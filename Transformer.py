import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) if d_model % 2 == 0 else torch.cos(position * div_term)[:, :-1]
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class GConv(nn.Module):
    # Spectral-based graph convolution function.
    # x: tensor, [batch_size, c_in, time_step, n_route].
    # theta: tensor, [ks*c_in, c_out], trainable kernel parameters.
    # ks: int, kernel size of graph convolution.
    # c_in: int, size of input channel.
    # c_out: int, size of output channel.
    # return: tensor, [batch_size, c_out, time_step, n_route].

    def __init__(self, ks, c_in, c_out, graph_kernel):
        super(GConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.ks = ks
        self.graph_kernel = graph_kernel
        self.theta = nn.Linear(ks*c_in, c_out)

    def forward(self, x):
        # graph kernel: tensor, [n_route, ks*n_route]
        kernel = self.graph_kernel
        # time_step, n_route
        _, _, t, n = x.shape
        # x:[batch_size, c_in, time_step, n_route] -> [batch_size, time_step, c_in, n_route]
        x_tmp = x.transpose(1, 2).contiguous()
        # x_ker = x_tmp * ker -> [batch_size, time_step, c_in, ks*n_route]
        x_ker = torch.matmul(x_tmp, kernel)
        # -> [batch_size, time_step, c_in*ks, n_route] -> [batch_size, time_step, n_route, c_in*ks]
        x_ker = x_ker.reshape(-1, t, self.c_in * self.ks, n).transpose(2, 3)
        # -> [batch_size, time_step, n_route, c_out]
        x_fig = self.theta(x_ker)
        # -> [batch_size, c_out, time_step, n_route]
        return x_fig.permute(0, 3, 1, 2).contiguous()


class TransformerModel(nn.Module):
    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.1):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        decoder_layers = TransformerDecoderLayer(ninp, nhead, nhid, dropout)
        self.decoder = TransformerDecoder(decoder_layers, nlayers)
        self.ninp = ninp

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tar, tgt_mask=None):
        if tgt_mask is None:
            device = tar.device
            tgt_mask = self._generate_square_subsequent_mask(tar.shape[0]).to(device)

        src = self.pos_encoder(src)
        tar = self.pos_encoder(tar)
        memory = self.transformer_encoder(src)
        output = self.decoder(tar, memory, tgt_mask)
        return output


class ShiftTransformer(nn.Module):
    def __init__(self, args, dropout=0.2):
        super(ShiftTransformer, self).__init__()
        self.args = args
        self.device = torch.device(
            'cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        self.out_dim = args.out_dim
        self.fc_1 = nn.Linear(1, args.hidden_dim)
        self.linear_layer = nn.Linear(args.hidden_dim, self.out_dim)
        self.transformer = TransformerModel(args.hidden_dim, 2, args.hidden_dim, args.num_layers, dropout)

    def forward(self, src, labels, tgt_mask=None):
        src = self.fc_1(src)
        labels = labels.view(1, -1, 1)
        tar = self.fc_1(labels)
        output = self.transformer(src, tar)
        output = self.linear_layer(output)
        return output.view(-1, 1)


class esm_Transformer(nn.Module):
    def __init__(self, args, dropout=0.2):
        super(esm_Transformer, self).__init__()
        self.args = args
        self.device = torch.device(
            'cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        self.out_dim = args.out_dim
        self.fc_1 = nn.Linear(1, args.hidden_dim)
        self.fc_2 = nn.Linear(args.hidden_dim, self.out_dim)
        self.transformer = TransformerModel(args.hidden_dim, 2, args.hidden_dim, args.num_layers, dropout)
        self.linear_layer_esm1 = nn.Linear(2 * self.out_dim, args.linear_hidden)
        self.linear_layer_esm2 = nn.Linear(args.linear_hidden, self.out_dim)

    def forward(self, src, labels, ari_res, tgt_mask=None):
        src = self.fc_1(src)
        labels = labels.view(1, -1, 1)
        tar = self.fc_1(labels)
        output = self.transformer(src, tar)
        output = self.fc_2(output)
        output = output.view(-1, 1)
        ari_res = torch.from_numpy(np.vstack(ari_res).reshape(-1, 1)).to(self.device)
        input = torch.cat((output, ari_res), 1)
        esm_1 = self.linear_layer_esm1(input)
        esm_pred = self.linear_layer_esm2(esm_1)
        return esm_pred


