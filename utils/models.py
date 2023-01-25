# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: models.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2020-01-13 (YYYY-MM-DD)
-----------------------------------------------
"""
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from torch.nn import Parameter

class WeightLayer(nn.Module):
    def __init__(self):
        super(WeightLayer, self).__init__()
        self.w = nn.Parameter(torch.rand(1, requires_grad=True))

    def forward(self, x):
        return x * self.w

class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.out_dim = args.out_dim
        self.num_layers = args.num_layers
        self.close_size = args.close_size
        self.period_size = args.period_size
        self.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

        self.lstm_close = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                                  batch_first=True, dropout=0.2)
        self.lstm_period = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                                   batch_first=True, dropout=0.2)

        self.weight_close = WeightLayer()
        self.weight_period = WeightLayer()

        self.linear_layer = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, xc, xp):
        bz = xc.size(0)
        h0 = Variable(torch.zeros(self.num_layers * 1, bz, self.hidden_dim)).to(self.device)
        c0 = Variable(torch.zeros(self.num_layers * 1, bz, self.hidden_dim)).to(self.device)

        self.lstm_close.flatten_parameters()
        self.lstm_period.flatten_parameters()

        xc_out, xc_hn = self.lstm_close(xc, (h0, c0))
        x = xc_out[:, -1, :]
        if self.period_size > 0:
            xp_out, xp_hn = self.lstm_period(xp, (h0, c0))
            y = xp_out[:, -1, :]
        out = x + y
        model_pred = self.linear_layer(out)
        return model_pred

class esm_LSTM(nn.Module):
    def __init__(self, args):
        super(esm_LSTM, self).__init__()
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.out_dim = args.out_dim
        self.num_layers = args.num_layers
        self.close_size = args.close_size
        self.period_size = args.period_size
        self.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

        self.lstm_close = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                                  batch_first=True, dropout=0.2)
        self.lstm_period = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                                   batch_first=True, dropout=0.2)

        self.weight_close = WeightLayer()
        self.weight_period = WeightLayer()

        self.linear_layer = nn.Linear(self.hidden_dim, self.out_dim)
        self.linear_layer_esm1 = nn.Linear(2 * self.out_dim, args.linear_hidden)
        self.linear_layer_esm2 = nn.Linear(args.linear_hidden, self.out_dim)


    def forward(self, xc, xp, ari_res):
        bz = xc.size(0)
        h0 = Variable(torch.zeros(self.num_layers * 1, bz, self.hidden_dim)).to(self.device)
        c0 = Variable(torch.zeros(self.num_layers * 1, bz, self.hidden_dim)).to(self.device)

        self.lstm_close.flatten_parameters()
        self.lstm_period.flatten_parameters()

        xc_out, xc_hn = self.lstm_close(xc, (h0, c0))
        x = xc_out[:, -1, :]
        if self.period_size > 0:
            xp_out, xp_hn = self.lstm_period(xp, (h0, c0))
            y = xp_out[:, -1, :]
        out = x + y
        model_pred = self.linear_layer(out)
        ari_res = torch.from_numpy(np.vstack(ari_res).reshape(-1, 1)).to(self.device)
        input = torch.cat((model_pred, ari_res), 1)
        esm_1 = self.linear_layer_esm1(input)
        esm_pred = self.linear_layer_esm2(esm_1)
        return esm_pred
