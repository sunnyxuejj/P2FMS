import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from utils.fed_update import test_inference

torch.manual_seed(2020)
np.random.seed(2020)


class FT_Update(object):
    def __init__(self, args, train, val, cell_idx):
        self.args = args
        self.train_loader = self.process_data(train)
        self.val_data = val
        self.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        self.criterion = nn.MSELoss().to(self.device)
        self.user = cell_idx

    def process_data(self, dataset):
        data = list(zip(*dataset))
        loader = DataLoader(data, shuffle=False, batch_size=self.args.local_bs)
        return loader

    def ft_update(self, model, personal_key):
        model.train()
        epoch_loss = []
        lr = self.args.lr
        if self.args.opt == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif self.args.opt == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=self.args.momentum)

        val_loss_cur = 10e9
        w_personal_cell = {}
        for iter in range(self.args.ft_epoch):
            for name, param in model.named_parameters():
                if name in personal_key:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            for batch_idx, (xc, xp, y) in enumerate(self.train_loader):
                xc, xp = xc.float().to(self.device), xp.float().to(self.device)
                y = y.float().to(self.device)

                model.zero_grad()
                pred = model(xc, xp)
                loss = self.criterion(y, pred)
                loss.backward()
                optimizer.step()

            val_loss, avg_mse, nrmse, prediction, truth = test_inference(self.args, model, self.val_data)
            if val_loss < val_loss_cur:
                val_loss_cur = val_loss
                for key in model.state_dict().keys() & personal_key:
                    w_personal_cell[key] = model.state_dict()[key]

        return w_personal_cell