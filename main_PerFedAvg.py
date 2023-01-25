# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: __init__.py
# This file is created by Jingjing Xue
# Email: xuejingjing20g@ict.ac.cn
# Date: 2022-08-02 (YYYY-MM-DD)
-----------------------------------------------
"""
import numpy as np
import copy
import torch
import pandas as pd
import random
from torch import nn
from torch.utils.data import DataLoader
from utils.misc import args_parser, average_weights, average_weights_att
from utils.misc import get_data, process_isolated
from utils.models import LSTM
from utils.fed_update import test_inference
from sklearn import metrics


class LocalUpdateMAML(object):
    def __init__(self, args, train, val, test, idx, cell):
        self.args = args
        self.train_loader = self.process_data(train)
        self.val_data = val
        self.test_data = test
        self.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        self.criterion_model = nn.MSELoss().to(self.device)
        self.idx = idx
        self.cell = cell

    def process_data(self, dataset):
        data = list(zip(*dataset))
        loader = DataLoader(data, shuffle=False, batch_size=self.args.local_bs)
        return loader

    def train(self, net, lr, local_epoch):
        net.train()
        lr_in = lr * 0.001

        if self.args.opt == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        elif self.args.opt == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(local_epoch):
            batch_loss = []
            for batch_idx, (xc, xp, y) in enumerate(self.train_loader):
                sup_xc, sup_xp = xc.float().to(self.device), xp.float().to(self.device)
                sup_y = y.float().to(self.device)
                targ_xc, targ_xp = xc.float().to(self.device), xp.float().to(self.device)
                targ_y = y.float().to(self.device)
                param_dict = dict()
                for name, param in net.named_parameters():
                    if param.requires_grad:
                        param_dict[name] = param.to(device=self.device)
                names_weights_copy = param_dict

                net.zero_grad()
                sup_pred = net(sup_xc, sup_xp)
                loss_sup = self.criterion_model(sup_y, sup_pred)
                grads = torch.autograd.grad(loss_sup, names_weights_copy.values(), create_graph=True, allow_unused=True)
                names_grads_copy = dict(zip(names_weights_copy.keys(), grads))

                for key, grad in names_grads_copy.items():
                    if grad is not None:
                        names_grads_copy[key] = names_grads_copy[key].sum(dim=0)
                        names_weights_copy[key] = names_weights_copy[key] - lr_in * names_grads_copy[key]

                targ_pred = net(targ_xc, targ_xp)
                loss_targ = self.criterion_model(targ_y, targ_pred)
                loss_targ.backward()
                optimizer.step()

                del targ_pred.grad
                del loss_targ.grad
                del loss_sup.grad
                del sup_pred.grad
                optimizer.zero_grad()
                net.zero_grad()

                batch_loss.append(loss_sup.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

def train_PerFedAvg(args, selected_cells, train, val, device, model_saved):
    global_model = LSTM(args).to(device)
    val_loss_cur = 10e9

    for epoch in range(args.epochs + 1):
        local_weights, local_losses, personal_loss = [], [], []
        global_model.train()

        m = max(int(args.frac * args.bs), 1)
        if epoch == args.epochs:
            m = args.bs
        cell_idx = random.sample(selected_cells, m)

        for idx, cell in enumerate(cell_idx):
            cell_train, cell_val, cell_test = train[cell], val[cell], test[cell]
            local = LocalUpdateMAML(args, cell_train, cell_val, cell_test, idx, cell)

            net_local = copy.deepcopy(global_model)
            w_local, loss = local.train(net_local.to(device), lr=args.lr, local_epoch=args.local_epoch)
            local_weights.append(copy.deepcopy(w_local))
            local_losses.append(copy.deepcopy(loss))

        # Update global model
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        val_loss_list = []
        val_pred, val_truth = {}, {}
        for cell in selected_cells:
            cell_val = val[cell]
            val_loss, val_mse, val_nrmse, val_pred[cell], val_truth[cell] = test_inference(args, global_model, cell_val)
            val_loss_list.append(val_loss)
        avg_val_loss = sum(val_loss_list) / len(val_loss_list)
        df_pred = pd.DataFrame.from_dict(val_pred)
        df_truth = pd.DataFrame.from_dict(val_truth)
        r2_score = metrics.r2_score(df_truth.values.ravel(), df_pred.values.ravel())
        print('Round: {}, Average training loss: {:.4f}, Val loss: {:4f}, Val R2 score: {:4f}'.format(epoch, sum(local_losses) / len(local_losses), avg_val_loss, r2_score), flush=True)

        if avg_val_loss < val_loss_cur:
            val_loss_cur = avg_val_loss
            with open(model_saved, 'wb') as f:
                print('Save model')
                torch.save(global_model, f)

if __name__ == '__main__':
    args = args_parser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data, _, selected_cells, mean, std, _, _ = get_data(args)
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # print(selected_cells)
    model_saved = './log/model_{}_{}_L2GD.pt'.format(args.file.split('.')[0], args.type)
    train, val, test = process_isolated(args, data)

    # Test model accuracy
    pred, truth = {}, {}
    test_loss_list = []
    test_mse_list = []
    nrmse = 0.0

    w_locals = train_PerFedAvg(args, selected_cells, train, val, device, model_saved)

    with open(model_saved, 'rb') as f:
        model_best = torch.load(f)

    for cell in selected_cells:
        cell_test = test[cell]
        test_loss, test_mse, test_nrmse, pred[cell], truth[cell] = test_inference(args, model_best, cell_test)
        # print(f'Cell {cell} MSE {test_mse:.4f}')
        nrmse += test_nrmse

        test_loss_list.append(test_loss)
        test_mse_list.append(test_mse)

    df_pred = pd.DataFrame.from_dict(pred)
    df_truth = pd.DataFrame.from_dict(truth)

    mse = metrics.mean_squared_error(df_pred.values.ravel(), df_truth.values.ravel())
    mae = metrics.mean_absolute_error(df_pred.values.ravel(), df_truth.values.ravel())
    nrmse = nrmse / len(selected_cells)
    print('Per_FedAvg Algo, File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(args.file, args.type, mse, mae, nrmse))