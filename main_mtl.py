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
import math
import random
from torch import nn
from torch.utils.data import DataLoader
from utils.misc import args_parser, average_weights, average_weights_att
from utils.misc import get_data, process_isolated
from utils.models import LSTM
from utils.fed_update import test_inference
from sklearn import metrics


class LocalUpdateMTL(object):
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

    def train(self, model, lr, local_epoch, omega, W_glob=None, w_glob_keys=None):
        model.train()
        # train and update

        if self.args.opt == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif self.args.opt == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(local_epoch):
            batch_loss = []
            for batch_idx, (xc, xp, y) in enumerate(self.train_loader):
                xc, xp = xc.float().to(self.device), xp.float().to(self.device)
                y = y.float().to(self.device)

                model.zero_grad()
                pred = model(xc, xp)
                loss = self.criterion_model(y, pred)
                batch_loss.append(loss.item())
                W = W_glob.clone().to(self.device)
                f = (int)(math.log10(W.shape[0]) + 1) + 1
                W_local = [model.state_dict(keep_vars=True)[key].flatten() for key in w_glob_keys]
                W_local = torch.cat(W_local)
                W[:, self.idx] = W_local

                loss_regularizer = 0
                loss_regularizer += W.norm() ** 2

                k = 9000
                for i in range(W.shape[0] // k):
                    x = W[i * k:(i + 1) * k, :]
                    loss_regularizer += x.mm(omega).mm(x.T).trace()
                loss_regularizer *= 10 ** (-f)

                loss = loss + loss_regularizer
                loss.backward()
                optimizer.step()

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            print('| client_{} | epoch {:3d} | model_loss {:5.4f} ' .format(self.cell, iter, sum(batch_loss)/len(batch_loss)), flush=True)

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

def train_MTL(args, selected_cells, train, val, device, model_saved):
    global_model = LSTM(args).to(device)
    global_model.train()
    global_weights = global_model.state_dict()
    w_glob_keys = global_weights.keys()

    # generate list of local models for each user
    w_locals, val_loss_p = {}, {}
    for cell in selected_cells:
        w_locals[cell] = global_weights
        val_loss_p[cell] = 10e9

    m = max(int(args.frac * args.bs), 1)
    i = torch.ones((m, 1))
    omega = i - 1 / m * i.mm(i.T)  # 这个omega是联邦多任务学习里的m*m的矩阵
    omega = omega ** 2
    omega = omega.to(device)

    W = [w_locals[selected_cells[0]][key].flatten() for key in w_glob_keys]
    W = torch.cat(W)
    d = len(W)
    del W
    val_loss_cur = 10e9

    for epoch in range(args.epochs + 1):
        local_weights, local_losses, personal_loss = [], [], []
        global_model.train()

        m = max(int(args.frac * args.bs), 1)
        if epoch == args.epochs:
            m = args.bs
        cell_idx = random.sample(selected_cells, m)

        W = torch.zeros((d, m)).to(device)
        for idx, cell in enumerate(cell_idx):
            W_local = [w_locals[cell][key].flatten() for key in w_glob_keys]
            W_local = torch.cat(W_local)
            W[:, idx] = W_local

        for idx, cell in enumerate(cell_idx):
            cell_train, cell_val, cell_test = train[cell], val[cell], test[cell]
            local = LocalUpdateMTL(args, cell_train, cell_val, cell_test, idx, cell)

            net_global = copy.deepcopy(global_model)
            w_glob_k = copy.deepcopy(net_global.state_dict()).keys()
            net_global.load_state_dict(w_locals[cell])
            w_local, loss = local.train(model=net_global.to(device), lr=args.lr, local_epoch=args.local_epoch, omega=omega, W_glob=W.clone(), w_glob_keys=w_glob_k)
            local_weights.append(copy.deepcopy(w_local))
            local_losses.append(copy.deepcopy(loss))

            net_local = copy.deepcopy(global_model)
            net_local.load_state_dict(w_local)
            val_loss_p_cell, _, _, _, _ = test_inference(args, net_local, cell_val)
            if val_loss_p_cell < val_loss_p[cell]:
                w_locals[cell] = copy.deepcopy(w_local)
                val_loss_p[cell] = copy.deepcopy(val_loss_p_cell)

        # get weighted average for global weights
        if args.agg == 'avg':
            global_weights = average_weights(local_weights)
        elif args.agg == 'att':
            global_weights = average_weights_att(local_weights, global_weights, args.epsilon)
        global_model.load_state_dict(global_weights)

        val_loss_list = []
        val_pred, val_truth = {}, {}
        for cell in selected_cells:
            cell_val = val[cell]
            val_loss, _, _, val_pred[cell], val_truth[cell] = test_inference(args, global_model, cell_val)
            val_loss_list.append(val_loss)
        avg_val_loss = sum(val_loss_list) / len(val_loss_list)

        df_pred = pd.DataFrame.from_dict(val_pred)
        df_truth = pd.DataFrame.from_dict(val_truth)
        r2_score = metrics.r2_score(df_truth.values.ravel(), df_pred.values.ravel())
        print('Round: {}, Average training loss of global model: {:.4f}, Val loss: {:4f}, Val R2 score: {:4f}'.format(epoch, sum(local_losses) / len(local_losses), avg_val_loss, r2_score), flush=True)
        if avg_val_loss < val_loss_cur:
            val_loss_cur = avg_val_loss
            with open(model_saved, 'wb') as f:
                print('Save model')
                torch.save(global_model, f)

    return w_locals

if __name__ == '__main__':
    args = args_parser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data, _, selected_cells, mean, std, _, _ = get_data(args)
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # print(selected_cells)
    model_saved = './log/model_{}_{}_ditto.pt'.format(args.file.split('.')[0], args.type)
    train, val, test = process_isolated(args, data)

    # Test model accuracy
    pred, truth = {}, {}
    test_loss_list = []
    test_mse_list = []
    nrmse = 0.0

    w_locals = train_MTL(args, selected_cells, train, val, device, model_saved)

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
    print('Global model FedAvg File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(args.file, args.type, mse, mae, nrmse), flush=True)

    for cell in selected_cells:
        cell_test = test[cell]
        cell_model = copy.deepcopy(model_best)
        # 加载个性化模型
        cell_model.load_state_dict(w_locals[cell])
        test_loss, test_mse, test_nrmse, pred[cell], truth[cell] = test_inference(args, cell_model, cell_test)
        # print(f'Cell {cell} MSE {test_mse:.4f}')
        nrmse += test_nrmse

        test_loss_list.append(test_loss)
        test_mse_list.append(test_mse)

    df_pred = pd.DataFrame.from_dict(pred)
    df_truth = pd.DataFrame.from_dict(truth)

    mse = metrics.mean_squared_error(df_pred.values.ravel(), df_truth.values.ravel())
    mae = metrics.mean_absolute_error(df_pred.values.ravel(), df_truth.values.ravel())
    nrmse = nrmse / len(selected_cells)
    print('Personal model FedAvg File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(args.file, args.type, mse, mae, nrmse), flush=True)