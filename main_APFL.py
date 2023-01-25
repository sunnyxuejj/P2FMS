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
import h5py
import tqdm
import copy
import torch
import pandas as pd
import os
import random
from torch import nn
from torch.utils.data import DataLoader
from utils.misc import args_parser, average_weights, average_weights_att
from utils.misc import get_data, process_isolated
from utils.models import LSTM
from utils.fed_update import LocalUpdate, test_inference
from sklearn import metrics


class LocalUpdateAPFL(object):
    def __init__(self, args, train, val, test, cell_idx):
        self.args = args
        self.train_loader = self.process_data(train)
        self.val_data = val
        self.test_data = test
        self.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        self.criterion_model = nn.MSELoss().to(self.device)
        self.user = cell_idx

    def process_data(self, dataset):
        data = list(zip(*dataset))
        loader = DataLoader(data, shuffle=False, batch_size=self.args.local_bs)
        return loader

    def train(self, model, lr, local_epoch, w_local_apfl=None):
        model.train()
        # train and update

        if self.args.opt == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif self.args.opt == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=self.args.momentum)

        epoch_loss, epoch_loss_p = [], []
        for iter in range(local_epoch):
            batch_loss, batch_loss_p = [], []
            for batch_idx, (xc, xp, y) in enumerate(self.train_loader):
                w_loc_new = {}
                w_glob = copy.deepcopy(model.state_dict())
                for k in model.state_dict().keys():
                    w_loc_new[k] = self.args.alpha_apfl * w_local_apfl[k] + self.args.alpha_apfl * w_glob[k]

                xc, xp = xc.float().to(self.device), xp.float().to(self.device)
                y = y.float().to(self.device)

                model.zero_grad()
                pred = model(xc, xp)
                loss = self.criterion_model(y, pred)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

                wt = copy.deepcopy(model.state_dict())
                model.load_state_dict(w_loc_new)
                pred = model(xc, xp)
                loss = self.args.alpha_apfl * self.criterion_model(y, pred)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss_p.append(loss.item())
                w_local_bar = copy.deepcopy(model.state_dict())
                for k in w_local_bar.keys():
                    w_local_apfl[k] = w_local_bar[k] - w_loc_new[k] + w_local_apfl[k]

                model.load_state_dict(wt)
                del wt
                del w_loc_new
                del w_glob

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_loss_p.append(sum(batch_loss_p) / len(batch_loss_p))

        return model.state_dict(), w_local_apfl, w_local_bar, sum(epoch_loss) / len(epoch_loss), sum(epoch_loss_p) / len(epoch_loss_p)

def train_apfl(args, selected_cells, train, val, device, model_saved):
    global_model = LSTM(args).to(device)
    global_model.train()
    global_weights = global_model.state_dict()

    # generate list of local models for each user
    w_locals, w_locals_save, val_loss_p = {}, {}, {}
    for cell in selected_cells:
        w_locals[cell] = global_weights
        w_locals_save[cell] = global_weights
        val_loss_p[cell] = 10e9

    val_loss_cur = 10e9
    for epoch in range(args.epochs + 1):
        local_weights, local_losses, personal_loss = [], [], []
        global_model.train()

        m = max(int(args.frac * args.bs), 1)
        if epoch == args.epochs:
            m = args.bs
        cell_idx = random.sample(selected_cells, m)

        for cell in cell_idx:
            cell_train, cell_val, cell_test = train[cell], val[cell], test[cell]
            local = LocalUpdateAPFL(args, cell_train, cell_val, cell_test, cell)

            net_global = copy.deepcopy(global_model)
            w_global, w_local, w_local_bar, loss, loss_p = local.train(model=net_global.to(device), lr=args.lr, local_epoch=args.local_epoch, w_local_apfl=w_locals[cell])
            local_weights.append(copy.deepcopy(w_global))
            local_losses.append(copy.deepcopy(loss))
            personal_loss.append(copy.deepcopy(loss_p))
            w_locals[cell] = copy.deepcopy(w_local)

            net_local = copy.deepcopy(global_model)
            net_local.load_state_dict(w_local_bar)
            val_loss_p_cell, _, _, _, _ = test_inference(args, net_local, cell_val)
            if val_loss_p_cell < val_loss_p[cell]:
                w_locals_save[cell] = copy.deepcopy(w_local_bar)
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
        print('Round: {}, Average training loss of global model: {:.4f}, Average training loss of personal model: {:.4f}, '
              'Val loss: {:4f}, Val R2 score: {:4f}'.format(epoch, sum(local_losses) / len(local_losses),
                                                            sum(personal_loss) / len(personal_loss), avg_val_loss, r2_score), flush=True)

        if avg_val_loss < val_loss_cur:
            val_loss_cur = avg_val_loss
            with open(model_saved, 'wb') as f:
                print('Save model')
                torch.save(global_model, f)

    return w_locals_save

if __name__ == '__main__':
    args = args_parser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data, _, selected_cells, mean, std, _, _ = get_data(args)
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # print(selected_cells)
    model_saved = './log/model_{}_{}_apfl.pt'.format(args.file.split('.')[0], args.type)
    train, val, test = process_isolated(args, data)

    # Test model accuracy
    pred, truth = {}, {}
    test_loss_list = []
    test_mse_list = []
    nrmse = 0.0

    w_locals = train_apfl(args, selected_cells, train, val, device, model_saved)

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
    print('Global model FedAvg File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(args.file, args.type, mse, mae, nrmse))

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
    print('Personal model FedAvg File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(args.file, args.type, mse, mae, nrmse))