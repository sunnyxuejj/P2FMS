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
import time
import random
from torch import nn
from torch.utils.data import DataLoader
from utils.misc import args_parser, average_weights, average_weights_att
from utils.misc import get_data, process_isolated
from utils.models import LSTM
from utils.fed_update import test_inference
from sklearn import metrics


class LocalUpdateFedPer(object):
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

    def train(self, model, personal_key, global_round=0):
        model.train()
        epoch_loss = []
        lr = self.args.lr
        if self.args.opt == 'adam':
            optimizer_model = torch.optim.Adam(model.parameters(), lr=lr)
        elif self.args.opt == 'sgd':
            optimizer_model = torch.optim.SGD(model.parameters(), lr=lr, momentum=self.args.momentum)

        start_time = time.time()
        loss_int_sum = 0
        val_loss_cur = 10e9

        w_personal_cell = {}
        for iter in range(self.args.client_all_epoch):
            if iter < self.args.personal_epoch:
                for name, param in model.named_parameters():
                    if name in personal_key:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            else:
                for name, param in model.named_parameters():
                    if name in personal_key:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

            batch_loss, esm_batch_loss = [], []
            begin_time = time.time()
            for batch_idx, (xc, xp, y) in enumerate(self.train_loader):
                xc, xp = xc.float().to(self.device), xp.float().to(self.device)
                y = y.float().to(self.device)
                model.train()
                model.zero_grad()
                model_pred = model(xc, xp)
                loss = self.criterion_model(y, model_pred)
                loss.backward()
                optimizer_model.step()

                batch_loss.append(loss.item())
                loss_int_sum += loss.data
                if batch_idx % self.args.log_interval == 0 and batch_idx > 0:
                    time_int = time.time() - begin_time
                    loss_int = loss_int_sum.item() / self.args.log_interval
                    # print('| client_{} | epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                    #                         'model_loss {:5.4f} '.format(
                    #                        self.user, iter, batch_idx, len(self.train_loader), time_int * 1000 / self.args.log_interval, loss_int))
                    loss_int_sum = 0
                    begin_time = time.time()

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            elapsed = time.time() - start_time
            # print('| client_{} | epoch {:3d} | {:5.2f} ms| '
            #      'model_loss {:5.4f} ' .format(
            #    self.user, iter, elapsed * 1000, sum(batch_loss)/len(batch_loss)))

            if iter < self.args.personal_epoch:
                val_loss, avg_mse, nrmse, prediction, truth = test_inference(self.args, model, self.val_data)
                # print('| client_{} | epoch {:3d} |val_loss {:5.4f} '.format(self.user, iter, val_loss))
                if val_loss < val_loss_cur:
                    #    print('save personal model', flush=True)
                    val_loss_cur = val_loss
                    for key in model.state_dict().keys() & personal_key:
                        w_personal_cell[key] = model.state_dict()[key]

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), w_personal_cell

def train_FedPer(args, selected_cells, train, val, device, model_saved):
    global_model = LSTM(args).to(device)
    global_weights = global_model.cpu().state_dict()
    w_personal = {}
    personal_key = ['linear_layer.weight', 'linear_layer.bias']
    for cell in selected_cells:
        w_personal[cell] = {key: global_weights[key] for key in global_weights.keys() & personal_key}
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
            local = LocalUpdateFedPer(args, cell_train, cell_val, cell_test, idx, cell)
            local_model = copy.deepcopy(global_model)
            w_local = global_model.state_dict()
            # 加载个性化层
            for k in w_local.keys():
                if k in personal_key:
                    w_local[k] = w_personal[cell][k]
            local_model.load_state_dict(w_local)
            local_model.train()
            w_local, loss, w_personal[cell] = local.train(model=copy.deepcopy(local_model), personal_key=personal_key, global_round=epoch)
            local_weights.append(copy.deepcopy(w_local))
            local_losses.append(copy.deepcopy(loss))

        # Update global model
        if args.agg == 'avg':
            global_weights = average_weights(local_weights)
        elif args.agg == 'att':
            global_weights = average_weights_att(local_weights, global_weights, args.epsilon)
        global_model.load_state_dict(global_weights)

        val_loss_list = []
        val_pred, val_truth = {}, {}
        for cell in selected_cells:
            cell_val = val[cell]
            cell_model = copy.deepcopy(global_model)
            w_cell = cell_model.state_dict()
            cell_model.load_state_dict(w_cell)
            val_loss, _, _, val_pred[cell], val_truth[cell] = test_inference(args, cell_model, cell_val)
            val_loss_list.append(val_loss)
        avg_val_loss = sum(val_loss_list) / len(val_loss_list)
        df_pred = pd.DataFrame.from_dict(val_pred)
        df_truth = pd.DataFrame.from_dict(val_truth)
        r2_score = metrics.r2_score(df_truth.values.ravel(), df_pred.values.ravel())
        print('Round: {}, Average training loss: {:.4f}, Val loss: {:4f}, Val R2 score: {:4f}'.format(epoch, sum(local_losses) / len(local_losses), avg_val_loss, r2_score), flush=True)

        #if avg_val_loss < val_loss_cur:
        #    val_loss_cur = avg_val_loss
        with open(model_saved, 'wb') as f:
            #print('Save model')
            torch.save(global_model, f)

    return w_personal

if __name__ == '__main__':
    args = args_parser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data, _, selected_cells, mean, std, _, _ = get_data(args)
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # print(selected_cells)
    model_saved = './log/model_{}_{}_FedPer.pt'.format(args.file.split('.')[0], args.type)
    train, val, test = process_isolated(args, data)

    # Test model accuracy
    pred, truth = {}, {}
    test_loss_list = []
    test_mse_list = []
    nrmse = 0.0

    w_personal = train_FedPer(args, selected_cells, train, val, device, model_saved)

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
    print('FedPer Algo Global model File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(args.file, args.type, mse, mae, nrmse))

    personal_key = ['linear_layer.weight', 'linear_layer.bias']
    for cell in selected_cells:
        cell_test = test[cell]
        cell_model = copy.deepcopy(model_best)
        w_cell = cell_model.state_dict()
        # 加载个性化层
        for k in w_cell.keys():
            if k in personal_key:
                w_cell[k] = w_personal[cell][k]
        cell_model.load_state_dict(w_cell)
        test_loss, test_mse, test_nrmse, pred[cell], truth[cell] = test_inference(args, cell_model, cell_test)
        # print(f'Cell {cell} MSE {test_mse:.4f}')
        nrmse += test_nrmse

        test_loss_list.append(test_loss)
        test_mse_list.append(test_mse)

    df_pred = pd.DataFrame.from_dict(pred)
    df_truth = pd.DataFrame.from_dict(truth)
    # df_pred.to_csv('./log/predict_{}_{}_esm_presonal_{}.csv'.format(args.file.split('.')[0], args.type, args.agg))

    mse = metrics.mean_squared_error(df_pred.values.ravel(), df_truth.values.ravel())
    mae = metrics.mean_absolute_error(df_pred.values.ravel(), df_truth.values.ravel())
    nrmse = nrmse / len(selected_cells)
    print('Personal model FedAvg File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(args.file, args.type, mse, mae, nrmse))