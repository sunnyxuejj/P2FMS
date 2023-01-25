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
import os
import random

from utils.misc import args_parser, average_weights_att
from utils.misc import get_data, process_isolated
from utils.models import LSTM
from utils.fed_update import LocalUpdate, test_inference
from sklearn import metrics

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train_att(args, selected_cells, train, val, device, model_saved):
    global_model = LSTM(args).to(device)
    global_model.train()
    global_weights = global_model.state_dict()

    cell_loss = []
    loss_hist = []
    val_loss_cur = 10e9

    for epoch in range(args.epochs):
        local_weights, local_losses = [], []
        # print(f'\n | Global Training Round: {epoch+1} |\n')
        global_model.train()

        m = max(int(args.frac * args.bs), 1)
        cell_idx = random.sample(selected_cells, m)
        # print(cell_idx)

        for cell in cell_idx:
            cell_train, cell_val, cell_test = train[cell], val[cell], test[cell]
            local_model = LocalUpdate(args, cell_train, cell_val, cell_test, cell)

            global_model.load_state_dict(global_weights)
            global_model.train()

            w, loss = local_model.update_weights(model=copy.deepcopy(global_model),
                                                             global_round=epoch)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            cell_loss.append(loss)

        loss_hist.append(sum(cell_loss) / len(cell_loss))

        # Update global model
        global_weights = average_weights_att(local_weights, global_weights, args.epsilon)
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
    model_saved = './log/model_{}_{}_avg.pt'.format(args.file.split('.')[0], args.type)
    parameter_list = 'FedAvg-data-{:}-type-{:}-'.format(args.file, args.type)
    parameter_list += '-frac-{:.2f}-le-{:}-lb-{:}-seed-{:}'.format(args.frac, args.local_epoch,
                                                                   args.local_bs,
                                                                   args.seed)
    log_id = args.directory + parameter_list
    train, val, test = process_isolated(args, data)

    # Test model accuracy
    pred, truth = {}, {}
    test_loss_list = []
    test_mse_list = []
    nrmse = 0.0

    if not os.path.exists(model_saved):
        train_att(args, selected_cells, train, val, device, model_saved)

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
    print('FedAtt File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(args.file, args.type, mse, mae,
                                                                                     nrmse))
