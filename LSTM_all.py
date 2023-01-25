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
import torch
import pandas as pd
from torch import nn
from utils.misc import args_parser
from utils.misc import get_data, process_centralized, process_isolated
from utils.models import LSTM
from torch.utils.data import DataLoader
from utils.fed_update import test_inference
from sklearn import metrics

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def process_data(args, dataset):
    data = list(zip(*dataset))
    loader = DataLoader(data, shuffle=False, batch_size=args.local_bs)
    return loader

def train_single(args, train, val, device, model_saved):
    single_model = LSTM(args).to(device)
    criterion_model = nn.MSELoss().to(device)
    cell_loss = []
    loss_hist = []
    val_loss_cur = 10e9

    lr = args.lr
    if args.opt == 'adam':
        optimizer = torch.optim.Adam(single_model.parameters(), lr=lr)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(single_model.parameters(), lr=lr, momentum=args.momentum)
    train_loader = process_data(args, train)

    for epoch in range(args.epochs):
        single_model.train()
        batch_loss = []
        for batch_idx, (xc, xp, y) in enumerate(train_loader):
            xc, xp = xc.float().to(device), xp.float().to(device)
            y = y.float().to(device)
            single_model = single_model.to(device)
            single_model.zero_grad()
            pred = single_model(xc, xp)
            loss = criterion_model(y, pred)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        epoch_loss = sum(batch_loss) / len(batch_loss)

        val_loss, val_mse, val_nrmse, val_pred, val_truth = test_inference(args, single_model, val)
        print('epoch: {}, training loss: {:.4f}, Val loss: {:4f}'.format(epoch, epoch_loss, val_loss), flush=True)
        if val_loss < val_loss_cur:
            val_loss_cur = val_loss
            with open(model_saved, 'wb') as f:
                print('Save model')
                torch.save(single_model, f)


if __name__ == '__main__':
    args = args_parser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data, _, selected_cells, mean, std, _, _ = get_data(args)
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    model_saved = './log/model_{}_{}_LSTM_all.pt'.format(args.file.split('.')[0], args.type)
    train, val, test = process_centralized(args, data)
    print('LSTM all File: {:} Type: {:}'.format(args.file, args.type))

    # Test model accuracy
    pred, truth = {}, {}

    #if not os.path.exists(model_saved):
    train_single(args, train, val, device, model_saved)

    with open(model_saved, 'rb') as f:
        model_best = torch.load(f)

    pred, truth = {}, {}
    test_loss_list = []
    test_mse_list = []
    nrmse = 0.0

    train, val, test = process_isolated(args, data)
    for cell in selected_cells:
        cell_test = test[cell]
        test_loss, test_mse, test_nrmse, pred[cell], truth[cell] = test_inference(args, model_best, cell_test)
        nrmse += test_nrmse

        test_loss_list.append(test_loss)
        test_mse_list.append(test_mse)

    df_pred = pd.DataFrame.from_dict(pred)
    df_truth = pd.DataFrame.from_dict(truth)

    mse = metrics.mean_squared_error(df_pred.values.ravel(), df_truth.values.ravel())
    mae = metrics.mean_absolute_error(df_pred.values.ravel(), df_truth.values.ravel())
    nrmse = nrmse / len(selected_cells)
    print('LSTM all File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(args.file, args.type, mse, mae,
                                                                                     nrmse))

    train, val, test = process_centralized(args, data)
    test_loss, test_mse, test_nrmse, pred, truth = test_inference(args, model_best, test)
    mse = metrics.mean_squared_error(pred, truth)
    mae = metrics.mean_absolute_error(pred, truth)
    nrmse = nrmse / len(selected_cells)
    print('LSTM all File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(args.file, args.type, mse, mae,
                                                                                     nrmse))