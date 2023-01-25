# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: __init__.py
# This file is created by Jingjing Xue
# Email: xuejingjing20g@ict.ac.cn
# Date: 2022-08-02 (YYYY-MM-DD)
-----------------------------------------------
"""
import h5py
import pandas as pd
import numpy as np
import copy
import torch
import torch.nn.functional as F
from sklearn import cluster
import random
import os
import tqdm
from scipy import linalg
from sklearn import metrics
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import ExponentialSmoothing
from utils.misc import args_parser, process_isolated
from utils.fed_update import LocalUpdate, test_inference_esm
import warnings
warnings.filterwarnings('ignore')

def get_one_data(args):
    path = os.getcwd()
    f = h5py.File(path + '/dataset/' + args.file, 'r')

    idx = f['idx'][()]
    cell = f['cell'][()]
    lng = f['lng'][()]
    lat = f['lat'][()]
    data = f[args.type][()][:, cell - 1]

    df = pd.DataFrame(data, index=pd.to_datetime(idx.ravel(), unit='s'), columns=cell)
    df.fillna(0, inplace=True)

    random.seed(args.seed)
    cell_pool = cell
    selected_cells = sorted(random.sample(list(cell_pool), args.bs))
    selected_cells_idx = np.where(np.isin(list(cell), selected_cells))

    random.seed(2)
    one_cell = random.sample(list(cell_pool), 1)
    one_cell_idx = np.where(np.isin(list(cell), one_cell))
    while one_cell[0] in selected_cells:
        one_cell = random.sample(list(cell_pool), 1)
        one_cell_idx = np.where(np.isin(list(cell), one_cell))
    cell_lng = lng[one_cell_idx]
    cell_lat = lat[one_cell_idx]
    # print('Selected cells:', selected_cells)

    random.seed(args.seed)
    df_one_cell = df[one_cell]
    # print(df_cells.head())

    # df_cells = df_cells.iloc[:-14 * 24]
    train_data = df_one_cell.iloc[:-args.test_days * 24]

    mean = train_data.mean()
    std = train_data.std()

    normalized_df = (df_one_cell - mean) / std

    return normalized_df, df_one_cell, one_cell, mean, std, cell_lng, cell_lat

def ets_pred(args, sub, pred_len, col):
    ets_model = ExponentialSmoothing(sub[:-3], trend="add", damped_trend=True, seasonal="add",
                                     seasonal_periods=24, use_boxcox=False,
                                     initialization_method="estimated").fit()
    predict_val = ets_model.forecast(3)
    #plot_predict(args, col, sub, predict_val, args.esm_m)
    return predict_val.values

def sarima_pred(args, sub, pred_len, col):
    S = 24
    sarima_model = sm.tsa.statespace.SARIMAX(sub[-24], order=(1, 1, 1), seasonal_order=(1, 1, 1, S))
    sarima_model = sarima_model.fit()
    # 模型预测
    predict_val = sarima_model.forecast(24)
    return predict_val.values

def process_isolated_esm(args, dataset):
    train, val, test = dict(), dict(), dict()
    column_names = [dataset.columns[0]]
    data_csv_path = 'dataset/{}_{}_{}_esm_data_{}.csv'.format(args.file.split('.')[0], args.type, args.esm_m, column_names[0])
    for col in column_names:
        cell_traffic = dataset[col]
        close_arr, period_arr, label_arr, ari_arr = [], [], [], []
        start_idx = max(args.close_size, args.period_size * 24)
        cell_pred_traffic = [None] * start_idx
        if os.path.exists(data_csv_path):
            dataset_ = pd.read_csv(data_csv_path)
            predict_val = dataset_[str(col) + '_{}_pred'.format(args.esm_m)]
        else:
            for idx in tqdm.tqdm(range(start_idx, len(cell_traffic), 3)):
                sub = cell_traffic[idx - args.period_size * 24: idx + 3]
                if args.esm_m == 'sarima':
                    pred = sarima_pred(args, sub, args.out_dim, col)
                elif args.esm_m == 'ets':
                    pred = ets_pred(args, sub, args.out_dim, col)
                for i in range(len(pred)):
                    cell_pred_traffic.append(pred[i])
            predict_val = cell_pred_traffic
            dataset[str(col) + '_{}_pred'.format(args.esm_m)] = cell_pred_traffic

        for idx in range(start_idx, len(cell_traffic) - args.out_dim):
            y_ = [cell_traffic.iloc[idx + i] for i in range(args.out_dim)]
            label_arr.append(y_)
            ari_arr.append([predict_val[idx + i] for i in range(args.out_dim)])

            if args.close_size > 0:
                x_close = [cell_traffic.iloc[idx - c] for c in range(1, args.close_size + 1)]
                close_arr.append(x_close)
            if args.period_size > 0:
                x_period = [cell_traffic.iloc[idx - p * 24] for p in range(1, args.period_size + 1)]
                period_arr.append(x_period)

        cell_arr_close = np.array(close_arr)
        cell_arr_close = cell_arr_close[:, :, np.newaxis]
        cell_label = np.array(label_arr)
        cell_ari_res = np.array(ari_arr)

        test_len = args.test_days * 24
        val_len = args.val_days * 24
        train_len = len(cell_arr_close) - test_len - val_len
        train_x_close = cell_arr_close[:train_len]
        val_x_close = cell_arr_close[train_len:train_len + val_len]
        test_x_close = cell_arr_close[-test_len:]

        train_label = cell_label[:train_len]
        train_ari_res = cell_ari_res[:train_len]
        val_label = cell_label[train_len:train_len + val_len]
        val_ari_res = cell_ari_res[train_len:train_len + val_len]
        test_label = cell_label[-test_len:]
        test_ari_res = cell_ari_res[-test_len:]

        if args.period_size > 0:
            cell_arr_period = np.array(period_arr)
            cell_arr_period = cell_arr_period[:, :, np.newaxis]
            train_x_period = cell_arr_period[:train_len]
            val_x_period = cell_arr_period[train_len:train_len + val_len]
            test_x_period = cell_arr_period[-test_len:]

        else:
            train_x_period = train_x_close
            val_x_period = val_x_close
            test_x_period = test_x_close

        train[col] = (train_x_close, train_x_period, train_label, train_ari_res)
        val[col] = (val_x_close, val_x_period, val_label, val_ari_res)
        test[col] = (test_x_close, test_x_period, test_label, test_ari_res)

    if not os.path.exists(data_csv_path):
        dataset.to_csv(data_csv_path)

    return train, val, test

if __name__ == '__main__':
    args = args_parser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data, _, selected_cells, mean, std, _, _ = get_one_data(args)

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    train, val, test = process_isolated_esm(args, data)
    model_saved = './log/model_{}_{}_esm_personal_{}.pt'.format(args.file.split('.')[0], args.type, args.agg)
    # personal_key = ['linear_layer.weight', 'linear_layer.bias', 'linear_layer_esm1.weight', 'linear_layer_esm1.bias', 'linear_layer_esm2.weight', 'linear_layer_esm2.bias']
    personal_key = ['linear_layer_esm1.weight', 'linear_layer_esm1.bias', 'linear_layer_esm2.weight', 'linear_layer_esm2.bias']

    with open(model_saved, 'rb') as f:
        global_model = torch.load(f)
    w_personal = {}
    global_weights = global_model.cpu().state_dict()
    for cell in selected_cells:
        w_personal[cell] = {key: global_weights[key] for key in global_weights.keys() & personal_key}

    cell_idx = selected_cells
    pred_test, truth_test = {}, {}
    for cell in cell_idx:
        cell_train, cell_val, cell_test = train[cell], val[cell], test[cell]
        local_update = LocalUpdate(args, cell_train, cell_val, cell_test, cell)
        local_model = copy.deepcopy(global_model)
        w_local = global_model.state_dict()
        for k in w_local.keys():
            if k in personal_key:
                w_local[k] = w_personal[cell][k]
        local_model.load_state_dict(w_local)
        local_model.train()

        w, loss, w_personal[cell] = local_update.personal_esm_update(model=copy.deepcopy(local_model), personal_key=personal_key)

    val_loss_list = []
    val_pred, val_truth = {}, {}
    for cell in selected_cells:
        cell_val = val[cell]
        cell_model = copy.deepcopy(global_model)
        w_cell = cell_model.state_dict()
        for k in w_cell.keys():
            if k in personal_key:
                w_cell[k] = w_personal[cell][k]
        cell_model.load_state_dict(w_cell)
        val_loss, _, _, val_pred[cell], val_truth[cell] = test_inference_esm(args, cell_model, cell_val)
        val_loss_list.append(val_loss)
    avg_val_loss = sum(val_loss_list) / len(val_loss_list)

    df_pred = pd.DataFrame.from_dict(val_pred)
    df_truth = pd.DataFrame.from_dict(val_truth)
    r2_score = metrics.r2_score(df_truth.values.ravel(), df_pred.values.ravel())
    print('Cluster: {}, Average training loss: {:.4f}, Val loss: {:4f}, Val R2 score: {:4f}'.format(selected_cells, loss, avg_val_loss, r2_score), flush=True)

    pred, truth = {}, {}
    test_loss_list = []
    for cell in selected_cells:
        cell_test = test[cell]
        cell_model = copy.deepcopy(global_model)
        w_cell = cell_model.state_dict()
        # 加载个性化层
        for k in w_cell.keys():
            if k in personal_key:
                w_cell[k] = w_personal[cell][k]
        cell_model.load_state_dict(w_cell)
        test_loss, test_mse, test_nrmse, pred[cell], truth[cell] = test_inference_esm(args, cell_model, cell_test)
        test_loss_list.append(test_loss)

    df_pred = pd.DataFrame.from_dict(pred)
    df_truth = pd.DataFrame.from_dict(truth)
    mse = metrics.mean_squared_error(df_pred.values.ravel(), df_truth.values.ravel())
    mae = metrics.mean_absolute_error(df_pred.values.ravel(), df_truth.values.ravel())
    print('Cluster: {}, Personal model FedAvg File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(selected_cells, args.file, args.type, mse, mae, test_nrmse))





