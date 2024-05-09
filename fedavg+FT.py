# -*- coding: utf-8 -*-

import numpy as np
import copy
import torch
import pandas as pd
import os

import tqdm

from utils.misc import args_parser
from utils.misc import get_data, process_isolated
from utils.FT_train import FT_Update
from utils.fed_update import test_inference
from sklearn import metrics
from fed_avg_algo import train_avg

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    args = args_parser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data, _, selected_cells, mean, std, _, _ = get_data(args)
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    personal_key = ['lstm_close.weight_hh_l0', 'lstm_close.bias_hh_l0', 'lstm_period.weight_hh_l0', 'lstm_period.bias_hh_l0', 'linear_layer.weight', 'linear_layer.bias']

    parameter_list = 'FedAvg-data-{:}-type-{:}-'.format(args.file, args.type)
    parameter_list += '-frac-{:.2f}-le-{:}-lb-{:}-seed-{:}'.format(args.frac, args.local_epoch,
                                                                   args.local_bs,
                                                                   args.seed)
    log_id = args.directory + parameter_list
    train, val, test = process_isolated(args, data)
    model_saved = './log/model_{}_{}_avg.pt'.format(args.file.split('.')[0], args.type)

    if not os.path.exists(model_saved):
        train_avg(args, selected_cells, train, val, device, model_saved)

    # Test model accuracy
    pred, truth = {}, {}
    test_loss_list = []
    test_mse_list = []
    nrmse = 0.0

    with open(model_saved, 'rb') as f:
        model_best = torch.load(f)

    for cell in tqdm.tqdm(selected_cells):
        cell_test = test[cell]
        cell_ft = FT_Update(args, train[cell], val[cell], cell)
        w_personal_cell = cell_ft.ft_update(model_best, personal_key) # 微调训练
        local_model = copy.deepcopy(model_best)
        w_local = local_model.state_dict()
        for k in w_local.keys():
            if k in personal_key:
                w_local[k] = w_personal_cell[k]
        local_model.load_state_dict(w_local)
        test_loss, test_mse, test_nrmse, pred[cell], truth[cell] = test_inference(args, local_model, cell_test)
        nrmse += test_nrmse

        test_loss_list.append(test_loss)
        test_mse_list.append(test_mse)

    df_pred = pd.DataFrame.from_dict(pred)
    df_truth = pd.DataFrame.from_dict(truth)

    mse = metrics.mean_squared_error(df_pred.values.ravel(), df_truth.values.ravel())
    mae = metrics.mean_absolute_error(df_pred.values.ravel(), df_truth.values.ravel())
    nrmse = nrmse / len(selected_cells)
    print('FedAvg File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(args.file, args.type, mse, mae, nrmse))
