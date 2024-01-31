# -*- coding: utf-8 -*-
import numpy as np
import tqdm
import time
import copy
import torch
import pandas as pd
import sys
import random
from utils.misc import args_parser, average_weights, average_weights_att
from utils.misc import get_data, process_isolated_esm_position
from utils.models import esm_LSTM_Pos
from utils.fed_update import LocalUpdate, test_inference_esm_pos
from sklearn import metrics

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def model_test(args, cell, model, cell_test):
    pred, truth = {}, {}
    test_loss, test_mse, test_nrmse, pred[cell], truth[cell] = test_inference_esm_pos(args, model, cell_test)
    # print(f'Cell {cell} MSE {test_mse:.4f}')
    df_pred = pd.DataFrame.from_dict(pred)
    df_truth = pd.DataFrame.from_dict(truth)

    mse = metrics.mean_squared_error(df_pred.values.ravel(), df_truth.values.ravel())
    mae = metrics.mean_absolute_error(df_pred.values.ravel(), df_truth.values.ravel())
    print('FedEsm File: {:} Type: {:}, Cluster: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(args.file, args.type, cell, mse, mae,
                                                                                                   test_nrmse))

if __name__ == '__main__':
    args = args_parser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data, _, selected_cells, mean, std, cell_lng, cell_lat = get_data(args)

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.device = device
    # print(selected_cells)

    train, val, test = process_isolated_esm_position(args, data, cell_lng, cell_lat)

    global_model = esm_LSTM_Pos(args).to(device)
    global_model.train()
    # print(global_model)

    global_weights = global_model.cpu().state_dict()
    w_personal = {}
    personal_key = ['linear_layer.weight', 'linear_layer.bias', 'linear_layer_esm1.weight', 'linear_layer_esm1.bias', 'linear_layer_esm2.weight', 'linear_layer_esm2.bias']
    # personal_key = ['linear_layer_esm1.weight', 'linear_layer_esm1.bias', 'linear_layer_esm2.weight', 'linear_layer_esm2.bias']

    for cell in selected_cells:
        w_personal[cell] = {key: global_weights[key] for key in global_weights.keys() & personal_key}

    model_saved = './log/model_{}_{}_esm_pos_personal_{}.pt'.format(args.file.split('.')[0], args.type, args.agg)
    best_val_loss = None
    val_loss = []
    val_acc = []
    cell_loss = []
    test_mse_hist, test_mae_hist = [], []
    val_loss_cur = 10e9
    for epoch in range(args.epochs+1):

        local_weights, avg_weight, local_losses = [], [], []
        # print(f'\n | Global Training Round: {epoch+1} |\n')
        global_model.train()

        m = max(int(args.frac * args.bs), 1)
        if epoch == args.epochs:
            m = args.bs
        cell_idx = random.sample(selected_cells, m)
        # print(cell_idx)

        pred_test, truth_test = {}, {}
        for cell in cell_idx:
            start_time = time.time()
            cell_train, cell_val, cell_test = train[cell], val[cell], test[cell]
            local_update = LocalUpdate(args, cell_train, cell_val, cell_test, cell)
            local_model = copy.deepcopy(global_model)
            w_local = global_model.state_dict()
            # 加载个性化层
            for k in w_local.keys():
                if k in personal_key:
                    w_local[k] = w_personal[cell][k]
            local_model.load_state_dict(w_local)
            local_model.train()
            start_time = time.time()
            w, loss, w_personal[cell] = local_update.personal_esm_pos_update(model=copy.deepcopy(local_model), personal_key=personal_key, global_round=epoch)
            print('| client_{} | local update time {:5.2f} ms|'.format(cell, (time.time() - start_time) * 1000))
            local_weights.append(copy.deepcopy(w))
            avg_weight.append(len(train[cell][0]))
            local_losses.append(copy.deepcopy(loss))
            cell_loss.append(loss)

        val_loss_list = []
        val_pred, val_truth = {}, {}
        for cell in selected_cells:
            cell_val = val[cell]
            cell_model = copy.deepcopy(global_model)
            w_cell = cell_model.state_dict()
            cell_model.load_state_dict(w_cell)
            val_loss, _, _, val_pred[cell], val_truth[cell] = test_inference_esm_pos(args, cell_model, cell_val)
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

        if epoch >= args.epochs - 10:
            pred_test, truth_test = {}, {}
            with open(model_saved, 'rb') as f:
                model_best = torch.load(f)
            for cell in selected_cells:
                cell_test = test[cell]
                cell_model = copy.deepcopy(model_best)
                w_cell = cell_model.state_dict()
                # 加载个性化层
                for k in w_cell.keys():
                    if k in personal_key:
                        w_cell[k] = w_personal[cell][k]
                cell_model.load_state_dict(w_cell)
                test_loss, _, _, pred_test[cell], truth_test[cell] = test_inference_esm_pos(args, cell_model, cell_test)

            df_pred = pd.DataFrame.from_dict(pred_test)
            df_truth = pd.DataFrame.from_dict(truth_test)
            test_mse = metrics.mean_squared_error(df_pred.values.ravel(), df_truth.values.ravel())
            test_mae = metrics.mean_absolute_error(df_pred.values.ravel(), df_truth.values.ravel())
            test_mse_hist.append(test_mse)
            test_mae_hist.append(test_mae)
            print('Round {}, Test MSE loss: {:.4f}, Test MAE loss: {:.4f}'.format(epoch, test_mse, test_mae))

        if epoch == args.epochs:
            print('Last ten round Average test MSE loss: {:.4f}, test MAE loss: {:.4f}'.format(
                sum(test_mse_hist) / len(test_mse_hist), sum(test_mae_hist) / len(test_mae_hist)))

        if args.agg == 'avg':
            global_weights = average_weights(local_weights)
        elif args.agg == 'att':
            global_weights = average_weights_att(local_weights, global_weights, args.epsilon)
        global_model.load_state_dict(global_weights)

        # if epoch % 10 == 0 and epoch > 5:
        #     args.lr = args.lr / 2
        #     if args.lr < 0.001:
        #         args.lr = 0.001

    # Test model accuracy
    pred, truth = {}, {}
    test_loss_list = []
    test_mse_list = []
    nrmse = 0.0

    with open(model_saved, 'rb') as f:
        model_best = torch.load(f)

    for cell in selected_cells:
        cell_test = test[cell]
        test_loss, test_mse, test_nrmse, pred[cell], truth[cell] = test_inference_esm_pos(args, model_best, cell_test)
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
        w_cell = cell_model.state_dict()
        # 加载个性化层
        for k in w_cell.keys():
            if k in personal_key:
                w_cell[k] = w_personal[cell][k]
        cell_model.load_state_dict(w_cell)
        test_loss, test_mse, test_nrmse, pred[cell], truth[cell] = test_inference_esm_pos(args, cell_model, cell_test)
        # print(f'Cell {cell} MSE {test_mse:.4f}')
        nrmse += test_nrmse

        test_loss_list.append(test_loss)
        test_mse_list.append(test_mse)

    df_pred = pd.DataFrame.from_dict(pred)
    df_truth = pd.DataFrame.from_dict(truth)
    df_pred.to_csv('./log/predict_{}_{}_esm_presonal_{}_new.csv'.format(args.file.split('.')[0], args.type, args.agg))

    mse = metrics.mean_squared_error(df_pred.values.ravel(), df_truth.values.ravel())
    mae = metrics.mean_absolute_error(df_pred.values.ravel(), df_truth.values.ravel())
    nrmse = nrmse / len(selected_cells)
    print('Personal model FedAvg File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(args.file, args.type, mse, mae, nrmse))

    '''for cell in data.columns.values:
        cell_test = test[cell]
        cell_model = copy.deepcopy(model_best)
        w_cell = cell_model.state_dict()
        # 加载个性化层
        for k in w_cell.keys():
            if k in personal_key:
                w_cell[k] = w_personal[cell][k]
        cell_model.load_state_dict(w_cell)
        model_test(args, cell, cell_model, cell_test)'''
