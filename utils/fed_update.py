# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn import metrics
import time
import copy

torch.manual_seed(2020)
np.random.seed(2020)


class LocalUpdate(object):
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

    def update_weights(self, model, global_round=0):
        model.train()
        epoch_loss = []
        lr = self.args.lr

        a = model.parameters()
        if self.args.opt == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif self.args.opt == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=self.args.momentum)

        for iter in range(self.args.local_epoch):
            batch_loss = []
            for batch_idx, (xc, xp, y) in enumerate(self.train_loader):
                xc, xp = xc.float().to(self.device), xp.float().to(self.device)
                y = y.float().to(self.device)

                model.zero_grad()
                pred = model(xc, xp)

                loss = self.criterion_model(y, pred)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return model.state_dict(), sum(epoch_loss)/len(epoch_loss)


    def update_weights_esm(self, model, global_round):
        model.train()
        epoch_loss = []
        lr = self.args.lr

        if self.args.opt == 'adam':
            optimizer_model = torch.optim.Adam(model.parameters(), lr=lr)
        elif self.args.opt == 'sgd':
            optimizer_model = torch.optim.SGD(model.parameters(), lr=lr, momentum=self.args.momentum)

        start_time = time.time()
        loss_int_sum = 0
        for iter in range(self.args.local_epoch):
            batch_loss, esm_batch_loss = [], []

            begin_time = time.time()
            for batch_idx, (xc, xp, y, ari_res) in enumerate(self.train_loader):
                xc, xp = xc.float().to(self.device), xp.float().to(self.device)
                y = y.float().to(self.device)
                ari_res = ari_res.float().to(self.device)

                model.zero_grad()
                esm_pred = model(xc, xp, ari_res)

                loss = self.criterion_model(y, esm_pred)
                loss.backward()
                optimizer_model.step()

                batch_loss.append(loss.item())
                loss_int_sum += loss.data
                if batch_idx % self.args.log_interval == 0 and batch_idx > 0:
                    time_int = time.time() - begin_time
                    loss_int = loss_int_sum.item() / self.args.log_interval
                    #print('| client_{} | epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                    #                          'model_loss {:5.4f} '.format(
                    #                        self.user, iter, batch_idx, len(self.train_loader), time_int * 1000 / self.args.log_interval, loss_int))
                    loss_int_sum = 0
                    begin_time = time.time()

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            elapsed = time.time() - start_time
            #print('| client_{} | epoch {:3d} | {:5.2f} ms| '
            #      'model_loss {:5.4f} ' .format(
            #    self.user, iter, elapsed * 1000, sum(batch_loss)/len(batch_loss)))

        return model.state_dict(), sum(epoch_loss)/len(epoch_loss)


    def personal_esm_update(self, model, personal_key, global_round=0):
        model.train()
        epoch_loss = []
        lr = self.args.lr
        if self.args.opt == 'adam':
            optimizer_model = torch.optim.Adam(model.parameters(), lr=lr)
        elif self.args.opt == 'sgd':
            optimizer_model = torch.optim.SGD(model.parameters(), lr=lr, momentum=self.args.momentum)
        loss_int_sum = 0
        val_loss_cur = 10e9

        w_personal_cell = {}
        test_mse, test_mae = 0.0, 0.0
        start_time = time.time()
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
            for batch_idx, (xc, xp, y, ari_res) in enumerate(self.train_loader):
                xc, xp = xc.float().to(self.device), xp.float().to(self.device)
                y = y.float().to(self.device)
                ari_res = ari_res.float().to(self.device)

                model.train()
                model.zero_grad()
                model_pred = model(xc, xp, ari_res)

                loss = self.criterion_model(y, model_pred)
                loss.backward()
                optimizer_model.step()

                batch_loss.append(loss.item())
                loss_int_sum += loss.data
                if batch_idx % self.args.log_interval == 0 and batch_idx > 0:
                    time_int = time.time() - begin_time
                    loss_int = loss_int_sum.item() / self.args.log_interval
                    #print('| client_{} | epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                    #                         'model_loss {:5.4f} '.format(
                    #                        self.user, iter, batch_idx, len(self.train_loader), time_int * 1000 / self.args.log_interval, loss_int))
                    loss_int_sum = 0
                    begin_time = time.time()

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            # elapsed = time.time() - start_time
            #print('| client_{} | epoch {:3d} | {:5.2f} ms| '
            #      'model_loss {:5.4f} ' .format(
            #    self.user, iter, elapsed * 1000, sum(batch_loss)/len(batch_loss)))

            if iter < self.args.personal_epoch:
                val_loss, avg_mse, nrmse, prediction, truth = test_inference_esm(self.args, model, self.val_data)
                #print('| client_{} | epoch {:3d} |val_loss {:5.4f} '.format(self.user, iter, val_loss))
                if val_loss < val_loss_cur:
                #    print('save personal model', flush=True)
                    val_loss_cur = val_loss
                    for key in model.state_dict().keys() & personal_key:
                        w_personal_cell[key] = model.state_dict()[key]
        # print('| client_{} | local update time {:5.2f} ms|' .format(self.user, (time.time() - start_time) * 1000))
        return model.state_dict(), sum(epoch_loss)/len(epoch_loss), w_personal_cell


def test_inference(args, model, dataset):
    model.eval()
    loss, mse = 0.0, 0.0
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    criterion = nn.MSELoss().to(device)
    data_loader = DataLoader(list(zip(*dataset)), batch_size=args.local_bs, shuffle=False)
    pred_list, truth_list = [], []

    with torch.no_grad():
        for batch_idx, (xc, xp, y) in enumerate(data_loader):
            xc, xp = xc.float().to(device), xp.float().to(device)
            y = y.float().to(device)
            pred = model(xc, xp)

            batch_loss = criterion(y, pred)
            loss += batch_loss.item()

            batch_mse = torch.mean((pred - y) ** 2)
            mse += batch_mse.item()

            pred_list.append(pred.detach().cpu())
            truth_list.append(y.detach().cpu())

    final_prediction = np.concatenate(pred_list).ravel()
    final_truth = np.concatenate(truth_list).ravel()
    nrmse= (metrics.mean_squared_error(final_truth, final_prediction) ** 0.5) / (max(final_truth) - min(final_truth))
    avg_loss = loss / len(data_loader)
    avg_mse = mse / len(data_loader)

    return avg_loss, avg_mse, nrmse, final_prediction, final_truth


def test_inference_esm(args, model, dataset):
    model.eval()
    loss, mse = 0.0, 0.0
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    criterion = nn.MSELoss().to(device)
    data_loader = DataLoader(list(zip(*dataset)), batch_size=args.local_bs, shuffle=False)
    pred_list, truth_list = [], []

    with torch.no_grad():
        for batch_idx, (xc, xp, y, ari_res) in enumerate(data_loader):
            xc, xp = xc.float().to(device), xp.float().to(device)
            ari_res = ari_res.float().to(device)
            y = y.float().to(device)
            pred = model(xc, xp, ari_res)

            batch_loss = criterion(y, pred)
            loss += batch_loss.item()

            pred_list.append(pred.detach().cpu())
            truth_list.append(y.detach().cpu())

    final_prediction = np.concatenate(pred_list).ravel()
    final_truth = np.concatenate(truth_list).ravel()
    nrmse= (metrics.mean_squared_error(final_truth, final_prediction) ** 0.5) / (max(final_truth) - min(final_truth))
    avg_loss = loss / len(data_loader)
    avg_mse = mse / len(data_loader)

    return avg_loss, avg_mse, nrmse, final_prediction, final_truth


def test_inference_esm_pos(args, model, dataset):
    model.eval()
    loss, mse = 0.0, 0.0
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    criterion = nn.MSELoss().to(device)
    data_loader = DataLoader(list(zip(*dataset)), batch_size=args.local_bs, shuffle=False)
    pred_list, truth_list = [], []

    with torch.no_grad():
        for batch_idx, (xc, xp, y, ari_res, position) in enumerate(data_loader):
            xc, xp = xc.float().to(device), xp.float().to(device)
            ari_res = ari_res.float().to(device)
            y = y.float().to(device)
            model = model.to(device)
            pred = model(xc, xp, ari_res, position)

            batch_loss = criterion(y, pred)
            loss += batch_loss.item()

            pred_list.append(pred.detach().cpu())
            truth_list.append(y.detach().cpu())

    final_prediction = np.concatenate(pred_list).ravel()
    final_truth = np.concatenate(truth_list).ravel()
    nrmse= (metrics.mean_squared_error(final_truth, final_prediction) ** 0.5) / (max(final_truth) - min(final_truth))
    avg_loss = loss / len(data_loader)
    avg_mse = mse / len(data_loader)

    return avg_loss, avg_mse, nrmse, final_prediction, final_truth


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
