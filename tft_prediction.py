import os
import collections
from typing import Dict, List, Tuple, Union
from datetime import datetime
import time
import copy
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder
from pytorch_forecasting.metrics import RMSE, MAE, MAPE, SMAPE, PoissonLoss
from pytorch_forecasting.metrics import MultiHorizonMetric
import matplotlib.pyplot as plt
from utils.misc import args_parser
from utils.misc import get_data
from sklearn import metrics

def data_process(data, df_data_file):
    df = pd.DataFrame()
    data['timestep'] = data.index

    for col in data.columns[:100]:
        df_col = data[['timestep', col]]
        df_col = df_col.sort_values(by=['timestep'])
        df_col.index = range(len(df_col))

        stamp = list(df_col["timestep"].array)
        month_day = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]
        cum_month_day = np.cumsum(month_day)
        time_idx = [0] * len(stamp)
        hour_in_day = [0] * len(stamp)
        day_in_month = [0] * len(stamp)
        day_in_week = [0] * len(stamp)
        for i in range(len(stamp)):
            dt = stamp[i]
            hour = dt.hour
            day = dt.day
            day_of_week = dt.strftime("%w")
            month = dt.month
            hour_in_day[i] = str(hour)
            day_in_month[i] = str(day)
            day_in_week[i] = str(day_of_week)
            time_idx[i] = (cum_month_day[month - 1] + day - 1) * 24 + hour
        time_idx = np.array(time_idx) - time_idx[0]
        df_col["time_idx"] = time_idx.astype(int)
        df_col["hour_in_day"] = hour_in_day
        df_col["day_in_week"] = day_in_week
        df_col['cluster_label'] = col
        df_col = df_col.rename(columns={col: 'metric'})
        df_col.index = range(len(df_col))
        df = pd.concat([df, df_col])
    df.index = range(len(df))
    df.to_csv(df_data_file)
    return df

if __name__ == '__main__':
    args = args_parser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    df_data_file = './dataset/select_100_{}_{}.csv'.format(args.file.split('.')[0], args.type)
    if not os.path.exists(df_data_file):
        data, _, selected_cells, mean, std, _, _ = get_data(args)
        data = data_process(data, df_data_file)
    else:
        data = pd.read_csv(df_data_file)

    data['hour_in_day'] = str(data['hour_in_day'])
    data['day_in_week'] = str(data['day_in_week'])

    max_prediction_length = 1
    min_prediction_length = 1
    max_encoder_length = 72  # 一个周期46个点
    min_encoder_length = 72
    training_cutoff = data["time_idx"].max() - 336

    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="metric",
        group_ids=['cluster_label'],
        min_encoder_length=min_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=min_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=[],
        static_reals=[],
        time_varying_known_categoricals=["hour_in_day", "day_in_week"],
        variable_groups={},
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=["metric"],
        target_normalizer=GroupNormalizer(groups=["cluster_label"], transformation=None),
        allow_missing_timesteps=True,
    )
    # set validation dataset parameters, prediction length as 1 weeks
    validation = TimeSeriesDataSet.from_dataset(training, data, min_prediction_length=1, max_prediction_length=1,
                                                min_prediction_idx=training_cutoff + 1, predict=True,
                                                stop_randomization=True)
    val_size = len(validation)
    batch_size = 128
    # generate dataloader
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    # early stopping and log settings
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()
    logger = TensorBoardLogger("log/abase_prediction_log")

    # define trainer and hypperparameters
    trainer = pl.Trainer(
        max_epochs=60,
        gpus=0,
        weights_summary="top",
        gradient_clip_val=0.10584967037537123,
        limit_train_batches=1.0,
        # fast_dev_run=True,
        callbacks=[lr_logger, early_stop_callback],
        logger=logger,
    )

    # define tft model and model hyperparameters
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.005,
        hidden_size=64,
        attention_head_size=2,
        dropout=0.2,
        hidden_continuous_size=32,
        # output_size=1,
        # loss=SMAPE(),
        output_size=1,
        loss=RMSE(),
        log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
        reduce_on_plateau_patience=4,
    )
    print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")

    # Train the model
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # # load best fit model for evaluation
    best_model_path = trainer.checkpoint_callback.best_model_path
    print("best model saved in path:", best_model_path)
    best_model_path = "./log/epoch=0-step=731.ckpt"
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    # smape error
    def smape(act, forc):
        return 100 / len(act) * np.sum(2 * np.abs(forc - act) / (np.abs(act) + np.abs(forc)))

    def mse(act, forc):
        pred = [i[0] for i in forc]
        return np.average((pred - act) ** 2)

    def mae(true, forc):
        pred = [i[0] for i in forc]
        return np.mean(np.abs(true - pred))

    # get predictions
    actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
    predictions, x = best_tft.predict(val_dataloader, return_x=True, mode="raw")

    mse = metrics.mean_squared_error(np.array(actuals).ravel(), np.array(predictions['prediction']).ravel())
    mae = metrics.mean_absolute_error(np.array(actuals).ravel(), np.array(predictions['prediction']).ravel())
    print(' TFT File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}'.format(args.file, args.type, mse, mae))