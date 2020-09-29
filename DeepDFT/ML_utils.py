#!/usr/bin/env python3
# @File    : ML_utils.py
# @Time    : 9/13/2020 10:27 PM
# @Author  : Zavier Cai
# @Email   : caizefeng18@gmail.com
# @Software: PyCharm
import math
import os

import numpy as np
import torch
from torch import nn
# a function used to standardize feature
from torch.utils.data import DataLoader, TensorDataset


def standardize(x):
    return (x - x.mean(0)) / x.std(0)


# RMSE if input criterion is nn.MSELoss
def evaluate_loss(test_val_iter, net, loss, train_mean, train_std, device=None):
    if device is None and isinstance(net, nn.Module):
        device = list(net.parameters())[0].device
    test_loss = 0.0
    with torch.no_grad():
        for X, y in test_val_iter:
            if isinstance(net, nn.Module):
                net.eval()  # close dropout
                X = ((X - train_mean) / train_std).to(device)
                y = y.to(device)
                y_hat = net(X)
                test_loss += loss(y_hat, y.view(-1, 1)).item() * X.size(0)
                net.train()  # restore the training mode
    return math.sqrt(test_loss / len(test_val_iter.dataset))


def standardization2D(read_saved=True, is_mmap=False, data_path=None, train_iter_more=None, num_feature=None,
                      train_path_list=None):
    mean_std_path = os.path.join(data_path, "mean_std.pt")
    if read_saved:
        train_mean, train_std = torch.load(mean_std_path, map_location=torch.device('cpu'))

    else:
        train_sum = torch.zeros(num_feature)
        train_var_sum = torch.zeros(num_feature)

        if is_mmap:
            for X, y in train_iter_more:
                train_sum += X.sum(0)
            train_mean = train_sum / len(train_iter_more.dataset)
            for X, y in train_iter_more:
                train_var_sum += ((X - train_mean) ** 2).sum(0)
            train_std = torch.sqrt(train_var_sum / (len(train_iter_more.dataset) - 1))

        else:
            instance_count = 0
            dataload_hp = {"batch_size": 50000, "shuffle": False, "num_workers": 8, "pin_memory": False}
            for train_file in train_path_list:
                all_data = torch.from_numpy(np.load(train_file))
                train_iter = DataLoader(TensorDataset(all_data[:, :-1], all_data[:, -1]), **dataload_hp)
                for X, y in train_iter:
                    train_sum += X.sum(0)
                    instance_count += X.size(0)
            train_mean = train_sum / instance_count
            for train_file in train_path_list:
                all_data = torch.from_numpy(np.load(train_file))
                train_iter = DataLoader(TensorDataset(all_data[:, :-1], all_data[:, -1]), **dataload_hp)
                for X, y in train_iter:
                    train_var_sum += ((X - train_mean) ** 2).sum(0)
            train_std = torch.sqrt(train_var_sum / instance_count - 1)
        torch.save((train_mean, train_std), mean_std_path)

    return train_mean, train_std
