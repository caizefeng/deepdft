#!/usr/bin/env python3
# @File    : ML_utils.py
# @Time    : 9/13/2020 10:27 PM
# @Author  : Zavier Cai
# @Email   : caizefeng18@gmail.com
# @Software: PyCharm
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from deep_dft.utils.AsyncDataloader import FileDataLoader


def evaluate_loss(test_val_iter, net, loss, train_mean=None, train_std=None, device=None):
    """evaluate RMSE if input criterion is nn.MSELoss"""
    if device is None and isinstance(net, nn.Module):
        device = list(net.parameters())[0].device
    test_loss = 0.0
    with torch.no_grad():
        for X, y in test_val_iter:
            if isinstance(net, nn.Module):
                net.eval()  # close dropout and batchnorm
                if train_mean is not None:
                    X = ((X - train_mean) / train_std).to(device)
                else:
                    X = X.to(device)
                y = y.to(device)
                y_hat = net(X)
                if y.ndim == 1:
                    test_loss += loss(y_hat, y.view(-1, 1)).item() * X.size(0)
                else:
                    test_loss += loss(y_hat, y).item() * X.size(0)
                net.train()  # restore the training mode
            else:
                raise RuntimeError("Inputed model to be evaluated is not a PyTorch nn.Module.")

    # Calling empty_cache() releases all **unused** cached memory from PyTorch
    # so that those can be used by other GPU applications. But it is blocking and will slow down the program
    # torch.cuda.empty_cache()

    return np.sqrt(test_loss / len(test_val_iter.dataset))


def standardize(x):
    """a function used to standardize feature"""
    return (x - x.mean(0)) / x.std(0)


def standardization2D(read_saved=True, data_path=None, num_feature=None, train_path_list=None, is_mmap=False,
                      train_iter_mmap=None):
    mean_std_path = os.path.join(data_path, "mean_std.pt")
    if read_saved:
        if os.path.exists(mean_std_path):
            train_mean, train_std = torch.load(mean_std_path, map_location=torch.device('cpu'))
            return train_mean, train_std

    train_sum = torch.zeros(num_feature)
    train_var_sum = torch.zeros(num_feature)

    if is_mmap:
        for X, y in train_iter_mmap:
            train_sum += X.sum(0)
        train_mean = train_sum / len(train_iter_mmap.dataset)
        for X, y in train_iter_mmap:
            train_var_sum += ((X - train_mean) ** 2).sum(0)
        train_std = torch.sqrt(train_var_sum / (len(train_iter_mmap.dataset) - 1))

    else:
        instance_count = 0
        train_file_iter_async = FileDataLoader(train_path_list, shuffle=False, queue_size=1)
        for train_file_data in train_file_iter_async:
            train_sum += train_file_data[:, :num_feature].sum(0)
            instance_count += train_file_data.size(0)
        train_mean = train_sum / instance_count

        for train_file_data in train_file_iter_async:
            train_var_sum += ((train_file_data[:, :num_feature] - train_mean) ** 2).sum(0)
        train_std = torch.sqrt(train_var_sum / (instance_count - 1))
    torch.save((train_mean, train_std), mean_std_path)

    return train_mean, train_std


def find_residual_fc(modulelist: nn.ModuleList):
    """function to find the starting and ending index of residual block"""
    # res-link starts after dropout and ends after BN
    dropout_index = [i for i, module in enumerate(modulelist) if isinstance(module, nn.Dropout)]
    linear_index = [i for i, module in enumerate(modulelist) if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    # linear_index = [i for i, module in enumerate(modulelist) if isinstance(module, nn.Linear)]
    starting_index = []
    ending_index = []
    for i in dropout_index:
        if ending_index:
            if i < ending_index[-1]:
                continue
        linear_candidate = [j for j in linear_index if j > i]
        if len(linear_candidate) >= 2:
            starting_index.append(i)
            ending_index.append(linear_candidate[1])

    return starting_index, ending_index


def smooth(scalar, weight=0.85):
    """calculate the exponential moving average of a list"""
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def seed_torch(seed=2018):
    """set random seed for PyTorch reproducing"""
    random.seed(seed)  # random
    os.environ['PYTHONHASHSEED'] = str(seed)  # python

    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # LSTM/RNN

    np.random.seed(seed)  # numpy

    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed_all(seed)  # all GPUs

    torch.backends.cudnn.benchmark = False  # disable cudnn
    torch.backends.cudnn.deterministic = True
