#!/usr/bin/env python3
# @File    : plot_charge.py
# @Time    : 9/18/2020 9:02 AM
# @Author  : Zavier Cai
# @Email   : caizefeng18@gmail.com
# @Software: PyCharm

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch

from deep_dft.utils.ML_utils import standardization2D
from deep_dft.SVTCharge import SVTNetCharge


def net_name_parse(net_basename, is_temp=True):
    raw_dropout = re.compile(r'dropout_(.*)_fc').findall(net_basename)[0]
    dropout_prob = [float(i) for i in raw_dropout.split('_')]
    if is_temp:
        raw_fc = re.compile(r'fc_(.*)_temp').findall(net_basename)[0]
    else:
        raw_fc = re.compile(r'fc_(.*)\.pt').findall(net_basename)[0]
    fc_list = [int(i) for i in raw_fc.split('_')]
    return {"dropout_prob": dropout_prob, "fc_list": fc_list}


if __name__ == '__main__':
    net_file_name = "SVTCharge_batch_7000_lr_0.00025_epoch_50_data_STO_600_cut9_gauss16_dropout_0.0_0.0_0.0_fc_300_300_300.pt"
    data_dir = "/public/WORK_backup/caizefeng/Datasets/STO_600_cut9_gauss16"
    test_file_path = "/public/WORK_backup/caizefeng/Datasets/STO_600_cut9_gauss16/test/10099.npy"

    net = SVTNetCharge(num_element=3, sigma_size=16, **net_name_parse(net_file_name, is_temp=False)).to(torch.float64)
    net.load_state_dict(torch.load(os.path.join("nets", net_file_name)))

    # net = torch.load(os.path.join("nets", net_file_name)).cpu()

    net.eval()
    data = np.load(test_file_path)

    # load pre-calculated mean and std to make the prediction
    train_mean, train_std = standardization2D(read_saved=True, data_path=data_dir)
    charge_dft = data[:, -1]
    with torch.no_grad():
        charge_pred = net((torch.from_numpy(data[:, :-1]) - train_mean) / train_std).numpy()

    # plot
    plt.figure(figsize=(10, 10))
    plt.scatter(charge_pred, charge_dft, c=charge_pred, cmap='viridis', s=10)
    plt.plot(np.linspace(0, 16, 50), np.linspace(0, 16, 50), ls="dashed", c="grey")
    plt.xlabel(r'$\rho_{\,\mathrm{pred}}$', fontsize=24)
    plt.ylabel(r'$\rho_{\,\mathrm{scf}}$', fontsize=24)
    # plt.title("Logrithm parity plot for the deep learning vs DFT charge density prediction", fontsize=16)
    plt.title("Parity plot for the deep learning vs DFT charge density prediction", fontsize=16)
    # plt.xscale("log")
    # plt.yscale("log")
    plt.xlim(-1,16)
    plt.xlim(-1,16)
    plt.grid(which='major')

    plt.show()
