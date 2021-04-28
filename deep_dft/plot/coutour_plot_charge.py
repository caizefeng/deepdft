#!/usr/bin/env python3
# @File    : coutour_plot_charge.py
# @Time    : 9/26/2020 8:18 PM
# @Author  : Zavier Cai
# @Email   : caizefeng18@gmail.com
# @Software: PyCharm
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from deep_dft.plot.parity_plot_charge import net_name_parse
from deep_dft.SVTCharge import SVTNetCharge
from deep_dft.utils import read_utils
from deep_dft.utils.ML_utils import standardization2D

if __name__ == '__main__':
    net_file_name = "SVTCharge_batch_7000_lr_0.00025_epoch_50_data_STO_600_cut9_gauss16_dropout_0.0_0.0_0.0_fc_300_300_300.pt"
    data_dir = "/public/WORK_backup/caizefeng/Datasets/STO_600_cut9_gauss16"
    test_file_path = "/public/WORK_backup/caizefeng/Datasets/STO_600_cut9_gauss16/test/10099.npy"
    test_structure_path = "/home/lzhpc/WORK/caizefeng/ML_DFT/Datasets/STO_600/CHGCAR_10099"
    cross_section = 1 / 2

    # recreate grid
    vec, coor_list, chg, ngxf, ngyf, ngzf = read_utils.split_read_chgcar(test_structure_path)
    # vec = torch.from_numpy(vec)
    # grid_coor = feature_utils.grid_gen(ngxf, ngyf, ngzf, vec)

    # load net
    # save as state_dict
    net = SVTNetCharge(num_element=3, sigma_size=16, **net_name_parse(net_file_name, is_temp=False)).to(torch.float64)
    net.load_state_dict(torch.load(os.path.join("nets", net_file_name)))

    # as as net
    # net = torch.load(os.path.join("nets", net_file_name))

    net.eval()
    data = np.load(test_file_path)

    # load pre-calculated mean and std to make the prediction
    train_mean, train_std = standardization2D(read_saved=True, data_path=data_dir)
    charge_dft = data[:, -1]
    with torch.no_grad():
        charge_pred = net((torch.from_numpy(data[:, :-1]) - train_mean) / train_std).numpy()
    # slice charge and reshape
    start_idx = int(ngxf * ngyf * ngzf * cross_section)
    end_idx = int(ngxf * ngyf * ngzf * cross_section + ngxf * ngyf)
    charge_dft_slice = charge_dft[start_idx: end_idx].reshape(ngyf, ngxf)
    charge_pred_slice = charge_pred[start_idx: end_idx].reshape(ngyf, ngxf)
    charge_diff = np.abs(charge_dft_slice - charge_pred_slice)

    # plot
    fig = plt.figure(figsize=(35, 10))

    ax = fig.add_subplot(131)
    im1 = ax.imshow(charge_dft_slice, cmap='viridis')
    ax.set_title("DFT charge density", fontsize=18)
    cbar1 = fig.colorbar(ax=ax, mappable=im1, )

    ax = fig.add_subplot(132)
    im2 = ax.imshow(charge_pred_slice, cmap='viridis')
    ax.set_title("DL charge density", fontsize=18)
    cbar2 = fig.colorbar(ax=ax, mappable=im2, )

    ax = fig.add_subplot(133)
    im3 = ax.imshow(charge_diff, cmap='viridis')
    ax.set_title("Absolute difference", fontsize=18)
    cbar3 = fig.colorbar(ax=ax, mappable=im3, )
    cbar3.set_label(r'$\rho$ / $e \cdot \AA^{-3}$', fontsize=18)

    plt.show()
