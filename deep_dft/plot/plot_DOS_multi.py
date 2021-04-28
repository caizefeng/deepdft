#!/usr/bin/env python3
# @File    : plot_charge.py
# @Time    : 9/18/2020 9:02 AM
# @Author  : Zavier Cai
# @Email   : caizefeng18@gmail.com
# @Software: PyCharm

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from deep_dft.SVTDOS import SVTNetDOS
from deep_dft.utils.ML_utils import standardization2D

if __name__ == '__main__':
    net_file_name = "state_dict_SVTDOS_300_300_300_0.3_0.3_0.3_41_10_1_1_0.0_fc_True_7500_2000_0.00025_cos-warm_5_2_single_vacuum_STO_600_cut9_gauss16_0.25_5_False_BN_seq_without_activate.pt"
    data_dir = "/public/WORK_backup/caizefeng/Datasets/vacuum_STO/vacuum_STO_600_cut9_gauss16_0.25_5"
    ldos_dir = "/public/WORK_backup/caizefeng/Datasets/vacuum_STO/vacuum_STO_600_cut9_gauss16_0.25_5"
    nets_dir = "/home/lzhpc/WORK/caizefeng/DeepDFT/nets"
    snapshot_label_list = ["10010", "10011", "10012", "10013"]
    level_name = "Vacuum"
    plot_type = "train"

    feature_file_path = os.path.join(data_dir, plot_type, snapshot_label_list[0]) + '.npy'
    energy_file_path_list = [os.path.join(ldos_dir, plot_type, "LDOS_" + snapshot_label) + '.txt' for snapshot_label in
                             snapshot_label_list]
    dos_file_path_list = [os.path.join(ldos_dir, plot_type, "LDOS_" + snapshot_label) + '.npy' for snapshot_label in
                          snapshot_label_list]

    # Save Module
    # net = torch.load(os.path.join(nets_dir, net_file_name))

    # Save state_dict()
    # checkpoint = torch.load(os.path.join(nets_dir, net_file_name))
    # net = SVTNetDOS(**checkpoint['model_hp'])
    # net.load_state_dict(checkpoint['model_state_dict'])
    # When haven't save hyperparameters of net
    # net_hp = {"num_element": 3, "sigma_size": 16,
    #           "fc_list": [300, 300, 300], "dropout_prob": [0.4, 0.3, 0.3],
    #           "num_windows": 41,
    #           "input_size_lstm": 10, "hidden_size_lstm": 1,
    #           "num_layers_lstm": 1, "dropout_lstm": 0.0,
    #           "output_style": "fc", "is_res": True, }
    # net = SVTNetDOS(**net_hp)
    # net.load_state_dict(checkpoint)

    # net.cpu().double()  # evaluate on CPU because OOM on GPU with such a large batch
    # feature = np.load(feature_file_path)
    energy = np.loadtxt(energy_file_path_list[0])
    interval = np.mean(np.diff(energy))
    dos_list = [np.load(dos_file_path) for dos_file_path in dos_file_path_list]

    # Load pre-calculated mean and std to make the prediction
    # train_mean, train_std = standardization2D(read_saved=True, data_path=data_dir)
    # sample = (torch.from_numpy(feature[:, :-1]) - train_mean) / train_std
    # with torch.no_grad():
    #     net.eval()
    #     dos_pred = (net(sample).sum(dim=0) / interval).numpy()

    dos_dft_list = [dos.sum(axis=0) / interval for dos in dos_list]

    # plot
    plt.figure(figsize=(10, 5))
    for idx, dos_dft in enumerate(dos_dft_list):
        plt.plot(energy, dos_dft, label=snapshot_label_list[idx])
    # plt.plot(energy, dos_pred, label="DOS (DeepDFT)", lw=3)
    # plt.axvline(0, ls='--', label="Fermi level", c='grey')
    plt.axvline(0, ls='--', label="{} Level".format(level_name), c='grey')
    plt.legend()
    plt.xlabel('Energy(eV)', fontsize=16)
    plt.ylabel('DOS', fontsize=16)
    # plt.title("Logrithm parity plot for the deep learning vs DFT charge density prediction", fontsize=16)
    title = "The Density of States (DOS) Prediction (Training Set)" if plot_type == 'train' else "The Density of States (DOS) Prediction (Test Set)"
    plt.title(title, fontsize=16)
    plt.xlim(np.min(energy), np.max(energy))
    plt.ylim(0, )
    # plt.savefig('dos_compare.png', dpi=600)
    plt.show()
