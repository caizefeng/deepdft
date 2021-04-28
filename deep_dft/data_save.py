#!/usr/bin/env python3
# @File    : data_from_chgcar.py
# @Time    : 9/12/2020 1:52 PM
# @Author  : Zavier Cai
# @Email   : caizefeng18@gmail.com
# @Software: PyCharm
import argparse
import os

import math
import numpy as np
import torch
from tqdm import tqdm

from deep_dft.utils import feature_utils, read_utils
from deep_dft.utils.datasets import AtomGridChargeDataset, BatchSVTChargeDataset


# save the files with all preprocessed feature
from deep_dft.utils.fs_utils import mkdir_without_override


def SVTFeatured_save(in_dir, out_dir, fileread_hp=None, preprocess_hp=None, ):
    os.chdir(out_dir)
    if fileread_hp is None:
        fileread_hp = {"root_dir": in_dir,
                       "rough_cutoff": 9, }
    if preprocess_hp is None:
        preprocess_hp = {"sigmas": torch.exp(torch.linspace(math.log(0.25), math.log(5), 16)),
                         "cutoff": 9 * torch.ones(3), "preprocess_batch": 50000,
                         "device": torch.device('cuda')}
    dataset_filewise = AtomGridChargeDataset(**fileread_hp, device=preprocess_hp["device"])
    for i in tqdm(range(len(dataset_filewise))):
        sample = dataset_filewise[i]
        file_name = os.path.basename(dataset_filewise.data_files[i]).split('_')[-1]
        dataset_batchwise = BatchSVTChargeDataset(sample, **preprocess_hp)
        grid_charge = []
        for j in tqdm(range(len(dataset_batchwise))):
            grid_charge_batch = torch.cat(dataset_batchwise[j], dim=1)
            grid_charge.append(grid_charge_batch.cpu())

        np.save(file_name, torch.cat(grid_charge, dim=0).numpy())


def save_parchg(in_dir, out_dir):
    os.chdir(out_dir)
    data_dirs = [x.path for x in os.scandir(in_dir) if x.name.startswith("LDOS")]
    for ldos_dir in tqdm(data_dirs):
        file_name = os.path.basename(ldos_dir)
        data_files = [x.path for x in os.scandir(ldos_dir) if x.name.startswith("PARCHG")]
        energy_array = np.asarray([float(os.path.basename(x).split('_')[-1]) for x in data_files])
        idx_array = np.argsort(energy_array)
        energy_array_sorted = np.sort(energy_array)
        np.savetxt(file_name + '.txt', energy_array_sorted)
        charge_arr_list = []
        for parchg_path in tqdm(data_files):
            vec, coor_list, chg, ngxf, ngyf, ngzf = read_utils.split_read_parchg(parchg_path)
            charge_arr = feature_utils.charge_label_numpy(vec, chg, ngxf, ngyf, ngzf)
            charge_arr_list.append(charge_arr)
        charge_arr_all = np.concatenate(charge_arr_list, axis=1)
        charge_arr_sorted = charge_arr_all[:, idx_array]
        np.save(file_name, charge_arr_sorted)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DeepDFT data preprocessing interface")
    parser.add_argument("-i", "--in_dir", help="The directary where the original CHGCARs/PARCHGs are")
    parser.add_argument("-o", "--out_dir", help="The directary where you wanna put your data files in")
    parser.add_argument("-c", "--cutoff", type=float, nargs='+', default=[9.0, ])
    parser.add_argument("--num_element", type=int, default=3)
    parser.add_argument("--gauss_min", type=float, default=0.25)
    parser.add_argument("--gauss_max", type=float, default=5)
    parser.add_argument("--gauss_num", type=int, default=16)
    parser.add_argument("-t", "--save_type", choices=['charge', 'ldos'], default='charge')
    args = parser.parse_args()

    if len(args.cutoff) == 1:
        cutoff_list = np.repeat(args.cutoff[0], args.num_element)
    else:
        cutoff_list = args.cutoff

    preprocess_hp = {
        "sigmas": torch.exp(torch.linspace(math.log(args.gauss_min), math.log(args.gauss_max), args.gauss_num)),
        "cutoff": torch.tensor(cutoff_list), "preprocess_batch": 50000,
        "device": torch.device('cuda:1')}

    mkdir_without_override(args.out_dir)

    if args.save_type == 'charge':
        SVTFeatured_save(args.in_dir, args.out_dir, preprocess_hp=preprocess_hp, )
    elif args.save_type == 'ldos':
        save_parchg(args.in_dir, args.out_dir)
    print("Transforming Finished: {}".format(args.out_dir))

# nohup python data_save.py -i /home/lzhpc/WORK/caizefeng/ML_DFT/Datasets/STO_600/CHGCAR_all -o /public/WORK_backup/caizefeng/Datasets/STO_600_cut9_gauss20_0.1_5/train -t charge &
# nohup python data_save.py -i /public/WORK_backup/caizefeng/Datasets/STO_600_cut9_gauss16/LDOS_all -o /public/WORK_backup/caizefeng/Datasets/STO_600_cut9_gauss20_0.1_5/train -t ldos &
