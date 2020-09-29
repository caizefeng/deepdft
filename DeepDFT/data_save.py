#!/usr/bin/env python3
# @File    : data_from_chgcar.py
# @Time    : 9/12/2020 1:52 PM
# @Author  : Zavier Cai
# @Email   : caizefeng18@gmail.com
# @Software: PyCharm
import argparse
import math
import os

import numpy as np
import torch
from tqdm import tqdm

import io_utils
import feature_utils
from datasets import AtomGridChargeDataset, BatchSVTChargeDataset

# create files that only has the cartesian coordinates of grid points, corresponding charge values,
# and atom coordinates and their corresponding augmentation occupancies
# lattice vectors `vec` is now not needed to be involved in the dataset
def atom_grid_charge_save(root_dir, rough_cutoff, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.chdir(root_dir)
    with os.scandir(root_dir) as it:
        for entry in it:
            if entry.name.startswith("CHGCAR") and entry.is_file():
                vec, coor_list, chg, ngxf, ngyf, ngzf = io_utils.split_read_chgcar(entry.path)
                vec, coor_list, chg = feature_utils.np2torch(vec, coor_list, chg, device)
                atom_coor_list = feature_utils.dir2cart(vec, coor_list)
                atom_coor_list_pbc = []
                for atom_coor_elementwise in atom_coor_list:
                    atom_coor_list_pbc.append(
                        feature_utils.PBC_padding(atom_coor_elementwise, vec, cutoff=rough_cutoff))
                grid_coor = feature_utils.grid_gen(ngxf, ngyf, ngzf, vec)
                charge_arr = feature_utils.charge_label(vec, chg, ngxf, ngyf, ngzf)
                torch.save([atom_coor_list_pbc, torch.cat((grid_coor, charge_arr), dim=1)],
                           entry.name.split("_")[1] + ".pt")


# save the files with all preprocessed feature
def SVTFeatured_save(in_dir, out_dir, fileread_hp=None, preproess_hp=None):
    os.chdir(out_dir)
    if fileread_hp is None:
        fileread_hp = {"root_dir": in_dir,
                       "rough_cutoff": 9, "device": torch.device('cuda')}
    if preproess_hp is None:
        preprocess_hp = {"sigmas": torch.exp(torch.linspace(math.log(0.25), math.log(5), 16)),
                         "cutoff": torch.tensor([9., 9., 9.]), "preprocess_batch": 50000,
                         "device": torch.device('cuda')}
    dataset_filewise = AtomGridChargeDataset(**fileread_hp)
    for i in tqdm(range(len(dataset_filewise))):
        sample = dataset_filewise[i]
        file_name = os.path.basename(dataset_filewise.data_files[i]).split('_')[-1]
        dataset_batchwise = BatchSVTChargeDataset(sample, **preprocess_hp)
        grid_charge = []
        for j in tqdm(range(len(dataset_batchwise))):
            grid_charge_batch = torch.cat(dataset_batchwise[j], dim=1)
            grid_charge.append(grid_charge_batch.cpu())

        np.save(file_name, torch.cat(grid_charge, dim=0).numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DeepDFT data dumping interface")
    parser.add_argument("-i", "--in_dir", help="The directary where the original CHGCARs are")
    parser.add_argument("-o", "--out_dir", help="The directary where you wanna put your data files in")
    args = parser.parse_args()
    SVTFeatured_save(args.in_dir, args.out_dir)
    # data_from_chgcar_save("/home/lzhpc/WORK/caizefeng/ML_DFT/Datasets/test", 9)

# fileread_hp = {"root_dir": "/home/lzhpc/WORK/caizefeng/ML_DFT/Datasets/test",
#                "rough_cutoff": 9, "device": torch.device('cuda')}
# preprocess_hp = {"sigmas": torch.exp(torch.linspace(math.log(0.25), math.log(5), 16)),
#                  "cutoff": torch.tensor([9., 9., 9.]), "preprocess_batch": 50000, "device": torch.device('cuda')}
