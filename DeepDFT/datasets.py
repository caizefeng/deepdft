#!/usr/bin/env python3
# @File    : datasets_gen.py
# @Time    : 9/12/2020 3:39 PM
# @Author  : Zavier Cai
# @Email   : caizefeng18@gmail.com
# @Software: PyCharm

import os

import numpy as np
import torch
from torch.utils.data import Dataset

import feature_utils
import io_utils


# extract atom coordinates, charge density values from CHGCAR
# generate grid points according to NGF
def data_from_chgcar(file_path, rough_cutoff, device):
    vec, coor_list, chg, ngxf, ngyf, ngzf = io_utils.split_read_chgcar(file_path)
    vec, coor_list, chg = feature_utils.np2torch(vec, coor_list, chg, device)
    atom_coor_list = feature_utils.dir2cart(vec, coor_list)
    atom_coor_list_pbc = []
    for atom_coor_elementwise in atom_coor_list:
        atom_coor_list_pbc.append(
            feature_utils.PBC_padding(atom_coor_elementwise, vec, cutoff=rough_cutoff))
    grid_coor = feature_utils.grid_gen(ngxf, ngyf, ngzf, vec)
    charge_arr = feature_utils.charge_label(vec, chg, ngxf, ngyf, ngzf)
    return [atom_coor_list_pbc, grid_coor, charge_arr]


# the total feature-generating routine
class SVTFeature(object):
    def __init__(self, cutoff, sigmas, device):
        self.device = device
        self.cutoff = cutoff
        self.sigmas = sigmas
        self.sigma_size = sigmas.numel()

    def __call__(self, atom_coor_list, grid_coor):
        descriptor_arr_all = torch.zeros(grid_coor.size(0), 5 * self.sigma_size * len(atom_coor_list),
                                         device=self.device)
        # print(descriptor_arr_all.device)
        for i, atom_coor in enumerate(atom_coor_list):  # through all elements
            feature_start_index = 5 * self.sigma_size * i  # starting index of each element，5=1s+1v+3t
            feature_end_index = 5 * self.sigma_size * (i + 1)  # ending index of each element
            dist_arr_middle = feature_utils.dist_gen(grid_coor, atom_coor)
            s_ini, v_ini, t_ini = feature_utils.des_initial_gen(dist_arr_middle, self.sigmas, self.cutoff[i])
            descriptor_arr = feature_utils.invariance_gen(s_ini, v_ini, t_ini)
            # print(descriptor_arr.device)
            descriptor_arr_all[:, feature_start_index:feature_end_index] = descriptor_arr

        # print(descriptor_arr_all.device)

        return descriptor_arr_all


# every indexing loads data from one single CHGCAR, could be used in visualization
class AtomGridChargeDataset(Dataset):
    def __init__(self, root_dir, rough_cutoff=9, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.data_files = [x.path for x in os.scandir(root_dir) if x.name.startswith("CHGCAR")]
        self.rough_cutoff = rough_cutoff

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        atom_coor_list, grid_coor, charge_arr = data_from_chgcar(self.data_files[idx],
                                                                 self.rough_cutoff, device=self.device)
        sample_dict = {"atom": atom_coor_list, "grid": grid_coor, "charge": charge_arr}
        return sample_dict


# have to preprocess raw data by chunks
class BatchSVTChargeDataset(Dataset):
    def __init__(self, sample, sigmas, cutoff=9 * torch.ones(3), preprocess_batch=50000, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.atom_list = sample["atom"]
        self.grid_list = torch.split(sample["grid"], preprocess_batch)
        self.charge_list = torch.split(sample["charge"], preprocess_batch)
        self.transform = SVTFeature(cutoff, sigmas, self.device)

    def __len__(self):
        return len(self.grid_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.transform(self.atom_list, self.grid_list[idx]), self.charge_list[idx]


# dataset from one file with memory mapping to reduce RSS
class MmapDataset2D(Dataset):
    def __init__(self, file_path, num_column, data_type="float64", offset=128):
        self.mmap = np.memmap(file_path, dtype=data_type, mode="r", offset=offset).reshape(-1, num_column)

    def __len__(self):
        return len(self.mmap)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = torch.tensor(self.mmap[idx])
        sample_tuple = tuple([sample[:-1], sample[-1]])

        return sample_tuple
