#!/usr/bin/env python3
# @File    : utils.py
# @Time    : 9/9/2020 8:19 PM
# @Author  : Zavier Cai
# @Email   : caizefeng18@gmail.com
# @Software: PyCharm


import math

import torch


# determine the number of the needed padding in each lattice direction
# according to the point-to-plane distance and cutoff in each direction
def padding_from_cutoff(vec, cutoff=9):
    device = vec.device
    volume = torch.det(vec)

    distance = torch.tensor([volume / (vec[1, :].cross(vec[2, :]).norm()),
                             volume / (vec[0, :].cross(vec[2, :]).norm()),
                             volume / (vec[0, :].cross(vec[1, :]).norm())],
                            device=device)
    padding = (cutoff / (distance.abs())).ceil().to(torch.int32)
    return padding


# padding the addtional cells to satisfy periodic boundary condition
# (inputs are cartesian coordinates and lattice vectors)
def PBC_padding(coor, vec, cutoff=9):
    device = coor.device
    padding = padding_from_cutoff(vec, cutoff)
    lattice_num_1 = padding[0] * 2 + 1
    lattice_num_2 = padding[1] * 2 + 1
    lattice_num_3 = padding[2] * 2 + 1
    lattice_num = lattice_num_1 * lattice_num_2 * lattice_num_3
    atom_num = coor.size()[0]
    coor_PBC = torch.zeros(atom_num * lattice_num, 3, device=device)
    coor_PBC[0:atom_num] = coor  # make sure original coor is the first part

    i = atom_num  # location of the first atom in one coor_transfer
    for x1_move in range(-padding[0], padding[0] + 1):
        for x2_move in range(-padding[1], padding[1] + 1):
            for x3_move in range(-padding[2], padding[2] + 1):
                if x1_move == 0 and x2_move == 0 and x3_move == 0:
                    continue
                else:
                    coor_moved = coor.clone().detach()
                    coor_moved[:, 0:3] += (vec[0] * x1_move + vec[1] * x2_move + vec[2] * x3_move)
                    coor_PBC[i:i + atom_num] = coor_moved
                    i += atom_num
    return coor_PBC


# generate the coordinates of grid by splitting the cell linearly
def grid_gen(ngxf, ngyf, ngzf, vec):
    grid_4d = torch.stack(torch.meshgrid(torch.linspace(0, 1, ngzf + 1)[:-1],
                                         torch.linspace(0, 1, ngyf + 1)[:-1],
                                         torch.linspace(0, 1, ngxf + 1)[:-1]))
    grid_coor = grid_4d.view(3, -1).T.flip(1).to(vec.device).matmul(vec.to(torch.float32))

    return grid_coor


# generate a tensor used to product with the input to representing cutoff function
def cutoff_gen(tensor_in, *r_cutoff, simple_cut=True):
    device = tensor_in.device
    if simple_cut:
        return torch.where(tensor_in < r_cutoff[0], torch.tensor(1, device=device), torch.tensor(0, device=device))
    else:
        r_cs = r_cutoff[0]
        r_cut = r_cutoff[1]
        return 1 / 2 * torch.cos(math.pi * (tensor_in - r_cs) / (r_cut - r_cs)) + 1 / 2


# transfer all data from NDArray to torch.Tensor
def np2torch(vec, coor_list, chg, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.from_numpy(vec).to(device), \
           [torch.from_numpy(i).to(device) for i in coor_list], \
           torch.from_numpy(chg).to(device)


# change primitive date in lattice coordinate (VASP default) to Cartesian coordinate
def dir2cart(vec, coor_list):
    return [coor.matmul(vec) for coor in coor_list]


# generate the difference of coordinates in 3 components x y z.
# fill the 1st layer of dist_arr by cross distance between grid coordinates and atom coordinates
def dist_gen(grid_coor, atom_coor):
    device = grid_coor.device
    dist_arr = torch.zeros(grid_coor.size()[0], atom_coor.size()[0], 4, device=device)
    for axis in range(3):
        for atom_index in range(atom_coor.size()[0]):
            dist_arr[:, atom_index, axis + 1] = atom_coor[atom_index, axis] - grid_coor[:, axis]
    dist_arr[:, :, 0] = torch.cdist(grid_coor, atom_coor, p=2, compute_mode="donot_use_mm_for_euclid_dist")
    return dist_arr


# split array into batches(tuple) to prevent memory error when processing it to generate descriptors
def batch_gen(input_arr, batch_size):
    # batch_num = dist_arr.size()[0] // batch_size + 1
    return torch.split(input_arr, batch_size, dim=0)


# divide charge values in CHGCAR by volumn of cell to generate real density
def charge_label(vec, chg, ngxf, ngyf, ngzf):
    cell_volume = torch.det(vec)
    chg_den_flat = chg.flatten() / cell_volume
    # the number of values in `chg` is completed to a multiple of 5
    # for convenience, now is the time to strip them away
    true_chg_den_flat = chg_den_flat[0:ngxf * ngyf * ngzf]
    charge_arr = true_chg_den_flat.view(-1, 1)
    return charge_arr


# generate the rudiment array with rotation-variant data, used to generate invariant feature
def des_initial_gen(dist_arr, sigmas, cutoff_distance):
    device = dist_arr.device
    sigma_size = sigmas.numel()
    s_ini = torch.zeros(dist_arr.size()[0], sigma_size, device=device)  # scalar
    v_ini = torch.zeros(dist_arr.size()[0], 3, sigma_size, device=device)  # vector
    t_ini = torch.zeros(dist_arr.size()[0], 3, 3, sigma_size, device=device)  # tensor
    tensor_cutoff = cutoff_gen(dist_arr[:, :, 0], cutoff_distance)
    # gauss_numerator = torch.exp(-dist_arr[:, :, 0] ** 2) * tensor_cutoff

    for index, sigma in enumerate(sigmas):
        c_k = 1 / ((2 * math.pi) ** 1.5 * sigma ** 3)  # (16,)
        gauss_denominator = 2 * sigma ** 2  # (16,)
        # gauss = torch.exp(-dist_arr[:, :, 0] ** 2) ** (1 / gauss_denominator) * tensor_cutoff
        gauss = torch.exp(-dist_arr[:, :, 0] ** 2 / gauss_denominator) * tensor_cutoff
        # so the iteration cannot be replaced by boardcasting

        s_ini[:, index] = c_k * torch.sum(gauss, 1)
        for i in range(3):  # i = x, y, z
            v_ini[:, i, index] = c_k * torch.sum(dist_arr[:, :, i + 1] / gauss_denominator * gauss, 1)
        for i in range(3):  # i, j = x, y, z
            for j in range(3):
                t_ini[:, i, j, index] = c_k * torch.sum(
                    dist_arr[:, :, i + 1] * dist_arr[:, :, j + 1] / gauss_denominator ** 2 * gauss, 1)
    return s_ini, v_ini, t_ini


# process the rudiment array for rotational invariance, des_arr with shape (grid, feature)
def invariance_gen(s_ini, v_ini, t_ini):
    device = s_ini.device
    sigma_size = s_ini.size()[1]  # the number of all sigmas
    descriptor_arr = torch.zeros(s_ini.size()[0], 5 * sigma_size, device=device)

    descriptor_arr[:, 0:sigma_size] = s_ini[:, :]
    descriptor_arr[:, sigma_size: 2 * sigma_size] = torch.sqrt(torch.sum(v_ini ** 2, 1))
    descriptor_arr[:, 2 * sigma_size: 3 * sigma_size] = t_ini[:, 0, 0, :] + \
                                                        t_ini[:, 1, 1, :] + t_ini[:, 2, 2, :]
    descriptor_arr[:, 3 * sigma_size: 4 * sigma_size] = t_ini[:, 0, 0, :] * \
                                                        t_ini[:, 1, 1, :] + t_ini[:, 1, 1, :] * t_ini[:, 2, 2, :] \
                                                        + t_ini[:, 0, 0, :] * t_ini[:, 2, 2, :] - \
                                                        t_ini[:, 0, 1, :] ** 2 - t_ini[:, 1, 2, :] ** 2 \
                                                        - t_ini[:, 0, 2, :] ** 2
    descriptor_arr[:, 4 * sigma_size: 5 * sigma_size] = \
        t_ini[:, 0, 0, :] * (
                t_ini[:, 1, 1, :] * t_ini[:, 2, 2, :] - t_ini[:, 1, 2, :] * t_ini[:, 2, 1, :]) \
        - t_ini[:, 0, 1, :] * (
                t_ini[:, 1, 0, :] * t_ini[:, 2, 2, :] - t_ini[:, 1, 2, :] * t_ini[:, 2, 0, :]) \
        + t_ini[:, 0, 2, ] * (
                t_ini[:, 1, 0, :] * t_ini[:, 2, 1, :] - t_ini[:, 1, 1, :] * t_ini[:, 2, 0, :])

    return descriptor_arr


# a function used to standardize feature
def standardize(input):
    return (input - input.mean(0)) / input.std(0)

# %%
# class SVTFeature(object):
#     def __init__(self, cutoff, sigmas, preprocessing_batch=50000):
#         self.cutoff = cutoff
#         self.sigmas = sigmas
#         self.sigma_size = sigmas.numel()
#         self.preprocessing_batch = preprocessing_batch
#
#     def __call__(self, sample):
#         atom_coor_list, grid_coor = sample["atom"], sample["grid"]
#         descriptor_arr_all = torch.zeros(grid_coor.size()[0], 5 * self.sigma_size * len(atom_coor_list))
#         for i, atom_coor in enumerate(atom_coor_list):  # through all elements
#             feature_start_index = 5 * self.sigma_size * i  # starting index of each element，5=1s+1v+3t
#             feature_end_index = 5 * self.sigma_size * (i + 1)  # ending index of each element
#             dist_arr_mid = feature_utils.dist_gen(grid_coor, atom_coor)
#             dist_arr_list = feature_utils.batch_gen(dist_arr_mid, self.preprocessing_batch)
#             instance_index = 0
#             for dist_arr in dist_arr_list:
#                 step = dist_arr.size()[0]  # step is the number of grid in each batch
#                 s_ini, v_ini, t_ini = des_initial_gen(dist_arr, self.sigmas, self.cutoff[i])
#                 descriptor_arr = invariance_gen(s_ini, v_ini, t_ini)
#                 descriptor_arr_all[instance_index:(instance_index + step), feature_start_index:feature_end_index] = descriptor_arr
#                 instance_index += step
#
#         return  descriptor_arr_all
