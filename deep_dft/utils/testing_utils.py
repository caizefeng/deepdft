#!/usr/bin/env python3
# @File    : test_utils.py
# @Time    : 9/12/2020 5:40 PM
# @Author  : Zavier Cai
# @Email   : caizefeng18@gmail.com
# @Software: PyCharm

import torch


def mat_z(theta):
    """rotation matrice to verify the invariance of the feature"""

    return torch.tensor([[torch.cos(theta), torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0],
                         [0, 0, 1]])


def mat_y(theta):
    """rotation matrice to verify the invariance of the feature"""

    return torch.tensor([[torch.cos(theta), 0, torch.sin(theta)],
                         [0, 1, 0],
                         [-torch.sin(theta), 0, torch.cos(theta)]])


def mat_x(theta):
    """rotation matrice to verify the invariance of the feature"""

    return torch.tensor([[1, 0, 0],
                         [0, torch.cos(theta), -torch.sin(theta)],
                         [0, torch.sin(theta), torch.cos(theta)]])
