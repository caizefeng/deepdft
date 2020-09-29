#!/usr/bin/env python3
# @File    : test.py
# @Time    : 9/9/2020 5:14 PM
# @Author  : Zavier Cai
# @Email   : caizefeng18@gmail.com
# @Software: PyCharm

# vec, coor_list, chg, ngxf, ngyf, ngzf, _, _ = split_read_chgcar("..", "CHGCAR_10000")
# print(vec)
import argparse
import time

import torch

import re

# a = torch.rand(500000, 240)
# b = torch.rand(500000, 240).cuda()
# a = torch.rand(500000, 240)
a = torch.rand(2000,1000)
c1 = time.time()
for _ in range(100000):
    b = a.size()[0]
print(time.time() - c1)

c1 = time.time()
for _ in range(100000):
    b2 = a.size(0)
print(time.time() - c1)




