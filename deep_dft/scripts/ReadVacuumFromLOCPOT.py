#!/usr/bin/env python3
# @File    : ReadVacuumFromLOCPOT.py
# @Time    : 12/9/2020 5:22 PM
# @Author  : Zavier Cai
# @Email   : caizefeng18@gmail.com

import os
import subprocess

import numpy as np

if not os.path.exists('POTPAVG.dat'):
    create_potfile_command = "(echo 42; echo 426; echo 3) | vaspkit"
    subprocess.call(create_potfile_command, shell=True, stdout=subprocess.PIPE)

pot_array = np.loadtxt('PLANAR_AVERAGE.dat', skiprows=1, dtype=np.float64)
vacuum_level = np.max(pot_array[:, 1])
print(vacuum_level)
