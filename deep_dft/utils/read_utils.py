#!/usr/bin/env python3
# @File    : read_chgcar.py
# @Time    : 9/9/2020 5:02 PM
# @Author  : Zavier Cai
# @Email   : caizefeng18@gmail.com
# @Software: PyCharm

import os
import subprocess

import numpy as np


def split_read_chgcar(chgcar_path):
    """read number of grids, lattice vectors, atom positions, charge density at each grids from CHGCAR"""
    with open(chgcar_path, 'r') as f:
        f_vec = open(chgcar_path + '_vec', 'w')
        f_chg = open(chgcar_path + '_chg', 'w')
        for line_num, line in enumerate(f):
            if line_num < 8:
                if line_num == 1:
                    scaling = float(line.split()[0])
                elif (line_num >= 2) and (line_num <= 4):
                    f_vec.write(line.rstrip() + '\n')
                elif line_num == 5:
                    atom_list = line.split()
                    atom_type_num = len(atom_list)
                elif line_num == 6:
                    atom_num_list = [int(i) for i in line.split()]
                    atom_num = sum(atom_num_list)
                    phantom_list = list(atom_num_list)  # copy the list
                    phantom_list.insert(0, 0)
                    # for cumulative summation after
                    phantom_list_np = np.cumsum(np.array(phantom_list))
                    f_coor = list(range(atom_type_num))
                    for index in range(atom_type_num):
                        f_coor[index] = open(
                            chgcar_path + '_coor_' + str(index), 'w')

            else:
                if line_num < 10 + atom_num:
                    if line_num <= (7 + atom_num):
                        j = 0
                        while j <= atom_type_num - 1:
                            if (8 + phantom_list_np[j]) <= line_num <= (
                                    7 + phantom_list_np[j + 1]):
                                f_coor[j].write(line.rstrip() + '\n')
                                break
                            j += 1
                    elif line_num == (9 + atom_num):
                        ngxf, ngyf, ngzf = [int(i) for i in line.split()]
                        ngf = ngxf * ngyf * ngzf
                        chg_line_num = ngf // 5
                        chg_space = 5 - ngf % 5
                else:
                    if (line_num >=
                        (10 + atom_num)) and (line_num <=
                                              (9 + atom_num + chg_line_num)):
                        f_chg.write(line.rstrip() + '\n')
                    elif chg_space != 5:
                        # lines is multiples of 5 or not determines whether completing is needed
                        if line_num == (10 + atom_num + chg_line_num):
                            f_chg.write(line.rstrip() + chg_space * ' 0' +
                                        '\n')
        f_vec.close()
        for i in range(atom_type_num):
            f_coor[i].close()
        f_chg.close()

    vec = scaling * np.loadtxt(chgcar_path + '_vec', dtype=np.float64)
    coor_list = []
    for index in range(atom_type_num):
        coor_list.append(
            np.loadtxt(chgcar_path + '_coor_' + str(index), dtype=np.float64))
    # charge arrays are too big to "define in advance and fill out"
    chg = np.loadtxt(chgcar_path + '_chg', dtype=np.float64)

    # remove auxiliary files
    os.remove(chgcar_path + '_vec')
    os.remove(chgcar_path + '_chg')
    for index in range(atom_type_num):
        os.remove(chgcar_path + '_coor_' + str(index))

    return vec, coor_list, chg, ngxf, ngyf, ngzf


def split_read_parchg(parchg_path):
    """read number of grids, lattice vectors, atom positions, charge density at each grids from CHGCAR"""
    with open(parchg_path, 'r') as f:
        f_vec = open(parchg_path + '_vec', 'w')
        f_chg = open(parchg_path + '_chg', 'w')
        for line_num, line in enumerate(f):
            if line_num < 8:
                if line_num == 1:
                    scaling = float(line.split()[0])
                elif (line_num >= 2) and (line_num <= 4):
                    f_vec.write(line.rstrip() + '\n')
                elif line_num == 5:
                    atom_list = line.split()
                    atom_type_num = len(atom_list)
                elif line_num == 6:
                    atom_num_list = [int(i) for i in line.split()]
                    atom_num = sum(atom_num_list)
                    phantom_list = list(atom_num_list)  # copy the list
                    phantom_list.insert(0, 0)
                    # for cumulative summation after
                    phantom_list_np = np.cumsum(np.array(phantom_list))
                    f_coor = list(range(atom_type_num))
                    for index in range(atom_type_num):
                        f_coor[index] = open(
                            parchg_path + '_coor_' + str(index), 'w')

            else:
                if line_num < 10 + atom_num:
                    if line_num <= (7 + atom_num):
                        j = 0
                        while j <= atom_type_num - 1:
                            if (8 + phantom_list_np[j]) <= line_num <= (
                                    7 + phantom_list_np[j + 1]):
                                f_coor[j].write(line.rstrip() + '\n')
                                break
                            j += 1
                    elif line_num == (9 + atom_num):
                        ngxf, ngyf, ngzf = [int(i) for i in line.split()]
                        ngf = ngxf * ngyf * ngzf
                        chg_line_num = ngf // 10
                        chg_space = 10 - ngf % 10
                else:
                    if (line_num >=
                        (10 + atom_num)) and (line_num <=
                                              (9 + atom_num + chg_line_num)):
                        f_chg.write(line.rstrip() + '\n')
                    elif chg_space != 10:
                        # lines is multiples of 10 or not determines whether completing is needed
                        if line_num == (10 + atom_num + chg_line_num):
                            f_chg.write(line.rstrip() + chg_space * ' 0' +
                                        '\n')
        f_vec.close()
        for i in range(atom_type_num):
            f_coor[i].close()
        f_chg.close()

    vec = scaling * np.loadtxt(parchg_path + '_vec', dtype=np.float64)
    coor_list = []
    for index in range(atom_type_num):
        coor_list.append(
            np.loadtxt(parchg_path + '_coor_' + str(index), dtype=np.float64))
    # charge arrays are too big to "define in advance and fill out"
    chg = np.loadtxt(parchg_path + '_chg', dtype=np.float64)

    # remove auxiliary files
    os.remove(parchg_path + '_vec')
    os.remove(parchg_path + '_chg')
    for index in range(atom_type_num):
        os.remove(parchg_path + '_coor_' + str(index))

    return vec, coor_list, chg, ngxf, ngyf, ngzf


def read_occ(chgcar_path):
    """read augmentation density of each channel(alpha,beta) of each atom from CHGCAR"""
    tail_command = ' '.join(('head -n 7', chgcar_path, '| tail -n 1'))
    sub = subprocess.Popen(tail_command, shell=True, stdout=subprocess.PIPE)
    atom_num_list = [int(i) for i in str(sub.stdout.read(), 'utf-8').split()]
    sub.kill()

    phantom_list = list(atom_num_list)
    phantom_list.insert(0, 1)
    index_array = np.cumsum(np.array(phantom_list))
    component_num_list = []
    occ_list = []
    # get number of components of each elements
    for first_index in index_array[:-1]:  # get rid of the last one
        grep_command_1 = 'grep \'augmentation occupancies{:>4d}\' {}'.format(
            first_index, chgcar_path)
        sub = subprocess.Popen(grep_command_1,
                               shell=True,
                               stdout=subprocess.PIPE)
        component_num = int(str(sub.stdout.read(), 'utf-8').split()[3])
        sub.kill()
        component_num_list.append(component_num)

    # initialize list(element) and array(atom)
    for i in range(len(atom_num_list)):
        occ_list.append(np.zeros((atom_num_list[i], component_num_list[i])))

    # extend number of components of each "element" to each "atom"
    component_num_arr = np.array(component_num_list).repeat(atom_num_list)

    for i in range(len(atom_num_list)):
        for atom_index in range(atom_num_list[i]):
            atom_index_nonpython = int(atom_index) + 1 + sum(
                atom_num_list[0:i])
            component_num = component_num_arr[int(atom_index) +
                                              sum(atom_num_list[0:i])]
            if component_num % 5 == 0:
                line_num_str = str(component_num // 5)
            else:
                line_num_str = str(component_num // 5 + 1)
            grep_command_2 = 'grep \'augmentation occupancies{:>4d}\' {} -A {} | tail -{}'.format(
                atom_index_nonpython, chgcar_path, line_num_str, line_num_str)
            sub = subprocess.Popen(grep_command_2,
                                   shell=True,
                                   stdout=subprocess.PIPE)
            occ = np.array(
                [float(i) for i in str(sub.stdout.read(), 'utf-8').split()])
            sub.kill()
            occ_list[i][atom_index:] = occ
    return occ_list
