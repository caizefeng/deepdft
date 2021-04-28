#!/usr/bin/env python3
# @File    : XDAT2POS.py
# @Time    : 11/3/2020 12:20 PM
# @Author  : Zavier Cai
# @Email   : caizefeng18@gmail.com

# %%
import argparse
import os
import subprocess

import numpy as np
from numpy.distutils.fcompiler import str2bool
from tqdm import tqdm


# %%

def xdat2pos(dir_name, is_sample=True, sample_num=10, isif=2):

    os.chdir(dir_name)

    # get number of atom in the POSCAR
    xdat_name = 'XDATCAR'
    tail_command = ' '.join(('head -n 7', xdat_name, '| tail -n 1'))
    sub = subprocess.Popen(tail_command, shell=True, stdout=subprocess.PIPE)
    atom_num = np.sum([int(i) for i in str(sub.stdout.read(), 'utf-8').split()])
    sub.kill()
    coor_length = atom_num + 1
    if isif == 2:
        main_length = coor_length

        # get vector part (the first 7 lines in XDATCAR)
        tail_command = ' '.join(('head -n 7', xdat_name))
        sub = subprocess.Popen(tail_command, shell=True, stdout=subprocess.PIPE)
        # Popen is non-blocking but read() is blocking
        vec_part = str(sub.stdout.read(), 'utf-8')
        sub.kill()
    elif isif == 3:
        main_length = atom_num + 8
    else:
        raise ValueError("Only isif = 2 / 3 is supported")

    # get total number of snapshots in the XDATCAR
    tail_command = ' '.join(('tail -n', str(coor_length), xdat_name, '| head -n 1'))
    sub = subprocess.Popen(tail_command, shell=True, stdout=subprocess.PIPE)
    snap_num = int(str(sub.stdout.read(), 'utf-8').split()[-1])
    sub.kill()

    if is_sample:
        sample_index_arr = np.random.choice(snap_num, sample_num, replace=False) + 1
    else:
        sample_index_arr = np.arange(snap_num) + 1
        sample_num = snap_num

    f_snap = list(range(sample_num))
    os.system('rm -rf POSCAR_all')
    os.system('mkdir POSCAR_all')

    # write each POSCAR and move it to dir /POSCAR_all in working dir
    for i, index in tqdm(enumerate(np.flip(sample_index_arr))):
        indexline = main_length * index
        tail_command = ' '.join(('tail -n', str(indexline), xdat_name, '| head -n', str(main_length)))
        sub = subprocess.Popen(tail_command, shell=True, stdout=subprocess.PIPE)
        coor_part = str(sub.stdout.read(), 'utf-8')
        sub.kill()
        f_name = 'POSCAR' + '_' + str(i + 10000)  # +10000 for decent sorting in Linux
        f_snap[i] = open(f_name, 'w')
        if isif == 2:
            f_snap[i].write(vec_part)
        f_snap[i].write(coor_part)
        f_snap[i].close()
        os.system('mv ' + f_name + ' POSCAR_all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract selected POSCARs from one XDATCAR")
    parser.add_argument("-d", "--dir_name", default=".")
    parser.add_argument("-i", "--is_sample", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-s", "--sample_num", type=int, default=10)
    parser.add_argument("-f", "--isif", type=int, choices=[2, 3], default=2)

    args = parser.parse_args()
    xdat2pos(args.dir_name, args.is_sample, sample_num=args.sample_num, isif=args.isif)
