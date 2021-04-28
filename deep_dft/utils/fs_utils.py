#!/usr/bin/env python3
# @File    : fs_utils.py
# @Time    : 12/11/2020 12:39 PM
# @Author  : Zavier Cai
# @Email   : caizefeng18@gmail.com
import os
import shutil


def mkdir_without_override(path):
    if not os.path.exists(path):
        os.makedirs(path)


def move_to_dir_without_override(in_path, out_dir):
    if not os.path.exists(os.path.join(out_dir, os.path.basename(in_path))):
        shutil.move(in_path, out_dir)
