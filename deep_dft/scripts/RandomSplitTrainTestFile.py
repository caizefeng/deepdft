#!/usr/bin/env python3
# @File    : RandomSplitTrainTestFile.py
# @Time    : 12/11/2020 4:19 PM
# @Author  : Zavier Cai
# @Email   : caizefeng18@gmail.com
import argparse
import os
import random
import re
import shutil

from deep_dft.utils.fs_utils import mkdir_without_override, move_to_dir_without_override
from deep_dft.utils.string_utils import str2bool


def random_split_train_test_file(data_dir, des_dir, is_ratio=False, split_option=None):
    """randomly split files to training and testing directories"""
    mkdir_without_override(os.path.join(des_dir, "train"))
    mkdir_without_override(os.path.join(des_dir, "test"))

    file_dict = {}
    for entry in os.scandir(data_dir):
        if entry.is_file():
            number_tag = re.compile(r'\d+').findall(entry.name)[0]
            if number_tag in file_dict:
                file_dict[number_tag].append(entry.path)
            else:
                file_dict[number_tag] = [entry.path, ]

    num_all = len(file_dict)
    if is_ratio and 0 < split_option < 1:
        num_test = int(num_all * split_option)
    elif not is_ratio and split_option < num_all:
        num_test = int(split_option)
    else:
        raise ValueError(
            "`split_option` should be between 0 and 1 if `is_ratio` is True, "
            " or smaller than the total number of data files if `is_ratio` is False!")

    test_keys = random.sample(list(file_dict.keys()), num_test)
    for test_key in test_keys:
        for file_path in file_dict[test_key]:
            move_to_dir_without_override(file_path, os.path.join(des_dir, "test"))
        del file_dict[test_key]
    for train_key in file_dict.keys():
        for file_path in file_dict[train_key]:
            move_to_dir_without_override(file_path, os.path.join(des_dir, "train"))

    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Randomly split files to training and testing directories")
    parser.add_argument("-i", "--in_dir", )
    parser.add_argument("-o", "--out_dir", )
    parser.add_argument("-r", "--is_ratio", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-s", "--split", type=float, default=0.2)
    args = parser.parse_args()

    random_split_train_test_file(args.in_dir, args.out_dir, args.is_ratio, args.split)
