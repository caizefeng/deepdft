#!/usr/bin/env python3
# @File    : string_utils.py
# @Time    : 12/11/2020 12:38 PM
# @Author  : Zavier Cai
# @Email   : caizefeng18@gmail.com
import argparse
from collections import OrderedDict
from typing import Dict, List, Iterable


def str2bool(v):
    """function that can be used by argparse as `type` argument"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def gen_name(tag: str, run_dict: Dict, run_extra: List):
    """function to generate names for each run in TensorBoard"""
    name_dict = OrderedDict()
    name = tag
    name_dict["tag"] = 1
    for key, value in run_dict.items():
        if isinstance(value, str):
            value_in_name = value
            name = '_'.join((name, value_in_name))
            name_dict[key] = 1
        elif isinstance(value, Iterable):
            value_in_name = [str(i) for i in value]
            name = '_'.join((name, *value_in_name))
            name_dict[key] = len(value)
        else:
            value_in_name = str(value)
            name = '_'.join((name, value_in_name))
            name_dict[key] = 1
    name = '_'.join((name, *run_extra))
    name_dict["extra"] = len(run_extra)
    return name, name_dict


def expand_list_in_dict(input_dict):
    """split the keys whose value is a list, e.g. {"a":[1,2,3], } => {"a1":1, "a2":2, "a3":3, }"""

    add_dict = {}
    redundant_key_list = []
    for key, value in input_dict.items():
        if isinstance(value, List):
            for idx, single_value in enumerate(value):
                add_dict['_'.join((key, str(idx)))] = single_value
            redundant_key_list.append(key)
    for key in redundant_key_list:
        del input_dict[key]
    input_dict.update(add_dict)
    return input_dict
