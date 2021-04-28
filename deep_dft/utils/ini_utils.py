#!/usr/bin/env python3
# @File    : ini_IO.py
# @Time    : 12/13/2020 11:17 AM
# @Author  : Zavier Cai
# @Email   : caizefeng18@gmail.com


import configparser


def read_section_as_dict(ini_path, section):
    cf = configparser.ConfigParser()
    cf.read(ini_path, encoding="utf-8")
    config_dict = {}
    for key, value in cf.items(section):
        config_dict[key] = value
    return config_dict


def read_or_write_default(ini_path, input_option_name, input_option):
    """read an option in "default" section if it is provided, otherwise set it"""
    cf = configparser.ConfigParser()
    cf.read(ini_path, encoding="utf-8")
    if input_option:
        cf.set("default", input_option_name, input_option)
        cf.write(open(ini_path, 'w'))
        return input_option
    else:
        return cf.get("default", input_option_name)


def read_ini_as_dict(ini_path):
    cf = configparser.ConfigParser()
    cf.read(ini_path, encoding="utf-8")
    config_dict_all = {}
    for section in cf.sections():
        config_dict = {}
        for key, value in cf.items(section):
            config_dict[key] = value
        config_dict_all[section] = config_dict
    return config_dict_all


def modify_one_section_by_dict(ini_path, section, wm_dict):
    cf = configparser.ConfigParser()
    cf.read(ini_path, encoding="utf-8")
    for key in wm_dict.keys():
        cf.set(section, key, wm_dict[key])
    cf.write(open(ini_path, 'w'))


if __name__ == '__main__':
    print(read_section_as_dict('/home/zhaogh/czf/DeepDFT/DeepDFT/configs/workload.ini', 'whut126')["export"])
