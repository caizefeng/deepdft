#!/usr/bin/env python3
# @File    : auto_vasp_utils.py
# @Time    : 12/10/2020 6:34 PM
# @Author  : Zavier Cai
# @Email   : caizefeng18@gmail.com
import os
import subprocess
import sys
from string import Template

import numpy as np

from deep_dft.utils.string_utils import str2bool


def change_pp(config_template_dir, wm_dict, pp='PAW'):
    """Change pseudo potential before Vaspkit processing and VASP calculation"""
    if pp == 'PAW':
        vaspkit_dict = {"POTCAR_TYPE": "PBE",
                        "RECOMMENDED_POTCAR": ".TRUE."}
    elif pp == 'USPP':
        # already set LDA_PATH = ~/soft/Pseudopotentials/US in ~/.vaspkit
        vaspkit_dict = {"POTCAR_TYPE": "LDA",
                        "RECOMMENDED_POTCAR": ".FALSE."}
    else:
        raise ValueError("Only two types of pseudo potential (PAW or USPP) are available now")
    vaspkit_dict.update(wm_dict)
    vaspkit_proto_path = os.path.join(config_template_dir, "vaspkit_template")
    vaspkit_out_path = os.path.expanduser(os.path.join('~', '.vaspkit'))
    fill_templates(out_path=vaspkit_out_path, proto_path_list=[vaspkit_proto_path, ], all_dict=vaspkit_dict)


def vasp_in_files_prep(calculate_dir, template_path, kit_path, wm_dict, vasp_dict, calculation_dict, dos_range,
                       window_interval, offset):
    """Prepare all input files for VASP"""
    incar_template_dir = os.path.join(template_path, 'incar_templates')
    submit_template_dir = os.path.join(template_path, 'submit_templates')
    incar_gen(calculate_dir, incar_template_dir, incar_type=calculation_dict["incar"])
    ptc_kpt_gen(calculate_dir, k_spacing=float(vasp_dict["static"]["k_spacing"]),
                aimd=str2bool(calculation_dict["gamma"]))
    if calculation_dict["submit"] == "fermi":
        ldos_submit_gen_fermi(calculate_dir, submit_template_dir, wm_dict, dos_range, window_interval)
    elif calculation_dict["submit"] == "vacuum":
        ldos_submit_gen_vacuum(calculate_dir, submit_template_dir, kit_path, wm_dict, dos_range,
                               window_interval,
                               offset)
    else:
        submit_gen_normal(calculate_dir, submit_template_dir, wm_dict, calculation_dict["submit"])


def incar_gen(calculate_dir, incar_template_dir, incar_type='aimd'):
    """generate INCAR"""
    # incar_type could be 'aimd', 'scf', 'parchg'
    incar_proto_path = os.path.join(incar_template_dir,
                                    'INCAR' + '_' + incar_type)
    incar_out_path = os.path.join(calculate_dir,
                                  'INCAR')
    os.system('cp ' + incar_proto_path + ' ' + incar_out_path)


def ptc_kpt_gen(calculate_dir, k_spacing=0.04, aimd=False):
    """generate KPOINTS and POTCAR"""
    current_dir = os.getcwd()
    os.chdir(calculate_dir)
    if os.path.exists('KPOINTS'):
        os.system("rm KPOINTS")

    # generate KPOINTS by vaspkit
    if os.path.exists('POSCAR'):
        if not aimd:
            subprocess.call('(echo 1; echo 102; echo 2; echo ' + str(k_spacing) + ') | vaspkit',
                            shell=True, stdout=subprocess.PIPE)
        # AIMD use kpoint with gamma only
        elif aimd:
            subprocess.call('(echo 1; echo 102; echo 2; echo 0) | vaspkit',
                            shell=True, stdout=subprocess.PIPE)

    elif os.path.exists('POSCAR_10000'):
        os.system('mv POSCAR_10000 POSCAR')
        if not aimd:
            subprocess.call('(echo 1; echo 102; echo 2; echo ' + str(k_spacing) + ') | vaspkit',
                            shell=True, stdout=subprocess.PIPE)
        # AIMD use kpoint with gamma only
        elif aimd:
            subprocess.call('(echo 1; echo 102; echo 2; echo 0) | vaspkit',
                            shell=True, stdout=subprocess.PIPE)
        os.system('mv POSCAR POSCAR_10000')

    else:
        raise ValueError('One POSCAR (i.e. POSCAR_10000) is needed at least')
    os.chdir(current_dir)
    # os.chdir(sys.path[0])  # back to directory where the script at.


def submit_gen_normal(calculate_dir, submit_template_dir, wm_dict, submit_type='normal', ):
    # submit_type could be 'normal', 'data', 'locpot'
    manager_proto_path = os.path.join(submit_template_dir, wm_dict["wm"] + '_head')
    submit_proto_path = os.path.join(submit_template_dir, 'submit' + '_' + submit_type)
    submit_out_path = os.path.join(calculate_dir, 'submit')
    fill_templates(submit_out_path, [manager_proto_path, submit_proto_path], wm_dict)


def check_range_interval(dos_range, window_interval):
    if dos_range / window_interval - int(dos_range / window_interval) > np.finfo(np.float32).eps:
        raise RuntimeError("The range of DOS better be a multiple of interval!")


def ldos_submit_gen_fermi(calculate_dir, submit_template_dir, wm_dict, dos_range=4.0, window_interval=0.1):
    check_range_interval(dos_range, window_interval)
    manager_proto_path = os.path.join(submit_template_dir, wm_dict["wm"] + '_head')
    submit_proto_path = os.path.join(submit_template_dir, 'submit_ldos_fermi')
    submit_out_path = os.path.join(calculate_dir, 'submit')

    num_grid_half = int(dos_range / window_interval / 2)
    all_dict = {"window_interval": str(window_interval),
                "num_grid_half": str(num_grid_half)}
    all_dict.update(wm_dict)
    fill_templates(submit_out_path, [manager_proto_path, submit_proto_path], all_dict)


def ldos_submit_gen_vacuum(calculate_dir, submit_template_dir, kit_path, wm_dict, dos_range=4.0, window_interval=0.1,
                           offset=0):
    check_range_interval(dos_range, window_interval)
    manager_proto_path = os.path.join(submit_template_dir, wm_dict["wm"] + '_head')
    submit_proto_path = os.path.join(submit_template_dir, 'submit_ldos_vacuum')
    submit_out_path = os.path.join(calculate_dir, 'submit')
    subprocess.call("cp {}/ReadVacuumFromLOCPOT.py {}".format(kit_path, calculate_dir), shell=True,
                    stdout=subprocess.PIPE)

    num_grid = int(dos_range / window_interval) + 1
    all_dict = {"window_interval": str(window_interval),
                "num_grid": str(num_grid),
                "offset": str(offset)}
    all_dict.update(wm_dict)
    fill_templates(submit_out_path, [manager_proto_path, submit_proto_path], all_dict)


def fill_templates(out_path, proto_path_list, all_dict):
    for i, proto_path in enumerate(proto_path_list):
        with open(proto_path, 'r') as f_in:
            temp = Template(f_in.read())
            out = temp.substitute(all_dict)  # `all_dict` could be larger than what `temp` requires
            if i == 0:
                f_out = open(out_path, 'w')
            else:
                f_out = open(out_path, 'a')
            f_out.write(out)
            f_out.close()
