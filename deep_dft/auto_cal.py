#!/usr/bin/env python3
# @File    : auto_cal.py
# @Time    : 9/9/2020 3:14 PM
# @Author  : Zavier Cai
# @Email   : caizefeng18@gmail.com
# @Software: PyCharm

import argparse
import configparser
import os
import sys

from deep_dft.utils.auto_vasp_utils import change_pp, vasp_in_files_prep
from deep_dft.utils.ini_utils import read_section_as_dict, read_ini_as_dict, \
    read_or_write_default
from deep_dft.utils.string_utils import str2bool


def ml_dft_cal(cluster, calculate_dir, cal_type, pp=None, k_spacing=None, dos_range=4.0, window_interval=0.1,
               offset=0, ):
    """Automatically generate input files and submit VASP tasks for a list of working directories"""
    kit_path = os.path.join(sys.path[0], "scripts")
    workload_config_path = os.path.join(sys.path[0], "configs", "workload.ini")
    vasp_config_path = os.path.join(sys.path[0], "configs", "vasp.ini")
    calculation_config_path = os.path.join(sys.path[0], "configs", "calculation.ini")

    template_path = os.path.join(sys.path[0], "templates")
    config_template_dir = os.path.join(template_path, 'config_templates')

    # Change and read cluster and cooresponding workload configuration
    cluster = read_or_write_default(workload_config_path, "cluster", cluster)
    wm_dict = read_section_as_dict(workload_config_path, cluster)

    # Change pseudo potential and read vasp configuration
    vasp_dict = read_ini_as_dict(vasp_config_path)
    if pp is None:
        pp = vasp_dict["basic"]["pseudo"]
    change_pp(config_template_dir, wm_dict, pp)
    if k_spacing is not None:
        vasp_dict["static"]["k_spacing"] = k_spacing

    # Read calculating configuration
    try:
        calculation_dict = read_section_as_dict(calculation_config_path, cal_type)
    except configparser.NoSectionError:
        print("Limited calculating types are supported: AIMD, Single, Charge, WF, LDOS-fermi, LDOS-vacuum!")
        sys.exit(1)

    # Prep and submit calculations
    vasp_in_files_prep(calculate_dir, template_path, kit_path, wm_dict, vasp_dict, calculation_dict,
                       dos_range=dos_range, window_interval=window_interval, offset=offset)
    os.chdir(calculate_dir)
    if wm_dict["wm"] == "SLURM":
        os.system('sbatch submit')
    elif wm_dict["wm"] == "PBS":
        os.system("qsub submit")

    print('Tasks have been submitted')


if __name__ == '__main__':
    print(
        "Note: Using the interface for the first time, you have to add your own server setting by executing: \
python auto_cal.py --setting")
    parser = argparse.ArgumentParser(description="DeepDFT Auto-calculating Interface")

    # Setting or Calculating
    parser.add_argument("--setting", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--setting_type", choices=['workload', 'vasp', 'calculation'], default='workload')

    # General
    parser.add_argument("-c", "--cluster")
    parser.add_argument("-i", "--calculate_dir", )
    parser.add_argument("-t", "--typecal", choices=['AIMD', 'Single', 'Charge', 'WF', 'LDOS-fermi', 'LDOS-vacuum'],
                        default='LDOS-vacuum')

    # Static
    parser.add_argument("-k", "--k_spacing", type=float, help="Kpoints spacing for VASP calculation")
    parser.add_argument("-p", "--pseudo", choices=['PAW', 'USPP'])

    # LDOS processing
    parser.add_argument("-d", "--dos_range", type=float, default=4.0, )
    parser.add_argument("-w", "--window_interval", type=float, default=0.1, )
    parser.add_argument("-f", "--offset", type=float, default=2, )

    args = parser.parse_args()

    if args.setting:
        ini_path = os.path.join(sys.path[0], "configs", args.setting_type + '.ini')
        os.system("vim " + ini_path)
    else:
        calculate_dir = args.calculate_dir if args.typecal == 'AIMD' else os.path.join(args.calculate_dir, 'POSCAR_all')
        ml_dft_cal(args.cluster, calculate_dir, cal_type=args.typecal, pp=args.pseudo, k_spacing=args.k_spacing,
                   dos_range=args.dos_range, window_interval=args.window_interval, offset=args.offset)
