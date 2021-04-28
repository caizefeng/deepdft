#!/usr/bin/env python3
# @File    : SVTDOS.py
# @Time    : 9/25/2020 4:43 PM
# @Author  : Zavier Cai
# @Email   : caizefeng18@gmail.com
# @Software: PyCharm


import argparse
import os
import sys
import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from deep_dft.SVTDOS import SVTNetDOS
from deep_dft.utils.AsyncDataloader import CudaDataLoader, FileDataLoader
from deep_dft.utils.ML_utils import standardization2D, smooth, evaluate_loss
from deep_dft.utils.fs_utils import mkdir_without_override
from deep_dft.utils.string_utils import str2bool, gen_name, expand_list_in_dict

if __name__ == '__main__':
    # Optimizing
    parser = argparse.ArgumentParser(description="DeepDFT SVT DOS")
    parser.add_argument("-b", "--batch_size", type=int, default=7500, )
    parser.add_argument("-l", "--lr", type=float, default=0.00025, )
    parser.add_argument("--decay", choices=["cos", "cos-warm", "exp"], default="cos-warm")
    parser.add_argument("--tmax", type=int, default=5)
    parser.add_argument("--t0", type=int, default=5)
    parser.add_argument("--tmult", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("-e", "--num_epoch", type=int, default=50)
    # Net Structure
    parser.add_argument("-d", "--dropout_prob", nargs='+', type=float, default=[0.0, ])
    parser.add_argument("-f", "--fc", nargs='+', type=int, default=[300, ])
    parser.add_argument("-i", "--input_size_lstm", type=int, default=10)
    parser.add_argument("--hidden_size_lstm", type=int, default=1)
    parser.add_argument("--num_layers_lstm", type=int, default=1)
    parser.add_argument("--dropout_lstm", type=float, default=0.0)
    parser.add_argument("-o", "--output_style", choices=['mean', 'fc'], default='fc')
    parser.add_argument("-r", "--is_res", type=str2bool, nargs='?', const=True, default=True)
    # Physics
    parser.add_argument("-w", "--num_windows", type=int, default=41)
    parser.add_argument("-g", "--sigma_size", type=int, default=16)
    parser.add_argument("--num_element", type=int, default=3)
    parser.add_argument("--is_half", type=str2bool, nargs='?', const=True, default=False)
    # File System
    parser.add_argument("-D", "--data_dir", default="/public/WORK_backup/caizefeng/Datasets/vacuum_STO/vacuum_STO_600_cut9_gauss16_0.25_5")
    parser.add_argument("--label_dir", )
    parser.add_argument("-u", "--runs_dir", default=os.path.dirname(sys.path[0]))
    parser.add_argument("-n", "--scalar_name", default="SVT_DOS")
    parser.add_argument("--device", nargs='+', type=str, default=['0', ])
    parser.add_argument("--early_dump", type=int)
    args = parser.parse_args()

    # General
    mkdir_without_override(os.path.join(args.runs_dir, "nets"))
    mkdir_without_override(os.path.join(args.runs_dir, "runs"))
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(args.device)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_element = args.num_element
    sigma_size = args.sigma_size
    num_feature = num_element * sigma_size * 5

    # Net
    if len(args.dropout_prob) == 1:
        dropout_prob_list = np.repeat(args.dropout_prob[0], 3).tolist()
    else:
        dropout_prob_list = args.dropout_prob
    if len(args.fc) == 1:
        fc_list = np.repeat(args.fc[0], 3).tolist()
    else:
        fc_list = args.fc
    net_hp = {"fc_list": fc_list, "dropout_prob": dropout_prob_list,
              "num_windows": args.num_windows if not args.is_half else int(np.floor(args.num_windows / 2)),
              "input_size_lstm": args.input_size_lstm, "hidden_size_lstm": args.hidden_size_lstm,
              "num_layers_lstm": args.num_layers_lstm, "dropout_lstm": args.dropout_lstm,
              "output_style": args.output_style, "is_res": args.is_res, }
    net = SVTNetDOS(num_element=num_element, sigma_size=sigma_size, **net_hp)

    # Optimizer
    train_hp = {"batch_size": args.batch_size, "num_epoch": args.num_epoch, "lr": args.lr}
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    schedule_dict = {"decay": args.decay}
    # learning rate decay
    if schedule_dict["decay"] == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.tmax, eta_min=5e-6)
        schedule_dict.update({"tmax": args.tmax})
    elif schedule_dict["decay"] == "cos-warm":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.t0,
                                                                   T_mult=args.tmult, eta_min=5e-6)
        schedule_dict.update({"t0": args.t0, "tmult": args.tmult})
    else:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
        schedule_dict.update({"gamma": args.gamma})

    # Data
    dataload_hp = {"batch_size": args.batch_size, "shuffle": True, "num_workers": 0, "pin_memory": False}
    dataload_hp_test = {"batch_size": 50000, "shuffle": False, "num_workers": 0, "pin_memory": False}
    data_dir = args.data_dir
    if args.label_dir:
        label_dir = args.label_dir
    else:
        label_dir = data_dir
    feature_train_dir = os.path.join(data_dir, "train")
    label_train_dir = os.path.join(label_dir, "train")
    feature_test_dir = os.path.join(data_dir, "test")
    label_test_dir = os.path.join(label_dir, "test")
    train_feature_path_list = [x.path for x in os.scandir(feature_train_dir) if
                               x.name.endswith("npy") and (not x.name.startswith("LDOS"))]
    train_ldos_path_list = [x.path for x in os.scandir(label_train_dir) if
                            x.name.endswith("npy") and x.name.startswith("LDOS")]
    test_feature_path_list = [x.path for x in os.scandir(feature_test_dir) if
                              x.name.endswith("npy") and (not x.name.startswith("LDOS"))]
    test_ldos_path_list = [x.path for x in os.scandir(label_test_dir) if
                           x.name.endswith("npy") and x.name.startswith("LDOS")]
    # testing dataloader
    dataset_test_list = []
    for test_feature_file, test_ldos_file in zip(test_feature_path_list, test_ldos_path_list):
        all_data_test = torch.from_numpy(np.load(test_feature_file)).float()
        ldos_data_test = torch.from_numpy(np.load(test_ldos_file)).float()
        dataset_test_list.append(TensorDataset(all_data_test[:, :-1], ldos_data_test[:, :net_hp["num_windows"]]))
    test_iter = DataLoader(ConcatDataset(dataset_test_list), **dataload_hp_test)
    test_iter_async = CudaDataLoader(test_iter, device, queue_size=4, repeat=True)
    # calculate mean and std over the whole training set
    train_mean, train_std = standardization2D(read_saved=True, data_path=data_dir, num_feature=num_feature,
                                              train_path_list=train_feature_path_list)
    train_mean = train_mean.to(device).float()
    train_std = train_std.to(device).float()
    # training and validiting
    training_loss, batch_count = 0.0, 0
    train_path_list = [x for x in zip(train_feature_path_list, train_ldos_path_list)]
    train_file_iter_async = FileDataLoader(train_path_list, shuffle=True, queue_size=1)

    # Generate the name for this run in TensorBoard
    run_dict = {}
    run_dict.update(net_hp)
    run_dict.update(train_hp)
    run_dict.update(schedule_dict)
    run_dict["data"] = os.path.basename(data_dir)
    run_dict["half"] = args.is_half
    run_extra = ["BN", "seq_without_activate"]
    run_name, _ = gen_name('SVTDOS', run_dict, run_extra)

    runs_path = os.path.join(args.runs_dir, "runs", run_name)
    train_writer = SummaryWriter('_'.join((runs_path, "train")))
    test_writer = SummaryWriter('_'.join((runs_path, "val")))

    # Tensorboard Hparam
    run_dict_expanded = expand_list_in_dict(run_dict)
    train_writer.add_hparams({**{"train": -1}, **run_dict_expanded, }, {'loss': -1})

    train_scalar_list = []
    test_scalar_list = []
    net.to(device)
    for epoch in range(train_hp["num_epoch"]):
        if args.early_dump:
            if epoch == args.early_dump:
                torch.save({'model_hp': dict(num_element=num_element, sigma_size=sigma_size, **net_hp),
                            'model_state_dict': net.state_dict(), },
                           os.path.join(args.runs_dir, "nets", "Checkpoint_{}_{}.pt".format(epoch, run_name)))

        for i, train_file_data in enumerate(train_file_iter_async):

            train_iter = DataLoader(
                TensorDataset(train_file_data[0][:, :-1], train_file_data[1][:, :net_hp["num_windows"]]),
                **dataload_hp)
            train_iter_async = CudaDataLoader(train_iter, device, queue_size=4, repeat=False)

            for j, (X, y) in enumerate(train_iter_async):
                a = time.time()

                X = ((X - train_mean) / train_std)

                # TensorBoard Graph
                if j == 0 and i == 0 and epoch == 0:
                    train_writer.add_graph(net, X)

                optimizer.zero_grad()
                # forward + backward + optimize
                y_hat = net(X)
                loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()

                training_loss += loss.item() * X.size(0)
                batch_count += X.size(0)

                # TensorBoard Scalar
                if (len(train_iter_async) * i + j) % 100 == 99:  # switch to 100 if data_dir is "test"
                    train_scalar = np.sqrt(training_loss / batch_count)
                    train_scalar_list.append(train_scalar)
                    train_writer.add_scalar(args.scalar_name + '/train',
                                            train_scalar,
                                            epoch * len(train_file_iter_async) * len(train_iter_async) + i * len(
                                                train_iter_async) + j)
                    training_loss, batch_count = 0.0, 0

                    test_scalar = evaluate_loss(test_iter_async, net, criterion, train_mean, train_std)
                    test_scalar_list.append(test_scalar)
                    test_writer.add_scalar(args.scalar_name + '/val',
                                           test_scalar,
                                           epoch * len(train_file_iter_async) * len(train_iter_async) + i * len(
                                               train_iter_async) + j)

                if j == 0 and i == 0 and epoch == (train_hp["num_epoch"] // 2):
                    torch.save({'model_hp': dict(num_element=num_element, sigma_size=sigma_size, **net_hp),
                                'model_state_dict': net.state_dict(), },
                               os.path.join(args.runs_dir, "nets", "Halfway_state_dict_{}.pt".format(run_name)))
        scheduler.step()
        # torch.cuda.empty_cache()
    print('Finished Training: {}'.format(run_name))

    # TensoBoard HParam
    train_writer.add_hparams({**{"train": 1}, **run_dict_expanded, },
                             {'loss': smooth(train_scalar_list, weight=0.995)[-1]})
    test_writer.add_hparams({**{"train": 0}, **run_dict_expanded, },
                            {'loss': smooth(test_scalar_list, weight=0.995)[-1]})

    torch.save({'model_hp': dict(num_element=num_element, sigma_size=sigma_size, **net_hp),
                'model_state_dict': net.state_dict(), }, os.path.join(args.runs_dir, "nets",
                                                                      "state_dict_{}.pt".format(run_name)))
    torch.save(net, os.path.join(args.runs_dir, "nets", "{}.pt".format(run_name)))

    train_writer.flush()
    train_writer.close()
    test_writer.flush()
    test_writer.close()
