#!/usr/bin/env python3
# @File    : SVTDOS.py
# @Time    : 9/25/2020 4:43 PM
# @Author  : Zavier Cai
# @Email   : caizefeng18@gmail.com
# @Software: PyCharm

import argparse
import os
import random
import sys

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from deep_dft.utils.ML_utils import standardization2D, smooth, \
    evaluate_loss, find_residual_fc
from deep_dft.utils.string_utils import str2bool, gen_name, expand_list_in_dict
from deep_dft.utils.fs_utils import mkdir_without_override


# Net
class SVTNetDOS(nn.Module):

    def __init__(self, num_element, sigma_size, dropout_prob, fc_list, num_windows, input_size_lstm, hidden_size_lstm,
                 num_layers_lstm, dropout_lstm, output_style="mean", is_res=False, ):
        super(SVTNetDOS, self).__init__()
        assert len(dropout_prob) == len(fc_list), "length of dropout probability unequals that of full-connected list"

        self.fc = nn.ModuleList([])
        self.is_res = is_res
        self.num_windows = num_windows
        self.num_input_lstm = input_size_lstm
        self.num_hidden_lstm = hidden_size_lstm
        self.num_layers_lstm = num_layers_lstm
        self.dropout_lstm = dropout_lstm
        self.output_style = output_style
        self.num_seq_feature = self.num_windows * self.num_input_lstm

        for idx in range(len(fc_list)):
            if idx == 0:
                self.fc.append(nn.Linear(num_element * sigma_size * 5, fc_list[idx]))
            else:
                self.fc.append(nn.Linear(fc_list[idx - 1], fc_list[idx]))
            self.fc.append(nn.BatchNorm1d(fc_list[idx]))
            self.fc.append(nn.ReLU())
            self.fc.append(nn.Dropout(dropout_prob[idx]))

        if self.is_res:
            self.res_starting, self.res_ending = find_residual_fc(self.fc)
        self.create_seq = nn.Sequential(nn.Linear(fc_list[-1], self.num_seq_feature),
                                        nn.BatchNorm1d(self.num_seq_feature),
                                        # nn.ReLU()
                                        )
        self.bilstm = nn.LSTM(self.num_input_lstm, self.num_hidden_lstm, self.num_layers_lstm, batch_first=True,
                              dropout=self.dropout_lstm, bidirectional=True)

        # self.output_dos = nn.ModuleList([nn.Linear(2 * self.num_hidden_lstm, 1) for _ in range(self.num_windows)])
        self.output_dos_fc = nn.Linear(2 * self.num_hidden_lstm, 1)

    def forward(self, x):

        if self.is_res:
            x_res = 0
            for idx, module in enumerate(self.fc):
                x = module(x)
                if idx in self.res_starting:
                    x_res = x
                if idx in self.res_ending:
                    x += x_res

        else:
            for module in self.fc:
                x = module(x)

        x = self.create_seq(x).view(-1, self.num_windows, self.num_input_lstm)
        x, _ = self.bilstm(x)  # x: batch*seq*feature

        if self.output_style == "mean":
            x = x.mean(2)
        elif self.output_style == "fc":
            x = self.output_dos_fc(x).squeeze()
        return x


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="DeepDFT SVT DOS")
    parser.add_argument("-b", "--batch_size", type=int, default=7000, )
    parser.add_argument("-l", "--lr", type=float, default=0.00025, )
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("-e", "--num_epoch", type=int, default=50)
    parser.add_argument("-d", "--dropout_prob", nargs='+', type=float, default=[0.0, ])
    parser.add_argument("-f", "--fc", nargs='+', type=int, default=[300, ])
    parser.add_argument("-w", "--num_windows", type=int, default=41)
    parser.add_argument("-i", "--input_size_lstm", type=int, default=10)
    parser.add_argument("--hidden_size_lstm", type=int, default=1)
    parser.add_argument("--num_layers_lstm", type=int, default=1)
    parser.add_argument("--dropout_lstm", type=float, default=0.0)
    parser.add_argument("-o", "--output_style", choices=['mean', 'fc'], default='fc')
    parser.add_argument("-r", "--is_res", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--is_half", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-D", "--data_dir", default="/public/WORK_backup/caizefeng/Datasets/STO_600_cut9_gauss16")
    parser.add_argument("--label_dir", )
    parser.add_argument("-g", "--sigma_size", default=16, type=int)
    parser.add_argument("--device", nargs='+', type=str, default=['0', ])
    parser.add_argument("-u", "--runs_dir", default=os.path.dirname(sys.path[0]))
    args = parser.parse_args()

    mkdir_without_override(os.path.join(args.runs_dir, "nets"))
    mkdir_without_override(os.path.join(args.runs_dir, "runs"))
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(args.device)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_element = 3
    sigma_size = args.sigma_size
    num_feature = num_element * sigma_size * 5

    if len(args.dropout_prob) == 1:
        dropout_prob_list = np.repeat(args.dropout_prob[0], 3).tolist()
    else:
        dropout_prob_list = args.dropout_prob

    if len(args.fc) == 1:
        fc_list = np.repeat(args.fc[0], 3).tolist()
    else:
        fc_list = args.fc

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

    dataload_hp = {"batch_size": args.batch_size, "shuffle": True, "num_workers": 4, "pin_memory": True}
    dataload_hp_test = {"batch_size": 50000, "shuffle": False, "num_workers": 4, "pin_memory": True}
    train_hp = {"batch_size": args.batch_size, "num_epoch": args.num_epoch, "lr": args.lr, "gamma": args.gamma}
    net_hp = {"fc_list": fc_list, "dropout_prob": dropout_prob_list,
              "num_windows": args.num_windows if not args.is_half else int(np.floor(args.num_windows / 2)),
              "input_size_lstm": args.input_size_lstm, "hidden_size_lstm": args.hidden_size_lstm,
              "num_layers_lstm": args.num_layers_lstm, "dropout_lstm": args.dropout_lstm,
              "output_style": args.output_style, "is_res": args.is_res, }

    # generate the name for this run in TensorBoard
    run_dict = dict(train_hp, **net_hp)
    run_dict["data"] = os.path.basename(data_dir)
    run_dict["half"] = args.is_half
    run_extra = ["BN", "seq_without_activate"]
    run_name, _ = gen_name('SVTDOS', run_dict, run_extra)

    runs_path = os.path.join(args.runs_dir, "runs", run_name)
    train_writer = SummaryWriter('_'.join((runs_path, "train")))
    test_writer = SummaryWriter('_'.join((runs_path, "val")))

    net = SVTNetDOS(num_element=num_element, sigma_size=sigma_size, **net_hp).to(torch.float64)
    run_dict_expanded = expand_list_in_dict(run_dict)
    train_writer.add_hparams({**{"train": -1}, **run_dict_expanded, }, {'loss': -1})

    # debug_writer = SummaryWriter('_'.join(("runs/" + run_name, "debug")))
    # tensor_debug = torch.rand(dataload_hp["batch_size"], 240).to(torch.float64)
    # debug_writer.add_graph(net, tensor_debug)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=train_hp["lr"])

    # learning rate decay
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=train_hp["gamma"])

    # validation and testing dataloader
    dataset_test_list = []
    for test_feature_file, test_ldos_file in zip(test_feature_path_list, test_ldos_path_list):
        all_data_test = torch.from_numpy(np.load(test_feature_file))
        ldos_data_test = torch.from_numpy(np.load(test_ldos_file))
        dataset_test_list.append(TensorDataset(all_data_test[:, :-1], ldos_data_test[:, :net_hp["num_windows"]]))
    test_iter = DataLoader(ConcatDataset(dataset_test_list), **dataload_hp_test)

    # calculate mean and std over the whole training set
    train_mean, train_std = standardization2D(read_saved=True, data_path=data_dir, num_feature=num_feature,
                                              train_path_list=train_feature_path_list)

    # training and validiting
    training_loss, batch_count = 0.0, 0
    train_path_list = [x for x in zip(train_feature_path_list, train_ldos_path_list)]

    train_scalar_list = []
    test_scalar_list = []

    net.to(device)
    for epoch in range(train_hp["num_epoch"]):
        random.shuffle(train_path_list)
        for i, train_file in enumerate(train_path_list):
            all_data = torch.from_numpy(np.load(train_file[0]))
            ldos_data = torch.from_numpy(np.load(train_file[1]))
            train_iter = DataLoader(TensorDataset(all_data[:, :-1], ldos_data[:, :net_hp["num_windows"]]),
                                    **dataload_hp)
            for j, (X, y) in enumerate(train_iter):

                X = ((X - train_mean) / train_std).to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

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
                if (len(train_iter) * i + j) % 100 == 99:  # switch to 100 if data_dir is "test"
                    train_scalar = np.sqrt(training_loss / batch_count)
                    train_scalar_list.append(train_scalar)
                    train_writer.add_scalar('SVT_DOS/train',
                                            train_scalar,
                                            epoch * len(train_path_list) * len(train_iter) + i * len(train_iter) + j)
                    training_loss, batch_count = 0.0, 0

                    test_scalar = evaluate_loss(test_iter, net, criterion, train_mean, train_std)
                    test_scalar_list.append(test_scalar)
                    test_writer.add_scalar('SVT_DOS/val',
                                           test_scalar,
                                           epoch * len(train_path_list) * len(train_iter) + i * len(train_iter) + j)

                if j == 0 and i == 0 and epoch == (train_hp["num_epoch"] // 2):
                    torch.save(net.state_dict(), os.path.join(args.runs_dir, "nets", run_name + "_temp_state_dict.pt"))
        scheduler.step()
    print('Finished Training')

    # TensoBoard HParam
    train_writer.add_hparams({**{"train": 1}, **run_dict_expanded, },
                             {'loss': smooth(train_scalar_list, weight=0.995)[-1]})
    test_writer.add_hparams({**{"train": 0}, **run_dict_expanded, },
                            {'loss': smooth(test_scalar_list, weight=0.995)[-1]})

    torch.save(net.state_dict(), os.path.join(args.runs_dir, "nets", run_name + "_state_dict.pt"))
    torch.save(net, os.path.join(args.runs_dir, "nets", run_name + ".pt"))

    train_writer.flush()
    train_writer.close()
    test_writer.flush()
    test_writer.close()
