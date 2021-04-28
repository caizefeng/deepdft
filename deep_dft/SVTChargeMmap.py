#!/usr/bin/env python3
# @File    : DeepChargeSVT.py
# @Time    : 9/9/2020 4:44 PM
# @Author  : Zavier Cai
# @Email   : caizefeng18@gmail.com
# @Software: PyCharm

# TODO(Zaveir): Get a SSD, otherwise the memory mapping method is extremely slow.

import argparse
import math
import os
import sys
import warnings

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

from deep_dft.utils.ML_utils import standardization2D, evaluate_loss
from deep_dft.utils.datasets import MmapDataset2D
from deep_dft.utils.fs_utils import mkdir_without_override

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="DeepDFT SVT Charge")
parser.add_argument("-b", "--batch_size", type=int, default=5000, )
parser.add_argument("-l", "--lr", type=float, default=0.0005, )
parser.add_argument("-e", "--num_epoch", type=int, default=50)
parser.add_argument("-d", "--dropout_prob", nargs="+", type=float, default=[0.0, ])

parser.add_argument("-D", "--data_dir", default="/public/WORK_backup/caizefeng/Datasets/STO_600_cut9_gauss16")
parser.add_argument("--device", default="cuda:1")
parser.add_argument("-u", "--runs_dir", default=os.path.dirname(sys.path[0]))
args = parser.parse_args()

mkdir_without_override(os.path.join(args.runs_dir, "nets"))
mkdir_without_override(os.path.join(args.runs_dir, "runs"))
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
num_element = 3
sigma_size = 16
num_feature = num_element * sigma_size * 5
num_column = num_element * sigma_size * 5 + 1

if len(args.dropout_prob) == 1:
    dropout_prob_list = np.repeat(args.dropout_prob[0], 3)
else:
    dropout_prob_list = args.dropout_prob


# MLP
class SVTNetCharge_naive(nn.Module):
    def __init__(self, num_element, sigma_size, dropout_prob, fc_list=None, ):
        super(SVTNetCharge_naive, self).__init__()
        if fc_list is None:
            fc_list = [300, 300, 300]
        self.fc = nn.Sequential(
            nn.Linear(num_element * sigma_size * 5, fc_list[0]),
            nn.ReLU(),
            nn.Dropout(dropout_prob[0]),
            nn.Linear(fc_list[0], fc_list[1]),
            nn.ReLU(),
            nn.Dropout(dropout_prob[1]),
            nn.Linear(fc_list[1], fc_list[2]),
            nn.ReLU(),
            nn.Dropout(dropout_prob[2]),
            nn.Linear(fc_list[2], 1),
        )

    def forward(self, x):
        output = self.fc(x)
        return output


data_dir = args.data_dir
dataload_hp = {"batch_size": args.batch_size, "shuffle": True, "num_workers": 8, "pin_memory": True}
dataload_hp_test = {"batch_size": 50000, "shuffle": False, "num_workers": 8, "pin_memory": True}
# dataload_hp_test = {"batch_size": 500000, "shuffle": False, "num_workers": 4, "pin_memory": False}
train_hp = {"num_epoch": args.num_epoch, "lr": args.lr, "dropout_prob": dropout_prob_list}

data_name = os.path.basename(data_dir)
run_name = '_'.join(('SVTCharge',
                     "batch", str(dataload_hp["batch_size"]),
                     "lr", str(train_hp["lr"]),
                     "epoch", str(train_hp["num_epoch"]),
                     "data", data_name,
                     "dropout", *[str(i) for i in train_hp["dropout_prob"]],
                     # "debug",
                     # "unstand",
                     ))

runs_path = os.path.join(args.runs_dir, "runs", run_name)
train_writer = SummaryWriter('_'.join((runs_path, "train")))
test_writer = SummaryWriter('_'.join((runs_path, "val")))

net = SVTNetCharge_naive(num_element=num_element, sigma_size=sigma_size, dropout_prob=train_hp["dropout_prob"]).to(
    torch.float64)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=train_hp["lr"])

train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
train_path_list = [x.path for x in os.scandir(train_dir) if x.name.endswith("npy")]
test_path_list = [x.path for x in os.scandir(test_dir) if x.name.endswith("npy")]

# validation and test dataset and dataloader
dataset_test_list = []
for i, test_file in enumerate(test_path_list):
    dataset = MmapDataset2D(test_file, num_column)
    dataset_test_list.append(dataset)
dataset_test = ConcatDataset(dataset_test_list)
test_iter = DataLoader(dataset_test, **dataload_hp_test)

# training dataset
dataset_train_list = []
for i, train_file in enumerate(train_path_list):
    dataset = MmapDataset2D(train_file, num_column)
    dataset_train_list.append(dataset)
dataset_train = ConcatDataset(dataset_train_list)
train_iter = DataLoader(dataset_train, **dataload_hp)

# calculate mean and std over the whole training set
train_iter_more = DataLoader(dataset_train, **dataload_hp_test)
train_mean, train_std = standardization2D(read_saved=True, data_path=data_dir, num_feature=num_feature,
                                          train_iter_mmap=train_iter_more)

# training
training_loss, batch_count = 0.0, 0
for epoch in range(train_hp["num_epoch"]):
    for i, (X, y) in enumerate(train_iter):
        # TensorBoard Graph
        if i == 0 and epoch == 0:
            train_writer.add_graph(net, X)

        net.to(device)
        X = ((X - train_mean) / train_std).to(device)
        y = y.to(device)
        optimizer.zero_grad()
        # forward + backward + optimize
        y_hat = net(X)
        loss = criterion(y_hat, y.view(-1, 1))
        loss.backward()
        optimizer.step()

        training_loss += loss.item() * X.size()[0]
        batch_count += X.size()[0]

        # TensorBoard Scalar
        if i % 1000 == 999:  # switch to 100 if data_dir is "test"
            train_writer.add_scalar('SVT_charge/train',
                                    math.sqrt(training_loss / batch_count),
                                    len(train_iter) * epoch + i)
            training_loss, batch_count = 0.0, 0

            test_writer.add_scalar('SVT_charge/val',
                                   evaluate_loss(test_iter, net, criterion, train_mean, train_std),
                                   len(train_iter) * epoch + i)

print('Finished Training')

torch.save(net.state_dict(), os.path.join(args.runs_dir, "nets", run_name + "_state_dict.pt"))
torch.save(net, os.path.join(args.runs_dir, "nets", run_name + ".pt"))

# net2 = SVTNetCharge_naive(num_element=3, sigma_size=16).to(torch.float64)
# net2.load_state_dict(torch.load(PATH))

train_writer.flush()
train_writer.close()
test_writer.flush()
test_writer.close()
