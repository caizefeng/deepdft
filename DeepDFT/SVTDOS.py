#!/usr/bin/env python3
# @File    : SVTDOS.py
# @Time    : 9/25/2020 4:43 PM
# @Author  : Zavier Cai
# @Email   : caizefeng18@gmail.com
# @Software: PyCharm

import argparse
import os

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


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

        for idx in range(len(fc_list)):
            if idx == 0:
                self.fc.append(nn.Linear(num_element * sigma_size * 5, fc_list[idx]))
            else:
                self.fc.append(nn.Linear(fc_list[idx - 1], fc_list[idx]))
            self.fc.append(nn.ReLU())
            self.fc.append(nn.Dropout(dropout_prob[idx]))

        if self.is_res:
            self.pre_res = nn.ModuleList(self.fc[:3])
            self.res = nn.ModuleList(self.fc[3:7])
            self.after_res = nn.ModuleList(self.fc[7:])

        self.create_seq = nn.Sequential(nn.Linear(fc_list[-1], self.num_windows * self.num_input_lstm),
                                        nn.ReLU())
        self.bilstm = nn.LSTM(self.num_input_lstm, self.num_hidden_lstm, self.num_layers_lstm, batch_first=True,
                              dropout=self.dropout_lstm, bidirectional=True)

        # self.output_dos = nn.ModuleList([nn.Linear(2 * self.num_hidden_lstm, 1) for _ in range(self.num_windows)])
        self.output_dos_fc = nn.Linear(2 * self.num_hidden_lstm, 1)

    def forward(self, x):
        if self.is_res:
            for module in self.pre_res:
                x = module(x)
            x_resident = x
            for module in self.res:
                x = module(x)
            x += x_resident
            for module in self.after_res:
                x = module(x)
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
    parser.add_argument("-e", "--num_epoch", type=int, default=50)
    parser.add_argument("-d", "--dropout_prob", nargs='+', type=float, default=[0.0, ])
    parser.add_argument("-f", "--fc", nargs='+', type=int, default=[300, ])
    parser.add_argument("-w", "--num_windows", type=int, default=40)
    parser.add_argument("-i", "--input_size_lstm", type=int, default=10)
    parser.add_argument("--hidden_size_lstm", type=int, default=1)
    parser.add_argument("--num_layers_lstm", type=int, default=1)
    parser.add_argument("--dropout_lstm", type=float, default=0.0)
    parser.add_argument("-o", "--output_style", choices=['mean', 'fc'], default='mean')
    parser.add_argument("-r", "--is_res", type=bool, default=False)
    parser.add_argument("-D", "--data_dir", default="/public/WORK_backup/caizefeng/Datasets/STO_600_cut9_gauss16")
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    num_element = 3
    sigma_size = 16
    num_feature = num_element * sigma_size * 5

    if len(args.dropout_prob) == 1:
        dropout_prob_list = np.repeat(args.dropout_prob[0], 3)
    else:
        dropout_prob_list = args.dropout_prob

    if len(args.fc) == 1:
        fc_list = np.repeat(args.fc[0], 3)
    else:
        fc_list = args.fc

    data_dir = args.data_dir
    dataload_hp = {"batch_size": args.batch_size, "shuffle": True, "num_workers": 8, "pin_memory": True}
    dataload_hp_test = {"batch_size": 50000, "shuffle": False, "num_workers": 8, "pin_memory": True}
    train_hp = {"num_epoch": args.num_epoch, "lr": args.lr, }
    net_hp = {"fc_list": fc_list, "dropout_prob": dropout_prob_list, "num_windows": args.num_windows,
              "input_size_lstm": args.input_size_lstm, "hidden_size_lstm": args.hidden_size_lstm,
              "num_layers_lstm": args.num_layers_lstm, "dropout_lstm": args.dropout_lstm,
              "output_style": args.output_style, "is_res": args.is_res, }

    data_name = os.path.basename(data_dir)
    run_name = '_'.join(('SVTDOS',
                         "batch", str(dataload_hp["batch_size"]),
                         "lr", str(train_hp["lr"]),
                         "epoch", str(train_hp["num_epoch"]),
                         "data", data_name,
                         "dropout", *[str(i) for i in net_hp["dropout_prob"]],
                         "fc", *[str(i) for i in net_hp["fc_list"]],
                         "lstm", str(net_hp["num_windows"]), str(net_hp["input_size_lstm"]),
                         str(net_hp["hidden_size_lstm"]), str(net_hp["num_layers_lstm"]), str(net_hp["dropout_lstm"]),
                         "output", str(net_hp["output_style"]),
                         "res", str(net_hp["is_res"]),
                         ))

    # train_writer = SummaryWriter('_'.join(("runs/" + run_name, "train"
    #                                        )))
    # test_writer = SummaryWriter('_'.join(("runs/" + run_name, "val"
    #                                       )))
    debug_writer = SummaryWriter('_'.join(("runs/" + run_name, "debug"
                                          )))

    net = SVTNetDOS(num_element=num_element, sigma_size=sigma_size, **net_hp).to(torch.float64)
    tensor_debug = torch.rand(dataload_hp["batch_size"], 240).to(torch.float64)
    debug_writer.add_graph(net, tensor_debug)
    debug_writer.flush()
    debug_writer.close()
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(net.parameters(), lr=train_hp["lr"])
    #
    # train_dir = os.path.join(data_dir, "train")
    # test_dir = os.path.join(data_dir, "test")
    # train_path_list = [x.path for x in os.scandir(train_dir) if x.name.endswith("npy")]
    # test_path_list = [x.path for x in os.scandir(test_dir) if x.name.endswith("npy")]
    #
    # # validation and testing dataloader
    # dataset_test_list = []
    # for i, test_file in enumerate(test_path_list):
    #     all_data_test = torch.from_numpy(np.load(test_file))
    #     dataset_test_list.append(TensorDataset(all_data_test[:, :-1], all_data_test[:, -1]))
    # test_iter = DataLoader(ConcatDataset(dataset_test_list), **dataload_hp_test)
    #
    # # calculate mean and std over the whole training set
    # train_mean, train_std = standardization2D(read_saved=True, data_path=data_dir, num_feature=num_feature,
    #                                           train_path_list=train_path_list)
    #
    # # training and validiting
    # training_loss, batch_count = 0.0, 0
    # for epoch in range(train_hp["num_epoch"]):
    #     random.shuffle(train_path_list)
    #     for i, train_file in enumerate(train_path_list):
    #         all_data = torch.from_numpy(np.load(train_file))
    #         train_iter = DataLoader(TensorDataset(all_data[:, :-1], all_data[:, -1]), **dataload_hp)
    #         for j, (X, y) in enumerate(train_iter):
    #             # TensorBoard Graph
    #             if j == 0 and i == 0 and epoch == 0:
    #                 train_writer.add_graph(net, X)
    #
    #             net.to(device)
    #             X = ((X - train_mean) / train_std).to(device)
    #             y = y.to(device)
    #             optimizer.zero_grad()
    #             # forward + backward + optimize
    #             y_hat = net(X)
    #             loss = criterion(y_hat, y.view(-1, 1))
    #             loss.backward()
    #             optimizer.step()
    #
    #             training_loss += loss.item() * X.size(0)
    #             batch_count += X.size(0)
    #
    #             # TensorBoard Scalar
    #             if (len(train_iter) * i + j) % 1000 == 999:  # switch to 100 if data_dir is "test"
    #                 train_writer.add_scalar('SVT_charge/train',
    #                                         math.sqrt(training_loss / batch_count),
    #                                         epoch * len(train_path_list) * len(train_iter) + i * len(train_iter) + j)
    #                 training_loss, batch_count = 0.0, 0
    #
    #                 test_writer.add_scalar('SVT_charge/val',
    #                                        evaluate_loss(test_iter, net, criterion, train_mean, train_std),
    #                                        epoch * len(train_path_list) * len(train_iter) + i * len(train_iter) + j)
    #
    #             if j == 0 and i == 0 and epoch == 20:
    #                 torch.save(net.state_dict(), os.path.join("nets", run_name + "_temp" + ".pt"))
    #
    # print('Finished Training')
    #
    # torch.save(net.state_dict(), os.path.join("nets", run_name + ".pt"))
    #
    # # net2 = SVTNetCharge(num_element=3, sigma_size=16).to(torch.float64)
    # # net2.load_state_dict(torch.load(PATH))
    #
    # train_writer.flush()
    # train_writer.close()
    # test_writer.flush()
    # test_writer.close()
