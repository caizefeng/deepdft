# coding: utf-8

# In[1]:

import os
import re
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as Data
from torch import nn

# change primitive date in lattice coordinate (VASP default) to Cartesian coordinate
from feature_utils import PBC_padding, dir2cart
from io_utils import split_read_chgcar, read_occ


# each atom : x y z element(0, 1, ...)
# ## Reading CHGCARs
# ## Preprocessing Functions


def coor_occ_gen(coor_list, atom_num_list):
    # concatenate the list of coordinates
    coor_without_element = np.concatenate(coor_list)
    # add the last element column
    coor = np.hstack(
        (coor_without_element,
         np.arange(len(atom_num_list)).repeat(atom_num_list).reshape(-1, 1)))
    return coor


# preprocessing input coordinates of atoms
def DPSE_feature_pre(coor, center, r_cut, r_cs):
    coor_i_raw = np.delete(coor, center, axis=0)
    coor_i = coor_i_raw[:, 0:3]
    element_array = coor_i_raw[:, 3]
    dist = np.sqrt(((coor_i - coor[center, 0:3]) ** 2).sum(axis=1))

    R_i = coor_i[dist < r_cut] - coor[center, 0:3]
    dist_i = dist[dist < r_cut]
    element_i = element_array[dist < r_cut]

    s_i = np.zeros((dist_i.shape[0], 2))
    for index, rji in enumerate(dist_i):
        if rji <= r_cs:
            s_i[index, 0] = 1 / rji
        elif r_cs < rji < r_cut:
            s_i[index, 0] = 1 / rji * (1 / 2 * np.cos(np.pi * (rji - r_cs) /
                                                      (r_cut - r_cs)) + 1 / 2)
        s_i[index, 1] = element_i[index]
    s_i_torch = torch.from_numpy(s_i).float()
    R_i_tilde = np.hstack((s_i[:, 0:1], s_i[:, 0:1] * R_i / dist_i.reshape(
        (-1, 1))))  # s_i[:,0:1] is s_ji
    R_i_tilde_torch = torch.from_numpy(R_i_tilde).float()
    return s_i_torch, R_i_tilde_torch


def flatten_and_padding(s_i_torch, R_i_tilde_torch, max_length):
    feature = torch.zeros(max_length * 2 + max_length * 4 + 1)
    # record s_i_torch.shape[0] at the last element of feature(for restore after DataLoader)
    feature[0:s_i_torch.shape[0] * 2] = s_i_torch.flatten()
    feature[max_length * 2:max_length * 2 +
                           R_i_tilde_torch.shape[0] * 4] = R_i_tilde_torch.flatten()
    feature[-1] = s_i_torch.shape[0]
    return feature


def occ_list_split_component(occ_list):
    occ_data = [
        np.hsplit(occ_array, occ_array.shape[1]) for occ_array in occ_list
    ]
    return occ_data


# ## Generating Dataset Iterators


# generate features and labels
def DPSE_data_gen_single(chgcar_path, r_cut, r_cs, padding_tuple):
    vec, coor_list, chg, ngxf, ngyf, ngzf, atom_num_list, phantom_list_np = split_read_chgcar(chgcar_path)
    coor_list_cart = dir2cart(vec, coor_list)
    coor = coor_occ_gen(coor_list_cart,
                        atom_num_list)  # concatenate coor like POSCAR
    coor_PBC = PBC_padding(coor, vec, r_cut)

    occ_list = read_occ(chgcar_path)

    max_length = coor_PBC.shape[0]  # used in flatten_and_padding

    feature_list = []  # len = element_num
    for element_index in range(len(atom_num_list)):
        features = np.zeros((atom_num_list[element_index], max_length * 6 + 1))
        for atom_index in range(phantom_list_np[element_index],
                                phantom_list_np[element_index + 1]):
            s_i_torch, R_i_tilde_torch = DPSE_feature_pre(
                coor_PBC, atom_index, r_cut, r_cs)
            feature = flatten_and_padding(s_i_torch, R_i_tilde_torch,
                                          max_length)
            features[atom_index - phantom_list_np[element_index]] = feature
        feature_list.append(features)
    return atom_num_list, feature_list, occ_list


# generate features and labels from the whole dataset
def dataset_directory(dir_name,
                      chgcar_keyword='CHGCAR',
                      r_cut=6,
                      r_cs=3,
                      padding_tuple=(1, 1, 1)):
    name_list = os.listdir(dir_name)
    pattern_str = r'[^.]*' + chgcar_keyword + r'.*'
    pattern_str = chgcar_keyword + r'.*'
    pattern = re.compile(pattern_str)
    name_list_match = [
        name for name in name_list if pattern.match(name) is not None
    ]
    feature_list_all, occ_list_all = [], []
    for chgcar_name in name_list_match:
        chgcar_path = '/'.join((dir_name, chgcar_name))
        atom_num_list, feature_list, occ_list = DPSE_data_gen_single(
            chgcar_path, r_cut, r_cs, padding_tuple)
        feature_list_all.append(feature_list)
        occ_list_all.append(occ_list)

    return atom_num_list, feature_list_all, occ_list_all


# generate iterators of data
def data_iter_gen(atom_num_list,
                  feature_list_all,
                  occ_list_all,
                  batch_size=2,
                  shuffle=True):
    data_iter_list = []
    for element_index in range(len(atom_num_list)):
        for file_index in range(len(feature_list_all)):
            feature = feature_list_all[file_index][element_index]
            label = occ_list_all[file_index][element_index]
            if file_index == 0:
                dataset = Data.TensorDataset(
                    torch.from_numpy(feature).float(),
                    torch.from_numpy(label).float())
            else:
                dataset += Data.TensorDataset(
                    torch.from_numpy(feature).float(),
                    torch.from_numpy(label).float())

        data_iter = Data.DataLoader(dataset, batch_size, shuffle=shuffle)
        #         print(element_index, len(dataset))
        data_iter_list.append(data_iter)

    return data_iter_list  # length of the list is the number of element types


# wrap above all
def DPSE_data_gen(dir_name='.',
                  chgcar_keyword='CHGCAR',
                  r_cut=6,
                  r_cs=3,
                  padding_tuple=(1, 1, 1),
                  batch_size=2,
                  shuffle=True):
    atom_num_list, feature_list_all, occ_list_all = dataset_directory(
        dir_name, chgcar_keyword, r_cut, r_cs, padding_tuple)
    data_iter_list = data_iter_gen(atom_num_list,
                                   feature_list_all,
                                   occ_list_all,
                                   batch_size,
                                   shuffle=shuffle)
    return data_iter_list


# ## Model and Network


class local_embedding(nn.Module):  # single atom j
    def __init__(self, n_o, n_h):
        super(local_embedding, self).__init__()
        self.n_list = [1, *n_h, n_o]  # input dim = 1
        self.linears = nn.ModuleList([])
        for i in range(len(self.n_list) - 1):
            self.linears.append(nn.Linear(self.n_list[i], self.n_list[i + 1]))
            if i != len(
                    self.n_list) - 2:  # no activation after the last hidden
                self.linears.append(nn.ReLU())

    def forward(self, x):
        for layer in self.linears:
            x = layer(x)
        return x


class local_embedding_cat(nn.Module):  # all atom j
    def __init__(self, n_o, n_h, num_element):
        super(local_embedding_cat, self).__init__()
        self.linears_list = nn.ModuleList(
            [local_embedding(n_o, n_h) for i in range(num_element)])
        self.n_o = n_o

    def forward(self, x):
        output = torch.zeros(x.shape[0], self.n_o).to(x.device)
        for i in range(x.shape[0]):
            # int(x[i, 1]) is the element sign of data i
            output[i] = self.linears_list[int(x[i, 1])](x[i, 0:1])
        return output


class BatchEmbedding(nn.Module):  # batch -> batch
    def __init__(self, M_1, M_2, embed_hidden_list, num_element):
        super(BatchEmbedding, self).__init__()
        self.M_1 = M_1
        self.M_2 = M_2
        self.embedding = local_embedding_cat(M_1, embed_hidden_list,
                                             num_element)

    def forward(self, x):  # x: (batch_size, PBC_atom_num*6 + 1)
        PBC_atom_num = int((x.shape[1] - 1) / 6)
        out = torch.zeros(x.shape[0], self.M_1 * self.M_2).to(x.device)

        for index, sample in enumerate(x[:]):  # through batch
            N_i = int(sample[-1])  # num of atoms in the cutoff sphere

            s_i_torch = sample[0:N_i * 2].view(N_i, 2)
            R_i_tilde_torch = sample[PBC_atom_num * 2:PBC_atom_num * 2 +
                                                      N_i * 4].view(N_i, 4)
            g_i1 = self.embedding(s_i_torch)
            g_i2 = g_i1[:, 0:self.M_2]
            D_i = g_i1.t().mm(R_i_tilde_torch).mm(R_i_tilde_torch.t()).mm(g_i2)

            out[index, :] = D_i.view(-1)
        return out  # (batch,M_1*M_2)


class MLP(nn.Module):  # for D_i
    def __init__(self, n_i, n_h):
        super(MLP, self).__init__()
        self.n_list = [n_i, *n_h, 1]  # output dim = 1
        self.linears = nn.ModuleList([])
        # normalization with no affine params
        #         self.linears.append(nn.BatchNorm1d(n_i, affine=False))
        for i in range(len(self.n_list) - 1):
            self.linears.append(nn.Linear(self.n_list[i], self.n_list[i + 1]))
            if i != len(
                    self.n_list) - 2:  # no activation after the last hidden
                self.linears.append(nn.ReLU())

    def forward(self, x):
        for layer in self.linears:
            x = layer(x)
        return x


class DPSE(nn.Module):  # 1 atom -> 1 component
    def __init__(self, M_1, M_2, embed_hidden_list, num_element,
                 mlp_hiddens_list):
        super(DPSE, self).__init__()
        self.embedding = BatchEmbedding(M_1, M_2, embed_hidden_list,
                                        num_element)
        self.reg_mlp = MLP(M_1 * M_2, mlp_hiddens_list)

    def forward(self, x):
        return self.reg_mlp(self.embedding(x))


# ## Training


def evaluate_loss(idx_component,
                  test_iter,
                  net,
                  loss,
                  device=None,
                  show_predict=False):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    test_l_sum, batch_count = 0.0, 0
    with torch.no_grad():
        for X, y in test_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()  # close dropout
                y = y[:, idx_component].view(-1, 1).to(device)
                y_hat = net(X.to(device))
                test_l_sum += loss(y_hat, y).cpu().item()
                if show_predict and batch_count == 0:
                    print("predicted density", "\n", "actual density")
                    print(torch.cat((y_hat, y), dim=1))
                net.train()  # restore the training mode
            batch_count += 1
    return test_l_sum / batch_count


# train overlap density(occ) for one element and one component
def train_occ(idx_component,
              optimizer_hyperparams,
              train_iter,
              test_iter,
              num_epochs=10):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    net = DPSE(50, 4, [16, 80, 160, 80, 16], 3, [600, 600, 600, 300])
    net = net.to(device)
    print("training on ", device)
    loss = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), **optimizer_hyperparams)
    train_ls, test_ls = [], []
    for epoch in range(num_epochs):
        train_l_sum, batch_count = 0, 0
        start = time.time()
        for X, y in train_iter:
            if epoch == 0:
                batch_size = X.shape[0]
            X = X.to(device)
            # only trained for one particular component
            y = y[:, idx_component].view(-1, 1).to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            batch_count += 1
        if (epoch + 1) % 1 == 0:
            print('epoch %d, loss %.8f, time %.1f sec' %
                  (epoch + 1, train_l_sum / batch_count, time.time() - start))
        train_ls.append(train_l_sum / batch_count)

        if epoch == (num_epochs - 1):
            test_ls.append(
                evaluate_loss(idx_component,
                              test_iter,
                              net,
                              loss,
                              device=None,
                              show_predict=True))
        else:
            test_ls.append(
                evaluate_loss(idx_component,
                              test_iter,
                              net,
                              loss,
                              device=None,
                              show_predict=False))

    plt.plot(np.arange(1, epoch + 2), train_ls, label='train')
    plt.plot(np.arange(1, epoch + 2), test_ls, label='test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    image_name = '{}_{}_{}.png'.format(str(optimizer_hyperparams['lr']),
                                       str(batch_size), str(num_epochs))

    plt.savefig(image_name, dpi=600)


# [2]: 3rd element (O) in this CHGCAR
def main(optimizer_hyperparams):
    lr = optimizer_hyperparams['lr']
    batch_size = optimizer_hyperparams['batch_size']
    num_epochs = optimizer_hyperparams['num_epochs']

    # redirect print
    file_name = 'new_{}_{}_{}.log'.format(str(optimizer_hyperparams['lr']),
                                          str(batch_size), str(num_epochs))
    output_file = open(file_name, "w")
    sys.stdout = output_file

    train_iter = DPSE_data_gen(dir_name='./CHGCAR_all/train',
                               chgcar_keyword='CHGCAR_',
                               r_cut=6,
                               r_cs=4,
                               padding_tuple=(1, 1, 1),
                               batch_size=batch_size,
                               shuffle=True)[2]
    test_iter = DPSE_data_gen(dir_name='./CHGCAR_all/test',
                              chgcar_keyword='CHGCAR_',
                              r_cut=6,
                              r_cs=4,
                              padding_tuple=(1, 1, 1),
                              batch_size=batch_size,
                              shuffle=True)[2]
    # 0: 1st component of O overlap density
    train_occ(0, {'lr': lr}, train_iter, test_iter, num_epochs=num_epochs)


if __name__ == '__main__':
    main({'lr': 0.001, 'batch_size': 64, 'num_epochs': 100})

# net = local_embedding_cat( 20, [16,32,16], 2)

# DPSE_feature_gen(*DPSE_feature_pre(a, 0, 0.8, 0.4), 20, 4, [16, 32, 16], 3)
