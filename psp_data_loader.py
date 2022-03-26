import numpy as np
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
from utils import random_index, TT_split, normalize
import torch
import random


def load_data(dataset, neg_prop, complete_prop, is_noise):
    all_data = []
    train_pairs = []
    label = []

    mat = sio.loadmat('./datasets/' + dataset + '.mat')
    if dataset == 'Scene15':
        data = mat['X'][0][0:2]  # 20, 59 dimensions
        label = np.squeeze(mat['Y'])

    elif dataset == 'Caltech101':
        data = mat['X'][0][3:5]
        label = np.squeeze(mat['Y'])

    elif dataset == 'Reuters_dim10':
        data = []  # 18758 samples
        data.append(normalize(np.vstack((mat['x_train'][0], mat['x_test'][0]))))
        data.append(normalize(np.vstack((mat['x_train'][1], mat['x_test'][1]))))
        label = np.squeeze(np.hstack((mat['y_train'], mat['y_test'])))

    elif dataset == 'NoisyMNIST-30000':
        data = []
        data.append(mat['X1'])
        data.append(mat['X2'])
        label = np.squeeze(mat['Y'])

    elif dataset == 'Caltech101-deepfea':
        data = []
        label = mat['gt']
        data.append(mat['X'][0][0].T)
        data.append(mat['X'][0][1].T)

    elif dataset == 'MNIST-USPS':
        data = []
        data.append(mat['X1'])
        data.append(normalize(mat['X2']))
        label = np.squeeze(mat['Y'])

    elif dataset == 'AWA-deepfea':
        data = []
        label = mat['gt']
        data.append(mat['X'][0][5].T)
        data.append(mat['X'][0][6].T)

    divide_seed = random.randint(1, 1000)
    all_data.append(data[0].T)
    all_data.append(data[1].T)
    mask = get_sn(2, len(data[0]), 1 - complete_prop)
    unit_mask = np.logical_and(mask[:, 0], mask[:, 1])

    train_label, test_label = label[unit_mask], label[unit_mask]
    all_label = label
    train_X, train_Y = data[0][unit_mask], data[1][unit_mask]

    # pair construction. view 0 and 1 refer to pairs constructed for training. noisy and real labels refer to 0/1 label of those pairs
    view0, view1, noisy_labels, real_labels, _, _ = get_pairs(train_X, train_Y, neg_prop, train_label)

    count = 0
    for i in range(len(noisy_labels)):
        if noisy_labels[i] != real_labels[i]:
            count += 1
    print('noise rate of the constructed neg. pairs is ', round(count / (len(noisy_labels) - len(train_X)), 2))

    if is_noise:  # training with noisy negative correspondence
        print("----------------------Training with noisy_labels----------------------")
        train_pair_labels = noisy_labels
    else:  # training with gt negative correspondence
        print("----------------------Training with real_labels----------------------")
        train_pair_labels = real_labels
    train_pairs.append(view0.T)
    train_pairs.append(view1.T)
    train_pair_real_labels = real_labels

    return train_pairs, train_pair_labels, train_pair_real_labels, all_data, all_label, all_label, all_label, divide_seed, mask


def get_pairs(train_X, train_Y, neg_prop, train_label):
    view0, view1, labels, real_labels, class_labels0, class_labels1 = [], [], [], [], [], []
    # construct pos. pairs
    for i in range(len(train_X)):
        view0.append(train_X[i])
        view1.append(train_Y[i])
        labels.append(1)
        real_labels.append(1)
        class_labels0.append(train_label[i])
        class_labels1.append(train_label[i])
    # construct neg. pairs by taking each sample in view0 as an anchor and randomly sample neg_prop samples from view1,
    # which may lead to the so called noisy labels, namely, some of the constructed neg. pairs may in the same category.
    for j in range(len(train_X)):
        neg_idx = random.sample(range(len(train_Y)), neg_prop)
        for k in range(neg_prop):
            view0.append(train_X[j])
            view1.append(train_Y[neg_idx[k]])
            labels.append(0)
            class_labels0.append(train_label[j])
            class_labels1.append(train_label[neg_idx[k]])
            if train_label[j] != train_label[neg_idx[k]]:
                real_labels.append(0)
            else:
                real_labels.append(1)

    labels = np.array(labels, dtype=np.int64)
    real_labels = np.array(real_labels, dtype=np.int64)
    class_labels0, class_labels1 = np.array(class_labels0, dtype=np.int64), np.array(class_labels1, dtype=np.int64)
    view0, view1 = np.array(view0, dtype=np.float32), np.array(view1, dtype=np.float32)
    return view0, view1, labels, real_labels, class_labels0, class_labels1


def get_sn(view_num, alldata_len, missing_rate):
    """Randomly generate incomplete data information, simulate partial view data with complete view data
    :param view_num:view number
    :param alldata_len:number of samples
    :param missing_rate:Defined in section 4.3 of the paper
    :return:Sn
    """
    missing_rate = missing_rate / 2
    one_rate = 1.0 - missing_rate
    if one_rate <= (1 / view_num):
        enc = OneHotEncoder()  # n_values=view_num
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
        return view_preserve
    error = 1
    if one_rate == 1:
        matrix = randint(1, 2, size=(alldata_len, view_num))
        return matrix
    while error >= 0.005:
        enc = OneHotEncoder()  # n_values=view_num
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
        one_num = view_num * alldata_len * one_rate - alldata_len
        ratio = one_num / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int)
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int))
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int)
        matrix = ((matrix_iter + view_preserve) > 0).astype(np.int)
        ratio = np.sum(matrix) / (view_num * alldata_len)
        error = abs(one_rate - ratio)
    return matrix


class GetDataset(Dataset):
    def __init__(self, data, labels, real_labels):
        self.data = data
        self.labels = labels
        self.real_labels = real_labels

    def __getitem__(self, index):
        fea0, fea1 = (torch.from_numpy(self.data[0][:, index])).type(torch.FloatTensor), (
            torch.from_numpy(self.data[1][:, index])).type(torch.FloatTensor)
        fea0, fea1 = fea0.unsqueeze(0), fea1.unsqueeze(0)
        label = np.int64(self.labels[index])
        if len(self.real_labels) == 0:
            return fea0, fea1, label
        real_label = np.int64(self.real_labels[index])
        return fea0, fea1, label, real_label

    def __len__(self):
        return len(self.labels)


class GetAllDataset(Dataset):
    def __init__(self, data, labels, class_labels0, class_labels1, mask):
        self.data = data
        self.labels = labels
        self.class_labels0 = class_labels0
        self.class_labels1 = class_labels1
        self.mask = mask

    def __getitem__(self, index):
        fea0, fea1 = (torch.from_numpy(self.data[0][:, index])).type(torch.FloatTensor), (
            torch.from_numpy(self.data[1][:, index])).type(torch.FloatTensor)
        fea0, fea1 = fea0.unsqueeze(0), fea1.unsqueeze(0)
        label = np.int64(self.labels[index])
        class_labels0 = np.int64(self.class_labels0[index])
        class_labels1 = np.int64(self.class_labels1[index])
        mask = np.int64(self.mask[index])
        return fea0, fea1, label, class_labels0, class_labels1, mask

    def __len__(self):
        return len(self.labels)


def loader(train_bs, neg_prop, aligned_prop, complete_prop, is_noise, dataset):
    """
    :param train_bs: batch size for training, default is 1024
    :param neg_prop: negative / positive pairs' ratio
    :param complete_prop: complete proportions for training SURE
    :param is_noise: training with noisy labels or not, 0 --- not, 1 --- yes
    :param dataset: choice of dataset
    :return: train_pair_loader including the constructed pos. and neg. pairs used for training MvCLN, all_loader including originally aligned and unaligned data used for testing SURE
    """
    train_pairs, train_pair_labels, train_pair_real_labels, all_data, all_label, all_label_X, all_label_Y, \
    divide_seed, mask = load_data(dataset, neg_prop, complete_prop, is_noise)
    train_pair_dataset = GetDataset(train_pairs, train_pair_labels, train_pair_real_labels)
    all_dataset = GetAllDataset(all_data, all_label, all_label_X, all_label_Y, mask)

    train_pair_loader = DataLoader(
        train_pair_dataset,
        batch_size=train_bs,
        shuffle=True,
        drop_last=True
    )
    all_loader = DataLoader(
        all_dataset,
        batch_size=1024,
        shuffle=True
    )
    return train_pair_loader, all_loader, divide_seed