# encoding=utf-8
"""
    Code adopted from Open-Source Code: https://github.com/jindongwang/Deep-learning-activity-recognition/
    @author: Jindong Wang
"""


import numpy as np
from torch.utils.data import Dataset, DataLoader
from os.path import expanduser
from utils.code_util import arguements
from scipy.signal import resample


def format_data_x(datafile):
    x_data = None
    for item in datafile:
        item_data = np.loadtxt(item, dtype=float)
        if x_data is None:
            x_data = np.zeros((len(item_data), 1))
        x_data = np.hstack((x_data, item_data))
    x_data = x_data[:, 1:]
    X = None
    for i in range(len(x_data)):
        row = np.asarray(x_data[i, :])
        row = row.reshape(9, 128).T
        if X is None:
            X = np.zeros((len(x_data), 128, 9))
        X[i] = row
    return X


def format_data_y(datafile):
    data = np.loadtxt(datafile, dtype=int) - 1
    YY = np.eye(6)[data]
    return YY


def load_data(data_folder):
    # After downloading the dataset, put it to somewhere that str_folder can find
    str_folder = data_folder + 'UCI HAR Dataset/'
    INPUT_SIGNAL_TYPES = [
        "body_acc_x_", "body_acc_y_", "body_acc_z_",
        "body_gyro_x_", "body_gyro_y_", "body_gyro_z_",
        "total_acc_x_", "total_acc_y_", "total_acc_z_"
    ]

    str_train_files = [str_folder + 'train/' + 'Inertial Signals/' + item + 'train.txt' for item in INPUT_SIGNAL_TYPES]
    str_test_files = [str_folder + 'test/' + 'Inertial Signals/' + item + 'test.txt' for item in INPUT_SIGNAL_TYPES]
    str_train_y = str_folder + 'train/y_train.txt'
    str_test_y = str_folder + 'test/y_test.txt'

    X_train = format_data_x(str_train_files)
    X_test = format_data_x(str_test_files)
    y_train = format_data_y(str_train_y)
    y_test = format_data_y(str_test_y)

    return X_train, onehot_to_label(y_train), X_test, onehot_to_label(y_test)


def onehot_to_label(y_onehot):
    a = np.argwhere(y_onehot == 1)
    return a[:, -1]


class data_loader(Dataset):
    def __init__(self, samples, labels, t):
        self.samples = samples
        self.labels = labels
        self.T = t

    def __getitem__(self, index):
        sample, target = self.samples[index], self.labels[index]

        if self.T:
            return self.T(sample), target
        else:
            return sample, target

    def __len__(self):
        return len(self.samples)


def load(path: str, batch_size: int = 64):

    x_train, y_train, x_test, y_test = load_data(path)

    x_train = x_train.reshape((-1, 9, 1, 128))
    x_test = x_test.reshape((-1, 9, 1, 128))

    transform = None
    train_set = data_loader(x_train, y_train, transform)
    test_set = data_loader(x_test, y_test, transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


if __name__ == '__main__':
    args = arguements()  # read arguments from command line

    data_path = expanduser("~") + '/DATASET/UCI-HAR/'
    train_loader, test_loader = load(path=data_path, batch_size=args.batch_size)

    for _, (data, labels) in enumerate(train_loader):
        pass
        # print(data.shape)
        # print(labels.shape)
