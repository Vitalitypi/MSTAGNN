import csv

import numpy as np
import pandas as pd
import torch
import datetime as dt
from datetime import datetime
from utils.norm import StandardScaler

def get_tod(periods,time_stamps,num_nodes,out_steps=12):
    tod = np.zeros((time_stamps, num_nodes, 2))
    for t in range(time_stamps):
        tod[t, :, 0] = np.ones(num_nodes) * ((t % periods)/periods)
        tod[t, :, 1] = np.ones(num_nodes) * (((t + out_steps) % periods)/periods)
    return tod

def get_tod_dow(dataset,time_stamps,num_nodes,periods,out_steps=12):
    day_dict = {
        'PEMS03':'2018-09-01',
        'PEMS04':'2018-01-01',
        'PEMS07':'2017-05-01',
        'PEMS08':'2016-07-01'
    }
    start_date = datetime.strptime(day_dict[dataset], "%Y-%m-%d")
    time_feature = np.zeros((time_stamps,num_nodes,4))
    current_date = start_date
    for t in range(time_stamps):

        time_feature[t, :, 0] = np.ones(num_nodes) * ((t % periods)/periods)
        time_feature[t, :, 2] = np.ones(num_nodes) * (((t + out_steps) % periods)/periods)

        date_info = np.ones((num_nodes))*(current_date.weekday())
        time_feature[t, :, 1] = date_info

        future_date = current_date + dt.timedelta(hours=1)
        date_info = np.ones((num_nodes))*(future_date.weekday())
        time_feature[t, :, 3] = date_info

        current_date = current_date + dt.timedelta(minutes=5)
    return time_feature


# For PEMS03/04/07/08 Datasets
def get_dataloader_pems(dataset, batch_size=64, val_ratio=0.2, test_ratio=0.2, in_steps=12, out_steps=12, periods=288):
    # load data
    data = np.load('./dataset/{}/{}.npz'.format(dataset, dataset))['data'][...,:1]
    # print the shape of data
    print(data.shape)
    time_stamps, num_nodes, _ = data.shape
    # tods = get_tod(periods,time_stamps,num_nodes)
    time_features = get_tod_dow(dataset,time_stamps,num_nodes,periods)
    # concatenate all data
    data = np.concatenate([data,time_features],axis=-1)

    # normalize data(only normalize the first dimension data)
    mean = data[..., 0].mean()
    std = data[..., 0].std()
    scaler = StandardScaler(mean, std)
    data[..., 0] = scaler.transform(data[..., 0])
    
    # spilit dataset by days or by ratio
    data_train, data_val, data_test = split_data_by_ratio(data, val_ratio, test_ratio)
    # add time window [B, N, 1]
    x_tra, y_tra = Add_Window_Horizon(data_train, in_steps, out_steps)
    x_val, y_val = Add_Window_Horizon(data_val, in_steps, out_steps)
    x_test, y_test = Add_Window_Horizon(data_test, in_steps, out_steps)

    print('Train: ', x_tra.shape, y_tra.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)

    ##############get dataloader######################
    train_dataloader = data_loader(x_tra, y_tra, batch_size, shuffle=True, drop_last=True)
    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader(x_val, y_val, batch_size, shuffle=False, drop_last=True)
    test_dataloader = data_loader(x_test, y_test, batch_size, shuffle=False, drop_last=False)

    return train_dataloader, val_dataloader, test_dataloader, scaler

# For PEMS-Bay and METR-LA Datasets
def get_dataloader_meta_la(args, normalizer='std', tod=False, dow=False, weather=False, single=True):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join("./dataset", args.dataset, category + '.npz'))
        data['x_' + category] = cat_data['x'] # [B, T, N, 2]
        data['y_' + category] = np.expand_dims(cat_data['y'][:, :, :, 0], axis=-1) # [B, T, N, 1]

    # data normalization method following DCRNN
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    for category in ['train', 'val', 'test']:
        data['x_' + category][:, :, :, 0] = scaler.transform(data['x_' + category][:, :, :, 0])
    if not args.real_value:
        data['y_' + category][:, :, :, 0] = scaler.transform(data['y_' + category][:, :, :, 0])

    x_tra, y_tra = data['x_train'], data['y_train']
    x_val, y_val = data['x_val'], data['y_val']
    x_test, y_test = data['x_test'], data['y_test']

    print('Train: ', x_tra.shape, y_tra.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)
    # print(x_tra[:10], x_val[:10], x_test[:10])
    # print(y_tra[:10], y_val[:10], y_test[:10])

    ##############get dataloader######################
    train_dataloader = data_loader(x_tra, y_tra, args.batch_size, shuffle=True, drop_last=True)
    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader(x_val, y_val, args.batch_size, shuffle=False, drop_last=True)
    test_dataloader = data_loader(x_test, y_test, args.batch_size, shuffle=False, drop_last=False)

    return train_dataloader, val_dataloader, test_dataloader, scaler


def get_adj_dis_matrix(dataset, adj_file, num_of_vertices, direction=False, id_filename=None):
    '''
    Parameters
    ----------
    adj_file: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    max_dict = {
        'PEMS03': 10.194,
        'PEMS04': 2712.1,
        'PEMS07': 20.539,
        'PEMS08': 3274.4
    }
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    # if node id in distance_df_file doesn't start from zero,
    # it needs to be remap via id_filename which contains the corresponding id with sorted index.
    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

        with open(adj_file, 'r') as f:
            f.readline()  # 略过表头那一行
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                if i==j:
                    continue
                A[id_dict[i], id_dict[j]] = 1
                distaneA[id_dict[i], id_dict[j]] = max_dict[dataset] / distance
                if not direction:
                    A[id_dict[j], id_dict[i]] = 1
                    distaneA[id_dict[j], id_dict[i]] = max_dict[dataset] / distance

        return A, distaneA  # adj matrix, distance matrix

    else:  # distance_df_file: node id starts from zero
        with open(adj_file, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[i, j] = 1
                distaneA[i, j] = max_dict[dataset] / distance
                if not direction:
                    A[j, i] = 1
                    distaneA[j, i] = max_dict[dataset] / distance
        return A, distaneA


def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len * test_ratio):]
    val_data = data[-int(data_len * (test_ratio + val_ratio)):-int(data_len * test_ratio)]
    train_data = data[:-int(data_len * (test_ratio + val_ratio))]

    return train_data, val_data, test_data


def Add_Window_Horizon(data, window=12, horizon=12):
    '''
    :param data: shape [B, N, D]
    :param window:
    :param horizon:
    :return: X is [B', W, N, D], Y is [B', H, N, D], B' = B - W - H + 1
    '''
    length = len(data)
    total_num = length - horizon - window + 1
    X = []  # windows
    Y = []  # horizon
    index = 0
    while index < total_num:
        X.append(data[index:index + window])
        Y.append(data[index + window:index + window + horizon, :, :1])
        index = index + 1
    X = np.array(X)
    Y = np.array(Y)

    return X, Y


def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)

    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             drop_last=drop_last)
    return dataloader


def norm_adj(W):
    '''
    compute  normalized Adj matrix

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    normalized Adj matrix: (D^hat)^{-1} A^hat; np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]

    N = W.shape[0]
    W = W + np.identity(N)  # add self-connection
    D = np.diag(1.0 / np.sum(W, axis=1))
    norm_Adj_matrix = np.dot(D, W)

    return norm_Adj_matrix

def construct_adj(A, steps=3):
    """
    构建local 时空图
    :param A: np.ndarray, adjacency matrix, shape is (N, N)
    :param steps: 选择几个时间步来构建图
    :return: new adjacency matrix: csr_matrix, shape is (N * steps, N * steps)
    """
    N = len(A)  # 获得行数
    adj = np.zeros((N * steps, N * steps))

    for i in range(steps):
        """对角线代表各个时间步自己的空间图，也就是A"""
        adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A

    for i in range(N):
        for k in range(steps - 1):
            """每个节点只会连接相邻时间步的自己"""
            adj[k * N + i, (k + 1) * N + i] = 1
            adj[(k + 1) * N + i, k * N + i] = 1

    for i in range(len(adj)):
        """加入自回"""
        adj[i, i] = 1

    return adj
