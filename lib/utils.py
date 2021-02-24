import os
import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from .metrics import masked_mape_np
from time import time
from scipy.sparse.linalg import eigs


def re_normalization(x, mean, std):
    x = x * std + mean
    return x


def max_min_normalization(x, _max, _min):
    x = 1. * (x - _min)/(_max - _min)
    x = x * 2. - 1.
    return x


def re_max_min_normalization(x, _max, _min):
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x


def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)

        # distance file中的id并不是从0开始的 所以要进行重新的映射；id_filename是节点的顺序
        if id_filename:

            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

            with open(distance_df_filename, 'r') as f:
                f.readline()  # 略过表头那一行
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
            return A, distaneA

        else:  # distance file中的id直接从0开始

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance
            return A, distaneA


def get_adjacency_matrix_2direction(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)

        # distance file中的id并不是从0开始的 所以要进行重新的映射；id_filename是节点的顺序
        if id_filename:

            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

            with open(distance_df_filename, 'r') as f:
                f.readline()  # 略过表头那一行
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    A[id_dict[j], id_dict[i]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
                    distaneA[id_dict[j], id_dict[i]] = distance
            return A, distaneA

        else:  # distance file中的id直接从0开始

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    A[j, i] = 1
                    distaneA[i, j] = distance
                    distaneA[j, i] = distance
            return A, distaneA


def get_Laplacian(A):
    '''
    compute the graph Laplacian, which can be represented as L = D − A

    Parameters
    ----------
    A: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    Laplacian matrix: np.ndarray, shape (N, N)

    '''

    assert (A-A.transpose()).sum() == 0  # 首先确保A是一个对称矩阵

    D = np.diag(np.sum(A, axis=1))  # D是度矩阵，只有对角线上有元素

    L = D - A  # L是实对称矩阵A，有n个不同特征值对应的特征向量是正交的。

    return L


def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))  # D是度矩阵，只有对角线上有元素

    L = D - W  # L是实对称矩阵A，有n个不同特征值对应的特征向量是正交的。

    lambda_max = eigs(L, k=1, which='LR')[0].real  # 求解拉普拉斯矩阵的最大奇异值

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def sym_norm_Adj(W):
    '''
    compute Symmetric normalized Adj matrix

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    Symmetric normalized Laplacian: (D^hat)^1/2 A^hat (D^hat)^1/2; np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]

    N = W.shape[0]
    W = W + np.identity(N) # 为邻居矩阵加上自连接
    D = np.diag(np.sum(W, axis=1))
    sym_norm_Adj_matrix = np.dot(np.sqrt(D),W)
    sym_norm_Adj_matrix = np.dot(sym_norm_Adj_matrix,np.sqrt(D))

    return sym_norm_Adj_matrix


def norm_Adj(W):
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
    W = W + np.identity(N)  # 为邻接矩阵加上自连接
    D = np.diag(1.0/np.sum(W, axis=1))
    norm_Adj_matrix = np.dot(D, W)

    return norm_Adj_matrix


def trans_norm_Adj(W):
    '''
    compute  normalized Adj matrix

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    Symmetric normalized Laplacian: (D^hat)^1/2 A^hat (D^hat)^1/2; np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]

    W = W.transpose()
    N = W.shape[0]
    W = W + np.identity(N)  # 为邻居矩阵加上自连接
    D = np.diag(1.0/np.sum(W, axis=1))
    trans_norm_Adj = np.dot(D, W)

    return trans_norm_Adj


def compute_val_loss(net, val_loader, criterion, sw, epoch):
    '''
    compute mean loss on validation set
    :param net: model
    :param val_loader: torch.utils.data.utils.DataLoader
    :param criterion: torch.nn.MSELoss
    :param sw: tensorboardX.SummaryWriter
    :param epoch: int, current epoch
    :return: val_loss
    '''

    net.train(False)  # ensure dropout layers are in evaluation mode

    with torch.no_grad():

        val_loader_length = len(val_loader)  # nb of batch

        tmp = []  # 记录了所有batch的loss

        start_time = time()

        for batch_index, batch_data in enumerate(val_loader):

            encoder_inputs, decoder_inputs, labels = batch_data

            encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)

            decoder_inputs = decoder_inputs.unsqueeze(-1)  # (B, N, T, 1)

            labels = labels.unsqueeze(-1)  # (B，N，T，1)

            predict_length = labels.shape[2]  # T
            # encode
            encoder_output = net.encode(encoder_inputs)
            # print('encoder_output:', encoder_output.shape)
            # decode
            decoder_start_inputs = decoder_inputs[:, :, :1, :]  # 只取输入的第一个值作为input，之后都用predict出来的值作为input
            decoder_input_list = [decoder_start_inputs]
            # 按着时间步进行预测
            for step in range(predict_length):
                decoder_inputs = torch.cat(decoder_input_list, dim=2)
                predict_output = net.decode(decoder_inputs, encoder_output)
                decoder_input_list = [decoder_start_inputs, predict_output]

            loss = criterion(predict_output, labels)  # 计算误差
            tmp.append(loss.item())
            if batch_index % 100 == 0:
                print('validation batch %s / %s, loss: %.2f' % (batch_index + 1, val_loader_length, loss.item()))

        print('validation cost time: %.4fs' %(time()-start_time))

        validation_loss = sum(tmp) / len(tmp)
        sw.add_scalar('validation_loss', validation_loss, epoch)

    return validation_loss


def predict_and_save_results(net, data_loader, data_target_tensor, epoch, _max, _min, params_path, type):
    '''
    for transformerGCN
    :param net: nn.Module
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param epoch: int
    :param _max: (1, 1, 3, 1)
    :param _min: (1, 1, 3, 1)
    :param params_path: the path for saving the results
    :return:
    '''
    net.train(False)  # ensure dropout layers are in test mode

    start_time = time()

    with torch.no_grad():

        data_target_tensor = data_target_tensor.cpu().numpy()

        loader_length = len(data_loader)  # nb of batch

        prediction = []

        input = []  # 存储所有batch的input

        start_time = time()

        for batch_index, batch_data in enumerate(data_loader):

            encoder_inputs, decoder_inputs, labels = batch_data

            encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)

            decoder_inputs = decoder_inputs.unsqueeze(-1)  # (B, N, T, 1)

            labels = labels.unsqueeze(-1)  # (B, N, T, 1)

            predict_length = labels.shape[2]  # T

            # encode
            encoder_output = net.encode(encoder_inputs)
            input.append(encoder_inputs[:, :, :, 0:1].cpu().numpy())  # (batch, T', 1)

            # decode
            decoder_start_inputs = decoder_inputs[:, :, :1, :]  # 只取输入的第一个值作为input，之后都用predict出来的值作为input
            decoder_input_list = [decoder_start_inputs]

            # 按着时间步进行预测
            for step in range(predict_length):
                decoder_inputs = torch.cat(decoder_input_list, dim=2)
                predict_output = net.decode(decoder_inputs, encoder_output)
                decoder_input_list = [decoder_start_inputs, predict_output]

            prediction.append(predict_output.detach().cpu().numpy())
            if batch_index % 100 == 0:
                print('predicting testing set batch %s / %s, time: %.2fs' % (batch_index + 1, loader_length, time() - start_time))

        print('test time on whole data:%.2fs' % (time() - start_time))
        input = np.concatenate(input, 0)
        input = re_max_min_normalization(input, _max[0, 0, 0, 0], _min[0, 0, 0, 0])

        prediction = np.concatenate(prediction, 0)  # (batch, N, T', 1)
        prediction = re_max_min_normalization(prediction, _max[0, 0, 0, 0], _min[0, 0, 0, 0])
        data_target_tensor = re_max_min_normalization(data_target_tensor, _max[0, 0, 0, 0], _min[0, 0, 0, 0])

        print('input:', input.shape)
        print('prediction:', prediction.shape)
        print('data_target_tensor:', data_target_tensor.shape)
        output_filename = os.path.join(params_path, 'output_epoch_%s_%s' % (epoch, type))
        np.savez(output_filename, input=input, prediction=prediction, data_target_tensor=data_target_tensor)

        # 计算误差
        excel_list = []
        prediction_length = prediction.shape[2]

        for i in range(prediction_length):
            assert data_target_tensor.shape[0] == prediction.shape[0]
            print('current epoch: %s, predict %s points' % (epoch, i))
            mae = mean_absolute_error(data_target_tensor[:, :, i], prediction[:, :, i, 0])
            rmse = mean_squared_error(data_target_tensor[:, :, i], prediction[:, :, i, 0]) ** 0.5
            mape = masked_mape_np(data_target_tensor[:, :, i], prediction[:, :, i, 0], 0)
            print('MAE: %.2f' % (mae))
            print('RMSE: %.2f' % (rmse))
            print('MAPE: %.2f' % (mape))
            excel_list.extend([mae, rmse, mape])

        # print overall results
        mae = mean_absolute_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1))
        rmse = mean_squared_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1)) ** 0.5
        mape = masked_mape_np(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)
        print('all MAE: %.2f' % (mae))
        print('all RMSE: %.2f' % (rmse))
        print('all MAPE: %.2f' % (mape))
        excel_list.extend([mae, rmse, mape])
        print(excel_list)


def load_graphdata_normY_channel1(graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size, shuffle=True, percent=1.0):
    '''
    将x,y都处理成归一化到[-1,1]之前的数据;
    每个样本同时包含所有监测点的数据，所以本函数构造的数据输入时空序列预测模型；
    该函数会把hour, day, week的时间串起来；
    注： 从文件读入的数据，x,y都是归一化后的值
    :param graph_signal_matrix_filename: str
    :param num_of_hours: int
    :param num_of_days: int
    :param num_of_weeks: int
    :param DEVICE:
    :param batch_size: int
    :return:
    three DataLoaders, each dataloader contains:
    test_x_tensor: (B, N_nodes, in_feature, T_input)
    test_decoder_input_tensor: (B, N_nodes, T_output)
    test_target_tensor: (B, N_nodes, T_output)

    '''

    file = os.path.basename(graph_signal_matrix_filename).split('.')[0]

    dirpath = os.path.dirname(graph_signal_matrix_filename)

    filename = os.path.join(dirpath,
                            file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks) + '.npz')

    print('load file:', filename)

    file_data = np.load(filename)
    train_x = file_data['train_x']  # (10181, 307, 3, 12)
    train_x = train_x[:, :, 0:1, :]
    train_target = file_data['train_target']  # (10181, 307, 12)
    train_timestamp = file_data['train_timestamp']  # (10181, 1)

    train_x_length = train_x.shape[0]
    scale = int(train_x_length*percent)
    print('ori length:', train_x_length, ', percent:', percent, ', scale:', scale)
    train_x = train_x[:scale]
    train_target = train_target[:scale]
    train_timestamp = train_timestamp[:scale]

    val_x = file_data['val_x']
    val_x = val_x[:, :, 0:1, :]
    val_target = file_data['val_target']
    val_timestamp = file_data['val_timestamp']

    test_x = file_data['test_x']
    test_x = test_x[:, :, 0:1, :]
    test_target = file_data['test_target']
    test_timestamp = file_data['test_timestamp']

    _max = file_data['mean']  # (1, 1, 3, 1)
    _min = file_data['std']  # (1, 1, 3, 1)

    # 统一对y进行归一化，变成[-1,1]之间的值
    train_target_norm = max_min_normalization(train_target, _max[:, :, 0, :], _min[:, :, 0, :])
    test_target_norm = max_min_normalization(test_target, _max[:, :, 0, :], _min[:, :, 0, :])
    val_target_norm = max_min_normalization(val_target, _max[:, :, 0, :], _min[:, :, 0, :])

    #  ------- train_loader -------
    train_decoder_input_start = train_x[:, :, 0:1, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
    train_decoder_input_start = np.squeeze(train_decoder_input_start, 2)  # (B,N,T(1))
    train_decoder_input = np.concatenate((train_decoder_input_start, train_target_norm[:, :, :-1]), axis=2)  # (B, N, T)

    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    train_decoder_input_tensor = torch.from_numpy(train_decoder_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    train_target_tensor = torch.from_numpy(train_target_norm).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_decoder_input_tensor, train_target_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    #  ------- val_loader -------
    val_decoder_input_start = val_x[:, :, 0:1, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
    val_decoder_input_start = np.squeeze(val_decoder_input_start, 2)  # (B,N,T(1))
    val_decoder_input = np.concatenate((val_decoder_input_start, val_target_norm[:, :, :-1]), axis=2)  # (B, N, T)

    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    val_decoder_input_tensor = torch.from_numpy(val_decoder_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    val_target_tensor = torch.from_numpy(val_target_norm).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_decoder_input_tensor, val_target_tensor)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    #  ------- test_loader -------
    test_decoder_input_start = test_x[:, :, 0:1, -1:]  # (B, N, 1(F), 1(T)),最后已知traffic flow作为decoder 的初始输入
    test_decoder_input_start = np.squeeze(test_decoder_input_start, 2)  # (B,N,T(1))
    test_decoder_input = np.concatenate((test_decoder_input_start, test_target_norm[:, :, :-1]), axis=2)  # (B, N, T)

    test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    test_decoder_input_tensor = torch.from_numpy(test_decoder_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    test_target_tensor = torch.from_numpy(test_target_norm).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_decoder_input_tensor, test_target_tensor)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    # print
    print('train:', train_x_tensor.size(), train_decoder_input_tensor.size(), train_target_tensor.size())
    print('val:', val_x_tensor.size(), val_decoder_input_tensor.size(), val_target_tensor.size())
    print('test:', test_x_tensor.size(), test_decoder_input_tensor.size(), test_target_tensor.size())

    return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _max, _min


#
# def load_data_normY_Metro_working(graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size, prediction_channel, shuffle=True):
#     '''
#     将x,y都处理成归一化到[-1,1]之前的数据;
#     batch*nd_node, 每个样本只是单个监测点的数据，所以本函数构造的数据输入纯时间序列模型;
#     该函数会把hour, day, week的时间串起来；
#     注： 从文件读入的数据，x是归一化的，但是y是真实值
#     :param graph_signal_matrix_filename: str
#     :param num_of_hours: int
#     :param num_of_days: int
#     :param num_of_weeks: int
#     :param DEVICE:
#     :param batch_size: int
#     :return:
#     three DataLoaders, each dataloader contains:
#     test_x_tensor: (B*N_nodes, T_input, in_feature)
#     test_decoder_input_tensor: (B*N_nodes, T_output, out_feature)
#     test_target_tensor: (B*N_nodes, T_output, out_feature)
#
#     '''
#     file = os.path.basename(graph_signal_matrix_filename).split('.')[0]
#
#     dirpath = os.path.dirname(graph_signal_matrix_filename)
#
#     filename = os.path.join(dirpath,
#                             file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks))
#
#     print('load file:', filename)
#
#     file_data = np.load(filename + '.npz')
#     train_x = file_data['train_x']  # (4479, 80, 2, 12)
#     train_target = file_data['train_target']  # (4479, 80, 2, 12)
#     train_timestamp = file_data['train_timestamp']  # (4479, 1)
#
#     val_x = file_data['val_x']
#     val_target = file_data['val_target']
#     val_timestamp = file_data['val_timestamp']
#
#     test_x = file_data['test_x']
#     test_target = file_data['test_target']
#     test_timestamp = file_data['test_timestamp']
#
#     _max = file_data['mean']  # (1, 1, 3, 1)
#     _min = file_data['std']  # (1, 1, 3, 1)
#
#     # 统一对y进行归一化，变成[-1,1]之间的值
#     train_target_norm = max_min_normalization(train_target, _max, _min)
#     val_target_norm = max_min_normalization(val_target, _max, _min)
#     test_target_norm = max_min_normalization(test_target, _max, _min)
#
#     # filter 0-6点的数据
#     train_retain = train_timestamp % (24 * 12) > 6 * 12
#     train_retain_index = np.where(train_retain == True)
#     train_x = train_x[train_retain_index[0], ...]
#     train_x = train_x.transpose(0, 1, 3, 2)[..., prediction_channel:prediction_channel+1]  #(3323, 80, 12, 1)
#     train_target_norm = train_target_norm[train_retain_index[0], ...]
#     train_target_norm = train_target_norm.transpose(0, 1, 3, 2)[..., prediction_channel:prediction_channel+1] #(3323, 80, 12, 1)
#     train_timestamp = train_timestamp[train_retain_index[0], :]
#
#     val_retain = val_timestamp % (24 * 12) > 6 * 12
#     val_retain_index = np.where(val_retain == True)
#     val_x = val_x[val_retain_index[0], ...]
#     val_x = val_x.transpose(0, 1, 3, 2)[..., prediction_channel:prediction_channel+1]  #(3323, 80, 12, 1)
#     val_target_norm = val_target_norm[val_retain_index[0], ...]
#     val_target_norm = val_target_norm.transpose(0, 1, 3, 2)[..., prediction_channel:prediction_channel+1] #(3323, 80, 12, 1)
#     val_timestamp = val_timestamp[val_retain_index[0], :]
#
#     test_retain = test_timestamp % (24 * 12) > 6 * 12
#     test_retain_index = np.where(test_retain == True)
#     test_x = test_x[test_retain_index[0], ...]
#     test_x = test_x.transpose(0, 1, 3, 2)[..., prediction_channel:prediction_channel+1]  #(3323, 80, 12, 1)
#     test_target_norm = test_target_norm[test_retain_index[0], ...]
#     test_target_norm = test_target_norm.transpose(0, 1, 3, 2)[..., prediction_channel:prediction_channel+1] #(3323, 80, 12, 1)
#     test_timestamp = test_timestamp[test_retain_index[0], :]
#
#     #  ------- train_loader -------
#     batch, nd_nodes, T_input, in_feature = train_x.shape
#     train_x = train_x.reshape((batch * nd_nodes, T_input, in_feature))  # (batch * nd_nodes, T_input, in_feature)
#
#     batch, nd_nodes, T_input, in_feature  = train_target_norm.shape
#     train_target_norm = train_target_norm.reshape((batch * nd_nodes, T_input, in_feature))
#
#     # 构造decoder的input
#     train_decoder_input_start = train_x[:, -1:, :]  # (batch * nd_nodes, 1,1),最后已知traffic flow作为decoder 的初始输入
#     train_decoder_input = np.concatenate((train_decoder_input_start, train_target_norm[:, :-1, :]), axis=1)
#
#     train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE)
#     train_decoder_input_tensor = torch.from_numpy(train_decoder_input).type(torch.FloatTensor).to(DEVICE)
#     train_target_tensor = torch.from_numpy(train_target_norm).type(torch.FloatTensor).to(DEVICE)
#
#     train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_decoder_input_tensor, train_target_tensor)
#
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
#
#     #  ------- val_loader -------
#     batch, nd_nodes, T_input, in_feature = val_x.shape
#     val_x = val_x.reshape((batch * nd_nodes, T_input, in_feature))  # (batch * nd_nodes, T_input, in_feature)
#
#     batch, nd_nodes, T_input, in_feature = val_target_norm.shape
#     val_target_norm = val_target_norm.reshape((batch * nd_nodes, T_input, in_feature))
#
#     # 构造decoder的input
#     val_decoder_input_start = val_x[:, -1:, :]  # (batch * nd_nodes, 1,1),最后已知traffic flow作为decoder 的初始输入
#     val_decoder_input = np.concatenate((val_decoder_input_start, val_target_norm[:, :-1, :]), axis=1)
#
#     val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(DEVICE)
#     val_decoder_input_tensor = torch.from_numpy(val_decoder_input).type(torch.FloatTensor).to(DEVICE)
#     val_target_tensor = torch.from_numpy(val_target_norm).type(torch.FloatTensor).to(DEVICE)
#
#     val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_decoder_input_tensor, val_target_tensor)
#
#     val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
#
#     #  ------- test_loader -------
#     batch, nd_nodes, T_input, in_feature = test_x.shape
#     test_x = test_x.reshape((batch * nd_nodes, T_input, in_feature))  # (batch * nd_nodes, T_input, in_feature)
#
#     batch, nd_nodes, T_input, in_feature = test_target_norm.shape
#     test_target_norm = test_target_norm.reshape((batch * nd_nodes, T_input, in_feature))
#
#     # 构造decoder的input
#     test_decoder_input_start = test_x[:, -1:, :]  # (batch * nd_nodes, 1,1),最后已知traffic flow作为decoder 的初始输入
#     test_decoder_input = np.concatenate((test_decoder_input_start, test_target_norm[:, :-1, :]), axis=1)
#
#     test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor).to(DEVICE)
#     test_decoder_input_tensor = torch.from_numpy(test_decoder_input).type(torch.FloatTensor).to(DEVICE)
#     test_target_tensor = torch.from_numpy(test_target_norm).type(torch.FloatTensor).to(DEVICE)
#
#     test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_decoder_input_tensor, test_target_tensor)
#
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
#
#     # print
#     print('train:', train_x_tensor.size(), train_decoder_input_tensor.size(), train_target_tensor.size())
#     print('val:', val_x_tensor.size(), val_decoder_input_tensor.size(), val_target_tensor.size())
#     print('test:', test_x_tensor.size(), test_decoder_input_tensor.size(), test_target_tensor.size())
#
#     return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _max, _min
#
# def load_graphdata_normY_channel2_Metro_working(graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size, prediction_channel, points_per_hour, shuffle=True, use_nni=False):
#     '''
#     抛去0-6点的数据
#     将x,y都处理成归一化到[-1,1]之前的数据;
#     每个样本同时包含所有监测点的数据，所以本函数构造的数据输入时空序列预测模型；
#     该函数会把hour, day, week的时间串起来；
#     注： 从文件读入的数据，x是归一化的，但是y是真实值
#     :param graph_signal_matrix_filename: str
#     :param num_of_hours: int
#     :param num_of_days: int
#     :param num_of_weeks: int
#     :param DEVICE:
#     :param batch_size: int
#     :return:
#     three DataLoaders, each dataloader contains:
#     test_x_tensor: (B, N_nodes, in_feature, T_input)
#     test_decoder_input_tensor: (B, N_nodes, T_output)
#     test_target_tensor: (B, N_nodes, T_output)
#
#     '''
#     if use_nni:
#         filename = graph_signal_matrix_filename
#
#     else:
#         file = os.path.basename(graph_signal_matrix_filename).split('.')[0]
#
#         dirpath = os.path.dirname(graph_signal_matrix_filename)
#
#         filename = os.path.join(dirpath,
#                                 file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks)+ '.npz')
#
#         print('load file:', filename)
#
#     file_data = np.load(filename)
#     train_x = file_data['train_x']  # [4479, 80, 2, 12]
#     print('original file train_x:', train_x.shape)
#     train_target = file_data['train_target']  # [4479, 80, 2, 12]
#     train_timestamp = file_data['train_timestamp']  # [104479181, 1]
#
#     val_x = file_data['val_x']
#     print('original file val_x:', val_x.shape)
#     val_target = file_data['val_target']
#     val_timestamp = file_data['val_timestamp']
#
#     test_x = file_data['test_x']
#     print('original file test_x:', test_x.shape)
#     test_target = file_data['test_target']
#     test_timestamp = file_data['test_timestamp']
#
#     _max = file_data['mean']  # (1, 1, 2, 1)
#     _min = file_data['std']  # (1, 1, 2, 1)
#
#     # 统一对y进行归一化，变成[-1,1]之间的值
#     train_target_norm = max_min_normalization(train_target, _max, _min)
#     test_target_norm = max_min_normalization(test_target, _max, _min)
#     val_target_norm = max_min_normalization(val_target, _max, _min)
#
#     # filter 0-6点的数据
#     train_retain = train_timestamp % (24 * points_per_hour) > 6 * points_per_hour
#     train_retain_index = np.where(train_retain == True)
#     train_x = train_x[train_retain_index[0], :, :]
#     train_target_norm = train_target_norm[train_retain_index[0], :, :]
#     train_timestamp = train_timestamp[train_retain_index[0], :]
#
#     val_retain = val_timestamp % (24 * points_per_hour) > 6 * points_per_hour
#     val_retain_index = np.where(val_retain == True)
#     val_x = val_x[val_retain_index[0], :, :]
#     val_target_norm = val_target_norm[val_retain_index[0], :, :]
#     val_timestamp = val_timestamp[val_retain_index[0], :]
#
#     test_retain = test_timestamp % (24 * points_per_hour) > 6 * points_per_hour
#     test_retain_index = np.where(test_retain == True)
#     test_x = test_x[test_retain_index[0], :, :]
#     test_target_norm = test_target_norm[test_retain_index[0], :, :]
#     test_timestamp = test_timestamp[test_retain_index[0], :]
#
#     #  ------- train_loader -------
#     train_decoder_input_start = train_x[:, :, :, -1:]  # (B, N, F, 1(T)),最后已知traffic flow作为decoder 的初始输入
#     train_decoder_input = np.concatenate((train_decoder_input_start, train_target_norm[:, :, :, :-1]),
#                                          axis=-1)  # (B, N, F, T)
#
#     train_x_tensor = torch.from_numpy(train_x[:, :, prediction_channel:prediction_channel+1, :]).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
#     train_decoder_input_tensor = torch.from_numpy(train_decoder_input[:, :, prediction_channel:prediction_channel+1, :]).type(torch.FloatTensor).to(
#         DEVICE)  # (B, N, F, T)
#     train_target_tensor = torch.from_numpy(train_target_norm[:, :, prediction_channel:prediction_channel+1, :]).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
#
#     train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_decoder_input_tensor, train_target_tensor)
#
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
#
#     #  ------- val_loader -------
#     val_decoder_input_start = val_x[:, :, :, -1:]  # (B, N, F, 1(T)),最后已知traffic flow作为decoder 的初始输入
#     val_decoder_input = np.concatenate((val_decoder_input_start, val_target_norm[:, :, :, :-1]),
#                                        axis=-1)  # (B, N, F, T)
#
#     val_x_tensor = torch.from_numpy(val_x[:, :, prediction_channel:prediction_channel+1, :]).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
#     val_decoder_input_tensor = torch.from_numpy(val_decoder_input[:, :, prediction_channel:prediction_channel+1, :]).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
#     val_target_tensor = torch.from_numpy(val_target_norm[:, :, prediction_channel:prediction_channel+1, :]).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
#
#     val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_decoder_input_tensor, val_target_tensor)
#
#     val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
#
#     #  ------- test_loader -------
#     test_decoder_input_start = test_x[:, :, :, -1:]  # (B, N, F, 1(T)),最后已知traffic flow作为decoder 的初始输入
#     test_decoder_input = np.concatenate((test_decoder_input_start, test_target_norm[:, :, :, :-1]),
#                                         axis=-1)  # (B, N, F, T)
#
#     test_x_tensor = torch.from_numpy(test_x[:, :, prediction_channel:prediction_channel+1, :]).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
#     test_decoder_input_tensor = torch.from_numpy(test_decoder_input[:, :, prediction_channel:prediction_channel+1, :]).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
#     test_target_tensor = torch.from_numpy(test_target_norm[:, :, prediction_channel:prediction_channel+1, :]).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
#
#     test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_decoder_input_tensor, test_target_tensor)
#
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
#
#     # print
#     print('train:', train_x_tensor.size(), train_decoder_input_tensor.size(), train_target_tensor.size())
#     print('val:', val_x_tensor.size(), val_decoder_input_tensor.size(), val_target_tensor.size())
#     print('test:', test_x_tensor.size(), test_decoder_input_tensor.size(), test_target_tensor.size())
#     print('_max, _min:', _max.shape, _min.shape)
#
#     return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _max, _min
#
# def compute_val_loss_ende(net, val_loader, criterion, sw, epoch, limit=None):
#     '''
#     for encoderdecoder, compute mean loss on validation set
#     :param net: model
#     :param val_loader: torch.utils.data.utils.DataLoader
#     :param criterion: torch.nn.MSELoss
#     :param sw: tensorboardX.SummaryWriter
#     :param epoch: int, current epoch
#     :return: val_loss
#     '''
#
#     net.train(False)  # ensure dropout layers are in evaluation mode
#
#     val_loader_length = len(val_loader)  # nb of batch
#
#     tmp = []  # 记录了所有batch的loss
#
#     with torch.no_grad():
#
#         for batch_index, batch_data in enumerate(val_loader):
#             encoder_inputs, decoder_inputs, labels = batch_data
#             predict_length = labels.shape[1]
#             # encode
#             encoder_output, encoder_hidden = net.encode(encoder_inputs[:, :, 0:1])
#             # decode
#             decoder_start_inputs = decoder_inputs[:, :1, :]  # 只取输入的第一个值作为input，之后都用predict出来的值作为input
#             predict_output = decoder_start_inputs
#             decoder_hidden = None
#             predict_output_list = []
#             # 按着时间步进行预测
#             for step in range(predict_length):
#                 predict_output, decoder_hidden = net.decode(predict_output, encoder_hidden, decoder_hidden)
#                 predict_output_list.append(predict_output)
#
#             predict_outputs = torch.cat(predict_output_list, dim=1)
#             loss = criterion(predict_outputs, labels)  # 计算误差
#             tmp.append(loss.item())
#             if batch_index % 100 == 0:
#                 print('validation batch %s / %s, loss: %.2f' % (batch_index + 1, val_loader_length, loss.item()))
#
#             if (limit is not None) and batch_index >= limit:
#                 break
#
#         validation_loss = sum(tmp) / len(tmp)
#         sw.add_scalar('validation_loss', validation_loss, epoch)
#
#     return validation_loss
#
#
# def evaluate_on_test_ende(net, test_loader, test_target_tensor, sw, epoch,  _max, _min):
#     '''
#     for encoderdecoder, compute MAE, RMSE, MAPE scores of the prediction for every time step on testing set.
#
#     :param net: model
#     :param test_loader: torch.utils.data.utils.DataLoader
#     :param test_target_tensor: torch.tensor
#     :param sw:
#     :param epoch: int, current epoch
#     :param _max: (1, 1, 3(features), 1)
#     :param _min: (1, 1, 3(features), 1)
#     '''
#
#     net.train(False)  # ensure dropout layers are in test mode
#
#     with torch.no_grad():
#
#         test_loader_length = len(test_loader)
#
#         test_target_tensor = test_target_tensor.cpu().numpy()
#
#         print('test_target_tensor:', type(test_target_tensor))
#
#         prediction = []
#         for batch_index, batch_data in enumerate(test_loader):
#             encoder_inputs, decoder_inputs, labels = batch_data
#             predict_length = labels.shape[1]
#             # encode
#             encoder_output, encoder_hidden = net.encode(encoder_inputs[:, :, 0:1])
#             # decode
#             decoder_start_inputs = decoder_inputs[:, :1, :]
#             predict_output = decoder_start_inputs
#             decoder_hidden = None
#             predict_output_list = []
#
#             # 按着时间步进行预测
#             for step in range(predict_length):
#                 predict_output, decoder_hidden = net.decode(predict_output, encoder_hidden, decoder_hidden)
#                 predict_output_list.append(predict_output)
#
#             predict_outputs = torch.cat(predict_output_list, dim=1)
#             prediction.append(predict_outputs.detach().cpu().numpy())
#             if batch_index % 100 == 0:
#                 print('predicting testing set batch %s / %s' % (batch_index + 1, test_loader_length))
#
#         prediction = np.concatenate(prediction, 0)  # (batch, T', 1)
#         prediction_length = prediction.shape[1]
#         prediction = re_max_min_normalization(prediction, _max[0, 0, 0, 0], _min[0, 0, 0, 0])
#         test_target_tensor = re_max_min_normalization(test_target_tensor, _max[0, 0, 0, 0], _min[0, 0, 0, 0])
#
#         for i in range(prediction_length):
#             print('current epoch: %s, predict %s points' % (epoch, i))
#             mae = mean_absolute_error(test_target_tensor[:, i], prediction[:, i])
#             rmse = mean_squared_error(test_target_tensor[:, i], prediction[:, i]) ** 0.5
#             mape = masked_mape_np(test_target_tensor[:, i], prediction[:, i], 0)
#             print('MAE: %.2f' % (mae))
#             print('RMSE: %.2f' % (rmse))
#             print('MAPE: %.2f' % (mape))
#             print()
#             sw.add_scalar('MAE_%s_points' % (i), mae, epoch)
#             sw.add_scalar('RMSE_%s_points' % (i), rmse, epoch)
#             sw.add_scalar('MAPE_%s_points' % (i), mape, epoch)
#
#
# def predict_and_save_results_ende(net, data_loader, data_target_tensor, epoch, _max, _min, params_path, type):
#     '''
#
#     :param net: nn.Module
#     :param data_loader: torch.utils.data.utils.DataLoader
#     :param data_target_tensor: tensor
#     :param epoch: int
#     :param _max: (1, 1, 3, 1)
#     :param _min: (1, 1, 3, 1)
#     :param params_path: the path for saving the results
#     :return:
#     '''
#     net.train(False)  # ensure dropout layers are in test mode
#
#     with torch.no_grad():
#
#         data_target_tensor = data_target_tensor.cpu().numpy()
#         loader_length = len(data_loader)  # nb of batch
#
#         prediction = []  # 存储所有batch的output
#         input = []  # 存储所有batch的input
#
#         for batch_index, batch_data in enumerate(data_loader):
#             encoder_inputs, decoder_inputs, labels = batch_data
#             predict_length = labels.shape[1]
#             # encode
#             encoder_output, encoder_hidden = net.encode(encoder_inputs[:, :, 0:1])
#             input.append(encoder_inputs[:, :, 0:1])  # (batch, T', 1)
#             # decode
#             decoder_start_inputs = decoder_inputs[:, :1, :]
#             predict_output = decoder_start_inputs
#             decoder_hidden = None
#             predict_output_list = []
#
#             # 按着时间步进行预测
#             for step in range(predict_length):
#                 predict_output, decoder_hidden = net.decode(predict_output, encoder_hidden, decoder_hidden)
#                 predict_output_list.append(predict_output)
#
#             predict_outputs = torch.cat(predict_output_list, dim=1)
#             prediction.append(predict_outputs.detach().cpu().numpy())
#
#             if batch_index % 100 == 0:
#                 print('predicting testing set batch %s / %s' % (batch_index + 1, loader_length))
#
#         input = np.concatenate(input, 0)
#         input = re_max_min_normalization(input, _max[0, 0, 0, 0], _min[0, 0, 0, 0])
#
#         prediction = np.concatenate(prediction, 0)  # (batch, T', 1)
#         prediction = re_max_min_normalization(prediction, _max[0, 0, 0, 0], _min[0, 0, 0, 0])
#
#         data_target_tensor = re_max_min_normalization(data_target_tensor, _max[0, 0, 0, 0], _min[0, 0, 0, 0])
#
#         print('input:', input.shape)
#         print('prediction:', prediction.shape)
#         print('data_target_tensor:', data_target_tensor.shape)
#         output_filename = os.path.join(params_path, 'output_epoch_%s_%s' % (epoch, type))
#         np.savez(output_filename, input=input, prediction=prediction, data_target_tensor=data_target_tensor)
#
#         prediction_length = prediction.shape[1]
#
#         for i in range(prediction_length):
#             assert data_target_tensor.shape[0] == prediction.shape[0]
#             print('current epoch: %s, predict %s points' % (epoch, i))
#             mae = mean_absolute_error(data_target_tensor[:, i], prediction[:, i])
#             rmse = mean_squared_error(data_target_tensor[:, i], prediction[:, i]) ** 0.5
#             mape = masked_mape_np(data_target_tensor[:, i], prediction[:, i], 0)
#             print('MAE: %.2f' % (mae))
#             print('RMSE: %.2f' % (rmse))
#             print('MAPE: %.2f' % (mape))
#
#         # print overall results
#         mae = mean_absolute_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1))
#         rmse = mean_squared_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1)) ** 0.5
#         mape = masked_mape_np(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)
#         print('all MAE: %.2f' % (mae))
#         print('all RMSE: %.2f' % (rmse))
#         print('all MAPE: %.2f' % (mape))
#
#
# def compute_val_loss_trf(net, val_loader, criterion, sw, epoch, limit=None):
#     '''
#     for transformer, compute mean loss on validation set
#     :param net: model
#     :param val_loader: torch.utils.data.utils.DataLoader
#     :param criterion: torch.nn.MSELoss
#     :param sw: tensorboardX.SummaryWriter
#     :param epoch: int, current epoch
#     :return: val_loss
#     '''
#
#     net.train(False)  # ensure dropout layers are in evaluation mode
#     with torch.no_grad():
#
#         val_loader_length = len(val_loader)  # nb of batch
#
#         tmp = []  # 记录了所有batch的loss
#
#         for batch_index, batch_data in enumerate(val_loader):
#             encoder_inputs, decoder_inputs, labels = batch_data
#             predict_length = labels.shape[1]
#             # encode
#             encoder_output = net.encode(encoder_inputs[:, :, 0:1])
#             # decode
#             decoder_start_inputs = decoder_inputs[:, :1, :]  # 只取输入的第一个值作为input，之后都用predict出来的值作为input
#             decoder_input_list = [decoder_start_inputs]
#             # 按着时间步进行预测
#             for step in range(predict_length):
#                 decoder_inputs = torch.cat(decoder_input_list, dim=1)
#                 predict_output = net.decode(decoder_inputs, encoder_output)
#                 decoder_input_list = [decoder_start_inputs, predict_output]
#
#             loss = criterion(predict_output, labels)  # 计算误差
#             tmp.append(loss.item())
#             if batch_index % 100 == 0:
#                 print('validation batch %s / %s, loss: %.2f' % (batch_index + 1, val_loader_length, loss.item()))
#             if (limit is not None) and batch_index >= limit:
#                 break
#
#         validation_loss = sum(tmp) / len(tmp)
#         sw.add_scalar('validation_loss', validation_loss, epoch)
#
#     return validation_loss
#
#
# def evaluate_on_test_trf(net, test_loader, test_target_tensor, sw, epoch,  _max, _min):
#     '''
#     for transformer, compute MAE, RMSE, MAPE scores of the prediction for every time step on testing set.
#
#     :param net: model
#     :param test_loader: torch.utils.data.utils.DataLoader
#     :param test_target_tensor: torch.tensor
#     :param sw:
#     :param epoch: int, current epoch
#     :param _max: (1, 1, 3(features), 1)
#     :param _min: (1, 1, 3(features), 1)
#     '''
#
#     net.train(False)  # ensure dropout layers are in test mode
#
#     with torch.no_grad():
#
#         test_loader_length = len(test_loader)
#
#         test_target_tensor = test_target_tensor.cpu().numpy()
#
#         prediction = []
#         for batch_index, batch_data in enumerate(test_loader):
#             encoder_inputs, decoder_inputs, labels = batch_data
#             predict_length = labels.shape[1]
#             # encode
#             encoder_output = net.encode(encoder_inputs[:, :, 0:1])
#             # decode
#             decoder_start_inputs = decoder_inputs[:, :1, :]  # 只取输入的第一个值作为input，之后都用predict出来的值作为input
#             decoder_input_list = [decoder_start_inputs]
#             # 按着时间步进行预测
#             for step in range(predict_length):
#                 decoder_inputs = torch.cat(decoder_input_list, dim=1)
#                 predict_output = net.decode(decoder_inputs, encoder_output)
#                 decoder_input_list = [decoder_start_inputs, predict_output]
#
#             prediction.append(predict_output.detach().cpu().numpy())
#             if batch_index % 100 == 0:
#                 print('predicting testing set batch %s / %s' % (batch_index + 1, test_loader_length))
#
#         prediction = np.concatenate(prediction, 0)  # (batch, T', 1)
#         prediction_length = prediction.shape[1]
#         prediction = re_max_min_normalization(prediction, _max[0, 0, 0, 0], _min[0, 0, 0, 0])
#         test_target_tensor = re_max_min_normalization(test_target_tensor, _max[0, 0, 0, 0], _min[0, 0, 0, 0])
#
#         for i in range(prediction_length):
#             print('current epoch: %s, predict %s points' % (epoch, i))
#             mae = mean_absolute_error(test_target_tensor[:, i], prediction[:, i])
#             rmse = mean_squared_error(test_target_tensor[:, i], prediction[:, i]) ** 0.5
#             mape = masked_mape_np(test_target_tensor[:, i], prediction[:, i], 0)
#             print('MAE: %.2f' % (mae))
#             print('RMSE: %.2f' % (rmse))
#             print('MAPE: %.2f' % (mape))
#             print()
#             sw.add_scalar('MAE_%s_points' % (i), mae, epoch)
#             sw.add_scalar('RMSE_%s_points' % (i), rmse, epoch)
#             sw.add_scalar('MAPE_%s_points' % (i), mape, epoch)
#
#
# def predict_and_save_results_trf(net, data_loader, data_target_tensor, epoch, _max, _min, params_path, type):
#     '''
#
#     :param net: nn.Module
#     :param data_loader: torch.utils.data.utils.DataLoader
#     :param data_target_tensor: tensor
#     :param epoch: int
#     :param _max: (1, 1, 3, 1)
#     :param _min: (1, 1, 3, 1)
#     :param params_path: the path for saving the results
#     :return:
#     '''
#     net.train(False)  # ensure dropout layers are in test mode
#
#     with torch.no_grad():
#
#         data_target_tensor = data_target_tensor.cpu().numpy()
#
#         loader_length = len(data_loader)  # nb of batch
#
#         prediction = []
#
#         input = []  # 存储所有batch的input
#
#         start_time = time()
#
#         for batch_index, batch_data in enumerate(data_loader):
#             encoder_inputs, decoder_inputs, labels = batch_data
#             predict_length = labels.shape[1]
#             # encode
#             encoder_output = net.encode(encoder_inputs[:, :, 0:1])
#             input.append(encoder_inputs[:, :, 0:1])  # (batch, T', 1)
#             # decode
#             decoder_start_inputs = decoder_inputs[:, :1, :]  # 只取输入的第一个值作为input，之后都用predict出来的值作为input
#             decoder_input_list = [decoder_start_inputs]
#             # 按着时间步进行预测
#             for step in range(predict_length):
#                 decoder_inputs = torch.cat(decoder_input_list, dim=1)
#                 predict_output = net.decode(decoder_inputs, encoder_output)
#                 decoder_input_list = [decoder_start_inputs, predict_output]
#
#             prediction.append(predict_output.detach().cpu().numpy())
#             print('predicting testing set batch %s / %s, time: %.2fs' % (batch_index + 1, loader_length, time() - start_time))
#
#         input = np.concatenate(input, 0)
#         input = re_max_min_normalization(input, _max[0, 0, 0, 0], _min[0, 0, 0, 0])
#
#         prediction = np.concatenate(prediction, 0)  # (batch, T', 1)
#         prediction = re_max_min_normalization(prediction, _max[0, 0, 0, 0], _min[0, 0, 0, 0])
#         data_target_tensor = re_max_min_normalization(data_target_tensor, _max[0, 0, 0, 0], _min[0, 0, 0, 0])
#
#         print('input:', input.shape)
#         print('prediction:', prediction.shape)
#         print('data_target_tensor:', data_target_tensor.shape)
#         output_filename = os.path.join(params_path, 'output_epoch_%s_%s' % (epoch, type))
#         np.savez(output_filename, input=input, prediction=prediction, data_target_tensor=data_target_tensor)
#
#         # 计算误差
#
#         prediction_length = prediction.shape[1]
#
#         for i in range(prediction_length):
#             assert data_target_tensor.shape[0] == prediction.shape[0]
#             print('current epoch: %s, predict %s points' % (epoch, i))
#             mae = mean_absolute_error(data_target_tensor[:, i], prediction[:, i])
#             rmse = mean_squared_error(data_target_tensor[:, i], prediction[:, i]) ** 0.5
#             mape = masked_mape_np(data_target_tensor[:, i], prediction[:, i], 0)
#             print('MAE: %.2f' % (mae))
#             print('RMSE: %.2f' % (rmse))
#             print('MAPE: %.2f' % (mape))
#
#         # print overall results
#         mae = mean_absolute_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1))
#         rmse = mean_squared_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1)) ** 0.5
#         mape = masked_mape_np(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)
#         print('all MAE: %.2f' % (mae))
#         print('all RMSE: %.2f' % (rmse))
#         print('all MAPE: %.2f' % (mape))
#
#
# def evaluate_on_test_rnn(net, test_loader, test_target_tensor, sw, epoch, _max, _min):
#     '''
#     for rnn, compute MAE, RMSE, MAPE scores of the prediction for every time step on testing set.
#
#     :param net: model
#     :param test_loader: torch.utils.data.utils.DataLoader
#     :param test_target_tensor: torch.tensor (B*N_nodes, T_output, out_feature)=(B*N_nodes, T_output, 1)
#     :param sw:
#     :param epoch: int, current epoch
#     :param _max: (1, 1, 3(features), 1)
#     :param _min: (1, 1, 3(features), 1)
#     '''
#
#     net.train(False)  # ensure dropout layers are in test mode
#
#     with torch.no_grad():
#
#         test_loader_length = len(test_loader)
#
#         test_target_tensor = test_target_tensor.cpu().numpy()
#
#         prediction = []  # 存储所有batch的output
#         for batch_index, batch_data in enumerate(test_loader):
#
#             encoder_inputs, decoder_inputs, labels = batch_data
#
#             predict_length = labels.shape[1]
#
#             outputs = net(encoder_inputs[:, :, 0:1], predict_length)
#
#             predict_output = outputs[:, -predict_length:]
#
#             prediction.append(predict_output.detach().cpu().numpy())
#
#             if batch_index % 100 == 0:
#                 print('predicting testing set batch %s / %s' % (batch_index + 1, test_loader_length))
#
#         prediction = np.concatenate(prediction, 0)  # (batch, T', 1)
#         prediction = re_max_min_normalization(prediction, _max[0, 0, 0, 0], _min[0, 0, 0, 0])
#         test_target_tensor = re_max_min_normalization(test_target_tensor, _max[0, 0, 0, 0], _min[0, 0, 0, 0])
#         prediction_length = prediction.shape[1]
#
#         for i in range(prediction_length):
#             assert test_target_tensor.shape[0] == prediction.shape[0]
#             print('current epoch: %s, predict %s points' % (epoch, i))
#             mae = mean_absolute_error(test_target_tensor[:, i], prediction[:, i])
#             rmse = mean_squared_error(test_target_tensor[:, i], prediction[:, i]) ** 0.5
#             mape = masked_mape_np(test_target_tensor[:, i], prediction[:, i], 0)
#             print('MAE: %.2f' % (mae))
#             print('RMSE: %.2f' % (rmse))
#             print('MAPE: %.2f' % (mape))
#             print()
#             if sw:
#                 sw.add_scalar('MAE_%s_points' % (i), mae, epoch)
#                 sw.add_scalar('RMSE_%s_points' % (i), rmse, epoch)
#                 sw.add_scalar('MAPE_%s_points' % (i), mape, epoch)
#
#
# def predict_and_save_results_rnn(net, data_loader, data_target_tensor, global_step, _max, _min, params_path, type):
#     '''
#
#     :param net: nn.Module
#     :param data_loader: torch.utils.data.utils.DataLoader
#     :param data_target_tensor: tensor
#     :param epoch: int
#     :param _max: (1, 1, 3, 1)
#     :param _min: (1, 1, 3, 1)
#     :param params_path: the path for saving the results
#     :return:
#     '''
#     net.train(False)  # ensure dropout layers are in test mode
#
#     with torch.no_grad():
#
#         data_target_tensor = data_target_tensor.cpu().numpy()
#         loader_length = len(data_loader)  # nb of batch
#
#         prediction = []  # 存储所有batch的output
#         input = []  # 存储所有batch的input
#         for batch_index, batch_data in enumerate(data_loader):
#
#             encoder_inputs, decoder_inputs, labels = batch_data
#
#             predict_length = labels.shape[1]
#
#             input.append(encoder_inputs[:, :, 0:1])  # (batch, T', 1)
#
#             outputs = net(encoder_inputs[:, :, 0:1], predict_length)
#
#             predict_output = outputs[:, -predict_length:]
#
#             prediction.append(predict_output.detach().cpu().numpy())
#
#             if batch_index % 100 == 0:
#                 print('predicting data set batch %s / %s' % (batch_index + 1, loader_length))
#
#         input = np.concatenate(input, 0)
#         input = re_max_min_normalization(input, _max[0, 0, 0, 0], _min[0, 0, 0, 0])
#
#         prediction = np.concatenate(prediction, 0)  # (batch, T', 1)
#         prediction = re_max_min_normalization(prediction, _max[0, 0, 0, 0], _min[0, 0, 0, 0])
#
#         data_target_tensor = re_max_min_normalization(data_target_tensor, _max[0, 0, 0, 0], _min[0, 0, 0, 0])
#
#         print('input:', input.shape)
#         print('prediction:', prediction.shape)
#         print('data_target_tensor:', data_target_tensor.shape)
#         output_filename = os.path.join(params_path, 'output_epoch_%s_%s' % (global_step, type))
#         np.savez(output_filename, input=input, prediction=prediction, data_target_tensor=data_target_tensor)
#
#         for i in range(data_target_tensor.shape[1]):
#             assert data_target_tensor.shape[0] == prediction.shape[0]
#             print('current global: %s, predict %s points' % (global_step, i))
#             mae = mean_absolute_error(data_target_tensor[:, i], prediction[:, i])
#             rmse = mean_squared_error(data_target_tensor[:, i], prediction[:, i]) ** 0.5
#             mape = masked_mape_np(data_target_tensor[:, i], prediction[:, i], 0)
#             print('MAE: %.2f' % (mae))
#             print('RMSE: %.2f' % (rmse))
#             print('MAPE: %.2f' % (mape))
#             print()
#
#         # print overall results
#         mae = mean_absolute_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1))
#         rmse = mean_squared_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1)) ** 0.5
#         mape = masked_mape_np(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)
#         print('all MAE: %.2f' % (mae))
#         print('all RMSE: %.2f' % (rmse))
#         print('all MAPE: %.2f' % (mape))
#
#
# def evaluate_on_test_trfGCN(net, test_loader, test_target_tensor, sw, epoch,  _max, _min):
#     '''
#     for transformerGCN, compute MAE, RMSE, MAPE scores of the prediction for every time step on testing set.
#
#     :param net: model
#     :param test_loader: torch.utils.data.utils.DataLoader
#     :param test_target_tensor: torch.tensor
#     :param sw:
#     :param epoch: int, current epoch
#     :param _max: (1, 1, 3(features), 1)
#     :param _min: (1, 1, 3(features), 1)
#     '''
#
#     net.train(False)  # ensure dropout layers are in test mode
#
#     with torch.no_grad():
#
#         test_loader_length = len(test_loader)
#
#         test_target_tensor = test_target_tensor.cpu().numpy()
#
#         prediction = []
#
#         start_time = time()
#
#         for batch_index, batch_data in enumerate(test_loader):
#
#             encoder_inputs, decoder_inputs, labels = batch_data
#
#             encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)
#
#             decoder_inputs = decoder_inputs.unsqueeze(-1)  # (B, N, T, 1)
#
#             labels = labels.unsqueeze(-1)
#
#             predict_length = labels.shape[2]
#             # encode
#             encoder_output = net.encode(encoder_inputs)
#             # decode
#             decoder_start_inputs = decoder_inputs[:, :, :1, :]  # 只取输入的第一个值作为input，之后都用predict出来的值作为input
#             decoder_input_list = [decoder_start_inputs]
#             # 按着时间步进行预测
#             for step in range(predict_length):
#                 decoder_inputs = torch.cat(decoder_input_list, dim=2)
#                 predict_output = net.decode(decoder_inputs, encoder_output)  # predict_output:(B,N,T,1)
#                 decoder_input_list = [decoder_start_inputs, predict_output]
#
#             prediction.append(predict_output.detach().cpu().numpy())
#
#             if batch_index % 100 == 0:
#                 print('predicting testing set batch %s / %s' % (batch_index + 1, test_loader_length))
#
#         print('predicting cost time: %.4fs'% (time()-start_time))
#
#         prediction = np.concatenate(prediction, 0)  # (batch, N, T', 1)
#         prediction_length = prediction.shape[2]
#         prediction = re_max_min_normalization(prediction, _max[0, 0, 0, 0], _min[0, 0, 0, 0])
#         test_target_tensor = re_max_min_normalization(test_target_tensor, _max[0, 0, 0, 0], _min[0, 0, 0, 0])
#
#         for i in range(prediction_length):
#             print('current epoch: %s, predict %s points' % (epoch, i))
#             mae = mean_absolute_error(test_target_tensor[:, :, i], prediction[:, :, i, 0])
#             rmse = mean_squared_error(test_target_tensor[:, :, i], prediction[:, :, i, 0]) ** 0.5
#             mape = masked_mape_np(test_target_tensor[:, :, i], prediction[:, :, i, 0], 0)
#             print('MAE: %.2f' % (mae))
#             print('RMSE: %.2f' % (rmse))
#             print('MAPE: %.2f' % (mape))
#             print()
#             sw.add_scalar('MAE_%s_points' % (i), mae, epoch)
#             sw.add_scalar('RMSE_%s_points' % (i), rmse, epoch)
#             sw.add_scalar('MAPE_%s_points' % (i), mape, epoch)
#
#         print('current epoch: %s, predict all points' % epoch)
#         mae = mean_absolute_error(test_target_tensor.reshape(-1), prediction[:, :, :, 0].reshape(-1))
#         rmse = mean_squared_error(test_target_tensor.reshape(-1), prediction[:, :, :, 0].reshape(-1)) ** 0.5
#         mape = masked_mape_np(test_target_tensor.reshape(-1), prediction[:, :, :, 0].reshape(-1), 0)
#         print('MAE: %.2f' % (mae))
#         print('RMSE: %.2f' % (rmse))
#         print('MAPE: %.2f' % (mape))
#         print()
#         sw.add_scalar('MAE_all_points', mae, epoch)
#         sw.add_scalar('RMSE_all_points', rmse, epoch)
#         sw.add_scalar('MAPE_all_points', mape, epoch)
#
#
# def evaluate_on_test_trfGCN_Tem(net, test_loader, test_target_tensor, sw, epoch,  _max, _min):
#     '''
#     for transformerGCN, compute MAE, RMSE, MAPE scores of the prediction for every time step on testing set.
#
#     :param net: model
#     :param test_loader: torch.utils.data.utils.DataLoader
#     :param test_target_tensor: torch.tensor
#     :param sw:
#     :param epoch: int, current epoch
#     :param _max: (1, 1, 3(features), 1)
#     :param _min: (1, 1, 3(features), 1)
#     '''
#
#     net.train(False)  # ensure dropout layers are in test mode
#
#     with torch.no_grad():
#
#         test_loader_length = len(test_loader)
#
#         test_target_tensor = test_target_tensor.cpu().numpy()
#
#         prediction = []
#
#         start_time = time()
#
#         for batch_index, batch_data in enumerate(test_loader):
#
#             encoder_inputs, decoder_inputs, labels, en_week_index, en_time_index, de_week_index, de_time_index = batch_data
#
#             encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)
#
#             decoder_inputs = decoder_inputs.unsqueeze(-1)  # (B, N, T, 1)
#
#             labels = labels.unsqueeze(-1)
#
#             predict_length = labels.shape[2]
#             # encode
#             encoder_output = net.encode(encoder_inputs, en_week_index, en_time_index)
#             # decode
#             decoder_start_inputs = decoder_inputs[:, :, :1, :]  # 只取输入的第一个值作为input，之后都用predict出来的值作为input
#             decoder_input_list = [decoder_start_inputs]
#             # 按着时间步进行预测
#             for step in range(predict_length):
#                 decoder_inputs = torch.cat(decoder_input_list, dim=2)
#                 predict_output = net.decode(decoder_inputs, encoder_output, de_week_index, de_time_index)  # predict_output:(B,N,T,1)
#                 decoder_input_list = [decoder_start_inputs, predict_output]
#
#             prediction.append(predict_output.detach().cpu().numpy())
#
#             if batch_index % 100 == 0:
#                 print('predicting testing set batch %s / %s' % (batch_index + 1, test_loader_length))
#
#         print('predicting cost time: %.4fs'% (time()-start_time))
#
#         prediction = np.concatenate(prediction, 0)  # (batch, N, T', 1)
#         prediction_length = prediction.shape[2]
#         prediction = re_max_min_normalization(prediction, _max[0, 0, 0, 0], _min[0, 0, 0, 0])
#         test_target_tensor = re_max_min_normalization(test_target_tensor, _max[0, 0, 0, 0], _min[0, 0, 0, 0])
#
#         for i in range(prediction_length):
#             print('current epoch: %s, predict %s points' % (epoch, i))
#             mae = mean_absolute_error(test_target_tensor[:, :, i], prediction[:, :, i, 0])
#             rmse = mean_squared_error(test_target_tensor[:, :, i], prediction[:, :, i, 0]) ** 0.5
#             mape = masked_mape_np(test_target_tensor[:, :, i], prediction[:, :, i, 0], 0)
#             print('MAE: %.2f' % (mae))
#             print('RMSE: %.2f' % (rmse))
#             print('MAPE: %.2f' % (mape))
#             print()
#             sw.add_scalar('MAE_%s_points' % (i), mae, epoch)
#             sw.add_scalar('RMSE_%s_points' % (i), rmse, epoch)
#             sw.add_scalar('MAPE_%s_points' % (i), mape, epoch)
#
#         print('current epoch: %s, predict all points' % epoch)
#         mae = mean_absolute_error(test_target_tensor.reshape(-1), prediction[:, :, :, 0].reshape(-1))
#         rmse = mean_squared_error(test_target_tensor.reshape(-1), prediction[:, :, :, 0].reshape(-1)) ** 0.5
#         mape = masked_mape_np(test_target_tensor.reshape(-1), prediction[:, :, :, 0].reshape(-1), 0)
#         print('MAE: %.2f' % (mae))
#         print('RMSE: %.2f' % (rmse))
#         print('MAPE: %.2f' % (mape))
#         print()
#         sw.add_scalar('MAE_all_points', mae, epoch)
#         sw.add_scalar('RMSE_all_points', rmse, epoch)
#         sw.add_scalar('MAPE_all_points', mape, epoch)
#
#
# #######
#
#
# def compute_val_loss_trfGCN_metro(net, val_loader, criterion, sw, epoch):
#     '''
#     for transformerGCN, compute mean loss on validation set
#     :param net: model
#     :param val_loader: torch.utils.data.utils.DataLoader
#     :param criterion: torch.nn.MSELoss
#     :param sw: tensorboardX.SummaryWriter
#     :param epoch: int, current epoch
#     :return: val_loss
#     '''
#
#     net.train(False)  # ensure dropout layers are in evaluation mode
#
#     with torch.no_grad():
#
#         val_loader_length = len(val_loader)  # nb of batch
#
#         tmp = []  # 记录了所有batch的loss
#
#         start_time = time()
#
#         for batch_index, batch_data in enumerate(val_loader):
#
#             encoder_inputs, decoder_inputs, labels = batch_data
#
#             encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)
#
#             decoder_inputs = decoder_inputs.transpose(-1, -2)  # (B, N, T, F)
#
#             labels = labels.transpose(-1, -2)  # (B, N, T, F)
#
#             predict_length = labels.shape[2]  # T
#             # encode
#             encoder_output = net.encode(encoder_inputs)
#             # decode
#             decoder_start_inputs = decoder_inputs[:, :, :1, :]  # 只取输入的第一个值作为input，之后都用predict出来的值作为input
#             decoder_input_list = [decoder_start_inputs]
#             # 按着时间步进行预测
#             for step in range(predict_length):
#                 decoder_inputs = torch.cat(decoder_input_list, dim=2)
#                 predict_output = net.decode(decoder_inputs, encoder_output)
#                 decoder_input_list = [decoder_start_inputs, predict_output]
#
#             loss = criterion(predict_output, labels)  # 计算误差
#             tmp.append(loss.item())
#             if batch_index % 100 == 0:
#                 print('validation batch %s / %s, loss: %.2f' % (batch_index + 1, val_loader_length, loss.item()))
#
#         print('validation cost time: %.4fs' %(time()-start_time))
#
#         validation_loss = sum(tmp) / len(tmp)
#         sw.add_scalar('validation_loss', validation_loss, epoch)
#
#     return validation_loss
#
#
# def compute_val_loss_trfGCN_metro_LR(net, val_loader, criterion, sw, epoch, Laplacian_matrix, LR_rate):
#     '''
#     for transformerGCN, compute mean loss on validation set
#     :param net: model
#     :param val_loader: torch.utils.data.utils.DataLoader
#     :param criterion: torch.nn.MSELoss
#     :param sw: tensorboardX.SummaryWriter
#     :param epoch: int, current epoch
#     :return: val_loss
#     '''
#
#     net.train(False)  # ensure dropout layers are in evaluation mode
#
#     with torch.no_grad():
#
#         val_loader_length = len(val_loader)  # nb of batch
#
#         tmp = []  # 记录了所有batch的loss
#
#         start_time = time()
#
#         for batch_index, batch_data in enumerate(val_loader):
#
#             encoder_inputs, decoder_inputs, labels = batch_data
#
#             encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)
#
#             decoder_inputs = decoder_inputs.transpose(-1, -2)  # (B, N, T, F)
#
#             labels = labels.transpose(-1, -2)  # (B, N, T, F)
#
#             predict_length = labels.shape[2]  # T
#             # encode
#             encoder_output = net.encode(encoder_inputs)
#             # decode
#             decoder_start_inputs = decoder_inputs[:, :, :1, :]  # 只取输入的第一个值作为input，之后都用predict出来的值作为input
#             decoder_input_list = [decoder_start_inputs]
#             # 按着时间步进行预测
#             for step in range(predict_length):
#                 decoder_inputs = torch.cat(decoder_input_list, dim=2)
#                 predict_output = net.decode(decoder_inputs, encoder_output)
#                 decoder_input_list = [decoder_start_inputs, predict_output]
#
#             error = criterion(predict_output, labels)  # 计算误差
#
#             SE_matrix_src = net.state_dict()['src_embed.2.embedding.weight']
#             SE_matrix_trg = net.state_dict()['trg_embed.2.embedding.weight']
#
#             reg_loss_SE_src = torch.trace(
#                 torch.matmul(torch.matmul(torch.transpose(SE_matrix_src, 0, 1), Laplacian_matrix), SE_matrix_src))
#             reg_loss_SE_trg = torch.trace(
#                 torch.matmul(torch.matmul(torch.transpose(SE_matrix_trg, 0, 1), Laplacian_matrix), SE_matrix_trg))
#
#             loss = error + LR_rate * reg_loss_SE_src + LR_rate * reg_loss_SE_trg  # 计算误差
#
#             tmp.append(loss.item())
#
#             if batch_index % 100 == 0:
#                 print('validation batch %s / %s, loss: %.2f' % (batch_index + 1, val_loader_length, loss.item()))
#
#         print('validation cost time: %.4fs' %(time()-start_time))
#
#         validation_loss = sum(tmp) / len(tmp)
#         sw.add_scalar('validation_loss', validation_loss, epoch)
#
#     return validation_loss
#
#
#
# def predict_and_save_results_trfGCN_metro(net, data_loader, data_target_tensor, epoch, _max, _min, params_path, type, prediction_channel):
#     '''
#     for transformerGCN
#     :param net: nn.Module
#     :param data_loader: torch.utils.data.utils.DataLoader
#     :param data_target_tensor: tensor
#     :param epoch: int
#     :param _max: (1, 1, 3, 1)
#     :param _min: (1, 1, 3, 1)
#     :param params_path: the path for saving the results
#     :return:
#     '''
#     net.train(False)  # ensure dropout layers are in test mode
#
#     with torch.no_grad():
#
#         data_target_tensor = data_target_tensor.cpu().numpy()
#
#         loader_length = len(data_loader)  # nb of batch
#
#         prediction = []
#
#         input = []  # 存储所有batch的input
#
#         start_time = time()
#
#         for batch_index, batch_data in enumerate(data_loader):
#
#             encoder_inputs, decoder_inputs, labels = batch_data
#
#             encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)
#
#             decoder_inputs = decoder_inputs.transpose(-1, -2)  # (B, N, T, F)
#
#             labels = labels.transpose(-1, -2)  # (B, N, T, F)
#
#             predict_length = labels.shape[2]  # T
#
#             # encode
#             encoder_output = net.encode(encoder_inputs)
#             input.append(encoder_inputs.cpu().numpy())  # (batch, T', 1)
#
#             # decode
#             decoder_start_inputs = decoder_inputs[:, :, :1, :]  # 只取输入的第一个值作为input，之后都用predict出来的值作为input
#             decoder_input_list = [decoder_start_inputs]
#
#             # 按着时间步进行预测
#             for step in range(predict_length):
#                 decoder_inputs = torch.cat(decoder_input_list, dim=2)
#                 predict_output = net.decode(decoder_inputs, encoder_output)
#                 decoder_input_list = [decoder_start_inputs, predict_output]
#
#             prediction.append(predict_output.detach().cpu().numpy())
#             if batch_index % 10 == 0:
#                 print('predicting testing set batch %s / %s, time: %.2fs' % (batch_index + 1, loader_length, time() - start_time))
#
#     _max = _max.transpose((0, 1, 3, 2))[:, :, :, prediction_channel:prediction_channel+1]
#     _min = _min.transpose((0, 1, 3, 2))[:, :, :, prediction_channel:prediction_channel+1]
#     print('_max:', _max.shape)
#
#     input = np.concatenate(input, 0)
#     input = re_max_min_normalization(input, _max, _min)
#
#     prediction = np.concatenate(prediction, 0)  # (batch, N, T', 1)
#     print('prediction:', prediction.shape)
#
#     prediction = re_max_min_normalization(prediction, _max, _min)
#     data_target_tensor = data_target_tensor.transpose((0, 1, 3, 2))  # (batch, N, T', F)
#     data_target_tensor = re_max_min_normalization(data_target_tensor, _max, _min)
#
#     print('input:', input.shape)
#     print('prediction:', prediction.shape)
#     print('data_target_tensor:', data_target_tensor.shape)
#     output_filename = os.path.join(params_path, 'output_epoch_%s_%s' % (epoch, type))
#     np.savez(output_filename, input=input, prediction=prediction, data_target_tensor=data_target_tensor)
#
#     # 计算误差
#     excel_list = []
#
#     prediction_length = prediction.shape[2]
#
#     for i in range(prediction_length):
#         print('current epoch: %s, predict %s points' % (epoch, i))
#         mae = mean_absolute_error(data_target_tensor[:, :, i, 0], prediction[:, :, i, 0])
#         rmse = mean_squared_error(data_target_tensor[:, :, i, 0], prediction[:, :, i, 0]) ** 0.5
#         mape = masked_mape_np(data_target_tensor[:, :, i, 0], prediction[:, :, i, 0], 0)
#         print('MAE: %.2f' % (mae))
#         print('RMSE: %.2f' % (rmse))
#         print('MAPE: %.2f' % (mape))
#         print()
#         excel_list.extend([mae, rmse, mape])
#
#     # print overall results
#     mae = mean_absolute_error(data_target_tensor[:, :, :, 0].reshape(-1), prediction[:, :, :, 0].reshape(-1))
#     rmse = mean_squared_error(data_target_tensor[:, :, :, 0].reshape(-1), prediction[:, :, :, 0].reshape(-1)) ** 0.5
#     mape = masked_mape_np(data_target_tensor[:, :, :, 0].reshape(-1), prediction[:, :, :, 0].reshape(-1), 0)
#     print('all MAE: %.2f' % (mae))
#     print('all RMSE: %.2f' % (rmse))
#     print('all MAPE: %.2f' % (mape))
#     excel_list.extend([mae, rmse, mape])
#     print(excel_list)
#
# def evaluate_on_test_trfGCN_metro(net, test_loader, test_target_tensor, sw, epoch,  _max, _min, prediction_channel):
#     '''
#     for transformerGCN, compute MAE, RMSE, MAPE scores of the prediction for every time step on testing set.
#
#     :param net: model
#     :param test_loader: torch.utils.data.utils.DataLoader
#     :param test_target_tensor: torch.tensor
#     :param sw:
#     :param epoch: int, current epoch
#     :param _max: (1, 1, 3(features), 1)
#     :param _min: (1, 1, 3(features), 1)
#     '''
#
#     net.train(False)  # ensure dropout layers are in test mode
#
#     with torch.no_grad():
#
#         test_loader_length = len(test_loader)
#
#         test_target_tensor = test_target_tensor.cpu().numpy()
#
#         prediction = []
#
#         start_time = time()
#
#         for batch_index, batch_data in enumerate(test_loader):
#
#             encoder_inputs, decoder_inputs, labels = batch_data
#
#             encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)
#
#             decoder_inputs = decoder_inputs.transpose(-1, -2)  # (B, N, T, F)
#
#             labels = labels.transpose(-1, -2)  # (B, N, T, F)
#
#             predict_length = labels.shape[2]
#             # encode
#             encoder_output = net.encode(encoder_inputs)
#             # decode
#             decoder_start_inputs = decoder_inputs[:, :, :1, :]  # 只取输入的第一个值作为input，之后都用predict出来的值作为input
#             decoder_input_list = [decoder_start_inputs]
#             # 按着时间步进行预测
#             for step in range(predict_length):
#                 decoder_inputs = torch.cat(decoder_input_list, dim=2)
#                 predict_output = net.decode(decoder_inputs, encoder_output)  # predict_output:(B,N,T,1)
#                 decoder_input_list = [decoder_start_inputs, predict_output]
#
#             prediction.append(predict_output.detach().cpu().numpy())
#
#             if batch_index % 100 == 0:
#                 print('predicting testing set batch %s / %s' % (batch_index + 1, test_loader_length))
#
#         print('predicting cost time: %.4fs'% (time()-start_time))
#         _max = _max.transpose((0, 1, 3, 2))[:, :, :, prediction_channel:prediction_channel+1]
#         _min = _min.transpose((0, 1, 3, 2))[:, :, :, prediction_channel:prediction_channel+1]
#         print('_max:', _max.shape)
#
#         prediction = np.concatenate(prediction, 0)  # (batch, N, T', 1)
#         prediction_length = prediction.shape[2]
#         print('prediction:', prediction.shape)
#
#         prediction = re_max_min_normalization(prediction, _max, _min)
#         test_target_tensor = test_target_tensor.transpose((0, 1, 3, 2))  # (batch, N, T', F)
#         test_target_tensor = re_max_min_normalization(test_target_tensor, _max, _min)
#
#         for i in range(prediction_length):
#             print('current epoch: %s, predict %s points' % (epoch, i))
#             mae = mean_absolute_error(test_target_tensor[:, :, i, 0], prediction[:, :, i, 0])
#             rmse = mean_squared_error(test_target_tensor[:, :, i, 0], prediction[:, :, i, 0]) ** 0.5
#             mape = masked_mape_np(test_target_tensor[:, :, i, 0], prediction[:, :, i, 0], 0)
#             print('MAE: %.2f' % (mae))
#             print('RMSE: %.2f' % (rmse))
#             print('MAPE: %.2f' % (mape))
#             print()
#             sw.add_scalar('MAE_%s_points' % (i), mae, epoch)
#             sw.add_scalar('RMSE_%s_points' % (i), rmse, epoch)
#             sw.add_scalar('MAPE_%s_points' % (i), mape, epoch)
#
#         print('current epoch: %s, predict all points' % epoch)
#         mae = mean_absolute_error(test_target_tensor.reshape(-1), prediction[:, :, :, 0].reshape(-1))
#         rmse = mean_squared_error(test_target_tensor.reshape(-1), prediction[:, :, :, 0].reshape(-1)) ** 0.5
#         mape = masked_mape_np(test_target_tensor.reshape(-1), prediction[:, :, :, 0].reshape(-1), 0)
#         print('MAE: %.2f' % (mae))
#         print('RMSE: %.2f' % (rmse))
#         print('MAPE: %.2f' % (mape))
#         print()
#         sw.add_scalar('MAE_all_points', mae, epoch)
#         sw.add_scalar('RMSE_all_points', rmse, epoch)
#         sw.add_scalar('MAPE_all_points', mape, epoch)
