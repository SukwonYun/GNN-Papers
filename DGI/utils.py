import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
from pathlib import Path
import pickle
import os
#os.chdir(os.getcwd() +'/GNN/DGI')
os.environ['CUDA_VISIBLE_DEVICES']='2'

def load_graph(data_name='./data/ind.cora', weight=False):
    with open(data_name + '.graph', 'rb') as f:
        data = pickle.load(f)
        print(data)
        graph = nx.Graph(data) #generate graph as object

    if not weight:
        for edge in graph.edges():
            graph[edge[0]][edge[1]]['weight'] = 1 #assign same weight between nodes

    else:
        for edge in graph.edges():
            graph[edge[0]][edge[1]]['weight'] = np.random.randint(0, 100) #assign random weight(0-100) between node

    return graph

def sparse_mx_to_torch_sparse(sparse_mx):
    """
    :param sparse_mx: sparse matrix
    :return: dense matrix
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32) # making COO format, by tocoo(), we can use .row, .col
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.long)) # locations wherer data locates
    value = torch.from_numpy(sparse_mx.data)

    return torch.sparse.FloatTensor(indices, value, torch.Size(sparse_mx.shape)).cuda() #Tensors in COO format

def preprocess_adj(adj_mx, sparse=False):
    """"
    :param adj_mx: Adjacency Matrix (node x node)
    :param sparse: Checking sparsity
    :return: A_hat
    """
    # Making A_hat
    I = np.eye(adj_mx.shape[0])
    A_temp = adj_mx + I
    D_temp = np.sum(A_temp, axis=1) #Degree Matrix
    D_inv_sqrt = np.power(D_temp, -0.5)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0 #1/0 -> 0
    D_inv_sqrt = np.diag(D_inv_sqrt) #Making diagonal matrix
    A_hat = np.dot(np.dot(D_inv_sqrt, A_temp), D_inv_sqrt)

    return sparse_mx_to_torch_sparse(sp.coo_matrix(A_hat)) if sparse else torch.from_numpy(A_hat).cuda()

def load_data(data_name = './data/ind.cora'):
    """
    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    objects = []
    datasets = ['x', 'y', 'allx', 'ally', 'tx', 'ty', 'graph'] #those datasets have sparse type

    for index,s in enumerate(datasets):
        datasets[index] = data_name + '.' + s

    for s in datasets:
        objects.append(pickle.load(open(Path(s), 'rb'), encoding = 'latin1'))

    x, y, allx, ally, tx, ty, graph = objects
    x, allx, tx = x.toarray(), allx.toarray(), tx.toarray()

    test_index = []
    with open(data_name + '.test.index', 'rb') as file:
        line = file.readline()
        while len(line) > 0:
            test_index.append(int(line[0:-1])) #extract only index number
            line = file.readline()

    max_idx, min_idx = max(test_index), min(test_index)

    #Combine train and test data
    tx_extended = np.zeros((max_idx - min_idx + 1, tx.shape[1]))
    feature_cat = np.vstack([allx, tx_extended]).astype(np.float)
    feature_cat[test_index] = tx

    ty_extended = np.zeros((max_idx - min_idx + 1, ty.shape[1]))
    label_cat = np.vstack([ally, ty_extended])
    label_cat[test_index] = ty

    # Graph into matrix format
    adj_mx = nx.adjacency_matrix(nx.convert.from_dict_of_lists(graph)).toarray() # graph format: {0: [633, 1862, 2582],...}

    # Setting index in train, valid, test set
    train_idx = range(len(y))
    valid_idx = range(len(y), len(y)+500)
    test_idx = test_index

    # Set 1 where index exists
    train_msk = sample_mask(train_idx, label_cat.shape[0])
    valid_msk = sample_mask(valid_idx, label_cat.shape[0])
    test_msk = sample_mask(test_idx, label_cat.shape[0])
    zero = np.zeros(label_cat.shape)

    train_label = zero.copy()
    valid_label = zero.copy()
    test_label = zero.copy()

    train_label[train_msk, :] = label_cat[train_msk, :]
    valid_label[valid_msk, :] = label_cat[valid_msk, :]
    test_label[test_msk, :] = label_cat[test_msk, :]

    feature_cat = normalize(feature_cat)

    feature_cat = torch.from_numpy(feature_cat).cuda()
    label_cat = torch.from_numpy(label_cat).cuda()

    train_msk = torch.from_numpy(train_msk).cuda()
    valid_msk = torch.from_numpy(valid_msk).cuda()
    test_msk = torch.from_numpy(test_msk).cuda()

    return adj_mx, feature_cat, label_cat, train_label, valid_label, test_label, train_msk, valid_msk, test_msk


def sample_mask(idx, length):
    """
    Create mask
    """
    msk = np.zeros(length)
    msk[idx] = 1
    return np.array(msk, dtype=np.bool)


def normalize(feature):
    """
    Row-normalize sparse matrix
    """
    rowsum = np.sum(feature, axis=1)
    rowsum_diag = np.diag(rowsum)
    rowsum_inv = np.power(rowsum_diag, -1)
    rowsum_inv[np.isinf(rowsum_inv)] = 0.0

    return np.dot(rowsum_inv, feature)