from __future__ import print_function

import os
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import from_scipy_sparse_matrix

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.split import split_report, stratified_train_test_split


def load_data_dblp(prefix='./example_data/', file_name='YelpChi.mat',
                   test_size=0.2, seed=48):
    data = sio.loadmat(os.path.join(prefix, file_name))
    truelabels, features = data['label'], data['features'].astype(float)
    truelabels = np.array(truelabels).reshape(-1)
    if sp.issparse(features):
        features = features.todense()
    features = np.asarray(features)
    rownetworks = [data['net_rur']]
    indices = np.arange(len(truelabels))
    train_idx, test_idx = stratified_train_test_split(
        truelabels, test_size=test_size, seed=seed, indices=indices)
    return rownetworks, features, truelabels, train_idx, test_idx


def load_mat_graph(prefix='../../dataset/', file_name='YelpChi.mat', normalize=True,
                   test_size=0.2, seed=48):
    adjs, feats, truelabels, train_idx, test_idx = load_data_dblp(
        prefix, file_name, test_size=test_size, seed=seed)
    if normalize:
        scaler = StandardScaler()
        feats = scaler.fit_transform(feats)
    adj = adjs[0]
    if not sp.issparse(adj):
        adj = sp.coo_matrix(adj)
    edge_index, _ = from_scipy_sparse_matrix(adj)
    x = torch.tensor(np.asarray(feats), dtype=torch.float32)
    y = torch.tensor(truelabels, dtype=torch.long)
    train_idx = torch.tensor(np.asarray(train_idx), dtype=torch.long)
    test_idx = torch.tensor(np.asarray(test_idx), dtype=torch.long)
    split_info = split_report(truelabels, train_idx.numpy(), test_idx.numpy())
    split_info.update({'test_fraction': float(test_size), 'seed': int(seed), 'stratified': True})
    return x, edge_index, y, train_idx, test_idx, split_info


def load_data(prefix='./example_data/', file_name='YelpChi.mat', normalize=True, load_walks=False,
              test_size=0.2, seed=48):
    return load_mat_graph(prefix, file_name, normalize, test_size, seed)
