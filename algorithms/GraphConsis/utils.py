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
from utils.data_loader import AMAZON_RELATIONS, YELPCHI_RELATIONS, load_tgraph_pyg


def _is_tgraph(file_name):
    name = os.path.basename(str(file_name)).lower().replace('.zip', '').replace('.bin', '')
    return name in ('tfinance', 'tsocial')


def _default_relations(file_name, data):
    name = os.path.basename(file_name).lower()
    if 'amazon' in name:
        candidates = list(AMAZON_RELATIONS)
    elif 'yelp' in name:
        candidates = list(YELPCHI_RELATIONS)
    else:
        candidates = [k for k in data.keys() if isinstance(k, str) and k.startswith('net_')]
    rels = [r for r in candidates if r in data]
    if not rels:
        raise KeyError('No relation matrices found in {}. Keys: {}'.format(
            file_name, [k for k in data.keys() if not str(k).startswith('__')]))
    return rels


def load_mat_full(prefix='./example_data/', file_name='YelpChi.mat',
                  relations=None, test_size=0.2, seed=48):
    data = sio.loadmat(os.path.join(prefix, file_name))
    if relations is None:
        relations = _default_relations(file_name, data)
    truelabels, features = data['label'], data['features'].astype(float)
    truelabels = np.array(truelabels).reshape(-1)
    if sp.issparse(features):
        features = features.todense()
    features = np.asarray(features)
    adj_mat = [data[relation] for relation in relations if relation in data]
    indices = np.arange(len(truelabels))
    train_idx, test_idx = stratified_train_test_split(
        truelabels, test_size=test_size, seed=seed, indices=indices)
    return adj_mat, features, truelabels, train_idx, test_idx


def load_data(prefix='../../dataset/', file_name='YelpChi.mat', relations=None,
              normalize=True, test_size=0.2, seed=48):
    if _is_tgraph(file_name):
        return load_tgraph_pyg(
            file_name, prefix=prefix, test_size=test_size, seed=seed, normalize=normalize)

    adjs, feats, truelabels, train_idx, test_idx = load_mat_full(
        prefix, file_name, relations, test_size, seed)
    if normalize:
        scaler = StandardScaler()
        feats = scaler.fit_transform(feats)
    edge_indices = []
    for adj in adjs:
        if not sp.issparse(adj):
            adj = sp.coo_matrix(adj)
        edge_index, _ = from_scipy_sparse_matrix(adj)
        edge_indices.append(edge_index)
    x = torch.tensor(np.asarray(feats), dtype=torch.float32)
    y = torch.tensor(truelabels, dtype=torch.long)
    train_idx_t = torch.tensor(np.asarray(train_idx), dtype=torch.long)
    test_idx_t = torch.tensor(np.asarray(test_idx), dtype=torch.long)
    split_info = split_report(truelabels, train_idx, test_idx)
    split_info.update({'test_fraction': float(test_size), 'seed': int(seed), 'stratified': True})
    return x, edge_indices, y, train_idx_t, test_idx_t, split_info
