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
from utils.split import protocol_split
from utils.data_loader import (
    AMAZON_RELATIONS, YELPCHI_RELATIONS, is_comp_name, is_elliptic_name,
    load_comp_pyg, load_elliptic_pyg, load_tgraph_pyg,
)


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


def load_mat_full(prefix='./example_data/', file_name='YelpChi.mat', relations=None):
    data = sio.loadmat(os.path.join(prefix, file_name))
    if relations is None:
        relations = _default_relations(file_name, data)
    truelabels, features = data['label'], data['features'].astype(float)
    truelabels = np.array(truelabels).reshape(-1)
    if sp.issparse(features):
        features = features.todense()
    features = np.asarray(features)
    adj_mat = [data[relation] for relation in relations if relation in data]
    return adj_mat, features, truelabels


def load_data(prefix='../../dataset/', file_name='YelpChi.mat', relations=None,
              normalize=True, train_ratio=0.4, seed=48, test_size=None):
    del test_size
    if _is_tgraph(file_name):
        x, edge_indices, y = load_tgraph_pyg(
            file_name, prefix=prefix, normalize=normalize, split=False)
    elif is_comp_name(file_name):
        x, edge_indices, y = load_comp_pyg(
            file_name, prefix=prefix, normalize=normalize, split=False)
    elif is_elliptic_name(file_name):
        x, edge_indices, y = load_elliptic_pyg(
            prefix=prefix, normalize=normalize, split=False)
    else:
        adjs, feats, truelabels = load_mat_full(prefix, file_name, relations)
        if normalize:
            feats = StandardScaler().fit_transform(feats)
        edge_indices = []
        for adj in adjs:
            if not sp.issparse(adj):
                adj = sp.coo_matrix(adj)
            edge_index, _ = from_scipy_sparse_matrix(adj)
            edge_indices.append(edge_index)
        x = torch.tensor(np.asarray(feats), dtype=torch.float32)
        y = torch.tensor(truelabels, dtype=torch.long)

    labels_np = y.detach().cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
    train_idx, valid_idx, test_idx, discarded_idx, split_info = protocol_split(
        labels_np, train_ratio=train_ratio, seed=seed)
    train_idx_t = torch.tensor(np.asarray(train_idx), dtype=torch.long)
    valid_idx_t = torch.tensor(np.asarray(valid_idx), dtype=torch.long)
    test_idx_t = torch.tensor(np.asarray(test_idx), dtype=torch.long)
    split_info['discarded_idx'] = discarded_idx
    return x, edge_indices, y, train_idx_t, valid_idx_t, test_idx_t, split_info
