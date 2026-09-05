'''
DGA-GNN (Duan et al., AAAI 2024): Dynamic Grouping Aggregation GNN.

Port of reference/DGA-GNN-main using scatter mean aggregation instead of DGL.
Decision-tree binning uses sklearn (official code used toad).
'''
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.tree import DecisionTreeClassifier


def decision_tree_bin_encoding(features, labels, train_idx, n_bins=10):
    """Concat original features with one-hot DT bins fit on the train split.

    Matches official ``bin_encoding2`` (toad Combiner, method='dt', min_samples=0.01).
    """
    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels).reshape(-1)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    n, n_feat = X.shape
    min_leaf = max(1, int(0.01 * max(len(train_idx), 1)))
    parts = [X.astype(np.float32)]
    for j in range(n_feat):
        col = X[:, j:j + 1]
        tree = DecisionTreeClassifier(
            max_leaf_nodes=max(2, int(n_bins)),
            min_samples_leaf=min_leaf,
            random_state=0,
        )
        try:
            tree.fit(col[train_idx], y[train_idx])
            leaves = tree.apply(col)
        except ValueError:
            leaves = np.zeros(n, dtype=np.int64)
        uniq, codes = np.unique(leaves, return_inverse=True)
        oh = np.zeros((n, int(uniq.size)), dtype=np.float32)
        oh[np.arange(n), codes] = 1.0
        parts.append(oh)
    return np.ascontiguousarray(np.concatenate(parts, axis=1), dtype=np.float32)


def _scatter_mean(src_feat, dst, n, edge_weight=None):
    """Mean of source features onto destination nodes (DGL copy_u + mean / weighted)."""
    dim = src_feat.size(1)
    out = torch.zeros(n, dim, device=src_feat.device, dtype=src_feat.dtype)
    if src_feat.size(0) == 0:
        return out
    if edge_weight is None:
        idx = dst.unsqueeze(-1).expand_as(src_feat)
        out.scatter_add_(0, idx, src_feat)
        deg = torch.bincount(dst, minlength=n).to(dtype=src_feat.dtype).clamp(min=1).unsqueeze(-1)
        return out / deg
    w = edge_weight.reshape(-1, 1).to(dtype=src_feat.dtype)
    idx = dst.unsqueeze(-1).expand_as(src_feat)
    out.scatter_add_(0, idx, src_feat * w)
    wsum = torch.zeros(n, 1, device=src_feat.device, dtype=src_feat.dtype)
    wsum.scatter_add_(0, dst.unsqueeze(-1), w)
    return out / (wsum + 1e-8)


class IntraConv(nn.Module):
    """GraphSAGE-style mean aggregator (official IntraConv_single / IntraConv_multi)."""

    def __init__(self, in_feats, out_feats, add_self=True, feat_drop=0.0,
                 norm=None, activation=None, bias=True):
        super(IntraConv, self).__init__()
        self.add_self = add_self
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        self.norm = norm
        self.fc_self = nn.Linear(in_feats, out_feats, bias=bias)
        self.fc_neigh = nn.Linear(in_feats, out_feats, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_feats)) if bias else None
        self._in_feats = in_feats
        self._out_feats = out_feats

    def forward(self, src, dst, feat, n, edge_weight=None):
        feat = self.feat_drop(feat)
        lin_before_mp = self._in_feats > self._out_feats
        src_h = self.fc_neigh(feat) if lin_before_mp else feat
        h_neigh = _scatter_mean(src_h[src], dst, n, edge_weight)
        if not lin_before_mp:
            h_neigh = self.fc_neigh(h_neigh)
        rst = self.fc_self(feat) + h_neigh if self.add_self else h_neigh
        if self.bias is not None:
            rst = rst + self.bias
        if self.activation is not None:
            rst = self.activation(rst)
        if self.norm is not None:
            rst = self.norm(rst)
        return rst


class DGAGNN(nn.Module):
    def __init__(self, in_feats, n_hidden, num_nodes, n_classes, n_etypes,
                 dropout=0.3, n_head=1, unclear_up=0.1, unclear_down=0.1):
        super(DGAGNN, self).__init__()
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_etypes = n_etypes
        self.n_head = n_head
        self.unclear_up = unclear_up
        self.unclear_down = unclear_down
        self.register_buffer('super_mask', torch.ones(num_nodes, n_classes))

        layers = []
        dim = in_feats
        for hidden in (n_hidden, n_hidden):
            layers.extend([
                nn.Dropout(dropout),
                nn.Linear(dim, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
            ])
            dim = hidden
        self.last_dim = dim
        self.emb_layer = nn.Sequential(*layers)
        self.emb_layer_fc = nn.Linear(n_hidden, n_classes)
        self.final_fc_layer = nn.Sequential(
            nn.Linear(n_head * n_hidden, n_hidden // 2),
            nn.ReLU(),
            nn.Linear(n_hidden // 2, n_classes),
        )
        attn_fn = nn.Tanh()
        self.W_f = nn.Sequential(nn.Linear(n_hidden, n_hidden * n_head), attn_fn)
        self.W_x = nn.Sequential(nn.Linear(n_hidden, n_hidden * n_head), attn_fn)

        dgas = []
        for _ in range(n_etypes):
            dgas.append(nn.ModuleDict({
                'all': IntraConv(self.last_dim, n_hidden, add_self=True, bias=False,
                                 norm=nn.BatchNorm1d(n_hidden), activation=nn.ReLU()),
                'gp0': IntraConv(self.last_dim, n_hidden, add_self=False, bias=False,
                                 norm=nn.BatchNorm1d(n_hidden), activation=nn.ReLU()),
                'gp1': IntraConv(self.last_dim, n_hidden, add_self=False, bias=False,
                                 norm=nn.BatchNorm1d(n_hidden), activation=nn.ReLU()),
            }))
        self.dgas = nn.ModuleList(dgas)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def update_super_mask(self, prob_hist):
        """``prob_hist`` is a list of (N, C) softmax tensors; official uses the last 10."""
        stacked = torch.stack(prob_hist[-10:], dim=0).mean(dim=0)
        self.super_mask.copy_(stacked.to(device=self.super_mask.device, dtype=self.super_mask.dtype))

    def forward(self, relations, feat):
        n = feat.size(0)
        x = self.emb_layer(feat)
        emb_out = self.emb_layer_fc(x)
        h_list = []
        for idx, (_name, src, dst) in enumerate(relations):
            w0 = (self.super_mask[src, 1] <= self.unclear_down).to(dtype=feat.dtype)
            w1 = (self.super_mask[src, 1] > self.unclear_up).to(dtype=feat.dtype)
            h_list.append(self.dgas[idx]['all'](src, dst, x, n, None))
            h_list.append(self.dgas[idx]['gp0'](src, dst, x, n, w0))
            h_list.append(self.dgas[idx]['gp1'](src, dst, x, n, w1))

        s_len = len(h_list)
        h_stack = torch.stack(h_list, dim=1)
        h_proj = self.W_f(h_stack).view(n, s_len, self.n_head, self.n_hidden)
        h_proj = h_proj.permute(0, 2, 1, 3).contiguous().view(-1, s_len, self.n_hidden)
        x_proj = self.W_x(x).view(n, self.n_head, self.n_hidden, 1).reshape(-1, self.n_hidden, 1)
        attn = F.softmax(torch.bmm(h_proj, x_proj), dim=1).transpose(1, 2)
        h_rep = h_stack.repeat(self.n_head, 1, 1)
        weighted = torch.bmm(attn, h_rep).squeeze(1)
        h = weighted.view(n, -1)
        return self.final_fc_layer(h), emb_out
