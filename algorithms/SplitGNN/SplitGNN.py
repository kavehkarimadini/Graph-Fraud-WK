'''
SplitGNN (Wu et al., CIKM 2023): spectral GNN for fraud detection against heterophily.

Port of reference/SplitGNN-master using torch.sparse instead of DGL.
'''
from __future__ import print_function

import numpy as np
import scipy
import sympy
import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_theta2(d):
    thetas = []
    x = sympy.symbols('x')
    for k in range(d + 1):
        f = sympy.poly((x / 2) ** k * (1 - x / 2) ** (d - k) / scipy.special.beta(k + 1, d + 1 - k))
        coeff = f.all_coeffs()
        thetas.append([float(coeff[d - j]) for j in range(d + 1)])
    return thetas


def hinge_loss(labels, scores):
    if scores.numel() == 0:
        return scores.new_zeros(())
    return F.relu(1.0 - labels.float() * scores).mean()


def _sparse_adj(src, dst, n, device, dtype):
    if src.numel() == 0:
        idx = torch.zeros(2, 0, dtype=torch.long, device=device)
        val = torch.zeros(0, dtype=dtype, device=device)
        return torch.sparse_coo_tensor(
            idx, val, (n, n), device=device, check_invariants=False).coalesce()
    idx = torch.stack([dst, src], dim=0)
    val = torch.ones(src.numel(), dtype=dtype, device=device)
    return torch.sparse_coo_tensor(
        idx, val, (n, n), device=device, check_invariants=False).coalesce()


def _unn_laplacian(feat, adj, d_invsqrt):
    return feat - torch.sparse.mm(adj, feat * d_invsqrt) * d_invsqrt


class RelationAware(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super(RelationAware, self).__init__()
        self.d_liner = nn.Linear(input_dim, output_dim)
        self.f_liner = nn.Linear(3 * output_dim, 1)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, dst):
        src = self.d_liner(src)
        dst = self.d_liner(dst)
        e_feats = torch.cat([src, dst, src - dst], dim=1)
        e_feats = self.dropout(e_feats)
        score = self.f_liner(e_feats).reshape(-1)
        return self.tanh(score)


class PolyConv(nn.Module):
    def __init__(self, in_feats, out_feats, relation_aware, thetas, K,
                 activation=F.leaky_relu, lin=False, bias=True):
        super(PolyConv, self).__init__()
        self._theta = thetas
        self._k = len(self._theta[0])
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.relation_aware = relation_aware
        self.K = K
        self.linear = nn.Linear(in_feats * len(thetas), out_feats, bias)
        self.linear1 = nn.Linear(in_feats * len(thetas), out_feats, bias)
        self.transh = nn.Linear(in_feats, out_feats, bias)
        self.lin = lin

    def forward(self, src, dst, feat):
        n = feat.size(0)
        score = self.relation_aware(feat[src], feat[dst])
        sign = torch.sign(score)
        pos = sign >= 0
        neg = sign < 0

        adj_all = _sparse_adj(src, dst, n, feat.device, feat.dtype)
        d_all = torch.pow(torch.bincount(dst, minlength=n).to(dtype=feat.dtype).clamp(min=1), -0.5).unsqueeze(-1)

        ones = torch.ones(src.numel(), device=feat.device, dtype=feat.dtype)
        pos_deg = torch.zeros(n, device=feat.device, dtype=feat.dtype)
        pos_deg.scatter_add_(0, dst, ones * pos.to(feat.dtype))
        neg_deg = torch.zeros(n, device=feat.device, dtype=feat.dtype)
        neg_deg.scatter_add_(0, dst, ones * neg.to(feat.dtype))
        d_pos = torch.pow(pos_deg.clamp(min=1), -0.5).unsqueeze(-1)
        d_neg = torch.pow(neg_deg.clamp(min=1), -0.5).unsqueeze(-1)
        adj_pos = _sparse_adj(src[pos], dst[pos], n, feat.device, feat.dtype)
        adj_neg = _sparse_adj(src[neg], dst[neg], n, feat.device, feat.dtype)

        transh = self.transh(feat)
        feat_orig = feat

        hs_o = []
        feat = feat_orig
        for theta in self._theta:
            h_o = theta[0] * feat
            for _k in range(1, self._k):
                feat = _unn_laplacian(feat, adj_all, d_all)
                h_o = h_o + theta[_k] * feat
            hs_o.append(h_o)

        hs_p = []
        feat = feat_orig
        for theta in self._theta[0:self.K + 1]:
            h_p = theta[0] * feat
            for _k in range(1, self._k):
                feat = _unn_laplacian(feat, adj_pos, d_pos)
                h_p = h_p + theta[_k] * feat
            hs_p.append(h_p)

        hs_n = []
        feat = feat_orig
        for theta in self._theta[self.K + 1:]:
            h_n = theta[0] * feat
            for _k in range(1, self._k):
                feat = _unn_laplacian(feat, adj_neg, d_neg)
                h_n = h_n + theta[_k] * feat
            hs_n.append(h_n)

        hs_o = torch.cat(hs_o, dim=1)
        if self.K != len(self._theta) - 1 and self.K != -1:
            hs_pn = torch.cat([torch.cat(hs_p, dim=1), torch.cat(hs_n, dim=1)], dim=1)
        elif self.K == -1:
            hs_pn = torch.cat(hs_n, dim=1)
        else:
            hs_pn = torch.cat(hs_p, dim=1)

        if self.lin:
            hs_o = self.activation(self.linear(hs_o))
            hs_pn = self.activation(self.linear1(hs_pn))
        return hs_o, hs_pn, transh


class MultiRelationSplitGNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, etypes, dropout, thetas, K):
        super(MultiRelationSplitGNNLayer, self).__init__()
        self.relation = [e for e in etypes if e != 'homo']
        if not self.relation:
            raise ValueError('SplitGNN needs at least one non-homo relation')
        self.liner = nn.Linear(len(self.relation) * output_dim * 3, output_dim)
        self.relation_aware = RelationAware(input_dim, output_dim, dropout)
        self.minelayers = nn.ModuleDict()
        self.dropout = nn.Dropout(dropout)
        for e in self.relation:
            self.minelayers[e] = PolyConv(
                input_dim, output_dim, self.relation_aware, thetas, K, lin=True)

    def forward(self, g, h):
        hs_o, hs_lh, hs_trans = [], [], []
        for e in self.relation:
            src, dst = g.edges(e)
            h_o, h_lh, h_trans = self.minelayers[e](src, dst, h)
            hs_o.append(h_o)
            hs_lh.append(h_lh)
            hs_trans.append(h_trans)
        h = torch.cat([torch.cat(hs_o, dim=1), torch.cat(hs_lh, dim=1), torch.cat(hs_trans, dim=1)], dim=1)
        return self.liner(self.dropout(h))

    def loss(self, g, h):
        agg_h = self.forward(g, h)
        src, dst = g.edges('homo')
        scores = self.relation_aware(h[src], h[dst])
        edge_train_mask = g.edata['homo']['train_mask'].bool()
        edge_train_label = g.edata['homo']['label'][edge_train_mask]
        edge_train_score = scores[edge_train_mask]
        pos = (edge_train_label == 1).nonzero(as_tuple=False).view(-1)
        neg = (edge_train_label == -1).nonzero(as_tuple=False).view(-1)
        if pos.numel() == 0 or neg.numel() == 0:
            return agg_h, hinge_loss(edge_train_label, edge_train_score)
        pos_np = pos.detach().cpu().numpy()
        neg_np = neg.detach().cpu().numpy()
        pos_np = np.random.choice(pos_np, size=len(neg_np))
        index = np.sort(np.concatenate([pos_np, neg_np]))
        index_t = torch.as_tensor(index, device=scores.device, dtype=torch.long)
        return agg_h, hinge_loss(edge_train_label[index_t], edge_train_score[index_t])


class SplitGNN(nn.Module):
    def __init__(self, in_feats, intra_dim, n_class, etypes, dropout, C, K, gamma):
        super(SplitGNN, self).__init__()
        self.gamma = gamma
        self.thetas = calculate_theta2(d=C)
        self.mine_layer = MultiRelationSplitGNNLayer(
            intra_dim, intra_dim, etypes, dropout, self.thetas, K)
        self.linear = nn.Linear(in_feats, intra_dim)
        self.linear2 = nn.Linear(intra_dim, n_class)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU()

    def encode(self, g):
        h = self.linear(g.ndata['feature'].float())
        h = self.mine_layer(g, h)
        h = self.relu(h)
        h = self.dropout(h)
        return self.linear2(h)

    def forward(self, g):
        return self.encode(g)

    def loss(self, g):
        h = self.linear(g.ndata['feature'].float())
        h, edge_loss = self.mine_layer.loss(g, h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.linear2(h)

        train_mask = g.ndata['train_mask'].bool()
        train_label = g.ndata['label'][train_mask]
        train_logit = h[train_mask]
        pos = (train_label == 1).nonzero(as_tuple=False).view(-1)
        neg = (train_label == 0).nonzero(as_tuple=False).view(-1)
        if pos.numel() == 0 or neg.numel() == 0:
            model_loss = F.cross_entropy(train_logit, train_label)
        else:
            pos_np = pos.detach().cpu().numpy()
            neg_np = neg.detach().cpu().numpy()
            if len(neg_np) >= len(pos_np):
                neg_np = np.random.choice(neg_np, size=len(pos_np), replace=False)
            else:
                neg_np = np.random.choice(neg_np, size=len(pos_np), replace=True)
            index = np.sort(np.concatenate([pos_np, neg_np]))
            index_t = torch.as_tensor(index, device=h.device, dtype=torch.long)
            model_loss = F.cross_entropy(train_logit[index_t], train_label[index_t])
        return model_loss + self.gamma * edge_loss
