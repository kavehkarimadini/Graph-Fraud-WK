"""
AMNet (Chai et al., IJCAI 2022): Adaptive Multi-frequency graph neural Network.

Port of the official PyG implementation (reference/AMNet-main, ``model.py`` and
``bernconv.py``) with a symmetric sparse CSR operator instead of
``MessagePassing``:

- a two-layer MLP projects node features to the hidden space;
- several Bernstein-polynomial spectral filters (``BernConv``) act on the scaled
  Laplacian ``L / 2 = (I - D^-1/2 A D^-1/2) / 2``; filter ``f`` computes
  ``sum_k sigmoid(w_fk) * C(K, k) (L/2)^k (I - L/2)^(K-k) x`` with learnable ``w_f``;
- a per-node attention ``softmax_f(tanh(W_f h_f) . tanh(W_x x))`` mixes the
  filter outputs, followed by dropout and a linear classifier;
- a marginal loss pushes normal training nodes towards filter 0 and anomalous
  training nodes towards filter 1, so the filters specialise to different
  frequency bands.
"""
import math
import warnings

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_homo_graph(relations, n):
    """Undirected, binary, de-duplicated union of the relations without self-loops
    (networkx ``to_undirected`` + PyG ``remove_self_loops``). Returns a scipy CSR matrix."""
    rows, cols = [], []
    for src, dst in relations:
        src = np.asarray(src, dtype=np.int64)
        dst = np.asarray(dst, dtype=np.int64)
        rows.extend([src, dst])
        cols.extend([dst, src])
    row = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    col = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
    keep = row != col
    adj = sp.csr_matrix((np.ones(int(keep.sum()), dtype=np.float32), (row[keep], col[keep])),
                        shape=(n, n))
    adj.sum_duplicates()
    adj.data[:] = 1.0
    return adj


def scaled_laplacian(adj, lambda_max=2.0):
    """``(I - D^-1/2 A D^-1/2) / lambda_max`` (PyG ``get_laplacian(..., 'sym')``).
    Isolated nodes keep a diagonal of ``1 / lambda_max``."""
    n = adj.shape[0]
    deg = np.asarray(adj.sum(axis=1)).ravel()
    with np.errstate(divide='ignore'):
        inv = np.power(deg, -0.5)
    inv[np.isinf(inv)] = 0.0
    d_inv = sp.diags(inv.astype(np.float32))
    lap = sp.identity(n, dtype=np.float32, format='csr') - d_inv @ adj @ d_inv
    lap = (lap / lambda_max).tocsr().astype(np.float32)
    lap.sum_duplicates()
    lap.sort_indices()
    return lap


def to_torch_csr(mat, device=None):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t = torch.sparse_csr_tensor(
            torch.as_tensor(mat.indptr.astype(np.int64)),
            torch.as_tensor(mat.indices.astype(np.int64)),
            torch.as_tensor(mat.data.astype(np.float32)),
            size=mat.shape)
    return t.to(device) if device is not None else t


class _SymSpMM(torch.autograd.Function):
    """``A @ x`` for a constant symmetric sparse ``A`` (backward is ``A @ grad``)."""

    @staticmethod
    def forward(ctx, mat, x):
        ctx.mat = mat
        return mat @ x

    @staticmethod
    def backward(ctx, grad):
        return None, ctx.mat @ grad


def sym_spmm(mat, x):
    return _SymSpMM.apply(mat, x)


def bernstein_coeff(order):
    """``C[k, i]``: monomial coefficient of ``x^i`` in ``C(K, k) x^k (1 - x)^(K - k)``
    (reference ``BernConv.get_bern_coeff``)."""
    coeff = np.zeros((order + 1, order + 1), dtype=np.float64)
    for k in range(order + 1):
        for j in range(order - k + 1):
            coeff[k, k + j] = math.comb(order, k) * math.comb(order - k, j) * (-1) ** j
    return coeff


class BernConv(nn.Module):
    """Bernstein filter with sigmoid-normalised coefficients (``normalization=True``).

    Operates on precomputed powers ``[x, Px, ..., P^K x]`` so several filters can
    share the propagation.
    """

    def __init__(self, order):
        super().__init__()
        self.order = order
        self.weight = nn.Parameter(torch.zeros(order + 1, 1))
        self.register_buffer('coeff', torch.tensor(bernstein_coeff(order), dtype=torch.float32))

    def forward(self, powers):
        mix = (torch.sigmoid(self.weight) * self.coeff).sum(dim=0)
        return torch.einsum('i,ind->nd', mix, powers)


class AMNet(nn.Module):
    def __init__(self, in_dim, hidden, n_class=2, order=5, n_filters=2, dropout=0.3):
        super().__init__()
        if n_filters < 2:
            raise ValueError('AMNet needs at least two filters for the marginal loss.')
        self.order = order
        self.n_filters = n_filters
        self.linear_transform_in = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.filters = nn.ModuleList([BernConv(order) for _ in range(n_filters)])
        self.W_f = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.W_x = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.linear_cls_out = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_class))

    def param_groups(self):
        """``(filter, linear, attention)`` parameters, optimised with separate settings."""
        filt = list(self.filters.parameters())
        lin = list(self.linear_transform_in.parameters()) + list(self.linear_cls_out.parameters())
        attn = list(self.W_x.parameters()) + list(self.W_f.parameters())
        return filt, lin, attn

    def forward(self, x, lap):
        """Returns ``(logits, attention)``; ``attention`` is ``(N, n_filters)``.

        ``lap`` is the scaled Laplacian as a torch sparse CSR tensor.
        """
        x = self.linear_transform_in(x)
        powers = [x]
        for _ in range(self.order):
            powers.append(sym_spmm(lap, powers[-1]))
        powers = torch.stack(powers, dim=0)
        h_filters = torch.stack([f(powers) for f in self.filters], dim=1)
        score_logit = torch.bmm(self.W_f(h_filters), self.W_x(x).unsqueeze(-1))
        score = F.softmax(score_logit, dim=1)
        res = (h_filters * score).sum(dim=1)
        return self.linear_cls_out(res), score.squeeze(-1)

    @staticmethod
    def marginal_loss(score, anomaly_idx, normal_idx):
        """``mean relu(s1 - s0)`` over normal nodes plus ``mean relu(s0 - s1)`` over anomalies."""
        loss = score.new_zeros(())
        if normal_idx is not None and normal_idx.numel() > 0:
            s = score[normal_idx]
            loss = loss + F.relu(s[:, 1] - s[:, 0]).mean()
        if anomaly_idx is not None and anomaly_idx.numel() > 0:
            s = score[anomaly_idx]
            loss = loss + F.relu(s[:, 0] - s[:, 1]).mean()
        return loss
