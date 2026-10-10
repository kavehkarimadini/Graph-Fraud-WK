"""
GGCN (Yan et al., ICDM 2022): "Two Sides of the Same Coin: Heterophily and
Oversmoothing in Graph Convolutional Neural Networks".

Port of the official implementation (reference/Heterophily_and_oversmoothing-main,
``model.py``: ``GGCN``, ``GGCNlayer``, ``GGCNlayer_SP``) on edge lists with sparse
CSR kernels instead of dense ``N x N`` matrices:

- **signed messages:** each layer scores every (non self-loop) edge by the cosine
  similarity of the transformed endpoints; positive and negative parts are
  propagated separately and mixed with the node's own message by learnable
  softmax weights, all scaled by a learnable softplus factor;
- **degree correction:** edge weights of the symmetric normalised adjacency are
  multiplied by ``softplus(a * (A_ii / A_ij - 1) + b)`` with learnable ``a, b``;
- **decaying aggregation:** layer outputs are accumulated with residual weights
  ``log(decay_rate / (l + 2)^exponent + 1)`` to limit oversmoothing.
"""
import math
import warnings

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def _csr(crow, col, val, n):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return torch.sparse_csr_tensor(crow, col, val, size=(n, n))


class EdgeGraph(object):
    """Symmetric normalised adjacency ``D^-1/2 (A + I) D^-1/2`` (reference
    ``sys_normalized_adjacency``) as a CSR edge list, plus its transpose layout.

    ``A`` is the binary union of the relations; ``row`` / ``col`` index
    ``A_hat[row, col]``, so ``(A_hat @ X)[i] = sum_j A_hat[i, j] X[j]``.
    """

    def __init__(self, relations, n, device):
        rows, cols = [], []
        for src, dst in relations:
            src = np.asarray(src, dtype=np.int64)
            dst = np.asarray(dst, dtype=np.int64)
            rows.extend([src, dst])
            cols.extend([dst, src])
        row = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
        col = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
        adj = sp.csr_matrix((np.ones(row.size, dtype=np.float32), (row, col)), shape=(n, n))
        adj.sum_duplicates()
        adj.data[:] = 1.0
        adj = (adj + sp.identity(n, dtype=np.float32, format='csr')).tocsr()
        rowsum = np.asarray(adj.sum(axis=1)).ravel()
        rowsum[rowsum == 0] = 1.0
        d_inv = sp.diags((rowsum ** -0.5).astype(np.float32))
        adj = (d_inv @ adj @ d_inv).tocsr()
        adj.sum_duplicates()
        adj.sort_indices()
        coo = adj.tocoo()
        row = coo.row.astype(np.int64)
        col = coo.col.astype(np.int64)
        val = coo.data.astype(np.float32)
        diag = np.asarray(adj.diagonal(), dtype=np.float32)
        perm = np.lexsort((row, col))

        self.n = n
        self.num_edges = int(row.size)
        self.row = torch.as_tensor(row, device=device)
        self.col = torch.as_tensor(col, device=device)
        self.crow = torch.as_tensor(adj.indptr.astype(np.int64), device=device)
        self.perm = torch.as_tensor(perm, device=device)
        self.col_t = self.row[self.perm]
        crow_t = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(np.bincount(col, minlength=n), out=crow_t[1:])
        self.crow_t = torch.as_tensor(crow_t, device=device)
        self.weight = torch.as_tensor(val, device=device)
        self.offdiag = self.row != self.col
        self.degree_term = torch.as_tensor(diag[row] / val - 1.0, device=device)

    def mm(self, val, x):
        return _csr(self.crow, self.col, val, self.n) @ x

    def mm_t(self, val, x):
        return _csr(self.crow_t, self.col_t, val[self.perm], self.n) @ x


def _edge_dot(graph, x, y, chunk=None):
    """``<x[row_e], y[col_e]>`` for every edge, computed in chunks."""
    e = graph.num_edges
    if chunk is None:
        chunk = max(1, (1 << 24) // max(x.size(1), 1))
    out = x.new_empty(e)
    for s in range(0, e, chunk):
        r = graph.row[s:s + chunk]
        c = graph.col[s:s + chunk]
        out[s:s + chunk] = (x[r] * y[c]).sum(dim=1)
    return out


class _SDDMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, graph, x, y):
        ctx.graph = graph
        ctx.save_for_backward(x, y)
        return _edge_dot(graph, x, y)

    @staticmethod
    def backward(ctx, grad):
        x, y = ctx.saved_tensors
        g = ctx.graph
        dx = g.mm(grad, y) if ctx.needs_input_grad[1] else None
        dy = g.mm_t(grad, x) if ctx.needs_input_grad[2] else None
        return None, dx, dy


class _SpMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, graph, val, x):
        ctx.graph = graph
        ctx.save_for_backward(val, x)
        return graph.mm(val, x)

    @staticmethod
    def backward(ctx, grad):
        val, x = ctx.saved_tensors
        g = ctx.graph
        dval = _edge_dot(g, grad, x) if ctx.needs_input_grad[1] else None
        dx = g.mm_t(val, grad) if ctx.needs_input_grad[2] else None
        return None, dval, dx


def edge_dot(graph, x, y):
    return _SDDMM.apply(graph, x, y)


def spmm(graph, val, x):
    return _SpMM.apply(graph, val, x)


class GGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, use_degree=True, use_sign=True, use_decay=True,
                 scale_init=0.5, deg_intercept_init=0.5):
        super().__init__()
        self.fcn = nn.Linear(in_dim, out_dim)
        self.use_degree = use_degree
        self.use_sign = use_sign
        if use_degree:
            self.deg_coeff = nn.Parameter(torch.tensor([0.5 if use_decay else deg_intercept_init, 0.0]))
        if use_sign:
            self.coeff = nn.Parameter(torch.zeros(3))
            self.scale = nn.Parameter(torch.full((1,), 2.0 if use_decay else scale_init))

    def forward(self, h, graph):
        wh = self.fcn(h)
        weight = graph.weight
        if self.use_degree:
            weight = weight * F.softplus(self.deg_coeff[0] * graph.degree_term + self.deg_coeff[1])
        if not self.use_sign:
            return spmm(graph, weight, wh)
        sq = (wh * wh).sum(dim=1)
        norm = sq.sqrt()
        cos = edge_dot(graph, wh, wh) / torch.clamp(norm[graph.row] * norm[graph.col], min=1e-9)
        att = torch.where(graph.offdiag, cos * weight, torch.zeros_like(cos))
        coeff = torch.softmax(self.coeff, dim=-1)
        val = coeff[0] * F.relu(att) - coeff[1] * F.relu(-att)
        return F.softplus(self.scale) * (spmm(graph, val, wh) + coeff[2] * wh)


class GGCN(nn.Module):
    def __init__(self, in_dim, hidden, n_class=2, nlayers=2, dropout=0.5, decay_rate=1.0,
                 exponent=3.0, use_degree=True, use_sign=True, use_decay=True,
                 scale_init=0.5, deg_intercept_init=0.5, use_bn=False, use_ln=False):
        super().__init__()
        if nlayers < 2:
            raise ValueError('GGCN needs at least 2 layers.')
        args = (use_degree, use_sign, use_decay, scale_init, deg_intercept_init)
        self.convs = nn.ModuleList([GGCNLayer(in_dim, hidden, *args)])
        for _ in range(nlayers - 2):
            self.convs.append(GGCNLayer(hidden, hidden, *args))
        self.convs.append(GGCNLayer(hidden, n_class, *args))
        self.fcn = nn.Linear(in_dim, hidden)
        self.dropout = dropout
        self.use_decay = use_decay
        self.decay = decay_rate
        self.exponent = exponent
        self.norms = None
        if use_bn and not use_decay:
            self.norms = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(nlayers - 1)])
        elif use_ln and not use_decay:
            self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(nlayers - 1)])

    def forward(self, x, graph):
        """Returns class logits (the reference applies ``log_softmax`` on top)."""
        x = F.dropout(x, self.dropout, training=self.training)
        layer_previous = F.elu(self.fcn(x))
        layer_inner = self.convs[0](x, graph)
        for i, conv in enumerate(self.convs[1:]):
            if self.norms is not None:
                layer_inner = self.norms[i](layer_inner)
            layer_inner = F.elu(layer_inner)
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            if i == 0:
                layer_previous = layer_inner + layer_previous
            else:
                coeff = math.log(self.decay / (i + 2) ** self.exponent + 1) if self.use_decay else 1.0
                layer_previous = coeff * layer_inner + layer_previous
            layer_inner = conv(layer_previous, graph)
        return layer_inner
