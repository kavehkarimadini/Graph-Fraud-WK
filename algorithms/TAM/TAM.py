"""
TAM (Qiao & Pang, NeurIPS 2023): Truncated Affinity Maximization, one-class
homophily modelling for (unsupervised) graph anomaly detection.

Port of the official implementation (reference/TAM-master-main: ``model.py``,
``utils.py``, ``train.py`` / ``train_large_graph.py``) on edge lists with sparse
CSR products instead of dense ``N x N`` matrices:

- **local affinity:** the anomaly measure is the mean cosine similarity of a
  node's representation to its neighbours (self-loop included) on the original
  graph; low affinity means anomalous;
- **LAMNet:** a two-layer GCN trained without labels to maximise the summed local
  affinity;
- **non-homophily truncation (NSGT):** every node drops the edges to neighbours
  whose raw-feature distance exceeds a random threshold between the graph-wide
  mean distance and its own maximum; truncation is applied ``K`` times on ``T``
  independent graph sequences, and a fresh LAMNet is trained on each truncated
  graph. The anomaly score is ``1 - minmax(mean affinity)``.

The affinity ``sum_j A_ij cos(z_i, z_j) / d_i`` equals ``z_i . (A z)_i / d_i`` for
unit-norm ``z``, so only sparse products over edges are needed.
"""
import warnings

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


def _csr_tensor(mat, device):
    mat = mat.tocsr()
    mat.sum_duplicates()
    mat.sort_indices()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t = torch.sparse_csr_tensor(
            torch.as_tensor(mat.indptr.astype(np.int64)),
            torch.as_tensor(mat.indices.astype(np.int64)),
            torch.as_tensor(mat.data.astype(np.float32)),
            size=mat.shape)
    return t.to(device)


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


class RawGraph(object):
    """Original graph ``A + I`` (binary union of the relations plus self-loops) as a
    symmetric CSR edge list, with reverse-edge indices and per-edge feature distances."""

    def __init__(self, relations, n, features, device, chunk=1 << 22):
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
        adj = (adj + sp.identity(n, dtype=np.float32, format='csr')).tocsr()
        adj.sort_indices()
        coo = adj.tocoo()
        row = coo.row.astype(np.int64)
        col = coo.col.astype(np.int64)
        key = row * n + col
        rev = np.searchsorted(key, col * n + row)

        self.n = n
        self.num_edges = int(row.size)
        self.row = torch.as_tensor(row, device=device)
        self.col = torch.as_tensor(col, device=device)
        self.rev = torch.as_tensor(rev, device=device)
        self.adj = _csr_tensor(adj, device)
        self.degree = torch.as_tensor(np.asarray(adj.sum(axis=0)).ravel(), dtype=torch.float32,
                                      device=device)
        x = torch.as_tensor(features, dtype=torch.float32, device=device)
        dis = torch.empty(self.num_edges, dtype=torch.float32, device=device)
        for s in range(0, self.num_edges, chunk):
            diff = x[self.row[s:s + chunk]] - x[self.col[s:s + chunk]]
            dis[s:s + chunk] = (diff * diff).sum(dim=1).sqrt()
        self.distance = dis

    def normalized_adj(self, mask):
        """``D^-1/2 A_cut D^-1/2`` (reference ``normalize_adj_tensor``) for the edges in ``mask``."""
        row = self.row[mask]
        col = self.col[mask]
        deg = torch.zeros(self.n, dtype=torch.float32, device=row.device).index_add_(
            0, col, torch.ones(row.numel(), dtype=torch.float32, device=row.device))
        inv = deg.pow(-0.5)
        inv[torch.isinf(inv)] = 0
        val = inv[row] * inv[col]
        crow = torch.zeros(self.n + 1, dtype=torch.long, device=row.device)
        crow[1:] = torch.cumsum(torch.bincount(row, minlength=self.n), 0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return torch.sparse_csr_tensor(crow, col, val, size=(self.n, self.n))


@torch.no_grad()
def truncate(graph, mask, rng):
    """One NSGT step (reference ``graph_nsgt``) on the edges in ``mask``.

    Node ``i`` with ``max_j d_ij > mean_d`` draws ``r_i ~ U(mean_d, max_j d_ij)`` and
    removes its edges with ``d_ij > r_i``; an edge survives if either endpoint kept it.
    ``mean_d`` is the mean non-zero distance over the current (directed) edges.
    """
    dis = graph.distance
    active = mask & (dis != 0)
    if not bool(active.any()):
        return mask.clone()
    mean_dis = dis[active].mean()
    neg = torch.full_like(dis, -1.0)
    max_dis = torch.full((graph.n,), -1.0, device=dis.device).scatter_reduce(
        0, graph.row, torch.where(mask, dis, neg), reduce='amax', include_self=True)
    qualify = max_dis > mean_dis
    r = torch.full((graph.n,), float('inf'), device=dis.device)
    idx = torch.nonzero(qualify).view(-1)
    if idx.numel() > 0:
        draws = torch.as_tensor(rng.random_sample(idx.numel()), dtype=torch.float32, device=dis.device)
        r[idx] = (max_dis[idx] - mean_dis) * draws + mean_dis
    kept = mask & ~(dis > r[graph.row])
    return kept | kept[graph.rev]


class GCNLayer(nn.Module):
    """``PReLU(A_hat (X W) + b)`` with Xavier-initialised ``W`` (reference ``GCN``)."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.fc.weight)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.act = nn.PReLU()

    def forward(self, x, adj):
        return self.act(sym_spmm(adj, self.fc(x)) + self.bias)


class LAMNet(nn.Module):
    """Two-layer GCN encoder (reference ``Model``); ``fc1`` feeds the edge regulariser."""

    def __init__(self, in_dim, embed_dim=128):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, 2 * embed_dim)
        self.gcn2 = GCNLayer(2 * embed_dim, embed_dim)
        self.fc1 = nn.Linear(embed_dim, 2 * embed_dim, bias=False)

    def forward(self, x, adj):
        emb = self.gcn2(self.gcn1(x, adj), adj)
        return emb, self.fc1(emb)


def _unit(emb):
    norm = emb.norm(dim=-1, keepdim=True)
    return torch.where(norm > 0, emb / norm, torch.zeros_like(emb))


def local_affinity(emb, graph):
    """``sum_j A_ij cos(z_i, z_j) / d_i`` on the original graph (reference ``inference``)."""
    z = _unit(emb)
    return (z * sym_spmm(graph.adj, z)).sum(dim=1) / graph.degree


def non_neighbor_similarity(emb, graph):
    """Reference ``reg_edge``: sum over nodes of the mean cosine similarity to non-neighbours."""
    z = _unit(emb)
    total = z @ z.sum(dim=0)
    nbr = (z * sym_spmm(graph.adj, z)).sum(dim=1)
    count = graph.n - graph.degree
    inv = torch.where(count > 0, 1.0 / count.clamp(min=1), torch.zeros_like(count))
    return ((total - nbr) * inv).sum()
