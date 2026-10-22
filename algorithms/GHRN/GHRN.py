"""
GHRN (Gao et al., WWW 2023): addressing heterophily in graph anomaly detection
from the perspective of the graph spectrum.

Port of the official DGL implementation (reference/GHRN-main: ``main.py``,
``BWGNN.py``, ``dataset.py``) with ``torch`` sparse CSR operators:

- a BWGNN detector (beta-wavelet polynomial filters of ``L = I - D^-1/2 A D^-1/2``)
  is trained on the original graph;
- its class probabilities ``Y`` give the high-frequency indicator
  ``LY = Y - A_hat Y`` (``A_hat`` normalised with self-loops); every edge is scored
  by ``<LY_u, LY_v>`` and the ``del_ratio`` fraction of edges with the smallest
  scores (likely inter-class, heterophilous edges) is deleted;
- a fresh BWGNN is trained on the pruned graph.

``BWGNN`` is the homogeneous detector; ``BWGNNHetero`` runs the filters on each
relation separately and sums the relation embeddings (reference ``BWGNN_Hetero``).
"""
import math
import warnings

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


def beta_wavelet_thetas(order):
    """Monomial coefficients (in ``L``) of the beta-wavelet kernels
    ``(L/2)^k (1 - L/2)^(d-k) / B(k+1, d+1-k)``, ``k = 0..d`` (reference ``calculate_theta2``)."""
    thetas = []
    for k in range(order + 1):
        beta = math.gamma(k + 1) * math.gamma(order + 1 - k) / math.gamma(order + 2)
        coeff = [0.0] * (order + 1)
        for j in range(order - k + 1):
            coeff[k + j] = math.comb(order - k, j) * (-1) ** j / 2 ** (k + j) / beta
        thetas.append(coeff)
    return thetas


def relation_edges(src, dst, n):
    """Binary, symmetric, de-duplicated edges of one relation without self-loops."""
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    adj = sp.csr_matrix((np.ones(2 * src.size, dtype=np.float32),
                         (np.concatenate([src, dst]), np.concatenate([dst, src]))), shape=(n, n))
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj = adj.tocoo()
    return adj.row.astype(np.int64), adj.col.astype(np.int64)


def add_self_loops(src, dst, n):
    loop = np.arange(n, dtype=np.int64)
    return np.concatenate([src, loop]), np.concatenate([dst, loop])


def remove_self_loops(src, dst):
    keep = src != dst
    return src[keep], dst[keep]


def _torch_csr(mat, device):
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


class NormAdj(object):
    """``A_hat = D^-1/2 A D^-1/2`` with ``A[v, u]`` = number of ``u -> v`` edges and ``D``
    the in-degree clamped at 1 (BWGNN ``PolyConv``: DGL ``copy_u`` + ``sum``).

    Holds ``A_hat`` and its transpose as torch CSR tensors for forward / backward.
    """

    def __init__(self, src, dst, n, device):
        src = np.asarray(src, dtype=np.int64)
        dst = np.asarray(dst, dtype=np.int64)
        deg = np.maximum(np.bincount(dst, minlength=n), 1).astype(np.float64)
        inv = deg ** -0.5
        val = (inv[src] * inv[dst]).astype(np.float32)
        mat = sp.csr_matrix((val, (dst, src)), shape=(n, n))
        self.mat = _torch_csr(mat, device)
        self.mat_t = _torch_csr(mat.T, device)
        self.num_edges = int(src.size)


class _SpMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, adj, x):
        ctx.adj = adj
        return adj.mat @ x

    @staticmethod
    def backward(ctx, grad):
        return None, ctx.adj.mat_t @ grad


def laplacian_powers(adj, h, order):
    """``[h, L h, ..., L^order h]`` with ``L = I - A_hat``."""
    powers = [h]
    for _ in range(order):
        powers.append(powers[-1] - _SpMM.apply(adj, powers[-1]))
    return powers


class BWGNN(nn.Module):
    """Homogeneous BWGNN (reference ``BWGNN.py``). All ``order + 1`` wavelet filters
    take the same input, so the Laplacian powers are computed once and shared."""

    def __init__(self, in_dim, hidden, n_class=2, order=2):
        super().__init__()
        self.order = order
        self.register_buffer('thetas', torch.tensor(beta_wavelet_thetas(order), dtype=torch.float32))
        self.linear = nn.Linear(in_dim, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.linear3 = nn.Linear(hidden * (order + 1), hidden)
        self.linear4 = nn.Linear(hidden, n_class)
        self.act = nn.ReLU()

    def wavelets(self, adj, h):
        powers = torch.stack(laplacian_powers(adj, h, self.order), dim=0)
        filtered = torch.einsum('fk,knd->fnd', self.thetas, powers)
        return torch.cat(list(filtered), dim=-1)

    def forward(self, x, adjs):
        h = self.act(self.linear(x))
        h = self.act(self.linear2(h))
        h = self.act(self.linear3(self.wavelets(adjs[0], h)))
        return self.linear4(h)


class BWGNNHetero(BWGNN):
    """Per-relation BWGNN (reference ``BWGNN_Hetero``): wavelets on each relation,
    a shared ``linear3`` per relation, summed, then LeakyReLU and the classifier."""

    def __init__(self, in_dim, hidden, n_class=2, order=2):
        super().__init__(in_dim, hidden, n_class=n_class, order=order)
        self.act = nn.LeakyReLU()

    def forward(self, x, adjs):
        h = self.act(self.linear(x))
        h = self.act(self.linear2(h))
        h_all = torch.stack([self.linear3(self.wavelets(adj, h)) for adj in adjs]).sum(0)
        return self.linear4(self.act(h_all))


@torch.no_grad()
def heterophily_scores(src, dst, n, probs, adj_type='sym'):
    """Edge indicator ``<LY_u, LY_v>`` with ``LY = Y - A_w Y`` (reference ``random_walk_update``).

    ``src`` / ``dst`` must already contain the self-loops. ``A_w`` uses DGL
    ``EdgeWeightNorm``: ``1 / sqrt(out_deg(u) in_deg(v))`` (``sym``) or ``1 / out_deg(u)`` (``rw``).
    """
    device = probs.device
    s = torch.as_tensor(src, dtype=torch.long, device=device)
    d = torch.as_tensor(dst, dtype=torch.long, device=device)
    ones = torch.ones(s.numel(), dtype=probs.dtype, device=device)
    out_deg = torch.zeros(n, dtype=probs.dtype, device=device).index_add_(0, s, ones)
    if adj_type == 'sym':
        in_deg = torch.zeros(n, dtype=probs.dtype, device=device).index_add_(0, d, ones)
        w = (out_deg[s] * in_deg[d]).rsqrt()
    else:
        w = out_deg[s].reciprocal()
    ay = torch.zeros_like(probs).index_add_(0, d, probs[s] * w.unsqueeze(1))
    ly = probs - ay
    return (ly[s] * ly[d]).sum(dim=1)


@torch.no_grad()
def prune_edges(src, dst, n, probs, del_ratio, adj_type='sym'):
    """GHRN edge deletion on one (self-loop free) edge list.

    Self-loops are added, the ``int(del_ratio * E)`` edges with the smallest
    indicator are removed, and the self-loops are dropped again.
    Returns the kept ``(src, dst)`` numpy arrays and the number of deleted non-loop edges.
    """
    full_src, full_dst = add_self_loops(src, dst, n)
    scores = heterophily_scores(full_src, full_dst, n, probs, adj_type)
    threshold = int(del_ratio * full_src.size)
    keep = np.ones(full_src.size, dtype=bool)
    if threshold > 0:
        order = torch.sort(scores, stable=True).indices[:threshold].cpu().numpy()
        keep[order] = False
    new_src, new_dst = remove_self_loops(full_src[keep], full_dst[keep])
    return new_src, new_dst, int(src.size - new_src.size)
