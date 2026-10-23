"""
CIE-GAD (Huang et al., IEEE TCYB 2025): Correlation Information Enhanced
Graph Anomaly Detection via Hypergraph Transformation.

Implementation from the paper (reference/CIE-GAD.md; no official code is shipped):

- **preliminary MLP** maps raw features to embedding size ``C``;
- **SCNAF** (Eqs. 8-15): Jacobi polynomial filters of order ``0..K`` on the
  symmetrically normalised adjacency ``A_hat = D^{-1/2} A D^{-1/2}``, fused by
  a per-node attention over the ``K+1`` filtered feature banks;
- **HCRE** (Eqs. 16-17): dual hypergraph transformation (nodes -> hyperedges,
  edges -> vertices) with equal vertex weights ``Upsilon = I``, followed by the
  hyperedge-centric convolution
  ``S = ReLU(D_e^{-1/2} H D_v^{-1/2} H^T Z_tilde Theta)``;
- **deep supervision** (Eqs. 18-21): separate linear classifiers on ``S`` and on
  ``Z_tilde``, with ``L = L_0 + gamma L_1``.

The dual-hypergraph operator is never materialised: for an undirected graph
without self-loops it equals
``D^{-1/2}(A + D) / sqrt(2)``, implemented with a shared CSR ``spmm``.
"""
import math
import warnings

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_homo_graph(relations, n):
    """Undirected, binary, de-duplicated union of the relations without self-loops."""
    rows, cols = [], []
    for src, dst in relations:
        src = np.asarray(src, dtype=np.int64)
        dst = np.asarray(dst, dtype=np.int64)
        rows.extend([src, dst])
        cols.extend([dst, src])
    row = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    col = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
    keep = row != col
    adj = sp.csr_matrix(
        (np.ones(int(keep.sum()), dtype=np.float32), (row[keep], col[keep])),
        shape=(n, n))
    adj.sum_duplicates()
    adj.data[:] = 1.0
    return adj


def normalized_adjacency(adj):
    """``A_hat = D^{-1/2} A D^{-1/2}`` (Eq. 8). Isolated nodes stay all-zero."""
    deg = np.asarray(adj.sum(axis=1)).ravel()
    with np.errstate(divide='ignore'):
        inv = np.power(deg, -0.5)
    inv[np.isinf(inv)] = 0.0
    d_inv = sp.diags(inv.astype(np.float32))
    a_hat = (d_inv @ adj @ d_inv).tocsr().astype(np.float32)
    a_hat.sum_duplicates()
    a_hat.sort_indices()
    return a_hat


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


def jacobi_coeffs(a, b, k):
    """Recursion coefficients ``(phi, phi', phi'')`` of ``P_k^{a,b}`` for ``k >= 2`` (Eq. 7)."""
    k = float(k)
    denom = 2.0 * k * (k + a + b)
    phi = (2.0 * k + a + b) * (2.0 * k + a + b - 1.0) / denom
    phi_p = ((2.0 * k + a + b - 1.0) * (a * a - b * b)
             / (denom * (2.0 * k + a + b - 2.0)))
    phi_pp = ((k + a - 1.0) * (k + b - 1.0) * (2.0 * k + a + b)
              / (k * (k + a + b) * (2.0 * k + a + b - 2.0)))
    return phi, phi_p, phi_pp


def hcre_propagate(z, adj, deg):
    """``D_e^{-1/2} H D_v^{-1/2} H^T z`` with ``Upsilon = I`` (Eq. 17, no Theta / sigma).

    For an undirected graph without self-loops this is
    ``D^{-1/2}(A z + D z) / sqrt(2)``. Isolated nodes stay zero.
    """
    az = sym_spmm(adj, z)
    out = az + deg.unsqueeze(1) * z
    scale = torch.zeros_like(deg)
    pos = deg > 0
    scale[pos] = 1.0 / (deg[pos].sqrt() * math.sqrt(2.0))
    return scale.unsqueeze(1) * out


class CIEGAD(nn.Module):
    def __init__(self, in_dim, hidden=32, order=6, n_class=2, gamma=1.0,
                 jacobi_a=1.0, jacobi_b=1.0, learn_jacobi=True, dropout=0.0,
                 use_scnaf=True, use_naf=True, use_hcre=True):
        super(CIEGAD, self).__init__()
        self.order = int(order)
        self.gamma = float(gamma)
        self.use_scnaf = bool(use_scnaf)
        self.use_naf = bool(use_naf)
        self.use_hcre = bool(use_hcre)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        n_banks = self.order + 1 if self.use_scnaf else 1
        self.W_k = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(n_banks)])
        if learn_jacobi:
            self.jacobi_a = nn.Parameter(torch.tensor(float(jacobi_a)))
            self.jacobi_b = nn.Parameter(torch.tensor(float(jacobi_b)))
        else:
            self.register_buffer('jacobi_a', torch.tensor(float(jacobi_a)))
            self.register_buffer('jacobi_b', torch.tensor(float(jacobi_b)))
        if self.use_naf:
            self.alpha_bias = None
        else:
            self.alpha_logit = nn.Parameter(torch.zeros(n_banks))
        self.theta = nn.Linear(hidden, hidden, bias=False)
        self.cls_hcre = nn.Linear(hidden, n_class)
        self.cls_scnaf = nn.Linear(hidden, n_class)
        self.dropout = nn.Dropout(dropout)

    def _ab(self):
        # Jacobi basis requires a, b > -1.
        return self.jacobi_a.clamp(min=-0.999), self.jacobi_b.clamp(min=-0.999)

    def jacobi_banks(self, x, a_hat):
        """``{Z_k}_{k=0}^K`` from Eqs. 9-10."""
        a, b = self._ab()
        banks = [x]
        if self.order >= 1:
            ax = sym_spmm(a_hat, x)
            banks.append(0.5 * (a - b) * x + 0.5 * (a + b + 2.0) * ax)
        for k in range(2, self.order + 1):
            phi, phi_p, phi_pp = jacobi_coeffs(a, b, k)
            ax = sym_spmm(a_hat, banks[-1])
            banks.append(phi * ax + phi_p * banks[-1] - phi_pp * banks[-2])
        return banks

    def fuse(self, banks):
        """Node-level (Eqs. 12-15) or scalar attention fusion of Jacobi banks."""
        projected = [self.W_k[k](z) for k, z in enumerate(banks)]
        if not self.use_naf:
            alpha = torch.softmax(self.alpha_logit, dim=0)
            fused = sum(alpha[k] * projected[k] for k in range(len(projected)))
            return F.relu(fused), alpha.unsqueeze(0).expand(banks[0].size(0), -1)
        scores = []
        for k, z in enumerate(banks):
            q = z.mean(dim=0, keepdim=True)
            scores.append(torch.tanh((projected[k] * q).sum(dim=-1)))
        alpha = torch.softmax(torch.stack(scores, dim=1), dim=1)
        fused = sum(alpha[:, k:k + 1] * projected[k] for k in range(len(projected)))
        return F.relu(fused), alpha

    def forward(self, x, a_hat, adj, deg):
        """Returns ``(logits_hcre, logits_scnaf, z_tilde, attention)``.

        ``logits_hcre`` is the primary prediction (Eq. 18). With ``use_hcre=False``
        both logits come from ``Z_tilde``. ``attention`` is ``(N, K+1)``.
        """
        h = self.dropout(self.mlp(x))
        if self.use_scnaf:
            banks = self.jacobi_banks(h, a_hat)
            z_tilde, attn = self.fuse(banks)
        else:
            z_tilde = F.relu(self.W_k[0](h))
            attn = z_tilde.new_ones(z_tilde.size(0), 1)
        logits_scnaf = self.cls_scnaf(self.dropout(z_tilde))
        if self.use_hcre:
            s = F.relu(self.theta(hcre_propagate(z_tilde, adj, deg)))
            logits_hcre = self.cls_hcre(self.dropout(s))
        else:
            logits_hcre = logits_scnaf
        return logits_hcre, logits_scnaf, z_tilde, attn

    def loss(self, logits_hcre, logits_scnaf, y, idx, weight=None):
        """``L_0 + gamma L_1`` (Eq. 21) on the labeled index set ``idx``."""
        l0 = F.cross_entropy(logits_hcre[idx], y[idx], weight=weight)
        if self.gamma == 0.0 or not self.use_hcre:
            return l0, l0, logits_hcre.new_zeros(())
        l1 = F.cross_entropy(logits_scnaf[idx], y[idx], weight=weight)
        return l0 + self.gamma * l1, l0, l1
