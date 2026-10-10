"""
ImGSL / IGSL (Ren et al., Machine Learning 2024): Improving fraud detection via
imbalanced graph structure learning.

Implementation from the paper (reference/ImGSL.md; no official code is shipped),
built on its two stated foundations, PC-GNN (sampler, label predictor, mini-batch
multi-relation aggregation) and IDGL (anchor-based iterative graph learning):

- **multi-relational class-balanced sampler** (Eqs. 2-3, in the training script):
  under every relation, train nodes are drawn with probability
  ``D_{r,v} / LF(y_v)`` (relation degree over class frequency);
- **graph-independent embedding** (Eqs. 4-9): a linear label predictor gives soft
  assignments ``f``. Edge ``(i, j)`` is pruned when ``S_ij = f_i . f_j < tau_minus``
  or when both endpoints are train nodes with different labels. During training
  every fraud centre also links its ``k_r`` most similar train fraud nodes, where
  ``k_r`` is half the average fraud degree under relation ``r``;
- **iterative graph structure learning** (Eqs. 10-15): per relation, a node-anchor
  graph ``K`` from a K-head weighted cosine thresholded at ``epsilon``, two-step
  anchor message passing ``Lambda^-1 K Delta^-1 K^T X``, and
  ``H_t = ReLU((beta * mean_{A~_r}(X) + (1 - beta) * msg_t) W_r)``. The first
  graph is learned from the raw features, later ones from ``H_{t-1}`` with a
  second metric. Iteration stops once ``||K_t - K_{t-1}||_F^2 / ||K_t||_F^2 < delta``
  or after ``max_iter`` steps;
- **embedding aggregation** (Eqs. 16-18): ``ReLU([x_v, H_1, ..., H_R] W)`` and a
  linear classifier. Loss: ``CE + alpha * CE(label predictor)``.
"""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def build_csr(src, dst, n):
    """Symmetrised, de-duplicated adjacency with self-loops."""
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    loops = np.arange(n, dtype=np.int64)
    rows = np.concatenate([src, dst, loops])
    cols = np.concatenate([dst, src, loops])
    adj = sp.csr_matrix((np.ones(rows.size, dtype=np.float32), (rows, cols)), shape=(n, n))
    adj.sum_duplicates()
    adj.sort_indices()
    return adj.indptr.astype(np.int64), adj.indices.astype(np.int64)


def gather_neighbors(indptr, indices, nodes):
    """Flatten the neighbour lists of ``nodes``: per-edge centre position and neighbour id."""
    start = indptr[nodes]
    deg = indptr[nodes + 1] - start
    seg = torch.repeat_interleave(torch.arange(nodes.numel(), device=nodes.device), deg)
    seg_start = torch.cumsum(deg, 0) - deg
    offset = torch.arange(seg.numel(), device=nodes.device) - seg_start[seg]
    return seg, indices[start[seg] + offset]


@torch.no_grad()
def refine_edges(seg, nbr, centers, prob_of, known, tau_minus, n,
                 oversample_rows=None, train_pos=None, k=0, remove_hetero=True):
    """Graph-independent edge refinement of one relation (Eqs. 8-9).

    ``prob_of(ids)`` returns the soft assignments of node ids; ``known`` holds train
    labels and -1 elsewhere. Returns the kept / added ``(seg, nbr)`` pairs, de-duplicated,
    always including the self-loop of every centre.
    """
    c = centers[seg]
    score = (prob_of(c) * prob_of(nbr)).sum(dim=1)
    keep = score >= tau_minus
    if remove_hetero:
        kc, kn = known[c], known[nbr]
        keep &= ~((kc >= 0) & (kn >= 0) & (kc != kn))
    seg_r = torch.cat([seg[keep], torch.arange(centers.numel(), device=seg.device)])
    nbr_r = torch.cat([nbr[keep], centers])
    if oversample_rows is not None and oversample_rows.numel() > 0 and k > 0 and train_pos.numel() > 0:
        kk = min(int(k), train_pos.numel())
        sim = prob_of(centers[oversample_rows]) @ prob_of(train_pos).t()
        top = torch.topk(sim, kk, dim=1).indices
        seg_r = torch.cat([seg_r, oversample_rows.unsqueeze(1).expand_as(top).reshape(-1)])
        nbr_r = torch.cat([nbr_r, train_pos[top].reshape(-1)])
    key = torch.unique(seg_r * n + nbr_r)
    return key // n, key % n


class WeightedCosine(nn.Module):
    """K-head weighted cosine node-anchor similarity, thresholded at ``epsilon`` (Eqs. 12-13)."""

    def __init__(self, dim, heads, epsilon):
        super().__init__()
        self.epsilon = float(epsilon)
        self.weight = nn.Parameter(torch.empty(heads, dim))
        init.xavier_uniform_(self.weight)

    def forward(self, h, anchor_h):
        a = F.normalize(h.unsqueeze(0) * self.weight.unsqueeze(1), dim=-1)
        b = F.normalize(anchor_h.unsqueeze(0) * self.weight.unsqueeze(1), dim=-1)
        sim = torch.matmul(a, b.transpose(1, 2)).mean(dim=0)
        return torch.where(sim >= self.epsilon, sim, torch.zeros_like(sim))


def anchor_message(k_mat, anchor_rows, x_anchor):
    """Two-step node-anchor message passing ``Lambda^-1 K Delta^-1 K^T X`` (Eq. 14).

    The anchor summaries ``Delta^-1 K^T X`` are averaged over the anchors themselves,
    a uniform node sample, as a Monte-Carlo estimate of the all-node sum.
    """
    k_aa = k_mat[anchor_rows]
    delta = k_aa.sum(dim=0, keepdim=True).t()
    summary = (k_aa.t() @ x_anchor) / delta.clamp(min=1e-12)
    lam = k_mat.sum(dim=1, keepdim=True)
    return (k_mat @ summary) / lam.clamp(min=1e-12)


class ImGSL(nn.Module):
    """One-layer ImGSL over ``R`` relations."""

    def __init__(self, in_dim, embed_dim, n_relations, n_class=2, beta=0.5, epsilon=0.0,
                 heads=4, tau_minus=0.2, max_iter=3, delta=1e-2, remove_hetero=True):
        super().__init__()
        self.n_relations = n_relations
        self.remove_hetero = bool(remove_hetero)
        self.beta = float(beta)
        self.tau_minus = float(tau_minus)
        self.max_iter = int(max_iter)
        self.delta = float(delta)
        self.label_clf = nn.Linear(in_dim, n_class)
        self.intra_weights = nn.ParameterList()
        self.feat_metric = nn.ModuleList()
        self.emb_metric = nn.ModuleList()
        for _ in range(n_relations):
            w = nn.Parameter(torch.empty(in_dim, embed_dim))
            init.xavier_uniform_(w)
            self.intra_weights.append(w)
            self.feat_metric.append(WeightedCosine(in_dim, heads, epsilon))
            self.emb_metric.append(WeightedCosine(embed_dim, heads, epsilon))
        self.weight = nn.Parameter(torch.empty(in_dim + n_relations * embed_dim, embed_dim))
        init.xavier_uniform_(self.weight)
        self.clf = nn.Linear(embed_dim, n_class)
        self.last_iterations = []

    def _relation_embedding(self, r, x_c, mean_c, anchor_rows):
        h_prev, k_prev = x_c, None
        x_anchor = x_c[anchor_rows]
        for t in range(self.max_iter):
            metric = self.feat_metric[r] if t == 0 else self.emb_metric[r]
            k_mat = metric(h_prev, h_prev[anchor_rows])
            msg = anchor_message(k_mat, anchor_rows, x_anchor)
            h = F.relu((self.beta * mean_c + (1 - self.beta) * msg) @ self.intra_weights[r])
            h_prev = h
            if k_prev is not None:
                with torch.no_grad():
                    change = (k_mat - k_prev).pow(2).sum() / k_mat.pow(2).sum().clamp(min=1e-12)
                if float(change) < self.delta:
                    return h, t + 1
            k_prev = k_mat
        return h_prev, self.max_iter

    def forward(self, x, batch, anchors, relations, known, train_pos=None, ks=None):
        """Logits and label-predictor logits of ``batch``.

        ``anchors``: node ids of the anchor set (also embedded as centres).
        With ``train_pos`` and ``ks`` (training), fraud centres oversample their
        ``ks[r]`` most similar train fraud nodes.
        """
        b, n = batch.numel(), x.size(0)
        centers = torch.cat([batch, anchors])
        anchor_rows = torch.arange(b, centers.numel(), device=x.device)
        gathered = [gather_neighbors(indptr, indices, centers) for indptr, indices in relations]
        oversample = train_pos is not None and ks is not None
        with torch.no_grad():
            ids = torch.cat([centers] + [nbr for _seg, nbr in gathered]
                            + ([train_pos] if oversample else []))
            uniq = torch.unique(ids)
            prob_uniq = torch.softmax(self.label_clf(x[uniq]), dim=-1)

            def prob_of(nodes):
                return prob_uniq[torch.searchsorted(uniq, nodes)]

            fraud_rows = (known[centers] == 1).nonzero(as_tuple=True)[0] if oversample else None

        x_c = x[centers]
        hs = [x[batch]]
        self.last_iterations = []
        for r, (seg, nbr) in enumerate(gathered):
            seg_r, nbr_r = refine_edges(seg, nbr, centers, prob_of, known, self.tau_minus, n,
                                        oversample_rows=fraud_rows, train_pos=train_pos,
                                        k=ks[r] if oversample else 0, remove_hetero=self.remove_hetero)
            cnt = torch.bincount(seg_r, minlength=centers.numel()).clamp(min=1).to(x.dtype)
            mean_c = x.new_zeros(centers.numel(), x.size(1)).index_add_(0, seg_r, x[nbr_r]) / cnt.unsqueeze(1)
            h_r, iters = self._relation_embedding(r, x_c, mean_c, anchor_rows)
            self.last_iterations.append(iters)
            hs.append(h_r[:b])
        h = F.relu(torch.cat(hs, dim=1) @ self.weight)
        return self.clf(h), self.label_clf(x[batch])
