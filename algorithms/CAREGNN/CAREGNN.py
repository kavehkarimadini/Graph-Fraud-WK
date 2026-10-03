"""
CARE-GNN (Dou et al., CIKM 2020): camouflage-resistant multi-relation GNN.

Port of the official implementation (reference/CARE-GNN-master). The three
modules are kept as in the paper / code:

- label-aware similarity: a linear label predictor scores every node and the
  L1 distance between centre and neighbour scores ranks neighbours (Eq. 2);
- similarity-aware neighbour selector: top-p filtering per relation with
  thresholds p_r updated by the RL module from epoch-to-epoch changes in the
  average neighbour distance of fraud nodes (Eq. 5-6);
- relation-aware aggregator: intra-relation mean of the kept neighbours, then
  inter-relation sum weighted by p_r (Eq. 8-9). ``Mean`` / ``Weight`` / ``Att``
  variants follow the reference ``layers.py``.

Neighbour filtering is vectorised over CSR adjacency (no per-node Python loop)
but keeps the reference rule: keep the ceil(deg * p_r) closest neighbours, or
all of them when deg <= ceil(deg * p_r) + 1.
"""
import math

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def build_csr(src, dst, n):
    """Symmetrised, de-duplicated adjacency with self-loops (as ``sparse_to_adjlist``)."""
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    loops = np.arange(n, dtype=np.int64)
    rows = np.concatenate([src, dst, loops])
    cols = np.concatenate([dst, src, loops])
    adj = sp.csr_matrix((np.ones(rows.size, dtype=np.float32), (rows, cols)), shape=(n, n))
    adj.sum_duplicates()
    adj.sort_indices()
    return adj.indptr.astype(np.int64), adj.indices.astype(np.int64)


def gather_neighbors(indptr, indices, batch):
    """Flatten the neighbour lists of ``batch``.

    Returns ``seg`` (position of the centre node in ``batch`` for each edge),
    ``nbr`` (neighbour id per edge) and ``deg`` (neighbour count per centre).
    """
    start = indptr[batch]
    deg = indptr[batch + 1] - start
    seg = torch.repeat_interleave(torch.arange(batch.numel(), device=batch.device), deg)
    seg_start = torch.cumsum(deg, 0) - deg
    offset = torch.arange(seg.numel(), device=batch.device) - seg_start[seg]
    nbr = indices[start[seg] + offset]
    return seg, nbr, deg


def filter_neighbors(center_score, nbr_score, seg, deg, threshold):
    """Top-p neighbour selection by label-aware distance (Section 3.3.1).

    Returns a boolean keep-mask over edges and the per-edge distances.
    """
    dist = (center_score[seg] - nbr_score).abs()
    order = torch.argsort(dist, stable=True)
    order = order[torch.argsort(seg[order], stable=True)]
    seg_sorted = seg[order]
    seg_start = torch.cumsum(deg, 0) - deg
    rank_sorted = torch.arange(seg.numel(), device=seg.device) - seg_start[seg_sorted]
    rank = torch.empty_like(rank_sorted)
    rank[order] = rank_sorted

    num_keep = torch.ceil(deg.to(torch.float64) * float(threshold)).to(torch.long)
    keep_all = deg <= num_keep + 1
    keep = keep_all[seg] | (rank < num_keep[seg])
    return keep, dist


class CAREGNN(nn.Module):
    """One-layer CARE-GNN over ``R`` relations (``OneLayerCARE`` + ``InterAgg``)."""

    def __init__(self, in_dim, embed_dim, n_relations, n_class=2, inter='GNN',
                 step_size=0.02, init_threshold=0.5, att_dropout=0.6):
        super().__init__()
        if inter not in ('GNN', 'Mean', 'Weight', 'Att'):
            raise ValueError("inter must be one of GNN / Mean / Weight / Att")
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.n_relations = n_relations
        self.inter = inter
        self.step_size = float(step_size)
        self.att_dropout = att_dropout

        self.register_buffer('thresholds', torch.full((n_relations,), float(init_threshold)))

        self.label_clf = nn.Linear(in_dim, n_class)
        self.weight = nn.Parameter(torch.empty(in_dim, embed_dim))
        init.xavier_uniform_(self.weight)
        self.out = nn.Parameter(torch.empty(embed_dim, n_class))
        init.xavier_uniform_(self.out)
        if inter == 'Weight':
            self.alpha = nn.Parameter(torch.empty(embed_dim, n_relations))
            init.xavier_uniform_(self.alpha)
        if inter == 'Att':
            self.a = nn.Parameter(torch.empty(2 * embed_dim, 1))
            init.xavier_uniform_(self.a)
            self.leakyrelu = nn.LeakyReLU(0.2)

    def _intra(self, x, batch, csr, center_score, all_score, threshold):
        indptr, indices = csr
        seg, nbr, deg = gather_neighbors(indptr, indices, batch)
        keep, dist = filter_neighbors(center_score, all_score[nbr], seg, deg, threshold)
        seg_k = seg[keep]
        agg = x.new_zeros(batch.numel(), x.size(1))
        agg.index_add_(0, seg_k, x[nbr[keep]])
        cnt = torch.bincount(seg_k, minlength=batch.numel()).clamp(min=1).to(x.dtype)
        return F.relu(agg / cnt.unsqueeze(1)), seg_k, dist[keep]

    def _inter(self, center_h, neigh_h):
        if self.inter == 'GNN':
            agg = sum(neigh_h[r] * self.thresholds[r] for r in range(self.n_relations))
            return F.relu(center_h + agg)
        if self.inter == 'Mean':
            agg = sum(neigh_h)
            return F.relu((center_h + agg) / float(self.n_relations + 1))
        if self.inter == 'Weight':
            w = F.softmax(self.alpha, dim=1)
            agg = sum(neigh_h[r] * w[:, r] for r in range(self.n_relations))
            return F.relu(center_h + agg)
        e = torch.cat([self.leakyrelu(torch.cat([center_h, h], dim=1).mm(self.a))
                       for h in neigh_h], dim=1)
        att = F.dropout(F.softmax(e, dim=1), self.att_dropout, training=self.training)
        agg = sum(att[:, r:r + 1] * neigh_h[r] for r in range(self.n_relations))
        return F.relu(center_h + agg)

    def forward(self, x, batch, relations, labels=None):
        """Score ``batch`` nodes.

        Returns ``(gnn_logits, label_logits, relation_scores)``. When ``labels``
        is given, ``relation_scores[r]`` is the mean kept-neighbour distance of
        the fraud nodes in the batch (the RL signal), else ``None``.
        """
        with torch.no_grad():
            all_score = self.label_clf(x)[:, 0]
        label_logits = self.label_clf(x[batch])
        center_score = all_score[batch]

        pos = None if labels is None else labels == 1
        neigh_h, rel_scores = [], []
        for r, csr in enumerate(relations):
            feats, seg_k, dist_k = self._intra(
                x, batch, csr, center_score, all_score, float(self.thresholds[r]))
            neigh_h.append(feats.mm(self.weight))
            if pos is not None:
                pos_edges = pos[seg_k]
                rel_scores.append(float(dist_k[pos_edges].mean()) if pos_edges.any() else math.nan)

        center_h = x[batch].mm(self.weight)
        combined = self._inter(center_h, neigh_h)
        gnn_logits = combined.mm(self.out)
        return gnn_logits, label_logits, (rel_scores if pos is not None else None)

    @torch.no_grad()
    def rl_update(self, previous_scores, current_scores):
        """Eq. 5-6: +step if the epoch's average distance did not increase, else -step."""
        rewards = []
        for r in range(self.n_relations):
            prev, cur = previous_scores[r], current_scores[r]
            if math.isnan(prev) or math.isnan(cur):
                rewards.append(0)
                continue
            reward = 1 if prev - cur >= 0 else -1
            rewards.append(reward)
            new = float(self.thresholds[r]) + reward * self.step_size
            if new > 1:
                new = 0.999
            elif new < 0:
                new = 0.001
            self.thresholds[r] = new
        return rewards
