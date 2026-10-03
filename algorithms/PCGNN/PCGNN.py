"""
PC-GNN (Liu et al., WWW 2021): Pick and Choose, a GNN-based imbalanced
learning approach for fraud detection.

Port of the official implementation (reference/PC-GNN-main) with vectorised
gather / scatter ops instead of per-node Python loops:

- Pick: each epoch draws a label-balanced node sample with probability
  ``degree / |class|`` (``pick_step``, done in the training script);
- Choose: under every relation a linear label predictor scores the nodes and
  each centre keeps its ``ceil(deg * p)`` neighbours closest in score (all of
  them when ``deg <= keep + 1``); in training, fraud centres additionally get
  the ``int(keep * rho)`` most similar train fraud nodes as neighbours;
- Aggregate: ``relu([x_v, mean(x_N)] W_r)`` per relation, then
  ``relu([x_v, h_1, ..., h_R] W)`` and a linear classifier.

Loss: ``CE(gnn) + alpha * CE(label predictor)`` on the centre nodes.
"""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def build_csr(src, dst, n):
    """Symmetrised, de-duplicated adjacency with self-loops (reference ``sparse_to_adjlist``)."""
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
    """Flatten the neighbour lists of ``nodes``: per-edge centre position, neighbour id, and degrees."""
    start = indptr[nodes]
    deg = indptr[nodes + 1] - start
    seg = torch.repeat_interleave(torch.arange(nodes.numel(), device=nodes.device), deg)
    seg_start = torch.cumsum(deg, 0) - deg
    offset = torch.arange(seg.numel(), device=nodes.device) - seg_start[seg]
    return seg, indices[start[seg] + offset], deg


def choose_neighbors(center_score, nbr_score, seg, deg, threshold):
    """Top-p neighbour selection by label-aware L1 distance (``choose_step_test``).

    Returns a keep-mask over edges and the number of neighbours ``ceil(deg * p)``.
    """
    dist = (center_score[seg] - nbr_score).abs()
    order = torch.argsort(dist, stable=True)
    order = order[torch.argsort(seg[order], stable=True)]
    seg_start = torch.cumsum(deg, 0) - deg
    rank = torch.empty_like(order)
    rank[order] = torch.arange(seg.numel(), device=seg.device) - seg_start[seg[order]]
    num_keep = torch.ceil(deg.to(torch.float64) * float(threshold)).to(torch.long)
    keep = (deg <= num_keep + 1)[seg] | (rank < num_keep[seg])
    return keep, num_keep


def oversample_minority(center_score, minor_score, minor_nodes, centers, k):
    """The ``k[i]`` train fraud nodes closest in label-aware score to each fraud centre.

    Returns ``(seg, nbr)``: positions in ``centers`` and the chosen node ids.
    """
    k = k.clamp(max=minor_nodes.numel())
    if centers.numel() == 0 or int(k.max()) <= 0:
        empty = centers.new_empty(0)
        return empty, empty
    dist = (center_score[centers].unsqueeze(1) - minor_score.unsqueeze(0)).abs()
    idx = torch.topk(dist, int(k.max()), dim=1, largest=False).indices
    mask = torch.arange(idx.size(1), device=idx.device).unsqueeze(0) < k.unsqueeze(1)
    seg = centers.unsqueeze(1).expand_as(idx)[mask]
    return seg, minor_nodes[idx[mask]]


class PCGNN(nn.Module):
    """One Pick-Choose-Aggregate layer over ``R`` relations (``PCALayer`` + ``InterAgg``)."""

    def __init__(self, in_dim, embed_dim, n_relations, n_class=2, threshold=0.5, rho=0.5):
        super().__init__()
        self.n_relations = n_relations
        self.threshold = float(threshold)
        self.rho = float(rho)
        self.label_clf = nn.Linear(in_dim, n_class)
        self.intra_weights = nn.ParameterList()
        for _ in range(n_relations):
            w = nn.Parameter(torch.empty(2 * in_dim, embed_dim))
            init.xavier_uniform_(w)
            self.intra_weights.append(w)
        self.weight = nn.Parameter(torch.empty(embed_dim * n_relations + in_dim, embed_dim))
        init.xavier_uniform_(self.weight)
        self.clf = nn.Parameter(torch.empty(n_class, embed_dim))
        init.xavier_uniform_(self.clf)

    def forward(self, x, batch, relations, labels=None, train_pos=None):
        """Logits and label-predictor logits of ``batch``.

        With ``labels`` and ``train_pos`` (train mode), fraud centres also
        aggregate their most similar train fraud nodes.
        """
        b = batch.numel()
        center_logits = self.label_clf(x[batch])
        self_feats = x[batch]

        gathered = [gather_neighbors(indptr, indices, batch) for indptr, indices in relations]
        oversample = labels is not None and train_pos is not None and self.rho > 0
        with torch.no_grad():
            nodes = torch.cat([batch] + [nbr for _s, nbr, _d in gathered]
                              + ([train_pos] if oversample else []))
            uniq, inv = torch.unique(nodes, return_inverse=True)
            scores = self.label_clf(x[uniq])[:, 0][inv]
            center_score = scores[:b]
            offset = b
            nbr_scores = []
            for _seg, nbr, _deg in gathered:
                nbr_scores.append(scores[offset:offset + nbr.numel()])
                offset += nbr.numel()
            minor_score = scores[offset:] if oversample else None
            fraud_centers = (labels == 1).nonzero(as_tuple=True)[0] if oversample else None

        hs = [self_feats]
        for r, (seg, nbr, deg) in enumerate(gathered):
            with torch.no_grad():
                keep, num_keep = choose_neighbors(center_score, nbr_scores[r], seg, deg, self.threshold)
                seg_r, nbr_r = seg[keep], nbr[keep]
                if oversample:
                    k = (num_keep[fraud_centers].to(torch.float64) * self.rho).to(torch.long)
                    o_seg, o_nbr = oversample_minority(center_score, minor_score, train_pos, fraud_centers, k)
                    n = x.size(0)
                    key = torch.unique(torch.cat([seg_r * n + nbr_r, o_seg * n + o_nbr]))
                    seg_r, nbr_r = key // n, key % n
                cnt = torch.bincount(seg_r, minlength=b).clamp(min=1).to(x.dtype)
            agg = x.new_zeros(b, x.size(1)).index_add_(0, seg_r, x[nbr_r]) / cnt.unsqueeze(1)
            hs.append(F.relu(torch.cat([self_feats, agg], dim=1) @ self.intra_weights[r]))
        h = F.relu(torch.cat(hs, dim=1) @ self.weight)
        return h @ self.clf.t(), center_logits
