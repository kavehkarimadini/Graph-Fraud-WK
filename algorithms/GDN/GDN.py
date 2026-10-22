"""
GDN (Gao et al., WSDM 2023): Alleviating Structural Distribution Shift in Graph
Anomaly Detection.

Port of the official implementation (reference/wsdm_GDN-main: ``models/model.py``,
``models/layers.py``, ``model_handler.py``) with vectorised gather / scatter ops
instead of per-node Python sets:

- **backbone:** a trainable node-feature table ``X``; per relation
  ``relu([x_v, mean(x_N(v))] W_r)`` (neighbours include ``v``), then
  ``relu([x_v, h_1, ..., h_R] W)`` and a linear classifier, trained with CE;
- **feature separation:** the ``topk`` feature dimensions with the largest mean
  absolute CE gradient on ``X`` are the anomaly features, the rest the normal
  features;
- **anomaly-feature constraint (fl):** on the anomaly dimensions, train fraud
  nodes are pulled towards the fraud prototype and pushed from the normal one,
  ``beta * exp(KL(p_pos || x) - KL(p_neg || x))``; prototypes are
  temperature-softmax weighted means of the train fraud / normal features;
- **normal-feature constraint (fn):** on the normal dimensions, each fraud node
  in the batch is pulled towards its neighbourhood mean and pushed from a random
  non-neighbour in every relation.

Both constraints only update ``X``.
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


def _kl(log_x, target):
    """``nn.KLDivLoss(reduction='batchmean')``."""
    return F.kl_div(log_x, target, reduction='batchmean')


class GDN(nn.Module):
    """GDN layer over ``R`` relations (``GDNLayer`` + ``InterAgg`` + ``IntraAgg``)."""

    def __init__(self, features, embed_dim, n_relations, train_pos, train_neg, n_class=2, temperature=5.0):
        super().__init__()
        features = torch.as_tensor(features, dtype=torch.float32)
        n, in_dim = features.shape
        self.n_relations = n_relations
        self.temperature = float(temperature)
        self.features = nn.Embedding(n, in_dim)
        self.features.weight = nn.Parameter(features.clone())
        self.intra_weights = nn.ParameterList()
        for _ in range(n_relations):
            w = nn.Parameter(torch.empty(2 * in_dim, embed_dim))
            init.xavier_uniform_(w)
            self.intra_weights.append(w)
        self.weight = nn.Parameter(torch.empty(embed_dim * n_relations + in_dim, embed_dim))
        init.xavier_uniform_(self.weight)
        self.clf = nn.Parameter(torch.empty(n_class, embed_dim))
        init.xavier_uniform_(self.clf)
        self.register_buffer('train_pos', torch.as_tensor(train_pos, dtype=torch.long))
        self.register_buffer('train_neg', torch.as_tensor(train_neg, dtype=torch.long))
        self.register_buffer('pos_vector', torch.zeros(1, in_dim))
        self.register_buffer('neg_vector', torch.zeros(1, in_dim))
        self.register_buffer('prototype_ready', torch.zeros((), dtype=torch.bool))

    def forward(self, batch, relations):
        """Logits of ``batch``, plus the per-relation neighbour means and gathers (for ``fn_loss``)."""
        x = self.features.weight
        b = batch.numel()
        self_feats = x[batch]
        hs, aggs, gathered = [self_feats], [], []
        for r, (indptr, indices) in enumerate(relations):
            seg, nbr, deg = gather_neighbors(indptr, indices, batch)
            agg = x.new_zeros(b, x.size(1)).index_add_(0, seg, x[nbr]) / deg.clamp(min=1).unsqueeze(1).to(x.dtype)
            aggs.append(agg)
            gathered.append((seg, nbr))
            hs.append(F.relu(torch.cat([self_feats, agg], dim=1) @ self.intra_weights[r]))
        h = F.relu(torch.cat(hs, dim=1) @ self.weight)
        return h @ self.clf.t(), aggs, gathered

    @torch.no_grad()
    def update_prototypes(self):
        """Reference ``update_label_vector``: mean at first, then temperature-softmax weighted means."""
        x = self.features.weight
        for idx, name in ((self.train_pos, 'pos_vector'), (self.train_neg, 'neg_vector')):
            feats = x[idx]
            if not bool(self.prototype_ready):
                new = feats.mean(dim=0, keepdim=True)
            else:
                cos = F.cosine_similarity(getattr(self, name), feats, dim=1, eps=1e-6)
                new = torch.softmax(cos / self.temperature, dim=0).unsqueeze(0) @ feats
            getattr(self, name).copy_(new)
        self.prototype_ready.fill_(True)

    def fl_loss(self, grads_idx):
        """KL of train fraud nodes to the fraud / normal prototypes on the anomaly dimensions."""
        log_x = F.log_softmax(self.features.weight[self.train_pos][:, grads_idx], dim=-1)
        target_pos = self.pos_vector[:, grads_idx].softmax(dim=-1).expand_as(log_x)
        target_neg = self.neg_vector[:, grads_idx].softmax(dim=-1).expand_as(log_x)
        return _kl(log_x, target_pos), _kl(log_x, target_neg)

    def fn_loss(self, batch, target_rows, aggs, negatives, non_grad_idx):
        """KL of batch fraud nodes to their neighbourhood means (pos) and to random
        non-neighbours (neg) on the normal dimensions, summed over relations.

        ``target_rows``: positions of the fraud nodes in ``batch``; ``negatives``:
        one tensor of sampled node ids per relation (see ``sample_negatives``).
        """
        x = self.features.weight
        log_x = F.log_softmax(x[batch[target_rows]][:, non_grad_idx], dim=-1)
        loss_pos = log_x.new_zeros(())
        loss_neg = log_x.new_zeros(())
        for agg, neg in zip(aggs, negatives):
            loss_pos = loss_pos + _kl(log_x, agg[target_rows][:, non_grad_idx].softmax(dim=-1))
            loss_neg = loss_neg + _kl(log_x, x[neg][:, non_grad_idx].softmax(dim=-1))
        return loss_pos, loss_neg


@torch.no_grad()
def sample_negatives(batch, target_rows, gathered, n, max_rounds=100):
    """For every relation, one node per target drawn uniformly from the nodes seen in
    the batch (batch nodes and all their neighbours) minus the target's neighbours in
    any relation (reference ``random.choice(all_nodes - all_neighs[i])``)."""
    m = target_rows.numel()
    device = batch.device
    pos_of = torch.full((batch.numel(),), -1, dtype=torch.long, device=device)
    pos_of[target_rows] = torch.arange(m, device=device)
    candidates = torch.unique(torch.cat([batch] + [nbr for _seg, nbr in gathered]))
    keys = []
    for seg, nbr in gathered:
        t = pos_of[seg]
        keep = t >= 0
        keys.append(t[keep] * n + nbr[keep])
    excluded = torch.unique(torch.cat(keys)) if keys else torch.zeros(0, dtype=torch.long, device=device)
    rows = torch.arange(m, device=device)
    out = []
    for _ in gathered:
        neg = candidates[torch.randint(candidates.numel(), (m,), device=device)]
        for _round in range(max_rounds):
            bad = torch.isin(rows * n + neg, excluded)
            if not bool(bad.any()):
                break
            neg[bad] = candidates[torch.randint(candidates.numel(), (int(bad.sum()),), device=device)]
        out.append(neg)
    return out
