"""
FRAUDRE (Zhang et al., ICDM 2021): fraud detection dual-resistant to graph
inconsistency and imbalance.

Port of the official implementation (reference/FRAUDRE-main) with vectorised
gather / scatter ops instead of per-node Python sets:

- ego-feature embedding ``relu(W x)``;
- fraud-aware convolutions: under every relation a node takes the mean of its
  (sampled) neighbours and the difference ``self - mean``; relations are mixed
  with per-dimension softmax weights and the result is concatenated to the
  node's own representation (``d -> 3d -> 7d`` for two layers);
- a two-layer classifier trained with logits shifted by ``log(prior)``
  (imbalance-aware loss); inference uses the raw logits.

Layer ``k`` aggregates only the last ``d * 2**k`` columns of its input (the
part produced by the previous convolution), as in the reference ``InterAgg``.
"""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def sample_adjacency(src, dst, n, max_neighbors=10, seed=0):
    """Symmetrised adjacency with self-loops, each row cut to a fixed random
    sample of ``max_neighbors`` (reference ``sparse_to_adjlist``).

    Returns CSR ``(indptr, indices)`` as int64 numpy arrays.
    """
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    loops = np.arange(n, dtype=np.int64)
    rows = np.concatenate([src, dst, loops])
    cols = np.concatenate([dst, src, loops])
    adj = sp.csr_matrix((np.ones(rows.size, dtype=np.float32), (rows, cols)), shape=(n, n))
    adj.sum_duplicates()
    adj.sort_indices()
    indptr, indices = adj.indptr.astype(np.int64), adj.indices.astype(np.int64)
    if not max_neighbors or max_neighbors <= 0:
        return indptr, indices

    deg = np.diff(indptr)
    row = np.repeat(np.arange(n, dtype=np.int64), deg)
    key = np.random.default_rng(seed).random(indices.size)
    order = np.lexsort((key, row))
    rank = np.arange(indices.size, dtype=np.int64) - indptr[row[order]]
    keep = np.sort(order[rank < max_neighbors])
    new_deg = np.minimum(deg, max_neighbors)
    new_indptr = np.concatenate([[0], np.cumsum(new_deg)]).astype(np.int64)
    return new_indptr, indices[keep]


def gather_neighbors(indptr, indices, nodes):
    """Flatten the neighbour lists of ``nodes``: per-edge centre position, neighbour id, and degrees."""
    start = indptr[nodes]
    deg = indptr[nodes + 1] - start
    seg = torch.repeat_interleave(torch.arange(nodes.numel(), device=nodes.device), deg)
    seg_start = torch.cumsum(deg, 0) - deg
    offset = torch.arange(seg.numel(), device=nodes.device) - seg_start[seg]
    return seg, indices[start[seg] + offset], deg


class FRAUDRE(nn.Module):
    def __init__(self, in_dim, n_relations, embed_dim=64, n_layers=2, hidden_dim=64, n_class=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_relations = n_relations
        self.mlp = nn.Linear(in_dim, embed_dim)
        self.alphas = nn.ParameterList()
        for k in range(n_layers):
            alpha = nn.Parameter(torch.empty(2 * embed_dim * 2 ** k, n_relations))
            init.xavier_uniform_(alpha)
            self.alphas.append(alpha)
        out_dim = (2 ** (n_layers + 1) - 1) * embed_dim
        self.weight_model = nn.Parameter(torch.empty(out_dim, hidden_dim))
        self.weight_model2 = nn.Parameter(torch.empty(hidden_dim, n_class))
        init.xavier_uniform_(self.weight_model)
        init.xavier_uniform_(self.weight_model2)

    def embed(self, x, batch, relations):
        """Final ``(2**(K+1) - 1) * d`` embeddings of ``batch`` over its K-hop sampled neighbourhood."""
        node_sets = [batch]
        for _ in range(self.n_layers):
            cur = node_sets[-1]
            hop = [cur] + [gather_neighbors(indptr, indices, cur)[1] for indptr, indices in relations]
            node_sets.append(torch.unique(torch.cat(hop)))

        h = F.relu(self.mlp(x[node_sets[-1]]))
        width = self.embed_dim
        for k in range(self.n_layers):
            nodes_in = node_sets[self.n_layers - k]
            nodes_out = node_sets[self.n_layers - k - 1]
            self_h = h[torch.searchsorted(nodes_in, nodes_out)]
            src = h[:, -width:]
            center = self_h[:, -width:]
            neigh = []
            for indptr, indices in relations:
                seg, nbr, deg = gather_neighbors(indptr, indices, nodes_out)
                total = src.new_zeros(nodes_out.numel(), width).index_add_(
                    0, seg, src[torch.searchsorted(nodes_in, nbr)])
                mean = total / deg.clamp(min=1).unsqueeze(1).to(src.dtype)
                neigh.append(torch.cat([mean, center - mean], dim=1))
            w = torch.softmax(self.alphas[k], dim=1)
            agg = sum(w[:, r] * neigh[r] for r in range(self.n_relations))
            h = torch.cat([self_h, agg], dim=1)
            width *= 2
        return h

    def forward(self, x, batch, relations):
        h = self.embed(x, batch, relations)
        return F.leaky_relu(h @ self.weight_model, 0.3) @ self.weight_model2


def prior_adjusted_loss(logits, labels, log_prior):
    """Cross-entropy on ``logits + log(prior)`` (reference ``MODEL.loss``)."""
    return F.cross_entropy(logits + log_prior, labels)
