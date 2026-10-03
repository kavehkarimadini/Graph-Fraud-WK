"""
SparseGAD (Gong et al., IJCAI 2023): beyond homophily, robust graph anomaly
detection via neural sparsification.

Port of the official DGL implementation (reference/SparseGAD-main, model
``gpr_att`` run by ``main.py``) with ``torch`` gather / scatter message passing:

- input projection, then a GPR-style GCN (``GPR_sparse``): hop outputs of
  ``relu(A_w (H W_l))`` are summed with learnable coefficients initialised to
  ``0.1 * 0.9**k``;
- an extractor MLP scores every edge with the cosine similarity of its
  endpoint embeddings; the GCN is run again with edge weights multiplied by
  this (signed) attention, so dissimilar edges are down-weighted or flipped;
- periodically the graph is augmented with a cosine kNN graph built from the
  extractor embeddings (``top_k_graph_based_on_edge_attn``) and re-normalised.
"""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_union_graph(relations, n):
    """Weighted union graph: one unit per relation an edge appears in, plus one self-loop
    per node (``dgl.to_homogeneous`` + ``add_self_loop`` + ``g.adj()``).

    Returns COO ``(src, dst, weight)`` numpy arrays.
    """
    adj = sp.identity(n, dtype=np.float32, format='csr')
    for src, dst in relations:
        src = np.asarray(src, dtype=np.int64)
        dst = np.asarray(dst, dtype=np.int64)
        rel = sp.csr_matrix((np.ones(2 * src.size, dtype=np.float32),
                             (np.concatenate([src, dst]), np.concatenate([dst, src]))), shape=(n, n))
        rel.sum_duplicates()
        rel.data[:] = 1.0
        adj = adj + rel
    adj = adj.tocoo()
    return adj.row.astype(np.int64), adj.col.astype(np.int64), adj.data.astype(np.float32)


def sym_normalize(src, dst, weight, n):
    """``w_uv / sqrt(d_u d_v)`` with ``d`` the weighted row (source) degree (reference ``normalize``)."""
    deg = torch.zeros(n, dtype=weight.dtype, device=weight.device).index_add_(0, src, weight)
    inv = deg.pow(-0.5)
    inv[torch.isinf(inv)] = 0
    return weight * inv[src] * inv[dst]


@torch.no_grad()
def knn_edges(emb, k, chunk=None):
    """Directed cosine kNN edges ``neighbour -> node`` (self included), brute force in chunks."""
    n = emb.size(0)
    k = min(k, n)
    z = F.normalize(emb, dim=1)
    if chunk is None:
        chunk = max(1, (1 << 26) // max(n, 1))
    nbrs = []
    for i in range(0, n, chunk):
        nbrs.append(torch.topk(z[i:i + chunk] @ z.t(), k, dim=1).indices)
    nbr = torch.cat(nbrs, dim=0)
    node = torch.arange(n, device=emb.device).unsqueeze(1).expand_as(nbr)
    return nbr.reshape(-1), node.reshape(-1)


def augment_graph(src, dst, new_src, new_dst, n):
    """Union of the current edges and the kNN edges, binary and de-duplicated
    (``dgl.to_simple``), then symmetrically normalised."""
    key = torch.unique(torch.cat([src * n + dst, new_src * n + new_dst]))
    src, dst = key // n, key % n
    weight = sym_normalize(src, dst, torch.ones(src.numel(), device=src.device), n)
    return src, dst, weight


def propagate(x, src, dst, weight):
    """``out[v] = sum_{(u, v)} w_uv * x[u]`` (DGL ``u_mul_e`` + ``sum``)."""
    return torch.zeros_like(x).index_add_(0, dst, x[src] * weight.unsqueeze(1))


class GPRSparse(nn.Module):
    def __init__(self, hidden, num_layers, dropout, dropout_adj, alpha=0.1):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(num_layers)])
        temp = alpha * (1 - alpha) ** np.arange(num_layers + 1)
        temp[-1] = (1 - alpha) ** num_layers
        self.temp = nn.Parameter(torch.tensor(temp, dtype=torch.float32))
        self.dropout = dropout
        self.dropout_adj = dropout_adj

    def forward(self, x, src, dst, weight):
        """Returns the GPR output and the (edge-dropped) weights it used."""
        weight = F.dropout(weight, p=self.dropout_adj, training=self.training)
        hidden = x * self.temp[0]
        for i, lin in enumerate(self.layers):
            x = F.relu(propagate(lin(x), src, dst, weight))
            x = F.dropout(x, p=self.dropout, training=self.training)
            hidden = hidden + x * self.temp[i + 1]
        return hidden, weight


class ExtractorMLP(nn.Module):
    def __init__(self, hidden, dropout=0.2):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Dropout(p=dropout), nn.ReLU(), nn.Linear(hidden, hidden))
        for m in self.feature_extractor:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, emb, src, dst):
        z = self.feature_extractor(emb)
        return F.cosine_similarity(z[src], z[dst], dim=1)


class SparseGAD(nn.Module):
    """``GPR_ATT``: GPR GCN with cosine edge attention from an extractor MLP."""

    def __init__(self, in_dim, hidden, n_class=2, num_layers=2, dropout=0.5, dropout_adj=0.25):
        super().__init__()
        self.inlinear = nn.Linear(in_dim, hidden)
        self.outlinear = nn.Linear(hidden, n_class)
        nn.init.xavier_uniform_(self.inlinear.weight)
        nn.init.xavier_uniform_(self.outlinear.weight)
        self.gnn = GPRSparse(hidden, num_layers, dropout, dropout_adj)
        self.extractor = ExtractorMLP(hidden)

    def forward(self, x, src, dst, weight):
        h = self.inlinear(x)
        h_gnn, weight = self.gnn(h, src, dst, weight)
        attn = self.extractor(h_gnn, src, dst)
        h_gnn, _ = self.gnn(h, src, dst, weight * attn)
        return self.outlinear(h_gnn)

    def node_embedding(self, x, src, dst, weight):
        """Extractor embeddings used to build the kNN graph (``gen_node_emb``)."""
        h_gnn, _ = self.gnn(self.inlinear(x), src, dst, weight)
        return self.extractor.feature_extractor(h_gnn)
