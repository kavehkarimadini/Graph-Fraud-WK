'''
SEC-GFD: "Revisiting Graph-based Fraud Detection in Sight of Heterophily and Spectrum"
(Xu et al., AAAI 2024).

Pure-PyTorch re-implementation (no DGL) of the released code in
reference/SEC-GFD-main.zip (``model/SECGFD.py``, ``utils.py``):

* hybrid filtering module: two-layer MLP encoder, then the Beta-wavelet band-pass
  filters W_0 .. W_C of order C (as in BWGNN) plus ``high_order`` high-pass filters
  L, L^2, ... with L = I - D^-1/2 A D^-1/2; the filter outputs are concatenated and
  classified by a two-layer MLP;
* local environmental constraint: a 2-layer GCN (graph without self-loops,
  symmetric normalisation) maps the input features to an embedding of the same size;
  the loss pushes the cosine similarity between a node's features and its GCN
  embedding up for benign training nodes and down for fraud training nodes.

Filters are evaluated with one shared sequence of Laplacian powers instead of
recomputing them inside every band, which gives identical outputs at a fraction of
the cost.
'''
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from algorithms.BWGNN.BWGNN import calculate_theta2  # noqa: E402  (Beta-wavelet coefficients)


class GraphConv(nn.Module):
    """DGL ``GraphConv`` (norm='both'): A_hat (X W) + b, xavier weight, zero bias."""

    def __init__(self, in_dim, out_dim):
        super(GraphConv, self).__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, norm_adj, x):
        return torch.sparse.mm(norm_adj, x @ self.weight) + self.bias


class SECGFD(nn.Module):
    def __init__(self, in_dim, hid_dim, num_classes=2, order=2, high_order=2):
        super(SECGFD, self).__init__()
        self.thetas = calculate_theta2(d=order)
        self.order = order
        self.high_order = high_order
        n_filters = len(self.thetas) + high_order
        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, hid_dim)
        self.linear3 = nn.Linear(hid_dim * n_filters, hid_dim)
        self.linear4 = nn.Linear(hid_dim, num_classes)
        self.gcn1 = GraphConv(in_dim, hid_dim)
        self.gcn2 = GraphConv(hid_dim, in_dim)

    @staticmethod
    def _laplacian(adj, d_invsqrt, h):
        """L h = h - D^-1/2 A D^-1/2 h (in-degrees clamped at 1, as in the released code)."""
        return h - torch.sparse.mm(adj, h * d_invsqrt) * d_invsqrt

    def forward(self, x, adj, d_invsqrt, gcn_adj):
        """Return (logits, environment embedding).

        ``adj``: sparse A[dst, src] of the filtering graph; ``d_invsqrt``: (N, 1) in-degree^-1/2;
        ``gcn_adj``: normalised adjacency of the graph without self-loops.
        """
        h = F.relu(self.linear1(x))
        h = F.relu(self.linear2(h))
        powers = [h]
        for _ in range(max(self.order, self.high_order)):
            powers.append(self._laplacian(adj, d_invsqrt, powers[-1]))
        bands = [sum(theta[k] * powers[k] for k in range(len(theta))) for theta in self.thetas]
        highs = [powers[j + 1] for j in range(self.high_order)]
        h = F.relu(self.linear3(torch.cat(bands + highs, dim=-1)))
        logits = self.linear4(h)
        emb = self.gcn2(gcn_adj, F.relu(self.gcn1(gcn_adj, x)))
        return logits, emb


def environment_loss(features, emb, labels, train_idx):
    """Local environmental constraint: -log( mean exp(cos) benign / mean exp(cos) fraud ).

    Training nodes are indexed by their node ids. The released ``cal_nceloss`` indexes
    by position inside ``idx_train`` (a bug) and averages raw cosines, whose ratio can
    be negative; the paper's equation averages exp(cos), which is used here (as in the
    repo's earlier SEC-GFD port).
    """
    sim = F.cosine_similarity(features, emb, dim=1)
    y = labels[train_idx]
    normal = train_idx[y == 0]
    fraud = train_idx[y == 1]
    if normal.numel() == 0 or fraud.numel() == 0:
        return sim.new_zeros(())
    return -torch.log(torch.exp(sim[normal]).mean() / torch.exp(sim[fraud]).mean())
