'''
PMP: Partitioning Message Passing for Graph Fraud Detection (Zhuo et al., ICLR 2024).

Pure-PyTorch re-implementation (no DGL) of the paper's model, following
paper/papers-to-read/PMP.pdf; details the paper leaves open follow the official
code (https://github.com/Xtra-Computing/PMP, ``LA-SAGE-S``).

One PMP layer for centre node v_i (paper Eq. 3-5, Algorithm 1):

    h_i' = ReLU( f_self(h_i)
                 + sum_{j in N_fr(i)}  h_j W_fr,i
                 + sum_{j in N_be(i)}  h_j W_be,i
                 + sum_{j unlabelled}  h_j (alpha_i W_fr,i + (1 - alpha_i) W_be,i) )

* neighbours are partitioned by their *training* label (fraud / benign); every
  other neighbour (validation, test, unlabelled) is "unlabelled" (Eq. 3);
* message fusion is a sum (Sec. 3);
* alpha_i = sigmoid(Phi(h_i)), Phi a single-layer MLP (Eq. 4);
* root-specific weights W_fr,i = Psi_fr(h_i) = MLP(diag(h_i)) (Eq. 5), implemented as
  in the official ``LILinear``: h_j W_fr,i = (h_j * g_fr(h_i)) W_fr with a linear g_fr;
* multi-relation graphs: the layer runs on every relation (shared parameters, as in
  the official code), the R outputs are concatenated and mixed by a linear layer
  (Sec. 3, "concatenation pooling followed by an MLP");
* output head: dropout, then a linear layer to 2 classes (official ``proj`` Conv1d).

With one layer (the paper's setting on every dataset, App. D.4-D.5) the partitioned
neighbour sums depend only on the input features and the training labels, so the
training script precomputes them once and trains on mini-batches of nodes without
touching the graph again. Deeper models propagate over the full graph.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class PMPConv(nn.Module):
    """One partitioning message-passing layer (shared across relations)."""

    def __init__(self, in_dim, out_dim):
        super(PMPConv, self).__init__()
        self.fc_self = nn.Linear(in_dim, out_dim)
        self.fc_fr = nn.Linear(in_dim, out_dim)     # W_fr
        self.fc_be = nn.Linear(in_dim, out_dim)     # W_be
        self.gen_fr = nn.Linear(in_dim, in_dim)     # Psi_fr: diag(g_fr(h_i)) W_fr
        self.gen_be = nn.Linear(in_dim, in_dim)     # Psi_be
        self.balance = nn.Linear(in_dim, 1)         # Phi (alpha_i)
        nn.init.xavier_uniform_(self.fc_self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, h_self, s_fr, s_be, s_unk):
        """``h_self``: centre-node input; ``s_*``: summed fraud / benign / unlabelled neighbours."""
        g_fr = self.gen_fr(h_self)
        g_be = self.gen_be(h_self)
        alpha = torch.sigmoid(self.balance(h_self))
        agg = (self.fc_fr(s_fr * g_fr) + self.fc_be(s_be * g_be)
               + alpha * self.fc_fr(s_unk * g_fr) + (1.0 - alpha) * self.fc_be(s_unk * g_be))
        return F.relu(self.fc_self(h_self) + agg)


class PMP(nn.Module):
    def __init__(self, in_dim, hid_dim, num_relations, num_layers=1, dropout=0.0, num_classes=2):
        super(PMP, self).__init__()
        if num_layers < 1:
            raise ValueError('num_layers must be >= 1')
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList(
            [PMPConv(in_dim if l == 0 else hid_dim, hid_dim) for l in range(num_layers)])
        self.relation_mlp = nn.ModuleList(
            [nn.Linear(hid_dim * num_relations, hid_dim) for _ in range(num_layers)])
        self.out = nn.Linear(hid_dim, num_classes)

    def layer(self, l, h_self, neigh):
        """``neigh``: per relation, the tuple (s_fr, s_be, s_unk) for the same nodes as ``h_self``."""
        h = torch.cat([self.convs[l](h_self, *rel) for rel in neigh], dim=-1)
        h = self.relation_mlp[l](h)
        if l < self.num_layers - 1:
            h = F.relu(h)
        return h

    def head(self, h):
        h = F.dropout(h, self.dropout, training=self.training)
        return self.out(h)

    def forward_precomputed(self, x, neigh):
        """One-layer model on precomputed partitioned neighbour sums (mini-batch friendly)."""
        if self.num_layers != 1:
            raise RuntimeError('forward_precomputed is only valid for num_layers == 1')
        return self.head(self.layer(0, x, neigh))

    def forward_full(self, x, adjs, masks):
        """Full-graph forward for any depth.

        ``adjs``: one sparse (N x N) tensor per relation with A[dst, src] = 1;
        ``masks``: (m_fr, m_be, m_unk), each an (N, 1) float tensor.
        """
        h = x
        for l in range(self.num_layers):
            neigh = [tuple(torch.sparse.mm(adj, h * m) for m in masks) for adj in adjs]
            h = self.layer(l, h, neigh)
        return self.head(h)
