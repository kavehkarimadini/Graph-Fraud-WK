"""
COFRAUD (Zang et al., IJCAI 2023): correlation-aware fraud detection on
multi-relation graphs (alienation + marginalization / synergistic camouflage).

Implementation from the paper (reference/COFRAUD.md), with hyperparameters
filled from the authors' released training script where the paper is silent.
No DGL / PyG — CSR neighbour lists and ``index_add_`` message passing.

- **Intra-relation** (Eqs. 1-6): ``z_v = W_v h``, ``z_r = W_r h_r`` with
  ``dim(z_r)=2·dim(z_v)``; frequency attention
  ``α_ij = tanh(z_r^T [z_i ‖ z_j]) ∈ [-1,1]``; residual aggregation
  ``c = (1-λ) W_align h + λ Σ_j α_ij/√(d_i d_j) z_j``; relation MLP ``h'_r``.
- **Inter-relation** (Eqs. 7-8): for each target relation ``r2``,
  ``h'_{v,r2} = Σ_ri β_{r2,ri} c_{v,ri}`` with
  ``β = softmax(LeakyReLU(q_{r2}^T c_{v,ri}))``.
- **Fusion** (Eqs. 9-10): ``γ`` from
  ``LeakyReLU((V_r h'_{v,r})^T U_r h'_r)``, then
  ``h_final = Σ_r γ_r V_r h'_{v,r}``; linear classifier + CE.

Multi-head attention follows the official code (8×32); the paper equations are
written for a single head. Intra uses the paper's tanh path (the released
``RelationGraphConv`` uses LeakyReLU+softmax instead).
"""
import math

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_csr(src, dst, n):
    """Symmetrised, de-duplicated adjacency without self-loops."""
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    adj = sp.csr_matrix(
        (np.ones(rows.size, dtype=np.float32), (rows, cols)), shape=(n, n))
    adj.sum_duplicates()
    adj.sort_indices()
    return adj.indptr.astype(np.int64), adj.indices.astype(np.int64)


def gather_neighbors(indptr, indices, nodes):
    """Per-edge centre position and neighbour id for ``nodes``."""
    start = indptr[nodes]
    deg = indptr[nodes + 1] - start
    seg = torch.repeat_interleave(torch.arange(nodes.numel(), device=nodes.device), deg)
    seg_start = torch.cumsum(deg, 0) - deg
    offset = torch.arange(seg.numel(), device=nodes.device) - seg_start[seg]
    return seg, indices[start[seg] + offset], deg


def _xavier_(t, gain=1.0):
    nn.init.xavier_normal_(t, gain=gain)


class COFRAUDLayer(nn.Module):
    """One stacked intra + inter block (paper Sec. 3.2-3.3)."""

    def __init__(self, in_dim, hidden, n_relations, relation_in_dim, relation_hidden,
                 n_heads=8, dropout=0.5, negative_slope=0.2, residual=True):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.n_relations = n_relations
        self.n_heads = n_heads
        self.out_dim = n_heads * hidden
        self.residual = residual
        self.dropout = nn.Dropout(dropout)
        self.leaky = nn.LeakyReLU(negative_slope)

        # Shared node transform W_v (Eqs. 1-2); per-relation W_r (Eq. 3).
        self.W_v = nn.Parameter(torch.empty(in_dim, n_heads * hidden))
        self.W_r = nn.ParameterList([
            nn.Parameter(torch.empty(relation_in_dim, n_heads * 2 * hidden))
            for _ in range(n_relations)
        ])
        if residual:
            self.W_align = nn.Linear(in_dim, n_heads * hidden, bias=False)
            self.lambda_logit = nn.Parameter(torch.zeros(1))
        self.relation_mlp = nn.ModuleList([
            nn.Linear(relation_in_dim, n_heads * relation_hidden)
            for _ in range(n_relations)
        ])
        # Inter-relation query q_r2 (Eq. 8), one per target relation.
        self.q = nn.ParameterList([
            nn.Parameter(torch.empty(n_heads, hidden))
            for _ in range(n_relations)
        ])
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        _xavier_(self.W_v, gain)
        for w in self.W_r:
            _xavier_(w, gain)
        if self.residual:
            _xavier_(self.W_align.weight, gain)
        for mlp in self.relation_mlp:
            _xavier_(mlp.weight, gain)
            nn.init.zeros_(mlp.bias)
        for q in self.q:
            _xavier_(q, gain)

    def _intra(self, h, indptr, indices, deg_inv_sqrt, rel_emb, W_r):
        """Eqs. 1-5 for one relation. ``h`` is (N, in_dim)."""
        n = h.size(0)
        h_drop = self.dropout(h)
        z = (h_drop @ self.W_v).view(n, self.n_heads, self.hidden)
        # z_r -> (H, 2D); α = tanh(a_l·z_i + a_r·z_j)
        z_r = (rel_emb @ W_r).view(self.n_heads, 2 * self.hidden)
        a_l, a_r = z_r[:, :self.hidden], z_r[:, self.hidden:]

        nodes = torch.arange(n, device=h.device)
        seg, nbr, _deg = gather_neighbors(indptr, indices, nodes)
        # Isolated nodes contribute no messages.
        if seg.numel() == 0:
            agg = z.new_zeros(n, self.n_heads * self.hidden)
        else:
            e_dst = (z[seg] * a_l).sum(dim=-1)   # (E, H)
            e_src = (z[nbr] * a_r).sum(dim=-1)
            alpha = torch.tanh(e_dst + e_src)    # Eq. 4
            scale = (deg_inv_sqrt[seg] * deg_inv_sqrt[nbr]).unsqueeze(-1)  # (E, 1)
            coef = (alpha * scale).unsqueeze(-1)  # (E, H, 1)
            msg = coef * z[nbr]
            agg = z.new_zeros(n, self.n_heads, self.hidden).index_add_(0, seg, msg)
            agg = agg.reshape(n, self.n_heads * self.hidden)
            agg = F.relu(agg)

        if self.residual:
            lam = torch.sigmoid(self.lambda_logit)
            c = (1.0 - lam) * self.W_align(h) + lam * agg
        else:
            c = agg
        return c

    def forward(self, h_per_rel, rel_embs, csrs, deg_inv_sqrt):
        """
        ``h_per_rel``: list of (N, in_dim) — same features per relation on layer 0,
        relation-specific after inter on later layers.
        ``rel_embs``: list of (relation_in_dim,) vectors.
        Returns ``h'_per_rel`` (N, out_dim) and updated relation embeddings.
        """
        # Intra (Eqs. 1-5).
        c_list = []
        for r in range(self.n_relations):
            indptr, indices = csrs[r]
            c_list.append(self._intra(
                h_per_rel[r], indptr, indices, deg_inv_sqrt[r],
                rel_embs[r], self.W_r[r]))

        # Inter (Eqs. 7-8): pass information across relation-specific c's.
        if self.n_relations == 1:
            h_out = c_list
        else:
            stacked = torch.stack(c_list, dim=0)  # (R, N, H*D)
            stacked_h = stacked.view(
                self.n_relations, -1, self.n_heads, self.hidden)
            h_out = []
            for r2 in range(self.n_relations):
                # (R, N, H, 1)
                score = (stacked_h * self.q[r2]).sum(dim=-1, keepdim=True)
                beta = torch.softmax(self.leaky(score), dim=0)
                mixed = (stacked_h * beta).sum(dim=0)  # (N, H, D)
                mixed = self.dropout(mixed)
                h_out.append(mixed.reshape(-1, self.n_heads * self.hidden))

        # Relation MLP (Eq. 6).
        rel_out = [self.relation_mlp[r](rel_embs[r]) for r in range(self.n_relations)]
        return h_out, rel_out


class RelationFusing(nn.Module):
    """Eqs. 9-10: fuse relation-specific node + relation embeddings."""

    def __init__(self, hidden, relation_hidden, n_heads, dropout=0.5, negative_slope=0.2):
        super().__init__()
        self.hidden = hidden
        self.relation_hidden = relation_hidden
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.leaky = nn.LeakyReLU(negative_slope)

    def forward(self, node_feats, rel_embs, V, U):
        """
        ``node_feats``: list of (N, H*D); ``rel_embs``: list of (H*Rhid,);
        ``V`` / ``U``: lists of (H, D, D) / (H, Rhid, D).
        """
        if len(node_feats) == 1:
            return node_feats[0]

        n_rel = len(node_feats)
        # (R, N, H, D)
        nodes = torch.stack(node_feats, dim=0).view(
            n_rel, -1, self.n_heads, self.hidden)
        rels = torch.stack(rel_embs, dim=0).view(
            n_rel, self.n_heads, self.relation_hidden)
        V_t = torch.stack(list(V), dim=0)  # (R, H, D, D)
        U_t = torch.stack(list(U), dim=0)  # (R, H, Rhid, D)

        nodes = torch.einsum('rnhd,rhdf->rnhf', nodes, V_t)
        rels = torch.einsum('rhd,rhdf->rhf', rels, U_t)
        # γ: (R, N, H, 1)
        score = (nodes * rels.unsqueeze(1)).sum(dim=-1, keepdim=True)
        gamma = torch.softmax(self.leaky(score), dim=0)
        out = (nodes * gamma).sum(dim=0)
        out = self.dropout(out)
        return out.reshape(-1, self.n_heads * self.hidden)


class COFRAUD(nn.Module):
    def __init__(self, in_dim, n_relations, hidden=32, relation_dim=16,
                 relation_hidden=16, n_layers=2, n_heads=8, n_class=2,
                 dropout=0.5, negative_slope=0.2, residual=True,
                 use_inter=True, use_fusion=True):
        super().__init__()
        self.n_relations = n_relations
        self.hidden = hidden
        self.relation_hidden = relation_hidden
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.use_inter = use_inter
        self.use_fusion = use_fusion
        self.out_dim = n_heads * hidden

        # Learnable relation embeddings (paper: one-hot; official: free vectors).
        self.relation_embedding = nn.ParameterList([
            nn.Parameter(torch.empty(relation_dim))
            for _ in range(n_relations)
        ])
        self.projection = nn.Linear(in_dim, n_heads * hidden)

        layers = []
        # Layer 0: projected node dim; relation_dim input.
        layers.append(COFRAUDLayer(
            n_heads * hidden, hidden, n_relations, relation_dim, relation_hidden,
            n_heads=n_heads, dropout=dropout, negative_slope=negative_slope,
            residual=residual))
        for _ in range(1, n_layers):
            layers.append(COFRAUDLayer(
                n_heads * hidden, hidden, n_relations,
                n_heads * relation_hidden, relation_hidden,
                n_heads=n_heads, dropout=dropout, negative_slope=negative_slope,
                residual=residual))
        self.layers = nn.ModuleList(layers)

        # Fusion transforms V_r, U_r (Eqs. 9-10).
        self.V = nn.ParameterList([
            nn.Parameter(torch.empty(n_heads, hidden, hidden))
            for _ in range(n_relations)
        ])
        self.U = nn.ParameterList([
            nn.Parameter(torch.empty(n_heads, relation_hidden, hidden))
            for _ in range(n_relations)
        ])
        self.fusing = RelationFusing(
            hidden, relation_hidden, n_heads, dropout=dropout,
            negative_slope=negative_slope)
        self.classifier = nn.Linear(n_heads * hidden, n_class)
        self.reset_parameters()

        if not use_inter:
            # Skip inter by replacing each layer's forward inter with identity;
            # handled in ``encode`` via a flag on the layer call path.
            pass

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for emb in self.relation_embedding:
            nn.init.normal_(emb, std=1.0 / math.sqrt(emb.numel()))
        _xavier_(self.projection.weight, gain)
        nn.init.zeros_(self.projection.bias)
        for v in self.V:
            _xavier_(v, gain)
        for u in self.U:
            _xavier_(u, gain)
        _xavier_(self.classifier.weight, gain)
        nn.init.zeros_(self.classifier.bias)

    def encode(self, x, csrs, deg_inv_sqrt):
        """Return fused node embeddings ``(N, H*D)``."""
        # Same projected features under every relation at the first layer.
        h0 = self.projection(x)
        h_per_rel = [h0 for _ in range(self.n_relations)]
        rel_embs = [emb for emb in self.relation_embedding]

        for layer in self.layers:
            if self.use_inter:
                h_per_rel, rel_embs = layer(h_per_rel, rel_embs, csrs, deg_inv_sqrt)
            else:
                # Ablation w/o inter: intra only, keep per-relation c as h'.
                c_list = []
                for r in range(self.n_relations):
                    indptr, indices = csrs[r]
                    c_list.append(layer._intra(
                        h_per_rel[r], indptr, indices, deg_inv_sqrt[r],
                        rel_embs[r], layer.W_r[r]))
                rel_embs = [layer.relation_mlp[r](rel_embs[r])
                            for r in range(self.n_relations)]
                h_per_rel = c_list

        if self.use_fusion:
            return self.fusing(h_per_rel, rel_embs, self.V, self.U)
        # Ablation w/o fusion: uniform mean of V_r h'_r (no relation semantics).
        parts = []
        for r in range(self.n_relations):
            n = h_per_rel[r].size(0)
            h = h_per_rel[r].view(n, self.n_heads, self.hidden)
            h = torch.einsum('nhd,hdf->nhf', h, self.V[r])
            parts.append(h)
        return torch.stack(parts, dim=0).mean(dim=0).reshape(
            -1, self.n_heads * self.hidden)

    def forward(self, x, csrs, deg_inv_sqrt):
        return self.classifier(self.encode(x, csrs, deg_inv_sqrt))
