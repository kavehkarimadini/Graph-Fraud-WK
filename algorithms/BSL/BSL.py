'''
BSL (Yu, Liu, Luo, AAAI 2024): Barely Supervised Learning for graph fraud.

GCN encoder, NA/AA/NN feature disentanglement, then weak/strong
augmentation of unlabeled nodes with labeled prototypes. Port of
reference/BSL/BSL.md without DGL.
'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


NA, AA, NN = 0, 1, 2
N_PARTS = 3


def _scatter_add(values, index, n):
    out = torch.zeros(n, device=values.device, dtype=values.dtype)
    if values.numel() == 0:
        return out
    out.scatter_add_(0, index, values)
    return out


def build_gcn_adj(src, dst, n, device, dtype):
    """Symmetric normalized adjacency with self-loops (Eq. 3)."""
    src = src.to(device=device, dtype=torch.long)
    dst = dst.to(device=device, dtype=torch.long)
    if src.numel() > 0:
        src, dst = torch.cat([src, dst], dim=0), torch.cat([dst, src], dim=0)
    sl = torch.arange(n, device=device, dtype=torch.long)
    src = torch.cat([src, sl], dim=0)
    dst = torch.cat([dst, sl], dim=0)
    idx = torch.stack([dst, src], dim=0)
    val = torch.ones(src.numel(), dtype=dtype, device=device)
    adj = torch.sparse_coo_tensor(
        idx, val, (n, n), device=device, check_invariants=False).coalesce()
    adj = torch.sparse_coo_tensor(
        adj.indices(), adj.values().clamp(max=1.0), adj.size(),
        device=device, check_invariants=False).coalesce()
    row, col = adj.indices()
    deg = _scatter_add(adj.values(), row, n).clamp(min=1.0)
    d_inv = torch.pow(deg, -0.5)
    norm_val = d_inv[row] * adj.values() * d_inv[col]
    return torch.sparse_coo_tensor(
        adj.indices(), norm_val, adj.size(),
        device=device, check_invariants=False).coalesce()


def edge_type_labels(yi, yj):
    """NA=0, AA=1, NN=2 from endpoint labels (1=fraud)."""
    mixed = yi != yj
    both_fraud = (yi == 1) & (yj == 1)
    types = torch.full_like(yi, NN)
    types = torch.where(mixed, types.new_full((), NA), types)
    types = torch.where(both_fraud & ~mixed, types.new_full((), AA), types)
    return types


def split_parts(z, part_dim):
    return z[:, 0:part_dim], z[:, part_dim:2 * part_dim], z[:, 2 * part_dim:3 * part_dim]


class BSL(nn.Module):
    """Barely Supervised Learning detector (Yu et al., AAAI 2024)."""

    def __init__(self, in_dim, hid_dim=96, n_class=2):
        super(BSL, self).__init__()
        if hid_dim % N_PARTS != 0:
            hid_dim = hid_dim + (N_PARTS - hid_dim % N_PARTS)
        self.hid_dim = hid_dim
        self.part_dim = hid_dim // N_PARTS
        self.n_class = n_class
        self.fc_in = nn.Linear(in_dim, hid_dim)
        self.theta = nn.Linear(hid_dim, hid_dim)
        self.link_fc = nn.Linear(self.part_dim, 1)
        self.attn_w = nn.ModuleList([
            nn.Linear(self.part_dim, self.part_dim) for _ in range(N_PARTS)
        ])
        self.attn_q = nn.Parameter(torch.randn(self.part_dim) * 0.1)
        self.cls = nn.Linear(self.part_dim, n_class)

    def encode(self, adj, feat):
        xhat = F.relu(self.fc_in(feat))
        return self.theta(torch.sparse.mm(adj, xhat))

    def attention(self, parts):
        omegas = []
        for k, z_k in enumerate(parts):
            h = torch.tanh(self.attn_w[k](z_k))
            omegas.append((h * self.attn_q).sum(dim=-1))
        omega = torch.stack(omegas, dim=-1)
        return torch.softmax(omega, dim=-1)

    def aggregate(self, parts, alpha):
        stacked = torch.stack(parts, dim=1)
        return (alpha.unsqueeze(-1) * stacked).sum(dim=1)

    def classify_parts(self, parts):
        alpha = self.attention(parts)
        zhat = self.aggregate(parts, alpha)
        return self.cls(zhat), alpha, zhat

    def link_loss(self, parts, src, dst, labels, labeled_mask):
        both = labeled_mask[src] & labeled_mask[dst]
        if not both.any():
            return parts[0].new_zeros(())
        s, d = src[both], dst[both]
        scores = []
        for z_k in parts:
            scores.append(torch.sigmoid(self.link_fc(z_k[d] - z_k[s])).squeeze(-1))
        logits = torch.stack(scores, dim=-1)
        y_edge = edge_type_labels(labels[d], labels[s])
        return F.cross_entropy(logits, y_edge)

    def attn_loss(self, alpha, labels, labeled_idx):
        """Increase same-class subspace weight (NN for normal, AA for fraud).

        Eq. 9 as printed has the opposite sign of the surrounding text; this
        follows the text so minimizing L_attn raises the matching weight.
        """
        if labeled_idx is None or labeled_idx.numel() == 0:
            return alpha.new_zeros(())
        a = alpha[labeled_idx]
        y = labels[labeled_idx].to(dtype=a.dtype)
        # y=0: minimize α_aa - α_nn; y=1: minimize α_nn - α_aa
        return ((1.0 - y) * (a[:, AA] - a[:, NN]) + y * (a[:, NN] - a[:, AA])).mean()

    def bsl_loss(self, parts, labels, labeled_idx, unlabeled_idx):
        if unlabeled_idx is None or unlabeled_idx.numel() == 0:
            return parts[0].new_zeros(())
        lab = labels[labeled_idx]
        norm = labeled_idx[lab == 0]
        fraud = labeled_idx[lab == 1]
        if norm.numel() == 0 or fraud.numel() == 0:
            return parts[0].new_zeros(())
        n_i = norm[torch.randint(0, norm.numel(), (1,), device=norm.device)].item()
        a_i = fraud[torch.randint(0, fraud.numel(), (1,), device=fraud.device)].item()
        z_na, z_aa, z_nn = parts
        u = unlabeled_idx
        n_u = int(u.numel())
        z_n_na = z_na[n_i].unsqueeze(0).expand(n_u, -1)
        z_a_na = z_na[a_i].unsqueeze(0).expand(n_u, -1)
        z_n_aa = z_aa[n_i].unsqueeze(0).expand(n_u, -1)
        z_n_nn = z_nn[n_i].unsqueeze(0).expand(n_u, -1)
        z_a_aa = z_aa[a_i].unsqueeze(0).expand(n_u, -1)
        z_a_nn = z_nn[a_i].unsqueeze(0).expand(n_u, -1)

        p_nw, _, _ = self.classify_parts((z_n_na, z_aa[u], z_nn[u]))
        p_aw, _, _ = self.classify_parts((z_a_na, z_aa[u], z_nn[u]))
        p_ns, _, _ = self.classify_parts((z_na[u], z_n_aa, z_n_nn))
        p_as, _, _ = self.classify_parts((z_na[u], z_a_aa, z_a_nn))
        p_nw = torch.softmax(p_nw, dim=-1)
        p_aw = torch.softmax(p_aw, dim=-1)
        p_ns = torch.softmax(p_ns, dim=-1)
        p_as = torch.softmax(p_as, dim=-1)
        l_con = ((p_nw - p_aw) ** 2).mean()
        l_incon = -((p_ns - p_as) ** 2).mean()
        return l_con + l_incon

    def forward(self, adj, feat, labels=None, labeled_idx=None, unlabeled_idx=None,
                src=None, dst=None):
        z = self.encode(adj, feat)
        parts = split_parts(z, self.part_dim)
        logits, alpha, _zhat = self.classify_parts(parts)
        losses = {
            'class': feat.new_zeros(()),
            'link': feat.new_zeros(()),
            'attn': feat.new_zeros(()),
            'bsl': feat.new_zeros(()),
        }
        if labels is not None and labeled_idx is not None and labeled_idx.numel() > 0:
            losses['class'] = F.cross_entropy(logits[labeled_idx], labels[labeled_idx])
            losses['attn'] = self.attn_loss(alpha, labels, labeled_idx)
            if src is not None and dst is not None:
                labeled_mask = torch.zeros(feat.size(0), dtype=torch.bool, device=feat.device)
                labeled_mask[labeled_idx] = True
                losses['link'] = self.link_loss(parts, src, dst, labels, labeled_mask)
            losses['bsl'] = self.bsl_loss(parts, labels, labeled_idx, unlabeled_idx)
        return logits, losses
