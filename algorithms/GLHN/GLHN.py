'''
GLHN / GLHAD (Guo et al., CIKM 2024): Graph Local Homophily Network.

Predicts node-level local homophily, then uses it to mix a low-pass
filter (I-L) with a high-pass filter (L). Two InfoNCE paradigms resist
feature camouflage. Port of reference/GLHN/GLHN.md without DGL.
'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


def _scatter_add(values, index, n):
    out = torch.zeros(n, device=values.device, dtype=values.dtype)
    if values.numel() == 0:
        return out
    out.scatter_add_(0, index, values)
    return out


def _symmetrize_edges(src, dst):
    return torch.cat([src, dst], dim=0), torch.cat([dst, src], dim=0)


def build_relation(src, dst, n, device, dtype, symmetrize=True):
    """Undirected sparse adjacency plus degree-normalized sqrt weights."""
    src = src.to(device=device, dtype=torch.long)
    dst = dst.to(device=device, dtype=torch.long)
    if symmetrize and src.numel() > 0:
        src, dst = _symmetrize_edges(src, dst)
    if src.numel() == 0:
        idx = torch.zeros(2, 0, dtype=torch.long, device=device)
        val = torch.zeros(0, dtype=dtype, device=device)
        adj = torch.sparse_coo_tensor(
            idx, val, (n, n), device=device, check_invariants=False).coalesce()
        d_invsqrt = torch.ones(n, 1, device=device, dtype=dtype)
        return adj, d_invsqrt, src, dst
    idx = torch.stack([dst, src], dim=0)
    val = torch.ones(src.numel(), dtype=dtype, device=device)
    adj = torch.sparse_coo_tensor(
        idx, val, (n, n), device=device, check_invariants=False).coalesce()
    adj = torch.sparse_coo_tensor(
        adj.indices(), adj.values().clamp(max=1.0), adj.size(),
        device=device, check_invariants=False).coalesce()
    row, col = adj.indices()
    deg = _scatter_add(adj.values(), row, n)
    d_invsqrt = torch.pow(deg.clamp(min=1.0), -0.5).unsqueeze(-1)
    return adj, d_invsqrt, col, row


def low_pass(feat, adj, d_invsqrt):
    """F_L X = (I - L) X = D^{-1/2} A D^{-1/2} X."""
    return torch.sparse.mm(adj, feat * d_invsqrt) * d_invsqrt


def high_pass(feat, adj, d_invsqrt):
    """F_H X = L X = X - D^{-1/2} A D^{-1/2} X."""
    return feat - low_pass(feat, adj, d_invsqrt)


def predict_local_homophily(feat, src, dst, fc_c, fc_n):
    """Eq. 6: mean sigmoid inner product of projected center/neighbor features."""
    n = feat.size(0)
    if src.numel() == 0:
        return feat.new_full((n, 1), 0.5)
    score = torch.sigmoid((fc_c(feat)[dst] * fc_n(feat)[src]).sum(dim=-1))
    h_sum = _scatter_add(score, dst, n)
    deg = torch.bincount(dst, minlength=n).to(dtype=feat.dtype)
    hom = h_sum / deg.clamp(min=1.0)
    return torch.where(deg > 0, hom, hom.new_full((), 0.5)).unsqueeze(-1)


def labeled_infonce(feat, src, dst, labels, labeled_mask, tau):
    """Eq. 7: InfoNCE on labeled centers (HMS vs HTS among labeled neighbors)."""
    both = labeled_mask[src] & labeled_mask[dst]
    if not both.any():
        return feat.new_zeros(())
    feat = F.normalize(feat, dim=-1)
    s, d = src[both], dst[both]
    exp_d = torch.exp(((feat[d] * feat[s]).sum(dim=-1) / tau).clamp(min=-40, max=40))
    same = (labels[d] == labels[s]).to(dtype=feat.dtype)
    n = feat.size(0)
    pos = _scatter_add(exp_d * same, d, n)
    tot = _scatter_add(exp_d, d, n)
    keep = pos > 0
    if not keep.any():
        return feat.new_zeros(())
    return -torch.log((pos[keep] / tot[keep].clamp(min=1e-12)).clamp(min=1e-12)).mean()


def unlabeled_infonce(feat, src, dst, labels, labeled_mask, tau):
    """Eq. 8: InfoNCE on unlabeled centers; pos/neg from NS vs AS closeness."""
    ok = (~labeled_mask[dst]) & labeled_mask[src]
    if not ok.any():
        return feat.new_zeros(())
    feat = F.normalize(feat, dim=-1)
    s, d = src[ok], dst[ok]
    cos = (feat[d] * feat[s]).sum(dim=-1)
    ns = (labels[s] == 0)
    anom = (labels[s] == 1)
    n = feat.size(0)
    ns_f = ns.to(dtype=feat.dtype)
    as_f = anom.to(dtype=feat.dtype)
    ns_sum = _scatter_add(cos * ns_f, d, n)
    as_sum = _scatter_add(cos * as_f, d, n)
    ns_cnt = _scatter_add(ns_f, d, n)
    as_cnt = _scatter_add(as_f, d, n)
    valid = (ns_cnt > 0) & (as_cnt > 0)
    if not valid.any():
        return feat.new_zeros(())
    closer_ns = (ns_sum / ns_cnt.clamp(min=1.0)) > (as_sum / as_cnt.clamp(min=1.0))
    edge_pos = torch.where(closer_ns[d], ns, anom).to(dtype=feat.dtype)
    exp_d = torch.exp((cos / tau).clamp(min=-40, max=40))
    pos = _scatter_add(exp_d * edge_pos, d, n)
    tot = _scatter_add(exp_d, d, n)
    keep = valid & (pos > 0)
    if not keep.any():
        return feat.new_zeros(())
    return -torch.log((pos[keep] / tot[keep].clamp(min=1e-12)).clamp(min=1e-12)).mean()


class GLHNLayer(nn.Module):
    """One layer: predict homophily, mix F_L / F_H, return contrastive terms."""

    def __init__(self, in_dim, out_dim, tau=0.1, filter_pair='laplacian'):
        super(GLHNLayer, self).__init__()
        self.tau = tau
        self.filter_pair = filter_pair
        self.fc_c = nn.Linear(in_dim, in_dim, bias=False)
        self.fc_n = nn.Linear(in_dim, in_dim, bias=False)
        self.fc_l = nn.Linear(in_dim, out_dim)
        self.fc_h = nn.Linear(in_dim, out_dim)

    def _filters(self, feat, adj, d_invsqrt):
        low = low_pass(feat, adj, d_invsqrt)
        high = feat - low
        if self.filter_pair == 'half':
            low = 0.5 * feat + 0.5 * low
            high = 0.5 * feat + 0.5 * high
        return low, high

    def forward_relation(self, feat, adj, d_invsqrt, src, dst,
                         labels=None, labeled_mask=None):
        hom = predict_local_homophily(feat, src, dst, self.fc_c, self.fc_n)
        low, high = self._filters(feat, adj, d_invsqrt)
        mixed = hom * F.relu(self.fc_l(low)) + (1.0 - hom) * F.relu(self.fc_h(high))
        lk = feat.new_zeros(())
        lu = feat.new_zeros(())
        if labels is not None and labeled_mask is not None and src.numel() > 0:
            lk = labeled_infonce(feat, src, dst, labels, labeled_mask, self.tau)
            lu = unlabeled_infonce(feat, src, dst, labels, labeled_mask, self.tau)
        return mixed, lk, lu


class GLHN(nn.Module):
    """Graph Local Homophily Network (paper name: GLHAD)."""

    def __init__(self, in_dim, hid_dim, n_class, n_layers=2, tau=0.1,
                 filter_pair='laplacian'):
        super(GLHN, self).__init__()
        self.n_layers = n_layers
        dims = [in_dim] + [hid_dim] * n_layers
        self.layers = nn.ModuleList([
            GLHNLayer(dims[i], dims[i + 1], tau=tau, filter_pair=filter_pair)
            for i in range(n_layers)
        ])
        self.out = nn.Linear(in_dim + hid_dim * n_layers, n_class)

    def forward(self, relations, feat, labels=None, labeled_mask=None):
        """``relations`` is a list of (name, adj, d_invsqrt, src, dst)."""
        xs = [feat]
        layer_lk = []
        layer_lu = []
        h = feat
        for layer in self.layers:
            outs = []
            lks = []
            lus = []
            for _name, adj, d_inv, src, dst in relations:
                mixed, lk, lu = layer.forward_relation(
                    h, adj, d_inv, src, dst, labels, labeled_mask)
                outs.append(mixed)
                lks.append(lk)
                lus.append(lu)
            h = torch.stack(outs, dim=0).mean(dim=0)
            xs.append(h)
            layer_lk.append(torch.stack(lks).mean() if lks else h.new_zeros(()))
            layer_lu.append(torch.stack(lus).mean() if lus else h.new_zeros(()))
        logits = self.out(torch.cat(xs, dim=1))
        return logits, layer_lk, layer_lu
