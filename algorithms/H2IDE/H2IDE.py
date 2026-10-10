"""
H2IDE (Fu, Liu, Yuan, Wu, IEEE TKDE 2025): fraud detection from multi-relation
graphs via disentangled homophily and heterophily identification.

Implementation from the paper (reference/H2IDE.md; no official code is shipped):

- **disentangled transformation** (Eq. 2): ``K`` factor embeddings per node,
  ``h_{u,k} = norm(LeakyReLU(W_k x_u + b_k))``;
- **homophily / heterophily identification** (Eqs. 3-6): factor-level edge
  likelihood ``q(v|u,k) = cos(h_{u,k}, h_{v,k}) / tau``, factor distribution
  ``p(k|u) = softmax_k |phi(h_{u,k}, c_{u,k})|`` against the neighbour prototype
  ``c_{u,k}`` (mean over all relations), ``Q(v|u) = E_{p(k|u)} q(v|u,k)`` and a
  hinge loss on edges whose two endpoints are labeled;
- **relation-aware aggregation** (Eqs. 7-14): per relation and factor, signed GAT
  attention ``a [W h_u || W m h_v]`` with ``m = sign(q)``, aggregation
  ``LeakyReLU(sum alpha m h_v)``, then relations are mixed by
  ``sum_r beta_{r,k} * s(k|u,r) * h^r_{u,k}`` where ``s`` compares each factor with
  its signed neighbour prototype under that relation;
- **independence constraint** (Eqs. 16-18): CLUB upper bound of the mutual
  information between every ordered pair of factors, with one Gaussian
  variational network per pair.

Every relation keeps a self-loop in the aggregation (Eqs. 8-10) but not in the
prototypes (Eqs. 4 and 12).
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
    adj = sp.csr_matrix((np.ones(rows.size, dtype=np.float32), (rows, cols)), shape=(n, n))
    adj.sum_duplicates()
    adj.sort_indices()
    return adj.indptr.astype(np.int64), adj.indices.astype(np.int64)


def sample_neighbors(indptr, indices, nodes, fanout):
    """Per-edge centre position and neighbour id of ``nodes``.

    ``fanout <= 0`` keeps every neighbour. Otherwise nodes with more than ``fanout``
    neighbours get ``fanout`` uniform draws with replacement (GraphSAGE).
    """
    start = indptr[nodes]
    deg = indptr[nodes + 1] - start
    take = deg if fanout <= 0 else deg.clamp(max=fanout)
    seg = torch.repeat_interleave(torch.arange(nodes.numel(), device=nodes.device), take)
    seg_start = torch.cumsum(take, 0) - take
    offset = torch.arange(seg.numel(), device=nodes.device) - seg_start[seg]
    if fanout > 0:
        d = deg[seg]
        draw = torch.minimum((torch.rand(seg.numel(), device=nodes.device) * d).long(), d - 1)
        offset = torch.where(d <= fanout, offset, draw)
    return seg, indices[start[seg] + offset]


def batch_graph(csrs, targets, fanouts):
    """Two-hop computation graph of ``targets``.

    Hop 1 (``fanouts[0]``) gives the neighbours used by the targets' aggregation,
    their relation prototypes (Eq. 12) and the identification task (Eqs. 4-6).
    Hop 2 (``fanouts[1]``) lets the remaining hop-1 nodes compute their own
    relation embeddings.
    """
    f1, f2 = fanouts
    nt = targets.numel()
    hop1 = [sample_neighbors(ip, ix, targets, f1) for ip, ix in csrs]
    nb = torch.cat([n for _s, n in hop1])
    cand = torch.unique(nb)
    extra = cand[~torch.isin(cand, targets)]
    u1 = torch.cat([targets, extra])
    hop2 = [sample_neighbors(ip, ix, extra, f2) for ip, ix in csrs]
    sorted_u1, order = torch.sort(u1)

    pieces = [u1] + [n for _s, n in hop1] + [n for _s, n in hop2]
    h_nodes, inv = torch.unique(torch.cat(pieces), return_inverse=True)
    inv = list(torch.split(inv, [p.numel() for p in pieces]))
    u1_in_h = inv[0]
    n_rel = len(csrs)
    nu1 = u1.numel()
    self_dst = torch.arange(nu1, device=targets.device)
    rels = []
    for r in range(n_rel):
        d1, n1 = hop1[r]
        d2, _n2 = hop2[r]
        rels.append({
            'agg_dst': torch.cat([d1, nt + d2, self_dst]),
            'agg_src': torch.cat([inv[1 + r], inv[1 + n_rel + r], u1_in_h]),
            'proto_dst': d1,
            'proto_src': order[torch.searchsorted(sorted_u1, n1)],
        })
    return {
        'h_nodes': h_nodes, 'u1_in_h': u1_in_h, 'n_targets': nt, 'n_u1': nu1, 'rels': rels,
        'disen_seg': torch.cat([d for d, _n in hop1]),
        'disen_src': torch.cat(inv[1:1 + n_rel]),
        'disen_nbr': nb,
    }


def degree_chunks(deg, budget, max_nodes):
    """Contiguous ranges of nodes whose summed ``deg`` stays near ``budget``."""
    cum = np.concatenate([[0], np.cumsum(deg)])
    out, a, n = [], 0, len(deg)
    while a < n:
        b = int(np.searchsorted(cum, cum[a] + budget, side='right')) - 1
        b = min(max(b, a + 1), n, a + max_nodes)
        out.append((a, b))
        a = b
    return out


def segment_softmax(e, index, n):
    """Softmax of ``e`` (E, K) over edges sharing the same ``index``."""
    idx = index.unsqueeze(1).expand_as(e)
    mx = e.new_full((n, e.size(1)), float('-inf')).scatter_reduce(
        0, idx, e, reduce='amax', include_self=True)
    ex = (e - mx.detach()[index]).exp()
    den = e.new_zeros(n, e.size(1)).index_add_(0, index, ex)
    return ex / den[index]


def _segment_mean(values, index, n):
    out = values.new_zeros((n,) + values.shape[1:]).index_add_(0, index, values)
    cnt = torch.bincount(index, minlength=n).clamp(min=1).to(values.dtype)
    return out / cnt.view(-1, *([1] * (values.dim() - 1)))


def _uniform(shape, fan_in):
    bound = 1.0 / math.sqrt(fan_in)
    return nn.Parameter(torch.empty(shape).uniform_(-bound, bound))


class PairwiseCLUB(nn.Module):
    """CLUB (Cheng et al. 2020) for every ordered factor pair ``(i, j)``, ``i != j``:
    a Gaussian ``o(h_i | h_j)`` whose mean and log-variance come from two-layer MLPs."""

    def __init__(self, n_factors, factor_dim, hidden):
        super(PairwiseCLUB, self).__init__()
        pairs = [(i, j) for i in range(n_factors) for j in range(n_factors) if i != j]
        self.register_buffer('y_idx', torch.tensor([i for i, _j in pairs], dtype=torch.long))
        self.register_buffer('x_idx', torch.tensor([j for _i, j in pairs], dtype=torch.long))
        p, d = len(pairs), factor_dim
        self.mu_w1, self.mu_b1 = _uniform((p, d, hidden), d), _uniform((p, hidden), d)
        self.mu_w2, self.mu_b2 = _uniform((p, hidden, d), hidden), _uniform((p, d), hidden)
        self.lv_w1, self.lv_b1 = _uniform((p, d, hidden), d), _uniform((p, hidden), d)
        self.lv_w2, self.lv_b2 = _uniform((p, hidden, d), hidden), _uniform((p, d), hidden)

    def _stats(self, z):
        x, y = z[:, self.x_idx], z[:, self.y_idx]
        hm = F.relu(torch.einsum('npd,pdh->nph', x, self.mu_w1) + self.mu_b1)
        mu = torch.einsum('nph,phd->npd', hm, self.mu_w2) + self.mu_b2
        hl = F.relu(torch.einsum('npd,pdh->nph', x, self.lv_w1) + self.lv_b1)
        logvar = torch.tanh(torch.einsum('nph,phd->npd', hl, self.lv_w2) + self.lv_b2)
        return y, mu, logvar

    def log_likelihood(self, z):
        """Eq. 17 summed over pairs: the estimator maximises this on detached embeddings."""
        y, mu, logvar = self._stats(z)
        return (-(mu - y) ** 2 / logvar.exp() - logvar).sum(-1).mean(0).sum()

    def mi_upper(self, z):
        """Eqs. 16 / 18: sum over pairs of the CLUB estimate (negative term in closed form)."""
        y, mu, logvar = self._stats(z)
        positive = -(mu - y) ** 2 / 2.0 / logvar.exp()
        y_mean = y.mean(0, keepdim=True)
        y2_mean = (y ** 2).mean(0, keepdim=True)
        negative = -(y2_mean - 2.0 * mu * y_mean + mu ** 2) / 2.0 / logvar.exp()
        return (positive.sum(-1) - negative.sum(-1)).mean(0).sum()


class H2IDE(nn.Module):
    def __init__(self, in_dim, n_relations, n_factors=8, factor_dim=8, tau=0.5, dropout=0.1,
                 n_class=2, mi_hidden=16, slope=0.2, residual=False):
        super(H2IDE, self).__init__()
        self.K, self.dk, self.tau, self.slope = n_factors, factor_dim, tau, slope
        self.residual = residual
        dim = n_factors * factor_dim
        self.factor_proj = nn.Linear(in_dim, dim)
        self.att_w = _uniform((n_relations, n_factors, factor_dim, factor_dim), factor_dim)
        self.att_a = _uniform((n_relations, n_factors, 2 * factor_dim), 2 * factor_dim)
        self.beta = nn.Parameter(torch.ones(n_relations, n_factors, factor_dim))
        self.mlp = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, n_class))
        self.in_drop = nn.Dropout(dropout)
        self.club = PairwiseCLUB(n_factors, factor_dim, mi_hidden)

    def embed(self, x):
        z = F.leaky_relu(self.factor_proj(self.in_drop(x)), self.slope)
        return F.normalize(z.view(-1, self.K, self.dk), dim=-1)

    def phi(self, a, b):
        return F.cosine_similarity(a, b, dim=-1) / self.tau

    def relation_aggregate(self, r, h_dst, h_src, dst, src, n_dst):
        """Eqs. 7-10 for relation ``r``: signed factor-level attention aggregation."""
        with torch.no_grad():
            m = torch.sign((h_dst[dst] * h_src[src]).sum(-1))
        w, a = self.att_w[r], self.att_a[r]
        a_dst = (torch.einsum('nkd,kde->nke', h_dst, w) * a[:, :self.dk]).sum(-1)
        a_src = (torch.einsum('nkd,kde->nke', h_src, w) * a[:, self.dk:]).sum(-1)
        e = F.leaky_relu(a_dst[dst] + m * a_src[src], self.slope)
        alpha = segment_softmax(e, dst, n_dst)
        msg = (alpha * m).unsqueeze(-1) * h_src[src]
        out = h_dst.new_zeros(n_dst, self.K, self.dk).index_add_(0, dst, msg)
        if self.residual:
            out = out + h_dst
        return F.leaky_relu(out, self.slope)

    def factor_scores(self, h_t, h_all, dst, src, n_t):
        """Eqs. 11-12: ``s(k|u,r)`` against the signed relation prototype."""
        with torch.no_grad():
            m = torch.sign(F.cosine_similarity(h_t[dst], h_all[src], dim=-1))
        proto = _segment_mean(m.unsqueeze(-1) * h_all[src], dst, n_t)
        return torch.softmax(self.phi(h_t, proto), dim=-1)

    def combine(self, h_rel, scores):
        """Eqs. 13-14."""
        return sum(self.beta[r] * scores[r].unsqueeze(-1) * h for r, h in enumerate(h_rel))

    def classify(self, z):
        return self.mlp(z.reshape(z.size(0), -1))

    def homophily(self, h_t, h_all, seg, src, n_t):
        """Eqs. 3-5: ``Q(v|u)`` for every (centre, neighbour) edge."""
        proto = _segment_mean(h_all[src], seg, n_t)
        p = torch.softmax(self.phi(h_t, proto).abs(), dim=-1)
        q = self.phi(h_t[seg], h_all[src])
        return (p[seg] * q).sum(-1)

    def forward_batch(self, x, g):
        """Logits, final factor embeddings and ``Q`` of the hop-1 edges of ``g['n_targets']`` targets."""
        nt, nu1 = g['n_targets'], g['n_u1']
        h = self.embed(x[g['h_nodes']])
        hu1 = h[g['u1_in_h']]
        h_rel, scores = [], []
        for r, e in enumerate(g['rels']):
            out = self.relation_aggregate(r, hu1, h, e['agg_dst'], e['agg_src'], nu1)
            h_rel.append(out[:nt])
            scores.append(self.factor_scores(out[:nt], out, e['proto_dst'], e['proto_src'], nt))
        z = self.combine(h_rel, scores)
        q = self.homophily(hu1[:nt], h, g['disen_seg'], g['disen_src'], nt)
        return self.classify(z), z, q

    @torch.no_grad()
    def inference(self, x, csrs, targets, edge_budget=1 << 19, node_chunk=65536):
        """Exact logits of ``targets`` with full neighbourhoods, computed layer-wise in chunks."""
        n = x.size(0)
        h = torch.cat([self.embed(x[i:i + node_chunk]) for i in range(0, n, node_chunk)])
        degs = [(ip[1:] - ip[:-1]).cpu().numpy() for ip, _ix in csrs]
        h_all = []
        for r, (ip, ix) in enumerate(csrs):
            out = h.new_empty(n, self.K, self.dk)
            for a, b in degree_chunks(degs[r] + 1, edge_budget, node_chunk):
                nodes = torch.arange(a, b, device=x.device)
                seg, nbr = sample_neighbors(ip, ix, nodes, 0)
                dst = torch.cat([seg, nodes - a])
                src = torch.cat([nbr, nodes])
                out[a:b] = self.relation_aggregate(r, h[a:b], h, dst, src, b - a)
            h_all.append(out)
        t_np = targets.cpu().numpy()
        tdeg = sum(d[t_np] for d in degs) + 1
        logits = []
        for a, b in degree_chunks(tdeg, edge_budget, node_chunk):
            t = targets[a:b]
            parts, scores = [], []
            for r, (ip, ix) in enumerate(csrs):
                seg, nbr = sample_neighbors(ip, ix, t, 0)
                parts.append(h_all[r][t])
                scores.append(self.factor_scores(parts[-1], h_all[r], seg, nbr, t.numel()))
            logits.append(self.classify(self.combine(parts, scores)))
        return torch.cat(logits, dim=0)
