"""
CGNN (Li, Yu, Luo, AAAI 2025): Context-aware graph neural network for graph-based
fraud detection with extremely limited labels.

Implementation from the paper (reference/CGNN.md; no official code is shipped):

- **latent mapping** (Eq. 1): ``h = LeakyReLU(FC(x))`` for every node;
- **category semantic decomposition** (Eq. 2): the latent vector of each
  neighbour is cut into halves ``h[:m]`` / ``h[m:]`` (``m = d // 2``) and projected
  by two linear layers into a normal component ``x^0`` and an abnormal component
  ``x^1`` (both of size ``d``). ``L_CSD`` (Eq. 3) is a supervised contrastive loss
  whose "labels" are the subspaces: normal components attract each other and repel
  abnormal ones;
- **denoising attention** (Eq. 4): ``alpha_j = softmax(W_att tanh(W (x^0 + x^1) + b))``
  taken over the two components of the neighbour, then
  ``m_j = alpha_j x^0 + (1 - alpha_j) x^1``;
- **aggregation and update** (Eq. 5): ``z_i = FC(h_i + sum_j m_j / sqrt(d_i))`` and an
  MLP classifier trained with class-weighted cross-entropy (Eq. 6);
- **subspace feature perturbation** (Eq. 9): for unlabeled centres the normal (or
  abnormal) components are randomly permuted among the centre's neighbours while
  every neighbour keeps its own attention weight; ``L_CL`` (Eq. 10) ties the clean
  and the two perturbed predictions together.
"""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_csr(src, dst, n):
    """Symmetrised, de-duplicated adjacency without self-loops (Eq. 5 adds ``h_i`` itself)."""
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


def gather_neighbors(indptr, indices, nodes):
    """Flatten the neighbour lists of ``nodes``: per-edge centre position and neighbour id."""
    start = indptr[nodes]
    deg = indptr[nodes + 1] - start
    seg = torch.repeat_interleave(torch.arange(nodes.numel(), device=nodes.device), deg)
    seg_start = torch.cumsum(deg, 0) - deg
    offset = torch.arange(seg.numel(), device=nodes.device) - seg_start[seg]
    return seg, indices[start[seg] + offset], deg


def permute_within_segments(seg, generator=None):
    """Random permutation of edge positions that never leaves a centre's neighbourhood.

    ``seg`` must be sorted (as produced by ``gather_neighbors``).
    """
    noise = torch.rand(seg.numel(), device=seg.device, dtype=torch.float64, generator=generator)
    return torch.argsort(seg.to(torch.float64) + noise)


def supcon_loss(z0, z1, tau):
    """Eq. 3 with the subspace as label (Khosla et al. 2020, averaged over anchors)."""
    if z0.size(0) < 2:
        return z0.sum() * 0.0
    z = F.normalize(torch.cat([z0, z1], dim=0), dim=1)
    lab = torch.cat([torch.zeros(z0.size(0), device=z.device),
                     torch.ones(z1.size(0), device=z.device)])
    logits = z @ z.t() / tau
    self_mask = torch.eye(z.size(0), dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(self_mask, float('-inf'))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos = (lab[:, None] == lab[None, :]) & ~self_mask
    log_prob = log_prob.masked_fill(~pos, 0.0)
    return -(log_prob.sum(dim=1) / pos.sum(dim=1)).mean()


def consistency_loss(prob, prob0, prob1, eps=1e-8):
    """Eq. 10: ``-mean( cos(y, y^0) * log cos(y, y^1) )``."""
    cos0 = F.cosine_similarity(prob, prob0, dim=1)
    cos1 = F.cosine_similarity(prob, prob1, dim=1).clamp_min(eps)
    return -(cos0 * torch.log(cos1)).mean()


class CGNN(nn.Module):
    def __init__(self, in_dim, emb_dim=64, n_class=2):
        super(CGNN, self).__init__()
        self.emb_dim = emb_dim
        self.m = emb_dim // 2
        self.latent = nn.Linear(in_dim, emb_dim)
        self.w_nor = nn.Linear(self.m, emb_dim)
        self.w_abnor = nn.Linear(emb_dim - self.m, emb_dim)
        self.att_hidden = nn.Linear(emb_dim, emb_dim)
        self.att_out = nn.Linear(emb_dim, 2, bias=False)
        self.update = nn.Linear(emb_dim, emb_dim)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.LeakyReLU(), nn.Linear(emb_dim, n_class))

    def embed(self, x):
        return F.leaky_relu(self.latent(x))

    def decompose(self, h):
        return self.w_nor(h[:, :self.m]), self.w_abnor(h[:, self.m:])

    def attention(self, x0, x1):
        return torch.softmax(self.att_out(torch.tanh(self.att_hidden(x0 + x1))), dim=-1)[:, :1]

    def aggregate(self, h_center, msg, seg, deg):
        agg = torch.zeros_like(h_center).index_add_(0, seg, msg)
        scale = deg.clamp_min(1).to(h_center.dtype).rsqrt().unsqueeze(1)
        return self.classifier(self.update(h_center + scale * agg))

    def forward(self, x, centers, indptr, indices, perturb=None, generator=None):
        """Logits of ``centers``.

        ``perturb`` (bool mask over ``centers``) selects centres that also get the two
        SFP views of Eq. 9; their logits are returned as ``(logits0, logits1)``.
        """
        seg, nbr, deg = gather_neighbors(indptr, indices, centers)
        n_c = centers.numel()
        uniq, inv = torch.unique(torch.cat([centers, nbr]), return_inverse=True)
        h = self.embed(x[uniq])
        x0, x1 = self.decompose(h)
        alpha = self.attention(x0, x1)
        c_pos, n_pos = inv[:n_c], inv[n_c:]
        a_e, x0_e, x1_e = alpha[n_pos], x0[n_pos], x1[n_pos]
        logits = self.aggregate(h[c_pos], a_e * x0_e + (1.0 - a_e) * x1_e, seg, deg)
        parts = {'x0': x0, 'x1': x1, 'nbr_pos': n_pos}
        if perturb is None or not bool(perturb.any()):
            return logits, None, parts

        sel = torch.nonzero(perturb, as_tuple=False).view(-1)
        remap = torch.full((n_c,), -1, dtype=torch.long, device=centers.device)
        remap[sel] = torch.arange(sel.numel(), device=centers.device)
        edge_keep = perturb[seg]
        p_seg = remap[seg[edge_keep]]
        p_a, p_x0, p_x1 = a_e[edge_keep], x0_e[edge_keep], x1_e[edge_keep]
        fa0 = p_x0[permute_within_segments(p_seg, generator)]
        fa1 = p_x1[permute_within_segments(p_seg, generator)]
        h_sel, deg_sel = h[c_pos[sel]], deg[sel]
        logits0 = self.aggregate(h_sel, p_a * fa0 + (1.0 - p_a) * p_x1, p_seg, deg_sel)
        logits1 = self.aggregate(h_sel, p_a * p_x0 + (1.0 - p_a) * fa1, p_seg, deg_sel)
        return logits, (logits0, logits1), parts
