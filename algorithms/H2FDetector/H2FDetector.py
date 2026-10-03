"""
H2-FDetector (Shi et al., WWW 2022): GNN fraud detector with homophilic and
heterophilic connections.

Port of the official DGL implementation (reference/H2-FDetector-master) with
``torch`` gather / scatter ops:

- relation-aware edge scorer ``tanh(f([d(h_u), d(h_v), d(h_u) - d(h_v)]))``
  trained with a hinge loss on train-edge homophily labels (+1 same class,
  -1 different class) over the union (``homo``) graph;
- per-relation attention aggregation where each message is multiplied by the
  sign of its edge score, so heterophilic neighbours push instead of pull;
- relation outputs concatenated and mixed by a linear layer; a prototype loss
  pulls train embeddings towards their class mean at every layer.

Edge signs carry no gradient (d sign / dx = 0), so they are computed under
``no_grad``: per edge in chunks while dropout is active, otherwise from
per-node terms. The attention logit ``atten([sign * h_src, h_dst])`` is
split into per-node terms, which is exact for a linear ``atten``.
"""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_edges(src, dst, n):
    """Symmetrised, de-duplicated COO edges with exactly one self-loop per node."""
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    loops = np.arange(n, dtype=np.int64)
    rows = np.concatenate([src, dst, loops])
    cols = np.concatenate([dst, src, loops])
    adj = sp.coo_matrix((np.ones(rows.size, dtype=np.float32), (rows, cols)), shape=(n, n)).tocsr()
    adj.sum_duplicates()
    adj = adj.tocoo()
    return adj.row.astype(np.int64), adj.col.astype(np.int64)


def segment_softmax(e, index, n):
    """Softmax of ``e`` (E, H) over edges sharing the same ``index`` (destination)."""
    idx = index.unsqueeze(1).expand_as(e)
    mx = e.new_full((n, e.size(1)), float('-inf')).scatter_reduce(
        0, idx, e, reduce='amax', include_self=True)
    ex = (e - mx.detach()[index]).exp()
    den = e.new_zeros(n, e.size(1)).index_add_(0, index, ex)
    return ex / den[index]


def hinge_loss(labels, scores, margin=1.0):
    return F.relu(margin - labels * scores).mean()


class RelationAware(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.d_liner = nn.Linear(in_dim, out_dim)
        self.f_liner = nn.Linear(3 * out_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def score(self, proj, src, dst):
        s, d = proj[src], proj[dst]
        e = self.dropout(torch.cat([s, d, s - d], dim=1))
        return torch.tanh(self.f_liner(e).squeeze(-1))

    @torch.no_grad()
    def signs(self, h, src, dst, chunk=1 << 20):
        proj = self.d_liner(h)
        if not self.training or self.dropout.p == 0:
            # Without dropout, f([s, d, s - d]) = (w1 + w3)·s + (w2 - w3)·d + b.
            w1, w2, w3 = self.f_liner.weight.view(3, -1)
            p_src = proj @ (w1 + w3)
            p_dst = proj @ (w2 - w3) + self.f_liner.bias
            return torch.sign(p_src[src] + p_dst[dst])
        out = torch.empty(src.numel(), device=h.device, dtype=h.dtype)
        for i in range(0, src.numel(), chunk):
            out[i:i + chunk] = torch.sign(self.score(proj, src[i:i + chunk], dst[i:i + chunk]))
        return out


class H2FDetectorLayer(nn.Module):
    """Signed multi-head attention aggregation over one relation."""

    def __init__(self, in_dim, out_dim, head, if_sum=False):
        super().__init__()
        self.head = head
        self.hd = out_dim
        self.if_sum = if_sum
        self.w_liner = nn.Linear(in_dim, out_dim * head)
        self.atten = nn.Linear(2 * out_dim, 1)

    def forward(self, h, src, dst, sign):
        n = h.size(0)
        z = self.w_liner(h).view(n, self.head, self.hd)
        w = self.atten.weight.view(-1)
        a_src = z @ w[:self.hd]
        a_dst = z @ w[self.hd:]
        e = F.leaky_relu(sign.unsqueeze(1) * a_src[src] + a_dst[dst] + self.atten.bias)
        alpha = segment_softmax(e, dst, n)
        msg = (alpha * sign.unsqueeze(1)).unsqueeze(-1) * z[src]
        out = z.new_zeros(n, self.head, self.hd).index_add_(0, dst, msg)
        return out.sum(dim=1) if self.if_sum else out.reshape(n, self.head * self.hd)


class MultiRelationH2FDetectorLayer(nn.Module):
    def __init__(self, in_dim, out_dim, head, n_relations, dropout, if_sum=False):
        super().__init__()
        width = out_dim if if_sum else out_dim * head
        self.relation_aware = RelationAware(in_dim, out_dim * head, dropout)
        self.layers = nn.ModuleList(
            [H2FDetectorLayer(in_dim, out_dim, head, if_sum) for _ in range(n_relations)])
        self.liner = nn.Linear(n_relations * width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, relations):
        hs = []
        for layer, (src, dst) in zip(self.layers, relations):
            sign = self.relation_aware.signs(h, src, dst)
            hs.append(layer(h, src, dst, sign))
        return self.liner(self.dropout(torch.cat(hs, dim=1)))

    def edge_loss(self, h, src, dst, label):
        if src.numel() == 0:
            return h.new_zeros(())
        proj = self.relation_aware.d_liner(h)
        return hinge_loss(label, self.relation_aware.score(proj, src, dst))


def prototype_loss(h, train_idx, train_label, sample_idx):
    """Cross-entropy over negative distances to the fraud / benign class means."""
    train_h = h[train_idx]
    pos = train_label == 1
    neg = train_label == 0
    if not pos.any() or not neg.any():
        return h.new_zeros(())
    pos_proto = train_h[pos].mean(dim=0, keepdim=True)
    neg_proto = train_h[neg].mean(dim=0, keepdim=True)
    sample_h = train_h[sample_idx]
    diff = torch.stack([
        -F.pairwise_distance(sample_h, neg_proto.expand_as(sample_h)),
        -F.pairwise_distance(sample_h, pos_proto.expand_as(sample_h)),
    ], dim=1)
    return F.cross_entropy(diff, train_label[sample_idx])


class H2FDetector(nn.Module):
    def __init__(self, in_dim, n_relations, intra_dim=8, head=4, n_layer=2, n_class=2, dropout=0.1):
        super().__init__()
        self.n_layer = n_layer
        if n_layer == 1:
            dims = [(in_dim, n_class, True)]
        else:
            dims = [(in_dim, intra_dim, False)]
            dims += [(intra_dim * head, intra_dim, False)] * (n_layer - 2)
            dims += [(intra_dim * head, n_class, True)]
        self.mine_layers = nn.ModuleList([
            MultiRelationH2FDetectorLayer(i, o, head, n_relations, dropout, if_sum=s)
            for i, o, s in dims])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, relations, aux=None):
        """Logits for every node.

        With ``aux`` (a callable returning per-layer edge and prototype samples),
        also returns the summed edge hinge loss and prototype loss.
        """
        h = x
        edge_loss = x.new_zeros(())
        proto_loss = x.new_zeros(())
        for i, layer in enumerate(self.mine_layers):
            out = layer(h, relations)
            if aux is not None:
                e_src, e_dst, e_label, train_idx, train_label, proto_idx = aux()
                edge_loss = edge_loss + layer.edge_loss(h, e_src, e_dst, e_label)
                proto_loss = proto_loss + prototype_loss(out, train_idx, train_label, proto_idx)
            h = out
            if i < len(self.mine_layers) - 1:
                h = self.dropout(F.relu(h))
        if aux is None:
            return h
        return h, edge_loss, proto_loss
