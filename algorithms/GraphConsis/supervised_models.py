'''GraphConsis (Liu et al., SIGIR 2020): context embeddings, consistency sampling, relation attention.'''
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_neighbor_csr(edge_indices, num_nodes, device):
    """Undirected CSR of (neighbor, relation) for all relations."""
    srcs, dsts, rels = [], [], []
    for r, edge_index in enumerate(edge_indices):
        e = edge_index.to(device=device, dtype=torch.long)
        if e.numel() == 0:
            continue
        src, dst = e[0], e[1]
        srcs.extend([src, dst])
        dsts.extend([dst, src])
        rel = torch.full((src.numel(),), r, device=device, dtype=torch.long)
        rels.extend([rel, rel])
    if not srcs:
        offsets = torch.zeros(num_nodes + 1, device=device, dtype=torch.long)
        empty = torch.zeros(0, device=device, dtype=torch.long)
        return offsets, empty, empty
    src = torch.cat(srcs)
    dst = torch.cat(dsts)
    rel = torch.cat(rels)
    order = torch.argsort(src)
    src, dst, rel = src[order], dst[order], rel[order]
    counts = torch.bincount(src, minlength=num_nodes)
    offsets = torch.zeros(num_nodes + 1, device=device, dtype=torch.long)
    offsets[1:] = torch.cumsum(counts, dim=0)
    return offsets, dst.contiguous(), rel.contiguous()


class GraphConsisLayer(nn.Module):
    """One paper layer: sample Q neighbors by Eq. 3–4, attend with Eq. 5–6, then ⊕."""

    def __init__(self, in_dim, out_dim, num_relations, num_samples, consis_eps=0.0,
                 max_candidates=256, dropout=0.0):
        super(GraphConsisLayer, self).__init__()
        self.num_samples = num_samples
        self.consis_eps = consis_eps
        self.max_candidates = max_candidates
        self.dropout = dropout
        self.relation_vectors = nn.Parameter(torch.empty(num_relations, in_dim))
        nn.init.xavier_uniform_(self.relation_vectors)
        self.attention_vec = nn.Linear(in_dim * 2, 1, bias=False)
        self.lin = nn.Linear(in_dim * 2, out_dim)

    def gather_candidates(self, nodes, offsets, nbrs, rels):
        batch = nodes.size(0)
        device = nodes.device
        starts = offsets[nodes]
        deg = (offsets[nodes + 1] - starts).clamp(min=0)
        if nbrs.numel() == 0:
            empty = nodes.new_zeros(batch, 1)
            valid = torch.zeros(batch, 1, dtype=torch.bool, device=device)
            return empty, empty, valid, deg
        k = min(self.max_candidates, max(int(deg.max().item()) if deg.numel() else 1, 1))
        pos = torch.arange(k, device=device).unsqueeze(0).expand(batch, k)
        rand_pos = (torch.rand(batch, k, device=device) * deg.clamp(min=1).unsqueeze(1).float()).long()
        pos = torch.where(deg.unsqueeze(1) > k, rand_pos, pos)
        valid = pos < deg.unsqueeze(1)
        idx = (starts.unsqueeze(1) + pos).clamp(0, nbrs.numel() - 1)
        return nbrs[idx], rels[idx], valid, deg

    def sample_from_candidates(self, nodes, h, cand_n, cand_r, valid, deg):
        q = self.num_samples
        batch = nodes.size(0)
        if cand_n.numel() == 0:
            neigh = nodes.unsqueeze(1).expand(batch, q).contiguous()
            rel = torch.zeros(batch, q, device=nodes.device, dtype=torch.long)
            return neigh, rel
        score = torch.exp(-((h[cand_n] - h[nodes].unsqueeze(1)) ** 2).sum(dim=-1))
        score = score.masked_fill(~valid, 0.0)
        if self.consis_eps > 0:
            score = score.masked_fill(score <= self.consis_eps, 0.0)
        empty = score.sum(dim=-1) <= 0
        if empty.any():
            fallback = valid.float() + (~valid.any(dim=-1, keepdim=True)).float()
            score = torch.where(empty.unsqueeze(1), fallback, score)
        prob = score / score.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        pick = torch.multinomial(prob, q, replacement=True)
        neigh = cand_n.gather(1, pick)
        rel = cand_r.gather(1, pick)
        isolated = deg == 0
        if isolated.any():
            neigh = neigh.clone()
            rel = rel.clone()
            neigh[isolated] = nodes[isolated].unsqueeze(1).expand(-1, q)
            rel[isolated] = 0
        return neigh, rel

    def aggregate(self, nodes, h, neigh, rel):
        h_self = h[nodes]
        h_q = h[neigh]
        t_r = self.relation_vectors[rel]
        logit = F.leaky_relu(self.attention_vec(torch.cat([h_q, t_r], dim=-1)), 0.2).squeeze(-1)
        attn = torch.softmax(logit, dim=-1)
        agg = (attn.unsqueeze(-1) * h_q).sum(dim=1)
        out = F.relu(self.lin(torch.cat([h_self, agg], dim=-1)))
        return F.dropout(out, p=self.dropout, training=self.training)

    def forward(self, nodes, h, offsets, nbrs, rels):
        cand_n, cand_r, valid, deg = self.gather_candidates(nodes, offsets, nbrs, rels)
        neigh, rel = self.sample_from_candidates(nodes, h, cand_n, cand_r, valid, deg)
        return self.aggregate(nodes, h, neigh, rel)


class SupervisedGraphconsis(nn.Module):
    """Paper GraphConsis: context || features, two sampled layers, relation attention."""

    def __init__(self, in_channels, hidden_channels, num_classes, num_relations,
                 context_dim=128, dropout=0.0, sigmoid_loss=False, aggregator='mean',
                 num_nodes=0, dim_2=100, samples_1=10, samples_2=5,
                 consis_eps=0.0, max_candidates=256):
        super(SupervisedGraphconsis, self).__init__()
        del aggregator
        self.num_relations = num_relations
        self.dropout = dropout
        self.sigmoid_loss = sigmoid_loss
        self.context_dim = context_dim
        self.use_context = context_dim > 0
        self.num_nodes = num_nodes
        if self.use_context:
            self.context_emb = nn.Parameter(torch.empty(num_nodes, context_dim))
            nn.init.xavier_uniform_(self.context_emb)
        in_dim = in_channels + context_dim
        self.layer1 = GraphConsisLayer(
            in_dim, hidden_channels, num_relations, samples_1,
            consis_eps=consis_eps, max_candidates=max_candidates, dropout=dropout)
        self.layer2 = GraphConsisLayer(
            hidden_channels, dim_2, num_relations, samples_2,
            consis_eps=consis_eps, max_candidates=max_candidates, dropout=dropout)
        self.classifier = nn.Linear(dim_2, num_classes)
        self.register_buffer('offsets', torch.zeros(1, dtype=torch.long), persistent=False)
        self.register_buffer('nbrs', torch.zeros(0, dtype=torch.long), persistent=False)
        self.register_buffer('rels', torch.zeros(0, dtype=torch.long), persistent=False)

    def set_graph(self, edge_indices, num_nodes=None):
        n = int(num_nodes if num_nodes is not None else self.num_nodes)
        device = self.classifier.weight.device
        offsets, nbrs, rels = build_neighbor_csr(edge_indices, n, device)
        self.offsets = offsets
        self.nbrs = nbrs
        self.rels = rels

    def _base(self, x):
        if self.use_context:
            return torch.cat([x, self.context_emb], dim=-1)
        return x

    def forward_nodes(self, x, nodes):
        """Embeddings + logits for ``nodes`` via 2-hop consistency sampling."""
        h0 = self._base(x)
        cand2_n, cand2_r, valid2, deg2 = self.layer2.gather_candidates(
            nodes, self.offsets, self.nbrs, self.rels)
        parts = [nodes]
        if valid2.any():
            parts.append(cand2_n[valid2])
        need_h1 = torch.unique(torch.cat(parts))
        h1_u = self.layer1(need_h1, h0, self.offsets, self.nbrs, self.rels)
        h1 = h1_u.new_zeros(x.size(0), h1_u.size(1))
        h1[need_h1] = h1_u
        neigh2, rel2 = self.layer2.sample_from_candidates(
            nodes, h1, cand2_n, cand2_r, valid2, deg2)
        h2 = self.layer2.aggregate(nodes, h1, neigh2, rel2)
        return self.classifier(F.normalize(h2, p=2, dim=-1))

    def forward(self, x, edge_indices=None, node_ids=None):
        if edge_indices is not None and self.offsets.numel() <= 1:
            self.set_graph(edge_indices, x.size(0))
        if node_ids is None:
            raise ValueError('GraphConsis expects node_ids (sampled minibatch).')
        return self.forward_nodes(x, node_ids)

    def loss(self, logits, labels):
        if self.sigmoid_loss:
            if labels.dim() == 1:
                labels = F.one_hot(labels.long(), num_classes=logits.size(-1)).float()
            return F.binary_cross_entropy_with_logits(logits, labels.float())
        if labels.dim() > 1:
            labels = labels.argmax(dim=-1)
        return F.cross_entropy(logits, labels.long())

    def predict(self, logits):
        if self.sigmoid_loss:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=-1)
