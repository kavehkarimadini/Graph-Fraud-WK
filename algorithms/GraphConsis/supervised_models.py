'''GraphConsis: multi-relation GraphSAGE with relation attention (PyG).'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class SupervisedGraphconsis(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_relations,
                 context_dim=0, dropout=0.0, sigmoid_loss=False, aggregator='mean',
                 num_nodes=0):
        super(SupervisedGraphconsis, self).__init__()
        self.num_relations = num_relations
        self.dropout = dropout
        self.sigmoid_loss = sigmoid_loss
        self.context_dim = context_dim
        self.use_context = context_dim > 0
        if self.use_context:
            self.context_emb = nn.Parameter(torch.empty(num_nodes, context_dim))
            nn.init.xavier_uniform_(self.context_emb)
        in_dim = in_channels + context_dim

        self.rel_convs = nn.ModuleList()
        for _ in range(num_relations):
            self.rel_convs.append(nn.ModuleList([
                SAGEConv(in_dim, hidden_channels, aggr=aggregator),
                SAGEConv(hidden_channels, hidden_channels, aggr=aggregator),
            ]))
        self.relation_vectors = nn.Parameter(torch.empty(num_relations, hidden_channels))
        nn.init.xavier_uniform_(self.relation_vectors)
        self.attention_vec = nn.Linear(hidden_channels * 2, 1, bias=False)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def encode_relation(self, x, edge_index, convs):
        h = F.relu(convs[0](x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return convs[1](h, edge_index)

    def forward(self, x, edge_indices, node_ids=None):
        if self.use_context:
            x = torch.cat([x, self.context_emb], dim=-1)
        outputs = []
        for r, convs in enumerate(self.rel_convs):
            h = self.encode_relation(x, edge_indices[r], convs)
            rel = self.relation_vectors[r].unsqueeze(0).expand(h.size(0), -1)
            attn_in = torch.cat([h, rel], dim=-1)
            weights = torch.sigmoid(self.attention_vec(attn_in))
            outputs.append(h * weights)
        out = torch.stack(outputs, dim=0).sum(dim=0)
        out = F.normalize(out, p=2, dim=-1)
        return self.classifier(out)

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
