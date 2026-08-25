'''GraphSAGE implemented with PyTorch Geometric SAGEConv.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class SupervisedGraphsage(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers=2,
                 dropout=0.0, concat=True, sigmoid_loss=False, aggregator='mean'):
        super(SupervisedGraphsage, self).__init__()
        self.dropout = dropout
        self.concat = concat
        self.sigmoid_loss = sigmoid_loss
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggregator))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggregator))
        out_hidden = hidden_channels * 2 if concat and num_layers > 1 else hidden_channels
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggregator))
        self.classifier = nn.Linear(hidden_channels if not concat else hidden_channels, num_classes)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)

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
