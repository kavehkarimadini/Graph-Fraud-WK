import torch.nn as nn
import torch.nn.functional as F

from base_models.layers import GraphConvolution


class GCN(nn.Module):
    """Two-layer GCN used by Player2Vec, FdGars and GAS."""

    def __init__(self, input_dim, dim1, output_dim, dropout=0.):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(input_dim, dim1, dropout=dropout, act=F.relu, norm=True)
        self.gc2 = GraphConvolution(dim1, output_dim, dropout=dropout, act=F.relu, norm=False)

    def forward(self, x, edge_index, edge_weight=None):
        h = self.gc1(x, edge_index, edge_weight)
        return self.gc2(h, edge_index, edge_weight)

    def embedding(self, x, edge_index, edge_weight=None):
        return self.forward(x, edge_index, edge_weight)
