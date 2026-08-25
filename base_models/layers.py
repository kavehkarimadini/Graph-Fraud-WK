'''
This code is due to Yutong Deng (@yutongD), Yingtong Dou (@YingtongDou) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection)
https://github.com/safe-graph/DGFraud
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


class GraphConvolution(nn.Module):
    """GCN layer wrapping PyG GCNConv, with optional batch-norm and L2 norm."""

    def __init__(self, input_dim, output_dim, dropout=0., act=F.relu,
                 bias=True, norm=False, add_self_loops=False, normalize=False):
        super(GraphConvolution, self).__init__()
        self.dropout = dropout
        self.act = act
        self.norm = norm
        self.conv = GCNConv(input_dim, output_dim, bias=bias,
                            add_self_loops=add_self_loops, normalize=normalize)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        output = self.conv(x, edge_index, edge_weight=edge_weight)
        mean = output.mean(dim=0, keepdim=True)
        var = output.var(dim=0, unbiased=False, keepdim=True)
        output = (output - mean) / torch.sqrt(var + 1e-3)
        output = self.act(output)
        if self.norm:
            return F.normalize(output, p=2, dim=-1, eps=1e-12)
        return output


class AttentionLayer(nn.Module):
    """View / node / scaled-dot-product attention used by Player2Vec, SemiGNN, GAS."""

    def __init__(self, hidden_size=None, attention_size=1, v_type='tanh', bias=True):
        super(AttentionLayer, self).__init__()
        self.v_type = v_type
        if hidden_size is not None:
            self.w_omega = nn.Parameter(torch.randn(hidden_size, attention_size) * 0.1)
            self.b_omega = nn.Parameter(torch.randn(attention_size) * 0.1) if bias else None
            self.u_omega = nn.Parameter(torch.randn(attention_size) * 0.1)
        else:
            self.w_omega = None

    def forward(self, inputs, joint_type='weighted_sum'):
        hidden_size = inputs.shape[-1]
        if self.w_omega is None or self.w_omega.shape[0] != hidden_size:
            device = inputs.device
            self.w_omega = nn.Parameter(torch.randn(hidden_size, 1, device=device) * 0.1)
            self.b_omega = nn.Parameter(torch.randn(1, device=device) * 0.1)
            self.u_omega = nn.Parameter(torch.randn(1, device=device) * 0.1)

        v = torch.tensordot(inputs, self.w_omega, dims=1)
        if self.b_omega is not None:
            v = v + self.b_omega
        if self.v_type == 'tanh':
            v = torch.tanh(v)
        elif self.v_type == 'relu':
            v = F.relu(v)
        vu = torch.tensordot(v, self.u_omega, dims=1)
        weights = torch.softmax(vu, dim=-1)
        if joint_type == 'weighted_sum':
            return torch.sum(inputs * weights.unsqueeze(-1), dim=-2)
        return torch.cat(list(inputs * weights.unsqueeze(-1)), dim=-1)

    @staticmethod
    def attention(inputs, attention_size, v_type=None, return_weights=False, bias=True,
                  joint_type='weighted_sum', multi_view=True):
        if multi_view and inputs.dim() == 2:
            inputs = inputs.unsqueeze(0)
        hidden_size = inputs.shape[-1]
        device = inputs.device
        w_omega = torch.randn(hidden_size, attention_size, device=device) * 0.1
        b_omega = torch.randn(attention_size, device=device) * 0.1
        u_omega = torch.randn(attention_size, device=device) * 0.1
        # Use module-free functional path via a small linear layer registered outside.
        # Callers that need trainable attention should use AttentionLayer as a module.
        v = torch.matmul(inputs, w_omega)
        if bias:
            v = v + b_omega
        if v_type == 'tanh':
            v = torch.tanh(v)
        elif v_type == 'relu':
            v = F.relu(v)
        vu = torch.matmul(v, u_omega)
        weights = torch.softmax(vu, dim=-1)
        if joint_type == 'weighted_sum':
            output = torch.sum(inputs * weights.unsqueeze(-1), dim=-2)
        else:
            output = torch.cat(list(inputs * weights.unsqueeze(-1)), dim=-1)
        if return_weights:
            return output, weights
        return output

    @staticmethod
    def node_attention(inputs, adj, return_weights=False):
        hidden_size = inputs.shape[-1]
        h_v = torch.randn(hidden_size, 1, device=inputs.device) * 0.1
        scores = torch.matmul(inputs, h_v).squeeze(-1)
        mask = (adj != 0).float()
        logits = scores.unsqueeze(0) * mask + (1.0 - mask) * (-1e9)
        weights = torch.softmax(logits, dim=-1)
        output = torch.matmul(weights, inputs)
        if return_weights:
            return output, weights
        return output

    @staticmethod
    def scaled_dot_product_attention(q, k, v, mask=None):
        dk = k.size(-1) ** 0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) / dk
        if mask is not None:
            scores = scores + mask
        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, v)
        return output, weights


class ViewAttention(nn.Module):
    def __init__(self, encoding1, encoding2, layer_size, meta, in_dim):
        super(ViewAttention, self).__init__()
        self.meta = meta
        self.layer_size = layer_size
        self.encoding2 = encoding2
        layers = []
        dim = in_dim
        for i in range(layer_size):
            out_dim = encoding1 if i == 0 else encoding2
            layers.append(nn.ModuleList([nn.Linear(dim, out_dim) for _ in range(meta)]))
            dim = out_dim
        self.layers = nn.ModuleList(layers)
        self.phi = nn.Parameter(torch.randn(encoding2) * 0.1)

    def forward(self, inputs):
        h = list(inputs)
        for layer in self.layers:
            h = [F.relu(lin(h_i)) for lin, h_i in zip(layer, h)]
        stacked = torch.stack(h, dim=0)
        weights = torch.softmax(stacked * self.phi, dim=0)
        mixed = stacked * weights
        return mixed.permute(1, 0, 2).reshape(inputs[0].shape[0], -1)


class NodeAttention(nn.Module):
    def __init__(self, hidden_size):
        super(NodeAttention, self).__init__()
        self.h_v = nn.Parameter(torch.randn(hidden_size, 1) * 0.1)

    def forward(self, inputs, adj):
        scores = torch.matmul(inputs, self.h_v).squeeze(-1)
        mask = (adj != 0).float()
        logits = scores.unsqueeze(0) * mask + (1.0 - mask) * (-1e9)
        weights = torch.softmax(logits, dim=-1)
        return torch.matmul(weights, inputs)


class ConcatenationAggregator(nn.Module):
    """Equation (3) in GAS."""

    def __init__(self, input_dim, output_dim, dropout=0., act=F.relu):
        super(ConcatenationAggregator, self).__init__()
        self.dropout = dropout
        self.act = act
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, review_vecs, user_vecs, item_vecs, review_user_adj, review_item_adj):
        review_vecs = F.dropout(review_vecs, p=self.dropout, training=self.training)
        user_vecs = F.dropout(user_vecs, p=self.dropout, training=self.training)
        item_vecs = F.dropout(item_vecs, p=self.dropout, training=self.training)
        ru = user_vecs[review_user_adj.long()]
        ri = item_vecs[review_item_adj.long()]
        concate_vecs = torch.cat([review_vecs, ru, ri], dim=1)
        return self.act(self.linear(concate_vecs))


class AttentionAggregator(nn.Module):
    """Equations (5) and (8) in GAS."""

    def __init__(self, input_dim1, input_dim2, output_dim, hid_dim,
                 dropout=0., bias=False, act=F.relu, concat=False):
        super(AttentionAggregator, self).__init__()
        self.dropout = dropout
        self.act = act
        self.concat = concat
        self.user_weights = nn.Linear(input_dim1, hid_dim, bias=bias)
        self.item_weights = nn.Linear(input_dim2, hid_dim, bias=bias)
        self.concate_user_weights = nn.Linear(hid_dim, output_dim)
        self.concate_item_weights = nn.Linear(hid_dim, output_dim)

    def forward(self, review_vecs, user_vecs, item_vecs,
                user_review_adj, user_item_adj, item_review_adj, item_user_adj):
        review_vecs = F.dropout(review_vecs, p=self.dropout, training=self.training)
        user_vecs = F.dropout(user_vecs, p=self.dropout, training=self.training)
        item_vecs = F.dropout(item_vecs, p=self.dropout, training=self.training)

        ur = review_vecs[user_review_adj.long()]
        ri = item_vecs[user_item_adj.long()]
        ir = review_vecs[item_review_adj.long()]
        ru = user_vecs[item_user_adj.long()]

        concate_user_vecs = torch.cat([ur, ri], dim=-1).reshape(user_vecs.size(0), -1)
        concate_item_vecs = torch.cat([ir, ru], dim=-1).reshape(item_vecs.size(0), -1)

        concate_user_vecs, _ = AttentionLayer.scaled_dot_product_attention(
            user_vecs, user_vecs, concate_user_vecs)
        concate_item_vecs, _ = AttentionLayer.scaled_dot_product_attention(
            item_vecs, item_vecs, concate_item_vecs)

        user_output = self.act(self.user_weights(concate_user_vecs))
        item_output = self.act(self.item_weights(concate_item_vecs))
        if self.concat:
            user_output = self.concate_user_weights(user_output)
            item_output = self.concate_item_weights(item_output)
            user_output = torch.cat([user_vecs, user_output], dim=1)
            item_output = torch.cat([item_vecs, item_output], dim=1)
        return user_output, item_output


class GASConcatenation(nn.Module):
    def forward(self, review_vecs, item_vecs, user_vecs, homo_vecs,
                review_item_adj, review_user_adj):
        ri = item_vecs[review_item_adj.long()]
        ru = user_vecs[review_user_adj.long()]
        return torch.cat([ri, review_vecs, ru, homo_vecs], dim=1)


class GEMLayer(nn.Module):
    """Equation (8) in GEM."""

    def __init__(self, embedding, encoding, device_num):
        super(GEMLayer, self).__init__()
        self.device_num = device_num
        self.encoding = encoding
        self.W = nn.Linear(embedding, encoding, bias=False)
        self.V = nn.Linear(encoding, encoding, bias=False)
        self.alpha = nn.Parameter(torch.empty(device_num, 1))
        nn.init.xavier_uniform_(self.alpha)

    def forward(self, x, h, adjs):
        h1 = self.W(x)
        h2_list = []
        for adj in adjs:
            h2_list.append(torch.matmul(adj, self.V(h)))
        h2 = torch.stack(h2_list, dim=0)
        weights = torch.softmax(self.alpha, dim=0)
        h2 = (h2 * weights.view(self.device_num, 1, 1)).sum(dim=0)
        return torch.sigmoid(h1 + h2)


class GAT(nn.Module):
    """Multi-head GAT using PyG GATConv."""

    def __init__(self, in_dim, dim, n_heads, attn_drop=0., ffd_drop=0., concat=False):
        super(GAT, self).__init__()
        self.conv = GATConv(in_dim, dim, heads=n_heads, dropout=attn_drop,
                            concat=concat)
        self.ffd_drop = ffd_drop

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.ffd_drop, training=self.training)
        return F.elu(self.conv(x, edge_index))


class GeniePathLayer(nn.Module):
    """Adaptive Path Layer: GAT (breadth) + LSTM (depth)."""

    def __init__(self, in_dim, dim, heads=1):
        super(GeniePathLayer, self).__init__()
        self.gat = GATConv(in_dim, dim, heads=heads, concat=False)
        self.lstm = nn.LSTM(dim, dim, batch_first=True)
        self.dim = dim

    def forward(self, x, edge_index, h, c):
        x = torch.tanh(self.gat(x, edge_index))
        x, (h, c) = self.lstm(x.unsqueeze(1), (h, c))
        return x.squeeze(1), (h, c)
