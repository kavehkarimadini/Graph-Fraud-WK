'''
BWGNN ('Rethinking Graph Neural Networks for Anomaly Detection', Tang et al., ICML 2022)

Adapted from https://github.com/squareRoot3/Rethinking-Anomaly-Detection
'''
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import scipy
import sympy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class PolyConv(nn.Module):
    def __init__(self, in_feats, out_feats, theta, activation=F.leaky_relu,
                 lin=False, bias=False):
        super(PolyConv, self).__init__()
        self._theta = theta
        self._k = len(self._theta)
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.linear = nn.Linear(in_feats, out_feats, bias)
        self.lin = lin

    def reset_parameters(self):
        if self.linear.weight is not None:
            init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, graph, feat):
        # I - D^{-1/2} A D^{-1/2}, same as DGL copy_u + sum on in-edges.
        d_invsqrt = torch.pow(graph.in_degrees().float().clamp(min=1), -0.5).unsqueeze(-1)
        d_invsqrt = d_invsqrt.to(device=feat.device, dtype=feat.dtype)
        adj = graph.adj
        if adj.dtype != feat.dtype:
            adj = adj.to(dtype=feat.dtype)
        h = self._theta[0] * feat
        for _k in range(1, self._k):
            feat = feat - torch.sparse.mm(adj, feat * d_invsqrt) * d_invsqrt
            h = h + self._theta[_k] * feat
        if self.lin:
            h = self.activation(self.linear(h))
        return h


def calculate_theta2(d):
    thetas = []
    x = sympy.symbols('x')
    for k in range(d + 1):
        f = sympy.poly((x / 2) ** k * (1 - x / 2) ** (d - k) / scipy.special.beta(k + 1, d + 1 - k))
        coeff = f.all_coeffs()
        thetas.append([float(coeff[d - j]) for j in range(d + 1)])
    return thetas


class BWGNN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, graph=None, d=2):
        super(BWGNN, self).__init__()
        self.g = graph
        self.thetas = calculate_theta2(d=d)
        self.conv = nn.ModuleList([
            PolyConv(h_feats, h_feats, theta, lin=False) for theta in self.thetas
        ])
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats * len(self.conv), h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.ReLU()
        self.d = d

    def set_graph(self, graph):
        self.g = graph

    def encode(self, graph, in_feat):
        h = self.act(self.linear(in_feat))
        h = self.act(self.linear2(h))
        h_final = torch.cat([conv(graph, h) for conv in self.conv], dim=-1)
        h = self.act(self.linear3(h_final))
        return self.linear4(h)

    def forward(self, in_feat, graph=None):
        g = self.g if graph is None else graph
        if g is None:
            raise ValueError('BWGNN needs a graph with .adj / in_degrees (pass graph=... or call set_graph).')
        return self.encode(g, in_feat)


class BWGNN_Hetero(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, graph, d=2):
        super(BWGNN_Hetero, self).__init__()
        self.g = graph
        self.thetas = calculate_theta2(d=d)
        self.conv = nn.ModuleList([
            PolyConv(h_feats, h_feats, theta, lin=False) for theta in self.thetas
        ])
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats * len(self.conv), h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.LeakyReLU()

    def forward(self, in_feat):
        h = self.act(self.linear(in_feat))
        h = self.act(self.linear2(h))
        h_all = []
        for relation in self.g.canonical_etypes:
            h_rel = h
            h_final = torch.cat([conv(self.g[relation], h_rel) for conv in self.conv], dim=-1)
            h_all.append(self.linear3(h_final))
        h_all = self.act(torch.stack(h_all).sum(0))
        return self.linear4(h_all)
