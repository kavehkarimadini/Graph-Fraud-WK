'''
This code is due to Yutong Deng (@yutongD), Yingtong Dou (@YingtongDou) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection)
https://github.com/safe-graph/DGFraud

GeniePath ('GeniePath: Graph Neural Networks with Adaptive Receptive Paths')
'''
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../..')))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.base_algorithm import Algorithm
from base_models.layers import GeniePathLayer
from utils.torch_helpers import (accuracy_from_logits, adj_to_edge_index,
                                 pred_from_logits, to_float_tensor, to_long_tensor)


class GeniePath(Algorithm):

    def __init__(self, nodes, in_dim, out_dim, dim, lstm_hidden, heads, layer_num,
                 class_size, weight_decay=0.01, device=None, **kwargs):
        super(GeniePath, self).__init__(weight_decay=weight_decay)
        self.nodes = nodes
        self.dim = dim
        self.out_dim = out_dim
        self.class_size = class_size
        self.device = device or torch.device('cpu')
        self.input_proj = nn.Linear(in_dim, dim)
        self.layers = nn.ModuleList([
            GeniePathLayer(dim, dim, heads=heads) for _ in range(layer_num)
        ])
        self.output_proj = nn.Linear(dim, out_dim)
        self.to(self.device)

    def forward(self, x, edge_index, batch_index):
        x = self.input_proj(x)
        h = torch.zeros(1, x.size(0), self.dim, device=x.device)
        c = torch.zeros(1, x.size(0), self.dim, device=x.device)
        for layer in self.layers:
            x, (h, c) = layer(x, edge_index, h, c)
        x = self.output_proj(x)
        return x[batch_index]

    def _edge_index(self, a):
        adj = a[0] if isinstance(a, (list, tuple)) else a
        edge_index, _ = adj_to_edge_index(adj, device=self.device)
        return edge_index

    def _run(self, x, a, t, b, optimizer=None):
        x = to_float_tensor(x, self.device)
        t = to_float_tensor(t, self.device)
        b = to_long_tensor(b, self.device)
        edge_index = self._edge_index(a)
        logits = self.forward(x, edge_index, b)
        loss = F.binary_cross_entropy_with_logits(logits, t)
        if optimizer is not None:
            loss = loss + self.weight_decay * self.l2_penalty()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(self.parameters(), 5.0)
            optimizer.step()
        pred, prob = pred_from_logits(logits, self.out_dim)
        acc = accuracy_from_logits(logits, t)
        return (loss.item(), acc.item(), pred.detach().cpu().numpy(),
                prob.detach().cpu().numpy(), t.detach().cpu().numpy())

    def fit_batch(self, x, a, t, b, learning_rate=1e-2, momentum=0.9, optimizer=None):
        self.train()
        return self._run(x, a, t, b, optimizer)[:4]

    def evaluate(self, x, a, t, b, learning_rate=1e-2, momentum=0.9):
        self.eval()
        with torch.no_grad():
            loss, acc, pred, prob, labels = self._run(x, a, t, b)
        tags = pred.argmax(axis=1) == np.asarray(labels).argmax(axis=1)
        return acc, pred, prob, tags
