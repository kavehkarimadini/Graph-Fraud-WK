'''
This code is due to Yutong Deng (@yutongD), Yingtong Dou (@YingtongDou) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection)
https://github.com/safe-graph/DGFraud

GEM ('Heterogeneous Graph Neural Networks for Malicious Account Detection')
'''
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../..')))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.base_algorithm import Algorithm
from base_models.layers import GEMLayer
from utils.torch_helpers import to_float_tensor, to_long_tensor


class GEM(Algorithm):

    def __init__(self, nodes, class_size, meta, embedding, encoding, hop,
                 weight_decay=0.01, device=None, **kwargs):
        super(GEM, self).__init__(weight_decay=weight_decay)
        self.nodes = nodes
        self.meta = meta
        self.class_size = class_size
        self.encoding = encoding
        self.device = device or torch.device('cpu')
        self.h0 = nn.Parameter(torch.empty(nodes, encoding))
        nn.init.xavier_uniform_(self.h0)
        self.layers = nn.ModuleList([
            GEMLayer(embedding, encoding, meta) for _ in range(hop)
        ])
        self.classifier = nn.Linear(encoding, class_size)
        self.to(self.device)

    def forward(self, x, adjs, batch_index):
        h = self.h0
        for layer in self.layers:
            h = layer(x, h, adjs)
        return self.classifier(h[batch_index])

    def _prepare(self, x, a, t, b):
        x = to_float_tensor(x, self.device)
        t = to_float_tensor(t, self.device)
        b = to_long_tensor(b, self.device)
        adjs = [to_float_tensor(adj, self.device) for adj in a]
        return x, adjs, t, b

    def _run(self, x, a, t, b, optimizer=None):
        x, adjs, t, b = self._prepare(x, a, t, b)
        logits = self.forward(x, adjs, b)
        loss = F.binary_cross_entropy_with_logits(logits, t)
        if optimizer is not None:
            loss = loss + self.weight_decay * self.l2_penalty()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(self.parameters(), 5.0)
            optimizer.step()
        prob = torch.sigmoid(logits)
        pred = (prob > 0.5).float()
        acc = (pred == t).float().mean()
        return (loss.item(), acc.item(), pred.detach().cpu().numpy(),
                prob.detach().cpu().numpy(), t.detach().cpu().numpy())

    def fit_batch(self, x, a, t, b, learning_rate=1e-2, momentum=0.9, optimizer=None):
        self.train()
        return self._run(x, a, t, b, optimizer)[:4]

    def evaluate(self, x, a, t, b, learning_rate=1e-2, momentum=0.9):
        self.eval()
        with torch.no_grad():
            loss, acc, pred, prob, labels = self._run(x, a, t, b)
        tags = pred == np.asarray(labels)
        return acc, pred, prob, tags
