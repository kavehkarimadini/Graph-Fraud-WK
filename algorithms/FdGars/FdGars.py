'''
This code is due to Yutong Deng (@yutongD), Yingtong Dou (@YingtongDou) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection)
https://github.com/safe-graph/DGFraud

FdGars ('FdGars: Fraudster Detection via Graph Convolutional Networks in Online App Review System')
'''
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../..')))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.base_algorithm import Algorithm
from base_models.models import GCN
from utils.torch_helpers import (accuracy_from_logits, adjs_to_edge_indices,
                                 pred_from_logits, to_float_tensor, to_long_tensor)


class FdGars(Algorithm):

    def __init__(self, nodes, class_size, gcn_output1, gcn_output2, meta,
                 embedding, encoding, weight_decay=0.01, device=None, **kwargs):
        super(FdGars, self).__init__(weight_decay=weight_decay)
        self.nodes = nodes
        self.meta = meta
        self.class_size = class_size
        self.device = device or torch.device('cpu')
        self.gcn = GCN(embedding, gcn_output1, encoding)
        self.classifier = nn.Linear(encoding, class_size)
        self.to(self.device)

    def forward(self, x, edge_packs, batch_index):
        edge_index, edge_weight = edge_packs[0]
        emb = self.gcn(x, edge_index, edge_weight)
        return self.classifier(emb[batch_index])

    def _prepare(self, x, a, t, b):
        x = to_float_tensor(x, self.device)
        t = to_float_tensor(t, self.device)
        b = to_long_tensor(b, self.device)
        edge_packs = adjs_to_edge_indices(a, device=self.device)
        return x, edge_packs, t, b

    def _run(self, x, a, t, b, optimizer=None):
        x, edge_packs, t, b = self._prepare(x, a, t, b)
        logits = self.forward(x, edge_packs, b)
        loss = F.binary_cross_entropy_with_logits(logits, t)
        if optimizer is not None:
            loss = loss + self.weight_decay * self.l2_penalty()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(self.parameters(), 5.0)
            optimizer.step()
        pred, prob = pred_from_logits(logits, self.class_size)
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
