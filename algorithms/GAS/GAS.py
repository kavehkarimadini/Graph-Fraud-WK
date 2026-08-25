'''
This code is due to Yutong Deng (@yutongD), Yingtong Dou (@YingtongDou) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection)
https://github.com/safe-graph/DGFraud

GAS ('Spam Review Detection with Graph Convolutional Networks')
'''
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../..')))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.base_algorithm import Algorithm
from base_models.layers import AttentionAggregator, ConcatenationAggregator, GASConcatenation
from utils.torch_helpers import accuracy_from_logits, pred_from_logits, to_float_tensor, to_long_tensor


class GAS(Algorithm):

    def __init__(self, nodes, class_size, embedding_i, embedding_u, embedding_r,
                 h_u_size, h_i_size, encoding1, encoding2, encoding3, encoding4,
                 gcn_dim, meta=1, concat=True, weight_decay=0.01, device=None, **kwargs):
        super(GAS, self).__init__(weight_decay=weight_decay)
        self.nodes = nodes
        self.class_size = class_size
        self.device = device or torch.device('cpu')
        self.r_aggregator = ConcatenationAggregator(
            embedding_r + embedding_u + embedding_i, encoding1)
        self.iu_aggregator = AttentionAggregator(
            h_u_size, h_i_size, encoding3, encoding2, concat=concat)
        self.concatenator = GASConcatenation()
        self.classifier = nn.LazyLinear(class_size)
        self.to(self.device)

    def forward(self, review_vecs, user_vecs, item_vecs, adj_info, batch_index):
        user_review_adj, user_item_adj, item_review_adj, item_user_adj, \
            review_user_adj, review_item_adj, homo_adj = adj_info
        h_r = self.r_aggregator(review_vecs, user_vecs, item_vecs,
                                review_user_adj, review_item_adj)
        h_u, h_i = self.iu_aggregator(
            review_vecs, user_vecs, item_vecs,
            user_review_adj, user_item_adj, item_review_adj, item_user_adj)
        concated_hr = self.concatenator(h_r, h_i, h_u, homo_adj,
                                        review_item_adj, review_user_adj)
        return self.classifier(concated_hr[batch_index])

    def _prepare(self, h, adj_info, t, b):
        review_vecs = to_float_tensor(h[0], self.device)
        user_vecs = to_float_tensor(h[1], self.device)
        item_vecs = to_float_tensor(h[2], self.device)
        t = to_float_tensor(t, self.device)
        b = to_long_tensor(b, self.device)
        packed = [
            to_long_tensor(adj_info[0], self.device),
            to_long_tensor(adj_info[1], self.device),
            to_long_tensor(adj_info[2], self.device),
            to_long_tensor(adj_info[3], self.device),
            to_long_tensor(adj_info[4], self.device),
            to_long_tensor(adj_info[5], self.device),
            to_float_tensor(adj_info[6], self.device),
        ]
        return review_vecs, user_vecs, item_vecs, packed, t, b

    def _run(self, h, adj_info, t, b, optimizer=None):
        review_vecs, user_vecs, item_vecs, packed, t, b = self._prepare(h, adj_info, t, b)
        logits = self.forward(review_vecs, user_vecs, item_vecs, packed, b)
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

    def fit_batch(self, h, adj_info, t, b, learning_rate=1e-2, momentum=0.9, optimizer=None):
        self.train()
        return self._run(h, adj_info, t, b, optimizer)[:4]

    def evaluate(self, h, adj_info, t, b):
        self.eval()
        with torch.no_grad():
            loss, acc, pred, prob, labels = self._run(h, adj_info, t, b)
        tags = pred.argmax(axis=1) == np.asarray(labels).argmax(axis=1)
        return acc, pred, prob, tags
