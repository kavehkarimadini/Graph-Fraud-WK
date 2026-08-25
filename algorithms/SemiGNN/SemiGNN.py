'''
This code is due to Yutong Deng (@yutongD), Yingtong Dou (@YingtongDou) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection)
https://github.com/safe-graph/DGFraud

SemiGNN ('A Semi-supervised Graph Attentive Network for Financial Fraud Detection')
'''
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../..')))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.base_algorithm import Algorithm
from base_models.layers import NodeAttention, ViewAttention
from utils.torch_helpers import accuracy_from_logits, pred_from_logits, to_float_tensor, to_long_tensor


class SemiGNN(Algorithm):

    def __init__(self, nodes, class_size, semi_encoding1, semi_encoding2, semi_encoding3,
                 init_emb_size, meta, ul, alpha, lamtha, weight_decay=0.01, device=None, **kwargs):
        super(SemiGNN, self).__init__(weight_decay=weight_decay)
        self.nodes = nodes
        self.meta = meta
        self.class_size = class_size
        self.ul = ul
        self.alpha = alpha
        self.lamtha = lamtha
        self.device = device or torch.device('cpu')
        self.init_emb = nn.Parameter(torch.empty(nodes, init_emb_size))
        nn.init.xavier_uniform_(self.init_emb)
        self.node_attns = nn.ModuleList([NodeAttention(init_emb_size) for _ in range(meta)])
        self.view_attn = ViewAttention(semi_encoding1, semi_encoding2, 2, meta, init_emb_size)
        self.mlp = nn.Linear(semi_encoding2 * meta, semi_encoding3)
        self.theta = nn.Linear(semi_encoding3, class_size, bias=False)
        self.to(self.device)

    def encode(self, adjs):
        h1 = [attn(self.init_emb, adj) for attn, adj in zip(self.node_attns, adjs)]
        h2 = self.view_attn(h1)
        return self.mlp(h2)

    def forward(self, adjs, batch_index):
        a_u = self.encode(adjs)
        return self.theta(a_u[batch_index]), a_u

    def _run(self, a, u_i, u_j, graph_label, batch_data, batch_sup_label, optimizer=None):
        adjs = [to_float_tensor(adj, self.device) for adj in a]
        batch_index = to_long_tensor(batch_data, self.device)
        sup_label = to_float_tensor(batch_sup_label, self.device)
        u_i = to_long_tensor(u_i, self.device)
        u_j = to_long_tensor(u_j, self.device)
        graph_label = to_float_tensor(graph_label, self.device).view(-1)

        logits, a_u = self.forward(adjs, batch_index)
        loss1 = -(1.0 / max(int(self.ul), 1)) * torch.sum(
            sup_label * torch.log(torch.softmax(logits, dim=-1) + 1e-12))
        inner = (a_u[u_i] * a_u[u_j]).sum(dim=1)
        loss2 = -torch.log(torch.sigmoid(graph_label * inner) + 1e-12).mean()
        loss = self.alpha * loss1 + (1 - self.alpha) * loss2
        if optimizer is not None:
            loss = loss + self.lamtha * self.l2_penalty()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(self.parameters(), 5.0)
            optimizer.step()
        pred, prob = pred_from_logits(logits, self.class_size)
        acc = accuracy_from_logits(logits, sup_label)
        return (loss.item(), acc.item(), pred.detach().cpu().numpy(),
                prob.detach().cpu().numpy(), sup_label.detach().cpu().numpy())

    def fit_batch(self, a, u_i, u_j, batch_graph_label, batch_data, batch_sup_label,
                  learning_rate=1e-2, momentum=0.9, optimizer=None):
        self.train()
        return self._run(a, u_i, u_j, batch_graph_label, batch_data, batch_sup_label, optimizer)[:4]

    def evaluate(self, a, u_i, u_j, batch_graph_label, batch_data, batch_sup_label,
                 learning_rate=1e-2, momentum=0.9):
        self.eval()
        with torch.no_grad():
            loss, acc, pred, prob, labels = self._run(
                a, u_i, u_j, batch_graph_label, batch_data, batch_sup_label)
        tags = pred.argmax(axis=1) == np.asarray(labels).argmax(axis=1)
        return acc, pred, prob, tags
