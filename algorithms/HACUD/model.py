'''
This code is due to Hengrui Zhang (@hengruizhang98) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection)
https://github.com/safe-graph/DGFraud
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import from_scipy_sparse_matrix


def _sparse_mm(adj, x):
    if adj.is_sparse:
        return torch.sparse.mm(adj.coalesce(), x)
    return torch.matmul(adj, x)


class Model(nn.Module):
    def __init__(self, data_config, pretrain_data, args):
        super(Model, self).__init__()
        self.n_nodes = data_config['n_nodes']
        self.n_metapath = data_config['n_metapath']
        self.n_class = data_config['n_class']
        self.n_fc = args.n_fc
        self.fc_sizes = eval(args.fc)
        self.reg = args.reg
        self.emb_dim = args.embed_size
        features = data_config['features']
        if hasattr(features, 'todense'):
            features = np.asarray(features.todense())
        self.register_buffer('features', torch.tensor(np.asarray(features), dtype=torch.float32))
        self.f_dim = self.features.shape[1]
        self.lr = args.lr

        adj_tensors = []
        for adj in data_config['norm_adj']:
            if hasattr(adj, 'tocoo'):
                edge_index, edge_weight = from_scipy_sparse_matrix(adj)
                sparse = torch.sparse_coo_tensor(
                    edge_index, edge_weight.float(), torch.Size(adj.shape))
            else:
                sparse = torch.tensor(np.asarray(adj), dtype=torch.float32)
            adj_tensors.append(sparse)
        for i, adj in enumerate(adj_tensors):
            self.register_buffer('adj_{}'.format(i), adj)

        self.W = nn.Linear(self.f_dim, self.emb_dim)
        self.W_rho = nn.ModuleList([nn.Linear(self.f_dim, self.emb_dim) for _ in range(self.n_metapath)])
        self.W_f = nn.ModuleList([nn.Linear(2 * self.emb_dim, self.emb_dim) for _ in range(self.n_metapath)])
        self.W_f1 = nn.Linear(2 * self.emb_dim, self.emb_dim)
        self.W_f2 = nn.Linear(self.emb_dim, self.emb_dim)
        self.z = nn.ParameterList([
            nn.Parameter(torch.empty(1, self.emb_dim * self.n_metapath))
            for _ in range(self.n_metapath)
        ])
        for z in self.z:
            nn.init.xavier_uniform_(z)

        dims = [self.emb_dim] + self.fc_sizes
        self.mlp = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(self.n_fc)])

    def _adj(self, n):
        return getattr(self, 'adj_{}'.format(n))

    def create_embedding(self):
        embeddings = self.features
        h = self.W(embeddings)
        f_tilde = []
        for n in range(self.n_metapath):
            neigh = _sparse_mm(self._adj(n), embeddings)
            h_u = self.W_rho[n](neigh)
            f_u = F.relu(self.W_f[n](torch.cat([h, h_u], dim=1)))
            v_u = F.relu(self.W_f1(torch.cat([h, f_u], dim=1)))
            alp_hat = F.softmax(F.relu(self.W_f2(v_u)), dim=1)
            f_tilde.append(alp_hat * f_u)
        f_c = torch.cat(f_tilde, dim=1)
        betas = []
        stacked = []
        for n in range(self.n_metapath):
            betas.append(torch.matmul(f_c, self.z[n].t()))
            stacked.append(f_tilde[n].unsqueeze(-1))
        beta_u = F.softmax(torch.cat(betas, dim=1), dim=1).unsqueeze(1)
        f = torch.cat(stacked, dim=2)
        return (beta_u * f).sum(dim=2)

    def pred(self, x):
        for n, layer in enumerate(self.mlp):
            x = layer(x)
            if n != self.n_fc - 1:
                x = F.relu(x)
        return x

    def forward(self, nodes):
        embeddings = self.create_embedding()
        return self.pred(embeddings[nodes])

    def create_loss(self, logits, labels):
        target = labels.argmax(dim=-1) if labels.dim() > 1 else labels.long()
        ce_loss = F.cross_entropy(logits, target, reduction='sum')
        reg_loss = sum(p.pow(2.0).sum() for p in self.parameters() if p.requires_grad)
        return ce_loss + self.reg * reg_loss, ce_loss, reg_loss

    def fit_batch(self, optimizer, nodes, labels):
        self.train()
        optimizer.zero_grad()
        logits = self.forward(nodes)
        loss, ce_loss, reg_loss = self.create_loss(logits, labels)
        loss.backward()
        optimizer.step()
        return loss.item(), ce_loss.item(), reg_loss.item()

    def eval_batch(self, nodes, labels):
        self.eval()
        with torch.no_grad():
            logits = self.forward(nodes)
            loss, ce_loss, reg_loss = self.create_loss(logits, labels)
        return loss.item(), ce_loss.item(), reg_loss.item(), logits.cpu().numpy()
