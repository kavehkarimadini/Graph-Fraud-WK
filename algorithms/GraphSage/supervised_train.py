'''
Supervised GraphSAGE training with PyTorch Geometric.
Usage:
  python supervised_train.py --train_prefix ../../dataset --file_name YelpChi.mat --sigmoid
'''
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.typing import WITH_PYG_LIB, WITH_TORCH_SPARSE

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../..')))
from algorithms.GraphSage.models import SupervisedGraphsage
from algorithms.GraphSage.utils import load_mat_graph
from utils.evaluation import compute_classification_metrics, run_output_dir, save_run_outputs
from utils.torch_helpers import get_device, set_seed
from utils.training import add_save_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='graphsage_mean')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--train_prefix', type=str, default='../../dataset/')
    parser.add_argument('--file_name', type=str, default='YelpChi.mat')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--samples_1', type=int, default=25)
    parser.add_argument('--samples_2', type=int, default=10)
    parser.add_argument('--dim_1', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--sigmoid', action='store_true')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=123)
    add_save_args(parser)
    parser.set_defaults(test_size=0.2)
    return parser.parse_args()


def train_epoch_sampled(model, optimizer, loader, device):
    total_loss = steps = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index)[:batch.batch_size]
        loss = model.loss(logits, batch.y[:batch.batch_size])
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()
        steps += 1
    return total_loss / max(steps, 1)


def train_epoch_full(model, optimizer, x, edge_index, y, train_idx, batch_size):
    permutation = train_idx[torch.randperm(train_idx.size(0), device=train_idx.device)]
    total_loss = steps = 0
    for start in range(0, permutation.size(0), batch_size):
        batch = permutation[start:start + batch_size]
        optimizer.zero_grad()
        logits = model(x, edge_index)
        loss = model.loss(logits[batch], y[batch])
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()
        steps += 1
    return total_loss / max(steps, 1)


def predict_split(model, data, idx, sigmoid):
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        preds = model.predict(logits)
        y_true = data.y[idx].cpu().numpy()
        y_pred = preds[idx].cpu().numpy()
    metrics = compute_classification_metrics(y_true, y_pred, y_pred, sigmoid=sigmoid)
    return metrics, y_true, y_pred


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.gpu)

    x, edge_index, y, train_idx, test_idx, split_info = load_mat_graph(
        args.train_prefix, args.file_name, test_size=args.test_size, seed=args.seed)
    num_classes = int(y.max().item()) + 1

    data = Data(x=x.float(), edge_index=edge_index, y=y)
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[train_idx] = True
    data.test_mask[test_idx] = True
    data = data.to(device)

    model = SupervisedGraphsage(
        in_channels=data.num_features,
        hidden_channels=args.dim_1,
        num_classes=num_classes,
        dropout=args.dropout,
        sigmoid_loss=args.sigmoid,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    train_idx = train_idx.to(device)
    test_idx = test_idx.to(device)
    use_sampling = bool(WITH_PYG_LIB or WITH_TORCH_SPARSE)
    train_loader = None
    if use_sampling:
        train_loader = NeighborLoader(
            data, num_neighbors=[args.samples_1, args.samples_2],
            batch_size=args.batch_size, input_nodes=data.train_mask, shuffle=True)
    else:
        print('NeighborLoader needs pyg-lib or torch-sparse; using full-graph minibatches.')

    t_start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        if use_sampling:
            avg_loss = train_epoch_sampled(model, optimizer, train_loader, device)
        else:
            avg_loss = train_epoch_full(
                model, optimizer, data.x, data.edge_index, data.y,
                train_idx, args.batch_size)
        print('Epoch {:d} loss: {:.4f}'.format(epoch, avg_loss))

    train_time = time.perf_counter() - t_start
    print('train time=', '{:.5f}'.format(train_time))
    model.eval()
    train_metrics, _, _ = predict_split(model, data, train_idx, args.sigmoid)
    test_metrics, y_true, y_pred = predict_split(model, data, test_idx, args.sigmoid)
    print('Train metrics:', train_metrics)
    print('Test metrics:', test_metrics)

    if not args.no_save:
        config = {k: v for k, v in vars(args).items()}
        config['model'] = 'GraphSage'
        out_dir = run_output_dir(args.save_dir, 'GraphSage', args.file_name.replace('.mat', ''), args.seed)
        save_run_outputs(
            out_dir, 'GraphSage', config, split_info,
            test_metrics, model=model, train_metrics=train_metrics,
            predictions=y_pred, probabilities=y_pred,
            train_indices=train_idx.cpu().numpy(), test_indices=test_idx.cpu().numpy(),
            y_train=data.y[train_idx].cpu().numpy(), y_test=y_true,
            train_time=train_time,
        )


if __name__ == '__main__':
    main()
