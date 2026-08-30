'''
GraphConsis training with PyTorch Geometric.
Usage:
  python supervised_train.py --train_prefix ../../dataset/ --file_name YelpChi.mat --sigmoid --epochs 3
'''
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from algorithms.GraphConsis.supervised_models import SupervisedGraphconsis
from algorithms.GraphConsis.utils import load_data
from utils.evaluation import compute_classification_metrics, run_output_dir, save_run_outputs
from utils.torch_helpers import get_device, set_seed
from utils.training import add_save_args, resolve_test_size


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='graphsage_mean')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--train_prefix', type=str, default='../../dataset/')
    parser.add_argument('--file_name', type=str, default='YelpChi.mat')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--dim_1', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--sigmoid', action='store_true')
    parser.add_argument('--context_dim', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--samples_1', type=int, default=25)
    parser.add_argument('--samples_2', type=int, default=10)
    parser.add_argument('--run_tag', type=str, default=None,
                        help='Optional suffix on the output folder (e.g. a timestamp).')
    add_save_args(parser)
    return parser.parse_args()


def predict_split(model, x, edge_indices, y, idx, sigmoid):
    with torch.no_grad():
        logits = model(x, edge_indices)
        preds = model.predict(logits)
        y_true = y[idx].cpu().numpy()
        y_pred = preds[idx].cpu().numpy()
    metrics = compute_classification_metrics(y_true, y_pred, y_pred, sigmoid=sigmoid)
    return metrics, y_true, y_pred


def train(args=None):
    """Train GraphConsis. ``args`` is an argparse Namespace (see ``parse_args``)."""
    if args is None:
        args = parse_args()
    set_seed(args.seed)
    device = get_device(args.gpu)
    dataset_name = os.path.splitext(os.path.basename(args.file_name))[0]
    if getattr(args, 'dataset_str', None) is None:
        args.dataset_str = dataset_name
    if args.test_size is None:
        args.test_size = resolve_test_size(args, 'GraphConsis')
    x, edge_indices, y, train_idx, test_idx, split_info = load_data(
        args.train_prefix, args.file_name, test_size=args.test_size, seed=args.seed)
    x = x.to(device)
    y = y.to(device)
    edge_indices = [e.to(device) for e in edge_indices]
    train_idx = train_idx.to(device)
    test_idx = test_idx.to(device)
    num_classes = int(y.max().item()) + 1

    model = SupervisedGraphconsis(
        in_channels=x.size(1),
        hidden_channels=args.dim_1,
        num_classes=num_classes,
        num_relations=len(edge_indices),
        context_dim=args.context_dim,
        dropout=args.dropout,
        sigmoid_loss=args.sigmoid,
        num_nodes=x.size(0),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    t_start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        permutation = train_idx[torch.randperm(train_idx.size(0), device=device)]
        total_loss = steps = 0
        for start in range(0, permutation.size(0), args.batch_size):
            batch = permutation[start:start + args.batch_size]
            optimizer.zero_grad()
            loss = model.loss(model(x, edge_indices)[batch], y[batch])
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        print('Epoch {:d} loss: {:.4f}'.format(epoch, total_loss / max(steps, 1)))

    print('train time=', '{:.5f}'.format(time.perf_counter() - t_start))
    model.eval()
    train_metrics, _, _ = predict_split(model, x, edge_indices, y, train_idx, args.sigmoid)
    test_metrics, y_true, y_pred = predict_split(model, x, edge_indices, y, test_idx, args.sigmoid)
    print('Train metrics:', train_metrics)
    print('Test metrics:', test_metrics)

    out_dir = None
    if not args.no_save:
        config = {k: v for k, v in vars(args).items()}
        config['model'] = 'GraphConsis'
        out_dir = run_output_dir(
            args.save_dir, 'GraphConsis', dataset_name, args.seed,
            run_tag=getattr(args, 'run_tag', None))
        save_run_outputs(
            out_dir, 'GraphConsis', config, split_info,
            test_metrics, model=model, train_metrics=train_metrics,
            predictions=y_pred, probabilities=y_pred,
            train_indices=train_idx.cpu().numpy(), test_indices=test_idx.cpu().numpy(),
            y_train=y[train_idx].cpu().numpy(), y_test=y_true,
        )
    return test_metrics, out_dir


def main():
    train(parse_args())


if __name__ == '__main__':
    main()
