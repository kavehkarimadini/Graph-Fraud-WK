'''
Train BWGNN (Tang et al., ICML 2022) with unified evaluation artifacts.

Usage (from algorithms/BWGNN):
  python BWGNN_main.py --dataset tfinance --gpu 0
  python BWGNN_main.py --dataset tsocial --hid_dim 10 --order 5 --gpu 0
'''
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from algorithms.BWGNN.BWGNN import BWGNN
from utils.data_loader import (
    is_comp_name, load_comp_graph, load_data_amazon, load_data_yelpchi,
    load_elliptic_graph, load_tgraph_dgl, scipy_to_dgl,
)
from utils.evaluation import compute_classification_metrics, run_output_dir, save_run_outputs
from utils.split import split_report, stratified_train_test_split
from utils.torch_helpers import get_device, set_seed
from utils.training import add_save_args, resolve_test_size


def parse_args():
    parser = argparse.ArgumentParser(description='BWGNN')
    parser.add_argument('--dataset', type=str, default='tfinance',
                        help="tfinance / tsocial / yelpchi / amazon / elliptic / comp")
    parser.add_argument('--train_ratio', type=float, default=None,
                        help='Train fraction (overrides --test_size). Paper uses 0.4.')
    parser.add_argument('--hid_dim', type=int, default=64)
    parser.add_argument('--order', type=int, default=2, help='Beta wavelet order C')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--data_prefix', type=str, default='../../dataset/')
    parser.add_argument('--run_tag', type=str, default=None,
                        help='Optional suffix on the output folder (e.g. a timestamp).')
    add_save_args(parser)
    return parser.parse_args()


def load_bwgnn_graph(args):
    ds = args.dataset.lower()
    prefix = args.data_prefix
    if ds in ('tfinance', 'tsocial'):
        return load_tgraph_dgl(ds, prefix=prefix)
    if ds in ('yelp', 'yelpchi'):
        adjs, features, train_idx, y_train, test_idx, y_test = load_data_yelpchi(
            os.path.join(prefix, 'YelpChi.mat'),
            test_size=args.test_size, seed=args.seed)
        labels = np.zeros(features.shape[0], dtype=np.int64)
        labels[train_idx] = y_train.argmax(axis=1)
        labels[test_idx] = y_test.argmax(axis=1)
        return scipy_to_dgl(adjs[0], features, labels)
    if ds == 'amazon':
        adjs, features, train_idx, y_train, test_idx, y_test = load_data_amazon(
            os.path.join(prefix, 'Amazon.mat'),
            test_size=args.test_size, seed=args.seed)
        labels = np.zeros(features.shape[0], dtype=np.int64)
        labels[train_idx] = y_train.argmax(axis=1)
        labels[test_idx] = y_test.argmax(axis=1)
        return scipy_to_dgl(adjs[0], features, labels)
    if ds in ('elliptic', 'elliptic_bitcoin', 'elliptic_bitcoin_dataset'):
        return load_elliptic_graph(prefix=prefix)
    if is_comp_name(ds):
        return load_comp_graph(ds, prefix=prefix)
    raise ValueError(
        "Unknown dataset '{}'. Use tfinance, tsocial, yelpchi, amazon, elliptic, or comp.".format(
            args.dataset))


def split_graph_labels(labels, test_size, seed):
    y = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
    train_idx, test_idx = stratified_train_test_split(y, test_size=test_size, seed=seed)
    info = split_report(y, train_idx, test_idx)
    info.update({'test_fraction': float(test_size), 'seed': int(seed), 'stratified': True})
    return (torch.as_tensor(train_idx, dtype=torch.long),
            torch.as_tensor(test_idx, dtype=torch.long), info)


def class_weight(labels, train_idx, device):
    y = labels[train_idx]
    n_pos = y.float().sum().clamp(min=1.0)
    n_neg = (y.numel() - n_pos).clamp(min=1.0)
    return torch.tensor([1.0, (n_neg / n_pos).item()], device=device)


def train(args=None):
    """Train BWGNN. ``args`` is an argparse Namespace (see ``parse_args``)."""
    if args is None:
        args = parse_args()
    args.dataset_str = args.dataset
    if args.train_ratio is not None:
        args.test_size = 1.0 - args.train_ratio
    elif args.test_size is None:
        args.test_size = resolve_test_size(args, 'BWGNN')
    set_seed(args.seed)
    device = get_device(args.gpu)

    graph = load_bwgnn_graph(args)
    labels_np = graph.ndata['label'].cpu().numpy()
    train_idx, test_idx, split_info = split_graph_labels(labels_np, args.test_size, args.seed)

    graph = graph.to(device)
    features = graph.ndata['feature']
    labels = graph.ndata['label']
    train_idx = train_idx.to(device)
    test_idx = test_idx.to(device)

    in_feats = features.size(1)
    num_classes = int(labels.max().item()) + 1
    model = BWGNN(in_feats, args.hid_dim, num_classes, graph, d=args.order).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    weight = class_weight(labels, train_idx, device)
    print('cross entropy weight:', weight.tolist())

    t_start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        logits = model(features)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx], weight=weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch {:d} loss: {:.4f}'.format(epoch, loss.item()))
    train_time = time.perf_counter() - t_start
    print('train time=', '{:.5f}'.format(train_time))

    model.eval()
    with torch.no_grad():
        logits = model(features)
        prob = torch.softmax(logits, dim=-1)
        pred = torch.nn.functional.one_hot(prob.argmax(dim=-1), num_classes=num_classes).float()

    y_train = labels[train_idx].cpu().numpy()
    y_test = labels[test_idx].cpu().numpy()
    train_metrics = compute_classification_metrics(
        y_train, pred[train_idx].cpu().numpy(), prob[train_idx].cpu().numpy(), sigmoid=False)
    test_metrics = compute_classification_metrics(
        y_test, pred[test_idx].cpu().numpy(), prob[test_idx].cpu().numpy(), sigmoid=False)
    print('Train metrics:', train_metrics)
    print('Test metrics:', test_metrics)

    out_dir = None
    if not args.no_save:
        config = {k: v for k, v in vars(args).items()}
        config['model'] = 'BWGNN'
        out_dir = run_output_dir(
            args.save_dir, 'BWGNN', args.dataset, args.seed,
            run_tag=getattr(args, 'run_tag', None))
        save_run_outputs(
            out_dir, 'BWGNN', config, split_info,
            test_metrics, model=model, train_metrics=train_metrics,
            predictions=pred[test_idx].cpu().numpy(),
            probabilities=prob[test_idx].cpu().numpy(),
            train_indices=train_idx.cpu().numpy(), test_indices=test_idx.cpu().numpy(),
            y_train=y_train, y_test=y_test,
            train_time=train_time,
        )
    return test_metrics, out_dir


def main():
    train(parse_args())


if __name__ == '__main__':
    main()
