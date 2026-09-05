'''
Train SplitGNN (Wu et al., CIKM 2023) with unified evaluation artifacts.

Usage (from algorithms/SplitGNN):
  python SplitGNN_main.py --dataset comp --gpu 0
  python SplitGNN_main.py --dataset yelpchi --gpu 0
  python SplitGNN_main.py --dataset amazon --gpu 0
  python SplitGNN_main.py --dataset elliptic --gpu 0
'''
from __future__ import print_function

import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from algorithms.SplitGNN.SplitGNN import SplitGNN
from utils.data_loader import load_splitgnn_graph
from utils.evaluation import compute_classification_metrics, run_output_dir, save_run_outputs
from utils.torch_helpers import get_device, set_seed
from utils.training import add_save_args

# Paper table (README "Reproduce Results"), not the placeholder yaml values.
DATASET_DEFAULTS = {
    'comp': dict(learning_rate=0.001, C=3, K=2, gamma=0.8, intra_dim=8,
                 dropout=0.1, weight_decay=5e-5, epochs=1000, early_stop=100),
    'fdcompcn': dict(learning_rate=0.001, C=3, K=2, gamma=0.8, intra_dim=8,
                     dropout=0.1, weight_decay=5e-5, epochs=1000, early_stop=100),
    'yelpchi': dict(learning_rate=0.01, C=2, K=1, gamma=0.6, intra_dim=8,
                    dropout=0.1, weight_decay=5e-5, epochs=1000, early_stop=100),
    'yelp': dict(learning_rate=0.01, C=2, K=1, gamma=0.6, intra_dim=8,
                 dropout=0.1, weight_decay=5e-5, epochs=1000, early_stop=100),
    'amazon': dict(learning_rate=0.1, C=2, K=1, gamma=0.4, intra_dim=8,
                   dropout=0.1, weight_decay=5e-5, epochs=1000, early_stop=50),
    'elliptic': dict(learning_rate=0.01, C=2, K=1, gamma=0.6, intra_dim=8,
                     dropout=0.1, weight_decay=5e-5, epochs=1000, early_stop=50),
    'tfinance': dict(learning_rate=0.01, C=2, K=1, gamma=0.6, intra_dim=8,
                     dropout=0.1, weight_decay=5e-5, epochs=1000, early_stop=50),
    'tsocial': dict(learning_rate=0.01, C=2, K=1, gamma=0.6, intra_dim=8,
                    dropout=0.1, weight_decay=5e-5, epochs=200, early_stop=50),
}


def parse_args():
    parser = argparse.ArgumentParser(description='SplitGNN')
    parser.add_argument('--dataset', type=str, default='comp',
                        help='comp / yelpchi / amazon / elliptic / tfinance / tsocial')
    parser.add_argument('--train_ratio', type=float, default=0.4,
                        help='Train fraction of labeled nodes (max 0.4). Valid=20%% and test=40%% are fixed.')
    parser.add_argument('--use_file_masks', action='store_true',
                        help='Use train/valid/test masks stored in comp.dgl instead of the 40/20/40 protocol.')
    parser.add_argument('--intra_dim', type=int, default=None)
    parser.add_argument('--C', type=int, default=None, help='Beta wavelet order')
    parser.add_argument('--K', type=int, default=None, help='Positive-band cutoff')
    parser.add_argument('--gamma', type=float, default=None, help='Hinge-loss weight')
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--early_stop', type=int, default=None,
                        help='Patience on valid AUC (0 disables).')
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=30)
    parser.add_argument('--data_prefix', type=str, default='../../dataset/')
    parser.add_argument('--run_tag', type=str, default=None)
    add_save_args(parser)
    return parser.parse_args()


def apply_dataset_defaults(args):
    ds = args.dataset.lower().replace('.dgl', '').replace('.mat', '')
    if ds == 'fdcompcn':
        ds = 'comp'
    if ds in ('elliptic_bitcoin', 'elliptic_bitcoin_dataset'):
        ds = 'elliptic'
    defaults = DATASET_DEFAULTS.get(ds, DATASET_DEFAULTS['comp'])
    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    if ds in ('comp', 'fdcompcn'):
        args.dataset_str = 'comp'
    elif ds in ('elliptic', 'elliptic_bitcoin', 'elliptic_bitcoin_dataset'):
        args.dataset_str = 'elliptic'
    else:
        args.dataset_str = ds
    return args


def _split_auc(logits, labels, idx):
    if idx is None or idx.numel() == 0:
        return None
    y = labels[idx].detach().cpu().numpy()
    if len(np.unique(y)) < 2:
        return None
    prob = torch.softmax(logits[idx], dim=-1)[:, 1].detach().cpu().numpy()
    return float(roc_auc_score(y, prob))


def train(args=None):
    """Train SplitGNN. ``args`` is an argparse Namespace (see ``parse_args``)."""
    if args is None:
        args = parse_args()
    args = apply_dataset_defaults(args)
    set_seed(args.seed)
    device = get_device(args.gpu)

    graph, train_idx, valid_idx, test_idx, split_info = load_splitgnn_graph(
        args.dataset, prefix=args.data_prefix, train_ratio=args.train_ratio,
        seed=args.seed, use_file_masks=getattr(args, 'use_file_masks', False))
    graph = graph.to(device)
    labels = graph.ndata['label']
    train_idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    valid_idx_t = (torch.as_tensor(valid_idx, dtype=torch.long, device=device)
                   if valid_idx is not None and len(valid_idx) else None)
    test_idx_t = torch.as_tensor(test_idx, dtype=torch.long, device=device)

    in_feats = graph.ndata['feature'].size(1)
    n_class = int(labels.max().item()) + 1
    model = SplitGNN(
        in_feats, args.intra_dim, n_class, graph.etypes,
        args.dropout, args.C, args.K, args.gamma).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    print('SplitGNN dataset={} nodes={} train={} valid={} test={} C={} K={} gamma={}'.format(
        args.dataset_str, graph.num_nodes(), int(train_idx_t.numel()),
        int(valid_idx_t.numel()) if valid_idx_t is not None else 0,
        int(test_idx_t.numel()), args.C, args.K, args.gamma))

    best_state = None
    best_auc = -1.0
    best_epoch = 0
    stop_steps = 0
    t_start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        loss = model.loss(graph)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        do_stop = False
        if valid_idx_t is not None and args.early_stop:
            model.eval()
            with torch.no_grad():
                logits = model(graph)
            auc = _split_auc(logits, labels, valid_idx_t)
            if auc is not None:
                if auc > best_auc:
                    best_auc = auc
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
                    stop_steps = 1
                else:
                    stop_steps += 1
                    if stop_steps >= args.early_stop:
                        do_stop = True
                if epoch % 10 == 0:
                    print('Epoch {:d} loss: {:.4f} valid AUC: {:.4f} (best {:.4f} @ {:d})'.format(
                        epoch, loss.item(), auc, best_auc, best_epoch))
            elif epoch % 10 == 0:
                print('Epoch {:d} loss: {:.4f}'.format(epoch, loss.item()))
        elif epoch % 10 == 0:
            print('Epoch {:d} loss: {:.4f}'.format(epoch, loss.item()))
        if do_stop:
            print('Early stop at epoch {:d} (best valid AUC {:.4f} @ {:d})'.format(
                epoch, best_auc, best_epoch))
            break
    train_time = time.perf_counter() - t_start
    print('train time=', '{:.5f}'.format(train_time))

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(graph)
        prob = torch.softmax(logits, dim=-1)
        pred = torch.nn.functional.one_hot(prob.argmax(dim=-1), num_classes=n_class).float()

    y_train = labels[train_idx_t].cpu().numpy()
    y_test = labels[test_idx_t].cpu().numpy()
    train_metrics = compute_classification_metrics(
        y_train, pred[train_idx_t].cpu().numpy(), prob[train_idx_t].cpu().numpy(), sigmoid=False)
    test_metrics = compute_classification_metrics(
        y_test, pred[test_idx_t].cpu().numpy(), prob[test_idx_t].cpu().numpy(), sigmoid=False)
    if valid_idx_t is not None:
        y_valid = labels[valid_idx_t].cpu().numpy()
        valid_metrics = compute_classification_metrics(
            y_valid, pred[valid_idx_t].cpu().numpy(), prob[valid_idx_t].cpu().numpy(), sigmoid=False)
        test_metrics = dict(test_metrics)
        test_metrics['valid_auc'] = valid_metrics.get('auroc')
        print('Valid metrics:', valid_metrics)
    print('Train metrics:', train_metrics)
    print('Test metrics:', test_metrics)

    out_dir = None
    if not args.no_save:
        config = {k: v for k, v in vars(args).items()}
        config['model'] = 'SplitGNN'
        config['best_valid_auc'] = best_auc if best_auc >= 0 else None
        config['best_epoch'] = best_epoch
        out_dir = run_output_dir(
            args.save_dir, 'SplitGNN', args.dataset_str, args.seed,
            run_tag=getattr(args, 'run_tag', None))
        save_run_outputs(
            out_dir, 'SplitGNN', config, split_info,
            test_metrics, model=model, train_metrics=train_metrics,
            predictions=pred[test_idx_t].cpu().numpy(),
            probabilities=prob[test_idx_t].cpu().numpy(),
            train_indices=train_idx_t.cpu().numpy(), test_indices=test_idx_t.cpu().numpy(),
            y_train=y_train, y_test=y_test,
            train_time=train_time,
        )
    return test_metrics, out_dir


def main():
    train(parse_args())


if __name__ == '__main__':
    main()
