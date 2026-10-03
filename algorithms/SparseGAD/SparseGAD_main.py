'''
Train SparseGAD (Gong et al., IJCAI 2023) with unified evaluation artifacts.

Usage (from algorithms/SparseGAD):
  python SparseGAD_main.py --dataset yelpchi --gpu 0
  python SparseGAD_main.py --dataset amazon --gpu 0
  python SparseGAD_main.py --dataset elliptic --gpu 0
  python SparseGAD_main.py --dataset tfinance --gpu 0
  python SparseGAD_main.py --dataset tsocial --gpu 0
  python SparseGAD_main.py --dataset comp --gpu 0
'''
from __future__ import print_function

import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from algorithms.SparseGAD.SparseGAD import (
    SparseGAD, augment_graph, build_union_graph, knn_edges, sym_normalize)
from utils.data_loader import load_dga_arrays
from utils.evaluation import compute_classification_metrics, run_output_dir, save_run_outputs
from utils.split import protocol_split
from utils.torch_helpers import get_device, set_seed
from utils.training import add_protocol_split_args, add_save_args

# Reference scripts/*_train_0.4.sh (lr 0.01, weight decay 5e-4, 2 layers, kNN
# k=10, graph update every 200 epochs, up to 1000 epochs). YelpChi: hidden 32,
# dropout 0.1, edge dropout 0.25, min-max features; Amazon: hidden 16, dropout
# 0.2, edge dropout 0.1, row-normalised features. Other datasets use the
# YelpChi settings with standard scaling; T-Social skips the O(n^2) kNN update.
DATASET_DEFAULTS = {
    'yelpchi': dict(hidden=32, dropout=0.1, dropout_adj=0.25, feat_norm='minmax',
                    epochs=1000, early_stop=200, update_adj_epoch=200),
    'amazon': dict(hidden=16, dropout=0.2, dropout_adj=0.1, feat_norm='row',
                   epochs=1000, early_stop=200, update_adj_epoch=200),
    'elliptic': dict(hidden=32, dropout=0.1, dropout_adj=0.25, feat_norm='standard',
                     epochs=1000, early_stop=200, update_adj_epoch=200),
    'tfinance': dict(hidden=32, dropout=0.1, dropout_adj=0.25, feat_norm='standard',
                     epochs=1000, early_stop=200, update_adj_epoch=200),
    'tsocial': dict(hidden=32, dropout=0.1, dropout_adj=0.25, feat_norm='standard',
                    epochs=300, early_stop=50, update_adj_epoch=0),
    'comp': dict(hidden=32, dropout=0.1, dropout_adj=0.25, feat_norm='standard',
                 epochs=1000, early_stop=200, update_adj_epoch=200),
}


def parse_args():
    parser = argparse.ArgumentParser(description='SparseGAD')
    parser.add_argument('--dataset', type=str, default='yelpchi',
                        help='yelpchi / amazon / elliptic / tfinance / tsocial / comp')
    parser.add_argument('--hidden', type=int, default=None)
    parser.add_argument('--nlayers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=None, help='Feature dropout (dropout2).')
    parser.add_argument('--dropout_adj', type=float, default=None, help='Edge-weight dropout (dropout_adj2).')
    parser.add_argument('--k', type=int, default=10, help='Neighbours per node in the kNN graph update.')
    parser.add_argument('--update_adj_epoch', type=int, default=None,
                        help='Add a kNN graph every this many epochs (0 disables).')
    parser.add_argument('--feat_norm', type=str, default=None,
                        choices=['row', 'minmax', 'standard', 'none'])
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--data_prefix', type=str, default='../../dataset/')
    parser.add_argument('--run_tag', type=str, default=None)
    add_save_args(parser)
    add_protocol_split_args(parser, early_stop=None)
    return parser.parse_args()


def apply_dataset_defaults(args):
    ds = args.dataset.lower().replace('.dgl', '').replace('.mat', '')
    if ds == 'fdcompcn':
        ds = 'comp'
    if ds in ('elliptic_bitcoin', 'elliptic_bitcoin_dataset'):
        ds = 'elliptic'
    if ds == 'yelp':
        ds = 'yelpchi'
    defaults = DATASET_DEFAULTS.get(ds, DATASET_DEFAULTS['yelpchi'])
    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    args.dataset_str = ds
    return args


def normalize_features(features, mode):
    if mode == 'row':
        rowsum = features.sum(axis=1, keepdims=True)
        if rowsum.min() <= 0:
            print('Some feature rows do not sum to a positive value; using standard scaling '
                  'instead of row normalisation.')
            mode = 'standard'
        else:
            return (features / (rowsum + 0.01)).astype(np.float32), mode
    if mode == 'minmax':
        return MinMaxScaler().fit_transform(features).astype(np.float32), mode
    if mode == 'standard':
        return StandardScaler().fit_transform(features).astype(np.float32), mode
    return features.astype(np.float32), mode


def _auc(labels, prob):
    y = labels.cpu().numpy()
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, prob[:, 1].cpu().numpy()))


def train(args=None):
    """Train SparseGAD. ``args`` is an argparse Namespace (see ``parse_args``)."""
    if args is None:
        args = parse_args()
    args = apply_dataset_defaults(args)
    if getattr(args, 'train_ratio', None) is None:
        args.train_ratio = 0.4
    set_seed(args.seed)
    device = get_device(args.gpu)

    features, labels_np, relations, labeled_from = load_dga_arrays(
        args.dataset, prefix=args.data_prefix)
    features, args.feat_norm = normalize_features(features, args.feat_norm)
    n = int(features.shape[0])
    index = np.arange(labeled_from, n, dtype=np.int64) if labeled_from is not None else None
    train_np, valid_np, test_np, discarded_np, split_info = protocol_split(
        labels_np, train_ratio=args.train_ratio, seed=args.seed, indices=index)

    rel_names = [name for name, _s, _d in relations]
    src_np, dst_np, w_np = build_union_graph([(s, d) for _n, s, d in relations], n)
    src = torch.as_tensor(src_np, device=device)
    dst = torch.as_tensor(dst_np, device=device)
    weight = sym_normalize(src, dst, torch.as_tensor(w_np, device=device), n)
    initial_edges = int(src.numel())
    del src_np, dst_np, w_np

    feat = torch.as_tensor(features, dtype=torch.float32, device=device)
    labels = torch.as_tensor(labels_np, dtype=torch.long, device=device)
    train_idx = torch.as_tensor(train_np, dtype=torch.long, device=device)
    valid_idx = torch.as_tensor(valid_np, dtype=torch.long, device=device)
    test_idx = torch.as_tensor(test_np, dtype=torch.long, device=device)

    n_class = int(labels.max().item()) + 1
    model = SparseGAD(feat.size(1), args.hidden, n_class=n_class, num_layers=args.nlayers,
                      dropout=args.dropout, dropout_adj=args.dropout_adj).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    print('SparseGAD dataset={} nodes={} rels={} edges={} feat={} ({}) train={} valid={} test={} '
          'hidden={} layers={} dropout={} dropout_adj={} k={} update_every={}'.format(
              args.dataset_str, n, rel_names, initial_edges, feat.size(1), args.feat_norm,
              int(train_idx.numel()), int(valid_idx.numel()), int(test_idx.numel()),
              args.hidden, args.nlayers, args.dropout, args.dropout_adj, args.k,
              args.update_adj_epoch))

    early_stop = int(getattr(args, 'early_stop', 0) or 0)
    best_state = None
    best_graph = (src, dst, weight)
    best_auc = -1.0
    best_epoch = 0
    stop_steps = 0
    graph_updates = []
    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        logits = model(feat, src, dst, weight)
        loss = F.nll_loss(F.log_softmax(logits[train_idx], dim=1), labels[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_prob = torch.softmax(model(feat, src, dst, weight)[valid_idx], dim=1)
        auc = _auc(labels[valid_idx], val_prob)
        if auc is None:
            auc = -1.0
        if auc > best_auc:
            best_auc = float(auc)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_graph = (src, dst, weight)
            stop_steps = 1
        else:
            stop_steps += 1
        print('Epoch {:d} loss: {:.4f} valid AUC: {:.4f} (best {:.4f} @ {:d})'.format(
            epoch, loss.item(), float(auc), best_auc, best_epoch))

        if args.update_adj_epoch and epoch % args.update_adj_epoch == 0:
            with torch.no_grad():
                emb = model.node_embedding(feat, src, dst, weight)
                new_src, new_dst = knn_edges(emb, args.k)
                src, dst, weight = augment_graph(src, dst, new_src, new_dst, n)
            graph_updates.append({'epoch': epoch, 'edges': int(src.numel())})
            print('Graph update at epoch {:d}: {:d} edges'.format(epoch, int(src.numel())))

        if early_stop and stop_steps >= early_stop:
            print('Early stop at epoch {:d}'.format(epoch))
            break
    train_time = time.perf_counter() - t_start
    epochs_executed = epoch
    print('train time=', '{:.5f}'.format(train_time))

    if best_state is not None:
        model.load_state_dict(best_state)
    src, dst, weight = best_graph
    model.eval()
    with torch.no_grad():
        prob = torch.softmax(model(feat, src, dst, weight), dim=1)
        pred = F.one_hot(prob.argmax(dim=-1), num_classes=n_class).float()

    y_train = labels[train_idx].cpu().numpy()
    y_valid = labels[valid_idx].cpu().numpy()
    y_test = labels[test_idx].cpu().numpy()
    train_metrics = compute_classification_metrics(
        y_train, pred[train_idx].cpu().numpy(), prob[train_idx].cpu().numpy(), sigmoid=False)
    valid_metrics = compute_classification_metrics(
        y_valid, pred[valid_idx].cpu().numpy(), prob[valid_idx].cpu().numpy(), sigmoid=False)
    test_metrics = compute_classification_metrics(
        y_test, pred[test_idx].cpu().numpy(), prob[test_idx].cpu().numpy(), sigmoid=False)
    test_metrics = dict(test_metrics)
    test_metrics['valid_auc'] = valid_metrics.get('auroc')
    test_metrics['best_epoch'] = best_epoch
    test_metrics['epochs_executed'] = epochs_executed
    print('Train metrics:', train_metrics)
    print('Valid metrics:', valid_metrics)
    print('Test metrics:', test_metrics)

    out_dir = None
    if not args.no_save:
        config = {k: v for k, v in vars(args).items()}
        config['model'] = 'SparseGAD'
        config['relations'] = rel_names
        config['best_valid_auc'] = best_auc if best_auc >= 0 else None
        config['best_epoch'] = best_epoch
        config['epochs_executed'] = epochs_executed
        config['feat_dim'] = int(feat.size(1))
        config['initial_edges'] = initial_edges
        config['best_graph_edges'] = int(src.numel())
        config['graph_updates'] = graph_updates
        out_dir = run_output_dir(
            args.save_dir, 'SparseGAD', args.dataset_str, args.seed,
            run_tag=getattr(args, 'run_tag', None))
        save_run_outputs(
            out_dir, 'SparseGAD', config, split_info,
            test_metrics, model=model, train_metrics=train_metrics,
            predictions=pred[test_idx].cpu().numpy(),
            probabilities=prob[test_idx].cpu().numpy(),
            train_indices=train_np, test_indices=test_np,
            valid_indices=valid_np,
            discarded_indices=discarded_np,
            y_train=y_train, y_test=y_test,
            train_time=train_time,
        )
    return test_metrics, out_dir


def main():
    train(parse_args())


if __name__ == '__main__':
    main()
