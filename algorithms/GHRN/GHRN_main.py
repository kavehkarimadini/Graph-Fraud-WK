'''
Train GHRN (Gao et al., WWW 2023) with unified evaluation artifacts.

Two stages in one run: BWGNN on the original graph, then GHRN edge pruning with
its predictions and a fresh BWGNN on the pruned graph (reported metrics).

Usage (from algorithms/GHRN):
  python GHRN_main.py --dataset yelpchi --gpu 0
  python GHRN_main.py --dataset amazon --gpu 0
  python GHRN_main.py --dataset amazon --homo 0 --gpu 0
  python GHRN_main.py --dataset elliptic --gpu 0
  python GHRN_main.py --dataset tfinance --gpu 0
  python GHRN_main.py --dataset tsocial --gpu 0
  python GHRN_main.py --dataset comp --gpu 0
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
from algorithms.GHRN.GHRN import (
    BWGNN, BWGNNHetero, NormAdj, add_self_loops, prune_edges, relation_edges)
from utils.data_loader import load_dga_arrays
from utils.evaluation import compute_classification_metrics, run_output_dir, save_run_outputs
from utils.split import protocol_split
from utils.torch_helpers import get_device, set_seed
from utils.training import add_protocol_split_args, add_save_args

# Reference main.py: BWGNN hidden 64, order 2, Adam lr 0.01, 100 epochs,
# class-weighted CE, symmetric indicator. Deleting ratio from the paper's
# sensitivity analysis (Sec. 4.5): ~0.1 on YelpChi, ~0.015 on Amazon / T-Social,
# about half the edge heterophily otherwise (0.015, as in the README example).
# YelpChi / Amazon follow the DGL FraudDataset path (union of relations with
# self-loops, raw features); T-Finance / T-Social use the graph as is without
# self-loops. T-Social uses BWGNN's paper setting (hidden 10, order 5).
# Elliptic and FDCompCN are standardised and follow the single-graph /
# multi-relation paths.
DATASET_DEFAULTS = {
    'yelpchi': dict(hid_dim=64, order=2, del_ratio=0.1, feat_norm='none', self_loop=1),
    'amazon': dict(hid_dim=64, order=2, del_ratio=0.015, feat_norm='none', self_loop=1),
    'elliptic': dict(hid_dim=64, order=2, del_ratio=0.015, feat_norm='standard', self_loop=0),
    'tfinance': dict(hid_dim=64, order=2, del_ratio=0.015, feat_norm='none', self_loop=0),
    'tsocial': dict(hid_dim=10, order=5, del_ratio=0.015, feat_norm='none', self_loop=0),
    'comp': dict(hid_dim=64, order=2, del_ratio=0.015, feat_norm='standard', self_loop=1),
}


def parse_args():
    parser = argparse.ArgumentParser(description='GHRN')
    parser.add_argument('--dataset', type=str, default='yelpchi',
                        help='yelpchi / amazon / elliptic / tfinance / tsocial / comp')
    parser.add_argument('--homo', type=int, default=1,
                        help='1: BWGNN on the union graph; 0: per-relation BWGNN (BWGNN_Hetero).')
    parser.add_argument('--hid_dim', type=int, default=None)
    parser.add_argument('--order', type=int, default=None, help='Beta wavelet order C.')
    parser.add_argument('--del_ratio', type=float, default=None,
                        help='Fraction of edges (self-loops included) deleted by the indicator '
                             '(dataset default); 0 = plain BWGNN.')
    parser.add_argument('--adj_type', type=str, default='sym', choices=['sym', 'rw'],
                        help='Normalisation of the indicator aggregation.')
    parser.add_argument('--self_loop', type=int, default=None,
                        help='Add self-loops to the homogeneous training graph (dataset default).')
    parser.add_argument('--feat_norm', type=str, default=None,
                        choices=['row', 'minmax', 'standard', 'none'])
    parser.add_argument('--epochs', type=int, default=100, help='Max epochs per stage.')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--data_prefix', type=str, default='../../dataset/')
    parser.add_argument('--run_tag', type=str, default=None)
    add_save_args(parser)
    add_protocol_split_args(parser)
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


def inter_class_share(src, dst, labels, labeled):
    """Fraction of edges between labeled nodes that join different classes (diagnostic only)."""
    mask = labeled[src] & labeled[dst]
    if not mask.any():
        return None
    return float((labels[src[mask]] != labels[dst[mask]]).mean())


def build_adjs(edge_sets, n, homo, self_loop, device):
    """Training operators: one union graph (homo) or one per relation (hetero)."""
    if homo:
        src = np.concatenate([s for s, _d in edge_sets])
        dst = np.concatenate([d for _s, d in edge_sets])
        if self_loop:
            src, dst = add_self_loops(src, dst, n)
        return [NormAdj(src, dst, n, device)]
    return [NormAdj(s, d, n, device) for s, d in edge_sets]


def fit(args, model, feat, adjs, labels, train_idx, valid_idx, ce_weight, stage):
    """Full-batch training with early stopping on valid AUC; restores the best state."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    early_stop = int(getattr(args, 'early_stop', 0) or 0)
    best_state = None
    best_auc = -1.0
    best_epoch = 0
    stop_steps = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        logits = model(feat, adjs)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx], weight=ce_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_prob = torch.softmax(model(feat, adjs)[valid_idx], dim=1)
        auc = _auc(labels[valid_idx], val_prob)
        if auc is None:
            auc = -1.0
        if auc > best_auc:
            best_auc = float(auc)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stop_steps = 1
        else:
            stop_steps += 1
        print('[{}] Epoch {:d} loss: {:.4f} valid AUC: {:.4f} (best {:.4f} @ {:d})'.format(
            stage, epoch, loss.item(), float(auc), best_auc, best_epoch))
        if early_stop and stop_steps >= early_stop:
            print('[{}] Early stop at epoch {:d}'.format(stage, epoch))
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return dict(best_auc=best_auc, best_epoch=best_epoch, epochs_executed=epoch)


def train(args=None):
    """Train GHRN. ``args`` is an argparse Namespace (see ``parse_args``)."""
    if args is None:
        args = parse_args()
    args = apply_dataset_defaults(args)
    if getattr(args, 'train_ratio', None) is None:
        args.train_ratio = 0.4
    device = get_device(args.gpu)

    features, labels_np, relations, labeled_from = load_dga_arrays(
        args.dataset, prefix=args.data_prefix)
    features, args.feat_norm = normalize_features(features, args.feat_norm)
    n = int(features.shape[0])
    index = np.arange(labeled_from, n, dtype=np.int64) if labeled_from is not None else None
    train_np, valid_np, test_np, discarded_np, split_info = protocol_split(
        labels_np, train_ratio=args.train_ratio, seed=args.seed, indices=index)

    rel_names = [name for name, _s, _d in relations]
    edge_sets = [relation_edges(s, d, n) for _n, s, d in relations]
    del relations
    homo = bool(args.homo)
    if homo:
        edge_sets = [(np.concatenate([s for s, _d in edge_sets]),
                      np.concatenate([d for _s, d in edge_sets]))]
    initial_edges = [int(s.size) for s, _d in edge_sets]

    feat = torch.as_tensor(features, dtype=torch.float32, device=device)
    labels = torch.as_tensor(labels_np, dtype=torch.long, device=device)
    train_idx = torch.as_tensor(train_np, dtype=torch.long, device=device)
    valid_idx = torch.as_tensor(valid_np, dtype=torch.long, device=device)
    test_idx = torch.as_tensor(test_np, dtype=torch.long, device=device)
    y_tr = labels[train_idx]
    n_pos = max(int(y_tr.sum().item()), 1)
    ce_weight = torch.tensor([1.0, (y_tr.numel() - n_pos) / n_pos], device=device)
    n_class = int(labels.max().item()) + 1
    model_cls = BWGNN if homo else BWGNNHetero
    print('GHRN dataset={} nodes={} rels={} homo={} self_loop={} edges={} feat={} ({}) '
          'train={} valid={} test={} hid_dim={} order={} del_ratio={} adj_type={} ce_weight={:.3f}'.format(
              args.dataset_str, n, rel_names, int(homo), args.self_loop, initial_edges, feat.size(1),
              args.feat_norm, int(train_idx.numel()), int(valid_idx.numel()), int(test_idx.numel()),
              args.hid_dim, args.order, args.del_ratio, args.adj_type, ce_weight[1].item()))

    t_start = time.perf_counter()
    adjs = build_adjs(edge_sets, n, homo, args.self_loop, device)
    set_seed(args.seed)
    model = model_cls(feat.size(1), args.hid_dim, n_class=n_class, order=args.order).to(device)
    stage1 = fit(args, model, feat, adjs, labels, train_idx, valid_idx, ce_weight, 'BWGNN')
    model.eval()
    with torch.no_grad():
        prob1 = torch.softmax(model(feat, adjs), dim=1)
    stage1_test = compute_classification_metrics(
        labels[test_idx].cpu().numpy(),
        F.one_hot(prob1[test_idx].argmax(dim=-1), num_classes=n_class).float().cpu().numpy(),
        prob1[test_idx].cpu().numpy(), sigmoid=False)
    print('Stage 1 (BWGNN) test metrics:', stage1_test)

    pruned_edges = list(initial_edges)
    deleted = [0] * len(edge_sets)
    heterophily = []
    labeled = np.ones(n, dtype=bool)
    if labeled_from is not None:
        labeled[:labeled_from] = False
    stage2 = None
    if args.del_ratio > 0:
        new_sets = []
        for i, (s, d) in enumerate(edge_sets):
            ns, nd, removed = prune_edges(s, d, n, prob1, args.del_ratio, args.adj_type)
            new_sets.append((ns, nd))
            deleted[i] = removed
            pruned_edges[i] = int(ns.size)
            before, after = inter_class_share(s, d, labels_np, labeled), inter_class_share(ns, nd, labels_np, labeled)
            heterophily.append({'before': before, 'after': after})
            print('Pruned graph {}: {} -> {} edges ({} deleted); inter-class share of labeled '
                  'edges {:.4f} -> {:.4f}'.format(i, s.size, ns.size, removed, before, after))
        edge_sets = new_sets
        del adjs
        adjs = build_adjs(edge_sets, n, homo, args.self_loop, device)
        set_seed(args.seed)
        model = model_cls(feat.size(1), args.hid_dim, n_class=n_class, order=args.order).to(device)
        stage2 = fit(args, model, feat, adjs, labels, train_idx, valid_idx, ce_weight, 'GHRN')
    train_time = time.perf_counter() - t_start
    final = stage2 if stage2 is not None else stage1
    epochs_executed = stage1['epochs_executed'] + (stage2['epochs_executed'] if stage2 else 0)
    print('train time=', '{:.5f}'.format(train_time))

    model.eval()
    with torch.no_grad():
        prob = torch.softmax(model(feat, adjs), dim=1)
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
    test_metrics['best_epoch'] = final['best_epoch']
    test_metrics['epochs_executed'] = epochs_executed
    print('Train metrics:', train_metrics)
    print('Valid metrics:', valid_metrics)
    print('Test metrics:', test_metrics)

    out_dir = None
    if not args.no_save:
        config = {k: v for k, v in vars(args).items()}
        config['model'] = 'GHRN'
        config['relations'] = rel_names
        config['best_valid_auc'] = final['best_auc'] if final['best_auc'] >= 0 else None
        config['best_epoch'] = final['best_epoch']
        config['epochs_executed'] = epochs_executed
        config['feat_dim'] = int(feat.size(1))
        config['initial_edges'] = initial_edges
        config['pruned_edges'] = pruned_edges
        config['deleted_edges'] = deleted
        config['inter_class_share'] = heterophily
        config['stage1'] = dict(stage1, test_metrics=stage1_test)
        config['stage2'] = stage2
        out_dir = run_output_dir(
            args.save_dir, 'GHRN', args.dataset_str, args.seed,
            run_tag=getattr(args, 'run_tag', None))
        save_run_outputs(
            out_dir, 'GHRN', config, split_info,
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
