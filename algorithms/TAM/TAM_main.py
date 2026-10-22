'''
Train TAM (Qiao & Pang, NeurIPS 2023) with unified evaluation artifacts.

TAM is unsupervised: labels are never used for training. Test metrics use the
anomaly score as the fraud probability; hard predictions flag the nodes whose
score exceeds the training-node quantile matching the training anomaly rate.

Usage (from algorithms/TAM):
  python TAM_main.py --dataset yelpchi --gpu 0
  python TAM_main.py --dataset amazon --gpu 0
  python TAM_main.py --dataset elliptic --gpu 0
  python TAM_main.py --dataset tfinance --gpu 0
  python TAM_main.py --dataset tsocial --gpu 0
  python TAM_main.py --dataset comp --gpu 0
'''
from __future__ import print_function

import argparse
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from algorithms.TAM.TAM import LAMNet, RawGraph, local_affinity, non_neighbor_similarity, truncate
from utils.data_loader import load_dga_arrays
from utils.evaluation import compute_classification_metrics, run_output_dir, save_run_outputs
from utils.split import protocol_split
from utils.torch_helpers import get_device, set_seed

# Reference (README / train.py / train_large_graph.py): embedding 128, Adam lr
# 1e-5, 500 epochs per LAMNet, lambda 0 on real-world data. Multi-relation union
# graphs follow the large-graph setting (Amazon-all / YelpChi-all: K = 7
# truncations, T = 1 sequence, min-max normalised affinity in the loss);
# single-relation graphs use K = 4, T = 3 (paper default) and the plain loss.
# Features: row-normalised on YelpChi / Amazon, raw on T-Finance (reference),
# raw on T-Social (same source), standardised on Elliptic / FDCompCN.
DATASET_DEFAULTS = {
    'yelpchi': dict(feat_norm='row'),
    'amazon': dict(feat_norm='row'),
    'elliptic': dict(feat_norm='standard'),
    'tfinance': dict(feat_norm='none'),
    'tsocial': dict(feat_norm='none'),
    'comp': dict(feat_norm='standard'),
}
UNION_SETTING = dict(cutting=7, n_tree=1, minmax_loss=1)
SINGLE_SETTING = dict(cutting=4, n_tree=3, minmax_loss=0)


def parse_args():
    parser = argparse.ArgumentParser(description='TAM')
    parser.add_argument('--dataset', type=str, default='yelpchi',
                        help='yelpchi / amazon / elliptic / tfinance / tsocial / comp')
    parser.add_argument('--relation', type=str, default='all',
                        help="Relation to use ('all' = union of every relation).")
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--cutting', type=int, default=None, help='Number of truncations K.')
    parser.add_argument('--n_tree', type=int, default=None, help='Number of truncation sequences T.')
    parser.add_argument('--lamda', type=float, default=0.0, help='Weight of the non-neighbour regulariser.')
    parser.add_argument('--minmax_loss', type=int, default=None,
                        help='Min-max normalise the affinity inside the loss (large-graph script).')
    parser.add_argument('--feat_norm', type=str, default=None,
                        choices=['row', 'minmax', 'standard', 'none'])
    parser.add_argument('--epochs', type=int, default=500, help='Epochs per LAMNet.')
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--data_prefix', type=str, default='../../dataset/')
    parser.add_argument('--run_tag', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Root directory for checkpoints and evaluation outputs.')
    parser.add_argument('--test_size', type=float, default=None, help='Unused (protocol split).')
    parser.add_argument('--no_save', action='store_true', help='Skip writing unified evaluation artifacts.')
    parser.add_argument('--train_ratio', type=float, default=0.4,
                        help='Train fraction of labeled nodes (max 0.4); only sets the anomaly-rate '
                             'estimate for the decision threshold.')
    return parser.parse_args()


def apply_dataset_defaults(args, n_relations):
    ds = args.dataset.lower().replace('.dgl', '').replace('.mat', '')
    if ds == 'fdcompcn':
        ds = 'comp'
    if ds in ('elliptic_bitcoin', 'elliptic_bitcoin_dataset'):
        ds = 'elliptic'
    if ds == 'yelp':
        ds = 'yelpchi'
    defaults = dict(DATASET_DEFAULTS.get(ds, DATASET_DEFAULTS['yelpchi']))
    defaults.update(UNION_SETTING if n_relations > 1 else SINGLE_SETTING)
    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    args.dataset_str = ds
    return args


def normalize_features(features, mode):
    """``row``: reference ``preprocess_features`` (rows divided by their sum)."""
    if mode == 'row':
        rowsum = features.sum(axis=1, keepdims=True)
        if rowsum.min() <= 0:
            print('Some feature rows do not sum to a positive value; using standard scaling '
                  'instead of row normalisation.')
            mode = 'standard'
        else:
            return (features / rowsum).astype(np.float32), mode
    if mode == 'minmax':
        return MinMaxScaler().fit_transform(features).astype(np.float32), mode
    if mode == 'standard':
        return StandardScaler().fit_transform(features).astype(np.float32), mode
    return features.astype(np.float32), mode


def select_relations(relations, name):
    if name in (None, 'all'):
        return relations
    chosen = [r for r in relations if r[0] == name]
    if not chosen:
        raise ValueError("Relation '{}' not found; available: {}".format(
            name, [r[0] for r in relations]))
    return chosen


def _minmax(x):
    span = x.max() - x.min()
    return (x - x.min()) / span if span > 0 else torch.zeros_like(x)


def _auc(y, score):
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, score))


def train(args=None):
    """Train TAM. ``args`` is an argparse Namespace (see ``parse_args``)."""
    if args is None:
        args = parse_args()
    if getattr(args, 'train_ratio', None) is None:
        args.train_ratio = 0.4
    set_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    device = get_device(args.gpu)

    features, labels_np, relations, labeled_from = load_dga_arrays(
        args.dataset, prefix=args.data_prefix)
    relations = select_relations(relations, args.relation)
    args = apply_dataset_defaults(args, len(relations))
    features, args.feat_norm = normalize_features(features, args.feat_norm)
    n = int(features.shape[0])
    index = np.arange(labeled_from, n, dtype=np.int64) if labeled_from is not None else None
    train_np, valid_np, test_np, discarded_np, split_info = protocol_split(
        labels_np, train_ratio=args.train_ratio, seed=args.seed, indices=index)

    rel_names = [name for name, _s, _d in relations]
    graph = RawGraph([(s, d) for _n, s, d in relations], n, features, device)
    del relations
    feat = torch.as_tensor(features, dtype=torch.float32, device=device)
    print('TAM dataset={} nodes={} rels={} edges(+self)={} feat={} ({}) train={} valid={} test={} '
          'K={} T={} emb={} lr={} epochs={} lamda={} minmax_loss={}'.format(
              args.dataset_str, n, rel_names, graph.num_edges, feat.size(1), args.feat_norm,
              len(train_np), len(valid_np), len(test_np), args.cutting, args.n_tree,
              args.embedding_dim, args.learning_rate, args.epochs, args.lamda, args.minmax_loss))

    y_valid = labels_np[valid_np]
    masks = [torch.ones(graph.num_edges, dtype=torch.bool, device=device) for _ in range(args.n_tree)]
    models = torch.nn.ModuleList()
    cut_means = []
    cut_log = []
    t_start = time.perf_counter()
    for k in range(args.cutting):
        tree_scores = []
        for t in range(args.n_tree):
            masks[t] = truncate(graph, masks[t], rng)
            adj_norm = graph.normalized_adj(masks[t])
            model = LAMNet(feat.size(1), args.embedding_dim).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                         weight_decay=args.weight_decay)
            model.train()
            for epoch in range(args.epochs):
                emb, feat1 = model(feat, adj_norm)
                affinity = local_affinity(emb, graph)
                loss = -(_minmax(affinity) if args.minmax_loss else affinity).sum()
                if args.lamda:
                    loss = loss + args.lamda * non_neighbor_similarity(feat1, graph)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if epoch % 100 == 0 or epoch == args.epochs - 1:
                    print('[cut {:d} tree {:d}] epoch {:d} loss {:.4f}'.format(k, t, epoch, loss.item()))
            tree_scores.append(affinity.detach())
            models.append(model)
            del adj_norm
        cut_means.append(torch.stack(tree_scores).mean(0))
        score_k = (1 - _minmax(torch.stack(cut_means).mean(0))).cpu().numpy()
        entry = {'cut': k, 'edges': [int(m.sum().item()) for m in masks],
                 'valid_auc': _auc(y_valid, score_k[valid_np])}
        cut_log.append(entry)
        print('Cut {:d}: edges {} valid AUROC (ensemble so far) {}'.format(
            k, entry['edges'], entry['valid_auc']))
    train_time = time.perf_counter() - t_start
    epochs_executed = args.cutting * args.n_tree * args.epochs
    print('train time=', '{:.5f}'.format(train_time))

    score = (1 - _minmax(torch.stack(cut_means).mean(0))).cpu().numpy().astype(np.float64)
    y_train = labels_np[train_np]
    rate = float(y_train.mean()) if len(y_train) else 0.0
    threshold = float(np.quantile(score[train_np], 1 - rate)) if 0 < rate < 1 else 1.0
    pred_all = (score >= threshold).astype(np.int64)
    prob_all = np.stack([1 - score, score], axis=1)
    onehot_all = np.eye(2, dtype=np.float32)[pred_all]

    y_test = labels_np[test_np]
    train_metrics = compute_classification_metrics(y_train, onehot_all[train_np], prob_all[train_np], sigmoid=False)
    valid_metrics = compute_classification_metrics(y_valid, onehot_all[valid_np], prob_all[valid_np], sigmoid=False)
    test_metrics = compute_classification_metrics(y_test, onehot_all[test_np], prob_all[test_np], sigmoid=False)
    test_metrics = dict(test_metrics)
    test_metrics['valid_auc'] = valid_metrics.get('auroc')
    test_metrics['best_epoch'] = epochs_executed
    test_metrics['epochs_executed'] = epochs_executed
    print('Decision threshold {:.4f} (train anomaly rate {:.4f})'.format(threshold, rate))
    print('Train metrics:', train_metrics)
    print('Valid metrics:', valid_metrics)
    print('Test metrics:', test_metrics)

    out_dir = None
    if not args.no_save:
        config = {k: v for k, v in vars(args).items()}
        config['model'] = 'TAM'
        config['relations'] = rel_names
        config['num_edges_with_self_loops'] = graph.num_edges
        config['epochs_executed'] = epochs_executed
        config['feat_dim'] = int(feat.size(1))
        config['decision_threshold'] = threshold
        config['train_anomaly_rate'] = rate
        config['truncation_log'] = cut_log
        out_dir = run_output_dir(
            args.save_dir, 'TAM', args.dataset_str, args.seed,
            run_tag=getattr(args, 'run_tag', None))
        save_run_outputs(
            out_dir, 'TAM', config, split_info,
            test_metrics, model=models, train_metrics=train_metrics,
            predictions=onehot_all[test_np],
            probabilities=prob_all[test_np],
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
