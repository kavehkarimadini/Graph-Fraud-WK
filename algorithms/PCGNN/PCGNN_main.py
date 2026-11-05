'''
Train PC-GNN (Liu et al., WWW 2021) with unified evaluation artifacts.

Usage (from algorithms/PCGNN):
  python PCGNN_main.py --dataset yelpchi --gpu 0
  python PCGNN_main.py --dataset amazon --gpu 0
  python PCGNN_main.py --dataset elliptic --gpu 0
  python PCGNN_main.py --dataset tfinance --gpu 0
  python PCGNN_main.py --dataset tsocial --gpu 0
  python PCGNN_main.py --dataset comp --gpu 0
'''
from __future__ import print_function
from pathlib import Path

import argparse
import copy
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from algorithms.PCGNN.PCGNN import PCGNN, build_csr
from utils.data_loader import load_dga_arrays
from utils.evaluation import compute_classification_metrics, run_output_dir, save_run_outputs
from utils.split import protocol_split
from utils.torch_helpers import get_device, set_seed
from utils.training import add_protocol_split_args, add_save_args

# Reference config/pcgnn_*.yml: emb 64, rho 0.5, alpha 2, 51 epochs;
# YelpChi lr 0.01 / L2 1e-3 / batch 1024 / thres 0.5;
# Amazon lr 0.005 / L2 5e-4 / batch 256 / thres 0.4.
# The reference row-normalises features; rows whose sum is not positive force
# standard scaling instead.
DATASET_DEFAULTS = {
    'yelpchi': dict(learning_rate=0.01, weight_decay=1e-3, batch_size=1024, thres=0.5,
                    feat_norm='row', epochs=100, early_stop=20),
    'amazon': dict(learning_rate=0.005, weight_decay=5e-4, batch_size=256, thres=0.4,
                   feat_norm='row', epochs=100, early_stop=20),
    'elliptic': dict(learning_rate=0.01, weight_decay=1e-3, batch_size=1024, thres=0.5,
                     feat_norm='standard', epochs=100, early_stop=20),
    'tfinance': dict(learning_rate=0.01, weight_decay=1e-3, batch_size=1024, thres=0.5,
                     feat_norm='standard', epochs=100, early_stop=20),
    'tsocial': dict(learning_rate=0.01, weight_decay=1e-3, batch_size=1024, thres=0.5,
                    feat_norm='standard', epochs=50, early_stop=10),
    'comp': dict(learning_rate=0.005, weight_decay=5e-4, batch_size=256, thres=0.5,
                 feat_norm='standard', epochs=100, early_stop=20),
}


def parse_args():
    parser = argparse.ArgumentParser(description='PC-GNN')
    parser.add_argument('--dataset', type=str, default='yelpchi',
                        help='yelpchi / amazon / elliptic / tfinance / tsocial / comp')
    parser.add_argument('--emb_size', type=int, default=64)
    parser.add_argument('--rho', type=float, default=0.5,
                        help='Ratio of oversampled fraud neighbours for fraud centres.')
    parser.add_argument('--alpha', type=float, default=2.0, help='Label-predictor loss weight.')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Fraction of neighbours kept per relation (choose step).')
    parser.add_argument('--thres', type=float, default=None,
                        help='Decision threshold on the fraud probability.')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--eval_batch_size', type=int, default=1024)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--feat_norm', type=str, default=None, choices=['row', 'standard', 'none'])
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--data_prefix', type=str, default='../../dataset/')
    parser.add_argument('--run_tag', type=str, default=None)
    parser.add_argument('--validation_only', action='store_true',
                        help='Tuning mode: stop after validation; never evaluate test.')
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
    if mode == 'standard':
        return StandardScaler().fit_transform(features).astype(np.float32), mode
    return features.astype(np.float32), mode


def pick_step(train_nodes, train_labels, degree, size, rng):
    """Label-balanced sampling with replacement, probability ``degree / |class|``."""
    n_pos = float(train_labels.sum())
    class_size = np.where(train_labels == 1, n_pos, float(len(train_labels)))
    prob = degree[train_nodes] / class_size
    return rng.choice(train_nodes, size=size, replace=True, p=prob / prob.sum())


@torch.no_grad()
def predict_fraud_prob(model, feat, idx, rels, batch_size):
    """``sigmoid`` of the fraud logit, as in the reference ``to_prob``."""
    model.eval()
    probs = []
    for i in range(0, idx.numel(), batch_size):
        logits, _ = model(feat, idx[i:i + batch_size], rels)
        probs.append(torch.sigmoid(logits[:, 1]))
    return torch.cat(probs, dim=0)


def _auc(labels, prob):
    y = labels.cpu().numpy()
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, prob.cpu().numpy()))


def train(args=None):
    """Train PC-GNN. ``args`` is an argparse Namespace (see ``parse_args``)."""
    if args is None:
        args = parse_args()
    args = apply_dataset_defaults(args)
    if getattr(args, 'train_ratio', None) is None:
        args.train_ratio = 0.4
    set_seed(args.seed)
    device = get_device(args.gpu)
    rng = np.random.default_rng(args.seed)

    features, labels_np, relations, labeled_from = load_dga_arrays(
        args.dataset, prefix=args.data_prefix)
    features, args.feat_norm = normalize_features(features, args.feat_norm)
    n = int(features.shape[0])
    index = np.arange(labeled_from, n, dtype=np.int64) if labeled_from is not None else None
    if args.dataset == 'tfinance':
        split_path = Path(__file__).resolve().parents[2] / 'splits' / 'tfinance' / f'seed_{args.seed}.npz'
        split = np.load(split_path)

        ratio_key = f"train_{int(round(args.train_ratio * 100))}"
        if ratio_key not in split.files:
            raise KeyError(f"Missing {ratio_key} in {split_path}")

        train_np = split[ratio_key].astype(np.int64)
        valid_np = split['val'].astype(np.int64)
        test_np = split['test'].astype(np.int64)

        discarded_key = f"discarded_{int(round(args.train_ratio * 100))}"
        discarded_np = (
            split[discarded_key].astype(np.int64)
            if discarded_key in split.files
            else np.empty(0, dtype=np.int64)
        )

        split_info = {
            'source': str(split_path),
            'seed': args.seed,
            'train_ratio': args.train_ratio,
        }
    else:
        train_np, valid_np, test_np, discarded_np, split_info = protocol_split(
            labels_np, train_ratio=args.train_ratio, seed=args.seed, indices=index)

    feat = torch.as_tensor(features, dtype=torch.float32, device=device)
    labels = torch.as_tensor(labels_np, dtype=torch.long, device=device)
    rel_names = [name for name, _s, _d in relations]
    rels = []
    for _name, src, dst in relations:
        indptr, indices = build_csr(src, dst, n)
        rels.append((torch.as_tensor(indptr, device=device), torch.as_tensor(indices, device=device)))
    homo_indptr, _ = build_csr(
        np.concatenate([r[1] for r in relations]), np.concatenate([r[2] for r in relations]), n)
    homo_degree = np.diff(homo_indptr).astype(np.float64)
    train_idx = torch.as_tensor(train_np, dtype=torch.long, device=device)
    valid_idx = torch.as_tensor(valid_np, dtype=torch.long, device=device)
    test_idx = torch.as_tensor(test_np, dtype=torch.long, device=device)
    train_labels_np = labels_np[train_np]
    train_pos_np = train_np[train_labels_np == 1]
    train_pos = torch.as_tensor(train_pos_np, dtype=torch.long, device=device)
    pick_size = 2 * len(train_pos_np)

    n_class = int(labels.max().item()) + 1
    model = PCGNN(feat.size(1), args.emb_size, len(rels), n_class=n_class,
                  threshold=args.threshold, rho=args.rho).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    print('PC-GNN dataset={} nodes={} rels={} feat={} ({}) train={} (fraud {}) valid={} test={} '
          'emb={} batch={} lr={} rho={} alpha={} thres={}'.format(
              args.dataset_str, n, rel_names, feat.size(1), args.feat_norm,
              int(train_idx.numel()), len(train_pos_np), int(valid_idx.numel()), int(test_idx.numel()),
              args.emb_size, args.batch_size, args.learning_rate, args.rho, args.alpha, args.thres))

    early_stop = int(getattr(args, 'early_stop', 0) or 0)
    best_state = None
    best_auc = -1.0
    best_epoch = 0
    stop_steps = 0
    t_start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        nodes = pick_step(train_np, train_labels_np, homo_degree, pick_size, rng)
        rng.shuffle(nodes)
        num_batches = max(1, math.ceil(len(nodes) / args.batch_size))
        epoch_loss = 0.0
        for b in range(num_batches):
            batch = torch.as_tensor(nodes[b * args.batch_size:(b + 1) * args.batch_size],
                                    dtype=torch.long, device=device)
            y = labels[batch]
            gnn_logits, label_logits = model(feat, batch, rels, labels=y, train_pos=train_pos)
            loss = F.cross_entropy(gnn_logits, y) + args.alpha * F.cross_entropy(label_logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        val_prob = predict_fraud_prob(model, feat, valid_idx, rels, args.eval_batch_size)
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
        print('Epoch {:d} loss: {:.4f} valid AUC: {:.4f} (best {:.4f} @ {:d})'.format(
            epoch, epoch_loss / num_batches, float(auc), best_auc, best_epoch))
        if early_stop and stop_steps >= early_stop:
            print('Early stop at epoch {:d}'.format(epoch))
            break
    train_time = time.perf_counter() - t_start
    epochs_executed = epoch + 1
    print('train time=', '{:.5f}'.format(train_time))

    if best_state is not None:
        model.load_state_dict(best_state)

    def _evaluate(idx):
        p = predict_fraud_prob(model, feat, idx, rels, args.eval_batch_size).cpu().numpy()
        prob = np.stack([1.0 - p, p], axis=1)
        pred = np.eye(2, dtype=np.float32)[(p >= args.thres).astype(np.int64)]
        return prob, pred

    prob_train, pred_train = _evaluate(train_idx)
    prob_valid, pred_valid = _evaluate(valid_idx)
    if args.validation_only:
        prob_valid, pred_valid = _evaluate(valid_idx)
        y_valid = labels[valid_idx].cpu().numpy()
        valid_metrics = compute_classification_metrics(
            y_valid, pred_valid, prob_valid, sigmoid=False
        )
        valid_metrics = dict(valid_metrics)
        valid_metrics['best_valid_auc'] = best_auc
        valid_metrics['best_epoch'] = best_epoch
        valid_metrics['epochs_executed'] = epochs_executed

        print('VALIDATION_ONLY_RESULT:', {
            'valid_auc': valid_metrics.get('auroc'),
            'best_valid_auc': best_auc,
            'best_epoch': best_epoch,
            'epochs_executed': epochs_executed,
        })
        return valid_metrics, None

    prob_test, pred_test = _evaluate(test_idx)
    y_train = labels[train_idx].cpu().numpy()
    y_valid = labels[valid_idx].cpu().numpy()
    y_test = labels[test_idx].cpu().numpy()
    train_metrics = compute_classification_metrics(y_train, pred_train, prob_train, sigmoid=False)
    valid_metrics = compute_classification_metrics(y_valid, pred_valid, prob_valid, sigmoid=False)
    test_metrics = compute_classification_metrics(y_test, pred_test, prob_test, sigmoid=False)
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
        config['model'] = 'PCGNN'
        config['relations'] = rel_names
        config['best_valid_auc'] = best_auc if best_auc >= 0 else None
        config['best_epoch'] = best_epoch
        config['epochs_executed'] = epochs_executed
        config['feat_dim'] = int(feat.size(1))
        config['pick_size'] = int(pick_size)
        out_dir = run_output_dir(
            args.save_dir, 'PCGNN', args.dataset_str, args.seed,
            run_tag=getattr(args, 'run_tag', None))
        save_run_outputs(
            out_dir, 'PCGNN', config, split_info,
            test_metrics, model=model, train_metrics=train_metrics,
            predictions=pred_test,
            probabilities=prob_test,
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
