'''
Train FRAUDRE (Zhang et al., ICDM 2021) with unified evaluation artifacts.

Usage (from algorithms/FRAUDRE):
  python FRAUDRE_main.py --dataset yelpchi --gpu 0
  python FRAUDRE_main.py --dataset amazon --gpu 0
  python FRAUDRE_main.py --dataset elliptic --gpu 0
  python FRAUDRE_main.py --dataset tfinance --gpu 0
  python FRAUDRE_main.py --dataset tsocial --gpu 0
  python FRAUDRE_main.py --dataset comp --gpu 0
'''
from __future__ import print_function

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
from algorithms.FRAUDRE.FRAUDRE import FRAUDRE, prior_adjusted_loss, sample_adjacency
from utils.data_loader import load_dga_arrays
from utils.evaluation import compute_classification_metrics, run_output_dir, save_run_outputs
from utils.split import protocol_split
from utils.torch_helpers import get_device, set_seed
from utils.training import add_protocol_split_args, add_save_args

# Reference train.py: embed 64, weight decay 1e-4, 2 convolution layers,
# lr 0.1 / batch 256 on Amazon and lr 0.001 / batch 1024 on YelpChi, 61 epochs.
# The other datasets use lr 0.01. The reference row-normalises features; rows
# whose sum is not positive force standard scaling instead.
DATASET_DEFAULTS = {
    'yelpchi': dict(learning_rate=0.001, batch_size=1024, feat_norm='row', epochs=100, early_stop=20),
    'amazon': dict(learning_rate=0.1, batch_size=256, feat_norm='row', epochs=100, early_stop=20),
    'elliptic': dict(learning_rate=0.01, batch_size=1024, feat_norm='standard', epochs=100, early_stop=20),
    'tfinance': dict(learning_rate=0.01, batch_size=1024, feat_norm='standard', epochs=100, early_stop=20),
    'tsocial': dict(learning_rate=0.01, batch_size=1024, feat_norm='standard', epochs=50, early_stop=10),
    'comp': dict(learning_rate=0.01, batch_size=256, feat_norm='standard', epochs=100, early_stop=20),
}


def parse_args():
    parser = argparse.ArgumentParser(description='FRAUDRE')
    parser.add_argument('--dataset', type=str, default='yelpchi',
                        help='yelpchi / amazon / elliptic / tfinance / tsocial / comp')
    parser.add_argument('--embed_dim', type=int, default=64, help='Ego-feature embedding size.')
    parser.add_argument('--n_layers', type=int, default=2, help='Fraud-aware convolution layers (K).')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Classifier hidden size.')
    parser.add_argument('--max_neighbors', type=int, default=10,
                        help='Fixed neighbour sample per node and relation (0 keeps all).')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--eval_batch_size', type=int, default=4096)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--feat_norm', type=str, default=None, choices=['row', 'standard', 'none'])
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
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
            return (features / rowsum).astype(np.float32), mode
    if mode == 'standard':
        return StandardScaler().fit_transform(features).astype(np.float32), mode
    return features.astype(np.float32), mode


@torch.no_grad()
def predict_proba(model, feat, idx, rels, batch_size):
    model.eval()
    probs = []
    for i in range(0, idx.numel(), batch_size):
        probs.append(torch.softmax(model(feat, idx[i:i + batch_size], rels), dim=-1))
    return torch.cat(probs, dim=0)


def _auc(labels, prob):
    y = labels.cpu().numpy()
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, prob[:, 1].cpu().numpy()))


def train(args=None):
    """Train FRAUDRE. ``args`` is an argparse Namespace (see ``parse_args``)."""
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
    train_np, valid_np, test_np, discarded_np, split_info = protocol_split(
        labels_np, train_ratio=args.train_ratio, seed=args.seed, indices=index)

    feat = torch.as_tensor(features, dtype=torch.float32, device=device)
    labels = torch.as_tensor(labels_np, dtype=torch.long, device=device)
    rel_names = [name for name, _s, _d in relations]
    rels = []
    for r, (_name, src, dst) in enumerate(relations):
        indptr, indices = sample_adjacency(src, dst, n, args.max_neighbors, seed=args.seed + r)
        rels.append((torch.as_tensor(indptr, device=device), torch.as_tensor(indices, device=device)))
    train_idx = torch.as_tensor(train_np, dtype=torch.long, device=device)
    valid_idx = torch.as_tensor(valid_np, dtype=torch.long, device=device)
    test_idx = torch.as_tensor(test_np, dtype=torch.long, device=device)

    n_class = int(labels.max().item()) + 1
    prior = np.bincount(labels_np[train_np], minlength=n_class) / float(len(train_np))
    log_prior = torch.log(torch.as_tensor(prior + 1e-8, dtype=torch.float32, device=device))

    model = FRAUDRE(
        feat.size(1), len(rels), embed_dim=args.embed_dim, n_layers=args.n_layers,
        hidden_dim=args.hidden_dim, n_class=n_class).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    print('FRAUDRE dataset={} nodes={} rels={} feat={} ({}) train={} valid={} test={} '
          'embed={} layers={} batch={} lr={} prior={}'.format(
              args.dataset_str, n, rel_names, feat.size(1), args.feat_norm,
              int(train_idx.numel()), int(valid_idx.numel()), int(test_idx.numel()),
              args.embed_dim, args.n_layers, args.batch_size, args.learning_rate,
              [round(float(p), 4) for p in prior]))

    early_stop = int(getattr(args, 'early_stop', 0) or 0)
    best_state = None
    best_auc = -1.0
    best_epoch = 0
    stop_steps = 0
    t_start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        nodes = rng.permutation(train_np)
        num_batches = max(1, math.ceil(len(nodes) / args.batch_size))
        epoch_loss = 0.0
        for b in range(num_batches):
            batch = torch.as_tensor(nodes[b * args.batch_size:(b + 1) * args.batch_size],
                                    dtype=torch.long, device=device)
            loss = prior_adjusted_loss(model(feat, batch, rels), labels[batch], log_prior)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        val_prob = predict_proba(model, feat, valid_idx, rels, args.eval_batch_size)
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
    prob_train = predict_proba(model, feat, train_idx, rels, args.eval_batch_size)
    prob_valid = predict_proba(model, feat, valid_idx, rels, args.eval_batch_size)
    prob_test = predict_proba(model, feat, test_idx, rels, args.eval_batch_size)

    def _hard(prob):
        return F.one_hot(prob.argmax(dim=-1), num_classes=n_class).float().cpu().numpy()

    y_train = labels[train_idx].cpu().numpy()
    y_valid = labels[valid_idx].cpu().numpy()
    y_test = labels[test_idx].cpu().numpy()
    train_metrics = compute_classification_metrics(
        y_train, _hard(prob_train), prob_train.cpu().numpy(), sigmoid=False)
    valid_metrics = compute_classification_metrics(
        y_valid, _hard(prob_valid), prob_valid.cpu().numpy(), sigmoid=False)
    test_metrics = compute_classification_metrics(
        y_test, _hard(prob_test), prob_test.cpu().numpy(), sigmoid=False)
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
        config['model'] = 'FRAUDRE'
        config['relations'] = rel_names
        config['best_valid_auc'] = best_auc if best_auc >= 0 else None
        config['best_epoch'] = best_epoch
        config['epochs_executed'] = epochs_executed
        config['feat_dim'] = int(feat.size(1))
        config['train_prior'] = [float(p) for p in prior]
        out_dir = run_output_dir(
            args.save_dir, 'FRAUDRE', args.dataset_str, args.seed,
            run_tag=getattr(args, 'run_tag', None))
        save_run_outputs(
            out_dir, 'FRAUDRE', config, split_info,
            test_metrics, model=model, train_metrics=train_metrics,
            predictions=_hard(prob_test),
            probabilities=prob_test.cpu().numpy(),
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
