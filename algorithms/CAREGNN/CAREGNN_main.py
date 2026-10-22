'''
Train CARE-GNN (Dou et al., CIKM 2020) with unified evaluation artifacts.

Usage (from algorithms/CAREGNN):
  python CAREGNN_main.py --dataset yelpchi --gpu 0
  python CAREGNN_main.py --dataset amazon --gpu 0
  python CAREGNN_main.py --dataset elliptic --gpu 0
  python CAREGNN_main.py --dataset tfinance --gpu 0
  python CAREGNN_main.py --dataset tsocial --gpu 0
  python CAREGNN_main.py --dataset comp --gpu 0
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
from algorithms.CAREGNN.CAREGNN import CAREGNN, build_csr
from utils.data_loader import load_dga_arrays
from utils.evaluation import compute_classification_metrics, run_output_dir, save_run_outputs
from utils.split import protocol_split
from utils.torch_helpers import get_device, set_seed
from utils.training import add_protocol_split_args, add_save_args

# Paper / reference train.py: emb 64, lr 0.01, lambda_1 2, L2 1e-3, step 0.02,
# batch 1024 (YelpChi) / 256 (Amazon). The reference row-normalises features;
# that is only well defined for non-negative features, so datasets with signed
# features (Amazon included) are standardised instead.
DATASET_DEFAULTS = {
    'yelpchi': dict(batch_size=1024, feat_norm='row', epochs=100, early_stop=25),
    'amazon': dict(batch_size=256, feat_norm='standard', epochs=100, early_stop=25),
    'elliptic': dict(batch_size=1024, feat_norm='standard', epochs=100, early_stop=25),
    'tfinance': dict(batch_size=1024, feat_norm='standard', epochs=100, early_stop=25),
    'tsocial': dict(batch_size=1024, feat_norm='standard', epochs=50, early_stop=10),
    'comp': dict(batch_size=256, feat_norm='standard', epochs=100, early_stop=25),
}


def parse_args():
    parser = argparse.ArgumentParser(description='CARE-GNN')
    parser.add_argument('--dataset', type=str, default='yelpchi',
                        help='yelpchi / amazon / elliptic / tfinance / tsocial / comp')
    parser.add_argument('--inter', type=str, default='GNN', choices=['GNN', 'Mean', 'Weight', 'Att'],
                        help='Inter-relation aggregator (GNN = CARE-GNN, others are paper variants).')
    parser.add_argument('--emb_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--eval_batch_size', type=int, default=4096)
    parser.add_argument('--lambda_1', type=float, default=2.0, help='Similarity (label) loss weight.')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='L2 weight (lambda_2).')
    parser.add_argument('--step_size', type=float, default=0.02, help='RL action step size.')
    parser.add_argument('--init_threshold', type=float, default=0.5)
    parser.add_argument('--under_sample', type=float, default=1.0,
                        help='Negatives per positive drawn each epoch.')
    parser.add_argument('--no_rl', action='store_true', help='Freeze filtering thresholds.')
    parser.add_argument('--feat_norm', type=str, default=None, choices=['row', 'standard', 'none'])
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=0.01)
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
    if mode == 'row' and features.min() < 0:
        print('Features contain negative values; using standard scaling instead of row normalisation.')
        mode = 'standard'
    if mode == 'row':
        rowsum = features.sum(axis=1, keepdims=True) + 0.01
        return (features / rowsum).astype(np.float32), mode
    if mode == 'standard':
        return StandardScaler().fit_transform(features).astype(np.float32), mode
    return features.astype(np.float32), mode


@torch.no_grad()
def predict_proba(model, feat, idx, rels, batch_size):
    model.eval()
    probs = []
    for i in range(0, idx.numel(), batch_size):
        logits, _, _ = model(feat, idx[i:i + batch_size], rels)
        probs.append(torch.softmax(logits, dim=-1))
    return torch.cat(probs, dim=0)


def _auc(labels, prob):
    y = labels.cpu().numpy()
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, prob[:, 1].cpu().numpy()))


def _epoch_relation_scores(batch_scores, n_rel):
    """Mean of per-batch relation scores, ignoring batches without fraud nodes."""
    out = []
    for r in range(n_rel):
        vals = [s[r] for s in batch_scores if not math.isnan(s[r])]
        out.append(float(np.mean(vals)) if vals else math.nan)
    return out


def train(args=None):
    """Train CARE-GNN. ``args`` is an argparse Namespace (see ``parse_args``)."""
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
    for _name, src, dst in relations:
        indptr, indices = build_csr(src, dst, n)
        rels.append((torch.as_tensor(indptr, device=device), torch.as_tensor(indices, device=device)))
    train_idx = torch.as_tensor(train_np, dtype=torch.long, device=device)
    valid_idx = torch.as_tensor(valid_np, dtype=torch.long, device=device)
    test_idx = torch.as_tensor(test_np, dtype=torch.long, device=device)
    train_pos = train_np[labels_np[train_np] == 1]
    train_neg = train_np[labels_np[train_np] != 1]

    n_class = int(labels.max().item()) + 1
    model = CAREGNN(
        feat.size(1), args.emb_size, len(rels), n_class=n_class, inter=args.inter,
        step_size=args.step_size, init_threshold=args.init_threshold).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    print('CARE-GNN dataset={} nodes={} rels={} feat={} ({}) train={} valid={} test={} '
          'emb={} inter={} batch={} lambda_1={} step={}'.format(
              args.dataset_str, n, rel_names, feat.size(1), args.feat_norm,
              int(train_idx.numel()), int(valid_idx.numel()), int(test_idx.numel()),
              args.emb_size, args.inter, args.batch_size, args.lambda_1, args.step_size))

    early_stop = int(getattr(args, 'early_stop', 0) or 0)
    best_state = None
    best_auc = -1.0
    best_epoch = 0
    stop_steps = 0
    prev_scores = None
    thresholds_log = [model.thresholds.tolist()]
    t_start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        n_neg = min(len(train_neg), int(round(len(train_pos) * args.under_sample)))
        nodes = np.concatenate([train_pos, rng.choice(train_neg, size=n_neg, replace=False)])
        rng.shuffle(nodes)
        num_batches = max(1, math.ceil(len(nodes) / args.batch_size))
        epoch_loss = 0.0
        batch_scores = []
        for b in range(num_batches):
            batch = torch.as_tensor(nodes[b * args.batch_size:(b + 1) * args.batch_size],
                                    dtype=torch.long, device=device)
            y = labels[batch]
            gnn_logits, label_logits, rel_scores = model(feat, batch, rels, y)
            loss = F.cross_entropy(gnn_logits, y) + args.lambda_1 * F.cross_entropy(label_logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_scores.append(rel_scores)

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

        epoch_scores = _epoch_relation_scores(batch_scores, len(rels))
        if not args.no_rl and prev_scores is not None:
            model.rl_update(prev_scores, epoch_scores)
        prev_scores = epoch_scores
        thresholds_log.append(model.thresholds.tolist())
        print('Epoch {:d} loss: {:.4f} valid AUC: {:.4f} (best {:.4f} @ {:d}) thresholds: {}'.format(
            epoch, epoch_loss / num_batches, float(auc), best_auc, best_epoch,
            [round(t, 3) for t in model.thresholds.tolist()]))
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
        config['model'] = 'CAREGNN'
        config['relations'] = rel_names
        config['best_valid_auc'] = best_auc if best_auc >= 0 else None
        config['best_epoch'] = best_epoch
        config['epochs_executed'] = epochs_executed
        config['feat_dim'] = int(feat.size(1))
        config['final_thresholds'] = model.thresholds.tolist()
        config['thresholds_log'] = thresholds_log
        out_dir = run_output_dir(
            args.save_dir, 'CAREGNN', args.dataset_str, args.seed,
            run_tag=getattr(args, 'run_tag', None))
        save_run_outputs(
            out_dir, 'CAREGNN', config, split_info,
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
