'''
Train DGA-GNN (Duan et al., AAAI 2024) with unified evaluation artifacts.

Usage (from algorithms/DGAGNN):
  python DGAGNN_main.py --dataset yelpchi --gpu 0
  python DGAGNN_main.py --dataset amazon --gpu 0
  python DGAGNN_main.py --dataset elliptic --gpu 0
  python DGAGNN_main.py --dataset tfinance --gpu 0
  python DGAGNN_main.py --dataset tsocial --gpu 0
  python DGAGNN_main.py --dataset comp --gpu 0
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
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from algorithms.DGAGNN.DGAGNN import DGAGNN, decision_tree_bin_encoding
from utils.data_loader import load_dga_arrays
from utils.evaluation import compute_classification_metrics, run_output_dir, save_run_outputs
from utils.split import protocol_split
from utils.torch_helpers import get_device, set_seed
from utils.training import add_protocol_split_args, add_save_args

# Official yaml hyperparameters (reference/DGA-GNN-main/code/configs).
DATASET_DEFAULTS = {
    'yelpchi': dict(n_hidden=64, n_head=2, n_bins=32, z=0.025, dropout=0.3,
                    epochs=200, early_stop=25),
    'yelp': dict(n_hidden=64, n_head=2, n_bins=32, z=0.025, dropout=0.3,
                 epochs=200, early_stop=25),
    'amazon': dict(n_hidden=128, n_head=1, n_bins=4, z=0.05, dropout=0.3,
                   epochs=200, early_stop=25),
    'elliptic': dict(n_hidden=128, n_head=2, n_bins=4, z=0.1, dropout=0.3,
                     epochs=200, early_stop=25),
    'tfinance': dict(n_hidden=64, n_head=1, n_bins=4, z=0.025, dropout=0.3,
                     epochs=200, early_stop=25),
    'tsocial': dict(n_hidden=32, n_head=1, n_bins=8, z=0.2, dropout=0.2,
                    epochs=100, early_stop=10),
    'comp': dict(n_hidden=64, n_head=1, n_bins=4, z=0.025, dropout=0.3,
                 epochs=200, early_stop=25),
    'fdcompcn': dict(n_hidden=64, n_head=1, n_bins=4, z=0.025, dropout=0.3,
                     epochs=200, early_stop=25),
}


def parse_args():
    parser = argparse.ArgumentParser(description='DGA-GNN')
    parser.add_argument('--dataset', type=str, default='yelpchi',
                        help='yelpchi / amazon / elliptic / tfinance / tsocial / comp')
    parser.add_argument('--n_hidden', type=int, default=None)
    parser.add_argument('--n_head', type=int, default=None)
    parser.add_argument('--n_bins', type=int, default=None, help='DT bins per feature (paper k)')
    parser.add_argument('--z', type=float, default=None, help='Grouping threshold (unclear_up/down)')
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--no_bin_encoding', action='store_true',
                        help='Skip decision-tree binning; use standardized features only.')
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


def _split_auc(logits, labels, idx):
    if idx is None or idx.numel() == 0:
        return None
    y = labels[idx].detach().cpu().numpy()
    if len(np.unique(y)) < 2:
        return None
    prob = torch.softmax(logits[idx], dim=-1)[:, 1].detach().cpu().numpy()
    return float(roc_auc_score(y, prob))


def train(args=None):
    """Train DGA-GNN. ``args`` is an argparse Namespace (see ``parse_args``)."""
    if args is None:
        args = parse_args()
    args = apply_dataset_defaults(args)
    if getattr(args, 'train_ratio', None) is None:
        args.train_ratio = 0.4
    set_seed(args.seed)
    device = get_device(args.gpu)

    features, labels_np, relations, labeled_from = load_dga_arrays(
        args.dataset, prefix=args.data_prefix)
    features = StandardScaler().fit_transform(features).astype(np.float32)
    n = int(features.shape[0])
    index = np.arange(labeled_from, n, dtype=np.int64) if labeled_from is not None else None
    train_np, valid_np, test_np, discarded_np, split_info = protocol_split(
        labels_np, train_ratio=args.train_ratio, seed=args.seed, indices=index)

    if not getattr(args, 'no_bin_encoding', False):
        features = decision_tree_bin_encoding(
            features, labels_np, train_np, n_bins=args.n_bins)
        print('after bin_encoding:', features.shape)

    feat = torch.as_tensor(features, dtype=torch.float32, device=device)
    labels = torch.as_tensor(labels_np, dtype=torch.long, device=device)
    rels = [(name,
             torch.as_tensor(src, dtype=torch.long, device=device),
             torch.as_tensor(dst, dtype=torch.long, device=device))
            for name, src, dst in relations]
    train_idx = torch.as_tensor(train_np, dtype=torch.long, device=device)
    valid_idx = torch.as_tensor(valid_np, dtype=torch.long, device=device)
    test_idx = torch.as_tensor(test_np, dtype=torch.long, device=device)

    n_class = int(labels.max().item()) + 1
    model = DGAGNN(
        feat.size(1), args.n_hidden, n, n_class, len(rels),
        dropout=args.dropout, n_head=args.n_head,
        unclear_up=args.z, unclear_down=args.z).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)
    print('DGA-GNN dataset={} nodes={} rels={} feat={} train={} valid={} test={} hid={} heads={} k={} z={}'.format(
        args.dataset_str, n, [r[0] for r in rels], feat.size(1),
        int(train_idx.numel()), int(valid_idx.numel()), int(test_idx.numel()),
        args.n_hidden, args.n_head, args.n_bins, args.z))

    early_stop = int(getattr(args, 'early_stop', 0) or 0)
    best_state = None
    best_auc = -1.0
    best_epoch = 0
    stop_steps = 0
    prob_hist = []
    t_start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        logits, _emb = model(rels, feat)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits, _ = model(rels, feat)
            val_prob = torch.softmax(val_logits, dim=-1)
            val_loss = F.cross_entropy(val_logits[valid_idx], labels[valid_idx])
        scheduler.step(val_loss)
        prob_hist.append(val_prob.detach().cpu())
        if len(prob_hist) > 10:
            prob_hist = prob_hist[-10:]
        model.update_super_mask(prob_hist)

        auc = _split_auc(val_logits, labels, valid_idx)
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
            epoch, loss.item(), float(auc), best_auc, best_epoch))
        if early_stop and stop_steps >= early_stop:
            print('Early stop at epoch {:d}'.format(epoch))
            break
    train_time = time.perf_counter() - t_start
    print('train time=', '{:.5f}'.format(train_time))

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, _ = model(rels, feat)
        prob = torch.softmax(logits, dim=-1)
        pred = torch.nn.functional.one_hot(prob.argmax(dim=-1), num_classes=n_class).float()

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
    print('Train metrics:', train_metrics)
    print('Valid metrics:', valid_metrics)
    print('Test metrics:', test_metrics)

    out_dir = None
    if not args.no_save:
        config = {k: v for k, v in vars(args).items()}
        config['model'] = 'DGAGNN'
        config['best_valid_auc'] = best_auc if best_auc >= 0 else None
        config['best_epoch'] = best_epoch
        config['feat_dim'] = int(feat.size(1))
        out_dir = run_output_dir(
            args.save_dir, 'DGAGNN', args.dataset_str, args.seed,
            run_tag=getattr(args, 'run_tag', None))
        save_run_outputs(
            out_dir, 'DGAGNN', config, split_info,
            test_metrics, model=model, train_metrics=train_metrics,
            predictions=pred[test_idx].cpu().numpy(),
            probabilities=prob[test_idx].cpu().numpy(),
            train_indices=train_idx.cpu().numpy(), test_indices=test_idx.cpu().numpy(),
            valid_indices=valid_idx.cpu().numpy(),
            discarded_indices=discarded_np,
            y_train=y_train, y_test=y_test,
            train_time=train_time,
        )
    return test_metrics, out_dir


def main():
    train(parse_args())


if __name__ == '__main__':
    main()
