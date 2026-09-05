'''
Train BSL (Yu, Liu, Luo, AAAI 2024) with unified evaluation artifacts.

Usage (from algorithms/BSL):
  python BSL_main.py --dataset yelpchi --gpu 0
  python BSL_main.py --dataset amazon --gpu 0
  python BSL_main.py --dataset elliptic --gpu 0
  python BSL_main.py --dataset tfinance --gpu 0
  python BSL_main.py --dataset tsocial --gpu 0
  python BSL_main.py --dataset comp --gpu 0
  python BSL_main.py --dataset yelpchi --paper_split --gpu 0
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
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from algorithms.BSL.BSL import BSL, build_gcn_adj
from utils.data_loader import load_dga_arrays
from utils.evaluation import compute_classification_metrics, run_output_dir, save_run_outputs
from utils.split import barely_split, protocol_split
from utils.torch_helpers import get_device, set_seed
from utils.training import add_protocol_split_args, add_save_args

DATASET_DEFAULTS = {
    'yelpchi': dict(hid_dim=96, epochs=100, early_stop=25, label_ratio=0.01, unlabeled_ratio=0.10),
    'yelp': dict(hid_dim=96, epochs=100, early_stop=25, label_ratio=0.01, unlabeled_ratio=0.10),
    'amazon': dict(hid_dim=96, epochs=100, early_stop=25, label_ratio=0.01, unlabeled_ratio=0.10),
    'elliptic': dict(hid_dim=96, epochs=100, early_stop=25, label_ratio=0.01, unlabeled_ratio=0.10),
    'tfinance': dict(hid_dim=96, epochs=100, early_stop=25, label_ratio=0.01, unlabeled_ratio=0.10),
    'tsocial': dict(hid_dim=48, epochs=50, early_stop=10, label_ratio=0.0001, unlabeled_ratio=0.001),
    'comp': dict(hid_dim=96, epochs=100, early_stop=25, label_ratio=0.01, unlabeled_ratio=0.10),
    'fdcompcn': dict(hid_dim=96, epochs=100, early_stop=25, label_ratio=0.01, unlabeled_ratio=0.10),
}


def parse_args():
    parser = argparse.ArgumentParser(description='BSL')
    parser.add_argument('--dataset', type=str, default='yelpchi',
                        help='yelpchi / amazon / elliptic / tfinance / tsocial / comp')
    parser.add_argument('--hid_dim', type=int, default=None, help='Must be divisible by 3 (paper 96)')
    parser.add_argument('--alpha', type=float, default=0.4, help='Weight of L_D (link + attn)')
    parser.add_argument('--beta', type=float, default=0.8, help='Weight of L_bsl')
    parser.add_argument('--batch_size', type=int, default=256, help='Unlabeled aux batch for L_bsl')
    parser.add_argument('--paper_split', action='store_true',
                        help='Use the paper 1%/10%/89% (0.01%/0.1%% on T-Social) split; checkpoint on train loss.')
    parser.add_argument('--label_ratio', type=float, default=None)
    parser.add_argument('--unlabeled_ratio', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
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


def _to_homo(relations):
    srcs = [np.asarray(src, dtype=np.int64) for _n, src, _d in relations]
    dsts = [np.asarray(dst, dtype=np.int64) for _n, _s, dst in relations]
    return np.concatenate(srcs), np.concatenate(dsts)


def _split_auc(logits, labels, idx):
    if idx is None or idx.numel() == 0:
        return None
    y = labels[idx].detach().cpu().numpy()
    if len(np.unique(y)) < 2:
        return None
    scores = torch.softmax(logits[idx], dim=-1)[:, 1].detach().cpu().numpy()
    if not np.isfinite(scores).all():
        return None
    return float(roc_auc_score(y, scores))


def train(args=None):
    """Train BSL. ``args`` is an argparse Namespace (see ``parse_args``)."""
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

    if getattr(args, 'paper_split', False):
        train_np, valid_np, test_np, unlabeled_np, split_info = barely_split(
            labels_np, label_ratio=args.label_ratio,
            unlabeled_ratio=args.unlabeled_ratio, seed=args.seed, indices=index)
        discarded_np = unlabeled_np
    else:
        train_np, valid_np, test_np, discarded_np, split_info = protocol_split(
            labels_np, train_ratio=args.train_ratio, seed=args.seed, indices=index)
        unlabeled_np = discarded_np if discarded_np.size else valid_np

    feat = torch.as_tensor(features, dtype=torch.float32, device=device)
    labels = torch.as_tensor(labels_np, dtype=torch.long, device=device)
    src_np, dst_np = _to_homo(relations)
    src = torch.as_tensor(src_np, dtype=torch.long, device=device)
    dst = torch.as_tensor(dst_np, dtype=torch.long, device=device)
    adj = build_gcn_adj(src, dst, n, device, feat.dtype)
    train_idx = torch.as_tensor(train_np, dtype=torch.long, device=device)
    valid_idx = torch.as_tensor(valid_np, dtype=torch.long, device=device)
    test_idx = torch.as_tensor(test_np, dtype=torch.long, device=device)
    unlabeled_idx = torch.as_tensor(unlabeled_np, dtype=torch.long, device=device)

    n_class = int(labels.max().item()) + 1
    model = BSL(feat.size(1), hid_dim=args.hid_dim, n_class=n_class).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    print('BSL dataset={} nodes={} feat={} hid={} train={} valid={} test={} '
          'unlab={} alpha={} beta={} paper_split={}'.format(
              args.dataset_str, n, feat.size(1), model.hid_dim,
              int(train_idx.numel()), int(valid_idx.numel()), int(test_idx.numel()),
              int(unlabeled_idx.numel()), args.alpha, args.beta,
              bool(getattr(args, 'paper_split', False))))

    early_stop = int(getattr(args, 'early_stop', 0) or 0)
    use_valid = valid_idx.numel() > 0
    best_state = None
    best_auc = -1.0
    best_loss = float('inf')
    best_epoch = 0
    stop_steps = 0
    t_start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        if unlabeled_idx.numel() > args.batch_size:
            perm = torch.randperm(unlabeled_idx.numel(), device=device)[:args.batch_size]
            unlab_batch = unlabeled_idx[perm]
        else:
            unlab_batch = unlabeled_idx
        logits, losses = model(
            adj, feat, labels, train_idx, unlab_batch, src, dst)
        loss_d = losses['link'] + losses['attn']
        loss = losses['class'] + args.alpha * loss_d + args.beta * losses['bsl']
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits, _ = model(adj, feat)
        improved = False
        auc = _split_auc(val_logits, labels, valid_idx) if use_valid else None
        if use_valid:
            if auc is None:
                auc = -1.0
            if auc > best_auc:
                best_auc = float(auc)
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                improved = True
        else:
            auc = -1.0
            if float(loss.item()) < best_loss:
                best_loss = float(loss.item())
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                improved = True
        if improved:
            stop_steps = 1
        else:
            stop_steps += 1
        print('Epoch {:d} loss: {:.4f} (ce {:.4f} link {:.4f} attn {:.4f} bsl {:.4f}) '
              'valid AUC: {:.4f} (best {:.4f} @ {:d})'.format(
                  epoch, loss.item(), losses['class'].item(), losses['link'].item(),
                  losses['attn'].item(), losses['bsl'].item(),
                  float(auc), best_auc if use_valid else best_loss, best_epoch))
        if early_stop and stop_steps >= early_stop:
            print('Early stop at epoch {:d}'.format(epoch))
            break
    train_time = time.perf_counter() - t_start
    print('train time=', '{:.5f}'.format(train_time))

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, _ = model(adj, feat)
        prob = torch.softmax(logits, dim=-1)
        pred = torch.nn.functional.one_hot(prob.argmax(dim=-1), num_classes=n_class).float()

    y_train = labels[train_idx].cpu().numpy()
    y_test = labels[test_idx].cpu().numpy()
    train_metrics = compute_classification_metrics(
        y_train, pred[train_idx].cpu().numpy(), prob[train_idx].cpu().numpy(), sigmoid=False)
    test_metrics = compute_classification_metrics(
        y_test, pred[test_idx].cpu().numpy(), prob[test_idx].cpu().numpy(), sigmoid=False)
    test_metrics = dict(test_metrics)
    if use_valid:
        y_valid = labels[valid_idx].cpu().numpy()
        valid_metrics = compute_classification_metrics(
            y_valid, pred[valid_idx].cpu().numpy(), prob[valid_idx].cpu().numpy(), sigmoid=False)
        test_metrics['valid_auc'] = valid_metrics.get('auroc')
        print('Valid metrics:', valid_metrics)
    else:
        valid_metrics = None
        test_metrics['valid_auc'] = None
    print('Train metrics:', train_metrics)
    print('Test metrics:', test_metrics)

    out_dir = None
    if not args.no_save:
        config = {k: v for k, v in vars(args).items()}
        config['model'] = 'BSL'
        config['best_valid_auc'] = best_auc if use_valid and best_auc >= 0 else None
        config['best_train_loss'] = best_loss if not use_valid else None
        config['best_epoch'] = best_epoch
        config['feat_dim'] = int(feat.size(1))
        config['hid_dim'] = int(model.hid_dim)
        out_dir = run_output_dir(
            args.save_dir, 'BSL', args.dataset_str, args.seed,
            run_tag=getattr(args, 'run_tag', None))
        save_run_outputs(
            out_dir, 'BSL', config, split_info,
            test_metrics, model=model, train_metrics=train_metrics,
            predictions=pred[test_idx].cpu().numpy(),
            probabilities=prob[test_idx].cpu().numpy(),
            train_indices=train_idx.cpu().numpy(), test_indices=test_idx.cpu().numpy(),
            valid_indices=valid_idx.cpu().numpy() if use_valid else None,
            discarded_indices=discarded_np,
            y_train=y_train, y_test=y_test,
            train_time=train_time,
        )
    return test_metrics, out_dir


def main():
    train(parse_args())


if __name__ == '__main__':
    main()
