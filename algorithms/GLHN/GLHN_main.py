'''
Train GLHN / GLHAD (Guo et al., CIKM 2024) with unified evaluation artifacts.

Usage (from algorithms/GLHN):
  python GLHN_main.py --dataset yelpchi --gpu 0
  python GLHN_main.py --dataset amazon --gpu 0
  python GLHN_main.py --dataset elliptic --gpu 0
  python GLHN_main.py --dataset tfinance --gpu 0
  python GLHN_main.py --dataset tsocial --gpu 0
  python GLHN_main.py --dataset comp --gpu 0
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
from algorithms.GLHN.GLHN import GLHN, build_relation
from utils.data_loader import load_dga_arrays
from utils.evaluation import compute_classification_metrics, run_output_dir, save_run_outputs
from utils.split import protocol_split
from utils.torch_helpers import get_device, set_seed
from utils.training import add_protocol_split_args, add_save_args

# Paper §4.1.4 plus γ1/γ2 from the sensitivity study (YelpChi / Amazon).
DATASET_DEFAULTS = {
    'yelpchi': dict(hid_dim=32, n_layers=2, gamma1=0.8, gamma2=1.0,
                    epochs=100, early_stop=25, graph_type='hetero'),
    'yelp': dict(hid_dim=32, n_layers=2, gamma1=0.8, gamma2=1.0,
                 epochs=100, early_stop=25, graph_type='hetero'),
    'amazon': dict(hid_dim=32, n_layers=2, gamma1=1.0, gamma2=0.8,
                   epochs=100, early_stop=25, graph_type='hetero'),
    'elliptic': dict(hid_dim=32, n_layers=2, gamma1=1.0, gamma2=1.0,
                     epochs=100, early_stop=25, graph_type='homo'),
    'tfinance': dict(hid_dim=32, n_layers=2, gamma1=1.0, gamma2=1.0,
                     epochs=100, early_stop=25, graph_type='homo'),
    'tsocial': dict(hid_dim=10, n_layers=2, gamma1=1.0, gamma2=1.0,
                    epochs=50, early_stop=10, graph_type='homo'),
    'comp': dict(hid_dim=32, n_layers=2, gamma1=0.8, gamma2=1.0,
                 epochs=100, early_stop=25, graph_type='hetero'),
    'fdcompcn': dict(hid_dim=32, n_layers=2, gamma1=0.8, gamma2=1.0,
                     epochs=100, early_stop=25, graph_type='hetero'),
}


def parse_args():
    parser = argparse.ArgumentParser(description='GLHN / GLHAD')
    parser.add_argument('--dataset', type=str, default='yelpchi',
                        help='yelpchi / amazon / elliptic / tfinance / tsocial / comp')
    parser.add_argument('--hid_dim', type=int, default=None)
    parser.add_argument('--n_layers', type=int, default=None)
    parser.add_argument('--tau', type=float, default=0.1, help='InfoNCE temperature')
    parser.add_argument('--gamma1', type=float, default=None, help='Weight of L_k')
    parser.add_argument('--gamma2', type=float, default=None, help='Weight of L_u')
    parser.add_argument('--graph_type', type=str, default=None, choices=['homo', 'hetero'],
                        help='Union all edges (homo) or per-relation then mean-pool (hetero).')
    parser.add_argument('--filter_pair', type=str, default='laplacian',
                        choices=['laplacian', 'half'],
                        help='laplacian: F_L=I-L, F_H=L; half: I-L/2 and L/2.')
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
    return [('homo', np.concatenate(srcs), np.concatenate(dsts))]


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


def _ce_weight(labels, train_idx, n_class, device):
    y = labels[train_idx]
    counts = torch.bincount(y, minlength=n_class).to(dtype=torch.float32)
    if n_class == 2 and counts[1] > 0:
        gamma = counts[0] / counts[1]
        return torch.tensor([1.0, float(gamma)], device=device)
    inv = counts.clamp(min=1.0)
    return (inv.sum() / (n_class * inv)).to(device)


def _prepare_relations(raw_rels, n, device, dtype, graph_type):
    if graph_type == 'homo' and len(raw_rels) > 1:
        raw_rels = _to_homo(raw_rels)
    built = []
    for name, src, dst in raw_rels:
        src_t = torch.as_tensor(src, dtype=torch.long, device=device)
        dst_t = torch.as_tensor(dst, dtype=torch.long, device=device)
        adj, d_inv, col, row = build_relation(src_t, dst_t, n, device, dtype)
        built.append((name, adj, d_inv, col, row))
    return built


def train(args=None):
    """Train GLHN. ``args`` is an argparse Namespace (see ``parse_args``)."""
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

    feat = torch.as_tensor(features, dtype=torch.float32, device=device)
    labels = torch.as_tensor(labels_np, dtype=torch.long, device=device)
    rels = _prepare_relations(relations, n, device, feat.dtype, args.graph_type)
    train_idx = torch.as_tensor(train_np, dtype=torch.long, device=device)
    valid_idx = torch.as_tensor(valid_np, dtype=torch.long, device=device)
    test_idx = torch.as_tensor(test_np, dtype=torch.long, device=device)
    labeled_mask = torch.zeros(n, dtype=torch.bool, device=device)
    labeled_mask[train_idx] = True

    n_class = int(labels.max().item()) + 1
    model = GLHN(
        feat.size(1), args.hid_dim, n_class, n_layers=args.n_layers,
        tau=args.tau, filter_pair=args.filter_pair).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    ce_w = _ce_weight(labels, train_idx, n_class, device)
    print('GLHN dataset={} nodes={} rels={} feat={} train={} valid={} test={} '
          'hid={} L={} type={} tau={} g1={} g2={}'.format(
              args.dataset_str, n, [r[0] for r in rels], feat.size(1),
              int(train_idx.numel()), int(valid_idx.numel()), int(test_idx.numel()),
              args.hid_dim, args.n_layers, args.graph_type, args.tau,
              args.gamma1, args.gamma2))

    early_stop = int(getattr(args, 'early_stop', 0) or 0)
    best_state = None
    best_auc = -1.0
    best_epoch = 0
    stop_steps = 0
    t_start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        logits, lks, lus = model(rels, feat, labels, labeled_mask)
        loss_ce = F.cross_entropy(logits[train_idx], labels[train_idx], weight=ce_w)
        loss_k = torch.stack(lks).sum() if lks else feat.new_zeros(())
        loss_u = torch.stack(lus).sum() if lus else feat.new_zeros(())
        loss = loss_ce + args.gamma1 * loss_k + args.gamma2 * loss_u
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits, _, _ = model(rels, feat)
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
        print('Epoch {:d} loss: {:.4f} (ce {:.4f} Lk {:.4f} Lu {:.4f}) '
              'valid AUC: {:.4f} (best {:.4f} @ {:d})'.format(
                  epoch, loss.item(), loss_ce.item(), loss_k.item(), loss_u.item(),
                  float(auc), best_auc, best_epoch))
        if early_stop and stop_steps >= early_stop:
            print('Early stop at epoch {:d}'.format(epoch))
            break
    train_time = time.perf_counter() - t_start
    print('train time=', '{:.5f}'.format(train_time))

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(rels, feat)
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
        config['model'] = 'GLHN'
        config['best_valid_auc'] = best_auc if best_auc >= 0 else None
        config['best_epoch'] = best_epoch
        config['feat_dim'] = int(feat.size(1))
        out_dir = run_output_dir(
            args.save_dir, 'GLHN', args.dataset_str, args.seed,
            run_tag=getattr(args, 'run_tag', None))
        save_run_outputs(
            out_dir, 'GLHN', config, split_info,
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
