'''
Train COFRAUD (Zang et al., IJCAI 2023) with unified evaluation artifacts.

Usage (from algorithms/COFRAUD):
  python COFRAUD_main.py --dataset yelpchi --gpu 0
  python COFRAUD_main.py --dataset amazon --gpu 0
  python COFRAUD_main.py --dataset elliptic --gpu 0
  python COFRAUD_main.py --dataset tfinance --gpu 0
  python COFRAUD_main.py --dataset tsocial --gpu 0
  python COFRAUD_main.py --dataset comp --gpu 0
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
from algorithms.COFRAUD.COFRAUD import COFRAUD, build_csr
from utils.data_loader import load_dga_arrays
from utils.evaluation import compute_classification_metrics, run_output_dir, save_run_outputs
from utils.split import protocol_split
from utils.torch_helpers import get_device, set_seed
from utils.training import add_protocol_split_args, add_save_args

# Paper Sec. 4: 2 layers, hidden 32 (Table 3 best); Amazon / Yelp only.
# Official train script (zyl199710/COFRAUD): 8 heads, relation_hidden 16,
# dropout 0.5, residual, lr 0.02, wd 0, 500 epochs, patience 50.
DATASET_DEFAULTS = {
    'yelpchi': dict(hidden=32, n_layers=2, n_heads=8, epochs=500, early_stop=50),
    'amazon': dict(hidden=32, n_layers=2, n_heads=8, epochs=500, early_stop=50),
    'elliptic': dict(hidden=32, n_layers=2, n_heads=8, epochs=500, early_stop=50),
    'tfinance': dict(hidden=32, n_layers=2, n_heads=8, epochs=500, early_stop=50),
    'tsocial': dict(hidden=32, n_layers=2, n_heads=4, epochs=200, early_stop=20),
    'comp': dict(hidden=32, n_layers=2, n_heads=8, epochs=500, early_stop=50),
}


def parse_args():
    parser = argparse.ArgumentParser(description='COFRAUD')
    parser.add_argument('--dataset', type=str, default='yelpchi',
                        help='yelpchi / amazon / elliptic / tfinance / tsocial / comp')
    parser.add_argument('--hidden', type=int, default=None,
                        help='Per-head hidden size (paper Table 3: 32).')
    parser.add_argument('--n_layers', type=int, default=None,
                        help='Intra/inter module layers (paper: 2).')
    parser.add_argument('--n_heads', type=int, default=None,
                        help='Attention heads (official: 8; paper equations are single-head).')
    parser.add_argument('--relation_dim', type=int, default=16,
                        help='Relation embedding size (official: 16; paper uses one-hot).')
    parser.add_argument('--relation_hidden', type=int, default=16,
                        help='Relation MLP hidden size (official: 16).')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--negative_slope', type=float, default=0.2)
    parser.add_argument('--residual', type=int, default=1,
                        help='1 = Eq. 5 residual with trainable λ; 0 = aggregation only.')
    parser.add_argument('--no_inter', action='store_true',
                        help='Ablation: skip inter-relation crossing (w/o inter).')
    parser.add_argument('--no_fusion', action='store_true',
                        help='Ablation: mean of V_r h\'_r instead of Eqs. 9-10 (w/o fusion).')
    parser.add_argument('--thres', type=float, default=0.5, help='Decision threshold on fraud prob.')
    parser.add_argument('--feat_norm', type=str, default='standard',
                        choices=['row', 'standard', 'none'])
    parser.add_argument('--loss_weight', type=str, default='none', choices=['none', 'balanced'],
                        help="Cross-entropy class weights ('none' matches the paper's CE).")
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=0.02)
    parser.add_argument('--weight_decay', type=float, default=0.0)
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
    if mode == 'standard':
        return StandardScaler().fit_transform(features).astype(np.float32), mode
    return features.astype(np.float32), mode


def _auc(labels, prob):
    y = labels.cpu().numpy()
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, prob.cpu().numpy()))


def train(args=None):
    """Train COFRAUD. ``args`` is an argparse Namespace (see ``parse_args``)."""
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
    csrs = []
    deg_inv_sqrt = []
    n_edges = 0
    for _name, src, dst in relations:
        indptr, indices = build_csr(src, dst, n)
        deg = np.diff(indptr).astype(np.float32)
        inv = np.zeros_like(deg)
        pos = deg > 0
        inv[pos] = 1.0 / np.sqrt(deg[pos])
        csrs.append((
            torch.as_tensor(indptr, dtype=torch.long, device=device),
            torch.as_tensor(indices, dtype=torch.long, device=device),
        ))
        deg_inv_sqrt.append(torch.as_tensor(inv, dtype=torch.float32, device=device))
        n_edges += int(deg.sum()) // 2
    del relations

    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    labels = torch.as_tensor(labels_np, dtype=torch.long, device=device)
    train_idx = torch.as_tensor(train_np, dtype=torch.long, device=device)
    valid_idx = torch.as_tensor(valid_np, dtype=torch.long, device=device)
    test_idx = torch.as_tensor(test_np, dtype=torch.long, device=device)

    n_class = int(labels.max().item()) + 1
    class_weight = None
    if args.loss_weight == 'balanced':
        counts = np.bincount(labels_np[train_np], minlength=n_class).astype(np.float64)
        class_weight = torch.as_tensor(
            counts.sum() / (n_class * np.maximum(counts, 1.0)),
            dtype=torch.float32, device=device)

    model = COFRAUD(
        x.size(1), n_relations=len(csrs), hidden=args.hidden,
        relation_dim=args.relation_dim, relation_hidden=args.relation_hidden,
        n_layers=args.n_layers, n_heads=args.n_heads, n_class=n_class,
        dropout=args.dropout, negative_slope=args.negative_slope,
        residual=bool(args.residual),
        use_inter=not args.no_inter, use_fusion=not args.no_fusion).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    print('COFRAUD dataset={} nodes={} edges={} rels={} feat={} ({}) train={} valid={} test={} '
          'hidden={} layers={} heads={} rel_dim={} rel_hid={} residual={} inter={} fusion={} '
          'lr={} wd={}'.format(
              args.dataset_str, n, n_edges, rel_names, x.size(1), args.feat_norm,
              len(train_np), len(valid_np), len(test_np),
              args.hidden, args.n_layers, args.n_heads, args.relation_dim,
              args.relation_hidden, bool(args.residual),
              not args.no_inter, not args.no_fusion,
              args.learning_rate, args.weight_decay))

    early_stop = int(getattr(args, 'early_stop', 0) or 0)
    best_state = None
    best_auc = -1.0
    best_epoch = 0
    stop_steps = 0
    t_start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        logits = model(x, csrs, deg_inv_sqrt)
        loss = torch.nn.functional.cross_entropy(
            logits[train_idx], labels[train_idx], weight=class_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x, csrs, deg_inv_sqrt)
            val_prob = torch.softmax(val_logits, dim=-1)[valid_idx, 1]
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
            epoch, loss.item(), float(auc), best_auc, best_epoch))
        if early_stop and stop_steps >= early_stop:
            print('Early stop at epoch {:d}'.format(epoch))
            break
    train_time = time.perf_counter() - t_start
    epochs_executed = epoch + 1
    print('train time=', '{:.5f}'.format(train_time))

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(x, csrs, deg_inv_sqrt)
        prob = torch.softmax(logits, dim=-1)
        pred = torch.nn.functional.one_hot(
            (prob[:, 1] >= args.thres).long(), num_classes=n_class).float()

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
        config['model'] = 'COFRAUD'
        config['relations'] = rel_names
        config['n_edges'] = n_edges
        config['best_valid_auc'] = best_auc if best_auc >= 0 else None
        config['best_epoch'] = best_epoch
        config['epochs_executed'] = epochs_executed
        config['feat_dim'] = int(x.size(1))
        out_dir = run_output_dir(
            args.save_dir, 'COFRAUD', args.dataset_str, args.seed,
            run_tag=getattr(args, 'run_tag', None))
        save_run_outputs(
            out_dir, 'COFRAUD', config, split_info,
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
