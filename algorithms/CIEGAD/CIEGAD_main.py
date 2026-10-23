'''
Train CIE-GAD (Huang et al., IEEE TCYB 2025) with unified evaluation artifacts.

Usage (from algorithms/CIEGAD):
  python CIEGAD_main.py --dataset yelpchi --gpu 0
  python CIEGAD_main.py --dataset amazon --gpu 0
  python CIEGAD_main.py --dataset elliptic --gpu 0
  python CIEGAD_main.py --dataset tfinance --gpu 0
  python CIEGAD_main.py --dataset tsocial --gpu 0
  python CIEGAD_main.py --dataset comp --gpu 0
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
from algorithms.CIEGAD.CIEGAD import (
    CIEGAD, build_homo_graph, normalized_adjacency, to_torch_csr)
from utils.data_loader import load_dga_arrays
from utils.evaluation import compute_classification_metrics, run_output_dir, save_run_outputs
from utils.split import protocol_split
from utils.torch_helpers import get_device, set_seed
from utils.training import add_protocol_split_args, add_save_args

# Paper Sec. V-B: embedding 32, 1000 epochs with early stopping, gamma 1, Adam,
# Optuna-tuned remaining hyperparameters; K = 12 on T-Finance and 6 elsewhere.
# Homogeneous union graph. Learning rate / weight decay are not reported; we use
# 1e-2 / 5e-5 (typical for full-batch spectral GAD models in this toolbox).
DATASET_DEFAULTS = {
    'yelpchi': dict(order=6, hidden=32, epochs=1000, early_stop=100),
    'amazon': dict(order=6, hidden=32, epochs=1000, early_stop=100),
    'elliptic': dict(order=6, hidden=32, epochs=1000, early_stop=100),
    'tfinance': dict(order=12, hidden=32, epochs=1000, early_stop=100),
    'tsocial': dict(order=6, hidden=32, epochs=200, early_stop=20),
    'comp': dict(order=6, hidden=32, epochs=1000, early_stop=100),
}


def parse_args():
    parser = argparse.ArgumentParser(description='CIE-GAD')
    parser.add_argument('--dataset', type=str, default='yelpchi',
                        help='yelpchi / amazon / elliptic / tfinance / tsocial / comp')
    parser.add_argument('--hidden', type=int, default=None, help='Embedding size C (paper 32).')
    parser.add_argument('--order', type=int, default=None, help='Jacobi order K (paper 6 / 12).')
    parser.add_argument('--gamma', type=float, default=1.0, help='Deep-supervision weight (Eq. 21).')
    parser.add_argument('--jacobi_a', type=float, default=1.0,
                        help='Jacobi weight a (paper leaves the value open; default 1).')
    parser.add_argument('--jacobi_b', type=float, default=1.0,
                        help='Jacobi weight b (paper leaves the value open; default 1).')
    parser.add_argument('--learn_jacobi', type=int, default=1,
                        help='1 = learn a,b (clamped to > -1); 0 = keep the given a,b fixed.')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--no_scnaf', action='store_true',
                        help='Ablation: drop Jacobi filters (CIE-GAD w/o SCNAF).')
    parser.add_argument('--no_naf', action='store_true',
                        help='Ablation: one scalar alpha_k per filter (CIE-GAD w/o NAF).')
    parser.add_argument('--no_hcre', action='store_true',
                        help='Ablation: classify from Z_tilde only (CIE-GAD w/o HCRE).')
    parser.add_argument('--thres', type=float, default=0.5, help='Decision threshold on fraud prob.')
    parser.add_argument('--feat_norm', type=str, default='standard',
                        choices=['row', 'standard', 'none'])
    parser.add_argument('--loss_weight', type=str, default='none', choices=['none', 'balanced'],
                        help="Cross-entropy class weights ('none' matches the paper's CE).")
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
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
    """Train CIE-GAD. ``args`` is an argparse Namespace (see ``parse_args``)."""
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
    adj = build_homo_graph([(src, dst) for _n, src, dst in relations], n)
    a_hat = normalized_adjacency(adj)
    deg_np = np.asarray(adj.sum(axis=1)).ravel().astype(np.float32)
    adj_t = to_torch_csr(adj, device)
    a_hat_t = to_torch_csr(a_hat, device)
    deg = torch.as_tensor(deg_np, dtype=torch.float32, device=device)
    del relations, adj, a_hat

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

    model = CIEGAD(
        x.size(1), hidden=args.hidden, order=args.order, n_class=n_class,
        gamma=args.gamma, jacobi_a=args.jacobi_a, jacobi_b=args.jacobi_b,
        learn_jacobi=bool(args.learn_jacobi), dropout=args.dropout,
        use_scnaf=not args.no_scnaf, use_naf=not args.no_naf,
        use_hcre=not args.no_hcre).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    print('CIE-GAD dataset={} nodes={} edges={} rels={} feat={} ({}) train={} valid={} test={} '
          'C={} K={} gamma={} a0={} b0={} learn_ab={} scnaf={} naf={} hcre={} lr={} wd={}'.format(
              args.dataset_str, n, int(deg_np.sum()) // 2, rel_names, x.size(1), args.feat_norm,
              len(train_np), len(valid_np), len(test_np), args.hidden, args.order, args.gamma,
              args.jacobi_a, args.jacobi_b, bool(args.learn_jacobi),
              not args.no_scnaf, not args.no_naf, not args.no_hcre,
              args.learning_rate, args.weight_decay))

    early_stop = int(getattr(args, 'early_stop', 0) or 0)
    best_state = None
    best_auc = -1.0
    best_epoch = 0
    stop_steps = 0
    t_start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        logits, logits_aux, _z, _attn = model(x, a_hat_t, adj_t, deg)
        loss, l0, l1 = model.loss(logits, logits_aux, labels, train_idx, weight=class_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits, _, _, _ = model(x, a_hat_t, adj_t, deg)
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
        a_now = float(model.jacobi_a.detach().cpu())
        b_now = float(model.jacobi_b.detach().cpu())
        print('Epoch {:d} loss: {:.4f} (L0 {:.4f} L1 {:.4f}) a={:.3f} b={:.3f} '
              'valid AUC: {:.4f} (best {:.4f} @ {:d})'.format(
                  epoch, loss.item(), l0.item(), l1.item(), a_now, b_now,
                  float(auc), best_auc, best_epoch))
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
        logits, _, _, attn = model(x, a_hat_t, adj_t, deg)
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
        config['model'] = 'CIEGAD'
        config['relations'] = rel_names
        config['n_edges'] = int(deg_np.sum()) // 2
        config['best_valid_auc'] = best_auc if best_auc >= 0 else None
        config['best_epoch'] = best_epoch
        config['epochs_executed'] = epochs_executed
        config['feat_dim'] = int(x.size(1))
        config['jacobi_a_final'] = float(model.jacobi_a.detach().cpu())
        config['jacobi_b_final'] = float(model.jacobi_b.detach().cpu())
        config['attn_mean'] = attn.mean(dim=0).detach().cpu().tolist()
        out_dir = run_output_dir(
            args.save_dir, 'CIEGAD', args.dataset_str, args.seed,
            run_tag=getattr(args, 'run_tag', None))
        save_run_outputs(
            out_dir, 'CIEGAD', config, split_info,
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
