'''
Train PMP (Zhuo et al., ICLR 2024) with unified evaluation artifacts. No DGL needed.

Usage (repo root):
  python algorithms/PMP/PMP_main.py --dataset amazon --data_prefix dataset/ --gpu 0
  python algorithms/PMP/PMP_main.py --dataset tsocial --data_prefix dataset/ --seed 123 --train_ratio 0.1 --gpu 0
'''
import argparse
import copy
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from algorithms.PMP.PMP import PMP
from utils.data_loader import load_dga_arrays
from utils.evaluation import compute_classification_metrics, run_dir_from_args, save_run_outputs
from utils.hparams import apply_tuned_hparams, canonical_dataset
from utils.run_guard import run_guarded
from utils.split import protocol_split
from utils.torch_helpers import get_device, set_seed
from utils.training import add_protocol_split_args, add_save_args

# Paper App. D.4 Table 5 (Adam, L = 1): lr / wd / dropout / batch size / hidden per dataset.
# The paper does not give an epoch budget; runs stop early on validation AUROC (patience
# 25) within at most 500 epochs, the protocol of the earlier PMP benchmark in this repo.
# Elliptic / FDCompCN are not in the paper: their values are only the fallback when
# configs/tuned/PMP/<dataset>.json is absent (they are the configurations the earlier
# validation-only search of this repo selected).
DATASET_DEFAULTS = {
    'yelpchi': dict(hid_dim=256, dropout=0.0, batch_size=512, learning_rate=0.01,
                    weight_decay=0.0, num_layers=1, epochs=500),
    'amazon': dict(hid_dim=128, dropout=0.6, batch_size=128, learning_rate=0.01,
                   weight_decay=0.0, num_layers=1, epochs=500),
    'tfinance': dict(hid_dim=64, dropout=0.4, batch_size=256, learning_rate=0.01,
                     weight_decay=0.0, num_layers=1, epochs=500),
    'tsocial': dict(hid_dim=128, dropout=0.0, batch_size=2 ** 17, learning_rate=0.001,
                    weight_decay=0.0, num_layers=1, epochs=500),
    'elliptic': dict(hid_dim=256, dropout=0.0, batch_size=512, learning_rate=0.01,
                     weight_decay=0.0, num_layers=1, epochs=500),
    'comp': dict(hid_dim=512, dropout=0.0, batch_size=256, learning_rate=0.005,
                 weight_decay=0.0, num_layers=1, epochs=500),
}


def apply_dataset_defaults(args):
    defaults = DATASET_DEFAULTS.get(canonical_dataset(args.dataset), DATASET_DEFAULTS['yelpchi'])
    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    return args


def parse_args():
    parser = argparse.ArgumentParser(description='PMP (partitioning message passing)')
    parser.add_argument('--dataset', type=str, default='amazon',
                        help='yelpchi / amazon / tfinance / tsocial / elliptic / comp')
    parser.add_argument('--hid_dim', type=int, default=None)
    parser.add_argument('--num_layers', type=int, default=None, help='PMP layers L (paper: 1)')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout before the output layer')
    parser.add_argument('--batch_size', type=int, default=None, help='Training nodes per mini-batch')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--eval_batch_size', type=int, default=65536,
                        help='Nodes per forward pass at evaluation time (memory only).')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--data_prefix', type=str, default='../../dataset/')
    parser.add_argument('--run_tag', type=str, default=None)
    add_save_args(parser)
    add_protocol_split_args(parser)
    return apply_tuned_hparams(parser.parse_args(), 'PMP', parser)


def row_normalize(features):
    """Official PMP feature normalisation: x / (sum(x) + 0.01) per node."""
    features = np.asarray(features, dtype=np.float32)
    row_sum = features.sum(axis=1, keepdims=True) + 0.01
    row_sum[~np.isfinite(row_sum)] = 1.0
    return (features / row_sum).astype(np.float32)


def load_pmp_data(args):
    """Features, labels, relation adjacencies (scipy CSR, A[dst, src]) and splittable nodes."""
    features, labels, relations, labeled_from = load_dga_arrays(args.dataset, prefix=args.data_prefix)
    n = features.shape[0]
    features = row_normalize(features)
    adjs, names = [], []
    for name, src, dst in relations:
        src = np.asarray(src, dtype=np.int64)
        dst = np.asarray(dst, dtype=np.int64)
        adj = sp.csr_matrix((np.ones(src.size, dtype=np.float32), (dst, src)), shape=(n, n))
        adjs.append(adj)
        names.append(str(name))
    index = np.arange(labeled_from, n, dtype=np.int64) if labeled_from is not None else None
    print('PMP data: {} nodes, {} features, relations {}'.format(
        n, features.shape[1], ', '.join('{}={}'.format(nm, a.nnz) for nm, a in zip(names, adjs))))
    return features, labels, adjs, names, index


def label_masks(n, labels, train_idx):
    """(fraud, benign, unlabelled) node masks; only training labels count as known."""
    is_train = np.zeros(n, dtype=bool)
    is_train[train_idx] = True
    fr = is_train & (labels == 1)
    be = is_train & (labels == 0)
    unk = ~is_train
    return [m.astype(np.float32) for m in (fr, be, unk)]


def _split_auc(prob_pos, labels_np, idx):
    y = labels_np[idx]
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, prob_pos[idx]))


def train(args=None):
    """Train PMP. ``args`` is an argparse Namespace (see ``parse_args``)."""
    if args is None:
        args = parse_args()
    args = apply_dataset_defaults(args)
    args.dataset_str = canonical_dataset(args.dataset)
    if getattr(args, 'train_ratio', None) is None:
        args.train_ratio = 0.4
    if getattr(args, 'eval_batch_size', None) is None:
        args.eval_batch_size = 65536
    set_seed(args.seed)
    device = get_device(args.gpu)

    features, labels_np, adjs, rel_names, index = load_pmp_data(args)
    n = features.shape[0]
    train_np, valid_np, test_np, discarded_np, split_info = protocol_split(
        labels_np, train_ratio=args.train_ratio, seed=args.seed, indices=index)
    masks_np = label_masks(n, labels_np, train_np)

    x = torch.from_numpy(features).to(device)
    labels = torch.as_tensor(labels_np, dtype=torch.long, device=device)
    train_idx = torch.as_tensor(train_np, dtype=torch.long, device=device)

    if args.num_layers == 1:
        # Layer-1 inputs are fixed: precompute every relation's partitioned neighbour sums once.
        neigh_all = [tuple(torch.from_numpy(np.ascontiguousarray(adj @ (features * m[:, None]))).to(device)
                           for m in masks_np) for adj in adjs]
        del adjs

        def forward_nodes(idx):
            return model.forward_precomputed(x[idx], [tuple(s[idx] for s in rel) for rel in neigh_all])
    else:
        adj_t = []
        for adj in adjs:
            coo = adj.tocoo()
            adj_t.append(torch.sparse_coo_tensor(
                np.vstack([coo.row, coo.col]), coo.data, (n, n)).coalesce().to(device))
        masks_t = [torch.from_numpy(m).view(-1, 1).to(device) for m in masks_np]
        del adjs

        def forward_nodes(idx):
            return model.forward_full(x, adj_t, masks_t)[idx]

    model = PMP(features.shape[1], args.hid_dim, len(rel_names), num_layers=args.num_layers,
                dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    def predict_all():
        model.eval()
        with torch.no_grad():
            if args.num_layers == 1:
                out = [forward_nodes(torch.arange(s, min(s + args.eval_batch_size, n), device=device))
                       for s in range(0, n, args.eval_batch_size)]
                logits = torch.cat(out, dim=0)
            else:
                logits = model.forward_full(x, adj_t, masks_t)
        return logits

    early_stop = int(getattr(args, 'early_stop', 0) or 0)
    gen = torch.Generator().manual_seed(int(args.seed))
    n_train = train_idx.numel()
    best_state, best_auc, best_epoch, stop_steps = None, -1.0, 0, 0
    t_start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        perm = train_idx[torch.randperm(n_train, generator=gen).to(device)]
        total_loss = 0.0
        for start in range(0, n_train, args.batch_size):
            batch = perm[start:start + args.batch_size]
            loss = F.cross_entropy(forward_nodes(batch), labels[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * batch.numel()
        prob_pos = torch.softmax(predict_all(), dim=-1)[:, 1].cpu().numpy()
        auc = _split_auc(prob_pos, labels_np, valid_np)
        auc = -1.0 if auc is None else auc
        if auc > best_auc:
            best_auc, best_epoch, stop_steps = float(auc), epoch, 1
            best_state = copy.deepcopy(model.state_dict())
        else:
            stop_steps += 1
        print('Epoch {:d} loss: {:.4f} valid AUC: {:.4f} (best {:.4f} @ {:d})'.format(
            epoch, total_loss / max(n_train, 1), float(auc), best_auc, best_epoch))
        if early_stop and stop_steps >= early_stop:
            print('Early stop at epoch {:d}'.format(epoch))
            break
    train_time = time.perf_counter() - t_start
    print('train time=', '{:.5f}'.format(train_time))

    if best_state is not None:
        model.load_state_dict(best_state)
    logits = predict_all()
    prob = torch.softmax(logits, dim=-1).cpu().numpy()
    pred = np.eye(prob.shape[1], dtype=np.float32)[prob.argmax(axis=1)]

    def split_metrics(idx):
        return compute_classification_metrics(labels_np[idx], pred[idx], prob[idx], sigmoid=False)

    train_metrics, valid_metrics = split_metrics(train_np), split_metrics(valid_np)
    test_metrics = dict(split_metrics(test_np))
    test_metrics['valid_auc'] = valid_metrics.get('auroc')
    print('Train metrics:', train_metrics)
    print('Valid metrics:', valid_metrics)
    print('Test metrics:', test_metrics)

    out_dir = None
    if not args.no_save:
        config = {k: v for k, v in vars(args).items()}
        config.update(model='PMP', relations=rel_names, best_valid_auc=best_auc if best_auc >= 0 else None,
                      best_epoch=best_epoch, feature_normalization='x / (sum(x) + 0.01) (official PMP)')
        out_dir = run_dir_from_args(args, 'PMP')
        save_run_outputs(
            out_dir, 'PMP', config, split_info, test_metrics, model=model, train_metrics=train_metrics,
            predictions=pred[test_np], probabilities=prob[test_np],
            train_indices=train_np, test_indices=test_np, valid_indices=valid_np,
            discarded_indices=discarded_np, y_train=labels_np[train_np], y_test=labels_np[test_np],
            train_time=train_time)
    return test_metrics, out_dir


def main():
    run_guarded(train, parse_args(), 'PMP')


if __name__ == '__main__':
    main()
