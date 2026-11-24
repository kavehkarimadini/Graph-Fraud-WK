'''
Train SEC-GFD (Xu et al., AAAI 2024) with unified evaluation artifacts. No DGL needed.

Usage (repo root):
  python algorithms/SEC_GFD/SEC_GFD_main.py --dataset amazon --data_prefix dataset/ --gpu 0
  python algorithms/SEC_GFD/SEC_GFD_main.py --dataset tfinance --data_prefix dataset/ --seed 123 --train_ratio 0.1 --gpu 0
'''
import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from algorithms.SEC_GFD.SEC_GFD import SECGFD, environment_loss
from utils.data_loader import load_comp_arrays, load_dga_arrays
from utils.evaluation import compute_classification_metrics, run_dir_from_args, save_run_outputs
from utils.hparams import apply_tuned_hparams, canonical_dataset
from utils.run_guard import run_guarded
from utils.split import protocol_split
from utils.torch_helpers import get_device, set_seed
from utils.training import add_protocol_split_args, add_save_args

# Released code (reference/SEC-GFD-main.zip, main.py): Adam lr 0.01 without weight decay,
# hidden 64, Beta-wavelet order C = 2, high_order 2, lambda (``--lemda``) 0.2.
# T-Social: hidden 16, C = 5 (the repo's earlier SEC-GFD configuration). The released code
# trains a fixed 100 epochs and reports the best *test* epoch; here training stops early on
# validation AUROC (patience 25) within at most 500 epochs, the protocol of the earlier
# SEC-GFD benchmark in this repo (100 epochs stops YelpChi while it is still improving). Elliptic /
# FDCompCN are not in the paper: their values are only the fallback when
# configs/tuned/SEC_GFD/<dataset>.json is absent (the earlier validation-only search of
# this repo selected them).
_BASE = dict(hid_dim=64, order=2, high_order=2, lemda=0.2, learning_rate=0.01, weight_decay=0.0, epochs=500)
DATASET_DEFAULTS = {
    'yelpchi': dict(_BASE),
    'amazon': dict(_BASE),
    'tfinance': dict(_BASE),
    'tsocial': dict(_BASE, hid_dim=16, order=5),
    'elliptic': dict(_BASE, hid_dim=16, order=2),
    'comp': dict(_BASE, hid_dim=64, order=2),
}


def apply_dataset_defaults(args):
    defaults = DATASET_DEFAULTS.get(canonical_dataset(args.dataset), DATASET_DEFAULTS['yelpchi'])
    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    return args


def parse_args():
    parser = argparse.ArgumentParser(description='SEC-GFD')
    parser.add_argument('--dataset', type=str, default='tfinance',
                        help='yelpchi / amazon / tfinance / tsocial / elliptic / comp')
    parser.add_argument('--hid_dim', type=int, default=None)
    parser.add_argument('--order', type=int, default=None, help='Beta-wavelet order C')
    parser.add_argument('--high_order', type=int, default=None, help='Number of high-pass filters')
    parser.add_argument('--lemda', type=float, default=None, help='Weight of the environmental loss')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--data_prefix', type=str, default='../../dataset/')
    parser.add_argument('--run_tag', type=str, default=None)
    add_save_args(parser)
    add_protocol_split_args(parser)
    return apply_tuned_hparams(parser.parse_args(), 'SEC_GFD', parser)


def load_secgfd_graph(args):
    """Homogeneous graph as in the released loader.

    YelpChi / Amazon: all relations merged (``dgl.to_homogeneous``, duplicate edges kept)
    plus self-loops. T-Finance / T-Social / Elliptic: the single graph as stored.
    FDCompCN: its homogeneous ``homo`` relation. Returns features, labels, src, dst and
    the splittable node index (Amazon: labelled nodes from 3305).
    """
    ds = canonical_dataset(args.dataset)
    if ds == 'comp':
        data = load_comp_arrays('comp', prefix=args.data_prefix)
        features, labels = data['features'], data['labels']
        rel = {name: (s, d) for name, s, d in data['relations']}
        src, dst = rel['homo'] if 'homo' in rel else next(iter(rel.values()))
        labeled_from = None
    else:
        features, labels, relations, labeled_from = load_dga_arrays(ds, prefix=args.data_prefix)
        src = np.concatenate([np.asarray(s, dtype=np.int64) for _n, s, _d in relations])
        dst = np.concatenate([np.asarray(d, dtype=np.int64) for _n, _s, d in relations])
    n = int(features.shape[0])
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    if ds in ('yelpchi', 'amazon'):
        loops = np.arange(n, dtype=np.int64)
        src, dst = np.concatenate([src, loops]), np.concatenate([dst, loops])
    index = np.arange(labeled_from, n, dtype=np.int64) if labeled_from is not None else None
    print('SEC-GFD graph: {} nodes, {} edges, {} features'.format(n, src.size, features.shape[1]))
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64), src, dst, index


def build_operators(n, src, dst, device):
    """Filtering adjacency + in-degree^-1/2, and the GCN operator on the graph without self-loops."""
    src_t = torch.as_tensor(src, dtype=torch.long)
    dst_t = torch.as_tensor(dst, dtype=torch.long)
    adj = torch.sparse_coo_tensor(torch.stack([dst_t, src_t]), torch.ones(src_t.numel()),
                                  (n, n)).coalesce()            # duplicate edges summed, as in DGL
    in_deg = torch.bincount(dst_t, minlength=n).float()
    d_invsqrt = in_deg.clamp(min=1).pow(-0.5).view(-1, 1)

    keep = src_t != dst_t
    s, d = src_t[keep], dst_t[keep]
    out_deg = torch.bincount(s, minlength=n).float().clamp(min=1)
    in_deg_nl = torch.bincount(d, minlength=n).float().clamp(min=1)
    vals = out_deg[s].pow(-0.5) * in_deg_nl[d].pow(-0.5)
    gcn_adj = torch.sparse_coo_tensor(torch.stack([d, s]), vals, (n, n)).coalesce()
    return adj.to(device), d_invsqrt.to(device), gcn_adj.to(device)


def _split_auprc(prob_pos, labels_np, idx):
    y = labels_np[idx]
    if len(np.unique(y)) < 2:
        return None
    return float(average_precision_score(y, prob_pos[idx]))


def train(args=None):
    """Train SEC-GFD. ``args`` is an argparse Namespace (see ``parse_args``)."""
    if args is None:
        args = parse_args()
    args = apply_dataset_defaults(args)
    args.dataset_str = canonical_dataset(args.dataset)
    if getattr(args, 'train_ratio', None) is None:
        args.train_ratio = 0.4
    set_seed(args.seed)
    device = get_device(args.gpu)

    features, labels_np, src, dst, index = load_secgfd_graph(args)
    n = features.shape[0]
    train_np, valid_np, test_np, discarded_np, split_info = protocol_split(
        labels_np, train_ratio=args.train_ratio, seed=args.seed, indices=index)
    adj, d_invsqrt, gcn_adj = build_operators(n, src, dst, device)
    del src, dst

    x = torch.from_numpy(features).to(device)
    labels = torch.as_tensor(labels_np, dtype=torch.long, device=device)
    train_idx = torch.as_tensor(train_np, dtype=torch.long, device=device)

    model = SECGFD(features.shape[1], args.hid_dim, 2, order=args.order, high_order=args.high_order).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    y_train = labels[train_idx]
    pos = y_train.sum().clamp(min=1).float()
    weight = torch.tensor([1.0, float((y_train.numel() - pos) / pos)], device=device)  # released code
    print('cross entropy weight:', weight.tolist())

    early_stop = int(getattr(args, 'early_stop', 0) or 0)
    best_state, best_auc, best_epoch, stop_steps = None, -1.0, 0, 0
    t_start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        logits, emb = model(x, adj, d_invsqrt, gcn_adj)
        ce = F.cross_entropy(logits[train_idx], y_train, weight=weight)
        env = environment_loss(x, emb, labels, train_idx)
        loss = ce + args.lemda * env
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits, _ = model(x, adj, d_invsqrt, gcn_adj)
        prob_pos = torch.softmax(val_logits, dim=-1)[:, 1].cpu().numpy()
        auc = _split_auprc(prob_pos, labels_np, valid_np)
        auc = -1.0 if auc is None else auc
        if auc > best_auc:
            best_auc, best_epoch, stop_steps = float(auc), epoch, 1
            best_state = copy.deepcopy(model.state_dict())
        else:
            stop_steps += 1
        print('Epoch {:d} loss: {:.4f} (ce {:.4f}, env {:.4f}) valid AUPRC: {:.4f} (best {:.4f} @ {:d})'.format(
            epoch, loss.item(), ce.item(), env.item(), float(auc), best_auc, best_epoch))
        if early_stop and stop_steps >= early_stop:
            print('Early stop at epoch {:d}'.format(epoch))
            break
    train_time = time.perf_counter() - t_start
    print('train time=', '{:.5f}'.format(train_time))

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, _ = model(x, adj, d_invsqrt, gcn_adj)
    prob = torch.softmax(logits, dim=-1).cpu().numpy()
    pred = np.eye(prob.shape[1], dtype=np.float32)[prob.argmax(axis=1)]

    def split_metrics(idx):
        return compute_classification_metrics(labels_np[idx], pred[idx], prob[idx], sigmoid=False)

    train_metrics, valid_metrics = split_metrics(train_np), split_metrics(valid_np)
    test_metrics = dict(split_metrics(test_np))
    test_metrics['valid_auc'] = valid_metrics.get('auroc')
    test_metrics['valid_auprc'] = valid_metrics.get('auprc')
    print('Train metrics:', train_metrics)
    print('Valid metrics:', valid_metrics)
    print('Test metrics:', test_metrics)

    out_dir = None
    if not args.no_save:
        config = {k: v for k, v in vars(args).items()}
        config.update(model='SEC_GFD', best_valid_auprc=best_auc if best_auc >= 0 else None,
                      best_epoch=best_epoch,
                      environment_loss='-log(mean exp(cos) benign / mean exp(cos) fraud), global node ids')
        out_dir = run_dir_from_args(args, 'SEC_GFD')
        save_run_outputs(
            out_dir, 'SEC_GFD', config, split_info, test_metrics, model=model, train_metrics=train_metrics,
            predictions=pred[test_np], probabilities=prob[test_np],
            train_indices=train_np, test_indices=test_np, valid_indices=valid_np,
            discarded_indices=discarded_np, y_train=labels_np[train_np], y_test=labels_np[test_np],
            train_time=train_time)
    return test_metrics, out_dir


def main():
    run_guarded(train, parse_args(), 'SEC_GFD')


if __name__ == '__main__':
    main()
