'''
Train BWGNN (Tang et al., ICML 2022) with unified evaluation artifacts.

Usage (from algorithms/BWGNN):
  python BWGNN_main.py --dataset tfinance --gpu 0
  python BWGNN_main.py --dataset tsocial --hid_dim 10 --order 5 --gpu 0
'''
import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from algorithms.BWGNN.BWGNN import BWGNN
from utils.data_loader import (
    is_comp_name, load_comp_graph, load_elliptic_graph, load_mat_homo_graph,
    load_tgraph_dgl,
)
from utils.evaluation import compute_classification_metrics, run_output_dir, save_run_outputs
from utils.split import protocol_split
from utils.torch_helpers import get_device, set_seed
from utils.training import add_protocol_split_args, add_save_args


def parse_args():
    parser = argparse.ArgumentParser(description='BWGNN')
    parser.add_argument('--dataset', type=str, default='tfinance',
                        help="tfinance / tsocial / yelpchi / amazon / elliptic / comp")
    parser.add_argument('--hid_dim', type=int, default=64)
    parser.add_argument('--order', type=int, default=2, help='Beta wavelet order C')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--data_prefix', type=str, default='../../dataset/')
    parser.add_argument('--run_tag', type=str, default=None,
                        help='Optional suffix on the output folder (e.g. a timestamp).')
    add_save_args(parser)
    add_protocol_split_args(parser)
    return parser.parse_args()


def load_bwgnn_graph(args):
    ds = args.dataset.lower()
    prefix = args.data_prefix
    if ds in ('tfinance', 'tsocial'):
        return load_tgraph_dgl(ds, prefix=prefix)
    if ds in ('yelp', 'yelpchi'):
        return load_mat_homo_graph(os.path.join(prefix, 'YelpChi.mat'), 'net_rur')
    if ds == 'amazon':
        return load_mat_homo_graph(os.path.join(prefix, 'Amazon.mat'), 'homo')
    if ds in ('elliptic', 'elliptic_bitcoin', 'elliptic_bitcoin_dataset'):
        return load_elliptic_graph(prefix=prefix)
    if is_comp_name(ds):
        return load_comp_graph(ds, prefix=prefix)
    raise ValueError(
        "Unknown dataset '{}'. Use tfinance, tsocial, yelpchi, amazon, elliptic, or comp.".format(
            args.dataset))


def class_weight(labels, train_idx, device):
    y = labels[train_idx]
    n_pos = y.float().sum().clamp(min=1.0)
    n_neg = (y.numel() - n_pos).clamp(min=1.0)
    return torch.tensor([1.0, (n_neg / n_pos).item()], device=device)


def _split_auc(logits, labels, idx):
    if idx is None or idx.numel() == 0:
        return None
    y = labels[idx].detach().cpu().numpy()
    if len(np.unique(y)) < 2:
        return None
    prob = torch.softmax(logits[idx], dim=-1)[:, 1].detach().cpu().numpy()
    return float(roc_auc_score(y, prob))


def train(args=None):
    """Train BWGNN. ``args`` is an argparse Namespace (see ``parse_args``)."""
    if args is None:
        args = parse_args()
    args.dataset_str = args.dataset
    train_ratio = getattr(args, 'train_ratio', 0.4)
    if train_ratio is None:
        train_ratio = 0.4
    args.train_ratio = train_ratio
    set_seed(args.seed)
    device = get_device(args.gpu)

    graph = load_bwgnn_graph(args)
    labels_np = graph.ndata['label'].cpu().numpy()
    train_np, valid_np, test_np, discarded_np, split_info = protocol_split(
        labels_np, train_ratio=train_ratio, seed=args.seed)

    graph = graph.to(device)
    features = graph.ndata['feature']
    labels = graph.ndata['label']
    train_idx = torch.as_tensor(train_np, dtype=torch.long, device=device)
    valid_idx = torch.as_tensor(valid_np, dtype=torch.long, device=device)
    test_idx = torch.as_tensor(test_np, dtype=torch.long, device=device)

    in_feats = features.size(1)
    num_classes = int(labels.max().item()) + 1
    model = BWGNN(in_feats, args.hid_dim, num_classes, graph, d=args.order).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    weight = class_weight(labels, train_idx, device)
    print('cross entropy weight:', weight.tolist())
    early_stop = int(getattr(args, 'early_stop', 0) or 0)

    best_state = None
    best_auc = -1.0
    best_epoch = 0
    stop_steps = 0
    t_start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        logits = model(features)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx], weight=weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(features)
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
        logits = model(features)
        prob = torch.softmax(logits, dim=-1)
        pred = torch.nn.functional.one_hot(prob.argmax(dim=-1), num_classes=num_classes).float()

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
        config['model'] = 'BWGNN'
        config['best_valid_auc'] = best_auc if best_auc >= 0 else None
        config['best_epoch'] = best_epoch
        out_dir = run_output_dir(
            args.save_dir, 'BWGNN', args.dataset, args.seed,
            run_tag=getattr(args, 'run_tag', None))
        save_run_outputs(
            out_dir, 'BWGNN', config, split_info,
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
