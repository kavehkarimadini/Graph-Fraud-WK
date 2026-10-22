'''
Train H2-FDetector (Shi et al., WWW 2022) with unified evaluation artifacts.

Usage (from algorithms/H2FDetector):
  python H2FDetector_main.py --dataset yelpchi --gpu 0
  python H2FDetector_main.py --dataset amazon --gpu 0
  python H2FDetector_main.py --dataset elliptic --gpu 0
  python H2FDetector_main.py --dataset tfinance --gpu 0
  python H2FDetector_main.py --dataset tsocial --gpu 0
  python H2FDetector_main.py --dataset comp --gpu 0
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
from algorithms.H2FDetector.H2FDetector import H2FDetector, build_edges
from utils.data_loader import load_dga_arrays
from utils.evaluation import compute_classification_metrics, run_output_dir, save_run_outputs
from utils.split import protocol_split
from utils.torch_helpers import get_device, set_seed
from utils.training import add_protocol_split_args, add_save_args

# Reference config/yelp.yaml and config/amazon.yaml (lr 0.1, weight decay 5e-5,
# intra_dim 8, 4 heads, 2 layers, dropout 0.1, up to 1000 epochs). Other
# datasets reuse the YelpChi loss weights. The reference row-normalises
# features; rows whose sum is not positive force standard scaling instead.
DATASET_DEFAULTS = {
    'yelpchi': dict(gamma1=1.2, gamma2=2.0, epochs=1000, early_stop=50, feat_norm='row'),
    'amazon': dict(gamma1=0.4, gamma2=1.4, epochs=1000, early_stop=200, feat_norm='row'),
    'elliptic': dict(gamma1=1.2, gamma2=2.0, epochs=1000, early_stop=50, feat_norm='standard'),
    'tfinance': dict(gamma1=1.2, gamma2=2.0, epochs=1000, early_stop=50, feat_norm='standard'),
    'tsocial': dict(gamma1=1.2, gamma2=2.0, epochs=200, early_stop=20, feat_norm='standard'),
    'comp': dict(gamma1=1.2, gamma2=2.0, epochs=1000, early_stop=50, feat_norm='standard'),
}


def parse_args():
    parser = argparse.ArgumentParser(description='H2-FDetector')
    parser.add_argument('--dataset', type=str, default='yelpchi',
                        help='yelpchi / amazon / elliptic / tfinance / tsocial / comp')
    parser.add_argument('--intra_dim', type=int, default=8)
    parser.add_argument('--head', type=int, default=4)
    parser.add_argument('--n_layer', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--gamma1', type=float, default=None, help='Weight of the edge hinge loss.')
    parser.add_argument('--gamma2', type=float, default=None, help='Weight of the prototype loss.')
    parser.add_argument('--feat_norm', type=str, default=None, choices=['row', 'standard', 'none'])
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=0.1)
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


def _auc(labels, logits):
    y = labels.cpu().numpy()
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()))


def train(args=None):
    """Train H2-FDetector. ``args`` is an argparse Namespace (see ``parse_args``)."""
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

    rel_names = [name for name, _s, _d in relations]
    rels = []
    for _name, src, dst in relations:
        s, d = build_edges(src, dst, n)
        rels.append((torch.as_tensor(s, device=device), torch.as_tensor(d, device=device)))
    homo_src, homo_dst = build_edges(
        np.concatenate([r[1] for r in relations]), np.concatenate([r[2] for r in relations]), n)

    in_train = np.zeros(n, dtype=bool)
    in_train[train_np] = True
    edge_train = in_train[homo_src] & in_train[homo_dst]
    same = labels_np[homo_src] == labels_np[homo_dst]
    edge_pos = np.flatnonzero(edge_train & same)
    edge_neg = np.flatnonzero(edge_train & ~same)
    homo_src_t = torch.as_tensor(homo_src, device=device)
    homo_dst_t = torch.as_tensor(homo_dst, device=device)
    del homo_src, homo_dst, edge_train, same

    feat = torch.as_tensor(features, dtype=torch.float32, device=device)
    labels = torch.as_tensor(labels_np, dtype=torch.long, device=device)
    train_idx = torch.as_tensor(train_np, dtype=torch.long, device=device)
    valid_idx = torch.as_tensor(valid_np, dtype=torch.long, device=device)
    test_idx = torch.as_tensor(test_np, dtype=torch.long, device=device)
    train_label = labels[train_idx]
    train_label_np = labels_np[train_np]
    train_pos = np.flatnonzero(train_label_np == 1)
    train_neg = np.flatnonzero(train_label_np == 0)

    def aux_samples():
        if len(edge_neg):
            e_idx = np.sort(np.concatenate([rng.choice(edge_pos, size=len(edge_neg)), edge_neg]))
        else:
            e_idx = np.asarray([], dtype=np.int64)
        e_idx = torch.as_tensor(e_idx, dtype=torch.long, device=device)
        e_src, e_dst = homo_src_t[e_idx], homo_dst_t[e_idx]
        e_label = torch.where(labels[e_src] == labels[e_dst], 1.0, -1.0)
        proto_idx = np.sort(np.concatenate([rng.choice(train_neg, size=len(train_pos)), train_pos]))
        return (e_src, e_dst, e_label, train_idx, train_label,
                torch.as_tensor(proto_idx, dtype=torch.long, device=device))

    n_class = int(labels.max().item()) + 1
    model = H2FDetector(
        feat.size(1), len(rels), intra_dim=args.intra_dim, head=args.head,
        n_layer=args.n_layer, n_class=n_class, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    print('H2-FDetector dataset={} nodes={} rels={} feat={} ({}) train={} valid={} test={} '
          'train edges +{}/-{} intra={} head={} layers={} g1={} g2={}'.format(
              args.dataset_str, n, rel_names, feat.size(1), args.feat_norm,
              int(train_idx.numel()), int(valid_idx.numel()), int(test_idx.numel()),
              len(edge_pos), len(edge_neg), args.intra_dim, args.head, args.n_layer,
              args.gamma1, args.gamma2))

    early_stop = int(getattr(args, 'early_stop', 0) or 0)
    best_state = None
    best_auc = -1.0
    best_epoch = 0
    stop_steps = 0
    t_start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        logits, edge_loss, proto_loss = model(feat, rels, aux=aux_samples)
        node_idx = np.sort(np.concatenate([
            train_pos, rng.choice(train_neg, size=len(train_pos), replace=False)]))
        node_idx = torch.as_tensor(node_idx, dtype=torch.long, device=device)
        model_loss = F.cross_entropy(logits[train_idx][node_idx], train_label[node_idx])
        loss = model_loss + args.gamma1 * edge_loss + args.gamma2 * proto_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(feat, rels)[valid_idx]
        auc = _auc(labels[valid_idx], val_logits)
        if auc is None:
            auc = -1.0
        if auc > best_auc:
            best_auc = float(auc)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stop_steps = 1
        else:
            stop_steps += 1
        print('Epoch {:d} loss: {:.4f} (ce {:.4f} edge {:.4f} proto {:.4f}) '
              'valid AUC: {:.4f} (best {:.4f} @ {:d})'.format(
                  epoch, loss.item(), model_loss.item(), edge_loss.item(), proto_loss.item(),
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
        prob = torch.softmax(model(feat, rels), dim=-1)
        pred = F.one_hot(prob.argmax(dim=-1), num_classes=n_class).float()

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
        config['model'] = 'H2FDetector'
        config['relations'] = rel_names
        config['best_valid_auc'] = best_auc if best_auc >= 0 else None
        config['best_epoch'] = best_epoch
        config['epochs_executed'] = epochs_executed
        config['feat_dim'] = int(feat.size(1))
        config['train_edges_pos'] = int(len(edge_pos))
        config['train_edges_neg'] = int(len(edge_neg))
        out_dir = run_output_dir(
            args.save_dir, 'H2FDetector', args.dataset_str, args.seed,
            run_tag=getattr(args, 'run_tag', None))
        save_run_outputs(
            out_dir, 'H2FDetector', config, split_info,
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
