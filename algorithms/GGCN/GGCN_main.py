'''
Train GGCN (Yan et al., ICDM 2022) with unified evaluation artifacts.

Usage (from algorithms/GGCN):
  python GGCN_main.py --dataset yelpchi --gpu 0
  python GGCN_main.py --dataset amazon --gpu 0
  python GGCN_main.py --dataset elliptic --gpu 0
  python GGCN_main.py --dataset tfinance --gpu 0
  python GGCN_main.py --dataset tsocial --gpu 0
  python GGCN_main.py --dataset comp --gpu 0
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
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from algorithms.GGCN.GGCN import GGCN, EdgeGraph
from utils.data_loader import load_dga_arrays
from utils.evaluation import compute_classification_metrics, run_output_dir, save_run_outputs
from utils.split import protocol_split
from utils.torch_helpers import get_device, set_seed
from utils.training import add_protocol_split_args, add_save_args

# Reference full-supervised.py: Adam lr 0.01, up to 1500 epochs with patience
# 100, decaying residuals (decay_rate 1.0, exponent 3.0), degree correction and
# signed messages on. Layers / hidden / dropout / weight decay follow the
# reference's Squirrel run (table_1_GGCN.sh: 2 layers, hidden 32, dropout 0.5,
# weight decay 4e-3), its largest heterophilous benchmark. T-Social uses a
# shorter schedule for full-batch training on 5.8M nodes. Features are
# standardised: the reference row normalisation targets bag-of-words features
# and collapses YelpChi's numeric features (valid AUROC ~0.6 vs ~0.8).
DATASET_DEFAULTS = {
    'yelpchi': dict(nlayers=2, hidden=32, dropout=0.5, weight_decay=4e-3, epochs=1500, early_stop=100),
    'amazon': dict(nlayers=2, hidden=32, dropout=0.5, weight_decay=4e-3, epochs=1500, early_stop=100),
    'elliptic': dict(nlayers=2, hidden=32, dropout=0.5, weight_decay=4e-3, epochs=1500, early_stop=100),
    'tfinance': dict(nlayers=2, hidden=32, dropout=0.5, weight_decay=4e-3, epochs=1500, early_stop=100),
    'tsocial': dict(nlayers=2, hidden=32, dropout=0.5, weight_decay=4e-3, epochs=500, early_stop=50),
    'comp': dict(nlayers=2, hidden=32, dropout=0.5, weight_decay=4e-3, epochs=1500, early_stop=100),
}


def parse_args():
    parser = argparse.ArgumentParser(description='GGCN')
    parser.add_argument('--dataset', type=str, default='yelpchi',
                        help='yelpchi / amazon / elliptic / tfinance / tsocial / comp')
    parser.add_argument('--relation', type=str, default='all',
                        help="Relation to use ('all' = union of every relation).")
    parser.add_argument('--nlayers', type=int, default=None, help='Number of GGCN layers (>= 2).')
    parser.add_argument('--hidden', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--decay_rate', type=float, default=1.0, help='decay_rate in the decay function.')
    parser.add_argument('--exponent', type=float, default=3.0, help='exponent in the decay function.')
    parser.add_argument('--no_degree', action='store_true', help='Disable degree correction.')
    parser.add_argument('--no_sign', action='store_true', help='Disable signed messages.')
    parser.add_argument('--no_decay', action='store_true', help='Disable decaying residual weights.')
    parser.add_argument('--use_bn', action='store_true', help='Batch norm between layers (only with --no_decay).')
    parser.add_argument('--use_ln', action='store_true', help='Layer norm between layers (only with --no_decay).')
    parser.add_argument('--scale_init', type=float, default=0.5,
                        help='Initial scale when the decaying combination is off.')
    parser.add_argument('--deg_intercept_init', type=float, default=0.5,
                        help='Initial degree slope when the decaying combination is off.')
    parser.add_argument('--feat_norm', type=str, default='standard',
                        choices=['row', 'minmax', 'standard', 'none'],
                        help="'row' is the reference preprocessing (meant for bag-of-words features).")
    parser.add_argument('--loss_weight', type=str, default='none', choices=['none', 'balanced'],
                        help="Cross-entropy class weights ('none' as in the reference).")
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=None)
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
    """``row``: reference ``preprocess_features`` (rows divided by their sum, zero rows kept);
    falls back to standard scaling when features are negative."""
    if mode == 'row':
        if features.min() < 0:
            print('Features contain negative values; using standard scaling instead of row normalisation.')
            mode = 'standard'
        else:
            rowsum = features.sum(axis=1, keepdims=True)
            rowsum[rowsum == 0] = 1.0
            return (features / rowsum).astype(np.float32), mode
    if mode == 'minmax':
        return MinMaxScaler().fit_transform(features).astype(np.float32), mode
    if mode == 'standard':
        return StandardScaler().fit_transform(features).astype(np.float32), mode
    return features.astype(np.float32), mode


def select_relations(relations, name):
    if name in (None, 'all'):
        return relations
    chosen = [r for r in relations if r[0] == name]
    if not chosen:
        raise ValueError("Relation '{}' not found; available: {}".format(
            name, [r[0] for r in relations]))
    return chosen


def _auc(labels, prob):
    y = labels.cpu().numpy()
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, prob[:, 1].cpu().numpy()))


def train(args=None):
    """Train GGCN. ``args`` is an argparse Namespace (see ``parse_args``)."""
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

    relations = select_relations(relations, args.relation)
    rel_names = [name for name, _s, _d in relations]
    graph = EdgeGraph([(s, d) for _n, s, d in relations], n, device)
    del relations

    feat = torch.as_tensor(features, dtype=torch.float32, device=device)
    labels = torch.as_tensor(labels_np, dtype=torch.long, device=device)
    train_idx = torch.as_tensor(train_np, dtype=torch.long, device=device)
    valid_idx = torch.as_tensor(valid_np, dtype=torch.long, device=device)
    test_idx = torch.as_tensor(test_np, dtype=torch.long, device=device)
    train_y = labels[train_idx]

    n_class = int(labels.max().item()) + 1
    use_decay = not args.no_decay
    model = GGCN(feat.size(1), args.hidden, n_class=n_class, nlayers=args.nlayers,
                 dropout=args.dropout, decay_rate=args.decay_rate, exponent=args.exponent,
                 use_degree=not args.no_degree, use_sign=not args.no_sign, use_decay=use_decay,
                 scale_init=args.scale_init, deg_intercept_init=args.deg_intercept_init,
                 use_bn=args.use_bn, use_ln=args.use_ln).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    if args.loss_weight == 'balanced':
        n_pos = max(int(train_y.sum().item()), 1)
        ce_weight = torch.tensor([1.0, (train_y.numel() - n_pos) / n_pos], device=device)
    else:
        ce_weight = None
    print('GGCN dataset={} nodes={} rels={} edges={} feat={} ({}) train={} valid={} test={} '
          'layers={} hidden={} dropout={} wd={} decay_rate={} exponent={} degree={} sign={} decay={}'.format(
              args.dataset_str, n, rel_names, graph.num_edges, feat.size(1), args.feat_norm,
              int(train_idx.numel()), int(valid_idx.numel()), int(test_idx.numel()),
              args.nlayers, args.hidden, args.dropout, args.weight_decay, args.decay_rate,
              args.exponent, not args.no_degree, not args.no_sign, use_decay))

    early_stop = int(getattr(args, 'early_stop', 0) or 0)
    best_state = None
    best_auc = -1.0
    best_epoch = 0
    stop_steps = 0
    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        logits = model(feat, graph)
        loss = F.cross_entropy(logits[train_idx], train_y, weight=ce_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_prob = torch.softmax(model(feat, graph)[valid_idx], dim=1)
        auc = _auc(labels[valid_idx], val_prob)
        if auc is None:
            auc = -1.0
        if auc > best_auc:
            best_auc = float(auc)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stop_steps = 0
        else:
            stop_steps += 1
        print('Epoch {:d} loss: {:.4f} valid AUC: {:.4f} (best {:.4f} @ {:d})'.format(
            epoch, loss.item(), float(auc), best_auc, best_epoch))
        if early_stop and stop_steps >= early_stop:
            print('Early stop at epoch {:d}'.format(epoch))
            break
    train_time = time.perf_counter() - t_start
    epochs_executed = epoch
    print('train time=', '{:.5f}'.format(train_time))

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prob = torch.softmax(model(feat, graph), dim=1)
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

    layer_mix = []
    with torch.no_grad():
        for conv in model.convs:
            entry = {}
            if hasattr(conv, 'coeff'):
                entry['pos_neg_self'] = torch.softmax(conv.coeff, dim=-1).tolist()
                entry['scale'] = F.softplus(conv.scale).item()
            if hasattr(conv, 'deg_coeff'):
                entry['deg_coeff'] = conv.deg_coeff.tolist()
            layer_mix.append(entry)
    print('Learned layer coefficients:', layer_mix)

    out_dir = None
    if not args.no_save:
        config = {k: v for k, v in vars(args).items()}
        config['model'] = 'GGCN'
        config['relations'] = rel_names
        config['num_edges'] = graph.num_edges
        config['best_valid_auc'] = best_auc if best_auc >= 0 else None
        config['best_epoch'] = best_epoch
        config['epochs_executed'] = epochs_executed
        config['feat_dim'] = int(feat.size(1))
        config['layer_coefficients'] = layer_mix
        out_dir = run_output_dir(
            args.save_dir, 'GGCN', args.dataset_str, args.seed,
            run_tag=getattr(args, 'run_tag', None))
        save_run_outputs(
            out_dir, 'GGCN', config, split_info,
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
