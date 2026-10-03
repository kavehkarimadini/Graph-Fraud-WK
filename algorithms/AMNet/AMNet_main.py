'''
Train AMNet (Chai et al., IJCAI 2022) with unified evaluation artifacts.

Usage (from algorithms/AMNet):
  python AMNet_main.py --dataset yelpchi --gpu 0
  python AMNet_main.py --dataset amazon --gpu 0
  python AMNet_main.py --dataset elliptic --gpu 0
  python AMNet_main.py --dataset tfinance --gpu 0
  python AMNet_main.py --dataset tsocial --gpu 0
  python AMNet_main.py --dataset comp --gpu 0
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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from algorithms.AMNet.AMNet import AMNet, build_homo_graph, scaled_laplacian, to_torch_csr
from utils.data_loader import load_dga_arrays
from utils.evaluation import compute_classification_metrics, run_output_dir, save_run_outputs
from utils.split import protocol_split
from utils.torch_helpers import get_device, set_seed
from utils.training import add_protocol_split_args, add_save_args

# Reference config.py: 2 filters, filter lr 5e-2, up to 2000 epochs with patience
# 200. ``order`` is the Bernstein order M. Elliptic uses the 'elliptic' profile,
# T-Finance the 'FinV' profile, and YelpChi the 'yelp' profile on its R-U-R
# relation (the reference's homophilous Yelp). Amazon and FDCompCN use the
# 'yelp' profile on the union graph. T-Social uses the 'yelp' profile with a
# smaller hidden size and a shorter schedule to fit full-batch training.
DATASET_DEFAULTS = {
    'yelpchi': dict(relation='rur', order=5, hidden=64, learning_rate=5e-4, weight_decay=1e-5,
                    beta=1.0, epochs=2000, early_stop=200),
    'amazon': dict(relation='all', order=5, hidden=64, learning_rate=5e-4, weight_decay=1e-5,
                   beta=1.0, epochs=2000, early_stop=200),
    'elliptic': dict(relation='all', order=4, hidden=128, learning_rate=1e-4, weight_decay=1e-5,
                     beta=0.5, epochs=2000, early_stop=200),
    'tfinance': dict(relation='all', order=7, hidden=128, learning_rate=5e-4, weight_decay=1e-5,
                     beta=1.0, epochs=2000, early_stop=200),
    'tsocial': dict(relation='all', order=5, hidden=32, learning_rate=5e-4, weight_decay=1e-5,
                    beta=1.0, epochs=500, early_stop=50),
    'comp': dict(relation='all', order=5, hidden=64, learning_rate=5e-4, weight_decay=1e-5,
                 beta=1.0, epochs=2000, early_stop=200),
}


def parse_args():
    parser = argparse.ArgumentParser(description='AMNet')
    parser.add_argument('--dataset', type=str, default='yelpchi',
                        help='yelpchi / amazon / elliptic / tfinance / tsocial / comp')
    parser.add_argument('--relation', type=str, default=None,
                        help="Relation to use ('all' = union of every relation).")
    parser.add_argument('--hidden', type=int, default=None)
    parser.add_argument('--order', type=int, default=None, help='Bernstein polynomial order (M).')
    parser.add_argument('--n_filters', type=int, default=2, help='Number of Bernstein filters (K).')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--beta', type=float, default=None, help='Weight of the marginal loss.')
    parser.add_argument('--feat_norm', type=str, default='standard',
                        choices=['row', 'minmax', 'standard', 'none'])
    parser.add_argument('--loss_weight', type=str, default='none', choices=['none', 'balanced'],
                        help="Cross-entropy class weights ('none' as in the reference).")
    parser.add_argument('--select_metric', type=str, default='auroc', choices=['auroc', 'auprc'],
                        help='Validation metric for model selection and early stopping.')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate of the MLP / attention / classifier.')
    parser.add_argument('--lr_filter', type=float, default=5e-2,
                        help='Learning rate of the Bernstein filter coefficients.')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='Weight decay of the non-filter parameters.')
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


def _valid_score(labels, prob, metric):
    y = labels.cpu().numpy()
    if len(np.unique(y)) < 2:
        return None
    p = prob[:, 1].cpu().numpy()
    if metric == 'auprc':
        return float(average_precision_score(y, p))
    return float(roc_auc_score(y, p))


def train(args=None):
    """Train AMNet. ``args`` is an argparse Namespace (see ``parse_args``)."""
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
    adj = build_homo_graph([(s, d) for _n, s, d in relations], n)
    num_edges = int(adj.nnz)
    lap = to_torch_csr(scaled_laplacian(adj), device=device)
    del adj, relations

    feat = torch.as_tensor(features, dtype=torch.float32, device=device)
    labels = torch.as_tensor(labels_np, dtype=torch.long, device=device)
    train_idx = torch.as_tensor(train_np, dtype=torch.long, device=device)
    valid_idx = torch.as_tensor(valid_np, dtype=torch.long, device=device)
    test_idx = torch.as_tensor(test_np, dtype=torch.long, device=device)
    train_y = labels[train_idx]
    anomaly_idx = train_idx[train_y == 1]
    normal_idx = train_idx[train_y == 0]

    n_class = int(labels.max().item()) + 1
    model = AMNet(feat.size(1), args.hidden, n_class=n_class, order=args.order,
                  n_filters=args.n_filters, dropout=args.dropout).to(device)
    filt, lin, attn = model.param_groups()
    optimizer = torch.optim.Adam([
        dict(params=filt, lr=args.lr_filter),
        dict(params=lin, lr=args.learning_rate, weight_decay=args.weight_decay),
        dict(params=attn, lr=args.learning_rate, weight_decay=args.weight_decay),
    ])
    if args.loss_weight == 'balanced':
        n_pos = max(int(anomaly_idx.numel()), 1)
        n_neg = max(int(normal_idx.numel()), 1)
        ce_weight = torch.tensor([1.0, n_neg / n_pos], device=device)
    else:
        ce_weight = None
    print('AMNet dataset={} nodes={} rels={} edges={} feat={} ({}) train={} valid={} test={} '
          'hidden={} order={} filters={} beta={} lr={} lr_filter={} wd={} select={}'.format(
              args.dataset_str, n, rel_names, num_edges, feat.size(1), args.feat_norm,
              int(train_idx.numel()), int(valid_idx.numel()), int(test_idx.numel()),
              args.hidden, args.order, args.n_filters, args.beta, args.learning_rate,
              args.lr_filter, args.weight_decay, args.select_metric))

    early_stop = int(getattr(args, 'early_stop', 0) or 0)
    best_state = None
    best_score = -1.0
    best_epoch = 0
    stop_steps = 0
    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        logits, score = model(feat, lap)
        ce = F.cross_entropy(logits[train_idx], train_y, weight=ce_weight)
        margin = model.marginal_loss(score, anomaly_idx, normal_idx)
        loss = ce + args.beta * margin
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_prob = torch.softmax(model(feat, lap)[0][valid_idx], dim=1)
        val_score = _valid_score(labels[valid_idx], val_prob, args.select_metric)
        if val_score is None:
            val_score = -1.0
        if val_score >= best_score:
            best_score = float(val_score)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stop_steps = 0
        else:
            stop_steps += 1
        print('Epoch {:d} loss: {:.4f} (ce {:.4f} margin {:.4f}) valid {}: {:.4f} '
              '(best {:.4f} @ {:d})'.format(
                  epoch, loss.item(), ce.item(), float(margin), args.select_metric,
                  float(val_score), best_score, best_epoch))
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
        logits, score = model(feat, lap)
        prob = torch.softmax(logits, dim=1)
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

    test_y = labels[test_idx]
    test_score = score[test_idx]
    filter_attention = {
        'test_anomaly': test_score[test_y == 1].mean(dim=0).tolist() if bool((test_y == 1).any()) else None,
        'test_normal': test_score[test_y == 0].mean(dim=0).tolist() if bool((test_y == 0).any()) else None,
    }
    print('Mean filter attention on test nodes:', filter_attention)

    out_dir = None
    if not args.no_save:
        config = {k: v for k, v in vars(args).items()}
        config['model'] = 'AMNet'
        config['relations'] = rel_names
        config['num_edges'] = num_edges
        config['best_valid_score'] = best_score if best_score >= 0 else None
        config['best_epoch'] = best_epoch
        config['epochs_executed'] = epochs_executed
        config['feat_dim'] = int(feat.size(1))
        config['filter_coefficients'] = [
            torch.sigmoid(f.weight).view(-1).tolist() for f in model.filters]
        config['filter_attention'] = filter_attention
        out_dir = run_output_dir(
            args.save_dir, 'AMNet', args.dataset_str, args.seed,
            run_tag=getattr(args, 'run_tag', None))
        save_run_outputs(
            out_dir, 'AMNet', config, split_info,
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
