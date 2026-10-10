'''
Train H2IDE (Fu, Liu, Yuan, Wu, IEEE TKDE 2025) with unified evaluation artifacts.

Usage (from algorithms/H2IDE):
  python H2IDE_main.py --dataset yelpchi --gpu 0
  python H2IDE_main.py --dataset amazon --gpu 0
  python H2IDE_main.py --dataset elliptic --gpu 0
  python H2IDE_main.py --dataset tfinance --gpu 0
  python H2IDE_main.py --dataset tsocial --gpu 0
  python H2IDE_main.py --dataset comp --gpu 0
'''
from __future__ import print_function

import argparse
import copy
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from algorithms.H2IDE.H2IDE import H2IDE, batch_graph, build_csr
from utils.data_loader import load_dga_arrays
from utils.evaluation import compute_classification_metrics, run_output_dir, save_run_outputs
from utils.split import protocol_split
from utils.torch_helpers import get_device, set_seed
from utils.training import add_protocol_split_args, add_save_args

# Paper Sec. V-A4: Adam lr 0.005, weight decay 5e-5, one layer, 500 epochs, factor
# dimension 8, dropout 0.1, beta 1, alpha3 1e-4. YelpChi: alpha1 1.2, alpha2 0.001,
# batch 256, K 12. Amazon: alpha1 0.8, alpha2 0.01, batch 1024, K 8. T-Finance /
# T-Social: alpha1 1.2, alpha2 0.01, batch 1024, K 8. Elliptic follows T-Finance
# (homogeneous) and FDCompCN follows Amazon (small multi-relation graph).
DATASET_DEFAULTS = {
    'yelpchi': dict(n_factors=12, alpha1=1.2, alpha2=0.001, batch_size=256, epochs=500, early_stop=100),
    'amazon': dict(n_factors=8, alpha1=0.8, alpha2=0.01, batch_size=1024, epochs=500, early_stop=100),
    'elliptic': dict(n_factors=8, alpha1=1.2, alpha2=0.01, batch_size=1024, epochs=500, early_stop=100),
    'tfinance': dict(n_factors=8, alpha1=1.2, alpha2=0.01, batch_size=1024, epochs=500, early_stop=100),
    'tsocial': dict(n_factors=8, alpha1=1.2, alpha2=0.01, batch_size=1024, epochs=50, early_stop=10),
    'comp': dict(n_factors=8, alpha1=0.8, alpha2=0.01, batch_size=1024, epochs=500, early_stop=100),
}


def parse_args():
    parser = argparse.ArgumentParser(description='H2IDE')
    parser.add_argument('--dataset', type=str, default='yelpchi',
                        help='yelpchi / amazon / elliptic / tfinance / tsocial / comp')
    parser.add_argument('--n_factors', type=int, default=None, help='Number of latent factors K.')
    parser.add_argument('--factor_dim', type=int, default=8, help='Embedding size of each factor.')
    parser.add_argument('--tau', type=float, default=0.5,
                        help='Temperature of the cosine proximity phi (not given in the paper).')
    parser.add_argument('--alpha1', type=float, default=None, help='Weight of the identification hinge loss.')
    parser.add_argument('--alpha2', type=float, default=None, help='Weight of the MI independence loss.')
    parser.add_argument('--alpha3', type=float, default=1e-4, help='Weight of the explicit L2 term.')
    parser.add_argument('--mi_beta', type=float, default=1.0,
                        help='Weight beta of the variational log-likelihood F (Eq. 18).')
    parser.add_argument('--mi_hidden', type=int, default=16, help='Hidden size of the CLUB networks.')
    parser.add_argument('--residual', type=int, default=0,
                        help='1 = add the centre factor embedding to its aggregated message (Eq. 10).')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--fanouts', type=int, nargs=2, default=[25, 10],
                        help='Training neighbours sampled per relation at hop 1 / hop 2 (0 = all).')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--eval_edge_budget', type=int, default=1 << 19)
    parser.add_argument('--thres', type=float, default=0.5, help='Decision threshold on the fraud probability.')
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--feat_norm', type=str, default='standard', choices=['row', 'standard', 'none'])
    parser.add_argument('--epochs', type=int, default=None)
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
    """Train H2IDE. ``args`` is an argparse Namespace (see ``parse_args``)."""
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
    csrs, n_edges = [], []
    for _name, src, dst in relations:
        indptr, indices = build_csr(src, dst, n)
        n_edges.append(int(indices.size))
        csrs.append((torch.as_tensor(indptr, device=device), torch.as_tensor(indices, device=device)))
    del relations

    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    labels = torch.as_tensor(labels_np, dtype=torch.long, device=device)
    valid_idx = torch.as_tensor(valid_np, dtype=torch.long, device=device)
    in_train = torch.zeros(n, dtype=torch.bool, device=device)
    in_train[torch.as_tensor(train_np, device=device)] = True
    train_labels_np = labels_np[train_np]
    train_pos = train_np[train_labels_np == 1]
    train_neg = train_np[train_labels_np == 0]

    n_class = int(labels.max().item()) + 1
    model = H2IDE(x.size(1), len(csrs), n_factors=args.n_factors, factor_dim=args.factor_dim,
                  tau=args.tau, dropout=args.dropout, n_class=n_class,
                  mi_hidden=args.mi_hidden, residual=bool(args.residual)).to(device)
    main_params = [p for name, p in model.named_parameters() if not name.startswith('club.')]
    optimizer = torch.optim.Adam(main_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    club_optimizer = torch.optim.Adam(model.club.parameters(), lr=args.learning_rate)
    n_balanced = 2 * min(len(train_pos), len(train_neg))
    steps = max(1, math.ceil(n_balanced / args.batch_size))
    print('H2IDE dataset={} nodes={} rels={} edges={} feat={} ({}) train={} (fraud {}) valid={} test={} '
          'K={} dk={} tau={} a1={} a2={} a3={} beta={} batch={} steps/epoch={} fanouts={} lr={} wd={}'.format(
              args.dataset_str, n, rel_names, n_edges, x.size(1), args.feat_norm,
              len(train_np), len(train_pos), len(valid_np), len(test_np), args.n_factors,
              args.factor_dim, args.tau, args.alpha1, args.alpha2, args.alpha3, args.mi_beta,
              args.batch_size, steps, args.fanouts, args.learning_rate, args.weight_decay))

    early_stop = int(getattr(args, 'early_stop', 0) or 0)
    best_state = None
    best_auc = -1.0
    best_epoch = 0
    stop_steps = 0
    t_start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        k = n_balanced // 2
        nodes = np.concatenate([rng.choice(train_pos, k, replace=False),
                                rng.choice(train_neg, k, replace=False)])
        rng.shuffle(nodes)
        sums = np.zeros(5)
        for b in range(steps):
            batch = torch.as_tensor(nodes[b * args.batch_size:(b + 1) * args.batch_size],
                                    dtype=torch.long, device=device)
            g = batch_graph(csrs, batch, args.fanouts)
            logits, z, q = model.forward_batch(x, g)

            loss_est = z.new_zeros(())
            loss_mi = z.new_zeros(())
            if args.alpha2 > 0:
                club_optimizer.zero_grad()
                loss_est = -args.mi_beta * model.club.log_likelihood(z.detach())
                loss_est.backward()
                club_optimizer.step()
                loss_mi = model.club.mi_upper(z)

            loss_ce = F.cross_entropy(logits, labels[batch])
            nbr = g['disen_nbr']
            known = in_train[nbr]
            loss_disen = z.new_zeros(())
            if bool(known.any()):
                same = labels[batch][g['disen_seg'][known]] == labels[nbr[known]]
                y_edge = torch.where(same, 1.0, -1.0)
                loss_disen = F.relu(1.0 - y_edge * q[known]).mean()
            loss_l2 = sum(p.pow(2).sum() for p in main_params)
            loss = loss_ce + args.alpha1 * loss_disen + args.alpha2 * loss_mi + args.alpha3 * loss_l2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sums += [loss.item(), loss_ce.item(), loss_disen.item(), loss_mi.item(), loss_est.item()]

        model.eval()
        val_prob = torch.softmax(model.inference(x, csrs, valid_idx, args.eval_edge_budget), dim=-1)[:, 1]
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
        sums /= steps
        print('Epoch {:d} loss: {:.4f} (ce {:.4f} disen {:.4f} mi {:.4f} est {:.4f}) valid AUC: {:.4f} '
              '(best {:.4f} @ {:d})'.format(epoch, *sums, float(auc), best_auc, best_epoch))
        if early_stop and stop_steps >= early_stop:
            print('Early stop at epoch {:d}'.format(epoch))
            break
    train_time = time.perf_counter() - t_start
    epochs_executed = epoch + 1
    print('train time=', '{:.5f}'.format(train_time))

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    all_np = np.concatenate([train_np, valid_np, test_np])
    p_all = torch.softmax(model.inference(
        x, csrs, torch.as_tensor(all_np, dtype=torch.long, device=device), args.eval_edge_budget),
        dim=-1)[:, 1].cpu().numpy()
    sizes = np.cumsum([len(train_np), len(valid_np)])

    def _evaluate(p):
        prob = np.stack([1.0 - p, p], axis=1)
        pred = np.eye(2, dtype=np.float32)[(p >= args.thres).astype(np.int64)]
        return prob, pred

    prob_train, pred_train = _evaluate(p_all[:sizes[0]])
    prob_valid, pred_valid = _evaluate(p_all[sizes[0]:sizes[1]])
    prob_test, pred_test = _evaluate(p_all[sizes[1]:])
    y_train, y_valid, y_test = labels_np[train_np], labels_np[valid_np], labels_np[test_np]
    train_metrics = compute_classification_metrics(y_train, pred_train, prob_train, sigmoid=False)
    valid_metrics = compute_classification_metrics(y_valid, pred_valid, prob_valid, sigmoid=False)
    test_metrics = compute_classification_metrics(y_test, pred_test, prob_test, sigmoid=False)
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
        config['model'] = 'H2IDE'
        config['relations'] = rel_names
        config['relation_edges'] = n_edges
        config['best_valid_auc'] = best_auc if best_auc >= 0 else None
        config['best_epoch'] = best_epoch
        config['epochs_executed'] = epochs_executed
        config['feat_dim'] = int(x.size(1))
        config['steps_per_epoch'] = int(steps)
        out_dir = run_output_dir(
            args.save_dir, 'H2IDE', args.dataset_str, args.seed,
            run_tag=getattr(args, 'run_tag', None))
        save_run_outputs(
            out_dir, 'H2IDE', config, split_info,
            test_metrics, model=model, train_metrics=train_metrics,
            predictions=pred_test,
            probabilities=prob_test,
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
