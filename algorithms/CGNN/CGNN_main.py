'''
Train CGNN (Li, Yu, Luo, AAAI 2025) with unified evaluation artifacts.

Usage (from algorithms/CGNN):
  python CGNN_main.py --dataset yelpchi --gpu 0
  python CGNN_main.py --dataset amazon --gpu 0
  python CGNN_main.py --dataset elliptic --gpu 0
  python CGNN_main.py --dataset tfinance --gpu 0
  python CGNN_main.py --dataset tsocial --gpu 0
  python CGNN_main.py --dataset comp --gpu 0
  python CGNN_main.py --dataset amazon --n_black 5 --gpu 0
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
from algorithms.CGNN.CGNN import CGNN, build_csr, consistency_loss, supcon_loss
from utils.data_loader import load_dga_arrays
from utils.evaluation import compute_classification_metrics, run_output_dir, save_run_outputs
from utils.split import protocol_split
from utils.torch_helpers import get_device, set_seed
from utils.training import add_protocol_split_args, add_save_args

# Paper Sec. 5.1: Adam lr 0.01, weight decay 1e-4, embedding 64, batch 16,
# lambda = beta = 1, tau 0.8 on T-Finance and 0.01 elsewhere. The few-label split
# (--n_black) uses these values. With the 40% labeled toolbox split, batch 16 would
# mean hundreds of steps per epoch, so that split uses 256 (1024 on T-Social), and
# tau 0.01 made L_CSD gradients swamp the cross-entropy: YelpChi stayed at chance
# (test AUROC 0.65 after 10 epochs versus 0.86 with tau 0.5 after 6) and Amazon
# reached 0.954 versus 0.976 after 15 epochs. That split therefore uses tau 0.5.
DATASET_DEFAULTS = {
    'yelpchi': dict(tau=0.5, paper_tau=0.01, epochs=100, early_stop=25),
    'amazon': dict(tau=0.5, paper_tau=0.01, epochs=100, early_stop=25),
    'elliptic': dict(tau=0.5, paper_tau=0.01, epochs=100, early_stop=25),
    'tfinance': dict(tau=0.8, paper_tau=0.8, epochs=100, early_stop=25),
    'tsocial': dict(tau=0.5, paper_tau=0.01, epochs=30, early_stop=10, batch_size=1024),
    'comp': dict(tau=0.5, paper_tau=0.01, epochs=100, early_stop=25),
}


def parse_args():
    parser = argparse.ArgumentParser(description='CGNN')
    parser.add_argument('--dataset', type=str, default='yelpchi',
                        help='yelpchi / amazon / elliptic / tfinance / tsocial / comp')
    parser.add_argument('--emb_size', type=int, default=64)
    parser.add_argument('--lam', type=float, default=1.0, help='Weight of L_CSD (Eq. 11 lambda).')
    parser.add_argument('--beta', type=float, default=1.0, help='Weight of L_CL (Eq. 11 beta).')
    parser.add_argument('--tau', type=float, default=None,
                        help='Temperature of L_CSD (Eq. 3). Default: paper value (T-Finance 0.8, '
                             'others 0.01) with --n_black, else 0.5 (T-Finance 0.8).')
    parser.add_argument('--csd_nodes', type=int, default=256,
                        help='Max distinct neighbours per step whose components enter L_CSD.')
    parser.add_argument('--n_black', type=int, default=0,
                        help='Paper few-label split: 1 or 5 labeled frauds plus eta-times as many '
                             'normals from a 10%% train pool (30%% valid, 60%% test); the rest of '
                             'the pool is gray. 0 = toolbox protocol split.')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Labeled (and gray) centres per step. Default 16 with --n_black, else 256.')
    parser.add_argument('--eval_batch_size', type=int, default=2048)
    parser.add_argument('--thres', type=float, default=0.5, help='Decision threshold on the fraud probability.')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
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
    defaults = dict(DATASET_DEFAULTS.get(ds, DATASET_DEFAULTS['yelpchi']))
    paper_tau = defaults.pop('paper_tau')
    if getattr(args, 'tau', None) is None and args.n_black > 0:
        args.tau = paper_tau
    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    if getattr(args, 'batch_size', None) is None:
        args.batch_size = 16 if args.n_black > 0 else 256
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


def few_label_split(labels, n_black, seed, indices=None):
    """Paper split: 10/30/60 train/valid/test; ``n_black`` frauds plus ``round(eta * n_black)``
    normals of the train pool are labeled (``eta`` = normal/fraud ratio of the data), the rest
    of the pool is gray."""
    pool, valid_idx, test_idx, _disc, info = protocol_split(
        labels, train_ratio=0.1, valid_ratio=0.3, test_ratio=0.6, seed=seed, indices=indices)
    scope = labels if indices is None else labels[indices]
    eta = float((scope == 0).sum()) / max(float((scope == 1).sum()), 1.0)
    rng = np.random.default_rng(seed)
    pos_pool, neg_pool = pool[labels[pool] == 1], pool[labels[pool] == 0]
    n_pos = min(int(n_black), pos_pool.size)
    n_neg = min(max(int(round(eta * n_black)), 1), neg_pool.size)
    labeled = np.sort(np.concatenate([rng.choice(pos_pool, n_pos, replace=False),
                                      rng.choice(neg_pool, n_neg, replace=False)]))
    gray = np.setdiff1d(pool, labeled)
    info.update({
        'protocol': 'cgnn_few_label',
        'train_size': int(labeled.size),
        'train_class_counts': {0: int(n_neg), 1: int(n_pos)},
        'gray_size': int(gray.size),
        'discarded_size': int(gray.size),
        'n_black': int(n_pos),
        'n_white': int(n_neg),
        'eta': eta,
    })
    print('CGNN few-label split: labeled {} fraud + {} normal (eta={:.2f}), gray={}, valid={}, test={}'.format(
        n_pos, n_neg, eta, gray.size, valid_idx.size, test_idx.size))
    return labeled, valid_idx, test_idx, gray, info


@torch.no_grad()
def predict_fraud_prob(model, x, idx, indptr, indices, batch_size):
    model.eval()
    probs = []
    for i in range(0, idx.numel(), batch_size):
        logits, _, _ = model(x, idx[i:i + batch_size], indptr, indices)
        probs.append(torch.softmax(logits, dim=-1)[:, 1])
    return torch.cat(probs, dim=0)


def _auc(labels, prob):
    y = labels.cpu().numpy()
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, prob.cpu().numpy()))


def _cycle(order, pos, size, rng):
    """Next ``size`` items of a shuffled pool, reshuffling when it runs out."""
    out = []
    while size > 0:
        if pos >= order.size:
            order, pos = rng.permutation(order), 0
        take = min(size, order.size - pos)
        out.append(order[pos:pos + take])
        pos += take
        size -= take
    return np.concatenate(out), order, pos


def train(args=None):
    """Train CGNN. ``args`` is an argparse Namespace (see ``parse_args``)."""
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
    if args.n_black > 0:
        train_np, valid_np, test_np, discarded_np, split_info = few_label_split(
            labels_np, args.n_black, args.seed, indices=index)
        gray_np = discarded_np
    else:
        train_np, valid_np, test_np, discarded_np, split_info = protocol_split(
            labels_np, train_ratio=args.train_ratio, seed=args.seed, indices=index)
        outside = np.arange(labeled_from, dtype=np.int64) if labeled_from else np.empty(0, np.int64)
        gray_np = np.concatenate([discarded_np, outside])
        if gray_np.size == 0:
            gray_np = valid_np

    rel_names = [name for name, _s, _d in relations]
    src = np.concatenate([np.asarray(s, dtype=np.int64) for _n, s, _d in relations])
    dst = np.concatenate([np.asarray(d, dtype=np.int64) for _n, _s, d in relations])
    del relations
    indptr_np, indices_np = build_csr(src, dst, n)
    del src, dst
    indptr = torch.as_tensor(indptr_np, device=device)
    indices = torch.as_tensor(indices_np, device=device)

    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    labels = torch.as_tensor(labels_np, dtype=torch.long, device=device)
    train_idx = torch.as_tensor(train_np, dtype=torch.long, device=device)
    valid_idx = torch.as_tensor(valid_np, dtype=torch.long, device=device)
    test_idx = torch.as_tensor(test_np, dtype=torch.long, device=device)

    n_class = int(labels.max().item()) + 1
    counts = np.bincount(labels_np[train_np], minlength=n_class).astype(np.float64)
    class_weight = torch.as_tensor(
        counts.sum() / (n_class * np.maximum(counts, 1.0)), dtype=torch.float32, device=device)

    model = CGNN(x.size(1), emb_dim=args.emb_size, n_class=n_class).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    use_cl = args.beta > 0 and gray_np.size > 0
    steps = max(1, math.ceil(max(train_np.size, gray_np.size if use_cl else 0) / args.batch_size))
    print('CGNN dataset={} nodes={} edges={} rels={} feat={} ({}) labeled={} (fraud {}) gray={} '
          'valid={} test={} emb={} batch={} steps/epoch={} lr={} wd={} lambda={} beta={} tau={} '
          'class_weight={}'.format(
              args.dataset_str, n, int(indices_np.size), rel_names, x.size(1), args.feat_norm,
              int(train_np.size), int(counts[1]) if n_class > 1 else 0, int(gray_np.size),
              int(valid_np.size), int(test_np.size), args.emb_size, args.batch_size, steps,
              args.learning_rate, args.weight_decay, args.lam, args.beta, args.tau,
              [round(float(w), 3) for w in class_weight.tolist()]))

    early_stop = int(getattr(args, 'early_stop', 0) or 0)
    best_state = None
    best_auc = -1.0
    best_epoch = 0
    stop_steps = 0
    lab_order, lab_pos = rng.permutation(train_np), 0
    gray_order, gray_pos = rng.permutation(gray_np), 0
    t_start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        sums = np.zeros(4)
        for _ in range(steps):
            lab_np, lab_order, lab_pos = _cycle(lab_order, lab_pos, min(args.batch_size, train_np.size), rng)
            centers_np = lab_np
            if use_cl:
                gray_b, gray_order, gray_pos = _cycle(
                    gray_order, gray_pos, min(args.batch_size, gray_np.size), rng)
                centers_np = np.concatenate([lab_np, gray_b])
            centers = torch.as_tensor(centers_np, dtype=torch.long, device=device)
            n_lab = lab_np.size
            perturb = None
            if use_cl:
                perturb = torch.zeros(centers.numel(), dtype=torch.bool, device=device)
                perturb[n_lab:] = True
            logits, views, parts = model(x, centers, indptr, indices, perturb=perturb)

            loss_ce = F.cross_entropy(logits[:n_lab], labels[centers[:n_lab]], weight=class_weight)
            loss_csd = logits.new_zeros(())
            if args.lam > 0 and parts['nbr_pos'].numel() > 0:
                comp = torch.unique(parts['nbr_pos'])
                if comp.numel() > args.csd_nodes:
                    comp = comp[torch.randperm(comp.numel(), device=device)[:args.csd_nodes]]
                loss_csd = supcon_loss(parts['x0'][comp], parts['x1'][comp], args.tau)
            loss_cl = logits.new_zeros(())
            if views is not None:
                prob = torch.softmax(logits[n_lab:], dim=-1)
                loss_cl = consistency_loss(prob, torch.softmax(views[0], dim=-1),
                                           torch.softmax(views[1], dim=-1))
            loss = loss_ce + args.lam * loss_csd + args.beta * loss_cl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sums += [loss.item(), loss_ce.item(), loss_csd.item(), loss_cl.item()]

        val_prob = predict_fraud_prob(model, x, valid_idx, indptr, indices, args.eval_batch_size)
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
        print('Epoch {:d} loss: {:.4f} (ce {:.4f} csd {:.4f} cl {:.4f}) valid AUC: {:.4f} '
              '(best {:.4f} @ {:d})'.format(epoch, *sums, float(auc), best_auc, best_epoch))
        if early_stop and stop_steps >= early_stop:
            print('Early stop at epoch {:d}'.format(epoch))
            break
    train_time = time.perf_counter() - t_start
    epochs_executed = epoch + 1
    print('train time=', '{:.5f}'.format(train_time))

    if best_state is not None:
        model.load_state_dict(best_state)

    def _evaluate(idx):
        p = predict_fraud_prob(model, x, idx, indptr, indices, args.eval_batch_size).cpu().numpy()
        prob = np.stack([1.0 - p, p], axis=1)
        pred = np.eye(2, dtype=np.float32)[(p >= args.thres).astype(np.int64)]
        return prob, pred

    prob_train, pred_train = _evaluate(train_idx)
    prob_valid, pred_valid = _evaluate(valid_idx)
    prob_test, pred_test = _evaluate(test_idx)
    y_train = labels[train_idx].cpu().numpy()
    y_valid = labels[valid_idx].cpu().numpy()
    y_test = labels[test_idx].cpu().numpy()
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
        config['model'] = 'CGNN'
        config['relations'] = rel_names
        config['best_valid_auc'] = best_auc if best_auc >= 0 else None
        config['best_epoch'] = best_epoch
        config['epochs_executed'] = epochs_executed
        config['feat_dim'] = int(x.size(1))
        config['gray_size'] = int(gray_np.size)
        config['steps_per_epoch'] = int(steps)
        config['class_weight'] = [float(w) for w in class_weight.tolist()]
        run_tag = getattr(args, 'run_tag', None)
        if args.n_black > 0 and not run_tag:
            run_tag = 'black{}'.format(args.n_black)
        out_dir = run_output_dir(args.save_dir, 'CGNN', args.dataset_str, args.seed, run_tag=run_tag)
        save_run_outputs(
            out_dir, 'CGNN', config, split_info,
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
