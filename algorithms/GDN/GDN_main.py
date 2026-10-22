'''
Train GDN (Gao et al., WSDM 2023) with unified evaluation artifacts.

Usage (from algorithms/GDN):
  python GDN_main.py --dataset yelpchi --gpu 0
  python GDN_main.py --dataset amazon --gpu 0
  python GDN_main.py --dataset elliptic --gpu 0
  python GDN_main.py --dataset tfinance --gpu 0
  python GDN_main.py --dataset tsocial --gpu 0
  python GDN_main.py --dataset comp --gpu 0
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
from algorithms.GDN.GDN import GDN, build_csr, sample_negatives
from utils.data_loader import load_dga_arrays
from utils.evaluation import compute_classification_metrics, run_output_dir, save_run_outputs
from utils.split import protocol_split
from utils.torch_helpers import get_device, set_seed
from utils.training import add_protocol_split_args, add_save_args

# Reference config/gdn_yelpchi.yml: emb 64, batch 1024, Adam lr 1e-3 / L2 1e-3 for
# the network and lr 1e-5 / L2 0.1 for the trainable feature table, Beta 0.1,
# topk 10, thres 0.2, 101 epochs. Features: raw on YelpChi, row-normalised
# (+0.01) on Amazon (model_handler.py); standardised elsewhere. Datasets with 10
# features keep topk at 3 so that both feature groups are non-empty.
DATASET_DEFAULTS = {
    'yelpchi': dict(feat_norm='none', topk=10, epochs=100, early_stop=20),
    'amazon': dict(feat_norm='row', topk=10, epochs=100, early_stop=20),
    'elliptic': dict(feat_norm='standard', topk=10, epochs=100, early_stop=20),
    'tfinance': dict(feat_norm='standard', topk=3, epochs=100, early_stop=20),
    'tsocial': dict(feat_norm='standard', topk=3, epochs=50, early_stop=10),
    'comp': dict(feat_norm='standard', topk=10, epochs=100, early_stop=20),
}


def parse_args():
    parser = argparse.ArgumentParser(description='GDN')
    parser.add_argument('--dataset', type=str, default='yelpchi',
                        help='yelpchi / amazon / elliptic / tfinance / tsocial / comp')
    parser.add_argument('--emb_size', type=int, default=64)
    parser.add_argument('--topk', type=int, default=None, help='Number of anomaly feature dimensions.')
    parser.add_argument('--beta', type=float, default=0.1, help='Weight of the feature constraints.')
    parser.add_argument('--add_constraint', type=int, default=1,
                        help='1 = CE + both feature constraints, 0 = backbone only.')
    parser.add_argument('--thres', type=float, default=0.2, help='Decision threshold on the fraud probability.')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--eval_batch_size', type=int, default=1024)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--feat_learning_rate', type=float, default=1e-5,
                        help='Learning rate of the trainable feature table.')
    parser.add_argument('--feat_weight_decay', type=float, default=0.1,
                        help='L2 weight of the trainable feature table.')
    parser.add_argument('--feat_norm', type=str, default=None, choices=['row', 'standard', 'none'])
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
    """``row``: reference ``normalize`` (rows divided by ``sum + 0.01``)."""
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


def train_step(model, optimizer, batch, y, rels, topk, beta, add_constraint, negatives=None):
    """One reference mini-batch step: CE, then the fl / fn constraints on the feature table.

    Returns ``(ce_loss, anomaly_feature_idx)``. ``negatives`` overrides the random
    non-neighbour draw of the fn constraint.
    """
    logits, aggs, gathered = model(batch, rels)
    model.update_prototypes()
    loss = F.cross_entropy(logits, y)
    optimizer.zero_grad()
    loss.backward(retain_graph=add_constraint)
    grads_idx = None
    if add_constraint:
        n_feat = model.features.weight.size(1)
        grads_idx = model.features.weight.grad.abs().mean(dim=0).topk(k=topk).indices
        keep = torch.ones(n_feat, dtype=torch.bool, device=grads_idx.device)
        keep[grads_idx] = False
        non_grads_idx = keep.nonzero(as_tuple=True)[0]

        loss_pos, loss_neg = model.fl_loss(grads_idx)
        (beta * torch.exp(loss_pos - loss_neg)).backward()

        target_rows = (y == 1).nonzero(as_tuple=True)[0]
        if target_rows.numel() > 0 and non_grads_idx.numel() > 0:
            if negatives is None:
                negatives = sample_negatives(batch, target_rows, gathered, model.features.num_embeddings)
            fn_pos, fn_neg = model.fn_loss(batch, target_rows, aggs, negatives, non_grads_idx)
            (beta * torch.exp(fn_pos - fn_neg)).backward()
    optimizer.step()
    return loss.item(), grads_idx


@torch.no_grad()
def predict_fraud_prob(model, idx, rels, batch_size):
    """Softmax fraud probability (reference ``to_prob``)."""
    model.eval()
    probs = []
    for i in range(0, idx.numel(), batch_size):
        logits, _aggs, _gathered = model(idx[i:i + batch_size], rels)
        probs.append(torch.softmax(logits, dim=-1)[:, 1])
    return torch.cat(probs, dim=0)


def _auc(labels, prob):
    y = labels.cpu().numpy()
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, prob.cpu().numpy()))


def train(args=None):
    """Train GDN. ``args`` is an argparse Namespace (see ``parse_args``)."""
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

    labels = torch.as_tensor(labels_np, dtype=torch.long, device=device)
    rel_names = [name for name, _s, _d in relations]
    rels = []
    for _name, src, dst in relations:
        indptr, indices = build_csr(src, dst, n)
        rels.append((torch.as_tensor(indptr, device=device), torch.as_tensor(indices, device=device)))
    del relations
    train_labels_np = labels_np[train_np]
    train_pos_np = train_np[train_labels_np == 1]
    train_neg_np = train_np[train_labels_np != 1]
    train_order = rng.permutation(train_np)
    train_idx = torch.as_tensor(train_np, dtype=torch.long, device=device)
    valid_idx = torch.as_tensor(valid_np, dtype=torch.long, device=device)
    test_idx = torch.as_tensor(test_np, dtype=torch.long, device=device)

    n_feat = int(features.shape[1])
    args.topk = max(1, min(int(args.topk), n_feat - 1 if n_feat > 1 else 1))
    n_class = int(labels.max().item()) + 1
    model = GDN(features, args.emb_size, len(rels), train_pos_np, train_neg_np, n_class=n_class).to(device)
    del features
    feat_params = [model.features.weight]
    net_params = [p for name, p in model.named_parameters() if name != 'features.weight']
    optimizer = torch.optim.Adam([
        dict(params=net_params, lr=args.learning_rate, weight_decay=args.weight_decay),
        dict(params=feat_params, lr=args.feat_learning_rate, weight_decay=args.feat_weight_decay),
    ])
    print('GDN dataset={} nodes={} rels={} feat={} ({}) train={} (fraud {}) valid={} test={} '
          'emb={} batch={} lr={} feat_lr={} topk={} beta={} constraint={} thres={}'.format(
              args.dataset_str, n, rel_names, n_feat, args.feat_norm,
              int(train_idx.numel()), len(train_pos_np), int(valid_idx.numel()), int(test_idx.numel()),
              args.emb_size, args.batch_size, args.learning_rate, args.feat_learning_rate,
              args.topk, args.beta, bool(args.add_constraint), args.thres))

    early_stop = int(getattr(args, 'early_stop', 0) or 0)
    anomaly_counts = torch.zeros(n_feat, dtype=torch.long, device=device)
    best_state = None
    best_auc = -1.0
    best_epoch = 0
    stop_steps = 0
    num_batches = max(1, math.ceil(len(train_order) / args.batch_size))
    t_start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for b in range(num_batches):
            batch = torch.as_tensor(train_order[b * args.batch_size:(b + 1) * args.batch_size],
                                    dtype=torch.long, device=device)
            loss, grads_idx = train_step(model, optimizer, batch, labels[batch], rels,
                                         args.topk, args.beta, bool(args.add_constraint))
            if grads_idx is not None:
                anomaly_counts[grads_idx] += 1
            epoch_loss += loss

        val_prob = predict_fraud_prob(model, valid_idx, rels, args.eval_batch_size)
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
            epoch, epoch_loss / num_batches, float(auc), best_auc, best_epoch))
        if early_stop and stop_steps >= early_stop:
            print('Early stop at epoch {:d}'.format(epoch))
            break
    train_time = time.perf_counter() - t_start
    epochs_executed = epoch + 1
    print('train time=', '{:.5f}'.format(train_time))

    if best_state is not None:
        model.load_state_dict(best_state)

    def _evaluate(idx):
        p = predict_fraud_prob(model, idx, rels, args.eval_batch_size).cpu().numpy()
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
        config['model'] = 'GDN'
        config['relations'] = rel_names
        config['best_valid_auc'] = best_auc if best_auc >= 0 else None
        config['best_epoch'] = best_epoch
        config['epochs_executed'] = epochs_executed
        config['feat_dim'] = n_feat
        config['anomaly_feature_counts'] = anomaly_counts.cpu().tolist()
        out_dir = run_output_dir(
            args.save_dir, 'GDN', args.dataset_str, args.seed,
            run_tag=getattr(args, 'run_tag', None))
        save_run_outputs(
            out_dir, 'GDN', config, split_info,
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
