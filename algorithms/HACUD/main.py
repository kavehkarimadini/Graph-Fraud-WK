'''
This code is due to Hengrui Zhang (@hengruizhang98) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection)
https://github.com/safe-graph/DGFraud
'''
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
import scipy.sparse as sp

from parse import parse_args
from get_data import Data
from model import Model
from utils.data_loader import get_split_info
from utils.evaluation import compute_classification_metrics, run_output_dir, save_run_outputs
from utils.torch_helpers import get_device, set_seed
from utils.training import resolve_test_size


def logits_to_prob(logits):
    logits = np.asarray(logits)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def eval_split(model, nodes, labels):
    _, _, _, logits = model.eval_batch(nodes, labels)
    prob = logits_to_prob(logits)
    if torch.is_tensor(labels):
        y_true = labels.detach().cpu().numpy()
    else:
        y_true = np.asarray(labels)
    y_pred = prob
    acc = float((y_pred.argmax(axis=1) == y_true.argmax(axis=1)).mean())
    metrics = compute_classification_metrics(y_true, y_pred, prob, sigmoid=False)
    metrics['accuracy'] = acc
    return acc, y_pred, prob, metrics


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.gpu)
    test_size = resolve_test_size(args, 'HACUD')

    path = '../../dataset/DBLP4057_GAT_with_idx_tra200_val_800.mat'
    save_path = os.path.join('..', 'HACUD', args.dataset)
    data_generator = Data(path=path, save_path=save_path, test_size=test_size, seed=args.seed)

    X_train = np.asarray(data_generator.X_train)
    X_test = np.asarray(data_generator.X_test)
    y_train = np.asarray(data_generator.y_train)
    y_test = np.asarray(data_generator.y_test)

    all_labels = np.zeros((data_generator.n_nodes, y_train.shape[1]))
    all_labels[X_train] = y_train
    all_labels[X_test] = y_test
    split_info = get_split_info(all_labels, X_train, X_test, test_size, args.seed)

    config = {
        'n_nodes': data_generator.n_nodes,
        'n_metapath': data_generator.n_metapath,
        'n_class': y_train.shape[1],
        'features': data_generator.features,
    }
    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()
    if args.adj_type == 'plain':
        config['norm_adj'] = plain_adj
        print('use the plain adjacency matrix')
    elif args.adj_type == 'norm':
        config['norm_adj'] = norm_adj
        print('use the normalized adjacency matrix')
    else:
        config['norm_adj'] = [mean_adj[i] + sp.eye(mean_adj[i].shape[0]) for i in range(len(mean_adj))]
        print('use the mean adjacency matrix')

    model = Model(data_config=config, pretrain_data=None, args=args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32, device=device)
    X_train_t = torch.tensor(X_train, dtype=torch.long, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.long, device=device)

    print('without pretraining.')
    for epoch in range(args.epoch):
        loss, ce_loss, reg_loss = 0., 0., 0.
        n_batch = (data_generator.n_train - 1) // args.batch_size + 1
        for idx in range(n_batch):
            if idx == n_batch - 1:
                nodes = X_train_t[idx * args.batch_size:]
                labels = y_train_t[idx * args.batch_size:]
            else:
                nodes = X_train_t[idx * args.batch_size:(idx + 1) * args.batch_size]
                labels = y_train_t[idx * args.batch_size:(idx + 1) * args.batch_size]
            batch_loss, batch_ce_loss, batch_reg = model.fit_batch(optimizer, nodes, labels)
            loss += batch_loss
            ce_loss += batch_ce_loss
            reg_loss += batch_reg

        _, _, _, test_metrics_epoch = eval_split(model, X_test_t, y_test_t)
        if np.isnan(loss):
            print('ERROR: loss is nan.')
            sys.exit()
        print('Epoch {} Train: {:.4f} CE: {:.4f} Reg: {:.4f} Test F1_mic: {:.4f} F1_mac: {:.4f} Accuracy: {:.4f}'.format(
            epoch, loss, ce_loss, reg_loss,
            test_metrics_epoch['f1_micro'], test_metrics_epoch['f1_macro'], test_metrics_epoch['accuracy']))

    _, train_pred, train_prob, train_metrics = eval_split(model, X_train_t, y_train_t)
    _, test_pred, test_prob, test_metrics = eval_split(model, X_test_t, y_test_t)
    print('Train metrics:', train_metrics)
    print('Test metrics:', test_metrics)

    if not args.no_save:
        run_config = {k: v for k, v in vars(args).items()}
        run_config['model'] = 'HACUD'
        run_config['dataset_str'] = args.dataset
        run_config['test_size'] = test_size
        out_dir = run_output_dir(args.save_dir, 'HACUD', args.dataset, args.seed)
        save_run_outputs(
            out_dir, 'HACUD', run_config, split_info,
            test_metrics, model=model, train_metrics=train_metrics,
            predictions=test_pred, probabilities=test_prob,
            train_indices=X_train, test_indices=X_test,
            y_train=y_train, y_test=y_test,
        )
