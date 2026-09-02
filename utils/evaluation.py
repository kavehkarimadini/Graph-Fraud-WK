"""Unified metrics computation and run artifact persistence."""
import json
import os

import numpy as np
import torch
from sklearn import metrics


def _to_class_indices(y):
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] > 1:
        return y.argmax(axis=1)
    return y.reshape(-1)


def _to_hard_predictions(y_pred, sigmoid=False):
    y_pred = np.asarray(y_pred)
    if sigmoid or (y_pred.ndim == 2 and not np.all(np.isin(y_pred, [0, 1]))):
        if y_pred.ndim == 1:
            return (y_pred > 0.5).astype(int)
        if y_pred.shape[1] == 1:
            return (y_pred.reshape(-1) > 0.5).astype(int)
        return (y_pred > 0.5).astype(int).argmax(axis=1) if y_pred.ndim == 2 else (y_pred > 0.5).astype(int)
    if y_pred.ndim == 2 and y_pred.shape[1] > 1:
        return y_pred.argmax(axis=1)
    return y_pred.reshape(-1).astype(int)


def _g_mean(y_true_idx, y_pred_idx):
    """Geometric mean of per-class recalls (sqrt(TPR * TNR) in the binary case)."""
    recalls = metrics.recall_score(y_true_idx, y_pred_idx, average=None, zero_division=0)
    recalls = np.asarray(recalls, dtype=float)
    if recalls.size == 0 or np.any(recalls <= 0):
        return 0.0
    return float(np.prod(recalls) ** (1.0 / recalls.size))


def _binary_positive_scores(y_prob):
    y_prob = np.asarray(y_prob)
    if y_prob.ndim == 2 and y_prob.shape[1] > 1:
        return y_prob[:, 1]
    return y_prob.reshape(-1)


def compute_classification_metrics(y_true, y_pred, y_prob=None, sigmoid=False):
    """Metrics suited to imbalanced fraud detection."""
    y_true_idx = _to_class_indices(y_true)
    y_pred_idx = _to_hard_predictions(y_pred, sigmoid=sigmoid)
    n_classes = len(np.unique(y_true_idx))

    out = {
        'accuracy': float(metrics.accuracy_score(y_true_idx, y_pred_idx)),
        'f1_micro': float(metrics.f1_score(y_true_idx, y_pred_idx, average='micro', zero_division=0)),
        'f1_macro': float(metrics.f1_score(y_true_idx, y_pred_idx, average='macro', zero_division=0)),
        'f1_weighted': float(metrics.f1_score(y_true_idx, y_pred_idx, average='weighted', zero_division=0)),
        'precision_macro': float(metrics.precision_score(y_true_idx, y_pred_idx, average='macro', zero_division=0)),
        'recall_macro': float(metrics.recall_score(y_true_idx, y_pred_idx, average='macro', zero_division=0)),
        'g_mean': _g_mean(y_true_idx, y_pred_idx),
    }

    scores = y_pred if y_prob is None else y_prob
    scores = np.asarray(scores)
    try:
        if n_classes == 2:
            pos = _binary_positive_scores(scores)
            auroc = float(metrics.roc_auc_score(y_true_idx, pos))
            auprc = float(metrics.average_precision_score(y_true_idx, pos))
            out['auroc'] = auroc
            out['auprc'] = auprc
            out['auc_roc'] = auroc
        elif scores.ndim == 2 and scores.shape[1] == n_classes:
            auroc = float(metrics.roc_auc_score(
                y_true_idx, scores, multi_class='ovr', average='macro'))
            y_true_oh = np.eye(n_classes)[y_true_idx]
            auprc = float(metrics.average_precision_score(y_true_oh, scores, average='macro'))
            out['auroc'] = auroc
            out['auprc'] = auprc
            out['auc_roc_macro'] = auroc
    except ValueError:
        pass

    return out


def count_model_parameters(model):
    """Return ``(total, trainable)`` parameter counts."""
    total = 0
    trainable = 0
    for param in model.parameters():
        n = int(param.numel())
        total += n
        if param.requires_grad:
            trainable += n
    return total, trainable


def run_output_dir(save_dir, model_name, dataset_name=None, seed=None, run_tag=None):
    parts = [save_dir, model_name]
    if dataset_name:
        parts.append(str(dataset_name))
    if seed is not None:
        seed_part = 'seed{}'.format(seed)
        if run_tag:
            seed_part = '{}_{}'.format(seed_part, run_tag)
        parts.append(seed_part)
    elif run_tag:
        parts.append(str(run_tag))
    return os.path.join(*parts)


def save_run_outputs(
    save_dir,
    model_name,
    config,
    split_info,
    test_metrics,
    model=None,
    train_metrics=None,
    predictions=None,
    probabilities=None,
    test_indices=None,
    train_indices=None,
    y_test=None,
    y_train=None,
    train_time=None,
):
    """
    Unified artifact layout under save_dir:

    - config.json
    - metrics.json
    - split.npz
    - predictions.npz
    - checkpoint.pt
    """
    os.makedirs(save_dir, exist_ok=True)

    complexity = {}
    if train_time is not None:
        complexity['train_time_sec'] = float(train_time)
    if model is not None and hasattr(model, 'parameters'):
        n_all, n_train = count_model_parameters(model)
        complexity['num_parameters'] = n_all
        complexity['num_trainable_parameters'] = n_train
    if complexity:
        config = dict(config)
        config.update(complexity)
        print('complexity:', complexity)

    config_path = os.path.join(save_dir, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, default=str)

    metrics_payload = {
        'model': model_name,
        'split': split_info,
        'test': test_metrics,
    }
    if train_metrics is not None:
        metrics_payload['train'] = train_metrics
    if complexity:
        metrics_payload['complexity'] = complexity

    metrics_path = os.path.join(save_dir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_payload, f, indent=2)

    split_arrays = {}
    if train_indices is not None:
        split_arrays['train_idx'] = np.asarray(train_indices)
    if test_indices is not None:
        split_arrays['test_idx'] = np.asarray(test_indices)
    if y_train is not None:
        split_arrays['y_train'] = np.asarray(y_train)
    if y_test is not None:
        split_arrays['y_test'] = np.asarray(y_test)
    if split_arrays:
        np.savez_compressed(os.path.join(save_dir, 'split.npz'), **split_arrays)

    if predictions is not None or probabilities is not None:
        np.savez_compressed(
            os.path.join(save_dir, 'predictions.npz'),
            predictions=np.asarray(predictions) if predictions is not None else np.array([]),
            probabilities=np.asarray(probabilities) if probabilities is not None else np.array([]),
            test_idx=np.asarray(test_indices) if test_indices is not None else np.array([]),
        )

    checkpoint_path = os.path.join(save_dir, 'checkpoint.pt')
    if model is not None:
        payload = {
            'model_name': model_name,
            'config': config,
            'state_dict': model.state_dict(),
        }
        torch.save(payload, checkpoint_path)

    print('Run saved to: {}'.format(os.path.abspath(save_dir)))
    for name in ['config.json', 'metrics.json', 'split.npz', 'predictions.npz', 'checkpoint.pt']:
        path = os.path.join(save_dir, name)
        if os.path.exists(path):
            print('  - {}'.format(name))
    return save_dir


def load_checkpoint(checkpoint_path, map_location='cpu'):
    return torch.load(checkpoint_path, map_location=map_location)


def load_run_metrics(save_dir):
    with open(os.path.join(save_dir, 'metrics.json'), 'r', encoding='utf-8') as f:
        return json.load(f)


def load_run_config(save_dir):
    with open(os.path.join(save_dir, 'config.json'), 'r', encoding='utf-8') as f:
        return json.load(f)
