#!/usr/bin/env python
"""Inspect or verify unified evaluation artifacts from a training run."""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.evaluation import compute_classification_metrics, load_run_config, load_run_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Inspect unified DGFraud run outputs.')
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Path to a run directory, e.g. results/GAS/example/seed123')
    parser.add_argument('--verify', action='store_true',
                        help='Recompute test metrics from predictions.npz and split.npz')
    return parser.parse_args()


def print_run_summary(run_dir):
    metrics_path = os.path.join(run_dir, 'metrics.json')
    config_path = os.path.join(run_dir, 'config.json')
    if not os.path.isdir(run_dir):
        raise FileNotFoundError('Run directory not found: {}'.format(run_dir))
    if not os.path.exists(metrics_path):
        raise FileNotFoundError('Missing metrics.json in {}'.format(run_dir))

    metrics = load_run_metrics(run_dir)
    config = load_run_config(run_dir) if os.path.exists(config_path) else {}

    print('Run directory:', os.path.abspath(run_dir))
    print('Model:', metrics.get('model', config.get('model', 'unknown')))
    if config:
        print('Config keys:', ', '.join(sorted(config.keys())))
    print('\nSplit info:')
    print(json.dumps(metrics.get('split', {}), indent=2))
    print('\nTrain metrics:')
    print(json.dumps(metrics.get('train', {}), indent=2))
    print('\nTest metrics:')
    print(json.dumps(metrics.get('test', {}), indent=2))

    for name in ['config.json', 'metrics.json', 'split.npz', 'predictions.npz', 'checkpoint.pt']:
        path = os.path.join(run_dir, name)
        status = 'present' if os.path.exists(path) else 'missing'
        print('  {}: {}'.format(name, status))


def verify_predictions(run_dir, metrics):
    pred_path = os.path.join(run_dir, 'predictions.npz')
    split_path = os.path.join(run_dir, 'split.npz')
    if not os.path.exists(pred_path) or not os.path.exists(split_path):
        print('Cannot verify: predictions.npz or split.npz missing.')
        return

    preds = np.load(pred_path)
    split = np.load(split_path)
    y_test = split['y_test']
    y_pred = preds['predictions']
    y_prob = preds['probabilities'] if preds['probabilities'].size else y_pred

    sigmoid = metrics.get('model') not in ('HACUD', 'GraphSage', 'GraphConsis')
    if metrics.get('model') in ('GraphSage', 'GraphConsis'):
        sigmoid = bool(load_run_config(run_dir).get('sigmoid', False))

    recomputed = compute_classification_metrics(y_test, y_pred, y_prob, sigmoid=sigmoid)
    print('\nRecomputed test metrics from saved predictions:')
    print(json.dumps(recomputed, indent=2))

    saved = metrics.get('test', {})
    for key in ['accuracy', 'f1_micro', 'f1_macro', 'f1_weighted']:
        if key in saved and key in recomputed:
            delta = abs(saved[key] - recomputed[key])
            print('  {} saved={:.6f} recomputed={:.6f} delta={:.6f}'.format(
                key, saved[key], recomputed[key], delta))


def main():
    args = parse_args()
    metrics = load_run_metrics(args.run_dir)
    print_run_summary(args.run_dir)
    if args.verify:
        verify_predictions(args.run_dir, metrics)


if __name__ == '__main__':
    main()
