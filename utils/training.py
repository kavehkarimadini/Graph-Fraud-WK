"""Shared train/eval finalization and artifact saving for all models."""
import argparse

from utils.evaluation import compute_classification_metrics, run_output_dir, save_run_outputs


def add_save_args(parser):
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Root directory for checkpoints and evaluation outputs.')
    parser.add_argument('--test_size', type=float, default=None,
                        help='Held-out test fraction (unused by GraphConsis/BWGNN/SplitGNN/DGA-GNN/GLHN).')
    parser.add_argument('--no_save', action='store_true',
                        help='Skip writing unified evaluation artifacts.')


def add_protocol_split_args(parser, train_ratio=0.4, early_stop=0):
    parser.add_argument('--train_ratio', type=float, default=train_ratio,
                        help='Train fraction of all labeled nodes (max 0.4). '
                             'Valid is fixed at 20%% and test at 40%%; unused train-pool nodes are discarded.')
    parser.add_argument('--early_stop', type=int, default=early_stop,
                        help='Patience on valid AUC (0 trains all epochs; best valid checkpoint is still kept).')


def default_test_size(model_name, dataset_str):
    ds = (dataset_str or '').lower()
    if ds in ('yelp', 'yelpchi'):
        return 0.2
    if ds in ('tfinance', 'tsocial', 'amazon', 'elliptic', 'elliptic_bitcoin',
              'comp', 'comp.dgl', 'fdcompcn'):
        return 0.4
    if ds == 'dblp':
        return 0.4
    if model_name in ('SemiGNN', 'GEM', 'GraphSage', 'GraphConsis'):
        return 0.2
    if model_name == 'HACUD':
        return 0.4
    return 0.4


def resolve_test_size(args, model_name):
    if getattr(args, 'test_size', None) is not None:
        return args.test_size
    dataset_str = getattr(args, 'dataset_str', getattr(args, 'dataset', 'example'))
    return default_test_size(model_name, dataset_str)


def args_to_config(args):
    if isinstance(args, argparse.Namespace):
        return {k: v for k, v in vars(args).items()}
    return dict(args)


def finalize_run(
    args,
    model_name,
    net,
    split_info,
    train_data,
    train_label,
    test_data,
    test_label,
    train_eval_fn,
    test_eval_fn,
    sigmoid=False,
    extra_config=None,
    train_time=None,
):
    """
    Evaluate train/test splits and persist unified artifacts.

    train_eval_fn / test_eval_fn must return (acc, pred, prob, tags).
    """
    _, train_pred, train_prob, _ = train_eval_fn()
    test_acc, test_pred, test_prob, _ = test_eval_fn()

    train_metrics = compute_classification_metrics(train_label, train_pred, train_prob, sigmoid=sigmoid)
    test_metrics = compute_classification_metrics(test_label, test_pred, test_prob, sigmoid=sigmoid)
    test_metrics['accuracy'] = float(test_acc) if test_acc is not None else test_metrics['accuracy']

    print('Train metrics:', train_metrics)
    print('Test metrics:', test_metrics)

    if getattr(args, 'no_save', False):
        return test_metrics

    config = args_to_config(args)
    if extra_config:
        config.update(extra_config)

    if train_time is None:
        train_time = config.get('train_time_sec', getattr(args, 'train_time_sec', None))

    dataset_name = config.get('dataset_str', config.get('dataset', 'unknown'))
    seed = config.get('seed', 123)
    out_dir = run_output_dir(args.save_dir, model_name, dataset_name, seed)

    save_run_outputs(
        save_dir=out_dir,
        model_name=model_name,
        config=config,
        split_info=split_info,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        model=net,
        predictions=test_pred,
        probabilities=test_prob,
        train_indices=train_data,
        test_indices=test_data,
        y_train=train_label,
        y_test=test_label,
        train_time=train_time,
    )
    return test_metrics
