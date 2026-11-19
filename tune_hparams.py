"""Validation-only hyperparameter tuning for (model, dataset) pairs without paper settings.

Policy (see ``utils/hparams.py``): a model uses its paper's hyperparameters on the
datasets its paper evaluates; on every other benchmark dataset its
hyperparameters are chosen here and saved to ``configs/tuned/<Model>/<dataset>.json``,
which the ``*_main.py`` scripts and ``utils.notebook_runner`` then load automatically.

Selection protocol (same as the PMP / SEC-GFD tuning already in the repo):
  * train_ratio 0.4 (the 40/20/40 protocol split),
  * every trial runs on the five benchmark seeds,
  * trials are ranked by mean validation AUROC (ties: lower std),
  * test metrics are never read.
Trial 0 is always the current (untuned) default configuration, so the tuned
config is never worse than the default on validation.

Usage (repo root):
  python tune_hparams.py --list                                   # policy matrix + status
  python tune_hparams.py --model BWGNN --dataset elliptic --gpu 0
  python tune_hparams.py --model GGCN --dataset yelpchi,amazon --n_trials 30
  python tune_hparams.py --all --gpu 0                            # every pair that needs tuning
  python tune_hparams.py --all --skip_dataset tsocial --resume    # continue an interrupted sweep
"""
from __future__ import print_function

import argparse
import contextlib
import gc
import importlib
import itertools
import json
import math
import os
import random
import sys
import time
import traceback
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils import notebook_runner as nr  # noqa: E402
from utils.hparam_spaces import search_space  # noqa: E402
from utils.hparams import (  # noqa: E402
    DATASETS, PAPER_EVALUATED, REPO_ROOT, canonical_dataset, needs_tuning, policy_table,
    tuned_path,
)

BENCHMARK_SEEDS = [123, 1234, 12345, 123456, 1234567]
LOG_DIR = os.path.join(REPO_ROOT, 'tuning_logs')

# model -> (notebook_runner defaults function, module with ``train(args)``)
MODELS = {
    'AMNet': ('_amnet_defaults', 'algorithms.AMNet.AMNet_main'),
    'BSL': ('_bsl_defaults', 'algorithms.BSL.BSL_main'),
    'BWGNN': ('_bwgnn_defaults', 'algorithms.BWGNN.BWGNN_main'),
    'CAREGNN': ('_caregnn_defaults', 'algorithms.CAREGNN.CAREGNN_main'),
    'CGNN': ('_cgnn_defaults', 'algorithms.CGNN.CGNN_main'),
    'CIEGAD': ('_ciegad_defaults', 'algorithms.CIEGAD.CIEGAD_main'),
    'COFRAUD': ('_cofraud_defaults', 'algorithms.COFRAUD.COFRAUD_main'),
    'DGAGNN': ('_dgagnn_defaults', 'algorithms.DGAGNN.DGAGNN_main'),
    'FRAUDRE': ('_fraudre_defaults', 'algorithms.FRAUDRE.FRAUDRE_main'),
    'GDN': ('_gdn_defaults', 'algorithms.GDN.GDN_main'),
    'GGCN': ('_ggcn_defaults', 'algorithms.GGCN.GGCN_main'),
    'GHRN': ('_ghrn_defaults', 'algorithms.GHRN.GHRN_main'),
    'GLHN': ('_glhn_defaults', 'algorithms.GLHN.GLHN_main'),
    'GraphConsis': ('_graphconsis_defaults', 'algorithms.GraphConsis.supervised_train'),
    'H2FDetector': ('_h2fdetector_defaults', 'algorithms.H2FDetector.H2FDetector_main'),
    'H2IDE': ('_h2ide_defaults', 'algorithms.H2IDE.H2IDE_main'),
    'ImGSL': ('_imgsl_defaults', 'algorithms.ImGSL.ImGSL_main'),
    'PCGNN': ('_pcgnn_defaults', 'algorithms.PCGNN.PCGNN_main'),
    'SparseGAD': ('_sparsegad_defaults', 'algorithms.SparseGAD.SparseGAD_main'),
    'SplitGNN': ('_splitgnn_defaults', 'algorithms.SplitGNN.SplitGNN_main'),
    'TAM': ('_tam_defaults', 'algorithms.TAM.TAM_main'),
}
assert set(MODELS) == set(PAPER_EVALUATED), 'MODELS and PAPER_EVALUATED must list the same models'

DATASET_FILES = {
    'yelpchi': 'YelpChi.mat', 'amazon': 'Amazon.mat', 'tfinance': 'tfinance',
    'tsocial': 'tsocial', 'elliptic': 'elliptic_bitcoin_dataset', 'comp': 'comp.dgl',
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', type=str, default=None, help='Comma-separated model names (algorithms/<folder>).')
    p.add_argument('--dataset', type=str, default=None, help='Comma-separated datasets (default: all that need tuning).')
    p.add_argument('--all', action='store_true', help='Tune every (model, dataset) pair without paper settings.')
    p.add_argument('--list', action='store_true', help='Print the paper / tune matrix and exit.')
    p.add_argument('--skip_dataset', type=str, default='', help='Comma-separated datasets to skip.')
    p.add_argument('--n_trials', type=int, default=20, help='Trials per pair, including the default (trial 0).')
    p.add_argument('--seeds', type=str, default=','.join(str(s) for s in BENCHMARK_SEEDS))
    p.add_argument('--train_ratio', type=float, default=0.4)
    p.add_argument('--sampler_seed', type=int, default=0, help='Seed of the random search itself.')
    p.add_argument('--max_epochs', type=int, default=None,
                   help='Optional cap on epochs per trial (budget); recorded in the output.')
    p.add_argument('--prune_margin', type=float, default=None,
                   help='Skip a trial\'s remaining seeds when its first-seed valid AUROC is this far '
                        'below the best mean so far (off by default).')
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--data_dir', type=str, default=None, help='Dataset folder (default: <repo>/dataset).')
    p.add_argument('--resume', action='store_true', help='Continue from <dataset>.partial.json.')
    p.add_argument('--force', action='store_true', help='Re-tune pairs that already have a tuned config.')
    p.add_argument('--verbose', action='store_true', help='Show model training output instead of logging it.')
    return p.parse_args()


def print_policy():
    print('{:<12} {}'.format('model', '  '.join('{:<9}'.format(d) for d in DATASETS)))
    by_model = {}
    for model, ds, kind, done in policy_table():
        by_model.setdefault(model, []).append('paper' if kind == 'paper' else ('tuned' if done else 'TUNE'))
    for model in sorted(by_model):
        print('{:<12} {}'.format(model, '  '.join('{:<9}'.format(c) for c in by_model[model])))
    print('\npaper = paper hyperparameters, tuned = configs/tuned/... present, TUNE = still to tune.')


def dataset_available(ds, data_dir):
    return os.path.exists(os.path.join(data_dir, DATASET_FILES[ds]))


def base_params(model, ds):
    defaults_fn = getattr(nr, MODELS[model][0])
    params = defaults_fn(ds, ROOT)
    return params


def anchor_trial(space, defaults):
    return {k: defaults.get(k) for k in space}


def sample_trials(space, n_trials, rng, anchor):
    keys = sorted(space)
    total = 1
    for k in keys:
        total *= len(space[k])
    trials = [anchor]
    seen = {json.dumps(anchor, sort_keys=True)}
    if total <= n_trials - 1:
        combos = [dict(zip(keys, vals)) for vals in itertools.product(*(space[k] for k in keys))]
        rng.shuffle(combos)
    else:
        combos = None
    attempts = 0
    while len(trials) < n_trials and attempts < 100 * n_trials:
        attempts += 1
        if combos is not None:
            if not combos:
                break
            cand = combos.pop()
        else:
            cand = {k: rng.choice(space[k]) for k in keys}
        sig = json.dumps(cand, sort_keys=True)
        if sig in seen:
            continue
        seen.add(sig)
        trials.append(cand)
    return trials


def _mean_std(values):
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    s = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
    return m, s


def run_one(model, train_fn, defaults, trial_params, seed, args, log_file):
    params = dict(defaults)
    params.update({k: v for k, v in trial_params.items()})
    params.update(seed=seed, train_ratio=args.train_ratio, gpu=args.gpu, no_save=True,
                  run_tag='tune', hparam_source='tuning')
    if args.max_epochs is not None and params.get('epochs') is not None:
        params['epochs'] = min(int(params['epochs']), args.max_epochs)
    ns = argparse.Namespace(**params)
    if args.verbose:
        out = train_fn(ns)
    else:
        with contextlib.redirect_stdout(log_file):
            out = train_fn(ns)
    metrics = out[0] if isinstance(out, tuple) else out
    auc = metrics.get('valid_auc') if isinstance(metrics, dict) else None
    return None if auc is None else float(auc)


def _cleanup():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover
        pass


def tune_pair(model, ds, args, seeds):
    out_path = tuned_path(model, ds)
    partial_path = out_path.replace('.json', '.partial.json')
    if os.path.isfile(out_path) and not args.force:
        print('[skip] {} / {}: already tuned ({})'.format(model, ds, os.path.relpath(out_path, ROOT)))
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.makedirs(os.path.join(LOG_DIR, model), exist_ok=True)

    defaults = base_params(model, ds)
    if args.data_dir:
        key = 'train_prefix' if model == 'GraphConsis' else 'data_prefix'
        defaults[key] = os.path.join(args.data_dir, '')
    space = search_space(model, ds)
    missing = [k for k in space if k not in defaults]
    if missing:
        raise KeyError('{}: search keys {} are not in the runner defaults'.format(model, missing))

    rng = random.Random('{}-{}-{}'.format(model, ds, args.sampler_seed))
    trials = sample_trials(space, args.n_trials, rng, anchor_trial(space, defaults))

    done = []
    if args.resume and os.path.isfile(partial_path):
        with open(partial_path, 'r', encoding='utf-8') as f:
            done = json.load(f).get('trials', [])
        print('Resuming {} / {} from trial {}'.format(model, ds, len(done)))

    train_fn = importlib.import_module(MODELS[model][1]).train
    log_path = os.path.join(LOG_DIR, model, '{}.log'.format(ds))
    print('=== Tuning {} on {}: {} trials x {} seeds (log: {})'.format(
        model, ds, len(trials), len(seeds), os.path.relpath(log_path, ROOT)))

    best = None
    for rec in done:
        if rec.get('mean') is not None and (best is None or (rec['mean'], -rec['std']) > (best['mean'], -best['std'])):
            best = rec

    with open(log_path, 'a', encoding='utf-8') as log_file:
        for i, trial in enumerate(trials):
            if i < len(done):
                continue
            t0 = time.perf_counter()
            aucs, error = [], None
            for j, seed in enumerate(seeds):
                print('\n##### trial {} seed {} params {}'.format(i, seed, trial), file=log_file, flush=True)
                try:
                    aucs.append(run_one(model, train_fn, defaults, trial, seed, args, log_file))
                except Exception as exc:  # OOM, invalid combination, ...
                    error = '{}: {}'.format(type(exc).__name__, exc)
                    traceback.print_exc(file=log_file)
                    _cleanup()
                    break
                finally:
                    log_file.flush()
                    _cleanup()
                if (j == 0 and args.prune_margin is not None and best is not None
                        and aucs[0] is not None and aucs[0] < best['mean'] - args.prune_margin):
                    error = 'pruned after first seed'
                    break
            mean, std = _mean_std(aucs)
            if error is not None:
                mean = std = None
            rec = dict(trial=i, params=trial, seeds=seeds[:len(aucs)], valid_auroc=aucs,
                       mean=mean, std=std, time_sec=round(time.perf_counter() - t0, 1), error=error)
            done.append(rec)
            if mean is not None and (best is None or (mean, -std) > (best['mean'], -best['std'])):
                best = rec
            print('  trial {:>2}/{} valid AUROC {} | best {:.4f} (trial {}) | {:.0f}s{}'.format(
                i, len(trials) - 1, 'n/a' if mean is None else '{:.4f} +/- {:.4f}'.format(mean, std),
                best['mean'] if best else float('nan'), best['trial'] if best else -1,
                rec['time_sec'], '' if error is None else ' [{}]'.format(error)))
            with open(partial_path, 'w', encoding='utf-8') as f:
                json.dump(dict(model=model, dataset=ds, trials=done), f, indent=2)

    if best is None:
        print('!!! {} / {}: every trial failed; nothing saved (see {})'.format(model, ds, log_path))
        return
    record = dict(
        model=model, dataset=ds,
        policy='Not evaluated in the original paper (paper/papers-to-read); hyperparameters '
               'selected by validation-only random search.',
        selection='mean validation AUROC over seeds; test split not used',
        params=best['params'],
        mean_valid_auroc=best['mean'], std_valid_auroc=best['std'],
        best_trial=best['trial'],
        default_trial_mean_valid_auroc=done[0]['mean'] if done else None,
        seeds=seeds, train_ratio=args.train_ratio, n_trials=len(trials),
        sampler_seed=args.sampler_seed, max_epochs=args.max_epochs, prune_margin=args.prune_margin,
        search_space=space, created=datetime.now().isoformat(timespec='seconds'),
        trials=done,
    )
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=2)
    if os.path.isfile(partial_path):
        os.remove(partial_path)
    print('>>> {} / {}: best trial {} valid AUROC {:.4f} +/- {:.4f} (default {}) -> {}'.format(
        model, ds, best['trial'], best['mean'], best['std'],
        'n/a' if record['default_trial_mean_valid_auroc'] is None
        else '{:.4f}'.format(record['default_trial_mean_valid_auroc']),
        os.path.relpath(out_path, ROOT)))


def main():
    args = parse_args()
    if args.list:
        print_policy()
        return
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    data_dir = args.data_dir or os.path.join(ROOT, 'dataset')

    if args.model:
        models = [m.strip() for m in args.model.split(',') if m.strip()]
    elif args.all:
        models = sorted(MODELS)
    else:
        raise SystemExit('Pass --model, --all or --list.')
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        raise SystemExit('Unknown model(s) {}. Choose from: {}'.format(unknown, ', '.join(sorted(MODELS))))
    datasets = ([canonical_dataset(d) for d in args.dataset.split(',') if d.strip()]
                if args.dataset else list(DATASETS))
    skip = {canonical_dataset(d) for d in args.skip_dataset.split(',') if d.strip()}

    for model in models:
        for ds in datasets:
            if ds in skip:
                continue
            if not needs_tuning(model, ds):
                if args.dataset:
                    print('[paper] {} / {}: evaluated in the paper; using paper hyperparameters, '
                          'not tuning.'.format(model, ds))
                continue
            if not dataset_available(ds, data_dir):
                print('[missing data] {} / {}: {} not found in {}'.format(
                    model, ds, DATASET_FILES[ds], data_dir))
                continue
            tune_pair(model, ds, args, seeds)


if __name__ == '__main__':
    main()
