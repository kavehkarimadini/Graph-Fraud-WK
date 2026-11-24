"""Hyperparameter policy: paper settings where the paper evaluated the dataset, tuned otherwise.

For every (model, dataset) pair of the benchmark:

* If the model's original paper (``paper/papers-to-read``) reports results on that
  dataset, the model runs with the paper's hyperparameters (the per-dataset
  ``DATASET_DEFAULTS`` / argparse defaults in each ``*_main.py``). Values the paper
  does not state fall back to the authors' official code.
* Otherwise the hyperparameters are selected by ``tune_hparams.py``: a
  validation-only random search (train_ratio 0.4, the five benchmark seeds, mean
  validation AUPRC; the test split is never used for selection). The selected
  values are stored in ``configs/tuned/<Model>/<dataset>.json`` and picked up
  automatically by the CLI entry points and ``utils.notebook_runner``.

Explicit command-line flags always take precedence over tuned values.

Every model uses the Adam optimizer. Its learning rate and weight decay follow the
same rule as everything else: paper values on paper datasets, tuned otherwise
(both are part of every search space in ``utils/hparam_spaces.py``).

PMP and SEC-GFD (``algorithms/PMP/PMP_main.py``, ``algorithms/SEC_GFD/SEC_GFD_main.py``)
follow the same policy as every other model.
"""
from __future__ import print_function

import json
import os
import sys
import warnings

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TUNED_DIR = os.path.join(REPO_ROOT, 'configs', 'tuned')

DATASETS = ('yelpchi', 'amazon', 'tfinance', 'tsocial', 'elliptic', 'comp')

_DATASET_ALIASES = {
    'yelp': 'yelpchi', 'yelpchi': 'yelpchi',
    'amazon': 'amazon',
    'tfinance': 'tfinance', 't-finance': 'tfinance', 't_finance': 'tfinance',
    'tsocial': 'tsocial', 't-social': 'tsocial', 't_social': 'tsocial',
    'elliptic': 'elliptic', 'elliptic_bitcoin': 'elliptic', 'elliptic_bitcoin_dataset': 'elliptic',
    'comp': 'comp', 'fdcompcn': 'comp',
}

# Datasets on which each model's original paper (in paper/papers-to-read) reports
# results, with the section that gives the hyperparameters. Every other benchmark
# dataset is tuned. Keys are the algorithms/<folder> names.
PAPER_EVALUATED = {
    'AMNet': dict(
        datasets={'yelpchi', 'elliptic'},
        source='AMNet (IJCAI 2022) Sec. 4.1: Yelp (R-U-R) and Elliptic; K = 2 filters, '
               'remaining values from the official code.'),
    'BSL': dict(
        datasets={'yelpchi', 'amazon', 'elliptic', 'tfinance', 'tsocial'},
        source='BSL (AAAI 2024) Experiment Settings: Adam lr 0.01, weight decay 1e-5, '
               'embedding 96, batch 256, alpha 0.4, beta 0.8.'),
    'BWGNN': dict(
        datasets={'yelpchi', 'amazon', 'tfinance', 'tsocial'},
        source='BWGNN (ICML 2022) Sec. 4.1: 100 epochs, Adam lr 0.01, h = 64, C = 2 '
               '(T-Social: h = 64, C = 5).'),
    'CAREGNN': dict(
        datasets={'yelpchi', 'amazon'},
        source='CARE-GNN (CIKM 2020) Sec. 4.1.4: embedding 64, batch 1024 / 256, 1 layer, '
               'lr 0.01, L2 1e-3, step 0.02, lambda_1 2.'),
    'CGNN': dict(
        datasets={'yelpchi', 'amazon', 'tfinance', 'elliptic'},
        source='CGNN (AAAI 2025) Implementation Details: lr 0.01, weight decay 1e-4, '
               'embedding 64, batch 16, lambda = beta = 1, tau 0.8 (T-Finance) / 0.01.'),
    'CIEGAD': dict(
        datasets={'yelpchi', 'amazon', 'tfinance', 'elliptic'},
        source='CIE-GAD (TCYB 2025) Sec. V-B: embedding 32, 1000 epochs, gamma 1, '
               'K = 12 (T-Finance) / 6.'),
    'COFRAUD': dict(
        datasets={'yelpchi', 'amazon'},
        source='COFRAUD (IJCAI 2023) Sec. 4.4-4.5: hidden 32, 2 layers, batch 1024; '
               'remaining values from the official train script.'),
    'DGAGNN': dict(
        datasets={'yelpchi', 'amazon', 'tfinance', 'tsocial', 'elliptic'},
        source='DGA-GNN (AAAI 2024) Hyperparameters: Adam lr 1e-3, weight decay 1e-3, '
               'up to 1000 iterations; k / z from the official per-dataset configs.'),
    'FRAUDRE': dict(
        datasets={'yelpchi', 'amazon'},
        source='FRAUDRE (ICDM 2021) Sec. V-A3: embedding 64, 2 layers, batch 256 / 1024, '
               'lr 0.1 (Amazon) / 0.001 (YelpChi).'),
    # Paper not in paper/papers-to-read: tuned on every dataset.
    'GDN': dict(datasets=set(), source='GDN paper not in paper/papers-to-read.'),
    'GGCN': dict(
        datasets=set(),
        source='GGCN (ICDM 2022) evaluates only Cora/Citeseer/Pubmed/WebKB/Chameleon/'
               'Squirrel/Actor.'),
    # Paper not in paper/papers-to-read: tuned on every dataset.
    'GHRN': dict(datasets=set(), source='GHRN paper not in paper/papers-to-read.'),
    'GLHN': dict(
        datasets={'yelpchi', 'amazon', 'tfinance', 'tsocial'},
        source='GLHAD (CIKM 2024) Sec. 4.1.4: hidden 32, 2 layers, tau 0.1, Adam lr 0.01, '
               'weight decay 1e-5; gamma1/gamma2 from the sensitivity study.'),
    'GraphConsis': dict(
        datasets={'yelpchi'},
        source='GraphConsis (SIGIR 2020) Sec. 4.1.3: 2 layers, 10 / 5 samples, hidden 200 / 100.'),
    'H2FDetector': dict(
        datasets={'yelpchi', 'amazon'},
        source='H2-FDetector (WWW 2022) Sec. 4.1.3: lr 0.1, weight decay 5e-5, embedding 8, '
               '4 heads, 2 layers, 1000 epochs, dropout 0.1; gamma1/gamma2 1.2/1.2 (YelpChi), '
               '0.4/1.4 (Amazon).'),
    'H2IDE': dict(
        datasets={'yelpchi', 'amazon', 'tfinance', 'tsocial'},
        source='H2IDE (TKDE) Sec. V-A4: lr 0.005, weight decay 5e-5, 1 layer, 500 epochs, '
               'factor dim 8, dropout 0.1; per-dataset alpha1/alpha2/batch/K.'),
    'ImGSL': dict(
        datasets={'yelpchi', 'amazon'},
        source='IGSL (Machine Learning 2024) Sec. 5.3: lr 0.01, weight decay 1e-3, 150 epochs, '
               'batch 256 / 1024, beta 0.5, epsilon 0.5 / 0, tau- 0.2, alpha 2.'),
    'PMP': dict(
        datasets={'yelpchi', 'amazon', 'tfinance', 'tsocial'},
        source='PMP (ICLR 2024) App. D.4 Table 5: Adam, 1 layer; lr / wd / dropout / batch / '
               'hidden = Yelp 0.01/0/0/512/256, Amazon 0.01/0/0.6/128/128, T-Finance '
               '0.01/0/0.4/256/64, T-Social 0.001/0/0/2^17/128.'),
    'SEC_GFD': dict(
        datasets={'yelpchi', 'amazon', 'tfinance', 'tsocial'},
        source='SEC-GFD (AAAI 2024) released code (reference/SEC-GFD-main.zip; the paper '
               'evaluates YelpChi, Amazon, T-Finance, T-Social): Adam lr 0.01, hidden 64, '
               'C = 2, high-order 2, lambda 0.2; T-Social hidden 16, C = 5.'),
    'PCGNN': dict(
        datasets={'yelpchi', 'amazon'},
        source='PC-GNN (WWW 2021) Sec. 4.1.3: Adam lr 0.01, 100 epochs, L = 1, d = 64, '
               'alpha 2, batch 1024; weight decay / threshold from the official configs.'),
    'SparseGAD': dict(
        datasets={'yelpchi', 'amazon'},
        source='SparseGAD (IJCAI 2023) Experiment Settings: 2000 epochs; per-dataset values '
               'from the official training scripts.'),
    'SplitGNN': dict(
        datasets={'yelpchi', 'amazon', 'comp'},
        source='SplitGNN (CIKM 2023) Sec. 4.1.3: lr 0.01 (YelpChi) / 0.1 (Amazon) / 0.001 '
               '(FDCompCN), weight decay 5e-5, embedding 8, 1000 epochs, dropout 0.1.'),
    'TAM': dict(
        datasets={'yelpchi', 'amazon', 'tfinance'},
        source='TAM (NeurIPS 2023) Sec. 4.1 / 4.3: 2-layer GCN, Adam lr 1e-5, 500 epochs, '
               'T = 3, lambda 0; K = 7 on Amazon-all / YelpChi-all / T-Finance.'),
}


def canonical_dataset(name):
    """Map any dataset spelling used in the repo (incl. file names) to a canonical key."""
    if name is None:
        return None
    key = os.path.basename(str(name)).strip().lower()
    for suffix in ('.mat', '.dgl'):
        if key.endswith(suffix):
            key = key[:-len(suffix)]
    return _DATASET_ALIASES.get(key, key)


def uses_paper_hparams(model, dataset):
    entry = PAPER_EVALUATED.get(model)
    if entry is None:
        raise KeyError('Unknown model {!r}; add it to utils.hparams.PAPER_EVALUATED.'.format(model))
    return canonical_dataset(dataset) in entry['datasets']


def needs_tuning(model, dataset):
    return not uses_paper_hparams(model, dataset)


def tuned_path(model, dataset):
    return os.path.join(TUNED_DIR, model, '{}.json'.format(canonical_dataset(dataset)))


def load_tuned(model, dataset):
    """Return the stored tuning record for (model, dataset), or ``None``."""
    path = tuned_path(model, dataset)
    if not os.path.isfile(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def tuned_params(model, dataset):
    """Hyperparameters to apply for (model, dataset) plus provenance keys.

    Paper datasets and untuned pairs return only the provenance keys, so callers
    keep their defaults. ``None`` values in a tuning record mean "use the model's
    own default" and are skipped.
    """
    ds = canonical_dataset(dataset)
    if uses_paper_hparams(model, ds):
        return {'hparam_source': 'paper'}
    record = load_tuned(model, ds)
    if record is None:
        warnings.warn(
            '{} on {} is not evaluated in its paper and has no tuned config yet ({}); '
            'falling back to the untuned defaults. Run: python tune_hparams.py --model {} '
            '--dataset {}'.format(model, ds, os.path.relpath(tuned_path(model, ds), REPO_ROOT),
                                  model, ds), stacklevel=2)
        return {'hparam_source': 'untuned_default'}
    params = {k: v for k, v in record['params'].items() if v is not None}
    params['hparam_source'] = 'tuned'
    params['hparam_file'] = os.path.relpath(tuned_path(model, ds), REPO_ROOT).replace(os.sep, '/')
    return params


def explicit_cli_dests(parser, argv=None):
    """``dest`` names of options the user actually typed on the command line."""
    argv = sys.argv[1:] if argv is None else argv
    dests = set()
    for action in parser._actions:  # pylint: disable=protected-access
        for opt in action.option_strings:
            if any(a == opt or a.startswith(opt + '=') for a in argv):
                dests.add(action.dest)
    return dests


def apply_tuned_hparams(args, model, parser=None, dataset=None, argv=None):
    """Overlay tuned values onto parsed CLI ``args`` (explicit flags win).

    Call right after ``parser.parse_args()`` and before the model's
    ``apply_dataset_defaults`` so tuned values are not overwritten.
    """
    explicit = explicit_cli_dests(parser, argv) if parser is not None else set()
    if dataset is None:
        dataset = getattr(args, 'dataset', None) or getattr(args, 'file_name', None)
    params = tuned_params(model, dataset)
    for key, value in params.items():
        if key in explicit:
            continue
        setattr(args, key, value)
    ds = canonical_dataset(dataset)
    print('Hyperparameters for {} on {}: {}{}'.format(
        model, ds, params['hparam_source'],
        ' ({})'.format(params['hparam_file']) if 'hparam_file' in params else ''))
    return args


def policy_table():
    """Rows of (model, dataset, 'paper' | 'tune', tuned-file-exists)."""
    rows = []
    for model in sorted(PAPER_EVALUATED):
        for ds in DATASETS:
            paper = uses_paper_hparams(model, ds)
            rows.append((model, ds, 'paper' if paper else 'tune',
                         (not paper) and os.path.isfile(tuned_path(model, ds))))
    return rows
