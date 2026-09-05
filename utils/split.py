"""Stratified train/test splitting for imbalanced fraud labels."""
import numpy as np
from sklearn.model_selection import train_test_split


def labels_to_stratify(labels):
    """Convert one-hot or multi-column labels to a 1D stratification vector."""
    y = np.asarray(labels)
    if y.ndim == 1:
        return y.reshape(-1)
    if y.shape[1] == 1:
        return y.reshape(-1)
    # one-hot / multi-label → primary class index
    return y.argmax(axis=1)


def class_distribution(labels):
    strat = labels_to_stratify(labels)
    unique, counts = np.unique(strat, return_counts=True)
    return {int(k): int(v) for k, v in zip(unique, counts)}


def stratified_train_test_split(labels, test_size=0.2, seed=48, indices=None):
    """
    Stratified split preserving class balance (important for fraud detection).

    Returns train_idx, test_idx as numpy int arrays.
    """
    if indices is None:
        indices = np.arange(len(labels))
    else:
        indices = np.asarray(indices)

    strat = labels_to_stratify(labels)
    # When ``indices`` is a subset of a full label vector, stratify on that subset.
    stratify = strat if strat.shape[0] == indices.shape[0] else strat[indices]
    try:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
    except ValueError as exc:
        # Tiny graphs may not support stratification for every class.
        print('Warning: stratified split failed ({}). Falling back to random split.'.format(exc))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
        )
    return np.asarray(train_idx, dtype=np.int64), np.asarray(test_idx, dtype=np.int64)


def split_report(labels, train_idx, test_idx, valid_idx=None, discarded_idx=None):
    """Human-readable class counts for train / valid / test."""
    strat = labels_to_stratify(labels)
    out = {
        'train_size': int(len(train_idx)),
        'test_size': int(len(test_idx)),
        'train_class_counts': class_distribution(strat[train_idx]),
        'test_class_counts': class_distribution(strat[test_idx]),
    }
    if valid_idx is not None:
        out['valid_size'] = int(len(valid_idx))
        out['valid_class_counts'] = class_distribution(strat[valid_idx])
    if discarded_idx is not None:
        out['discarded_size'] = int(len(discarded_idx))
        if len(discarded_idx):
            out['discarded_class_counts'] = class_distribution(strat[discarded_idx])
    return out


# GraphConsis / BWGNN / SplitGNN: fixed hold-outs; only train_ratio is tunable.
VALID_RATIO = 0.2
TEST_RATIO = 0.4
TRAIN_POOL_RATIO = 1.0 - VALID_RATIO - TEST_RATIO  # 0.4


def protocol_split(labels, train_ratio=0.4, valid_ratio=VALID_RATIO, test_ratio=TEST_RATIO,
                   seed=48, indices=None):
    """Stratified  train-pool / valid / test  split with a tunable train slice.

    ``valid_ratio`` and ``test_ratio`` are fractions of the index set (default
    20% / 40%) and stay fixed. The remaining 40% is the train pool. If
    ``train_ratio`` is smaller than that pool, a stratified subset is kept and
    the rest of the pool is discarded (not added to valid or test).
    ``train_ratio`` larger than the pool is clamped.
    """
    strat = labels_to_stratify(labels)
    if indices is None:
        indices = np.arange(strat.size)
    indices = np.asarray(indices, dtype=np.int64)
    n = float(indices.size)
    if n < 3:
        raise ValueError('Need at least 3 nodes to form train/valid/test')

    rest_idx, test_idx = stratified_train_test_split(
        strat, test_size=test_ratio, seed=seed, indices=indices)
    valid_of_rest = valid_ratio / max(1.0 - test_ratio, 1e-8)
    train_pool, valid_idx = stratified_train_test_split(
        strat, test_size=valid_of_rest, seed=seed, indices=rest_idx)

    max_train = 1.0 - valid_ratio - test_ratio
    used_ratio = float(train_ratio)
    if used_ratio > max_train + 1e-9:
        print('Warning: train_ratio={:.3f} exceeds the {:.0f}% train pool; clamping.'.format(
            used_ratio, 100 * max_train))
        used_ratio = max_train
    if used_ratio <= 0:
        raise ValueError('train_ratio must be positive')

    discarded_idx = np.asarray([], dtype=np.int64)
    if used_ratio >= max_train - 1e-9:
        train_idx = train_pool
    else:
        keep_frac = used_ratio / max_train
        train_idx, discarded_idx = stratified_train_test_split(
            strat, test_size=1.0 - keep_frac, seed=seed, indices=train_pool)

    info = split_report(strat, train_idx, test_idx, valid_idx=valid_idx, discarded_idx=discarded_idx)
    info.update({
        'train_ratio': float(used_ratio),
        'valid_ratio': float(valid_ratio),
        'test_ratio': float(test_ratio),
        'train_pool_ratio': float(max_train),
        'seed': int(seed),
        'stratified': True,
        'protocol': 'fixed_valid_test',
    })
    print('Split protocol: train={}/{} ({:.1%}) valid={}/{} ({:.1%}) test={}/{} ({:.1%}) discarded={}'.format(
        len(train_idx), int(n), len(train_idx) / n,
        len(valid_idx), int(n), len(valid_idx) / n,
        len(test_idx), int(n), len(test_idx) / n,
        len(discarded_idx)))
    return train_idx, valid_idx, test_idx, discarded_idx, info
