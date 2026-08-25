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
    try:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=strat,
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


def split_report(labels, train_idx, test_idx):
    """Human-readable class counts for train / test."""
    strat = labels_to_stratify(labels)
    train_counts = class_distribution(strat[train_idx])
    test_counts = class_distribution(strat[test_idx])
    return {
        'train_size': int(len(train_idx)),
        'test_size': int(len(test_idx)),
        'train_class_counts': train_counts,
        'test_class_counts': test_counts,
    }
