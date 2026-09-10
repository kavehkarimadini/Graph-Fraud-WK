import os
import numpy as np
from sklearn.model_selection import train_test_split
from dgl.data import FraudDataset

SEEDS = [123, 1234, 12345, 123456, 1234567]

# Load Amazon graph only for graph structure/labels.
# We ignore DGL's built-in train/val/test masks.
ds = FraudDataset(
    name="amazon",
    raw_dir="dataset",
    random_seed=717,
    train_size=0.4,
    val_size=0.2,
    force_reload=False,
)

g = ds[0]
labels = g.ndata["label"].cpu().numpy()

# Amazon nodes 0-3304 are unlabeled in this dataset.
eligible = np.arange(3305, g.num_nodes())

os.makedirs("splits/amazon", exist_ok=True)

for seed in SEEDS:
    # 40% train pool + 60% remainder, stratified
    train40, remainder = train_test_split(
        eligible,
        train_size=0.4,
        random_state=seed,
        shuffle=True,
        stratify=labels[eligible],
    )

    # Remainder = 20% validation + 40% test
    val, test = train_test_split(
        remainder,
        train_size=1/3,
        random_state=seed + 1,
        shuffle=True,
        stratify=labels[remainder],
    )

    # Nested training subsets:
    # Train10 ⊂ Train20 ⊂ Train30 ⊂ Train40
    train30, _ = train_test_split(
        train40,
        train_size=0.75,
        random_state=seed + 2,
        shuffle=True,
        stratify=labels[train40],
    )

    train20, _ = train_test_split(
        train30,
        train_size=2/3,
        random_state=seed + 3,
        shuffle=True,
        stratify=labels[train30],
    )

    train10, _ = train_test_split(
        train20,
        train_size=0.5,
        random_state=seed + 4,
        shuffle=True,
        stratify=labels[train20],
    )

    # Sort only for stable/reproducible storage.
    train40 = np.sort(train40)
    train30 = np.sort(train30)
    train20 = np.sort(train20)
    train10 = np.sort(train10)
    val = np.sort(val)
    test = np.sort(test)

    # Protocol checks
    assert set(train10).issubset(set(train20))
    assert set(train20).issubset(set(train30))
    assert set(train30).issubset(set(train40))

    assert set(train40).isdisjoint(val)
    assert set(train40).isdisjoint(test)
    assert set(val).isdisjoint(test)

    assert set(train40) | set(val) | set(test) == set(eligible)

    discarded30 = np.setdiff1d(train40, train30)
    discarded20 = np.setdiff1d(train40, train20)
    discarded10 = np.setdiff1d(train40, train10)

    out = f"splits/amazon/seed_{seed}.npz"

    np.savez_compressed(
        out,
        seed=seed,
        train_40=train40,
        train_30=train30,
        train_20=train20,
        train_10=train10,
        val=val,
        test=test,
        discarded_30=discarded30,
        discarded_20=discarded20,
        discarded_10=discarded10,
        eligible=eligible,
    )

    print(f"\nSeed {seed}")
    for name, idx in [
        ("Train40", train40),
        ("Train30", train30),
        ("Train20", train20),
        ("Train10", train10),
        ("Valid20", val),
        ("Test40", test),
    ]:
        counts = np.bincount(labels[idx], minlength=2)
        print(
            f"{name:8s}: {len(idx):4d} nodes | "
            f"benign={counts[0]:4d} fraud={counts[1]:4d}"
        )

    print("Nestedness: PASS")
    print("Disjoint train/val/test: PASS")
    print("Saved:", out)
