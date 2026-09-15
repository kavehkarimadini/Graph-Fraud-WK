"""Generate canonical nested 10/20/30/40% train splits for PMP benchmark datasets."""
import argparse
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from algorithms.PMP.pmp_config import canonical_dataset_name
from algorithms.PMP.pmp_data import load_pmp_dataset

SEEDS = [123, 1234, 12345, 123456, 1234567]


def generate(dataset):
    dataset = canonical_dataset_name(dataset)
    graph, _, labels_t, _, eligible = load_pmp_dataset(dataset, ROOT)
    labels = labels_t.cpu().numpy()

    out_dir = os.path.join(ROOT, "splits", dataset)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Dataset: {dataset}")
    print(f"Nodes: {graph.num_nodes()} | Eligible labelled nodes: {len(eligible)}")

    for seed in SEEDS:
        train40, remainder = train_test_split(
            eligible, train_size=0.4, random_state=seed, shuffle=True,
            stratify=labels[eligible],
        )
        val, test = train_test_split(
            remainder, train_size=1/3, random_state=seed + 1, shuffle=True,
            stratify=labels[remainder],
        )
        train30, _ = train_test_split(
            train40, train_size=0.75, random_state=seed + 2, shuffle=True,
            stratify=labels[train40],
        )
        train20, _ = train_test_split(
            train30, train_size=2/3, random_state=seed + 3, shuffle=True,
            stratify=labels[train30],
        )
        train10, _ = train_test_split(
            train20, train_size=0.5, random_state=seed + 4, shuffle=True,
            stratify=labels[train20],
        )

        arrays = {k: np.sort(v.astype(np.int64)) for k, v in {
            "train_40": train40, "train_30": train30, "train_20": train20,
            "train_10": train10, "val": val, "test": test,
        }.items()}

        assert set(arrays["train_10"]).issubset(set(arrays["train_20"]))
        assert set(arrays["train_20"]).issubset(set(arrays["train_30"]))
        assert set(arrays["train_30"]).issubset(set(arrays["train_40"]))
        assert set(arrays["train_40"]).isdisjoint(arrays["val"])
        assert set(arrays["train_40"]).isdisjoint(arrays["test"])
        assert set(arrays["val"]).isdisjoint(arrays["test"])
        assert set(arrays["train_40"]) | set(arrays["val"]) | set(arrays["test"]) == set(eligible)

        out = os.path.join(out_dir, f"seed_{seed}.npz")
        np.savez_compressed(
            out,
            seed=seed,
            **arrays,
            discarded_30=np.setdiff1d(arrays["train_40"], arrays["train_30"]),
            discarded_20=np.setdiff1d(arrays["train_40"], arrays["train_20"]),
            discarded_10=np.setdiff1d(arrays["train_40"], arrays["train_10"]),
            eligible=np.sort(eligible),
        )

        print(f"\nSeed {seed}")
        for name in ["train_40", "train_30", "train_20", "train_10", "val", "test"]:
            idx = arrays[name]
            counts = np.bincount(labels[idx], minlength=2)
            print(f"{name:8s}: {len(idx):7d} | benign={counts[0]:7d} fraud={counts[1]:7d}")
        print("Nestedness/disjointness/coverage: PASS")
        print("Saved:", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["amazon", "yelp", "yelpchi", "tfinance", "elliptic"])
    args = parser.parse_args()
    generate(args.dataset)
