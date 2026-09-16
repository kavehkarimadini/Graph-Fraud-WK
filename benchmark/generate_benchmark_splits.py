"""Generate canonical nested 10/20/30/40% benchmark train splits."""

import argparse
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split


ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from algorithms.PMP.pmp_config import canonical_dataset_name


SEEDS = [
    123,
    1234,
    12345,
    123456,
    1234567,
]


def load_split_source(dataset):
    """
    Load only the information required for benchmark split generation.

    FDCompCN deliberately uses the repository's DGL-free loader so that
    canonical splits can be generated locally without importing DGL.
    Other datasets retain the existing PMP dataset pathway.
    """

    dataset = canonical_dataset_name(dataset)

    if dataset == "fdcompcn":
        from utils.data_loader import load_comp_arrays

        data = load_comp_arrays(
            "fdcompcn",
            prefix=os.path.join(
                ROOT,
                "dataset",
            ),
        )

        labels = np.asarray(
            data["labels"],
            dtype=np.int64,
        ).reshape(-1)

        num_nodes = int(
            labels.shape[0]
        )

        eligible = np.arange(
            num_nodes,
            dtype=np.int64,
        )

        eligible = eligible[
            np.isin(
                labels[eligible],
                [0, 1],
            )
        ]

        return (
            num_nodes,
            labels,
            eligible,
        )

    # Existing pathway for all other benchmark datasets.
    # Import lazily so local FDCompCN split generation does not import DGL.
    from algorithms.PMP.pmp_data import load_pmp_dataset

    (
        graph,
        _,
        labels_t,
        _,
        eligible,
    ) = load_pmp_dataset(
        dataset,
        ROOT,
    )

    labels = (
        labels_t
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64)
    )

    return (
        graph.num_nodes(),
        labels,
        np.asarray(
            eligible,
            dtype=np.int64,
        ),
    )


def generate(dataset):
    dataset = canonical_dataset_name(
        dataset
    )

    (
        num_nodes,
        labels,
        eligible,
    ) = load_split_source(
        dataset
    )

    out_dir = os.path.join(
        ROOT,
        "splits",
        dataset,
    )

    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    print(
        f"Dataset: {dataset}"
    )

    print(
        f"Nodes: {num_nodes} | "
        f"Eligible labelled nodes: {len(eligible)}"
    )

    for seed in SEEDS:

        train40, remainder = train_test_split(
            eligible,
            train_size=0.4,
            random_state=seed,
            shuffle=True,
            stratify=labels[eligible],
        )

        val, test = train_test_split(
            remainder,
            train_size=1 / 3,
            random_state=seed + 1,
            shuffle=True,
            stratify=labels[remainder],
        )

        train30, _ = train_test_split(
            train40,
            train_size=0.75,
            random_state=seed + 2,
            shuffle=True,
            stratify=labels[train40],
        )

        train20, _ = train_test_split(
            train30,
            train_size=2 / 3,
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

        arrays = {
            k: np.sort(
                v.astype(np.int64)
            )
            for k, v in {
                "train_40": train40,
                "train_30": train30,
                "train_20": train20,
                "train_10": train10,
                "val": val,
                "test": test,
            }.items()
        }

        assert set(
            arrays["train_10"]
        ).issubset(
            set(arrays["train_20"])
        )

        assert set(
            arrays["train_20"]
        ).issubset(
            set(arrays["train_30"])
        )

        assert set(
            arrays["train_30"]
        ).issubset(
            set(arrays["train_40"])
        )

        assert set(
            arrays["train_40"]
        ).isdisjoint(
            arrays["val"]
        )

        assert set(
            arrays["train_40"]
        ).isdisjoint(
            arrays["test"]
        )

        assert set(
            arrays["val"]
        ).isdisjoint(
            arrays["test"]
        )

        assert (
            set(arrays["train_40"])
            | set(arrays["val"])
            | set(arrays["test"])
            == set(eligible)
        )

        out = os.path.join(
            out_dir,
            f"seed_{seed}.npz",
        )

        np.savez_compressed(
            out,
            seed=seed,
            **arrays,

            discarded_30=np.setdiff1d(
                arrays["train_40"],
                arrays["train_30"],
            ),

            discarded_20=np.setdiff1d(
                arrays["train_40"],
                arrays["train_20"],
            ),

            discarded_10=np.setdiff1d(
                arrays["train_40"],
                arrays["train_10"],
            ),

            eligible=np.sort(
                eligible
            ),
        )

        print(
            f"\nSeed {seed}"
        )

        for name in [
            "train_40",
            "train_30",
            "train_20",
            "train_10",
            "val",
            "test",
        ]:

            idx = arrays[name]

            counts = np.bincount(
                labels[idx],
                minlength=2,
            )

            print(
                f"{name:8s}: "
                f"{len(idx):7d} | "
                f"benign={counts[0]:7d} "
                f"fraud={counts[1]:7d}"
            )

        print(
            "Nestedness/disjointness/coverage: PASS"
        )

        print(
            "Saved:",
            os.path.relpath(
                out,
                ROOT,
            ),
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "amazon",
            "yelp",
            "yelpchi",
            "tfinance",
            "elliptic",
            "fdcompcn",
        ],
    )

    args = parser.parse_args()

    generate(
        args.dataset
    )