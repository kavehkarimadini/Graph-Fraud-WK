"""Validation-only SEC-GFD hyperparameter search for Elliptic.

IMPORTANT:
- Uses only train_40 and validation nodes.
- Never reads or evaluates the test split.
- Hyperparameters are selected by mean validation AUROC across 5 seeds.
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from algorithms.SEC_GFD.model.SECGFD import SECGFD
from algorithms.SEC_GFD.secgfd_data import load_secgfd_dataset
from algorithms.SEC_GFD.SEC_GFD_benchmark import (
    corrected_environment_loss,
    set_seed,
)


DATASET = "elliptic"

SEEDS = [
    123,
    1234,
    12345,
    123456,
    1234567,
]

HIDDEN_DIMS = [16, 64]
ORDERS = [2, 5]

LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0
LAMBDA_ENV = 0.2
HIGH_ORDER = 2

MAX_EPOCHS = 500
EARLY_STOP = 25

OUTPUT_DIR = os.path.join(
    ROOT,
    "tuning",
    "SEC_GFD",
    DATASET,
)


def load_train_validation_split(seed):
    path = os.path.join(
        ROOT,
        "splits",
        DATASET,
        "seed_{}.npz".format(seed),
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            "Elliptic benchmark split not found: {}".format(path)
        )

    data = np.load(path)

    # Deliberately read ONLY training and validation.
    # The test split is untouched during tuning.
    train_idx = data[
        "train_40"
    ].astype(np.int64)

    valid_idx = data[
        "val"
    ].astype(np.int64)

    if set(train_idx.tolist()) & set(valid_idx.tolist()):
        raise RuntimeError(
            "Training and validation splits overlap."
        )

    return train_idx, valid_idx


def evaluate_validation_auc(
    model,
    features,
    labels,
    valid_idx,
):
    model.eval()

    idx = torch.as_tensor(
        valid_idx,
        dtype=torch.long,
        device=labels.device,
    )

    with torch.no_grad():
        logits, _ = model(features)

        probabilities = torch.softmax(
            logits[idx],
            dim=1,
        )

    y_true = labels[
        idx
    ].detach().cpu().numpy()

    y_score = probabilities[
        :, 1
    ].detach().cpu().numpy()

    return float(
        roc_auc_score(
            y_true,
            y_score,
        )
    )


def train_validation_run(
    graph,
    features,
    labels,
    eligible,
    seed,
    hidden_dim,
    order_c,
):
    set_seed(seed)

    train_idx, valid_idx = (
        load_train_validation_split(seed)
    )

    eligible_set = set(
        np.asarray(
            eligible,
            dtype=np.int64,
        ).tolist()
    )

    if not set(train_idx.tolist()).issubset(
        eligible_set
    ):
        raise RuntimeError(
            "Training split contains ineligible nodes."
        )

    if not set(valid_idx.tolist()).issubset(
        eligible_set
    ):
        raise RuntimeError(
            "Validation split contains ineligible nodes."
        )

    device = torch.device("cuda:0")

    model = SECGFD(
        in_dim=int(features.shape[1]),
        hid_dim=int(hidden_dim),
        out_dim=2,
        graph=graph,
        d=int(order_c),
        high_order=HIGH_ORDER,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    train_tensor = torch.as_tensor(
        train_idx,
        dtype=torch.long,
        device=device,
    )

    train_labels = labels[
        train_tensor
    ]

    benign_count = int(
        (train_labels == 0).sum().item()
    )

    fraud_count = int(
        (train_labels == 1).sum().item()
    )

    if benign_count == 0:
        raise RuntimeError(
            "Training split contains zero benign nodes."
        )

    if fraud_count == 0:
        raise RuntimeError(
            "Training split contains zero fraud nodes."
        )

    fraud_weight = (
        benign_count / fraud_count
    )

    class_weight = torch.tensor(
        [
            1.0,
            fraud_weight,
        ],
        dtype=torch.float32,
        device=device,
    )

    best_val_auc = -np.inf
    best_epoch = 0
    wait = 1
    actual_epochs = 0

    start = time.perf_counter()

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        actual_epochs = epoch

        model.train()

        optimizer.zero_grad()

        logits, embeddings = model(
            features
        )

        classification_loss = (
            F.cross_entropy(
                logits[
                    train_tensor
                ],
                labels[
                    train_tensor
                ],
                weight=class_weight,
            )
        )

        environment_loss = (
            corrected_environment_loss(
                embeddings=embeddings,
                features=features,
                labels=labels,
                train_idx=train_idx,
            )
        )

        loss = (
            classification_loss
            + LAMBDA_ENV
            * environment_loss
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite loss at epoch {}."
                .format(epoch)
            )

        loss.backward()
        optimizer.step()

        val_auc = evaluate_validation_auc(
            model,
            features,
            labels,
            valid_idx,
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            wait = 1
        else:
            wait += 1

        print(
            "hidden={} C={} seed={} "
            "epoch={:03d} "
            "loss={:.6f} "
            "val_auc={:.6f} "
            "best={:.6f}@{} "
            "wait={}/{}".format(
                hidden_dim,
                order_c,
                seed,
                epoch,
                float(
                    loss.detach().cpu()
                ),
                val_auc,
                best_val_auc,
                best_epoch,
                wait,
                EARLY_STOP,
            )
        )

        if wait >= EARLY_STOP:
            print(
                "Early stopping at epoch {}."
                .format(epoch)
            )
            break

    elapsed = (
        time.perf_counter()
        - start
    )

    return {
        "dataset": DATASET,
        "seed": int(seed),
        "train_ratio": 0.4,
        "hidden_dim": int(hidden_dim),
        "order_C": int(order_c),
        "high_order": int(HIGH_ORDER),
        "learning_rate": float(
            LEARNING_RATE
        ),
        "weight_decay": float(
            WEIGHT_DECAY
        ),
        "lambda_env": float(
            LAMBDA_ENV
        ),
        "optimizer": "Adam",
        "max_epochs": int(
            MAX_EPOCHS
        ),
        "early_stop": int(
            EARLY_STOP
        ),
        "early_stop_metric": (
            "validation_AUROC"
        ),
        "early_stop_rule": (
            "strict_greater_than"
        ),
        "early_stop_counter_reset": 1,
        "best_val_auroc": float(
            best_val_auc
        ),
        "best_epoch": int(
            best_epoch
        ),
        "epochs_executed": int(
            actual_epochs
        ),
        "train_size": int(
            len(train_idx)
        ),
        "validation_size": int(
            len(valid_idx)
        ),
        "fraud_class_weight": float(
            fraud_weight
        ),
        "train_time_seconds": float(
            elapsed
        ),
        "test_accessed": False,
    }


def save_run(result):
    run_dir = os.path.join(
        OUTPUT_DIR,
        "hidden_{}".format(
            result["hidden_dim"]
        ),
        "C_{}".format(
            result["order_C"]
        ),
    )

    os.makedirs(
        run_dir,
        exist_ok=True,
    )

    path = os.path.join(
        run_dir,
        "seed_{}.json".format(
            result["seed"]
        ),
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    return path


def existing_run(
    hidden_dim,
    order_c,
    seed,
):
    path = os.path.join(
        OUTPUT_DIR,
        "hidden_{}".format(
            hidden_dim
        ),
        "C_{}".format(
            order_c
        ),
        "seed_{}.json".format(
            seed
        ),
    )

    if not os.path.exists(path):
        return None

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def summarize(results):
    rows = []

    for hidden_dim in HIDDEN_DIMS:
        for order_c in ORDERS:
            group = [
                r
                for r in results
                if (
                    r["hidden_dim"]
                    == hidden_dim
                    and r["order_C"]
                    == order_c
                )
            ]

            if len(group) != len(SEEDS):
                raise RuntimeError(
                    "Incomplete tuning configuration: "
                    "hidden={} C={} has {}/{} seeds."
                    .format(
                        hidden_dim,
                        order_c,
                        len(group),
                        len(SEEDS),
                    )
                )

            values = np.asarray(
                [
                    r["best_val_auroc"]
                    for r in group
                ],
                dtype=np.float64,
            )

            rows.append({
                "hidden_dim": int(
                    hidden_dim
                ),
                "order_C": int(
                    order_c
                ),
                "num_seeds": int(
                    len(values)
                ),
                "mean_validation_auroc": float(
                    values.mean()
                ),
                "std_validation_auroc": float(
                    values.std()
                ),
            })

    # Primary criterion:
    # highest mean validation AUROC.
    #
    # Predeclared deterministic tie-break:
    # lower validation-AUROC std,
    # then smaller hidden dimension,
    # then smaller C.
    ranked = sorted(
        rows,
        key=lambda r: (
            -r[
                "mean_validation_auroc"
            ],
            r[
                "std_validation_auroc"
            ],
            r[
                "hidden_dim"
            ],
            r[
                "order_C"
            ],
        ),
    )

    winner = ranked[0]

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    csv_path = os.path.join(
        OUTPUT_DIR,
        "secgfd_elliptic_tuning_summary.csv",
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "hidden_dim",
                "order_C",
                "num_seeds",
                "mean_validation_auroc",
                "std_validation_auroc",
            ],
        )

        writer.writeheader()
        writer.writerows(ranked)

    json_path = os.path.join(
        OUTPUT_DIR,
        "secgfd_elliptic_tuning_summary.json",
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "dataset": DATASET,
                "selection_metric": (
                    "mean_validation_AUROC"
                ),
                "test_accessed": False,
                "search_space": {
                    "hidden_dim": HIDDEN_DIMS,
                    "order_C": ORDERS,
                },
                "fixed_hyperparameters": {
                    "optimizer": "Adam",
                    "learning_rate": (
                        LEARNING_RATE
                    ),
                    "weight_decay": (
                        WEIGHT_DECAY
                    ),
                    "lambda_env": (
                        LAMBDA_ENV
                    ),
                    "high_order": (
                        HIGH_ORDER
                    ),
                    "max_epochs": (
                        MAX_EPOCHS
                    ),
                    "early_stop": (
                        EARLY_STOP
                    ),
                },
                "results": ranked,
                "selected_configuration": (
                    winner
                ),
            },
            f,
            indent=2,
        )

    print("=" * 70)
    print("VALIDATION-ONLY TUNING COMPLETE")
    print("=" * 70)

    for row in ranked:
        print(
            "hidden={} C={} | "
            "mean val AUROC={:.6f} "
            "std={:.6f}".format(
                row["hidden_dim"],
                row["order_C"],
                row[
                    "mean_validation_auroc"
                ],
                row[
                    "std_validation_auroc"
                ],
            )
        )

    print("")
    print(
        "SELECTED: hidden_dim={} C={}".format(
            winner["hidden_dim"],
            winner["order_C"],
        )
    )

    print(
        "Test set accessed: False"
    )

    print(
        "Saved:",
        OUTPUT_DIR,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validation-only SEC-GFD "
            "Elliptic hyperparameter search"
        )
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip tuning runs that already "
            "have a saved JSON result."
        ),
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for SEC-GFD tuning."
        )

    device = torch.device(
        "cuda:0"
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    (
        graph,
        _,
        _,
        eligible,
    ) = load_secgfd_dataset(
        DATASET,
        ROOT,
    )

    graph = graph.to(device)

    features = graph.ndata[
        "feature"
    ].float()

    labels = graph.ndata[
        "label"
    ].long().view(-1)

    print(
        "Nodes={} Edges={} Features={} Eligible={}".format(
            graph.num_nodes(),
            graph.num_edges(),
            features.shape[1],
            len(eligible),
        )
    )

    results = []

    for hidden_dim in HIDDEN_DIMS:
        for order_c in ORDERS:
            for seed in SEEDS:

                if args.resume:
                    previous = existing_run(
                        hidden_dim,
                        order_c,
                        seed,
                    )

                    if previous is not None:
                        print(
                            "SKIP complete tuning run: "
                            "hidden={} C={} seed={}"
                            .format(
                                hidden_dim,
                                order_c,
                                seed,
                            )
                        )

                        results.append(
                            previous
                        )
                        continue

                print("=" * 70)
                print(
                    "TUNING | hidden={} | C={} | seed={}"
                    .format(
                        hidden_dim,
                        order_c,
                        seed,
                    )
                )
                print("=" * 70)

                result = train_validation_run(
                    graph=graph,
                    features=features,
                    labels=labels,
                    eligible=eligible,
                    seed=seed,
                    hidden_dim=hidden_dim,
                    order_c=order_c,
                )

                path = save_run(
                    result
                )

                print(
                    "Saved:",
                    path,
                )

                results.append(
                    result
                )

    summarize(results)


if __name__ == "__main__":
    main()