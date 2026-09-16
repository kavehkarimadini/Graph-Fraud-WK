"""Validation-only hyperparameter tuning for SEC-GFD on FDCompCN."""

import argparse
import copy
import csv
import json
import os
import sys
import time

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from algorithms.SEC_GFD.model.SECGFD import SECGFD
from algorithms.SEC_GFD.secgfd_data import load_secgfd_dataset
from algorithms.SEC_GFD.SEC_GFD_benchmark import (
    corrected_environment_loss,
    set_seed,
)


DATASET = "fdcompcn"

SEEDS = [
    123,
    1234,
    12345,
    123456,
    1234567,
]

HIDDEN_DIMS = [
    16,
    64,
]

ORDERS = [
    2,
    5,
]

LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0
LAMBDA_ENV = 0.2
HIGH_ORDER = 2

MAX_EPOCHS = 500
EARLY_STOP = 25

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "tuning",
    "SEC_GFD",
    DATASET,
)


def load_train_validation_split(seed):
    """
    Load only train_40 and validation nodes.

    The test split is deliberately never read by this tuning script.
    """

    path = os.path.join(
        PROJECT_ROOT,
        "splits",
        DATASET,
        "seed_{}.npz".format(seed),
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            "Missing benchmark split: {}".format(path)
        )

    data = np.load(path)

    train_idx = data[
        "train_40"
    ].astype(np.int64)

    val_idx = data[
        "val"
    ].astype(np.int64)

    if not set(
        train_idx.tolist()
    ).isdisjoint(
        set(val_idx.tolist())
    ):
        raise RuntimeError(
            "Training and validation nodes overlap."
        )

    return (
        train_idx,
        val_idx,
        path,
    )


def output_path(
    hidden_dim,
    order_c,
    seed,
):
    return os.path.join(
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


def evaluate_validation_auc(
    model,
    features,
    labels,
    val_idx,
):
    model.eval()

    idx = torch.as_tensor(
        val_idx,
        dtype=torch.long,
        device=labels.device,
    )

    with torch.no_grad():
        logits, _ = model(
            features
        )

        probabilities = torch.softmax(
            logits[idx],
            dim=1,
        )[:, 1]

    y_true = (
        labels[idx]
        .detach()
        .cpu()
        .numpy()
    )

    y_prob = (
        probabilities
        .detach()
        .cpu()
        .numpy()
    )

    return float(
        roc_auc_score(
            y_true,
            y_prob,
        )
    )


def run_one(
    graph,
    features,
    labels,
    eligible,
    hidden_dim,
    order_c,
    seed,
    resume=False,
):
    save_path = output_path(
        hidden_dim,
        order_c,
        seed,
    )

    if (
        resume
        and os.path.exists(
            save_path
        )
    ):
        print(
            "SKIP complete: "
            "hidden={} C={} seed={}".format(
                hidden_dim,
                order_c,
                seed,
            )
        )

        with open(
            save_path,
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)

    set_seed(seed)

    (
        train_idx,
        val_idx,
        split_path,
    ) = load_train_validation_split(
        seed
    )

    eligible_set = set(
        np.asarray(
            eligible,
            dtype=np.int64,
        ).tolist()
    )

    if (
        set(train_idx.tolist())
        - eligible_set
    ):
        raise RuntimeError(
            "Training split contains "
            "ineligible nodes."
        )

    if (
        set(val_idx.tolist())
        - eligible_set
    ):
        raise RuntimeError(
            "Validation split contains "
            "ineligible nodes."
        )

    print("=" * 70)

    print(
        "SEC-GFD FDCompCN TUNING | "
        "hidden={} | C={} | seed={}".format(
            hidden_dim,
            order_c,
            seed,
        )
    )

    print("=" * 70)

    print(
        "Train={} Validation={}".format(
            len(train_idx),
            len(val_idx),
        )
    )

    model = SECGFD(
        in_dim=int(
            features.shape[1]
        ),
        hid_dim=int(
            hidden_dim
        ),
        out_dim=2,
        graph=graph,
        d=int(
            order_c
        ),
        high_order=HIGH_ORDER,
    ).to(
        features.device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    train_tensor = torch.as_tensor(
        train_idx,
        dtype=torch.long,
        device=labels.device,
    )

    train_labels = labels[
        train_tensor
    ]

    benign_count = int(
        (
            train_labels == 0
        ).sum().item()
    )

    fraud_count = int(
        (
            train_labels == 1
        ).sum().item()
    )

    if benign_count == 0:
        raise RuntimeError(
            "Training split contains "
            "zero benign nodes."
        )

    if fraud_count == 0:
        raise RuntimeError(
            "Training split contains "
            "zero fraud nodes."
        )

    fraud_weight = (
        benign_count
        / fraud_count
    )

    class_weight = torch.tensor(
        [
            1.0,
            fraud_weight,
        ],
        dtype=torch.float32,
        device=labels.device,
    )

    best_val_auroc = -np.inf
    best_epoch = 0
    best_state = None

    wait = 1
    epochs_executed = 0

    start_time = time.perf_counter()

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        epochs_executed = epoch

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

        if not torch.isfinite(
            loss
        ):
            raise RuntimeError(
                "Non-finite training loss "
                "at epoch {}.".format(
                    epoch
                )
            )

        loss.backward()

        optimizer.step()

        val_auc = (
            evaluate_validation_auc(
                model,
                features,
                labels,
                val_idx,
            )
        )

        if (
            val_auc
            > best_val_auroc
        ):
            best_val_auroc = float(
                val_auc
            )

            best_epoch = int(
                epoch
            )

            best_state = copy.deepcopy(
                model.state_dict()
            )

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
                best_val_auroc,
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

    train_time_seconds = (
        time.perf_counter()
        - start_time
    )

    if best_state is None:
        raise RuntimeError(
            "No best validation checkpoint "
            "was recorded."
        )

    # Restore validation-selected checkpoint.
    # No test evaluation is performed here.
    model.load_state_dict(
        best_state
    )

    result = {
        "dataset": DATASET,
        "seed": int(seed),
        "train_ratio": 0.4,

        "hidden_dim": int(
            hidden_dim
        ),

        "order_C": int(
            order_c
        ),

        "high_order": HIGH_ORDER,

        "learning_rate": (
            LEARNING_RATE
        ),

        "weight_decay": (
            WEIGHT_DECAY
        ),

        "lambda_env": (
            LAMBDA_ENV
        ),

        "optimizer": "Adam",

        "max_epochs": (
            MAX_EPOCHS
        ),

        "early_stop": (
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
            best_val_auroc
        ),

        "best_epoch": int(
            best_epoch
        ),

        "epochs_executed": int(
            epochs_executed
        ),

        "train_size": int(
            len(train_idx)
        ),

        "validation_size": int(
            len(val_idx)
        ),

        "fraud_class_weight": float(
            fraud_weight
        ),

        "train_time_seconds": float(
            train_time_seconds
        ),

        "split_file": os.path.relpath(
            split_path,
            PROJECT_ROOT,
        ),

        "test_accessed": False,

        "selection_note": (
            "Hyperparameter tuning uses train_40 "
            "and validation only. Test is untouched."
        ),
    }

    os.makedirs(
        os.path.dirname(
            save_path
        ),
        exist_ok=True,
    )

    with open(
        save_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print(
        "Saved:",
        save_path,
    )

    return result


def summarize():
    rows = []

    for hidden_dim in HIDDEN_DIMS:
        for order_c in ORDERS:
            values = []

            for seed in SEEDS:
                path = output_path(
                    hidden_dim,
                    order_c,
                    seed,
                )

                if not os.path.exists(
                    path
                ):
                    raise RuntimeError(
                        "Missing tuning run: {}"
                        .format(path)
                    )

                with open(
                    path,
                    "r",
                    encoding="utf-8",
                ) as f:
                    result = json.load(
                        f
                    )

                if (
                    result.get(
                        "test_accessed"
                    )
                    is not False
                ):
                    raise RuntimeError(
                        "Tuning record indicates "
                        "test access: {}".format(
                            path
                        )
                    )

                values.append(
                    float(
                        result[
                            "best_val_auroc"
                        ]
                    )
                )

            rows.append(
                {
                    "hidden_dim": int(
                        hidden_dim
                    ),

                    "order_C": int(
                        order_c
                    ),

                    "num_seeds": len(
                        values
                    ),

                    "mean_val_auroc": float(
                        np.mean(values)
                    ),

                    "std_val_auroc": float(
                        np.std(values)
                    ),

                    "test_accessed": False,
                }
            )

    ranked = sorted(
        rows,
        key=lambda x: (
            -x[
                "mean_val_auroc"
            ],
            x[
                "std_val_auroc"
            ],
            x[
                "hidden_dim"
            ],
            x[
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
        "secgfd_fdcompcn_tuning_summary.csv",
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
                "mean_val_auroc",
                "std_val_auroc",
                "test_accessed",
            ],
        )

        writer.writeheader()
        writer.writerows(
            ranked
        )

    json_path = os.path.join(
        OUTPUT_DIR,
        "secgfd_fdcompcn_tuning_summary.json",
    )

    payload = {
        "dataset": DATASET,

        "selection_metric": (
            "mean validation AUROC "
            "across five predefined seeds"
        ),

        "train_ratio_used_for_tuning": 0.4,

        "test_accessed": False,

        "search_space": {
            "hidden_dim": HIDDEN_DIMS,
            "order_C": ORDERS,
        },

        "fixed_settings": {
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

        "selected": {
            "hidden_dim": int(
                winner[
                    "hidden_dim"
                ]
            ),

            "order_C": int(
                winner[
                    "order_C"
                ]
            ),

            "mean_val_auroc": float(
                winner[
                    "mean_val_auroc"
                ]
            ),

            "std_val_auroc": float(
                winner[
                    "std_val_auroc"
                ]
            ),
        },
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
        )

    print("=" * 70)
    print(
        "VALIDATION-ONLY TUNING COMPLETE"
    )
    print("=" * 70)

    for row in ranked:
        print(
            "hidden={} C={} | "
            "mean val AUROC={:.6f} "
            "std={:.6f}".format(
                row[
                    "hidden_dim"
                ],
                row[
                    "order_C"
                ],
                row[
                    "mean_val_auroc"
                ],
                row[
                    "std_val_auroc"
                ],
            )
        )

    print()

    print(
        "SELECTED: hidden_dim={} C={}"
        .format(
            winner[
                "hidden_dim"
            ],
            winner[
                "order_C"
            ],
        )
    )

    print(
        "Test set accessed: False"
    )

    print(
        "Saved:",
        OUTPUT_DIR,
    )


def load_dataset_on_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "Run tuning on a GPU instance."
        )

    device = torch.device(
        "cuda:0"
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(
            0
        ),
    )

    (
        graph,
        _,
        _,
        eligible,
    ) = load_secgfd_dataset(
        DATASET,
        PROJECT_ROOT,
    )

    print(
        "Nodes={} Edges={} Eligible={}".format(
            graph.num_nodes(),
            graph.num_edges(),
            len(eligible),
        )
    )

    graph = graph.to(
        device
    )

    features = graph.ndata[
        "feature"
    ].float()

    labels = graph.ndata[
        "label"
    ].long().view(-1)

    print(
        "Features={}".format(
            tuple(
                features.shape
            )
        )
    )

    print(
        "Benign={} Fraud={}".format(
            int(
                (
                    labels == 0
                ).sum().item()
            ),
            int(
                (
                    labels == 1
                ).sum().item()
            ),
        )
    )

    return (
        graph,
        features,
        labels,
        eligible,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validation-only SEC-GFD "
            "FDCompCN hyperparameter tuning"
        )
    )

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run one tuning configuration only: "
            "hidden=64, C=2, seed=123."
        ),
    )

    group.add_argument(
        "--all",
        action="store_true",
        help=(
            "Run all 4 configurations "
            "x 5 seeds."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip tuning runs whose JSON "
            "output already exists."
        ),
    )

    args = parser.parse_args()

    (
        graph,
        features,
        labels,
        eligible,
    ) = load_dataset_on_gpu()

    if args.smoke:
        run_one(
            graph=graph,
            features=features,
            labels=labels,
            eligible=eligible,
            hidden_dim=64,
            order_c=2,
            seed=123,
            resume=args.resume,
        )

        print("=" * 70)
        print(
            "SMOKE TUNING RUN COMPLETE"
        )
        print(
            "No test data was accessed."
        )
        print("=" * 70)

        return

    for hidden_dim in HIDDEN_DIMS:
        for order_c in ORDERS:
            for seed in SEEDS:
                run_one(
                    graph=graph,
                    features=features,
                    labels=labels,
                    eligible=eligible,
                    hidden_dim=hidden_dim,
                    order_c=order_c,
                    seed=seed,
                    resume=args.resume,
                )

    summarize()


if __name__ == "__main__":
    main()