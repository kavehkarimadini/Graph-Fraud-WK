from __future__ import print_function

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import copy
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from algorithms.FRAUDRE.FRAUDRE import (
    FRAUDRE,
    prior_adjusted_loss,
    sample_adjacency,
)
from utils.data_loader import load_dga_arrays


DATASET = "elliptic"

SEEDS = [123, 1234, 12345, 123456, 1234567]

EMBED_CANDIDATES = [16, 32, 64, 128]
LAYER_CANDIDATES = [1, 2]
BATCH_CANDIDATES = [256, 1024]

LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 64
MAX_NEIGHBORS = 10

MAX_EPOCHS = 500
EARLY_STOP = 25

TRAIN_KEY = "train_40"
VALID_KEY = "val"

ROOT = Path(__file__).resolve().parents[1]
SPLIT_ROOT = ROOT / "splits" / DATASET
TUNING_ROOT = ROOT / "tuning" / "FRAUDRE" / DATASET


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def config_name(embed_dim, n_layers, batch_size):
    return "embed_{}_layers_{}_batch_{}".format(
        embed_dim, n_layers, batch_size
    )


def run_dir(embed_dim, n_layers, batch_size, seed):
    return (
        TUNING_ROOT
        / config_name(embed_dim, n_layers, batch_size)
        / "seed_{}".format(seed)
    )


def result_path(embed_dim, n_layers, batch_size, seed):
    return run_dir(
        embed_dim, n_layers, batch_size, seed
    ) / "tuning_result.json"


def load_split(seed):
    path = SPLIT_ROOT / "seed_{}.npz".format(seed)

    if not path.exists():
        raise FileNotFoundError("Missing canonical split: {}".format(path))

    split = np.load(path)

    # Intentionally access only train_40 and val.
    # The test split is never loaded or evaluated during tuning.
    train_idx = np.asarray(split[TRAIN_KEY], dtype=np.int64)
    val_idx = np.asarray(split[VALID_KEY], dtype=np.int64)

    return train_idx, val_idx


def prepare_dataset():
    features, labels, relations, labeled_from = load_dga_arrays(DATASET)

    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    # T-Finance is unsupported by the FRAUDRE paper.
    # Keep preprocessing fixed rather than tune it.
    # This matches the repository FRAUDRE Elliptic preprocessing convention.
    features = StandardScaler().fit_transform(features).astype(np.float32)

    return features, labels, relations, labeled_from


@torch.no_grad()
def validation_auc(model, feat, labels, val_idx, rels, batch_size):
    model.eval()

    score_parts = []

    for start in range(0, len(val_idx), batch_size):
        batch_np = val_idx[start:start + batch_size]
        batch = torch.as_tensor(
            batch_np,
            dtype=torch.long,
            device=feat.device,
        )

        logits = model(feat, batch, rels)

        # Match the unified FRAUDRE benchmark inference rule.
        scores = torch.sigmoid(logits)
        score_parts.append(scores[:, 1].detach().cpu().numpy())

    y_val = labels[val_idx]
    y_score = np.concatenate(score_parts)

    return float(roc_auc_score(y_val, y_score))


def train_validation_only(
    prepared,
    embed_dim,
    n_layers,
    batch_size,
    seed,
    device,
):
    out_dir = run_dir(embed_dim, n_layers, batch_size, seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)

    features, labels_np, relations, labeled_from = prepared
    n = int(features.shape[0])

    train_np, val_np = load_split(seed)

    if np.intersect1d(train_np, val_np).size != 0:
        raise RuntimeError("train_40 and val overlap.")

    if train_np.min() < 0 or train_np.max() >= n:
        raise RuntimeError("Invalid train indices.")

    if val_np.min() < 0 or val_np.max() >= n:
        raise RuntimeError("Invalid validation indices.")

    feat = torch.as_tensor(
        features,
        dtype=torch.float32,
        device=device,
    )

    labels = torch.as_tensor(
        labels_np,
        dtype=torch.long,
        device=device,
    )

    rel_names = [name for name, _src, _dst in relations]
    rels = []

    for r, (_name, src, dst) in enumerate(relations):
        indptr, indices = sample_adjacency(
            src,
            dst,
            n,
            MAX_NEIGHBORS,
            seed=seed + r,
        )

        rels.append(
            (
                torch.as_tensor(
                    indptr,
                    dtype=torch.long,
                    device=device,
                ),
                torch.as_tensor(
                    indices,
                    dtype=torch.long,
                    device=device,
                ),
            )
        )

    n_class = int(labels.max().item()) + 1

    prior = np.bincount(
        labels_np[train_np],
        minlength=n_class,
    ).astype(np.float64)

    prior /= float(len(train_np))

    log_prior = torch.log(
        torch.as_tensor(
            prior + 1e-8,
            dtype=torch.float32,
            device=device,
        )
    )

    model = FRAUDRE(
        feat.size(1),
        len(rels),
        embed_dim=embed_dim,
        n_layers=n_layers,
        hidden_dim=HIDDEN_DIM,
        n_class=n_class,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    rng = np.random.default_rng(seed)

    print("=" * 72)
    print(
        "FRAUDRE Elliptic tuning | "
        "embed={} | layers={} | batch={} | seed={}".format(
            embed_dim,
            n_layers,
            batch_size,
            seed,
        )
    )
    print(
        "Train={} | Val={} | Relations={} | test split NOT loaded".format(
            len(train_np),
            len(val_np),
            rel_names,
        )
    )

    best_val_auc = -np.inf
    best_epoch = 0
    best_state = None
    wait = 1
    epochs_executed = 0

    start_time = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        epochs_executed = epoch
        model.train()

        shuffled = np.asarray(train_np, dtype=np.int64).copy()
        rng.shuffle(shuffled)

        num_batches = max(
            1,
            math.ceil(len(shuffled) / batch_size),
        )

        epoch_loss = 0.0

        for start in range(0, len(shuffled), batch_size):
            batch_np = shuffled[start:start + batch_size]

            batch = torch.as_tensor(
                batch_np,
                dtype=torch.long,
                device=device,
            )

            optimizer.zero_grad(set_to_none=True)

            logits = model(
                feat,
                batch,
                rels,
            )

            loss = prior_adjusted_loss(
                logits,
                labels[batch],
                log_prior,
            )

            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())

        val_auc = validation_auc(
            model,
            feat,
            labels_np,
            val_np,
            rels,
            batch_size,
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

            # Locked project convention.
            wait = 1
        else:
            wait += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                "Epoch {:03d} | loss {:.6f} | "
                "val_AUROC {:.6f} | best {:.6f} @ {}".format(
                    epoch,
                    epoch_loss / num_batches,
                    val_auc,
                    best_val_auc,
                    best_epoch,
                )
            )

        if wait >= EARLY_STOP:
            print("Early stopping at epoch {}".format(epoch))
            break

    train_time = time.perf_counter() - start_time

    if best_state is None:
        raise RuntimeError("No best validation checkpoint was recorded.")

    model.load_state_dict(best_state)

    # Validation-only tuning ends here.
    # No test indices, labels, predictions, or metrics are accessed.

    result = {
        "model": "FRAUDRE",
        "dataset": DATASET,
        "purpose": "validation_only_hyperparameter_tuning",

        "embed_dim": int(embed_dim),
        "n_layers": int(n_layers),
        "batch_size": int(batch_size),

        "hidden_dim": HIDDEN_DIM,
        "max_neighbors": MAX_NEIGHBORS,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,

        "feature_preprocessing": "standard_scaler_fixed_not_tuned",

        "seed": int(seed),
        "training_split_key": TRAIN_KEY,
        "validation_split_key": VALID_KEY,

        "train_size": int(len(train_np)),
        "validation_size": int(len(val_np)),

        "best_validation_auroc": float(best_val_auc),
        "best_epoch": int(best_epoch),
        "epochs_executed": int(epochs_executed),
        "train_time_seconds": float(train_time),

        "max_epochs": MAX_EPOCHS,
        "early_stop": EARLY_STOP,
        "early_stop_counter_reset": 1,
        "checkpoint_metric": "validation_AUROC",
        "checkpoint_rule": "strict_greater_than",

        "inference_score_rule": "sigmoid_on_each_output_logit",

        "test_split_loaded": False,
        "test_used_during_training": False,
        "test_used_for_early_stopping": False,
        "test_used_for_checkpoint_selection": False,
        "test_used_for_hyperparameter_selection": False,

        "train_prior": [float(x) for x in prior],

        "relations": rel_names,

        "num_parameters": int(
            sum(p.numel() for p in model.parameters())
        ),
        "num_trainable_parameters": int(
            sum(
                p.numel()
                for p in model.parameters()
                if p.requires_grad
            )
        ),
    }

    path = out_dir / "tuning_result.json"

    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(
        "DONE | best validation AUROC={:.6f} | epoch={} | {:.2f}s".format(
            best_val_auc,
            best_epoch,
            train_time,
        )
    )
    print("Saved:", path)

    return result


def summarise():
    rows = []

    for embed_dim in EMBED_CANDIDATES:
        for n_layers in LAYER_CANDIDATES:
            for batch_size in BATCH_CANDIDATES:
                for seed in SEEDS:
                    path = result_path(
                        embed_dim,
                        n_layers,
                        batch_size,
                        seed,
                    )

                    if not path.exists():
                        continue

                    with path.open(
                        "r",
                        encoding="utf-8",
                    ) as handle:
                        result = json.load(handle)

                    rows.append(
                        {
                            "embed_dim": embed_dim,
                            "n_layers": n_layers,
                            "batch_size": batch_size,
                            "seed": seed,
                            "best_validation_auroc": float(
                                result["best_validation_auroc"]
                            ),
                            "best_epoch": int(
                                result["best_epoch"]
                            ),
                            "epochs_executed": int(
                                result["epochs_executed"]
                            ),
                            "train_time_seconds": float(
                                result["train_time_seconds"]
                            ),
                        }
                    )

    if not rows:
        raise RuntimeError("No completed tuning runs found.")

    TUNING_ROOT.mkdir(parents=True, exist_ok=True)

    runs_csv = TUNING_ROOT / "fraudre_elliptic_tuning_runs.csv"

    with runs_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    summaries = []

    for embed_dim in EMBED_CANDIDATES:
        for n_layers in LAYER_CANDIDATES:
            for batch_size in BATCH_CANDIDATES:
                config_rows = [
                    r
                    for r in rows
                    if r["embed_dim"] == embed_dim
                    and r["n_layers"] == n_layers
                    and r["batch_size"] == batch_size
                ]

                if len(config_rows) != len(SEEDS):
                    continue

                values = np.asarray(
                    [
                        r["best_validation_auroc"]
                        for r in config_rows
                    ],
                    dtype=np.float64,
                )

                summaries.append(
                    {
                        "embed_dim": embed_dim,
                        "n_layers": n_layers,
                        "batch_size": batch_size,
                        "num_seeds": len(config_rows),
                        "mean_validation_auroc": float(
                            values.mean()
                        ),
                        "std_validation_auroc": float(
                            values.std(ddof=1)
                        ),
                    }
                )

    if not summaries:
        raise RuntimeError(
            "No configuration has all five completed seeds."
        )

    # Highest mean validation AUROC wins.
    # Exact tie: smaller model first, then smaller batch.
    summaries.sort(
        key=lambda r: (
            -r["mean_validation_auroc"],
            r["embed_dim"],
            r["n_layers"],
            r["batch_size"],
        )
    )

    summary_csv = (
        TUNING_ROOT
        / "fraudre_elliptic_tuning_summary.csv"
    )

    with summary_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(summaries[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summaries)

    winner = summaries[0]

    selected = {
        "model": "FRAUDRE",
        "dataset": DATASET,
        "purpose": "freeze_configuration_before_final_benchmark",

        "embed_dim": int(winner["embed_dim"]),
        "n_layers": int(winner["n_layers"]),
        "batch_size": int(winner["batch_size"]),

        "hidden_dim": HIDDEN_DIM,
        "max_neighbors": MAX_NEIGHBORS,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,

        "feature_preprocessing": "standard_scaler_fixed_not_tuned",

        "selection_training_split": TRAIN_KEY,
        "selection_validation_split": VALID_KEY,
        "selection_seeds": list(SEEDS),
        "selection_metric": "mean_validation_AUROC",

        "selected_mean_validation_auroc": float(
            winner["mean_validation_auroc"]
        ),
        "selected_std_validation_auroc": float(
            winner["std_validation_auroc"]
        ),

        "test_used_for_hyperparameter_selection": False,

        "search_space": {
            "embed_dim": EMBED_CANDIDATES,
            "n_layers": LAYER_CANDIDATES,
            "batch_size": BATCH_CANDIDATES,
        },

        "max_epochs": MAX_EPOCHS,
        "early_stop": EARLY_STOP,
        "early_stop_counter_reset": 1,
        "checkpoint_metric": "validation_AUROC",
        "checkpoint_rule": "strict_greater_than",
    }

    selected_path = (
        TUNING_ROOT
        / "fraudre_elliptic_selected_config.json"
    )

    with selected_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(selected, handle, indent=2)

    print("\nTUNING COMPLETE")
    print("=" * 72)

    for row in summaries:
        print(
            "embed={:<3} layers={} batch={:<4} | "
            "mean val AUROC={:.6f} +/- {:.6f}".format(
                row["embed_dim"],
                row["n_layers"],
                row["batch_size"],
                row["mean_validation_auroc"],
                row["std_validation_auroc"],
            )
        )

    print("\nSELECTED CONFIG:")
    print(json.dumps(selected, indent=2))
    print("\nSaved:", runs_csv)
    print("Saved:", summary_csv)
    print("Saved:", selected_path)


def complete(embed_dim, n_layers, batch_size, seed):
    return result_path(
        embed_dim,
        n_layers,
        batch_size,
        seed,
    ).exists()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validation-only FRAUDRE Elliptic "
            "hyperparameter tuner"
        )
    )

    parser.add_argument(
        "--embed-dim",
        type=int,
        choices=EMBED_CANDIDATES,
    )

    parser.add_argument(
        "--n-layers",
        type=int,
        choices=LAYER_CANDIDATES,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        choices=BATCH_CANDIDATES,
    )

    parser.add_argument(
        "--seed",
        type=int,
        choices=SEEDS,
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all 16 configs across all five fixed seeds.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed tuning_result.json runs.",
    )

    parser.add_argument(
        "--summarise-only",
        action="store_true",
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.summarise_only:
        summarise()
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Run tuning on the GPU instance."
        )

    device = torch.device(
        "cuda:{}".format(args.gpu)
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(device),
    )

    prepared = prepare_dataset()

    if args.all:
        for embed_dim in EMBED_CANDIDATES:
            for n_layers in LAYER_CANDIDATES:
                for batch_size in BATCH_CANDIDATES:
                    for seed in SEEDS:
                        if (
                            args.resume
                            and complete(
                                embed_dim,
                                n_layers,
                                batch_size,
                                seed,
                            )
                        ):
                            print(
                                "SKIP completed:",
                                config_name(
                                    embed_dim,
                                    n_layers,
                                    batch_size,
                                ),
                                "seed={}".format(seed),
                            )
                            continue

                        train_validation_only(
                            prepared,
                            embed_dim,
                            n_layers,
                            batch_size,
                            seed,
                            device,
                        )

        summarise()
        return

    if (
        args.embed_dim is None
        or args.n_layers is None
        or args.batch_size is None
        or args.seed is None
    ):
        raise ValueError(
            "For one run provide --embed-dim, --n-layers, "
            "--batch-size and --seed, or use --all."
        )

    train_validation_only(
        prepared,
        args.embed_dim,
        args.n_layers,
        args.batch_size,
        args.seed,
        device,
    )


if __name__ == "__main__":
    main()
