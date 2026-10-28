"""
Validation-only CARE-GNN hyperparameter tuning for T-Finance.

This tuner intentionally never reads the canonical test split and never
computes test predictions or test metrics.

Protocol:
- dataset: T-Finance
- training-label budget: 40% only
- seeds: 123, 1234, 12345, 123456, 1234567
- candidate batch sizes: 256, 1024
- selection metric: mean validation AUROC across five seeds
- max epochs: 500
- early stopping: patience 25, strict validation-AUROC improvement
- preprocessing: official CARE-GNN row normalization
"""

from __future__ import print_function

import argparse
import copy
import csv
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BENCHMARK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCHMARK_DIR.parent

if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import CAREGNN_benchmark as core
from algorithms.CAREGNN.CAREGNN import CAREGNN, build_csr
from utils.data_loader import load_dga_arrays
from utils.torch_helpers import set_seed

DATASET = "elliptic"
TRAIN_RATIO = 0.4
TRAIN_KEY = "train_40"
VALID_KEY = "val"
SEEDS = tuple(core.ALLOWED_SEEDS)
BATCH_CANDIDATES = (256, 1024)

BASE_CONFIG = {
    "hidden_dim": 64,
    "learning_rate": 0.01,
    "weight_decay": 0.001,
    "lambda_1": 2.0,
    "step_size": 0.02,
    "initial_threshold": 0.5,
    "under_sampling_scale": 1.0,
    "num_layers": 1,
    "optimizer": "Adam",
    "feature_preprocessing": "row",
    "source_note": (
        "CARE-GNN paper/official implementation defaults; "
        "T-Finance batch size selected by validation-only tuning."
    ),
}

TUNING_ROOT = PROJECT_ROOT / "tuning" / "CAREGNN" / DATASET


def result_path(batch_size, seed):
    return (
        TUNING_ROOT
        / "batch_{}".format(batch_size)
        / "seed_{}".format(seed)
        / "tuning_result.json"
    )


def load_tuning_split(seed):
    """Load only train_40 and validation arrays; never index/read test."""
    path = PROJECT_ROOT / "splits" / DATASET / "seed_{}.npz".format(seed)
    if not path.exists():
        raise FileNotFoundError("Missing canonical split file: {}".format(path))

    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in (TRAIN_KEY, VALID_KEY) if key not in data.files]
        if missing:
            raise RuntimeError("Missing tuning split arrays {} in {}".format(missing, path))
        train_np = np.asarray(data[TRAIN_KEY], dtype=np.int64).copy()
        valid_np = np.asarray(data[VALID_KEY], dtype=np.int64).copy()

    if train_np.ndim != 1 or valid_np.ndim != 1:
        raise RuntimeError("Tuning split arrays must be one-dimensional.")
    if len(np.intersect1d(train_np, valid_np)) != 0:
        raise RuntimeError("Training and validation tuning splits overlap.")
    return train_np, valid_np, path


def validate_tuning_indices(train_np, valid_np, eligible, labels_np):
    eligible_set = set(np.asarray(eligible, dtype=np.int64).tolist())
    for name, indices in (("train", train_np), ("validation", valid_np)):
        if len(indices) == 0:
            raise RuntimeError("{} split is empty.".format(name))
        if len(np.unique(indices)) != len(indices):
            raise RuntimeError("{} split contains duplicate indices.".format(name))
        if not set(indices.tolist()).issubset(eligible_set):
            raise RuntimeError("{} split contains ineligible indices.".format(name))
        if not np.isin(labels_np[indices], [0, 1]).all():
            raise RuntimeError("{} split contains non-binary labels.".format(name))

    if len(np.unique(labels_np[valid_np])) < 2:
        raise RuntimeError("Validation split contains fewer than two classes.")


def prepare_dataset(device):
    features_np, labels_np, relation_arrays, labeled_from = load_dga_arrays(
        DATASET,
        prefix=str(PROJECT_ROOT / "dataset"),
    )

    features_np = np.asarray(features_np, dtype=np.float32)
    labels_np = np.asarray(labels_np, dtype=np.int64).reshape(-1)

    if features_np.ndim != 2:
        raise RuntimeError("Expected a 2-D feature matrix.")
    if features_np.shape[0] != labels_np.shape[0]:
        raise RuntimeError("Feature/label node counts do not match.")
    if not np.isfinite(features_np).all():
        raise RuntimeError("T-Finance features contain non-finite values.")

    num_nodes = int(features_np.shape[0])
    if labeled_from is None:
        eligible = np.arange(num_nodes, dtype=np.int64)
    else:
        eligible = np.arange(int(labeled_from), num_nodes, dtype=np.int64)
    eligible = eligible[np.isin(labels_np[eligible], [0, 1])]

    features_np = core.normalize_features(features_np, "standard")
    features = torch.as_tensor(features_np, dtype=torch.float32, device=device)
    labels = torch.as_tensor(labels_np, dtype=torch.long, device=device)

    relation_names = []
    relations = []
    for relation_name, src, dst in relation_arrays:
        relation_names.append(relation_name)
        indptr, indices = build_csr(src, dst, num_nodes)
        relations.append(
            (
                torch.as_tensor(indptr, dtype=torch.long, device=device),
                torch.as_tensor(indices, dtype=torch.long, device=device),
            )
        )
        del indptr, indices

    del relation_arrays
    gc.collect()

    return {
        "features": features,
        "labels": labels,
        "labels_np": labels_np,
        "eligible": eligible,
        "relations": relations,
        "relation_names": relation_names,
    }


def train_validation_only(batch_size, seed, prepared, device):
    if batch_size not in BATCH_CANDIDATES:
        raise ValueError("Unsupported tuning batch size: {}".format(batch_size))

    set_seed(seed)
    rng = np.random.default_rng(seed)
    train_np, valid_np, split_path = load_tuning_split(seed)
    validate_tuning_indices(
        train_np,
        valid_np,
        prepared["eligible"],
        prepared["labels_np"],
    )

    config = dict(BASE_CONFIG)
    config["batch_size"] = int(batch_size)

    features = prepared["features"]
    labels = prepared["labels"]
    labels_np = prepared["labels_np"]
    relations = prepared["relations"]

    train_pos = train_np[labels_np[train_np] == 1]
    train_neg = train_np[labels_np[train_np] == 0]
    if len(train_pos) == 0 or len(train_neg) == 0:
        raise RuntimeError("Training split must contain both classes.")

    valid_tensor = torch.as_tensor(valid_np, dtype=torch.long, device=device)
    num_classes = int(labels.max().item()) + 1

    model = CAREGNN(
        features.size(1),
        config["hidden_dim"],
        len(relations),
        n_class=num_classes,
        inter="GNN",
        step_size=config["step_size"],
        init_threshold=config["initial_threshold"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    print("=" * 72)
    print(
        "CAREGNN T-Finance TUNING | batch={} | seed={} | train_ratio=0.4".format(
            batch_size, seed
        )
    )
    print(
        "Train={} | Val={} | Relations={} | test split NOT loaded".format(
            len(train_np), len(valid_np), prepared["relation_names"]
        )
    )

    best_val_auc = -np.inf
    best_epoch = 0
    best_state = None
    best_thresholds = None
    wait = 1
    previous_relation_scores = None
    thresholds_log = [model.thresholds.tolist()]
    start_time = time.perf_counter()
    epochs_executed = 0

    for epoch in range(1, core.MAX_EPOCHS + 1):
        epochs_executed = epoch
        model.train()

        num_negative_samples = min(
            len(train_neg),
            int(round(len(train_pos) * config["under_sampling_scale"])),
        )
        sampled_negative = rng.choice(
            train_neg,
            size=num_negative_samples,
            replace=False,
        )
        training_nodes = np.concatenate([train_pos, sampled_negative])
        rng.shuffle(training_nodes)

        num_batches = max(1, math.ceil(len(training_nodes) / config["batch_size"]))
        epoch_loss = 0.0
        batch_relation_scores = []

        for batch_number in range(num_batches):
            start = batch_number * config["batch_size"]
            end = (batch_number + 1) * config["batch_size"]
            batch = torch.as_tensor(
                training_nodes[start:end],
                dtype=torch.long,
                device=device,
            )
            batch_labels = labels[batch]

            gnn_logits, label_logits, relation_scores = model(
                features,
                batch,
                relations,
                batch_labels,
            )
            classification_loss = F.cross_entropy(gnn_logits, batch_labels)
            similarity_loss = F.cross_entropy(label_logits, batch_labels)
            loss = classification_loss + config["lambda_1"] * similarity_loss

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "CARE-GNN loss became non-finite at epoch {}.".format(epoch)
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            batch_relation_scores.append(relation_scores)

        validation_probabilities = core.predict_proba(
            model=model,
            features=features,
            indices=valid_tensor,
            relations=relations,
            batch_size=core.EVAL_BATCH_SIZE,
        )
        val_auc = core.validation_auc(labels[valid_tensor], validation_probabilities)

        if val_auc > best_val_auc:
            best_val_auc = float(val_auc)
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            best_thresholds = model.thresholds.detach().clone()
            wait = 1
        else:
            wait += 1

        current_relation_scores = core.epoch_relation_scores(
            batch_relation_scores,
            len(relations),
        )
        if previous_relation_scores is not None:
            model.rl_update(previous_relation_scores, current_relation_scores)
        previous_relation_scores = current_relation_scores
        thresholds_log.append(model.thresholds.tolist())

        print(
            "Epoch {:03d} | Loss {:.6f} | Val AUROC {:.6f} | "
            "Best {:.6f} @ {} | Wait {}/{} | Thresholds {}".format(
                epoch,
                epoch_loss / num_batches,
                val_auc,
                best_val_auc,
                best_epoch,
                wait,
                core.EARLY_STOP,
                [round(float(v), 3) for v in model.thresholds.tolist()],
            )
        )

        if wait >= core.EARLY_STOP:
            print("Early stopping at epoch {}.".format(epoch))
            break

    train_time = time.perf_counter() - start_time
    if best_state is None:
        raise RuntimeError("No best validation checkpoint was recorded.")

    model.load_state_dict(best_state)
    if best_thresholds is not None:
        with torch.no_grad():
            model.thresholds.copy_(best_thresholds.to(model.thresholds.device))

    return {
        "model": "CAREGNN",
        "dataset": DATASET,
        "purpose": "validation_only_hyperparameter_tuning",
        "seed": int(seed),
        "train_ratio": TRAIN_RATIO,
        "batch_size": int(batch_size),
        "best_validation_auroc": float(best_val_auc),
        "best_epoch": int(best_epoch),
        "epochs_executed": int(epochs_executed),
        "train_time_seconds": float(train_time),
        "hidden_dim": int(config["hidden_dim"]),
        "learning_rate": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
        "lambda_1": float(config["lambda_1"]),
        "step_size": float(config["step_size"]),
        "initial_threshold": float(config["initial_threshold"]),
        "under_sampling_scale": float(config["under_sampling_scale"]),
        "num_layers": int(config["num_layers"]),
        "optimizer": config["optimizer"],
        "feature_preprocessing": config["feature_preprocessing"],
        "relations": prepared["relation_names"],
        "final_restored_thresholds": model.thresholds.tolist(),
        "thresholds_log": thresholds_log,
        "max_epochs": int(core.MAX_EPOCHS),
        "early_stop": int(core.EARLY_STOP),
        "checkpoint_metric": "validation_AUROC",
        "checkpoint_rule": "strict_greater_than",
        "split_file": os.path.relpath(split_path, PROJECT_ROOT),
        "train_split_key": TRAIN_KEY,
        "validation_split_key": VALID_KEY,
        "test_split_loaded": False,
        "test_predictions_computed": False,
        "test_metrics_computed": False,
        "hyperparameter_source": config["source_note"],
    }


def save_result(result):
    path = result_path(result["batch_size"], result["seed"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print("Saved:", path)


def is_complete(batch_size, seed):
    return result_path(batch_size, seed).exists()


def aggregate_if_complete():
    rows = []
    for batch_size in BATCH_CANDIDATES:
        for seed in SEEDS:
            path = result_path(batch_size, seed)
            if not path.exists():
                return False
            with path.open("r", encoding="utf-8") as handle:
                result = json.load(handle)
            rows.append(
                {
                    "batch_size": int(result["batch_size"]),
                    "seed": int(result["seed"]),
                    "best_validation_auroc": float(result["best_validation_auroc"]),
                    "best_epoch": int(result["best_epoch"]),
                    "epochs_executed": int(result["epochs_executed"]),
                    "train_time_seconds": float(result["train_time_seconds"]),
                }
            )

    TUNING_ROOT.mkdir(parents=True, exist_ok=True)
    runs_csv = TUNING_ROOT / "caregnn_elliptic_tuning_runs.csv"
    with runs_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summaries = []
    for batch_size in BATCH_CANDIDATES:
        values = np.asarray(
            [
                row["best_validation_auroc"]
                for row in rows
                if row["batch_size"] == batch_size
            ],
            dtype=np.float64,
        )
        summaries.append(
            {
                "batch_size": int(batch_size),
                "num_seeds": int(values.size),
                "mean_validation_auroc": float(values.mean()),
                "std_validation_auroc": float(values.std(ddof=1)),
            }
        )

    summaries.sort(key=lambda row: (-row["mean_validation_auroc"], row["batch_size"]))
    summary_csv = TUNING_ROOT / "caregnn_elliptic_tuning_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    winner = summaries[0]
    selected = {
        "model": "CAREGNN",
        "dataset": DATASET,
        "selection_budget": TRAIN_RATIO,
        "selection_seeds": list(SEEDS),
        "selection_metric": "mean_validation_AUROC",
        "selection_rule": "highest mean validation AUROC; exact tie -> smaller batch size",
        "candidates": summaries,
        "selected_batch_size": int(winner["batch_size"]),
        "selected_mean_validation_auroc": float(winner["mean_validation_auroc"]),
        "test_split_loaded_during_tuning": False,
        "test_predictions_computed_during_tuning": False,
        "test_metrics_computed_during_tuning": False,
    }

    selected_path = TUNING_ROOT / "caregnn_elliptic_selected_config.json"
    with selected_path.open("w", encoding="utf-8") as handle:
        json.dump(selected, handle, indent=2)

    print("=" * 72)
    print("TUNING COMPLETE")
    for row in summaries:
        print(
            "Batch {} | mean val AUROC {:.10f} | std {:.10f}".format(
                row["batch_size"],
                row["mean_validation_auroc"],
                row["std_validation_auroc"],
            )
        )
    print("SELECTED batch_size={}".format(winner["batch_size"]))
    print("Saved:", runs_csv)
    print("Saved:", summary_csv)
    print("Saved:", selected_path)
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validation-only CARE-GNN T-Finance hyperparameter tuner"
    )
    parser.add_argument("--batch-size", type=int, choices=BATCH_CANDIDATES)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run both batch-size candidates across all five fixed seeds.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tuning runs whose tuning_result.json already exists.",
    )
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run tuning on the GPU instance.")

    device = torch.device("cuda:{}".format(args.gpu))
    print("GPU:", torch.cuda.get_device_name(device))
    prepared = prepare_dataset(device)

    try:
        if args.all:
            for batch_size in BATCH_CANDIDATES:
                for seed in SEEDS:
                    if args.resume and is_complete(batch_size, seed):
                        print("SKIP complete: batch={} seed={}".format(batch_size, seed))
                        continue
                    result = train_validation_only(
                        batch_size=batch_size,
                        seed=seed,
                        prepared=prepared,
                        device=device,
                    )
                    save_result(result)
                    torch.cuda.empty_cache()
            aggregate_if_complete()
            return

        if args.batch_size is None or args.seed is None:
            raise SystemExit(
                "Single-run mode requires both --batch-size and --seed, or use --all."
            )

        if args.resume and is_complete(args.batch_size, args.seed):
            print(
                "SKIP complete: batch={} seed={}".format(args.batch_size, args.seed)
            )
        else:
            result = train_validation_only(
                batch_size=args.batch_size,
                seed=args.seed,
                prepared=prepared,
                device=device,
            )
            save_result(result)

        aggregate_if_complete()
    finally:
        del prepared
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
