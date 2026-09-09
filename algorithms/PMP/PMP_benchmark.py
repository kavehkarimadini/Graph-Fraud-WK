import argparse
import copy
import json
import os
import random
import sys
import time

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from algorithms.PMP.LASAGE_S import LASAGE_S
from utils.evaluation import (
    compute_classification_metrics,
    save_run_outputs,
)


# ---------------------------------------------------------------------
# Benchmark constants
# ---------------------------------------------------------------------

MODEL_NAME = "PMP"
DATASET_NAME = "amazon"

ALLOWED_SEEDS = [
    123,
    1234,
    12345,
    123456,
    1234567,
]

ALLOWED_TRAIN_RATIOS = [
    0.4,
    0.3,
    0.2,
    0.1,
]

# Unified benchmark protocol
MAX_EPOCHS = 500
EARLY_STOP = 25
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0

# PMP Amazon hyperparameters
# Paper values take priority over repository config where they differ.
HIDDEN_DIM = 128
DROPOUT = 0.6
NUM_LAYERS = 1
BATCH_SIZE = 128

# Other settings from official PMP implementation/config
NUM_TRANS = 1
AGGREGATION = "mean"
RELATION_AGGREGATION = "cat"
PROJECTION = True

# Official Amazon config uses ELU for the adaptive MLP.
MLP_ACTIVATION = nn.ELU(inplace=True)


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    dgl.seed(seed)

    # Improve reproducibility where supported.
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# ---------------------------------------------------------------------
# Feature normalization
# ---------------------------------------------------------------------

def row_normalize_features(features):
    """
    Reproduce PMP's row-normalization behaviour.

    Official PMP uses:
        rowsum = sum(row) + 0.01
        feature = feature / rowsum
    """
    features = features.float()

    row_sum = features.sum(dim=1, keepdim=True) + 0.01

    row_sum = torch.where(
        torch.isfinite(row_sum),
        row_sum,
        torch.ones_like(row_sum),
    )

    return features / row_sum


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

def load_amazon():
    from dgl.data import FraudDataset

    dataset_dir = os.path.join(
        PROJECT_ROOT,
        "dataset",
    )

    dataset = FraudDataset(
        name="amazon",
        raw_dir=dataset_dir,

        # These built-in masks are NOT used by our benchmark.
        # They only allow DGL to load the cached dataset.
        train_size=0.4,
        val_size=0.2,
        random_seed=717,

        force_reload=False,
    )

    graph = dataset[0]

    # Preserve the multi-relation Amazon graph.
    relations = list(graph.etypes)

    features = row_normalize_features(
        graph.ndata["feature"]
    )

    labels = graph.ndata["label"].long().squeeze()

    # Keep normalized features in graph node data.
    graph.ndata["feature"] = features

    return graph, features, labels, relations


# ---------------------------------------------------------------------
# Benchmark split loading
# ---------------------------------------------------------------------

def ratio_to_key(train_ratio):
    mapping = {
        0.4: "train_40",
        0.3: "train_30",
        0.2: "train_20",
        0.1: "train_10",
    }

    return mapping[float(train_ratio)]


def load_split(seed, train_ratio):
    split_path = os.path.join(
        PROJECT_ROOT,
        "splits",
        "amazon",
        f"seed_{seed}.npz",
    )

    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"Amazon benchmark split not found:\n{split_path}"
        )

    split_data = np.load(split_path)

    train_key = ratio_to_key(train_ratio)

    train_idx = split_data[train_key].astype(np.int64)
    valid_idx = split_data["val"].astype(np.int64)
    test_idx = split_data["test"].astype(np.int64)

    train40_idx = split_data["train_40"].astype(np.int64)

    # Anything from the 40% train pool not used by this
    # training-ratio experiment is discarded.
    discarded_idx = np.setdiff1d(
        train40_idx,
        train_idx,
        assume_unique=False,
    ).astype(np.int64)

    return (
        train_idx,
        valid_idx,
        test_idx,
        discarded_idx,
        split_path,
    )


# ---------------------------------------------------------------------
# Split validation
# ---------------------------------------------------------------------

def validate_split(
    train_idx,
    valid_idx,
    test_idx,
):
    train_set = set(train_idx.tolist())
    valid_set = set(valid_idx.tolist())
    test_set = set(test_idx.tolist())

    if train_set & valid_set:
        raise RuntimeError(
            "Split error: train and validation overlap."
        )

    if train_set & test_set:
        raise RuntimeError(
            "Split error: train and test overlap."
        )

    if valid_set & test_set:
        raise RuntimeError(
            "Split error: validation and test overlap."
        )


# ---------------------------------------------------------------------
# PMP label-status construction
# ---------------------------------------------------------------------

def configure_pmp_labels(
    graph,
    labels,
    train_idx,
):
    """
    PMP partitions neighbor messages according to known label status.

    0 = known benign training node
    1 = known fraud training node
    2 = unlabeled/unknown node

    Critically, validation and test labels are never exposed to PMP.
    """

    label_unk = torch.full(
        (graph.num_nodes(),),
        fill_value=2,
        dtype=torch.long,
    )

    train_tensor = torch.as_tensor(
        train_idx,
        dtype=torch.long,
    )

    label_unk[train_tensor] = labels[train_tensor]

    graph.ndata["label_unk"] = label_unk

    return graph


# ---------------------------------------------------------------------
# PMP model
# ---------------------------------------------------------------------

def build_model(
    features,
    relations,
    device,
):
    """
    Match the structure used by PMP's official prepare_model().

    With projection enabled:
      internal output dimension = hidden dimension
      projection output dimension = number of classes (2)
    """

    model = LASAGE_S(
        in_size=features.shape[1],
        hid_size=HIDDEN_DIM,

        # Official PMP behaviour when proj=True
        out_size=HIDDEN_DIM,

        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        proj=PROJECTION,

        num_relations=len(relations),
        batch_size=BATCH_SIZE,
        num_trans=NUM_TRANS,

        mlp_activation=MLP_ACTIVATION,

        # Binary fraud classification
        out_proj_size=2,

        agg=AGGREGATION,
        relation_agg=RELATION_AGGREGATION,
    )

    return model.to(device)


# ---------------------------------------------------------------------
# DGL loader
# ---------------------------------------------------------------------

def make_loader(
    graph,
    node_indices,
    batch_size,
    shuffle,
):
    sampler = MultiLayerFullNeighborSampler(
        num_layers=NUM_LAYERS
    )

    node_tensor = torch.as_tensor(
        node_indices,
        dtype=torch.long,
    )

    loader = DataLoader(
        graph,
        node_tensor,
        sampler,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
    )

    return loader


# ---------------------------------------------------------------------
# Forward helper
# ---------------------------------------------------------------------

def forward_batch(
    model,
    blocks,
    relations,
    features,
    input_nodes,
    device,
):
    blocks = [
        block.to(device)
        for block in blocks
    ]

    # Use input-node ordering generated by the sampler.
    batch_features = features[input_nodes].to(device)

    # Correct PMP signature:
    # model(blocks, relations, feats)
    logits = model(
        blocks,
        relations,
        batch_features,
    )

    return logits


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_nodes(
    model,
    graph,
    features,
    labels,
    relations,
    node_indices,
    device,
):
    model.eval()

    # Larger evaluation batch is safe and reduces overhead.
    eval_batch_size = 4096

    loader = make_loader(
        graph=graph,
        node_indices=node_indices,
        batch_size=eval_batch_size,
        shuffle=False,
    )

    all_indices = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for (
            input_nodes,
            output_nodes,
            blocks,
        ) in loader:

            logits = forward_batch(
                model=model,
                blocks=blocks,
                relations=relations,
                features=features,
                input_nodes=input_nodes,
                device=device,
            )

            batch_labels = labels[
                output_nodes
            ]

            all_indices.append(
                output_nodes.cpu()
            )

            all_labels.append(
                batch_labels.cpu()
            )

            all_logits.append(
                logits.detach().cpu()
            )

    output_indices = torch.cat(
        all_indices,
        dim=0,
    ).numpy()

    y_true = torch.cat(
        all_labels,
        dim=0,
    ).numpy()

    logits = torch.cat(
        all_logits,
        dim=0,
    )

    probabilities = torch.softmax(
        logits,
        dim=1,
    ).numpy()

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    return (
        output_indices,
        y_true,
        predictions,
        probabilities,
    )


# ---------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------

def train_one_run(
    seed,
    train_ratio,
):
    print("=" * 70)
    print("PMP BENCHMARK RUN")
    print("=" * 70)

    print(f"Dataset     : {DATASET_NAME}")
    print(f"Model       : {MODEL_NAME}")
    print(f"Seed        : {seed}")
    print(f"Train ratio : {train_ratio}")
    print(f"Max epochs  : {MAX_EPOCHS}")
    print(f"Early stop  : {EARLY_STOP}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("=" * 70)

    set_seed(seed)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "PMP benchmark must run on the Vast.ai GB10."
        )

    device = torch.device("cuda:0")

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    # -------------------------------------------------------------
    # Load graph
    # -------------------------------------------------------------

    (
        graph,
        features,
        labels,
        relations,
    ) = load_amazon()

    print(
        f"Nodes: {graph.num_nodes()}"
    )

    print(
        f"Edges: {graph.num_edges()}"
    )

    print(
        f"Relations: {relations}"
    )

    # -------------------------------------------------------------
    # Load exact canonical benchmark split
    # -------------------------------------------------------------

    (
        train_idx,
        valid_idx,
        test_idx,
        discarded_idx,
        split_path,
    ) = load_split(
        seed=seed,
        train_ratio=train_ratio,
    )

    validate_split(
        train_idx,
        valid_idx,
        test_idx,
    )

    print(
        f"Train nodes     : {len(train_idx)}"
    )

    print(
        f"Validation nodes: {len(valid_idx)}"
    )

    print(
        f"Test nodes      : {len(test_idx)}"
    )

    print(
        f"Discarded nodes : {len(discarded_idx)}"
    )

    # -------------------------------------------------------------
    # Give PMP access ONLY to training labels
    # -------------------------------------------------------------

    graph = configure_pmp_labels(
        graph=graph,
        labels=labels,
        train_idx=train_idx,
    )

    # Graph remains on CPU.
    # DGL sampled blocks are transferred batch-by-batch to GPU.

    # -------------------------------------------------------------
    # Build fresh model
    # -------------------------------------------------------------

    model = build_model(
        features=features,
        relations=relations,
        device=device,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Official PMP Amazon config uses unweighted CE.
    loss_function = nn.CrossEntropyLoss()

    train_loader = make_loader(
        graph=graph,
        node_indices=train_idx,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # -------------------------------------------------------------
    # Early stopping state
    # -------------------------------------------------------------

    best_val_auc = -np.inf
    best_state = None
    best_epoch = 0

    # Team's exact early-stop convention.
    wait = 1

    actual_epochs = 0

    start_time = time.perf_counter()

    # -------------------------------------------------------------
    # Train
    # -------------------------------------------------------------

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        actual_epochs = epoch

        model.train()

        total_loss = 0.0
        total_batches = 0

        for (
            input_nodes,
            output_nodes,
            blocks,
        ) in train_loader:

            logits = forward_batch(
                model=model,
                blocks=blocks,
                relations=relations,
                features=features,
                input_nodes=input_nodes,
                device=device,
            )

            batch_labels = labels[
                output_nodes
            ].to(device)

            optimizer.zero_grad()

            loss = loss_function(
                logits,
                batch_labels,
            )

            loss.backward()

            optimizer.step()

            total_loss += float(
                loss.detach().cpu().item()
            )

            total_batches += 1

        average_loss = (
            total_loss /
            max(total_batches, 1)
        )

        # ---------------------------------------------------------
        # Validation only
        # ---------------------------------------------------------

        (
            _,
            y_val,
            _,
            val_probabilities,
        ) = evaluate_nodes(
            model=model,
            graph=graph,
            features=features,
            labels=labels,
            relations=relations,
            node_indices=valid_idx,
            device=device,
        )

        val_auc = roc_auc_score(
            y_val,
            val_probabilities[:, 1],
        )

        improved = (
            val_auc > best_val_auc
        )

        if improved:
            best_val_auc = float(
                val_auc
            )

            best_epoch = int(
                epoch
            )

            best_state = copy.deepcopy(
                model.state_dict()
            )

            # Exact team convention
            wait = 1

        else:
            wait += 1

        print(
            f"Epoch {epoch:03d} | "
            f"Loss {average_loss:.6f} | "
            f"Val AUROC {val_auc:.6f} | "
            f"Best {best_val_auc:.6f} | "
            f"Best epoch {best_epoch} | "
            f"Wait {wait}/{EARLY_STOP}"
        )

        if wait >= EARLY_STOP:
            print(
                f"\nEarly stopping triggered "
                f"at epoch {epoch}."
            )

            break

    train_time = (
        time.perf_counter()
        - start_time
    )

    # -------------------------------------------------------------
    # Restore best validation checkpoint
    # -------------------------------------------------------------

    if best_state is None:
        raise RuntimeError(
            "No best model checkpoint was recorded."
        )

    model.load_state_dict(
        best_state
    )

    print(
        f"\nRestored best checkpoint "
        f"from epoch {best_epoch}"
    )

    print(
        f"Best validation AUROC: "
        f"{best_val_auc:.6f}"
    )

    # -------------------------------------------------------------
    # TEST SET — touched only now
    # -------------------------------------------------------------

    (
        test_order,
        y_test,
        test_predictions,
        test_probabilities,
    ) = evaluate_nodes(
        model=model,
        graph=graph,
        features=features,
        labels=labels,
        relations=relations,
        node_indices=test_idx,
        device=device,
    )

    test_metrics = (
        compute_classification_metrics(
            y_true=y_test,
            y_pred=test_predictions,
            y_prob=test_probabilities,
            sigmoid=False,
        )
    )

    # Add benchmark metadata to metrics.
    test_metrics[
        "best_epoch"
    ] = int(best_epoch)

    test_metrics[
        "best_val_auroc"
    ] = float(best_val_auc)

    test_metrics[
        "train_time_seconds"
    ] = float(train_time)

    test_metrics[
        "epochs_executed"
    ] = int(actual_epochs)

    # -------------------------------------------------------------
    # Split information
    # -------------------------------------------------------------

    split_info = {
        "seed": int(seed),

        "train_ratio":
            float(train_ratio),

        "train_size":
            int(len(train_idx)),

        "validation_size":
            int(len(valid_idx)),

        "test_size":
            int(len(test_idx)),

        "discarded_size":
            int(len(discarded_idx)),

        "train_pool_ratio":
            0.4,

        "validation_ratio":
            0.2,

        "test_ratio":
            0.4,

        "nested_training_subsets":
            True,

        "split_file":
            os.path.relpath(
                split_path,
                PROJECT_ROOT,
            ),
    }

    # -------------------------------------------------------------
    # Full configuration
    # -------------------------------------------------------------

    config = {
        "model":
            MODEL_NAME,

        "dataset":
            DATASET_NAME,

        "seed":
            int(seed),

        "train_ratio":
            float(train_ratio),

        # Benchmark-wide settings
        "optimizer":
            "Adam",

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "max_epochs":
            MAX_EPOCHS,

        "early_stop":
            EARLY_STOP,

        "early_stop_metric":
            "validation_AUROC",

        "early_stop_rule":
            "strict_greater_than",

        "early_stop_counter_reset":
            1,

        # PMP settings
        "hidden_dim":
            HIDDEN_DIM,

        "dropout":
            DROPOUT,

        "num_layers":
            NUM_LAYERS,

        "batch_size":
            BATCH_SIZE,

        "num_trans":
            NUM_TRANS,

        "aggregation":
            AGGREGATION,

        "relation_aggregation":
            RELATION_AGGREGATION,

        "projection":
            PROJECTION,

        "output_classes":
            2,

        "mlp_activation":
            "ELU",

        "feature_normalization":
            "PMP row normalization",

        "weighted_loss":
            False,

        "loss":
            "CrossEntropyLoss",

        # Hardware/software
        "platform":
            "Vast.ai",

        "gpu":
            torch.cuda.get_device_name(0),

        "torch_version":
            torch.__version__,

        "dgl_version":
            dgl.__version__,

        "cuda_version":
            torch.version.cuda,
    }

    # -------------------------------------------------------------
    # Output location
    # -------------------------------------------------------------

    ratio_percent = int(
        round(train_ratio * 100)
    )

    output_dir = os.path.join(
        PROJECT_ROOT,
        "results",
        MODEL_NAME,
        DATASET_NAME,
        f"seed_{seed}",
        f"train_{ratio_percent}",
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Save standardized project outputs
    # -------------------------------------------------------------

    save_run_outputs(
        save_dir=output_dir,

        model_name=MODEL_NAME,

        config=config,

        split_info=split_info,

        test_metrics=test_metrics,

        model=model,

        predictions=test_predictions,

        probabilities=test_probabilities,

        test_indices=test_order,

        train_indices=train_idx,

        valid_indices=valid_idx,

        discarded_indices=discarded_idx,

        y_test=y_test,

        y_train=labels[
            torch.as_tensor(
                train_idx,
                dtype=torch.long,
            )
        ].cpu().numpy(),

        train_time=train_time,
    )

    # Extra benchmark-specific metadata.
    metadata_path = os.path.join(
        output_dir,
        "benchmark_metadata.json",
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "model":
                    MODEL_NAME,

                "dataset":
                    DATASET_NAME,

                "seed":
                    int(seed),

                "train_ratio":
                    float(train_ratio),

                "best_epoch":
                    int(best_epoch),

                "best_validation_auroc":
                    float(best_val_auc),

                "epochs_executed":
                    int(actual_epochs),

                "train_time_seconds":
                    float(train_time),

                "test_evaluated_after_model_selection":
                    True,
            },
            file,
            indent=2,
        )

    # -------------------------------------------------------------
    # Final terminal output
    # -------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    ordered_metrics = [
        "auroc",
        "auprc",
        "g_mean",
        "f1_macro",
        "f1_micro",
        "f1_weighted",
        "precision_macro",
        "recall_macro",
        "accuracy",
        "best_val_auroc",
        "best_epoch",
        "epochs_executed",
        "train_time_seconds",
    ]

    printed = set()

    for metric in ordered_metrics:
        if metric in test_metrics:
            value = test_metrics[
                metric
            ]

            if isinstance(
                value,
                (
                    float,
                    np.floating,
                ),
            ):
                print(
                    f"{metric:25s}: "
                    f"{float(value):.6f}"
                )
            else:
                print(
                    f"{metric:25s}: "
                    f"{value}"
                )

            printed.add(metric)

    # Print any additional metrics produced by
    # utils/evaluation.py.
    for metric, value in (
        test_metrics.items()
    ):
        if metric in printed:
            continue

        print(
            f"{metric:25s}: "
            f"{value}"
        )

    print("=" * 70)

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best validation AUROC: "
        f"{best_val_auc:.6f}"
    )

    print(
        f"Training time: "
        f"{train_time:.2f} seconds"
    )

    print(
        f"Results saved to:\n"
        f"{output_dir}"
    )

    print("=" * 70)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "PMP unified benchmark runner"
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        choices=ALLOWED_SEEDS,
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        required=True,
        choices=ALLOWED_TRAIN_RATIOS,
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    train_one_run(
        seed=args.seed,
        train_ratio=args.train_ratio,
    )