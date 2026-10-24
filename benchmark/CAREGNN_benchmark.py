"""
CARE-GNN runner for the unified graph-fraud benchmark.

Current paper-supported datasets:
    YelpChi
    Amazon

Unified benchmark protocol:
    - Canonical saved 40/20/40 train/validation/test split
    - Nested 10/20/30/40% training-label budgets
    - Seeds: 123, 1234, 12345, 123456, 1234567
    - Maximum 500 epochs
    - Early stopping patience 25
    - Validation AUROC used for checkpoint selection
    - Strict validation improvement
    - Best validation checkpoint restored before testing
    - Test set accessed only after model selection
"""

from __future__ import print_function

import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from algorithms.CAREGNN.CAREGNN import CAREGNN, build_csr

from caregnn_config import (
    canonical_dataset_name,
    get_caregnn_config,
)

from utils.data_loader import load_dga_arrays

from utils.evaluation import (
    compute_classification_metrics,
    save_run_outputs,
)

from utils.torch_helpers import set_seed


# ---------------------------------------------------------------------
# Unified benchmark constants
# ---------------------------------------------------------------------

MODEL_NAME = "CAREGNN"

ALLOWED_SEEDS = [
    123,
    1234,
    12345,
    123456,
    1234567,
]

ALLOWED_TRAIN_RATIOS = [
    0.1,
    0.2,
    0.3,
    0.4,
]

MAX_EPOCHS = 500
EARLY_STOP = 25
EVAL_BATCH_SIZE = 4096


# ---------------------------------------------------------------------
# Dataset / split helpers
# ---------------------------------------------------------------------

def ratio_to_key(train_ratio):
    return {
        0.1: "train_10",
        0.2: "train_20",
        0.3: "train_30",
        0.4: "train_40",
    }[float(train_ratio)]


def loader_dataset_name(dataset):
    """
    Convert canonical benchmark names to names expected by load_dga_arrays.
    """
    if dataset == "yelp":
        return "yelpchi"

    if dataset == "fdcompcn":
        return "comp"

    return dataset


def output_dir_for(dataset, seed, train_ratio):
    pct = int(
        round(
            float(train_ratio) * 100
        )
    )

    return os.path.join(
        PROJECT_ROOT,
        "results",
        MODEL_NAME,
        dataset,
        "seed_{}".format(seed),
        "train_{}".format(pct),
    )


def run_complete(dataset, seed, train_ratio):
    out_dir = output_dir_for(
        dataset,
        seed,
        train_ratio,
    )

    required = [
        "metrics.json",
        "config.json",
        "benchmark_metadata.json",
        "split.npz",
        "predictions.npz",
        "checkpoint.pt",
    ]

    return all(
        os.path.exists(
            os.path.join(
                out_dir,
                filename,
            )
        )
        for filename in required
    )


def load_split(dataset, seed, train_ratio):
    """
    Load exactly the same canonical split files used by PMP and SEC-GFD.
    """
    path = os.path.join(
        PROJECT_ROOT,
        "splits",
        dataset,
        "seed_{}.npz".format(seed),
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            "Canonical benchmark split not found: {}\n"
            "Do not generate a new split inside CARE-GNN."
            .format(path)
        )

    data = np.load(path)

    train_idx = data[
        ratio_to_key(train_ratio)
    ].astype(np.int64)

    valid_idx = data[
        "val"
    ].astype(np.int64)

    test_idx = data[
        "test"
    ].astype(np.int64)

    train40_idx = data[
        "train_40"
    ].astype(np.int64)

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
        path,
    )


def validate_split(
    train_idx,
    valid_idx,
    test_idx,
):
    train_set = set(
        train_idx.tolist()
    )
    valid_set = set(
        valid_idx.tolist()
    )
    test_set = set(
        test_idx.tolist()
    )

    if (
        train_set & valid_set
        or train_set & test_set
        or valid_set & test_set
    ):
        raise RuntimeError(
            "Split validation failed: "
            "train/validation/test overlap detected."
        )


def validate_eligible_indices(
    train_idx,
    valid_idx,
    test_idx,
    eligible,
):
    eligible_set = set(
        np.asarray(
            eligible,
            dtype=np.int64,
        ).tolist()
    )

    for name, values in [
        ("train", train_idx),
        ("validation", valid_idx),
        ("test", test_idx),
    ]:
        invalid = (
            set(values.tolist())
            - eligible_set
        )

        if invalid:
            raise RuntimeError(
                "{} split contains nodes that "
                "are not eligible labelled nodes."
                .format(name)
            )


# ---------------------------------------------------------------------
# Feature preprocessing
# ---------------------------------------------------------------------

def normalize_features(
    features,
    mode,
):
    features = np.asarray(
        features,
        dtype=np.float32,
    )

    if mode == "row":
        if features.min() < 0:
            raise RuntimeError(
                "Row normalisation requested for features "
                "containing negative values."
            )

        row_sum = (
            features.sum(
                axis=1,
                keepdims=True,
            )
            + 0.01
        )

        return (
            features / row_sum
        ).astype(np.float32)

    if mode == "standard":
        return (
            StandardScaler()
            .fit_transform(features)
            .astype(np.float32)
        )

    if mode == "none":
        return features.astype(
            np.float32
        )

    raise ValueError(
        "Unknown feature normalisation mode: {}"
        .format(mode)
    )


def feature_normalisation_for(dataset):
    """
    Preserve the preprocessing pathway used by the current CARE-GNN port.

    YelpChi:
        row normalisation

    Amazon:
        standard scaling because its loaded feature representation
        may contain signed values.
    """
    if dataset == "yelp":
        return "row"

    if dataset == "amazon":
        return "standard"

    return "standard"


# ---------------------------------------------------------------------
# CARE-GNN evaluation helpers
# ---------------------------------------------------------------------

@torch.no_grad()
def predict_proba(
    model,
    features,
    indices,
    relations,
    batch_size,
):
    model.eval()

    probabilities = []

    for start in range(
        0,
        indices.numel(),
        batch_size,
    ):
        batch = indices[
            start:start + batch_size
        ]

        logits, _, _ = model(
            features,
            batch,
            relations,
        )

        probabilities.append(
            torch.softmax(
                logits,
                dim=-1,
            )
        )

    return torch.cat(
        probabilities,
        dim=0,
    )


def validation_auc(
    labels,
    probabilities,
):
    y_true = (
        labels
        .detach()
        .cpu()
        .numpy()
    )

    if len(
        np.unique(y_true)
    ) < 2:
        raise RuntimeError(
            "Validation set contains fewer than two classes."
        )

    return float(
        roc_auc_score(
            y_true,
            probabilities[
                :, 1
            ]
            .detach()
            .cpu()
            .numpy(),
        )
    )


def epoch_relation_scores(
    batch_scores,
    num_relations,
):
    """
    Mean relation score over mini-batches.

    Batches without usable fraud-node relation scores are ignored.
    """
    output = []

    for relation_id in range(
        num_relations
    ):
        values = [
            scores[relation_id]
            for scores in batch_scores
            if not math.isnan(
                scores[relation_id]
            )
        ]

        if values:
            output.append(
                float(
                    np.mean(values)
                )
            )
        else:
            output.append(
                math.nan
            )

    return output


@torch.no_grad()
def evaluate_indices(
    model,
    features,
    labels,
    indices_np,
    relations,
):
    index_tensor = torch.as_tensor(
        indices_np,
        dtype=torch.long,
        device=labels.device,
    )

    probabilities = predict_proba(
        model=model,
        features=features,
        indices=index_tensor,
        relations=relations,
        batch_size=EVAL_BATCH_SIZE,
    )

    predictions = torch.argmax(
        probabilities,
        dim=1,
    )

    y_true = (
        labels[
            index_tensor
        ]
        .detach()
        .cpu()
        .numpy()
    )

    return (
        np.asarray(
            indices_np,
            dtype=np.int64,
        ),
        y_true,
        predictions.detach().cpu().numpy(),
        probabilities.detach().cpu().numpy(),
    )


# ---------------------------------------------------------------------
# One complete benchmark run
# ---------------------------------------------------------------------

def train_one_run(
    dataset,
    seed,
    train_ratio,
    gpu=0,
):
    dataset = canonical_dataset_name(
        dataset
    )

    # For now this intentionally allows only datasets with
    # paper-defined CARE-GNN settings.
    config = get_caregnn_config(
        dataset
    )

    print(
        "=" * 72
    )
    print(
        "CARE-GNN | dataset={} | seed={} | train_ratio={}".format(
            dataset,
            seed,
            train_ratio,
        )
    )
    print(
        "=" * 72
    )

    set_seed(
        seed
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "Run CARE-GNN benchmark training on a GPU instance."
        )

    device = torch.device(
        "cuda:{}".format(gpu)
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(
            device
        ),
    )

    rng = np.random.default_rng(
        seed
    )

    loader_name = loader_dataset_name(
        dataset
    )

    data_prefix = os.path.join(
        PROJECT_ROOT,
        "dataset",
    )

    (
        features_np,
        labels_np,
        relation_arrays,
        labeled_from,
    ) = load_dga_arrays(
        loader_name,
        prefix=data_prefix,
    )

    features_np = np.asarray(
        features_np,
        dtype=np.float32,
    )

    labels_np = np.asarray(
        labels_np,
        dtype=np.int64,
    ).reshape(-1)

    num_nodes = int(
        features_np.shape[0]
    )

    if labeled_from is None:
        eligible = np.arange(
            num_nodes,
            dtype=np.int64,
        )
    else:
        eligible = np.arange(
            int(labeled_from),
            num_nodes,
            dtype=np.int64,
        )

    eligible = eligible[
        np.isin(
            labels_np[
                eligible
            ],
            [0, 1],
        )
    ]

    (
        train_np,
        valid_np,
        test_np,
        discarded_np,
        split_path,
    ) = load_split(
        dataset=dataset,
        seed=seed,
        train_ratio=train_ratio,
    )

    validate_split(
        train_np,
        valid_np,
        test_np,
    )

    validate_eligible_indices(
        train_np,
        valid_np,
        test_np,
        eligible,
    )

    feature_norm = (
        feature_normalisation_for(
            dataset
        )
    )

    features_np = normalize_features(
        features_np,
        feature_norm,
    )

    features = torch.as_tensor(
        features_np,
        dtype=torch.float32,
        device=device,
    )

    labels = torch.as_tensor(
        labels_np,
        dtype=torch.long,
        device=device,
    )

    relation_names = []

    relations = []

    for (
        relation_name,
        src,
        dst,
    ) in relation_arrays:

        relation_names.append(
            relation_name
        )

        indptr, indices = build_csr(
            src,
            dst,
            num_nodes,
        )

        relations.append(
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

    train_tensor = torch.as_tensor(
        train_np,
        dtype=torch.long,
        device=device,
    )

    valid_tensor = torch.as_tensor(
        valid_np,
        dtype=torch.long,
        device=device,
    )

    train_pos = train_np[
        labels_np[
            train_np
        ] == 1
    ]

    train_neg = train_np[
        labels_np[
            train_np
        ] == 0
    ]

    if len(train_pos) == 0:
        raise RuntimeError(
            "Training split contains zero fraud nodes."
        )

    if len(train_neg) == 0:
        raise RuntimeError(
            "Training split contains zero benign nodes."
        )

    num_classes = int(
        labels.max().item()
    ) + 1

    model = CAREGNN(
        features.size(1),
        config[
            "hidden_dim"
        ],
        len(
            relations
        ),
        n_class=num_classes,
        inter="GNN",
        step_size=config[
            "step_size"
        ],
        init_threshold=config[
            "initial_threshold"
        ],
    ).to(
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config[
            "learning_rate"
        ],
        weight_decay=config[
            "weight_decay"
        ],
    )

    print(
        "Nodes={} | Features={} | Relations={} | "
        "Train={} | Val={} | Test={}".format(
            num_nodes,
            int(
                features.size(1)
            ),
            relation_names,
            len(
                train_np
            ),
            len(
                valid_np
            ),
            len(
                test_np
            ),
        )
    )

    print(
        "Hidden={} | LR={} | Weight decay={} | "
        "Batch={} | lambda_1={} | RL step={} | "
        "Initial threshold={}".format(
            config[
                "hidden_dim"
            ],
            config[
                "learning_rate"
            ],
            config[
                "weight_decay"
            ],
            config[
                "batch_size"
            ],
            config[
                "lambda_1"
            ],
            config[
                "step_size"
            ],
            config[
                "initial_threshold"
            ],
        )
    )

    best_val_auc = -np.inf
    best_epoch = 0
    best_state = None
    best_thresholds = None

    wait = 1

    previous_relation_scores = None

    thresholds_log = [
        model.thresholds.tolist()
    ]

    start_time = (
        time.perf_counter()
    )

    epochs_executed = 0

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        epochs_executed = epoch

        model.train()

        num_negative_samples = min(
            len(
                train_neg
            ),
            int(
                round(
                    len(
                        train_pos
                    )
                    * config[
                        "under_sampling_scale"
                    ]
                )
            ),
        )

        sampled_negative = rng.choice(
            train_neg,
            size=num_negative_samples,
            replace=False,
        )

        training_nodes = np.concatenate(
            [
                train_pos,
                sampled_negative,
            ]
        )

        rng.shuffle(
            training_nodes
        )

        num_batches = max(
            1,
            math.ceil(
                len(
                    training_nodes
                )
                / config[
                    "batch_size"
                ]
            ),
        )

        epoch_loss = 0.0

        batch_relation_scores = []

        for batch_number in range(
            num_batches
        ):
            start = (
                batch_number
                * config[
                    "batch_size"
                ]
            )

            end = (
                (
                    batch_number
                    + 1
                )
                * config[
                    "batch_size"
                ]
            )

            batch_nodes = (
                training_nodes[
                    start:end
                ]
            )

            batch = torch.as_tensor(
                batch_nodes,
                dtype=torch.long,
                device=device,
            )

            batch_labels = labels[
                batch
            ]

            (
                gnn_logits,
                label_logits,
                relation_scores,
            ) = model(
                features,
                batch,
                relations,
                batch_labels,
            )

            classification_loss = (
                F.cross_entropy(
                    gnn_logits,
                    batch_labels,
                )
            )

            similarity_loss = (
                F.cross_entropy(
                    label_logits,
                    batch_labels,
                )
            )

            loss = (
                classification_loss
                + config[
                    "lambda_1"
                ]
                * similarity_loss
            )

            if not torch.isfinite(
                loss
            ):
                raise RuntimeError(
                    "CARE-GNN loss became non-finite "
                    "at epoch {}.".format(
                        epoch
                    )
                )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            epoch_loss += float(
                loss.detach().cpu()
            )

            batch_relation_scores.append(
                relation_scores
            )

        # -------------------------------------------------------------
        # Validation-only checkpoint selection
        # -------------------------------------------------------------

        validation_probabilities = (
            predict_proba(
                model=model,
                features=features,
                indices=valid_tensor,
                relations=relations,
                batch_size=EVAL_BATCH_SIZE,
            )
        )

        val_auc = validation_auc(
            labels[
                valid_tensor
            ],
            validation_probabilities,
        )

        if (
            val_auc
            > best_val_auc
        ):
            best_val_auc = float(
                val_auc
            )

            best_epoch = int(
                epoch
            )

            best_state = copy.deepcopy(
                model.state_dict()
            )

            best_thresholds = (
                model.thresholds
                .detach()
                .clone()
            )

            wait = 1

        else:
            wait += 1

        # -------------------------------------------------------------
        # CARE-GNN reinforcement-learning threshold update
        # -------------------------------------------------------------

        current_relation_scores = (
            epoch_relation_scores(
                batch_relation_scores,
                len(
                    relations
                ),
            )
        )

        if (
            previous_relation_scores
            is not None
        ):
            model.rl_update(
                previous_relation_scores,
                current_relation_scores,
            )

        previous_relation_scores = (
            current_relation_scores
        )

        thresholds_log.append(
            model.thresholds.tolist()
        )

        print(
            "Epoch {:03d} | "
            "Loss {:.6f} | "
            "Val AUROC {:.6f} | "
            "Best {:.6f} @ {} | "
            "Wait {}/{} | "
            "Thresholds {}".format(
                epoch,
                epoch_loss
                / num_batches,
                val_auc,
                best_val_auc,
                best_epoch,
                wait,
                EARLY_STOP,
                [
                    round(
                        float(value),
                        3,
                    )
                    for value
                    in model.thresholds.tolist()
                ],
            )
        )

        if wait >= EARLY_STOP:
            print(
                "Early stopping at epoch {}."
                .format(
                    epoch
                )
            )

            break

    train_time = (
        time.perf_counter()
        - start_time
    )

    if best_state is None:
        raise RuntimeError(
            "No best validation checkpoint "
            "was recorded."
        )

    # -------------------------------------------------------------
    # Restore validation-selected checkpoint
    # -------------------------------------------------------------

    model.load_state_dict(
        best_state
    )

    if best_thresholds is not None:
        with torch.no_grad():
            model.thresholds.copy_(
                best_thresholds.to(
                    model.thresholds.device
                )
            )

    # -------------------------------------------------------------
    # Test is evaluated only here, after model selection
    # -------------------------------------------------------------

    (
        test_order,
        y_test,
        test_predictions,
        test_probabilities,
    ) = evaluate_indices(
        model=model,
        features=features,
        labels=labels,
        indices_np=test_np,
        relations=relations,
    )

    test_metrics = (
        compute_classification_metrics(
            y_true=y_test,
            y_pred=test_predictions,
            y_prob=test_probabilities,
            sigmoid=False,
        )
    )

    test_metrics = dict(
        test_metrics
    )

    test_metrics.update(
        {
            "best_val_auroc": float(
                best_val_auc
            ),
            "best_epoch": int(
                best_epoch
            ),
            "epochs_executed": int(
                epochs_executed
            ),
            "train_time_seconds": float(
                train_time
            ),
        }
    )

    # Optional training metrics are produced only after model selection.
    (
        _,
        y_train,
        train_predictions,
        train_probabilities,
    ) = evaluate_indices(
        model=model,
        features=features,
        labels=labels,
        indices_np=train_np,
        relations=relations,
    )

    train_metrics = (
        compute_classification_metrics(
            y_true=y_train,
            y_pred=train_predictions,
            y_prob=train_probabilities,
            sigmoid=False,
        )
    )

    split_info = {
        "seed": int(
            seed
        ),
        "train_ratio": float(
            train_ratio
        ),
        "train_size": int(
            len(
                train_np
            )
        ),
        "validation_size": int(
            len(
                valid_np
            )
        ),
        "test_size": int(
            len(
                test_np
            )
        ),
        "discarded_size": int(
            len(
                discarded_np
            )
        ),
        "train_pool_ratio": 0.4,
        "validation_ratio": 0.2,
        "test_ratio": 0.4,
        "nested_training_subsets": True,
        "split_file": os.path.relpath(
            split_path,
            PROJECT_ROOT,
        ),
    }

    saved_config = {
        "model": MODEL_NAME,
        "dataset": dataset,
        "seed": int(
            seed
        ),
        "train_ratio": float(
            train_ratio
        ),

        "hidden_dim": int(
            config[
                "hidden_dim"
            ]
        ),

        "num_layers": int(
            config[
                "num_layers"
            ]
        ),

        "learning_rate": float(
            config[
                "learning_rate"
            ]
        ),

        "weight_decay": float(
            config[
                "weight_decay"
            ]
        ),

        "lambda_1": float(
            config[
                "lambda_1"
            ]
        ),

        "step_size": float(
            config[
                "step_size"
            ]
        ),

        "initial_threshold": float(
            config[
                "initial_threshold"
            ]
        ),

        "under_sampling_scale": float(
            config[
                "under_sampling_scale"
            ]
        ),

        "batch_size": int(
            config[
                "batch_size"
            ]
        ),

        "eval_batch_size": int(
            EVAL_BATCH_SIZE
        ),

        "optimizer": config[
            "optimizer"
        ],

        "inter_relation_aggregator": "GNN",

        "relations": relation_names,

        "feature_dim": int(
            features.size(1)
        ),

        "feature_preprocessing": (
            feature_norm
        ),

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

        "hard_prediction_rule": (
            "argmax_softmax"
        ),

        "best_valid_auroc": float(
            best_val_auc
        ),

        "best_epoch": int(
            best_epoch
        ),

        "epochs_executed": int(
            epochs_executed
        ),

        "final_thresholds": (
            model.thresholds.tolist()
        ),

        "thresholds_log": (
            thresholds_log
        ),

        "hyperparameter_source": (
            config[
                "source_note"
            ]
        ),

        "benchmark_stopping_note": (
            "Unified benchmark uses max 500 epochs "
            "and patience 25 with validation AUROC "
            "checkpoint selection."
        ),

        "test_selection_note": (
            "Test set is not used for training, "
            "early stopping, checkpoint selection, "
            "or hyperparameter selection."
        ),

        "platform": "Vast.ai",

        "gpu": torch.cuda.get_device_name(
            device
        ),

        "torch_version": (
            torch.__version__
        ),

        "cuda_version": (
            torch.version.cuda
        ),
    }

    out_dir = output_dir_for(
        dataset,
        seed,
        train_ratio,
    )

    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    save_run_outputs(
        save_dir=out_dir,
        model_name=MODEL_NAME,
        config=saved_config,
        split_info=split_info,
        test_metrics=test_metrics,
        model=model,
        train_metrics=train_metrics,
        predictions=test_predictions,
        probabilities=test_probabilities,
        train_indices=train_np,
        test_indices=test_order,
        valid_indices=valid_np,
        discarded_indices=discarded_np,
        y_train=y_train,
        y_test=y_test,
        train_time=train_time,
    )

    metadata_path = os.path.join(
        out_dir,
        "benchmark_metadata.json",
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file_handle:
        json.dump(
            {
                "model": MODEL_NAME,
                "dataset": dataset,
                "seed": int(
                    seed
                ),
                "train_ratio": float(
                    train_ratio
                ),
                "best_epoch": int(
                    best_epoch
                ),
                "best_validation_auroc": float(
                    best_val_auc
                ),
                "epochs_executed": int(
                    epochs_executed
                ),
                "train_time_seconds": float(
                    train_time
                ),
                "canonical_split_used": True,
                "nested_training_subsets": True,
                "test_evaluated_after_model_selection": True,
                "test_used_during_training": False,
                "test_used_for_early_stopping": False,
                "test_used_for_checkpoint_selection": False,
                "max_epochs": MAX_EPOCHS,
                "early_stop": EARLY_STOP,
                "checkpoint_metric": "validation_AUROC",
                "checkpoint_rule": "strict_greater_than",
                "epoch_resume_enabled": False,
            },
            file_handle,
            indent=2,
        )

    print(
        "=" * 72
    )

    print(
        "FINAL TEST RESULTS"
    )

    for key in [
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
    ]:
        if key not in test_metrics:
            continue

        value = test_metrics[
            key
        ]

        if isinstance(
            value,
            (
                float,
                np.floating,
            ),
        ):
            print(
                "{:25s}: {:.6f}".format(
                    key,
                    value,
                )
            )

        else:
            print(
                "{:25s}: {}".format(
                    key,
                    value,
                )
            )

    print(
        "Saved:",
        out_dir,
    )

    return (
        test_metrics,
        out_dir,
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "CARE-GNN unified benchmark runner"
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "yelp",
            "yelpchi",
            "amazon",
        ],
        help=(
            "Currently restricted to CARE-GNN "
            "paper-supported datasets. "
            "Unsupported datasets will be enabled "
            "after validation-only tuning."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        choices=ALLOWED_SEEDS,
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        choices=ALLOWED_TRAIN_RATIOS,
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Run all 5 seeds x 4 training-label budgets."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip runs that already contain a complete "
            "saved benchmark output. An interrupted "
            "individual run is restarted from the beginning."
        ),
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="CUDA device index.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    dataset = canonical_dataset_name(
        args.dataset
    )

    if args.all:

        for train_ratio in (
            ALLOWED_TRAIN_RATIOS
        ):

            for seed in (
                ALLOWED_SEEDS
            ):

                if (
                    args.resume
                    and run_complete(
                        dataset,
                        seed,
                        train_ratio,
                    )
                ):
                    print(
                        "SKIP complete: "
                        "{} seed={} train_ratio={}".format(
                            dataset,
                            seed,
                            train_ratio,
                        )
                    )

                    continue

                train_one_run(
                    dataset=dataset,
                    seed=seed,
                    train_ratio=train_ratio,
                    gpu=args.gpu,
                )

        return

    if (
        args.seed is None
        or args.train_ratio is None
    ):
        raise SystemExit(
            "Single-run mode requires both "
            "--seed and --train-ratio, "
            "or use --all."
        )

    if (
        args.resume
        and run_complete(
            dataset,
            args.seed,
            args.train_ratio,
        )
    ):
        print(
            "SKIP complete: "
            "{} seed={} train_ratio={}".format(
                dataset,
                args.seed,
                args.train_ratio,
            )
        )

        return

    train_one_run(
        dataset=dataset,
        seed=args.seed,
        train_ratio=args.train_ratio,
        gpu=args.gpu,
    )


if __name__ == "__main__":
    main()