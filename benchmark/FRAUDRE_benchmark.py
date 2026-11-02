"""FRAUDRE runner for the unified graph-fraud benchmark."""

from __future__ import print_function

import argparse
import copy
import json
import math
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from algorithms.FRAUDRE.FRAUDRE import (
    FRAUDRE,
    prior_adjusted_loss,
    sample_adjacency,
)

from fraudre_config import (
    canonical_dataset_name,
    get_fraudre_config,
)

from utils.data_loader import load_dga_arrays
from utils.evaluation import (
    compute_classification_metrics,
    save_run_outputs,
)


MODEL_NAME = "FRAUDRE"

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

def set_seed(seed):
    import random

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



def ratio_to_key(train_ratio):
    return {
        0.1: "train_10",
        0.2: "train_20",
        0.3: "train_30",
        0.4: "train_40",
    }[float(train_ratio)]


def loader_dataset_name(dataset):
    if dataset == "yelp":
        return "yelpchi"

    if dataset == "fdcompcn":
        return "comp"

    return dataset


def output_dir_for(dataset, seed, train_ratio):
    pct = int(round(float(train_ratio) * 100))

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
            os.path.join(out_dir, filename)
        )
        for filename in required
    )


def load_split(dataset, seed, train_ratio):
    path = os.path.join(
        PROJECT_ROOT,
        "splits",
        dataset,
        "seed_{}.npz".format(seed),
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            "Canonical benchmark split not found: {}\n"
            "Do not generate a new split inside FRAUDRE."
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


def validate_split(train_idx, valid_idx, test_idx):
    tr = set(train_idx.tolist())
    va = set(valid_idx.tolist())
    te = set(test_idx.tolist())

    if tr & va or tr & te or va & te:
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

    for name, indices in [
        ("train", train_idx),
        ("validation", valid_idx),
        ("test", test_idx),
    ]:
        invalid = set(indices.tolist()) - eligible_set

        if invalid:
            raise RuntimeError(
                "{} split contains ineligible nodes."
                .format(name)
            )


def row_normalize_features(features):
    """
    Match the official FRAUDRE row-normalisation pathway.
    """
    features = np.asarray(
        features,
        dtype=np.float32,
    )

    rowsum = np.asarray(
        features.sum(axis=1),
        dtype=np.float64,
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        r_inv = np.power(
            rowsum,
            -1.0,
        )

    r_inv[
        ~np.isfinite(r_inv)
    ] = 0.0

    normalized = (
        features
        * r_inv[:, None]
    )

    return np.ascontiguousarray(
        normalized,
        dtype=np.float32,
    )


@torch.no_grad()
def predict_scores(
    model,
    features,
    indices,
    relations,
    batch_size,
):
    """
    FRAUDRE official evaluation applies sigmoid to the two output logits.
    """
    model.eval()

    scores = []

    for start in range(
        0,
        indices.numel(),
        batch_size,
    ):
        batch = indices[
            start:start + batch_size
        ]

        logits = model(
            features,
            batch,
            relations,
        )

        scores.append(
            torch.sigmoid(
                logits
            )
        )

    return torch.cat(
        scores,
        dim=0,
    )


def validation_auc(
    labels,
    scores,
):
    y_true = (
        labels
        .detach()
        .cpu()
        .numpy()
    )

    if len(np.unique(y_true)) < 2:
        raise RuntimeError(
            "Validation split contains fewer than two classes."
        )

    return float(
        roc_auc_score(
            y_true,
            scores[:, 1]
            .detach()
            .cpu()
            .numpy(),
        )
    )


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

    scores = predict_scores(
        model=model,
        features=features,
        indices=index_tensor,
        relations=relations,
        batch_size=EVAL_BATCH_SIZE,
    )

    predictions = torch.argmax(
        scores,
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
        scores.detach().cpu().numpy(),
    )


def train_one_run(
    dataset,
    seed,
    train_ratio,
    gpu=0,
    force=False,
):
    dataset = canonical_dataset_name(
        dataset
    )

    if seed not in ALLOWED_SEEDS:
        raise ValueError(
            "Seed {} is not one of the fixed benchmark seeds."
            .format(seed)
        )

    if float(train_ratio) not in ALLOWED_TRAIN_RATIOS:
        raise ValueError(
            "train_ratio {} is not one of the fixed benchmark ratios."
            .format(train_ratio)
        )

    if (
        not force
        and run_complete(
            dataset,
            seed,
            train_ratio,
        )
    ):
        print(
            "SKIP complete run: {} seed={} train_ratio={}"
            .format(
                dataset,
                seed,
                train_ratio,
            )
        )
        return

    config = get_fraudre_config(
        dataset
    )

    print("=" * 72)
    print(
        "FRAUDRE | dataset={} | seed={} | train_ratio={}"
        .format(
            dataset,
            seed,
            train_ratio,
        )
    )
    print("=" * 72)

    set_seed(
        seed
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
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

    feature_normalisation = config["feature_normalisation"]

    if feature_normalisation == "row":
        features_np = row_normalize_features(
            features_np
        )

    elif feature_normalisation == "standard":
        from sklearn.preprocessing import StandardScaler

        features_np = StandardScaler().fit_transform(
            features_np
        ).astype(np.float32)

    else:
        raise RuntimeError(
            "Unsupported FRAUDRE feature_normalisation: {}"
            .format(feature_normalisation)
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

    for relation_id, (
        relation_name,
        src,
        dst,
    ) in enumerate(
        relation_arrays
    ):
        relation_names.append(
            relation_name
        )

        indptr, indices = sample_adjacency(
            src=src,
            dst=dst,
            n=num_nodes,
            max_neighbors=config[
                "max_neighbors"
            ],
            seed=seed + relation_id,
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

    if len(relations) != 3:
        raise RuntimeError(
            "Expected 3 FRAUDRE relations for {}, found {}."
            .format(
                dataset,
                len(relations),
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

    num_classes = int(
        labels.max().item()
    ) + 1

    if num_classes != 2:
        raise RuntimeError(
            "FRAUDRE benchmark expects binary labels."
        )

    class_counts = np.bincount(
        labels_np[
            train_np
        ],
        minlength=num_classes,
    )

    if np.any(class_counts == 0):
        raise RuntimeError(
            "Training split contains a class with zero nodes."
        )

    prior = (
        class_counts.astype(
            np.float64
        )
        / float(
            len(train_np)
        )
    )

    log_prior = torch.log(
        torch.as_tensor(
            prior + 1e-8,
            dtype=torch.float32,
            device=device,
        )
    )

    model = FRAUDRE(
        in_dim=int(
            features.size(1)
        ),
        n_relations=len(
            relations
        ),
        embed_dim=config[
            "embed_dim"
        ],
        n_layers=config[
            "n_layers"
        ],
        hidden_dim=config[
            "hidden_dim"
        ],
        n_class=num_classes,
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
        "Train={} | Val={} | Test={}"
        .format(
            num_nodes,
            int(
                features.size(1)
            ),
            relation_names,
            len(train_np),
            len(valid_np),
            len(test_np),
        )
    )

    print(
        "Embed={} | Layers={} | Hidden={} | "
        "Batch={} | LR={} | Weight decay={} | "
        "Max neighbors={} | Prior={}"
        .format(
            config[
                "embed_dim"
            ],
            config[
                "n_layers"
            ],
            config[
                "hidden_dim"
            ],
            config[
                "batch_size"
            ],
            config[
                "learning_rate"
            ],
            config[
                "weight_decay"
            ],
            config[
                "max_neighbors"
            ],
            [
                round(
                    float(value),
                    6,
                )
                for value in prior
            ],
        )
    )

    best_val_auc = -np.inf
    best_epoch = 0
    best_state = None

    # Locked unified benchmark mechanism:
    # improvement -> wait resets to 1.
    wait = 1

    start_time = time.perf_counter()
    epochs_executed = 0

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        epochs_executed = epoch

        model.train()

        training_nodes = rng.permutation(
            train_np
        )

        num_batches = max(
            1,
            math.ceil(
                len(training_nodes)
                / config[
                    "batch_size"
                ]
            ),
        )

        epoch_loss = 0.0

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

            batch_nodes = training_nodes[
                start:end
            ]

            batch = torch.as_tensor(
                batch_nodes,
                dtype=torch.long,
                device=device,
            )

            logits = model(
                features,
                batch,
                relations,
            )

            loss = prior_adjusted_loss(
                logits,
                labels[
                    batch
                ],
                log_prior,
            )

            if not torch.isfinite(
                loss
            ):
                raise RuntimeError(
                    "FRAUDRE loss became non-finite at epoch {}."
                    .format(epoch)
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(
                loss.detach().cpu()
            )

        # ---------------------------------------------------------
        # Validation only. Test is not accessed here.
        # ---------------------------------------------------------

        validation_scores = predict_scores(
            model=model,
            features=features,
            indices=valid_tensor,
            relations=relations,
            batch_size=EVAL_BATCH_SIZE,
        )

        val_auc = validation_auc(
            labels[
                valid_tensor
            ],
            validation_scores,
        )

        if val_auc > best_val_auc:
            best_val_auc = float(
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
            "Epoch {:03d} | Loss {:.6f} | "
            "Val AUROC {:.6f} | Best {:.6f} @ {} | "
            "Wait {}/{}"
            .format(
                epoch,
                epoch_loss
                / num_batches,
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

    train_time = (
        time.perf_counter()
        - start_time
    )

    if best_state is None:
        raise RuntimeError(
            "No best validation checkpoint was recorded."
        )

    # ---------------------------------------------------------
    # Restore best validation-selected checkpoint.
    # ---------------------------------------------------------

    model.load_state_dict(
        best_state
    )

    # ---------------------------------------------------------
    # Test is evaluated only after model selection.
    # ---------------------------------------------------------

    (
        test_order,
        y_test,
        test_predictions,
        test_scores,
    ) = evaluate_indices(
        model=model,
        features=features,
        labels=labels,
        indices_np=test_np,
        relations=relations,
    )

    test_metrics = compute_classification_metrics(
        y_true=y_test,
        y_pred=test_predictions,
        y_prob=test_scores,
        sigmoid=False,
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

    (
        _,
        y_train,
        train_predictions,
        train_scores,
    ) = evaluate_indices(
        model=model,
        features=features,
        labels=labels,
        indices_np=train_np,
        relations=relations,
    )

    train_metrics = compute_classification_metrics(
        y_true=y_train,
        y_pred=train_predictions,
        y_prob=train_scores,
        sigmoid=False,
    )

    split_info = {
        "seed": int(
            seed
        ),
        "train_ratio": float(
            train_ratio
        ),
        "train_size": int(
            len(train_np)
        ),
        "validation_size": int(
            len(valid_np)
        ),
        "test_size": int(
            len(test_np)
        ),
        "discarded_size": int(
            len(discarded_np)
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
        "seed": int(seed),
        "train_ratio": float(
            train_ratio
        ),
        "embed_dim": int(
            config[
                "embed_dim"
            ]
        ),
        "n_layers": int(
            config[
                "n_layers"
            ]
        ),
        "hidden_dim": int(
            config[
                "hidden_dim"
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
        "max_neighbors": int(
            config[
                "max_neighbors"
            ]
        ),
        "optimizer": config[
            "optimizer"
        ],
        "relations": relation_names,
        "feature_dim": int(
            features.size(1)
        ),
        "feature_preprocessing": (
            "official_FRAUDRE_row_normalisation"
        ),
        "loss": (
            "prior_adjusted_cross_entropy"
        ),
        "train_prior": [
            float(value)
            for value in prior
        ],
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
        "inference_score_rule": (
            "sigmoid_on_each_output_logit"
        ),
        "hard_prediction_rule": (
            "argmax_sigmoid_scores"
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
        "hyperparameter_source": (
            config[
                "source_note"
            ]
        ),
        "benchmark_stopping_note": (
            "Unified benchmark uses max 500 epochs "
            "and early-stop counter 25. Strict validation "
            "AUROC improvement saves the checkpoint and "
            "resets the counter to 1."
        ),
        "test_selection_note": (
            "Test set is not used for training, early stopping, "
            "checkpoint selection, or hyperparameter selection."
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
        probabilities=test_scores,
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
    ) as handle:
        json.dump(
            {
                "model": MODEL_NAME,
                "dataset": dataset,
                "seed": int(seed),
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
                "early_stop_counter_reset": 1,
                "checkpoint_metric": "validation_AUROC",
                "checkpoint_rule": "strict_greater_than",
                "epoch_resume_enabled": False,
            },
            handle,
            indent=2,
        )

    print("=" * 72)
    print("FINAL TEST RESULTS")

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
        if key in test_metrics:
            print(
                "{}: {}".format(
                    key,
                    test_metrics[key],
                )
            )

    print(
        "Saved:",
        out_dir,
    )
    print("=" * 72)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Unified FRAUDRE benchmark runner"
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--train_ratio",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--all",
        action="store_true",
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    dataset = canonical_dataset_name(
        args.dataset
    )

    if args.all:
        for train_ratio in ALLOWED_TRAIN_RATIOS:
            for seed in ALLOWED_SEEDS:
                train_one_run(
                    dataset=dataset,
                    seed=seed,
                    train_ratio=train_ratio,
                    gpu=args.gpu,
                    force=args.force,
                )

        return

    if args.seed is None:
        raise ValueError(
            "--seed is required unless --all is used."
        )

    if args.train_ratio is None:
        raise ValueError(
            "--train_ratio is required unless --all is used."
        )

    train_one_run(
        dataset=dataset,
        seed=args.seed,
        train_ratio=args.train_ratio,
        gpu=args.gpu,
        force=args.force,
    )


if __name__ == "__main__":
    main()
