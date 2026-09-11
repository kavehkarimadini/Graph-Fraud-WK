"""SEC-GFD runner for the unified graph-fraud benchmark."""

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
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from algorithms.SEC_GFD.model.SECGFD import SECGFD
from algorithms.SEC_GFD.secgfd_config import (
    canonical_dataset_name,
    get_secgfd_config,
)
from algorithms.SEC_GFD.secgfd_data import load_secgfd_dataset
from utils.evaluation import (
    compute_classification_metrics,
    save_run_outputs,
)


MODEL_NAME = "SEC_GFD"

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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    dgl.seed(seed)

    try:
        dgl.random.seed(seed)
    except Exception:
        pass

    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()

    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])

    if (
        torch.cuda.is_available()
        and "cuda" in state
        and state["cuda"] is not None
    ):
        torch.cuda.set_rng_state_all(state["cuda"])


def ratio_to_key(train_ratio):
    return {
        0.1: "train_10",
        0.2: "train_20",
        0.3: "train_30",
        0.4: "train_40",
    }[float(train_ratio)]


def output_dir_for(dataset, seed, train_ratio):
    pct = int(round(train_ratio * 100))

    return os.path.join(
        PROJECT_ROOT,
        "results",
        MODEL_NAME,
        dataset,
        "seed_{}".format(seed),
        "train_{}".format(pct),
    )


def run_complete(dataset, seed, train_ratio):
    out = output_dir_for(
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
            os.path.join(out, name)
        )
        for name in required
    )


def training_state_path(dataset, seed, train_ratio):
    return os.path.join(
        output_dir_for(
            dataset,
            seed,
            train_ratio,
        ),
        "training_state.pt",
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
            "Benchmark split not found: {}\n"
            "Run the benchmark split generator first.".format(
                path
            )
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
    tr = set(train_idx.tolist())
    va = set(valid_idx.tolist())
    te = set(test_idx.tolist())

    if tr & va or tr & te or va & te:
        raise RuntimeError(
            "Split error: train/validation/test sets overlap."
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
                "{} split contains nodes that are not "
                "eligible for SEC-GFD.".format(name)
            )


def build_model(
    graph,
    in_features,
    config,
    device,
):
    model = SECGFD(
        in_dim=in_features,
        hid_dim=config["hidden_dim"],
        out_dim=2,
        graph=graph,
        d=config["order"],
        high_order=config["high_order"],
    )

    return model.to(device)


def corrected_environment_loss(
    embeddings,
    features,
    labels,
    train_idx,
):
    """
    SEC-GFD environmental loss.

    Two corrections are applied relative to the released repository:

    1. Actual global training-node IDs are used rather than positions
       inside labels[idx_train].

    2. exp(cosine_similarity) is used before class-wise averaging,
       matching the environmental-constraint equation in the paper.
       This also guarantees positive values inside the logarithm.
    """

    train_idx = torch.as_tensor(
        train_idx,
        dtype=torch.long,
        device=labels.device,
    )

    train_labels = labels[
        train_idx
    ]

    normal_nodes = train_idx[
        train_labels == 0
    ]

    anomaly_nodes = train_idx[
        train_labels == 1
    ]

    if normal_nodes.numel() == 0:
        raise RuntimeError(
            "Training split contains no benign nodes."
        )

    if anomaly_nodes.numel() == 0:
        raise RuntimeError(
            "Training split contains no fraud nodes."
        )

    similarity = F.cosine_similarity(
        features,
        embeddings,
        dim=1,
    )

    normal_similarity = torch.mean(
        torch.exp(
            similarity[normal_nodes]
        )
    )

    anomaly_similarity = torch.mean(
        torch.exp(
            similarity[anomaly_nodes]
        )
    )

    loss = -torch.log(
        normal_similarity
        / anomaly_similarity
    )

    if not torch.isfinite(loss):
        raise RuntimeError(
            "SEC-GFD environmental loss became non-finite. "
            "normal_similarity={} anomaly_similarity={}".format(
                float(
                    normal_similarity.detach().cpu()
                ),
                float(
                    anomaly_similarity.detach().cpu()
                ),
            )
        )

    return loss


def evaluate_indices(
    model,
    features,
    labels,
    node_indices,
):
    model.eval()

    index_tensor = torch.as_tensor(
        node_indices,
        dtype=torch.long,
        device=labels.device,
    )

    with torch.no_grad():
        logits, _ = model(
            features
        )

        probabilities = torch.softmax(
            logits[index_tensor],
            dim=1,
        )

        predictions = torch.argmax(
            probabilities,
            dim=1,
        )

    return (
        np.asarray(
            node_indices,
            dtype=np.int64,
        ),
        labels[
            index_tensor
        ].detach().cpu().numpy(),
        predictions.detach().cpu().numpy(),
        probabilities.detach().cpu().numpy(),
    )


def save_training_state(
    path,
    epoch,
    model,
    optimizer,
    best_val_auc,
    best_epoch,
    best_state,
    wait,
    elapsed_seconds,
):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_auc": float(best_val_auc),
        "best_epoch": int(best_epoch),
        "best_state_dict": best_state,
        "wait": int(wait),
        "elapsed_seconds": float(
            elapsed_seconds
        ),
        "rng_state": capture_rng_state(),
    }

    temporary_path = (
        path + ".tmp"
    )

    torch.save(
        payload,
        temporary_path,
    )

    os.replace(
        temporary_path,
        path,
    )


def load_training_state(
    path,
    model,
    optimizer,
    device,
):
    try:
        state = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        state = torch.load(
            path,
            map_location=device,
        )

    model.load_state_dict(
        state["model_state_dict"]
    )

    optimizer.load_state_dict(
        state["optimizer_state_dict"]
    )

    restore_rng_state(
        state["rng_state"]
    )

    return state


def train_one_run(
    dataset,
    seed,
    train_ratio,
    resume=False,
):
    dataset = canonical_dataset_name(
        dataset
    )

    config = get_secgfd_config(
        dataset
    )

    print("=" * 70)
    print(
        "SEC-GFD | dataset={} | seed={} | train_ratio={}".format(
            dataset,
            seed,
            train_ratio,
        )
    )
    print("=" * 70)

    set_seed(seed)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "Run benchmark training on a GPU instance."
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
        features,
        labels,
        eligible,
    ) = load_secgfd_dataset(
        dataset,
        PROJECT_ROOT,
    )

    print(
        "Nodes={} Edges={} Features={}".format(
            graph.num_nodes(),
            graph.num_edges(),
            features.shape[1],
        )
    )

    (
        train_idx,
        valid_idx,
        test_idx,
        discarded_idx,
        split_path,
    ) = load_split(
        dataset,
        seed,
        train_ratio,
    )

    validate_split(
        train_idx,
        valid_idx,
        test_idx,
    )

    validate_eligible_indices(
        train_idx,
        valid_idx,
        test_idx,
        eligible,
    )

    print(
        "Train={} Val={} Test={} Discarded={}".format(
            len(train_idx),
            len(valid_idx),
            len(test_idx),
            len(discarded_idx),
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

    model = build_model(
        graph=graph,
        in_features=int(
            features.shape[1]
        ),
        config=config,
        device=device,
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

    train_tensor = torch.as_tensor(
        train_idx,
        dtype=torch.long,
        device=device,
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

    if fraud_count == 0:
        raise RuntimeError(
            "Training split contains zero fraud nodes."
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
        device=device,
    )

    print(
        "Cross-entropy fraud weight: {:.6f}".format(
            fraud_weight
        )
    )

    out_dir = output_dir_for(
        dataset,
        seed,
        train_ratio,
    )

    state_path = training_state_path(
        dataset,
        seed,
        train_ratio,
    )

    best_val_auc = -np.inf
    best_epoch = 0
    best_state = None

    # Preserve the same patience-counter convention
    # used by the completed PMP benchmark.
    wait = 1

    start_epoch = 1
    elapsed_before_resume = 0.0

    if (
        resume
        and os.path.exists(
            state_path
        )
    ):
        state = load_training_state(
            state_path,
            model,
            optimizer,
            device,
        )

        start_epoch = (
            int(
                state["epoch"]
            )
            + 1
        )

        best_val_auc = float(
            state[
                "best_val_auc"
            ]
        )

        best_epoch = int(
            state[
                "best_epoch"
            ]
        )

        best_state = state[
            "best_state_dict"
        ]

        wait = int(
            state[
                "wait"
            ]
        )

        elapsed_before_resume = float(
            state.get(
                "elapsed_seconds",
                0.0,
            )
        )

        print(
            "RESUME from epoch {} | "
            "best val AUROC {:.6f} @ {} | "
            "wait {}/{}".format(
                start_epoch - 1,
                best_val_auc,
                best_epoch,
                wait,
                EARLY_STOP,
            )
        )

    segment_start = (
        time.perf_counter()
    )

    actual_epochs = (
        start_epoch - 1
    )

    if wait < EARLY_STOP:

        for epoch in range(
            start_epoch,
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
                + config[
                    "lambda_env"
                ]
                * environment_loss
            )

            if not torch.isfinite(
                loss
            ):
                raise RuntimeError(
                    "SEC-GFD total training loss "
                    "became non-finite at epoch {}."
                    .format(epoch)
                )

            loss.backward()

            optimizer.step()

            (
                _,
                y_val,
                _,
                val_probabilities,
            ) = evaluate_indices(
                model,
                features,
                labels,
                valid_idx,
            )

            val_auc = roc_auc_score(
                y_val,
                val_probabilities[
                    :, 1
                ],
            )

            if val_auc > best_val_auc:

                best_val_auc = float(
                    val_auc
                )

                best_epoch = int(
                    epoch
                )

                best_state = (
                    copy.deepcopy(
                        model.state_dict()
                    )
                )

                wait = 1

            else:
                wait += 1

            elapsed_now = (
                elapsed_before_resume
                + time.perf_counter()
                - segment_start
            )

            save_training_state(
                path=state_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                best_val_auc=best_val_auc,
                best_epoch=best_epoch,
                best_state=best_state,
                wait=wait,
                elapsed_seconds=elapsed_now,
            )

            print(
                "Epoch {:03d} | "
                "Loss {:.6f} | "
                "CE {:.6f} | "
                "Env {:.6f} | "
                "Val AUROC {:.6f} | "
                "Best {:.6f} @ {} | "
                "Wait {}/{}".format(
                    epoch,
                    float(
                        loss.detach().cpu()
                    ),
                    float(
                        classification_loss.detach().cpu()
                    ),
                    float(
                        environment_loss.detach().cpu()
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
                    "Early stopping at epoch {}.".format(
                        epoch
                    )
                )
                break

    else:
        print(
            "Early-stopping condition was already "
            "reached in the recovered checkpoint."
        )

    train_time = (
        elapsed_before_resume
        + time.perf_counter()
        - segment_start
    )

    if best_state is None:
        raise RuntimeError(
            "No best validation checkpoint was recorded."
        )

    model.load_state_dict(
        best_state
    )

    (
        test_order,
        y_test,
        test_predictions,
        test_probabilities,
    ) = evaluate_indices(
        model,
        features,
        labels,
        test_idx,
    )

    test_metrics = (
        compute_classification_metrics(
            y_true=y_test,
            y_pred=test_predictions,
            y_prob=test_probabilities,
            sigmoid=False,
        )
    )

    test_metrics.update(
        {
            "best_epoch": int(
                best_epoch
            ),
            "best_val_auroc": float(
                best_val_auc
            ),
            "train_time_seconds": float(
                train_time
            ),
            "epochs_executed": int(
                actual_epochs
            ),
        }
    )

    split_info = {
        "seed": int(seed),
        "train_ratio": float(
            train_ratio
        ),
        "train_size": len(
            train_idx
        ),
        "validation_size": len(
            valid_idx
        ),
        "test_size": len(
            test_idx
        ),
        "discarded_size": len(
            discarded_idx
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

        "optimizer": config[
            "optimizer"
        ],

        "learning_rate": config[
            "learning_rate"
        ],

        "weight_decay": config[
            "weight_decay"
        ],

        "hidden_dim": config[
            "hidden_dim"
        ],

        "order_C": config[
            "order"
        ],

        "high_order": config[
            "high_order"
        ],

        "lambda_env": config[
            "lambda_env"
        ],

        "loss": (
            "weighted CrossEntropy + "
            "lambda_env * environmental loss"
        ),

        "class_weight_rule": (
            "fraud_weight = "
            "num_benign_train / "
            "num_fraud_train"
        ),

        "fraud_class_weight": float(
            fraud_weight
        ),

        "max_epochs": MAX_EPOCHS,

        "early_stop": EARLY_STOP,

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

        "feature_preprocessing": (
            "SEC-GFD raw feature pathway; "
            "no PMP row normalization"
        ),

        "environment_loss_indexing": (
            "corrected to use actual global "
            "training-node IDs"
        ),

        "environment_loss_similarity": (
            "exp(cosine_similarity) before "
            "class-wise averaging, matching "
            "SEC-GFD paper environmental loss"
        ),

        "official_code_high_order_note": (
            "Released SEC-GFD main.py uses "
            "high_order=2; preserved here."
        ),

        "hyperparameter_source": config[
            "source_note"
        ],

        "benchmark_stopping_note": (
            "Unified benchmark uses max 500 "
            "epochs and patience 25 instead of "
            "SEC-GFD paper's fixed 100 epochs."
        ),

        "platform": "Vast.ai",

        "gpu": torch.cuda.get_device_name(
            0
        ),

        "torch_version": torch.__version__,

        "dgl_version": dgl.__version__,

        "cuda_version": torch.version.cuda,
    }

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
        predictions=test_predictions,
        probabilities=test_probabilities,
        test_indices=test_order,
        train_indices=train_idx,
        valid_indices=valid_idx,
        discarded_indices=discarded_idx,
        y_test=y_test,
        y_train=labels[
            train_tensor
        ].detach().cpu().numpy(),
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
    ) as f:

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
                    actual_epochs
                ),
                "train_time_seconds": float(
                    train_time
                ),
                "test_evaluated_after_model_selection": True,
                "test_used_during_training": False,
                "epoch_resume_enabled": True,
                "environment_loss_indexing_corrected": True,
                "environment_loss_exp_similarity": True,
            },
            f,
            indent=2,
        )

    if os.path.exists(
        state_path
    ):
        os.remove(
            state_path
        )

    print("=" * 70)
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

        if key in test_metrics:

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


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "SEC-GFD unified benchmark runner"
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "amazon",
            "yelp",
            "yelpchi",
            "tfinance",
            "tsocial",
        ],
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
            "Run all 5 seeds x "
            "4 train ratios."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip complete runs and recover "
            "an interrupted run from its "
            "latest epoch checkpoint."
        ),
    )

    return parser.parse_args()


def main():

    args = parse_args()

    dataset = canonical_dataset_name(
        args.dataset
    )

    if args.all:

        for ratio in ALLOWED_TRAIN_RATIOS:

            for seed in ALLOWED_SEEDS:

                if (
                    args.resume
                    and run_complete(
                        dataset,
                        seed,
                        ratio,
                    )
                ):
                    print(
                        "SKIP complete: "
                        "{} seed={} "
                        "train_ratio={}".format(
                            dataset,
                            seed,
                            ratio,
                        )
                    )
                    continue

                train_one_run(
                    dataset=dataset,
                    seed=seed,
                    train_ratio=ratio,
                    resume=args.resume,
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
        resume=args.resume,
    )


if __name__ == "__main__":
    main()