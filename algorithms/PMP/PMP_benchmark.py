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
from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from algorithms.PMP.LASAGE_S import LASAGE_S
from utils.evaluation import compute_classification_metrics, save_run_outputs


MODEL_NAME = "PMP"
DATASET_NAME = "amazon"

MAX_EPOCHS = 500
EARLY_STOP = 25
LEARNING_RATE = 0.01

# PMP paper settings for Amazon.
HIDDEN_DIM = 128
DROPOUT = 0.6
NUM_LAYERS = 1
BATCH_SIZE = 128
WEIGHT_DECAY = 0.0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)


def load_amazon():
    from dgl.data import FraudDataset

    dataset_path = os.path.join(PROJECT_ROOT, "dataset")

    ds = FraudDataset(
        name="amazon",
        raw_dir=dataset_path,
        train_size=0.4,
        val_size=0.2,
        random_seed=717,
        force_reload=False,
    )

    graph = ds[0]

    features = graph.ndata["feature"].float()
    labels = graph.ndata["label"].long()

    return graph, features, labels


def load_split(seed, train_ratio):
    split_path = os.path.join(
        PROJECT_ROOT,
        "splits",
        "amazon",
        f"seed_{seed}.npz",
    )

    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"Split file not found: {split_path}"
        )

    data = np.load(split_path)

    ratio_key = {
        0.4: "train_40",
        0.3: "train_30",
        0.2: "train_20",
        0.1: "train_10",
    }[train_ratio]

    train_idx = data[ratio_key].astype(np.int64)
    valid_idx = data["val"].astype(np.int64)
    test_idx = data["test"].astype(np.int64)

    return train_idx, valid_idx, test_idx


def build_model(graph, features, device):
    relations = list(graph.etypes)

    model = LASAGE_S(
        in_dim=features.shape[1],
        hid_dim=HIDDEN_DIM,
        out_dim=2,
        n_layer=NUM_LAYERS,
        relations=relations,
        dropout=DROPOUT,
        agg="mean",
        relation_agg="cat",
        proj=True,
        num_trans=1,
        resi=0.2,
    )

    return model.to(device)


def make_loader(graph, node_indices, shuffle):
    sampler = MultiLayerFullNeighborSampler(NUM_LAYERS)

    loader = DataLoader(
        graph,
        torch.tensor(node_indices, dtype=torch.long),
        sampler,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
    )

    return loader


def run_inference(model, graph, features, labels, node_indices, device):
    model.eval()

    loader = make_loader(graph, node_indices, shuffle=False)

    all_logits = []
    all_labels = []
    all_indices = []

    with torch.no_grad():
        for input_nodes, output_nodes, blocks in loader:
            blocks = [b.to(device) for b in blocks]

            batch_features = features[input_nodes].to(device)

            logits = model(
                blocks,
                batch_features,
            )

            all_logits.append(logits.detach().cpu())
            all_labels.append(labels[output_nodes].detach().cpu())
            all_indices.append(output_nodes.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    y_true = torch.cat(all_labels, dim=0).numpy()
    indices = torch.cat(all_indices, dim=0).numpy()

    probabilities = torch.softmax(logits, dim=1).numpy()
    predictions = np.argmax(probabilities, axis=1)

    return indices, y_true, predictions, probabilities


def train_one_run(seed, train_ratio):
    set_seed(seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this benchmark.")

    device = torch.device("cuda:0")

    graph, features, labels = load_amazon()

    train_idx, valid_idx, test_idx = load_split(
        seed,
        train_ratio,
    )

    graph = graph.to("cpu")

    model = build_model(
        graph,
        features,
        device,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    train_loader = make_loader(
        graph,
        train_idx,
        shuffle=True,
    )

    best_val_auc = -np.inf
    best_state = None
    best_epoch = 0

    # Team protocol: counter resets to 1 on improvement.
    wait = 1

    start_time = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        epoch_loss = 0.0
        batch_count = 0

        for input_nodes, output_nodes, blocks in train_loader:
            blocks = [b.to(device) for b in blocks]

            batch_features = features[input_nodes].to(device)
            batch_labels = labels[output_nodes].to(device)

            optimizer.zero_grad()

            logits = model(
                blocks,
                batch_features,
            )

            loss = F.cross_entropy(
                logits,
                batch_labels,
            )

            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            batch_count += 1

        (
            _,
            val_true,
            val_pred,
            val_prob,
        ) = run_inference(
            model,
            graph,
            features,
            labels,
            valid_idx,
            device,
        )

        val_auc = roc_auc_score(
            val_true,
            val_prob[:, 1],
        )

        avg_loss = epoch_loss / max(batch_count, 1)

        print(
            f"Epoch {epoch:03d} | "
            f"loss={avg_loss:.6f} | "
            f"val_AUROC={val_auc:.6f} | "
            f"best={best_val_auc:.6f} | "
            f"wait={wait}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 1
        else:
            wait += 1

        if wait >= EARLY_STOP:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best epoch: {best_epoch}"
            )
            break

    train_time = time.perf_counter() - start_time

    if best_state is None:
        raise RuntimeError("No best model checkpoint was recorded.")

    model.load_state_dict(best_state)

    (
        test_order,
        y_test,
        test_predictions,
        test_probabilities,
    ) = run_inference(
        model,
        graph,
        features,
        labels,
        test_idx,
        device,
    )

    test_metrics = compute_classification_metrics(
        y_test,
        test_predictions,
        test_probabilities,
        sigmoid=False,
    )

    test_metrics["best_epoch"] = int(best_epoch)
    test_metrics["best_val_auroc"] = float(best_val_auc)
    test_metrics["train_time_seconds"] = float(train_time)

    split_info = {
        "seed": int(seed),
        "train_ratio": float(train_ratio),
        "train_size": int(len(train_idx)),
        "valid_size": int(len(valid_idx)),
        "test_size": int(len(test_idx)),
        "nested_split_protocol": True,
        "validation_ratio": 0.2,
        "test_ratio": 0.4,
    }

    config = {
        "model": MODEL_NAME,
        "dataset": DATASET_NAME,
        "seed": int(seed),
        "train_ratio": float(train_ratio),

        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,

        "max_epochs": MAX_EPOCHS,
        "early_stop": EARLY_STOP,
        "early_stop_metric": "validation_AUROC",
        "early_stop_improvement": "strict_greater_than",
        "early_stop_reset_value": 1,

        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "num_layers": NUM_LAYERS,
        "batch_size": BATCH_SIZE,

        "pmp_aggregation": "mean",
        "relation_aggregation": "cat",
        "projection": True,
        "num_transformations": 1,
        "residual_parameter": 0.2,

        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "dgl_version": dgl.__version__,
        "cuda_version": torch.version.cuda,
    }

    run_tag = f"train_{int(train_ratio * 100)}"

    output_dir = os.path.join(
        PROJECT_ROOT,
        "results",
        MODEL_NAME,
        DATASET_NAME,
        f"seed_{seed}",
        run_tag,
    )

    os.makedirs(output_dir, exist_ok=True)

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
        y_test=y_test,
        train_time=train_time,
    )

    with open(
        os.path.join(output_dir, "benchmark_metadata.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_validation_auroc": best_val_auc,
                "actual_epochs_executed": epoch,
            },
            f,
            indent=2,
        )

    print("\nFINAL TEST METRICS")
    print("=" * 60)

    for key, value in test_metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            print(f"{key}: {value}")

    print("=" * 60)
    print("Saved to:", output_dir)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        choices=[
            123,
            1234,
            12345,
            123456,
            1234567,
        ],
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        required=True,
        choices=[
            0.4,
            0.3,
            0.2,
            0.1,
        ],
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train_one_run(
        seed=args.seed,
        train_ratio=args.train_ratio,
    )