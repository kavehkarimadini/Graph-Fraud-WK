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
from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from algorithms.PMP.LASAGE_S import LASAGE_S
from algorithms.PMP.pmp_config import canonical_dataset_name, get_pmp_config
from algorithms.PMP.pmp_data import load_pmp_dataset
from utils.evaluation import compute_classification_metrics, save_run_outputs

MODEL_NAME = "PMP"
ALLOWED_SEEDS = [123, 1234, 12345, 123456, 1234567]
ALLOWED_TRAIN_RATIOS = [0.1, 0.2, 0.3, 0.4]
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
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def ratio_to_key(train_ratio):
    return {0.1: "train_10", 0.2: "train_20", 0.3: "train_30", 0.4: "train_40"}[float(train_ratio)]


def output_dir_for(dataset, seed, train_ratio):
    pct = int(round(train_ratio * 100))
    return os.path.join(PROJECT_ROOT, "results", MODEL_NAME, dataset, f"seed_{seed}", f"train_{pct}")


def run_complete(dataset, seed, train_ratio):
    out = output_dir_for(dataset, seed, train_ratio)
    required = [
        "metrics.json", "config.json", "benchmark_metadata.json",
        "split.npz", "predictions.npz", "checkpoint.pt",
    ]
    return all(os.path.exists(os.path.join(out, name)) for name in required)


def load_split(dataset, seed, train_ratio):
    path = os.path.join(PROJECT_ROOT, "splits", dataset, f"seed_{seed}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Benchmark split not found: {path}\n"
            f"Run: python benchmark/generate_benchmark_splits.py --dataset {dataset}"
        )
    data = np.load(path)
    train_idx = data[ratio_to_key(train_ratio)].astype(np.int64)
    valid_idx = data["val"].astype(np.int64)
    test_idx = data["test"].astype(np.int64)
    train40_idx = data["train_40"].astype(np.int64)
    discarded_idx = np.setdiff1d(train40_idx, train_idx, assume_unique=False).astype(np.int64)
    return train_idx, valid_idx, test_idx, discarded_idx, path


def validate_split(train_idx, valid_idx, test_idx):
    tr, va, te = set(train_idx.tolist()), set(valid_idx.tolist()), set(test_idx.tolist())
    if tr & va or tr & te or va & te:
        raise RuntimeError("Split error: train/validation/test sets overlap.")


def configure_pmp_labels(graph, labels, train_idx):
    label_unk = torch.full((graph.num_nodes(),), 2, dtype=torch.long)
    idx = torch.as_tensor(train_idx, dtype=torch.long)
    label_unk[idx] = labels[idx]
    graph.ndata["label_unk"] = label_unk
    return graph


def activation_from_name(name):
    name = name.lower()
    if name == "elu":
        return nn.ELU(inplace=True)
    if name == "relu":
        return nn.ReLU(inplace=True)
    raise ValueError(f"Unsupported MLP activation: {name}")


def build_model(features, relations, device, cfg):
    model = LASAGE_S(
        in_size=features.shape[1],
        hid_size=cfg["hidden_dim"],
        out_size=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        proj=cfg["projection"],
        num_relations=len(relations),
        batch_size=cfg["batch_size"],
        num_trans=cfg["num_trans"],
        mlp_activation=activation_from_name(cfg["mlp_activation"]),
        out_proj_size=2,
        agg=cfg["aggregation"],
        relation_agg=cfg["relation_aggregation"],
    )
    return model.to(device)


def make_loader(graph, node_indices, batch_size, shuffle, num_layers):
    sampler = MultiLayerFullNeighborSampler(num_layers=num_layers)
    node_tensor = torch.as_tensor(node_indices, dtype=torch.long)
    return DataLoader(
        graph, node_tensor, sampler,
        batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=0,
    )


def forward_batch(model, blocks, relations, features, input_nodes, device):
    blocks = [block.to(device) for block in blocks]
    batch_features = features[input_nodes].to(device)
    return model(blocks, relations, batch_features)


def evaluate_nodes(model, graph, features, labels, relations, node_indices, device, cfg):
    model.eval()
    loader = make_loader(
        graph, node_indices, cfg["eval_batch_size"], False, cfg["num_layers"]
    )
    all_indices, all_labels, all_logits = [], [], []
    with torch.no_grad():
        for input_nodes, output_nodes, blocks in loader:
            logits = forward_batch(model, blocks, relations, features, input_nodes, device)
            all_indices.append(output_nodes.cpu())
            all_labels.append(labels[output_nodes].cpu())
            all_logits.append(logits.detach().cpu())

    output_indices = torch.cat(all_indices).numpy()
    y_true = torch.cat(all_labels).numpy()
    logits = torch.cat(all_logits)
    probabilities = torch.softmax(logits, dim=1).numpy()
    predictions = np.argmax(probabilities, axis=1)
    return output_indices, y_true, predictions, probabilities


def train_one_run(dataset, seed, train_ratio):
    dataset = canonical_dataset_name(dataset)
    cfg = get_pmp_config(dataset)

    print("=" * 70)
    print(f"PMP | dataset={dataset} | seed={seed} | train_ratio={train_ratio}")
    print("=" * 70)

    set_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run the training benchmark on the GPU instance.")
    device = torch.device("cuda:0")
    print("GPU:", torch.cuda.get_device_name(0))

    graph, features, labels, relations, _ = load_pmp_dataset(dataset, PROJECT_ROOT)
    print(f"Nodes={graph.num_nodes()} Edges={graph.num_edges()} Relations={relations}")

    train_idx, valid_idx, test_idx, discarded_idx, split_path = load_split(dataset, seed, train_ratio)
    validate_split(train_idx, valid_idx, test_idx)
    print(f"Train={len(train_idx)} Val={len(valid_idx)} Test={len(test_idx)} Discarded={len(discarded_idx)}")

    graph = configure_pmp_labels(graph, labels, train_idx)
    model = build_model(features, relations, device, cfg)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    if cfg["weighted_loss"]:
        raise NotImplementedError("Weighted PMP loss is intentionally disabled in current official configs.")
    loss_function = nn.CrossEntropyLoss()
    train_loader = make_loader(
        graph, train_idx, cfg["batch_size"], True, cfg["num_layers"]
    )

    best_val_auc = -np.inf
    best_state = None
    best_epoch = 0
    wait = 1  # preserve completed Amazon convention
    actual_epochs = 0
    start_time = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        actual_epochs = epoch
        model.train()
        total_loss, total_batches = 0.0, 0

        for input_nodes, output_nodes, blocks in train_loader:
            logits = forward_batch(model, blocks, relations, features, input_nodes, device)
            batch_labels = labels[output_nodes].to(device)
            optimizer.zero_grad()
            loss = loss_function(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu().item())
            total_batches += 1

        average_loss = total_loss / max(total_batches, 1)
        _, y_val, _, val_prob = evaluate_nodes(
            model, graph, features, labels, relations, valid_idx, device, cfg
        )
        val_auc = roc_auc_score(y_val, val_prob[:, 1])

        if val_auc > best_val_auc:
            best_val_auc = float(val_auc)
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            wait = 1
        else:
            wait += 1

        print(
            f"Epoch {epoch:03d} | Loss {average_loss:.6f} | Val AUROC {val_auc:.6f} | "
            f"Best {best_val_auc:.6f} @ {best_epoch} | Wait {wait}/{EARLY_STOP}"
        )
        if wait >= EARLY_STOP:
            print(f"Early stopping at epoch {epoch}.")
            break

    train_time = time.perf_counter() - start_time
    if best_state is None:
        raise RuntimeError("No best validation checkpoint recorded.")
    model.load_state_dict(best_state)

    test_order, y_test, test_predictions, test_probabilities = evaluate_nodes(
        model, graph, features, labels, relations, test_idx, device, cfg
    )
    test_metrics = compute_classification_metrics(
        y_true=y_test, y_pred=test_predictions, y_prob=test_probabilities, sigmoid=False
    )
    test_metrics.update({
        "best_epoch": int(best_epoch),
        "best_val_auroc": float(best_val_auc),
        "train_time_seconds": float(train_time),
        "epochs_executed": int(actual_epochs),
    })

    split_info = {
        "seed": int(seed), "train_ratio": float(train_ratio),
        "train_size": len(train_idx), "validation_size": len(valid_idx),
        "test_size": len(test_idx), "discarded_size": len(discarded_idx),
        "train_pool_ratio": 0.4, "validation_ratio": 0.2, "test_ratio": 0.4,
        "nested_training_subsets": True,
        "split_file": os.path.relpath(split_path, PROJECT_ROOT),
    }

    config = {
        "model": MODEL_NAME, "dataset": dataset, "seed": int(seed),
        "train_ratio": float(train_ratio), "optimizer": "Adam",
        "learning_rate": cfg["learning_rate"], "weight_decay": cfg["weight_decay"],
        "max_epochs": MAX_EPOCHS, "early_stop": EARLY_STOP,
        "early_stop_metric": "validation_AUROC", "early_stop_rule": "strict_greater_than",
        "early_stop_counter_reset": 1,
        "hidden_dim": cfg["hidden_dim"], "dropout": cfg["dropout"],
        "num_layers": cfg["num_layers"], "batch_size": cfg["batch_size"],
        "num_trans": cfg["num_trans"], "aggregation": cfg["aggregation"],
        "relation_aggregation": cfg["relation_aggregation"], "projection": cfg["projection"],
        "output_classes": 2, "mlp_activation": cfg["mlp_activation"].upper(),
        "feature_normalization": "PMP row normalization", "weighted_loss": cfg["weighted_loss"],
        "loss": "CrossEntropyLoss", "hyperparameter_source": cfg["source_note"],
        "platform": "Vast.ai", "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__, "dgl_version": dgl.__version__,
        "cuda_version": torch.version.cuda,
    }

    output_dir = output_dir_for(dataset, seed, train_ratio)
    os.makedirs(output_dir, exist_ok=True)
    save_run_outputs(
        save_dir=output_dir, model_name=MODEL_NAME, config=config,
        split_info=split_info, test_metrics=test_metrics, model=model,
        predictions=test_predictions, probabilities=test_probabilities,
        test_indices=test_order, train_indices=train_idx, valid_indices=valid_idx,
        discarded_indices=discarded_idx, y_test=y_test,
        y_train=labels[torch.as_tensor(train_idx, dtype=torch.long)].cpu().numpy(),
        train_time=train_time,
    )

    with open(os.path.join(output_dir, "benchmark_metadata.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model": MODEL_NAME, "dataset": dataset, "seed": int(seed),
            "train_ratio": float(train_ratio), "best_epoch": int(best_epoch),
            "best_validation_auroc": float(best_val_auc),
            "epochs_executed": int(actual_epochs), "train_time_seconds": float(train_time),
            "test_evaluated_after_model_selection": True,
        }, f, indent=2)

    print("=" * 70)
    print("FINAL TEST RESULTS")
    for key in [
        "auroc", "auprc", "g_mean", "f1_macro", "f1_micro", "f1_weighted",
        "precision_macro", "recall_macro", "accuracy", "best_val_auroc",
        "best_epoch", "epochs_executed", "train_time_seconds",
    ]:
        if key in test_metrics:
            value = test_metrics[key]
            print(f"{key:25s}: {value:.6f}" if isinstance(value, (float, np.floating)) else f"{key:25s}: {value}")
    print("Saved:", output_dir)
    return test_metrics, output_dir


def parse_args():
    parser = argparse.ArgumentParser(description="PMP unified benchmark runner")
    parser.add_argument("--dataset", required=True, choices=["amazon", "yelp", "yelpchi", "tfinance", "fdcompcn", "elliptic"])
    parser.add_argument("--seed", type=int, choices=ALLOWED_SEEDS)
    parser.add_argument("--train-ratio", type=float, choices=ALLOWED_TRAIN_RATIOS)
    parser.add_argument("--all", action="store_true", help="Run all 5 seeds x 4 train ratios.")
    parser.add_argument("--resume", action="store_true", help="Skip runs with a complete six-file artifact set.")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = canonical_dataset_name(args.dataset)

    if args.all:
        for ratio in ALLOWED_TRAIN_RATIOS:
            for seed in ALLOWED_SEEDS:
                if args.resume and run_complete(dataset, seed, ratio):
                    print(f"SKIP complete: {dataset} seed={seed} train_ratio={ratio}")
                    continue
                train_one_run(dataset, seed, ratio)
        return

    if args.seed is None or args.train_ratio is None:
        raise SystemExit("Single-run mode requires both --seed and --train-ratio, or use --all.")

    if args.resume and run_complete(dataset, args.seed, args.train_ratio):
        print(f"SKIP complete: {dataset} seed={args.seed} train_ratio={args.train_ratio}")
        return
    train_one_run(dataset, args.seed, args.train_ratio)


if __name__ == "__main__":
    main()
