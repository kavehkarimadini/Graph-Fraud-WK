import argparse
import copy
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.PMP.PMP_benchmark import (
    EARLY_STOP,
    MAX_EPOCHS,
    build_model,
    configure_pmp_labels,
    evaluate_nodes,
    forward_batch,
    make_loader,
    set_seed,
)
from algorithms.PMP.pmp_data import load_pmp_dataset


SEEDS = [123, 1234, 12345, 123456, 1234567]

CANDIDATES = [
    {
        "name": "paper_yelp",
        "hidden_dim": 256,
        "dropout": 0.0,
        "batch_size": 512,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "mlp_activation": "relu",
        "source": "PMP paper Table 5 Yelp setting",
    },
    {
        "name": "release_yelp",
        "hidden_dim": 48,
        "dropout": 0.0,
        "batch_size": 512,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "mlp_activation": "relu",
        "source": "Released Xtra-Computing/PMP yelp.yml",
    },
    {
        "name": "paper_amazon",
        "hidden_dim": 128,
        "dropout": 0.6,
        "batch_size": 128,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "mlp_activation": "elu",
        "source": "PMP paper Table 5 Amazon numeric setting; ELU from released Amazon implementation",
    },
    {
        "name": "release_amazon",
        "hidden_dim": 256,
        "dropout": 0.6,
        "batch_size": 128,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "mlp_activation": "elu",
        "source": "Released Xtra-Computing/PMP amazon.yml",
    },
    {
        "name": "paper_tfinance",
        "hidden_dim": 64,
        "dropout": 0.4,
        "batch_size": 256,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "mlp_activation": "relu",
        "source": "PMP paper Table 5 T-Finance setting",
    },
    {
        "name": "release_tfinance",
        "hidden_dim": 64,
        "dropout": 0.4,
        "batch_size": 256,
        "learning_rate": 0.005,
        "weight_decay": 0.0,
        "mlp_activation": "relu",
        "source": "Released Xtra-Computing/PMP tfinance.yml",
    },
]


def full_cfg(candidate):
    return {
        **candidate,
        "num_layers": 1,
        "num_trans": 1,
        "aggregation": "mean",
        "relation_aggregation": "cat",
        "projection": True,
        "eval_batch_size": 4096,
        "weighted_loss": False,
    }


def load_train_validation(dataset, seed):
    path = ROOT / "splits" / dataset / f"seed_{seed}.npz"
    if not path.exists():
        raise FileNotFoundError(path)

    # Intentionally read ONLY train_40 and val.
    # Test indices are never accessed during tuning.
    with np.load(path, allow_pickle=False) as data:
        train_idx = data["train_40"].astype(np.int64)
        val_idx = data["val"].astype(np.int64)

    if set(train_idx.tolist()) & set(val_idx.tolist()):
        raise RuntimeError("Training/validation overlap detected.")

    return train_idx, val_idx, path


def result_path(dataset, candidate_name, seed):
    return (
        ROOT
        / "tuning"
        / "PMP"
        / dataset
        / candidate_name
        / f"seed_{seed}.json"
    )


def run_one(dataset, candidate, seed, graph, features, labels, relations, device):
    cfg = full_cfg(candidate)

    out = result_path(dataset, candidate["name"], seed)
    out.parent.mkdir(parents=True, exist_ok=True)

    set_seed(seed)

    train_idx, val_idx, split_path = load_train_validation(dataset, seed)

    graph = configure_pmp_labels(graph, labels, train_idx)

    model = build_model(features, relations, device, cfg)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    loss_fn = nn.CrossEntropyLoss()

    train_loader = make_loader(
        graph,
        train_idx,
        cfg["batch_size"],
        True,
        cfg["num_layers"],
    )

    best_val_auc = -np.inf
    best_state = None
    best_epoch = 0
    wait = 1
    actual_epochs = 0

    start = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        actual_epochs = epoch

        model.train()
        total_loss = 0.0
        batches = 0

        for input_nodes, output_nodes, blocks in train_loader:
            logits = forward_batch(
                model,
                blocks,
                relations,
                features,
                input_nodes,
                device,
            )

            batch_labels = labels[output_nodes].to(device)

            optimizer.zero_grad()
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            batches += 1

        _, y_val, _, val_prob = evaluate_nodes(
            model,
            graph,
            features,
            labels,
            relations,
            val_idx,
            device,
            cfg,
        )

        val_auc = float(
            roc_auc_score(y_val, val_prob[:, 1])
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 1
        else:
            wait += 1

        avg_loss = total_loss / max(batches, 1)

        print(
            f"{dataset} | {candidate['name']} | seed={seed} | "
            f"epoch={epoch:03d} loss={avg_loss:.6f} "
            f"val_auc={val_auc:.6f} "
            f"best={best_val_auc:.6f}@{best_epoch} "
            f"wait={wait}/{EARLY_STOP}"
        )

        if wait >= EARLY_STOP:
            print(f"Early stopping at epoch {epoch}.")
            break

    elapsed = time.perf_counter() - start

    if best_state is None:
        raise RuntimeError("No best validation checkpoint recorded.")

    payload = {
        "model": "PMP",
        "dataset": dataset,
        "candidate": candidate["name"],
        "source": candidate["source"],
        "seed": seed,
        "train_ratio": 0.4,
        "train_size": int(len(train_idx)),
        "validation_size": int(len(val_idx)),
        "hidden_dim": cfg["hidden_dim"],
        "dropout": cfg["dropout"],
        "batch_size": cfg["batch_size"],
        "learning_rate": cfg["learning_rate"],
        "weight_decay": cfg["weight_decay"],
        "num_layers": cfg["num_layers"],
        "num_trans": cfg["num_trans"],
        "aggregation": cfg["aggregation"],
        "relation_aggregation": cfg["relation_aggregation"],
        "projection": cfg["projection"],
        "mlp_activation": cfg["mlp_activation"],
        "best_val_auroc": best_val_auc,
        "best_epoch": best_epoch,
        "epochs_executed": actual_epochs,
        "train_time_seconds": elapsed,
        "split_file": os.path.relpath(split_path, ROOT),
        "selection_metric": "validation_AUROC",
        "test_accessed": False,
    }

    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Saved:", out)

    return payload


def summarize(dataset):
    rows = []

    for candidate in CANDIDATES:
        values = []

        for seed in SEEDS:
            path = result_path(dataset, candidate["name"], seed)

            if not path.exists():
                raise RuntimeError(f"Missing tuning run: {path}")

            with path.open("r", encoding="utf-8") as f:
                d = json.load(f)

            if d.get("test_accessed") is not False:
                raise RuntimeError(
                    f"Test-access audit failed: {path}"
                )

            values.append(float(d["best_val_auroc"]))

        rows.append({
            "candidate": candidate["name"],
            "hidden_dim": candidate["hidden_dim"],
            "dropout": candidate["dropout"],
            "batch_size": candidate["batch_size"],
            "learning_rate": candidate["learning_rate"],
            "weight_decay": candidate["weight_decay"],
            "mlp_activation": candidate["mlp_activation"],
            "mean_val_auroc": float(np.mean(values)),
            "std_val_auroc": float(np.std(values)),
            "num_seeds": len(values),
            "test_accessed": False,
        })

    rows.sort(
        key=lambda r: (
            -r["mean_val_auroc"],
            r["std_val_auroc"],
        )
    )

    winner = rows[0]

    out_dir = ROOT / "tuning" / "PMP" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / f"pmp_{dataset}_tuning_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with (out_dir / f"pmp_{dataset}_tuning_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "model": "PMP",
                "dataset": dataset,
                "selection_metric": "mean validation AUROC across five seeds",
                "test_accessed": False,
                "candidates": rows,
                "selected": winner,
            },
            f,
            indent=2,
        )

    print("=" * 80)
    print("VALIDATION-ONLY PMP TUNING COMPLETE")
    print("=" * 80)

    for row in rows:
        print(
            f"{row['candidate']:20s} | "
            f"mean val AUROC={row['mean_val_auroc']:.6f} "
            f"std={row['std_val_auroc']:.6f}"
        )

    print()
    print("SELECTED:", winner["candidate"])
    print(
        f"hidden={winner['hidden_dim']} "
        f"dropout={winner['dropout']} "
        f"batch={winner['batch_size']} "
        f"lr={winner['learning_rate']}"
    )
    print("Test set accessed: False")
    print("Saved:", out_dir)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["fdcompcn", "elliptic"],
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    device = torch.device("cuda:0")

    graph, features, labels, relations, _ = load_pmp_dataset(
        args.dataset,
        str(ROOT),
    )

    print("GPU:", torch.cuda.get_device_name(0))
    print(
        f"Dataset={args.dataset} "
        f"nodes={graph.num_nodes()} "
        f"edges={graph.num_edges()} "
        f"relations={relations}"
    )

    for candidate in CANDIDATES:
        for seed in SEEDS:
            out = result_path(
                args.dataset,
                candidate["name"],
                seed,
            )

            if args.resume and out.exists():
                print(
                    f"SKIP complete: "
                    f"{candidate['name']} seed={seed}"
                )
                continue

            run_one(
                args.dataset,
                candidate,
                seed,
                graph,
                features,
                labels,
                relations,
                device,
            )

    summarize(args.dataset)


if __name__ == "__main__":
    main()
