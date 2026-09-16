import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "base_pmp_tuner",
    ROOT / "benchmark" / "tune_pmp_unsupported.py",
)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

SEEDS = [123, 1234, 12345, 123456, 1234567]
OUT = ROOT / "tuning" / "PMP" / "fdcompcn_paper_grid"

CANDIDATES = []

for hidden in [256, 512]:
    for lr in [0.01, 0.005]:
        for dropout in [0.0, 0.4, 0.6]:
            CANDIDATES.append({
                "name": f"h{hidden}_lr{str(lr).replace('.','p')}_d{str(dropout).replace('.','p')}",
                "hidden_dim": hidden,
                "dropout": dropout,
                "batch_size": 256,
                "learning_rate": lr,
                "weight_decay": 0.0,
                "mlp_activation": "relu",
                "source": (
                    "Predefined PMP FDCompCN validation-only search using "
                    "hyperparameter values reported in PMP ICLR 2024 Appendix D.4."
                ),
            })


def result_path(dataset, candidate_name, seed):
    return OUT / candidate_name / f"seed_{seed}.json"


base.result_path = result_path


def summarize():
    rows = []

    for c in CANDIDATES:
        vals = []

        for seed in SEEDS:
            p = result_path("fdcompcn", c["name"], seed)

            if not p.exists():
                raise RuntimeError(f"Missing: {p}")

            d = json.loads(p.read_text())

            if d.get("test_accessed") is not False:
                raise RuntimeError(f"TEST ACCESS VIOLATION: {p}")

            vals.append(float(d["best_val_auroc"]))

        rows.append({
            "candidate": c["name"],
            "hidden_dim": c["hidden_dim"],
            "learning_rate": c["learning_rate"],
            "dropout": c["dropout"],
            "batch_size": c["batch_size"],
            "weight_decay": c["weight_decay"],
            "mean_val_auroc": float(np.mean(vals)),
            "std_val_auroc": float(np.std(vals)),
            "num_seeds": 5,
            "test_accessed": False,
        })

    rows.sort(key=lambda r: (-r["mean_val_auroc"], r["std_val_auroc"]))

    OUT.mkdir(parents=True, exist_ok=True)

    with (OUT / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    payload = {
        "model": "PMP",
        "dataset": "fdcompcn",
        "selection": "highest mean validation AUROC across five seeds",
        "train_budget": 0.4,
        "test_accessed": False,
        "search_space": {
            "hidden_dim": [256, 512],
            "learning_rate": [0.01, 0.005],
            "dropout": [0.0, 0.4, 0.6],
            "batch_size": [256],
            "weight_decay": [0.0],
            "num_layers": [1],
        },
        "results": rows,
        "selected": rows[0],
    }

    (OUT / "summary.json").write_text(json.dumps(payload, indent=2))

    print("=" * 72)
    print("FINAL PAPER-GROUNDED VALIDATION SEARCH")
    print("=" * 72)

    for r in rows:
        print(
            f"{r['candidate']:25s} "
            f"{r['mean_val_auroc']:.6f} ± {r['std_val_auroc']:.6f}"
        )

    print("\nSELECTED:")
    print(json.dumps(rows[0], indent=2))
    print("\nTEST ACCESSED: False")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    device = torch.device("cuda:0")

    graph, features, labels, relations, _ = base.load_pmp_dataset(
        "fdcompcn",
        str(ROOT),
    )

    print("Candidates:", len(CANDIDATES))
    print("Seeds:", len(SEEDS))
    print("Total validation-only runs:", len(CANDIDATES) * len(SEEDS))

    for c in CANDIDATES:
        for seed in SEEDS:
            p = result_path("fdcompcn", c["name"], seed)

            if args.resume and p.exists():
                print("SKIP", c["name"], seed)
                continue

            base.run_one(
                "fdcompcn",
                c,
                seed,
                graph,
                features,
                labels,
                relations,
                device,
            )

    summarize()


if __name__ == "__main__":
    main()
