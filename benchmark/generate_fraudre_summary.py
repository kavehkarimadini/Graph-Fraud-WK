"""Create run-level and per-train-ratio FRAUDRE summaries for any completed dataset."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [123, 1234, 12345, 123456, 1234567]
RATIOS = [10, 20, 30, 40]
METRICS = [
    "accuracy", "f1_micro", "f1_macro", "f1_weighted",
    "precision_macro", "recall_macro", "g_mean", "auroc", "auprc",
    "best_val_auroc", "train_time_seconds", "best_epoch", "epochs_executed",
]


def build(dataset):
    dataset = dataset.lower().replace("yelpchi", "yelp")
    results_dir = ROOT / "results" / "FRAUDRE" / dataset
    output_dir = ROOT / "summaries" / "FRAUDRE" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for metrics_path in sorted(results_dir.glob("seed_*/train_*/metrics.json")):
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        split, test = metrics["split"], metrics["test"]
        run = {
            "seed": int(split["seed"]),
            "train_ratio": float(split["train_ratio"]),
            "train_ratio_pct": int(round(float(split["train_ratio"]) * 100)),
            "train_size": int(split["train_size"]),
            "validation_size": int(split["validation_size"]),
            "test_size": int(split["test_size"]),
        }
        for key in METRICS:
            if key in test:
                run[key] = test[key]
        run["num_parameters"] = metrics.get("complexity", {}).get("num_parameters")
        runs.append(run)

    runs.sort(key=lambda x: (x["train_ratio_pct"], x["seed"]))
    if not runs:
        raise FileNotFoundError(f"No metrics.json files found under {results_dir}")

    expected = len(SEEDS) * len(RATIOS)
    print(f"Found {len(runs)}/{expected} runs for FRAUDRE {dataset}.")

    with (output_dir / "runs.json").open("w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)
    with (output_dir / f"fraudre_{dataset}_20_runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=runs[0].keys())
        writer.writeheader(); writer.writerows(runs)

    summary = []
    for ratio in RATIOS:
        ratio_runs = [r for r in runs if r["train_ratio_pct"] == ratio]
        if not ratio_runs:
            continue
        item = {
            "train_ratio_pct": ratio,
            "num_seeds": len(ratio_runs),
            "seeds": [r["seed"] for r in ratio_runs],
        }
        for metric in METRICS:
            values = [float(r[metric]) for r in ratio_runs if metric in r and r[metric] is not None]
            if values:
                item[metric] = {
                    "mean": mean(values),
                    "std": stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                }
        summary.append(item)

    payload = {
        "model": "FRAUDRE", "dataset": dataset, "num_runs": len(runs),
        "num_seeds": len(SEEDS), "train_ratios_pct": RATIOS,
        "summary": summary,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    csv_rows = []
    for item in summary:
        row = {"train_ratio_pct": item["train_ratio_pct"], "num_seeds": item["num_seeds"]}
        for metric in METRICS:
            if metric in item:
                row[f"{metric}_mean"] = item[metric]["mean"]
                row[f"{metric}_std"] = item[metric]["std"]
        csv_rows.append(row)
    with (output_dir / f"fraudre_{dataset}_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader(); writer.writerows(csv_rows)

    print("Summary saved to:", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    build(args.dataset)
