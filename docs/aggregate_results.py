"""
Aggregate benchmark metrics from results/ into docs/data/.

Produces mean ± std over seeds for:
  - test metrics (g_mean, auroc, auprc, f1_macro) at 10/20/30/40% train
  - per-epoch train time and parameter counts
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent / "data"

DATASETS = ["Amazon", "YelpChi", "T-Finance", "Elliptic", "FDCompCN"]
METRICS = ["g_mean", "auroc", "auprc", "f1_macro"]
RATIOS = [0.1, 0.2, 0.3, 0.4]

DATASET_ALIASES = {
    "amazon": "Amazon",
    "yelpchi": "YelpChi",
    "yelp": "YelpChi",
    "tfinance": "T-Finance",
    "t-finance": "T-Finance",
    "elliptic": "Elliptic",
    "comp": "FDCompCN",
    "fdcompcn": "FDCompCN",
}


def norm_dataset(name: str) -> str | None:
    key = name.strip().lower().replace("_", "").replace("-", "")
    # normalize without dashes
    mapping = {
        "amazon": "Amazon",
        "yelpchi": "YelpChi",
        "yelp": "YelpChi",
        "tfinance": "T-Finance",
        "elliptic": "Elliptic",
        "comp": "FDCompCN",
        "fdcompcn": "FDCompCN",
    }
    # also try original aliases
    low = name.strip().lower()
    if low in DATASET_ALIASES:
        return DATASET_ALIASES[low]
    compact = re.sub(r"[^a-z0-9]", "", low)
    return mapping.get(compact)


def _safe_float(v):
    if v is None:
        return None
    try:
        f = float(v)
        if np.isnan(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _epochs_run(metrics: dict, config: dict | None) -> float | None:
    test = metrics.get("test") or {}
    if isinstance(test, dict) and test.get("epochs_executed") is not None:
        return float(test["epochs_executed"])
    cfg = config or {}
    best = cfg.get("best_epoch")
    if best is None and isinstance(test, dict):
        best = test.get("best_epoch")
    early = cfg.get("early_stop")
    epochs = cfg.get("epochs")
    if best is not None and early is not None:
        run = float(best) + float(early)
        if epochs is not None:
            run = min(run, float(epochs))
        return max(run, 1.0)
    if epochs is not None:
        return float(epochs)
    return None


def _load_json(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _record_from_metrics(
    model: str,
    dataset: str,
    ratio: float,
    metrics_path: Path,
) -> dict | None:
    metrics = _load_json(metrics_path)
    if not metrics:
        return None
    config = _load_json(metrics_path.with_name("config.json")) or {}

    model_name = metrics.get("model") or config.get("model") or model
    split = metrics.get("split") or {}
    train_ratio = split.get("train_ratio")
    if train_ratio is None:
        train_ratio = config.get("train_ratio", ratio)
    train_ratio = float(train_ratio)

    test = metrics.get("test") or {}
    complexity = metrics.get("complexity") or {}
    train_time = complexity.get("train_time_sec")
    if train_time is None and isinstance(test, dict):
        train_time = test.get("train_time_seconds")
    if train_time is None:
        train_time = config.get("train_time_sec")
    train_time = _safe_float(train_time)

    n_params = complexity.get("num_parameters")
    if n_params is None:
        n_params = config.get("num_parameters")
    n_params = _safe_float(n_params)

    epochs = _epochs_run(metrics, config)
    per_epoch = None
    if train_time is not None and epochs is not None and epochs > 0:
        per_epoch = train_time / epochs

    row = {
        "model": str(model_name),
        "dataset": dataset,
        "train_ratio": train_ratio,
        "train_time_sec": train_time,
        "epochs_run": epochs,
        "per_epoch_sec": per_epoch,
        "num_parameters": n_params,
        "source": str(metrics_path.relative_to(ROOT)),
    }
    for m in METRICS:
        row[m] = _safe_float(test.get(m)) if isinstance(test, dict) else None
    return row


def collect_runs() -> list[dict]:
    rows: list[dict] = []

    # BWGNN_{10,20,30,40}_percent train / {ds} / seed*
    for pct, ratio in [(10, 0.1), (20, 0.2), (30, 0.3), (40, 0.4)]:
        base = RESULTS / f"BWGNN_{pct}_percent train"
        if not base.exists():
            continue
        for ds_dir in base.iterdir():
            if not ds_dir.is_dir():
                continue
            ds = norm_dataset(ds_dir.name)
            if not ds:
                continue
            for seed_dir in ds_dir.iterdir():
                mp = seed_dir / "metrics.json"
                if mp.exists():
                    r = _record_from_metrics("BWGNN", ds, ratio, mp)
                    if r:
                        rows.append(r)

    # train_{10,20,30,40}pct / {GLHN,GraphConsis} / {Dataset} / seed*
    for pct, ratio in [(10, 0.1), (20, 0.2), (30, 0.3), (40, 0.4)]:
        base = RESULTS / f"train_{pct}pct"
        if not base.exists():
            continue
        for model_dir in base.iterdir():
            if not model_dir.is_dir():
                continue
            for ds_dir in model_dir.iterdir():
                if not ds_dir.is_dir():
                    continue
                ds = norm_dataset(ds_dir.name)
                if not ds:
                    continue
                for seed_dir in ds_dir.iterdir():
                    mp = seed_dir / "metrics.json"
                    if mp.exists():
                        r = _record_from_metrics(model_dir.name, ds, ratio, mp)
                        if r:
                            rows.append(r)

    # PMP / SEC_GFD: {ds}/seed_*/train_{10,20,30,40}/metrics.json
    for model in ["PMP", "SEC_GFD"]:
        base = RESULTS / model
        if not base.exists():
            continue
        for ds_dir in base.iterdir():
            if not ds_dir.is_dir():
                continue
            ds = norm_dataset(ds_dir.name)
            if not ds:
                continue
            for seed_dir in ds_dir.iterdir():
                if not seed_dir.is_dir():
                    continue
                for train_dir in seed_dir.iterdir():
                    if not train_dir.is_dir():
                        continue
                    m = re.match(r"train_(\d+)", train_dir.name)
                    if not m:
                        continue
                    ratio = int(m.group(1)) / 100.0
                    mp = train_dir / "metrics.json"
                    if mp.exists():
                        r = _record_from_metrics(model, ds, ratio, mp)
                        if r:
                            rows.append(r)

    # SplitGNN_{20,30,40}_percent train
    for pct, ratio in [(20, 0.2), (30, 0.3), (40, 0.4)]:
        base = RESULTS / f"SplitGNN_{pct}_percent train"
        if not base.exists():
            continue
        for mp in base.rglob("metrics.json"):
            # find dataset folder name in path
            ds = None
            for part in mp.parts:
                ds = norm_dataset(part)
                if ds:
                    break
            if not ds:
                continue
            r = _record_from_metrics("SplitGNN", ds, ratio, mp)
            if r:
                rows.append(r)

    # GAGA_*/GHRN_* flat folders
    for folder in RESULTS.iterdir():
        if not folder.is_dir():
            continue
        m = re.match(
            r"(GAGA|GHRN)_([a-zA-Z0-9]+)_split\d+_train(\d+)_seed\d+",
            folder.name,
        )
        if not m:
            continue
        model, ds_raw, pct = m.group(1), m.group(2), int(m.group(3))
        ds = norm_dataset(ds_raw)
        if not ds:
            continue
        mp = folder / "metrics.json"
        if mp.exists():
            r = _record_from_metrics(model, ds, pct / 100.0, mp)
            if r:
                rows.append(r)

    # Extracted BSL / DGAGNN: results/_extracted/results/tr{10,20,30,40}/{model}/{ds}/seed*
    extracted = RESULTS / "_extracted" / "results"
    if extracted.exists():
        for tr_dir in extracted.iterdir():
            m = re.match(r"tr(\d+)", tr_dir.name)
            if not m:
                continue
            ratio = int(m.group(1)) / 100.0
            for model_dir in tr_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                for ds_dir in model_dir.iterdir():
                    if not ds_dir.is_dir():
                        continue
                    ds = norm_dataset(ds_dir.name)
                    if not ds:
                        continue
                    for seed_dir in ds_dir.iterdir():
                        mp = seed_dir / "metrics.json"
                        if mp.exists():
                            r = _record_from_metrics(model_dir.name, ds, ratio, mp)
                            if r:
                                rows.append(r)

    return rows


def aggregate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Normalize model display names
    rename = {"DGAGNN": "DGA-GNN", "SEC_GFD": "SEC-GFD"}
    df = df.copy()
    df["model"] = df["model"].replace(rename)

    metric_rows = []
    for (model, dataset, ratio), g in df.groupby(["model", "dataset", "train_ratio"]):
        row = {
            "model": model,
            "dataset": dataset,
            "train_ratio": ratio,
            "n_seeds": len(g),
        }
        for m in METRICS:
            vals = g[m].dropna().astype(float)
            if len(vals) == 0:
                row[f"{m}_mean"] = np.nan
                row[f"{m}_std"] = np.nan
            else:
                row[f"{m}_mean"] = float(vals.mean())
                row[f"{m}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        metric_rows.append(row)
    metrics_agg = pd.DataFrame(metric_rows)

    # Runtime / complexity: average over seeds and train ratios (params stable;
    # per-epoch time may vary slightly by ratio — we also keep ratio-specific)
    runtime_rows = []
    for (model, dataset, ratio), g in df.groupby(["model", "dataset", "train_ratio"]):
        pe = g["per_epoch_sec"].dropna().astype(float)
        tt = g["train_time_sec"].dropna().astype(float)
        pp = g["num_parameters"].dropna().astype(float)
        runtime_rows.append(
            {
                "model": model,
                "dataset": dataset,
                "train_ratio": ratio,
                "n_seeds": len(g),
                "per_epoch_sec_mean": float(pe.mean()) if len(pe) else np.nan,
                "per_epoch_sec_std": float(pe.std(ddof=1)) if len(pe) > 1 else (0.0 if len(pe) else np.nan),
                "train_time_sec_mean": float(tt.mean()) if len(tt) else np.nan,
                "num_parameters_mean": float(pp.mean()) if len(pp) else np.nan,
            }
        )
    runtime_agg = pd.DataFrame(runtime_rows)
    return metrics_agg, runtime_agg


def fmt_mean_std(mean, std, digits=3) -> str:
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "—"
    if std is None or (isinstance(std, float) and np.isnan(std)):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f}±{std:.{digits}f}"


def build_markdown_tables(metrics_agg: pd.DataFrame) -> dict[float, str]:
    """Return ratio -> markdown table string with multi-header."""
    models = sorted(metrics_agg["model"].unique())
    tables = {}
    for ratio in RATIOS:
        sub = metrics_agg[np.isclose(metrics_agg["train_ratio"], ratio)]
        # header
        lines = []
        h1 = ["Model"]
        h2 = [""]
        for ds in DATASETS:
            h1.extend([ds, "", "", ""])
            h2.extend(["G-Mean", "AUROC", "AUPRC", "F1-Macro"])
        lines.append("| " + " | ".join(h1) + " |")
        lines.append("| " + " | ".join(["---"] * len(h1)) + " |")
        lines.append("| " + " | ".join(h2) + " |")

        for model in models:
            cells = [model]
            msub = sub[sub["model"] == model]
            for ds in DATASETS:
                row = msub[msub["dataset"] == ds]
                if row.empty:
                    cells.extend(["—"] * 4)
                else:
                    r = row.iloc[0]
                    for m in METRICS:
                        cells.append(fmt_mean_std(r[f"{m}_mean"], r[f"{m}_std"]))
            lines.append("| " + " | ".join(cells) + " |")
        tables[ratio] = "\n".join(lines)
    return tables


def build_runtime_markdown(runtime_agg: pd.DataFrame) -> str:
    """Per-epoch seconds averaged across train ratios for each model×dataset."""
    models = sorted(runtime_agg["model"].unique())
    # average per-epoch across ratios
    collapsed = (
        runtime_agg.groupby(["model", "dataset"])
        .agg(
            per_epoch_sec_mean=("per_epoch_sec_mean", "mean"),
            num_parameters_mean=("num_parameters_mean", "mean"),
        )
        .reset_index()
    )

    lines = []
    h1 = ["Model"] + DATASETS
    lines.append("| " + " | ".join(h1) + " |")
    lines.append("| " + " | ".join(["---"] * len(h1)) + " |")
    for model in models:
        cells = [model]
        for ds in DATASETS:
            row = collapsed[(collapsed["model"] == model) & (collapsed["dataset"] == ds)]
            if row.empty or np.isnan(row.iloc[0]["per_epoch_sec_mean"]):
                cells.append("—")
            else:
                cells.append(f"{row.iloc[0]['per_epoch_sec_mean']:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_params_markdown(runtime_agg: pd.DataFrame) -> str:
    models = sorted(runtime_agg["model"].unique())
    collapsed = (
        runtime_agg.groupby(["model", "dataset"])
        .agg(num_parameters_mean=("num_parameters_mean", "mean"))
        .reset_index()
    )
    lines = []
    h1 = ["Model"] + DATASETS
    lines.append("| " + " | ".join(h1) + " |")
    lines.append("| " + " | ".join(["---"] * len(h1)) + " |")
    for model in models:
        cells = [model]
        for ds in DATASETS:
            row = collapsed[(collapsed["model"] == model) & (collapsed["dataset"] == ds)]
            if row.empty or np.isnan(row.iloc[0]["num_parameters_mean"]):
                cells.append("—")
            else:
                n = int(round(row.iloc[0]["num_parameters_mean"]))
                cells.append(f"{n:,}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = collect_runs()
    if not rows:
        raise SystemExit("No metrics found under results/")
    df = pd.DataFrame(rows)
    # Keep only known datasets / ratios
    df = df[df["dataset"].isin(DATASETS)]
    df = df[df["train_ratio"].isin(RATIOS)]
    df.to_csv(OUT / "runs_raw.csv", index=False)

    metrics_agg, runtime_agg = aggregate(df)
    metrics_agg.to_csv(OUT / "metrics_agg.csv", index=False)
    runtime_agg.to_csv(OUT / "runtime_agg.csv", index=False)

    tables = build_markdown_tables(metrics_agg)
    with open(OUT / "tables_metrics.md", "w", encoding="utf-8") as f:
        for ratio in RATIOS:
            pct = int(ratio * 100)
            f.write(f"## {pct}% training ratio\n\n")
            f.write(tables[ratio] + "\n\n")
        f.write("## Per-epoch average time (seconds)\n\n")
        f.write(build_runtime_markdown(runtime_agg) + "\n\n")
        f.write("## Parameter counts\n\n")
        f.write(build_params_markdown(runtime_agg) + "\n")

    summary = {
        "n_runs": len(df),
        "models": sorted(df["model"].replace({"DGAGNN": "DGA-GNN", "SEC_GFD": "SEC-GFD"}).unique().tolist())
        if "model" in df
        else [],
        "datasets": DATASETS,
        "ratios": RATIOS,
        "coverage": metrics_agg.groupby("model")["dataset"]
        .nunique()
        .to_dict(),
    }
    # fix models list after rename in aggregate
    summary["models"] = sorted(metrics_agg["model"].unique().tolist())
    summary["coverage"] = {
        m: int(metrics_agg[metrics_agg["model"] == m]["dataset"].nunique())
        for m in summary["models"]
    }
    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Collected {len(df)} runs -> {OUT}")
    print("Models:", summary["models"])
    print("Coverage:", summary["coverage"])


if __name__ == "__main__":
    main()
