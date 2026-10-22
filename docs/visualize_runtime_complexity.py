"""
Static visualizations of per-epoch runtimes and parameter complexity.

Reads docs/data/runtime_agg.csv (run aggregate_results.py first) and writes
captioned figures to docs/figures/.

Usage (from repo root):
    python docs/aggregate_results.py
    python docs/visualize_runtime_complexity.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

DOCS = Path(__file__).resolve().parent
DATA = DOCS / "data"
FIG = DOCS / "figures"

DATASETS = ["Amazon", "YelpChi", "T-Finance", "Elliptic", "FDCompCN"]
CMAP = LinearSegmentedColormap.from_list(
    "warm",
    ["#fff8f0", "#f0d4b0", "#d4894a", "#8c4a1a", "#3d220c"],
)
CMAP_COOL = LinearSegmentedColormap.from_list(
    "cool",
    ["#f4f8fb", "#c5d6e4", "#6a8fad", "#2f4f6b", "#132433"],
)


def collapse(runtime: pd.DataFrame) -> pd.DataFrame:
    return (
        runtime.groupby(["model", "dataset"], as_index=False)
        .agg(
            per_epoch_sec=("per_epoch_sec_mean", "mean"),
            num_parameters=("num_parameters_mean", "mean"),
        )
    )


def plot_runtime_table(collapsed: pd.DataFrame, out_path: Path) -> None:
    models = sorted(collapsed["model"].unique())
    cell_text = []
    heat = []
    for model in models:
        row_t, row_h = [], []
        for ds in DATASETS:
            r = collapsed[(collapsed["model"] == model) & (collapsed["dataset"] == ds)]
            if r.empty or pd.isna(r.iloc[0]["per_epoch_sec"]):
                row_t.append("—")
                row_h.append(np.nan)
            else:
                v = float(r.iloc[0]["per_epoch_sec"])
                row_t.append(f"{v:.3f}")
                row_h.append(v)
        cell_text.append(row_t)
        heat.append(row_h)
    heat_arr = np.array(heat, dtype=float)
    log_heat = np.log10(np.clip(heat_arr, 1e-4, None))

    fig, ax = plt.subplots(figsize=(10, max(4, len(models) * 0.5 + 1.5)))
    ax.set_axis_off()
    table = ax.table(
        cellText=cell_text,
        rowLabels=models,
        colLabels=DATASETS,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.15, 1.6)

    valid = log_heat[~np.isnan(log_heat)]
    lo, hi = (valid.min(), valid.max()) if len(valid) else (0, 1)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d0c4b4")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor("#3d220c")
            cell.set_text_props(color="white", weight="bold")
        elif c == -1:
            cell.set_facecolor("#f5ebe0")
            cell.set_text_props(weight="bold")
        elif r > 0 and c >= 0:
            v = log_heat[r - 1, c]
            if np.isnan(v):
                cell.set_facecolor("#f0f0f0")
                cell.get_text().set_color("#888888")
            else:
                t = (v - lo) / (hi - lo + 1e-12)
                cell.set_facecolor(CMAP(t))
                cell.get_text().set_color("#ffffff" if t > 0.55 else "#2a1a0c")

    fig.suptitle(
        "Table R1. Average per-epoch training time (seconds)",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.02,
        "Caption: Mean wall-clock seconds per training epoch, averaged over "
        "seeds and training ratios. Colour uses a log scale (darker = slower). "
        "T-Finance denseness dominates cost for spectral / sampling-heavy models.",
        ha="center",
        fontsize=8,
        color="#443322",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_params_table(collapsed: pd.DataFrame, out_path: Path) -> None:
    models = sorted(collapsed["model"].unique())
    cell_text, heat = [], []
    for model in models:
        row_t, row_h = [], []
        for ds in DATASETS:
            r = collapsed[(collapsed["model"] == model) & (collapsed["dataset"] == ds)]
            if r.empty or pd.isna(r.iloc[0]["num_parameters"]):
                row_t.append("—")
                row_h.append(np.nan)
            else:
                n = int(round(r.iloc[0]["num_parameters"]))
                row_t.append(f"{n:,}")
                row_h.append(float(n))
        cell_text.append(row_t)
        heat.append(row_h)
    heat_arr = np.array(heat, dtype=float)
    log_heat = np.log10(np.clip(heat_arr, 1.0, None))

    fig, ax = plt.subplots(figsize=(10, max(4, len(models) * 0.5 + 1.5)))
    ax.set_axis_off()
    table = ax.table(
        cellText=cell_text,
        rowLabels=models,
        colLabels=DATASETS,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.15, 1.6)

    valid = log_heat[~np.isnan(log_heat)]
    lo, hi = (valid.min(), valid.max()) if len(valid) else (0, 1)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#b4c4d0")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor("#132433")
            cell.set_text_props(color="white", weight="bold")
        elif c == -1:
            cell.set_facecolor("#e4eef4")
            cell.set_text_props(weight="bold")
        elif r > 0 and c >= 0:
            v = log_heat[r - 1, c]
            if np.isnan(v):
                cell.set_facecolor("#f0f0f0")
                cell.get_text().set_color("#888888")
            else:
                t = (v - lo) / (hi - lo + 1e-12)
                cell.set_facecolor(CMAP_COOL(t))
                cell.get_text().set_color("#ffffff" if t > 0.55 else "#102030")

    fig.suptitle(
        "Table R2. Trainable parameter counts by model and dataset",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.02,
        "Caption: Mean number of trainable parameters. Counts vary with input "
        "feature dimension and model-specific widths; colour uses a log scale.",
        ha="center",
        fontsize=8,
        color="#223344",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_runtime_bars(collapsed: pd.DataFrame, out_path: Path) -> None:
    models = sorted(collapsed["model"].unique())
    x = np.arange(len(DATASETS))
    width = 0.08
    fig, ax = plt.subplots(figsize=(13, 5.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    for i, model in enumerate(models):
        vals = []
        for ds in DATASETS:
            r = collapsed[(collapsed["model"] == model) & (collapsed["dataset"] == ds)]
            vals.append(
                r.iloc[0]["per_epoch_sec"]
                if not r.empty and not pd.isna(r.iloc[0]["per_epoch_sec"])
                else np.nan
            )
        ax.bar(
            x + (i - len(models) / 2) * width + width / 2,
            vals,
            width=width,
            label=model,
            color=colors[i],
            edgecolor="white",
            linewidth=0.3,
        )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_ylabel("Seconds / epoch (log scale)")
    ax.set_title(
        "Figure R1. Per-epoch runtime comparison (log scale)",
        fontweight="bold",
    )
    ax.legend(ncol=5, fontsize=8, frameon=False, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3, which="both", linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(
        0.5,
        0.01,
        "Caption: Grouped bars of average per-epoch time. Log scale highlights "
        "orders-of-magnitude gaps between sparse graphs (FDCompCN, Elliptic) "
        "and dense transaction graphs (T-Finance).",
        ha="center",
        fontsize=8,
        color="#443322",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_params_bars(collapsed: pd.DataFrame, out_path: Path) -> None:
    # Use Amazon params as a representative architecture size when available,
    # else mean across datasets.
    rows = []
    for model, g in collapsed.groupby("model"):
        amz = g[g["dataset"] == "Amazon"]["num_parameters"]
        if not amz.empty and not pd.isna(amz.iloc[0]):
            rows.append((model, float(amz.iloc[0])))
        else:
            rows.append((model, float(g["num_parameters"].mean())))
    rows.sort(key=lambda t: t[1])
    models = [r[0] for r in rows]
    vals = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.barh(models, vals, color="#2f4f6b", edgecolor="white")
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_title(
        "Figure R2. Parameter complexity (representative / mean count)",
        fontweight="bold",
    )
    for bar, v in zip(bars, vals):
        ax.text(
            v * 1.05,
            bar.get_y() + bar.get_height() / 2,
            f"{int(v):,}",
            va="center",
            fontsize=8,
            color="#132433",
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(
        0.5,
        0.01,
        "Caption: Model capacity on a log axis. SplitGNN / GAGA / GLHN remain "
        "compact; PMP and DGA-GNN grow with feature width and grouping heads.",
        ha="center",
        fontsize=8,
        color="#223344",
    )
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_runtime_vs_params(collapsed: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6))
    markers = ["o", "s", "^", "D", "P"]
    for ds, mk in zip(DATASETS, markers):
        sub = collapsed[collapsed["dataset"] == ds].dropna(
            subset=["per_epoch_sec", "num_parameters"]
        )
        if sub.empty:
            continue
        ax.scatter(
            sub["num_parameters"],
            sub["per_epoch_sec"],
            s=55,
            marker=mk,
            label=ds,
            alpha=0.85,
        )
        for _, row in sub.iterrows():
            ax.annotate(
                row["model"],
                (row["num_parameters"], row["per_epoch_sec"]),
                textcoords="offset points",
                xytext=(4, 3),
                fontsize=6,
                alpha=0.85,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Parameters (log)")
    ax.set_ylabel("Seconds / epoch (log)")
    ax.set_title(
        "Figure R3. Runtime versus parameter complexity",
        fontweight="bold",
    )
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, which="both", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(
        0.5,
        0.01,
        "Caption: Points far above the diagonal trend are dominated by graph "
        "density / spectral ops rather than parameter count (e.g. GraphConsis "
        "and SplitGNN on T-Finance).",
        ha="center",
        fontsize=8,
        color="#334433",
    )
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    path = DATA / "runtime_agg.csv"
    if not path.exists():
        raise SystemExit("Missing docs/data/runtime_agg.csv — run docs/aggregate_results.py first.")
    runtime = pd.read_csv(path)
    collapsed = collapse(runtime)

    plot_runtime_table(collapsed, FIG / "runtime_table.png")
    plot_params_table(collapsed, FIG / "params_table.png")
    plot_runtime_bars(collapsed, FIG / "runtime_bars.png")
    plot_params_bars(collapsed, FIG / "params_bars.png")
    plot_runtime_vs_params(collapsed, FIG / "runtime_vs_params.png")
    print(f"Wrote runtime/complexity figures under {FIG}")


if __name__ == "__main__":
    main()
