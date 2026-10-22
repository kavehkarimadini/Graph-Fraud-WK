"""
Static visualizations of benchmark metric tables.

Reads docs/data/metrics_agg.csv (run aggregate_results.py first) and writes
captioned figures to docs/figures/.

Usage (from repo root):
    python docs/aggregate_results.py
    python docs/visualize_metrics.py
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
METRICS = ["g_mean", "auroc", "auprc", "f1_macro"]
METRIC_LABELS = {
    "g_mean": "G-Mean",
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "f1_macro": "F1-Macro",
}
RATIOS = [0.1, 0.2, 0.3, 0.4]

# Soft blue→teal sequential map (avoid purple defaults)
CMAP = LinearSegmentedColormap.from_list(
    "score",
    ["#f7fbfc", "#d0e8e6", "#7fb3a8", "#2f6f68", "#1a3c38"],
)


def _fmt(mean, std) -> str:
    if pd.isna(mean):
        return "—"
    if pd.isna(std):
        return f"{mean:.3f}"
    return f"{mean:.3f}\n±{std:.3f}"


def plot_ratio_table(metrics: pd.DataFrame, ratio: float, out_path: Path) -> None:
    sub = metrics[np.isclose(metrics["train_ratio"], ratio)].copy()
    models = sorted(sub["model"].unique())
    # columns: dataset × metric
    col_labels = []
    for ds in DATASETS:
        for m in METRICS:
            col_labels.append(f"{ds}\n{METRIC_LABELS[m]}")

    cell_text = []
    heat = []
    for model in models:
        row_txt = []
        row_heat = []
        msub = sub[sub["model"] == model]
        for ds in DATASETS:
            r = msub[msub["dataset"] == ds]
            for m in METRICS:
                if r.empty:
                    row_txt.append("—")
                    row_heat.append(np.nan)
                else:
                    mean = r.iloc[0][f"{m}_mean"]
                    std = r.iloc[0][f"{m}_std"]
                    row_txt.append(_fmt(mean, std))
                    row_heat.append(mean if not pd.isna(mean) else np.nan)
        cell_text.append(row_txt)
        heat.append(row_heat)

    heat_arr = np.array(heat, dtype=float)
    # normalize per-metric column group for color only
    color = np.full_like(heat_arr, np.nan)
    for i in range(len(DATASETS)):
        for j in range(len(METRICS)):
            col = i * len(METRICS) + j
            vals = heat_arr[:, col]
            valid = vals[~np.isnan(vals)]
            if len(valid) == 0:
                continue
            lo, hi = valid.min(), valid.max()
            if hi - lo < 1e-9:
                color[:, col] = 0.7
            else:
                color[:, col] = (vals - lo) / (hi - lo)

    n_rows, n_cols = len(models), len(col_labels)
    fig_w = max(18, n_cols * 0.95)
    fig_h = max(4.5, n_rows * 0.55 + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_axis_off()

    table = ax.table(
        cellText=cell_text,
        rowLabels=models,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.55)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#c5d0ce")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor("#1a3c38")
            cell.set_text_props(color="white", weight="bold", fontsize=7)
            cell.set_height(0.12)
        elif c == -1:
            cell.set_facecolor("#e8f0ef")
            cell.set_text_props(weight="bold", fontsize=8)
        elif r > 0 and c >= 0:
            v = color[r - 1, c]
            if np.isnan(v):
                cell.set_facecolor("#f4f4f4")
                cell.get_text().set_color("#888888")
            else:
                cell.set_facecolor(CMAP(v))
                cell.get_text().set_color("#102422" if v < 0.65 else "#ffffff")

    pct = int(ratio * 100)
    fig.suptitle(
        f"Table {pct // 10}. Test metrics at {pct}% training ratio "
        f"(mean ± std over seeds)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.01,
        f"Figure {pct // 10}. Heat-coded comparison of G-Mean, AUROC, AUPRC and "
        f"F1-Macro across five datasets (rows = models). Darker teal = higher "
        f"score within each metric column. Em dash (—) = missing run.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#334433",
        wrap=True,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_metric_heatmap_grid(metrics: pd.DataFrame, out_path: Path) -> None:
    """One AUROC heatmap per training ratio for a compact overview."""
    models = sorted(metrics["model"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    axes = axes.ravel()
    for ax, ratio in zip(axes, RATIOS):
        sub = metrics[np.isclose(metrics["train_ratio"], ratio)]
        mat = np.full((len(models), len(DATASETS)), np.nan)
        for i, model in enumerate(models):
            for j, ds in enumerate(DATASETS):
                r = sub[(sub["model"] == model) & (sub["dataset"] == ds)]
                if not r.empty:
                    mat[i, j] = r.iloc[0]["auroc_mean"]
        im = ax.imshow(mat, cmap=CMAP, vmin=0.5, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(DATASETS)))
        ax.set_xticklabels(DATASETS, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=8)
        ax.set_title(f"{int(ratio * 100)}% train", fontsize=10, fontweight="bold")
        for i in range(len(models)):
            for j in range(len(DATASETS)):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(
                        j,
                        i,
                        f"{v:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color="white" if v > 0.75 else "#102422",
                    )
    fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.02, label="AUROC")
    fig.suptitle(
        "Figure 5. AUROC overview across training ratios",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.02,
        "Caption: Mean AUROC over seeds for each model–dataset pair. "
        "Grey cells indicate incomplete coverage for that setting.",
        ha="center",
        fontsize=8,
        color="#334433",
    )
    fig.tight_layout(rect=[0, 0.05, 0.95, 0.95])
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_label_efficiency(metrics: pd.DataFrame, out_path: Path) -> None:
    """AUROC vs training ratio for full-coverage models."""
    coverage = metrics.groupby("model")["dataset"].nunique()
    full_models = sorted(coverage[coverage >= 5].index.tolist())
    # drop partial for clarity in this figure
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6), sharey=True)
    palette = {
        "BWGNN": "#1a3c38",
        "GLHN": "#2f6f68",
        "GraphConsis": "#c47b3a",
        "PMP": "#3a5f8c",
        "SEC-GFD": "#6b8f71",
        "BSL": "#8c4a3a",
        "DGA-GNN": "#5c6b8c",
        "SplitGNN": "#7a6b4a",
    }
    for ax, ds in zip(axes, DATASETS):
        for model in full_models:
            ys = []
            xs = []
            for ratio in RATIOS:
                r = metrics[
                    (metrics["model"] == model)
                    & (metrics["dataset"] == ds)
                    & np.isclose(metrics["train_ratio"], ratio)
                ]
                if not r.empty and not pd.isna(r.iloc[0]["auroc_mean"]):
                    xs.append(int(ratio * 100))
                    ys.append(r.iloc[0]["auroc_mean"])
            if xs:
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    linewidth=1.6,
                    markersize=4,
                    label=model,
                    color=palette.get(model, "#444444"),
                )
        ax.set_title(ds, fontsize=10, fontweight="bold")
        ax.set_xlabel("Train %")
        ax.set_xticks([10, 20, 30, 40])
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("AUROC")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(len(full_models), 8),
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(
        "Figure 6. Label efficiency: AUROC vs training ratio",
        fontsize=12,
        fontweight="bold",
        y=1.12,
    )
    fig.text(
        0.5,
        -0.02,
        "Caption: How each fully evaluated model responds to more labeled "
        "training data. Steeper slopes indicate stronger label dependence; "
        "flat high curves indicate robust low-label behaviour.",
        ha="center",
        fontsize=8,
        color="#334433",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    path = DATA / "metrics_agg.csv"
    if not path.exists():
        raise SystemExit("Missing docs/data/metrics_agg.csv — run docs/aggregate_results.py first.")
    metrics = pd.read_csv(path)

    for ratio in RATIOS:
        pct = int(ratio * 100)
        plot_ratio_table(metrics, ratio, FIG / f"metrics_table_{pct}pct.png")
        print(f"Wrote metrics_table_{pct}pct.png")

    plot_metric_heatmap_grid(metrics, FIG / "auroc_overview.png")
    print("Wrote auroc_overview.png")
    plot_label_efficiency(metrics, FIG / "label_efficiency.png")
    print("Wrote label_efficiency.png")
    print(f"Figures saved under {FIG}")


if __name__ == "__main__":
    main()
