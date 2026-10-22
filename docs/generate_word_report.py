"""
Generate a Word (.docx) Results & Discussion report with embedded figures.

Usage (from repo root):
    python docs/aggregate_results.py
    python docs/visualize_metrics.py
    python docs/visualize_runtime_complexity.py
    python docs/generate_word_report.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Inches, Pt, RGBColor

DOCS = Path(__file__).resolve().parent
DATA = DOCS / "data"
FIG = DOCS / "figures"
OUT = DOCS / "Results_and_Discussion.docx"

DATASETS = ["Amazon", "YelpChi", "T-Finance", "Elliptic", "FDCompCN"]
METRICS = ["g_mean", "auroc", "auprc", "f1_macro"]
METRIC_LABELS = ["G-Mean", "AUROC", "AUPRC", "F1-Macro"]
RATIOS = [0.1, 0.2, 0.3, 0.4]


def set_cell_shading(cell, hex_color: str) -> None:
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), hex_color)
    shd.set(qn("w:val"), "clear")
    tcPr.append(shd)


def add_caption(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.italic = True
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x33, 0x44, 0x33)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(10)


def add_heading_styled(doc: Document, text: str, level: int = 1) -> None:
    doc.add_heading(text, level=level)


def add_body(doc: Document, text: str) -> None:
    p = doc.add_paragraph(text)
    p.paragraph_format.space_after = Pt(8)
    for run in p.runs:
        run.font.size = Pt(11)


def add_figure(doc: Document, path: Path, caption: str, width: float = 6.5) -> None:
    if not path.exists():
        add_body(doc, f"[Missing figure: {path.name}]")
        return
    doc.add_picture(str(path), width=Inches(width))
    last = doc.paragraphs[-1]
    last.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_caption(doc, caption)


def fmt(mean, std) -> str:
    if pd.isna(mean):
        return "—"
    if pd.isna(std):
        return f"{mean:.3f}"
    return f"{mean:.3f}±{std:.3f}"


def add_metrics_table(doc: Document, metrics: pd.DataFrame, ratio: float, table_no: int) -> None:
    sub = metrics[np.isclose(metrics["train_ratio"], ratio)]
    models = sorted(sub["model"].unique())
    n_cols = 1 + len(DATASETS) * len(METRICS)
    table = doc.add_table(rows=2 + len(models), cols=n_cols)
    table.style = "Table Grid"

    # Header row 1: datasets
    table.cell(0, 0).text = "Model"
    set_cell_shading(table.cell(0, 0), "1A3C38")
    col = 1
    for ds in DATASETS:
        cell = table.cell(0, col)
        cell.text = ds
        # merge across 4 metric columns
        end = table.cell(0, col + 3)
        cell.merge(end)
        set_cell_shading(cell, "1A3C38")
        for p in cell.paragraphs:
            for r in p.runs:
                r.font.bold = True
                r.font.size = Pt(8)
                r.font.color.rgb = RGBColor(255, 255, 255)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        col += 4

    # Header row 2: metrics
    table.cell(1, 0).text = ""
    set_cell_shading(table.cell(1, 0), "2F6F68")
    col = 1
    for _ in DATASETS:
        for lab in METRIC_LABELS:
            cell = table.cell(1, col)
            cell.text = lab
            set_cell_shading(cell, "2F6F68")
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.bold = True
                    r.font.size = Pt(7)
                    r.font.color.rgb = RGBColor(255, 255, 255)
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            col += 1

    for i, model in enumerate(models):
        row_i = i + 2
        table.cell(row_i, 0).text = model
        for p in table.cell(row_i, 0).paragraphs:
            for r in p.runs:
                r.font.bold = True
                r.font.size = Pt(8)
        msub = sub[sub["model"] == model]
        col = 1
        for ds in DATASETS:
            r = msub[msub["dataset"] == ds]
            for m in METRICS:
                cell = table.cell(row_i, col)
                if r.empty:
                    cell.text = "—"
                else:
                    cell.text = fmt(r.iloc[0][f"{m}_mean"], r.iloc[0][f"{m}_std"])
                for p in cell.paragraphs:
                    for run in p.runs:
                        run.font.size = Pt(7)
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                col += 1

    pct = int(ratio * 100)
    add_caption(
        doc,
        f"Table {table_no}. Mean ± std test metrics at {pct}% training ratio "
        f"across five datasets (G-Mean, AUROC, AUPRC, F1-Macro). Em dash = not run.",
    )


def add_simple_table(
    doc: Document,
    collapsed: pd.DataFrame,
    value_col: str,
    title_caption: str,
    value_fmt,
) -> None:
    models = sorted(collapsed["model"].unique())
    table = doc.add_table(rows=1 + len(models), cols=1 + len(DATASETS))
    table.style = "Table Grid"
    headers = ["Model"] + DATASETS
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = h
        set_cell_shading(cell, "3D220C" if "time" in title_caption.lower() or "epoch" in title_caption.lower() else "132433")
        for p in cell.paragraphs:
            for r in p.runs:
                r.font.bold = True
                r.font.size = Pt(8)
                r.font.color.rgb = RGBColor(255, 255, 255)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    for i, model in enumerate(models):
        table.cell(i + 1, 0).text = model
        for j, ds in enumerate(DATASETS):
            r = collapsed[(collapsed["model"] == model) & (collapsed["dataset"] == ds)]
            cell = table.cell(i + 1, j + 1)
            if r.empty or pd.isna(r.iloc[0][value_col]):
                cell.text = "—"
            else:
                cell.text = value_fmt(r.iloc[0][value_col])
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(8)
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_caption(doc, title_caption)


def build_discussion_paragraphs() -> list[tuple[str, int | None]]:
    """Return (text, heading_level|None) blocks."""
    return []


def main():
    metrics = pd.read_csv(DATA / "metrics_agg.csv")
    runtime = pd.read_csv(DATA / "runtime_agg.csv")
    collapsed = (
        runtime.groupby(["model", "dataset"], as_index=False)
        .agg(
            per_epoch_sec=("per_epoch_sec_mean", "mean"),
            num_parameters=("num_parameters_mean", "mean"),
        )
    )

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    title = doc.add_heading("Results and Discussion", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph(
        "Benchmarking Graph-Based Fraud Detection: Architecture Comparison Across "
        "Training Ratios, Datasets, Runtime and Complexity"
    )
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER

    add_heading_styled(doc, "1. Experimental setup", 1)
    add_body(
        doc,
        "We evaluate graph neural fraud detectors under a shared protocol: fixed "
        "validation (20%) and test (40%) pools, with the labeled training pool "
        "restricted to 10%, 20%, 30% or 40% of nodes (stratified seeds "
        "123–1234567). Reported scores are means ± standard deviation over five "
        "seeds. Primary metrics are G-Mean, AUROC, AUPRC and Macro-F1. The five "
        "benchmark graphs are Amazon, YelpChi, T-Finance, Elliptic and FDCompCN "
        "(T-Social excluded pending heterophily recomputation). Models with "
        "complete five-dataset coverage include BWGNN, GLHN, GraphConsis, PMP, "
        "SEC-GFD, BSL and DGA-GNN; SplitGNN is partial at 10%/40%; GAGA and GHRN "
        "cover a subset of datasets.",
    )

    add_heading_styled(doc, "2. Dataset characteristics and comparison", 1)
    add_body(
        doc,
        "The collection spans four application domains and deliberately "
        "orthogonal difficulty axes: class imbalance, local camouflage "
        "(heterophily), multi-relation structure, sparsity and scale "
        "(Dataset_Report_Full).",
    )
    add_body(
        doc,
        "YelpChi (45,954 reviews, 14.53% spam, 32 features) is a "
        "multi-relation review graph (R-U-R, R-T-R, R-S-R). Local heterophily is "
        "severe on the product/time relations (~0.80) but low on R-U-R (~0.075), "
        "so methods that treat relations equally dilute the informative "
        "reviewer link. Amazon (11,944 users, 6.87% fraud over all nodes / ~9.5% "
        "over labeled, 25 features) shows extreme local camouflage (~0.90) with "
        "dense multi-relation user links and a large unlabeled set retained for "
        "message passing. T-Finance (39,357 accounts, 4.58% fraud, 10 features) "
        "is a dense homogeneous transaction graph (~21.2M edges, ~539 edges/node) "
        "with comparatively milder local heterophily (0.456), so anomalies partly "
        "cluster. Elliptic (203,769 Bitcoin transactions, 2.23% illicit among all "
        "nodes, 166 features) is large but sparse (~1.15 edges/node) with many "
        "unknown labels kept in-graph; features already encode one-hop aggregates, "
        "so GNNs partly re-aggregate precomputed structure. FDCompCN (5,317 "
        "listed companies, 10.51% fraud, 57 features) is extremely sparse "
        "(~1.37 mean degree after removing self-loops) with severe camouflage "
        "(~0.86) and thousands of isolated firms for which a GNN reduces to a "
        "feature-only classifier.",
    )
    add_body(
        doc,
        "Comparing datasets: Amazon and YelpChi stress relation-aware and "
        "camouflage-resistant designs; T-Finance stresses scalability under "
        "density; Elliptic stresses scale under sparsity plus already-rich "
        "features; FDCompCN stresses structural sparsity and isolation. Fraud "
        "rate, camouflage and scale are largely independent axes—no single "
        "property predicts the others—so architecture winners shift by dataset.",
    )

    add_heading_styled(doc, "3. Performance tables by training ratio", 1)
    add_body(
        doc,
        "The following tables report mean ± std over seeds. Colour-coded static "
        "visualizations of the same tables appear as Figures 1–4.",
    )
    for i, ratio in enumerate(RATIOS, start=1):
        add_heading_styled(doc, f"3.{i} Training ratio {int(ratio * 100)}%", 2)
        add_metrics_table(doc, metrics, ratio, table_no=i)
        add_figure(
            doc,
            FIG / f"metrics_table_{int(ratio * 100)}pct.png",
            f"Figure {i}. Heat-coded metric table at {int(ratio * 100)}% training "
            f"ratio (darker teal = higher score within each metric column).",
            width=6.7,
        )

    add_figure(
        doc,
        FIG / "auroc_overview.png",
        "Figure 5. AUROC overview heatmaps across all training ratios.",
        width=6.5,
    )
    add_figure(
        doc,
        FIG / "label_efficiency.png",
        "Figure 6. Label efficiency curves (AUROC vs training ratio) for "
        "fully covered models.",
        width=6.5,
    )

    add_heading_styled(doc, "4. Architecture comparison and interpretation", 1)
    add_body(
        doc,
        "Spectral / high-frequency models. BWGNN’s beta-wavelet filters target "
        "band-pass anomaly signals and remain strong on Amazon, T-Finance and "
        "Elliptic even at 10% labels (AUROC typically >0.95). Gains plateau as "
        "labels increase, consistent with a spectral prior that needs fewer "
        "labels when anomalies are frequency-separable. SplitGNN extends this "
        "idea with signed multi-relation splits and is competitive on dense "
        "YelpChi/T-Finance at 20–30%, but coverage gaps and weaker Elliptic/"
        "FDCompCN scores show sensitivity to sparsity and incomplete runs.",
    )
    add_body(
        doc,
        "Relation-aware aggregators. GraphConsis was built for inconsistency "
        "across relations and delivers top-tier Amazon AUROC/AUPRC and strong "
        "T-Finance / Elliptic Macro-F1, aligning with multi-relation review and "
        "transaction semantics. PMP similarly ranks at or near the top on Amazon "
        "and T-Finance AUPRC / F1, indicating that prototype / message-passing "
        "refinements help under camouflage when neighbourhood structure is "
        "informative—but both are among the slowest per epoch on dense graphs.",
    )
    add_body(
        doc,
        "Homophily-aware and grouping models. GLHN mixes low- and high-pass "
        "filters using predicted local homophily and is consistently solid "
        "across datasets without extreme failures, with a notable relative "
        "advantage on FDCompCN G-Mean where isolation makes pure neighbourhood "
        "aggregation unreliable. DGA-GNN’s dynamic grouping aggregation "
        "dominates YelpChi and Elliptic (AUROC ≈0.98 / 0.99 at 40%) and leads "
        "FDCompCN AUROC/AUPRC, suggesting grouped aggregation helps when "
        "neighbourhoods are noisy or heterophilic; its higher parameter count "
        "reflects grouping heads and binning capacity.",
    )
    add_body(
        doc,
        "Efficiency-oriented detectors. SEC-GFD achieves competitive Amazon / "
        "T-Finance / Elliptic scores with very low per-epoch time and modest "
        "parameters—attractive when wall-clock budget matters. BSL’s barely-"
        "supervised disentanglement helps on Elliptic/T-Finance under scarce "
        "labels but collapses on YelpChi (AUROC ≈0.60), indicating that weak/"
        "strong unlabeled augmentations do not substitute for relation-selective "
        "modelling when camouflage is relation-specific. GAGA/GHRN (partial "
        "coverage) track BWGNN-like Amazon quality; GHRN is slower per epoch.",
    )
    add_body(
        doc,
        "Why winners differ by dataset. (1) Multi-relation camouflage (YelpChi, "
        "Amazon): relation-aware or grouping models (DGA-GNN, GraphConsis, PMP) "
        "outrun vanilla spectral filters on YelpChi AUPRC. (2) Dense mild-"
        "heterophily transactions (T-Finance): spectral and consistency models "
        "all saturate high AUROC; ranking shifts to AUPRC/F1 and runtime. "
        "(3) Sparse large graphs (Elliptic): DGA-GNN and BWGNN exploit structure "
        "without drowning in density cost. (4) Isolated sparse corporate graph "
        "(FDCompCN): absolute scores drop sharply (AUPRC often <0.4); methods "
        "that degrade gracefully to features (DGA-GNN, GraphConsis, GLHN) retain "
        "relative advantages, while structure-heavy filters cannot invent missing "
        "edges.",
    )

    add_heading_styled(doc, "5. Runtime and parameter complexity", 1)
    add_simple_table(
        doc,
        collapsed,
        "per_epoch_sec",
        "Table 5. Average per-epoch training time in seconds (mean over seeds "
        "and training ratios).",
        lambda v: f"{float(v):.3f}",
    )
    add_simple_table(
        doc,
        collapsed,
        "num_parameters",
        "Table 6. Trainable parameter counts by model and dataset.",
        lambda v: f"{int(round(float(v))):,}",
    )
    add_figure(
        doc,
        FIG / "runtime_table.png",
        "Figure 7. Visual table of average per-epoch training time (seconds).",
        width=6.2,
    )
    add_figure(
        doc,
        FIG / "params_table.png",
        "Figure 8. Visual table of trainable parameter counts.",
        width=6.2,
    )
    add_figure(
        doc,
        FIG / "runtime_bars.png",
        "Figure 9. Grouped per-epoch runtime bars (log scale).",
        width=6.5,
    )
    add_figure(
        doc,
        FIG / "params_bars.png",
        "Figure 10. Parameter complexity comparison (log scale).",
        width=6.0,
    )
    add_figure(
        doc,
        FIG / "runtime_vs_params.png",
        "Figure 11. Runtime versus parameter complexity by dataset.",
        width=5.8,
    )
    add_body(
        doc,
        "Runtime is driven more by graph density and spectral/sampling cost than "
        "by parameter count. GraphConsis and SplitGNN are expensive on T-Finance "
        "despite moderate or small parameter budgets; SEC-GFD and BSL remain "
        "fast across datasets. PMP and DGA-GNN combine larger capacity with "
        "higher epoch cost on Amazon/YelpChi. SplitGNN is the most compact "
        "parameterisation; PMP’s FDCompCN parameter blow-up reflects "
        "feature-dependent widths in that implementation.",
    )

    add_heading_styled(doc, "6. Conclusions", 1)
    add_body(
        doc,
        "No single architecture wins universally. Spectral anomaly filters "
        "(BWGNN) provide strong, label-efficient baselines on frequency-"
        "separable fraud; relation-aware and grouping models (GraphConsis, PMP, "
        "DGA-GNN) unlock YelpChi and further lift precision-oriented metrics; "
        "homophily-mixing (GLHN) and efficient spectral variants (SEC-GFD) offer "
        "balanced accuracy–cost trade-offs; barely-supervised designs need "
        "dataset conditions that match their assumptions. FDCompCN remains the "
        "hardest stress test because isolation removes the inductive bias GNNs "
        "rely on. Practitioners should match model family to the dominant "
        "dataset pathology—camouflage, density, sparsity or isolation—rather "
        "than expecting a global ranking.",
    )

    doc.save(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
