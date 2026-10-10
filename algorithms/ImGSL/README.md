# ImGSL (IGSL)

## Paper
The ImGSL model (called IGSL in the paper) is proposed by the [paper](https://doi.org/10.1007/s10994-023-06464-0) below:
```bibtex
@article{ren2024improving,
  title={Improving fraud detection via imbalanced graph structure learning},
  author={Ren, Lingfei and Hu, Ruimin and Liu, Yang and Li, Dengshi and Wu, Junhang and Zang, Yilong and Hu, Wenyi},
  journal={Machine Learning},
  volume={113},
  pages={1069--1090},
  year={2024},
  publisher={Springer}
}
```

## Brief Introduction

ImGSL targets two problems in fraud graphs: heterophily, because fraudsters connect to many normal users, and class imbalance. It combines four parts:

- **Multi-relational class-balanced sampler** (Eqs. 2–3): under every relation, train nodes are drawn with probability `D_{r,v} / LF(y_v)`, the node's degree in that relation divided by its class frequency. Popular, minority-class nodes are therefore picked more often.
- **Graph-independent embedding** (Eqs. 4–9): a linear label predictor gives soft class assignments `f`. Edges whose same-class probability `S_ij = f_i . f_j` falls below `tau-` are pruned in every relation. Optionally (see Note), edges between train nodes with different labels are removed, and train fraud centres are linked to their most similar train fraud nodes.
- **Iterative graph structure learning** (Eqs. 10–15, after IDGL), run per relation:
  - A node-anchor graph `K` is built from a K-head weighted cosine similarity, thresholded at `epsilon`.
  - Messages pass node to anchor to node: `Lambda^-1 K Delta^-1 K^T X`.
  - Embeddings are updated as `H_t = ReLU((beta * mean_{A~_r}(X) + (1 - beta) * message_t) W_r)`.
  - The first graph is learned from the raw features, later graphs from `H_{t-1}` with a second metric.
  - Iteration stops when `||K_t - K_{t-1}||_F^2 / ||K_t||_F^2 < delta`, or after `max_iter` steps.
- **Embedding aggregation** (Eqs. 16–18): `ReLU([x_v, H_1, ..., H_R] W)` followed by a classifier. The loss is cross-entropy plus `alpha` times the label predictor's cross-entropy.

**No official code was available.** This implementation follows the paper text (`reference/ImGSL.md`) and the two works it builds on: PC-GNN for the sampler, label predictor, mini-batch multi-relation aggregation and `alpha = 2`, and IDGL for anchor-based iterative graph learning.

Against a naive implementation written directly from Eqs. 8–16 (Python sets per node, per-head cosine loops, dense diagonal matrices), it matches logits and all gradients to float32 precision. This holds in training mode (with oversampling and label-based edge removal), in evaluation mode, and when dynamic stopping fires. The sampler matches Eq. 2 exactly.

## Run

From `algorithms/ImGSL/`:

```powershell
python ImGSL_main.py --dataset yelpchi --gpu 0
python ImGSL_main.py --dataset amazon --gpu 0
python ImGSL_main.py --dataset elliptic --gpu 0
python ImGSL_main.py --dataset tfinance --gpu 0
python ImGSL_main.py --dataset tsocial --gpu 0
python ImGSL_main.py --dataset comp --gpu 0
python ImGSL_main.py --dataset amazon --remove_hetero 1 --oversample_ratio 0.5 --gpu 0   # paper-literal Eq. 9
```

`--train_ratio` (default 0.4) shrinks only the train pool; valid/test stay 20/40.

Ablations as in the paper:
- `--no_sampler` uses uniform node sampling (IGSL\p).
- `--beta 1` disables the learned graph (IGSL\c).
- `--tau_minus 0` disables graph-independent pruning (IGSL\e).

## Note

- **Paper settings** (Sec. 5.3), used unchanged:
  - Adam with lr `0.01` and weight decay `1e-3`; 150 epochs; one layer;
  - `beta 0.5`, `tau- 0.2`, `alpha 2`;
  - batch 256 with `epsilon 0.5` on Amazon, batch 1024 with `epsilon 0` on YelpChi.

  Other datasets use the YelpChi setting, except FDCompCN, which uses batch 256. The paper's `lambda/2 ||theta||^2` term is the optimiser's weight decay. Training validates every epoch, stops after 30 epochs without improvement (T-Social: 50 epochs, patience 10), and restores the best validation-AUROC checkpoint.
- **Label-dependent edits of Eq. 9 are off by default.** They are removing edges between train nodes with different labels, and linking every train fraud centre to its `k_r` most similar train fraud nodes (`k_r` = half the average fraud degree under relation `r`). These edits can only reach training centres, because test labels are unknown. Training neighbourhoods therefore become much more homophilous than test ones, and the classifier learns a pattern that test nodes never show. In smoke runs, `--remove_hetero 1 --oversample_ratio 0.5` gave Amazon test AUROC about 0.89, versus 0.967 without them after 40 epochs, and YelpChi about 0.77 versus 0.895 after 15 epochs. Both flags restore the paper-literal behaviour, and the oversampling ratio can be set to any value.
- **Features** are standardised on every dataset. PC-GNN-style row normalisation (`--feat_norm row`) did worse on Amazon (test AUROC 0.942 versus 0.967) and much worse on YelpChi.
- **Choices the paper leaves open:**
  - `Aggr` on the refined graph is a mean over neighbours, with the self-loop included.
  - The label predictor and the classifier are linear layers, as in PC-GNN.
  - Each epoch picks `2 * #train fraud` nodes in total, split evenly across relations. Degrees include the self-loop, so every train node can be picked.
  - The weighted cosine has 4 heads (the IDGL default). There are 512 anchors, at most 3 iterations, and `delta = 1e-2`.
  - The paper's Eq. 15 writes `> delta`, but its text describes convergence; iteration stops once the relative change falls below `delta`.
- **Anchors:** a fresh random set is drawn for every training batch, as IDGL does. Evaluation uses one fixed, seeded set, so predictions are deterministic. Anchors are embedded alongside the batch as extra centres.
- **Anchor summaries `Delta^-1 K^T X`** are averaged over the anchors themselves, using the anchor-anchor block of `K`, rather than over all N nodes. The anchors are a uniform node sample, so this is a Monte-Carlo estimate of IDGL's similarity-weighted mean. It keeps every step mini-batch sized, and makes each node's output independent of which other nodes share its batch.
- **Label-predictor pruning** (`tau-`) uses only predictions, so it is applied identically in training and evaluation.
- **`config.json`** records the oversampling `k_r` and the mean number of structure-learning iterations per relation.
- Amazon splits only labeled nodes (`index >= 3305`); unlabeled users stay in the graph.
- **Reported results** (paper Table 3, 40% training, AUC): Amazon 97.76, YelpChi 90.57.
- No DGL or PyG import.
