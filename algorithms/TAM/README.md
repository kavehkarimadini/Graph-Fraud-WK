# TAM

## Paper
The TAM model is proposed by the [paper](https://arxiv.org/abs/2306.00006) below:
```bibtex
@inproceedings{qiao2023truncated,
  title={Truncated Affinity Maximization: One-class Homophily Modeling for Graph Anomaly Detection},
  author={Qiao, Hezhe and Pang, Guansong},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```

## Brief Introduction

TAM (Truncated Affinity Maximization) is an **unsupervised** graph anomaly detector built on one-class homophily: normal nodes tend to be strongly connected to similar normal nodes, while anomalies are weakly similar to their neighbours. Its parts:

- **Local affinity:** a node's anomaly measure is the mean cosine similarity between its representation and its neighbours' (self-loop included) on the original graph. Low affinity means anomalous.
- **LAMNet:** a two-layer GCN trained without labels to maximise the summed local affinity over all nodes.
- **Non-homophily truncation (NSGT):** edges between dissimilar nodes would teach LAMNet to make anomalies look like their normal neighbours. Every node therefore drops the edges to neighbours whose raw-feature distance exceeds a random threshold between the graph-wide mean distance and its own maximum. Truncation is applied `K` times on each of `T` independent graph sequences, and a fresh LAMNet is trained on every truncated graph. The affinity, still measured on the original graph, is averaged over sequences and truncations, and the anomaly score is `1 - minmax(mean affinity)`.

This port follows `reference/TAM-master-main`: `model.py`, `utils.py` (`graph_nsgt`), and `train.py` / `train_large_graph.py`. The reference works on dense `N x N` matrices and loops over nodes in Python, which cannot handle fraud graphs. Here everything runs on edge lists:
- the affinity is `z_i . (A z)_i / d_i` for unit-norm `z`, a CSR sparse product;
- the GCN uses the CSR normalised adjacency of the truncated graph;
- truncation is a vectorised per-node maximum, threshold and mask, with reverse-edge indices for symmetrisation.

On a random two-relation graph, the port reproduces several reference functions exactly:
- edge distances (`calc_distance`);
- three cumulative truncation steps (`graph_nsgt`, fed the same random stream);
- the normalised adjacency (`normalize_adj_tensor`).

Losses and gradients (`max_message`, its min-max variant and `reg_edge`) match to float32 precision.

## Run

From `algorithms/TAM/`:

```powershell
python TAM_main.py --dataset yelpchi --gpu 0
python TAM_main.py --dataset amazon --gpu 0
python TAM_main.py --dataset elliptic --gpu 0
python TAM_main.py --dataset tfinance --gpu 0
python TAM_main.py --dataset tsocial --gpu 0
python TAM_main.py --dataset comp --gpu 0
```

Labels are never used for training. The 40/20/40 split only defines the evaluation sets. The training split also supplies the anomaly-rate estimate for the decision threshold, so `--train_ratio` barely affects results.

## Note

- **Reference hyperparameters:** embedding size 128, Adam with lr `1e-5`, 500 epochs per LAMNet, and `lamda` 0 (the non-neighbour regulariser is off on real data).
- **Truncation settings follow the graph type.** Both can be overridden with `--cutting` / `--n_tree`.
  - With the binary union of all relations (the default; analogous to the paper's Amazon-all / YelpChi-all), the large-graph setting applies: `K = 7` truncations, `T = 1`, and the affinity min-max normalised inside the loss (`train_large_graph.py`).
  - With a single relation (`--relation upu`, `--relation rur`, ...), the paper's default applies: `K = 4`, `T = 3`, plain loss (`train.py`).
- **Features:**
  - YelpChi and Amazon are row-normalised, as in the reference (`preprocess_features`). If any row sum is not positive, standard scaling is used instead.
  - T-Finance stays raw, as in the reference, and so does T-Social.
  - Elliptic and FDCompCN are standardised.
  - `--feat_norm` overrides these defaults.
- **Isolated nodes:** a node whose only neighbour is itself has affinity 1, so it is always ranked most normal. This matters on sparse relations. In YelpChi R-U-R, 82% of fraudsters are isolated versus 42% of normal reviews, so TAM scores below chance there (smoke test AUROC about 0.34). The union graph is the default for this reason.
- **Reported results** (paper, AUROC): Amazon 0.7064, Amazon-all 0.8476, YelpChi 0.5643, YelpChi-all 0.5818, T-Finance 0.6175.
- **No model selection.** TAM has no early stopping or model selection, as in the reference: the affinity from the final epoch of each LAMNet is used. The log shows the validation AUROC of the running ensemble after each truncation, but this is for information only.
- **Thresholding:** `probabilities` are `[1 - score, score]`. Hard predictions flag nodes whose score is at or above the training-node quantile `1 - (training anomaly rate)`; this is the only use of training labels.
- **Training time:** `epochs_executed` is `K * T * epochs`, and `train_time` covers the truncations and every LAMNet. The checkpoint stores all `K * T` LAMNets.
- **`config.json`** records the kept directed edges per sequence after each truncation, the running validation AUROC, and the threshold.
- **Amazon** evaluation uses labeled nodes only (`index >= 3305`); unlabeled users stay in the graph.
- **Memory:** it is linear in the number of edges (edge list, distances, CSR). T-Social's union graph has roughly 150M directed edges, so it needs a large GPU.
- No DGL or PyG import.
