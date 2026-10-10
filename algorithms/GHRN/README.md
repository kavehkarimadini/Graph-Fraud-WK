# GHRN

## Paper
The GHRN model is proposed by the [paper](https://dl.acm.org/doi/10.1145/3543507.3583268) below:
```bibtex
@inproceedings{gao2023ghrn,
  title={Addressing Heterophily in Graph Anomaly Detection: A Perspective of Graph Spectrum},
  author={Gao, Yuan and Wang, Xiang and He, Xiangnan and Liu, Zhenguang and Feng, Huamin and Zhang, Yongdong},
  booktitle={Proceedings of the ACM Web Conference 2023 (WWW)},
  pages={1528--1538},
  year={2023}
}
```

## Brief Introduction

GHRN (Graph Heterophily Reduction Network) links heterophily (edges between a fraudster and a normal node) to the high-frequency part of the graph spectrum, and removes heterophilous edges before training a spectral detector.

- **Stage 1, detector:** BWGNN, a bank of beta-wavelet band-pass filters on `L = I - D^-1/2 A D^-1/2`, is trained on the original graph with class-weighted cross-entropy.
- **Indicator:** with the predicted class probabilities `Y`, the post-aggregation residual `LY = Y - A_hat Y` (normalised adjacency with self-loops) measures how much each node's prediction differs from its neighbourhood. Each edge is scored by `<LY_u, LY_v>`. A small (negative) score means the two endpoints deviate in opposite directions, so the edge is likely inter-class.
- **Pruning:** the `del_ratio` fraction of edges with the smallest scores is deleted (`0.1` on YelpChi, `0.015` elsewhere).
- **Stage 2:** a fresh BWGNN is trained on the pruned graph; its predictions are reported.

`--homo 1` (default) trains BWGNN on the union of all relations. `--homo 0` uses the per-relation BWGNN (reference `BWGNN_Hetero`) and prunes every relation separately.

This port follows `reference/GHRN-main` (official DGL code, `main.py`, `BWGNN.py`, `dataset.py`). The two runs of the reference (`--del_ratio 0` to save predictions, then `--del_ratio r` to load them) are chained in one script. Graph filtering uses `torch.sparse` CSR products (with a transposed operator for the backward pass), and the `order + 1` wavelet filters share one set of Laplacian powers because they all take the same input. On a random multigraph it reproduces a dense transcription of the reference forward pass and gradients for both variants to float precision. It also deletes exactly the same edges as a transcription of `random_walk_update`. Wavelet coefficients are computed in closed form and match the reference's `sympy` construction.

## Run

From `algorithms/GHRN/`:

```powershell
python GHRN_main.py --dataset yelpchi --gpu 0
python GHRN_main.py --dataset amazon --gpu 0
python GHRN_main.py --dataset elliptic --gpu 0
python GHRN_main.py --dataset tfinance --gpu 0
python GHRN_main.py --dataset tsocial --gpu 0
python GHRN_main.py --dataset comp --gpu 0
python GHRN_main.py --dataset yelpchi --homo 0 --gpu 0   # per-relation BWGNN
```

`--train_ratio` (default 0.4) shrinks only the train pool; valid/test stay 20/40. `--del_ratio 0` reduces the run to plain BWGNN.

## Note

- Reference hyperparameters (`main.py`): hidden 64, wavelet order 2, Adam lr `0.01` without weight decay, 100 epochs per stage, cross-entropy weight `#normal / #fraud` on the training nodes, symmetric indicator (`--adj_type sym`). The deleting ratio follows the paper's sensitivity analysis (Sec. 4.5): about `0.1` on YelpChi and `0.015` on Amazon and T-Social, otherwise about half the edge heterophily. T-Finance, Elliptic and FDCompCN therefore use `0.015` (the reference README example). GADBench searches `del_ratio` in `[0.01, 0.1]`, so it is worth tuning per dataset. T-Social uses BWGNN's paper setting (hidden 10, order 5) to keep full-batch training in memory.
- Graphs follow the reference loaders. YelpChi, Amazon and FDCompCN use the DGL `FraudDataset` path: every relation is made binary, symmetric and free of self-loops, the union keeps one copy per relation (`dgl.to_homogeneous`, so an edge in two relations counts twice), and one self-loop per node is added (`--self_loop`). T-Finance, T-Social and Elliptic are single graphs used without self-loops. Per-relation graphs (`--homo 0`) have no self-loops.
- Pruning, as in `random_walk_update`: self-loops are added, `int(del_ratio * E)` of all edges (self-loops included) with the smallest indicator are deleted, then the self-loops are removed and, for the union graph, re-added. Indicator weights follow DGL `EdgeWeightNorm`: `1 / sqrt(out_deg(u) in_deg(v))` for `sym`, `1 / out_deg(u)` for `rw`. Ties are broken by a stable sort.
- Stage 1 predictions are those of the best-validation checkpoint. The reference loaded the predictions saved at epoch 100, which are also those of its best-validation (lowest valid loss) epoch. Both stages start from the same seed, as the reference's two separate runs did.
- The reference kept the epoch with the lowest validation loss and reported test metrics at the validation-tuned macro-F1 threshold. For consistency with the rest of the toolbox, each stage here early-stops on validation AUROC (patience 25), and predictions use argmax.
- Features are raw for YelpChi, Amazon, T-Finance and T-Social, as in the reference. Elliptic and FDCompCN are standardised. Override with `--feat_norm`.
- The reference split is random 40/20/40 (DGL `FraudDataset` masks, or `train_test_split` with `random_state=2`). Here the toolbox's stratified protocol is used. Amazon splits only labeled nodes (`index >= 3305`); unlabeled users stay in the graph.
- `train_time_sec` and `epochs_executed` cover both stages plus pruning, so the per-epoch time is comparable with single-stage models. `config.json` records the stage-1 (BWGNN) metrics, edge counts before and after pruning, and the inter-class share of labeled edges before and after pruning (a diagnostic only, not used in training).
- No DGL or sympy import.
