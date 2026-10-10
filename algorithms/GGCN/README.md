# GGCN

## Paper
The GGCN model is proposed by the [paper](https://arxiv.org/abs/2102.06462) below:
```bibtex
@inproceedings{yan2022two,
  title={Two Sides of the Same Coin: Heterophily and Oversmoothing in Graph Convolutional Neural Networks},
  author={Yan, Yujun and Hashemi, Milad and Swersky, Kevin and Yang, Yaoqing and Koutra, Danai},
  booktitle={2022 IEEE International Conference on Data Mining (ICDM)},
  pages={1287--1292},
  year={2022},
  organization={IEEE}
}
```

## Brief Introduction

GGCN (Generalised GCN) argues that heterophily and oversmoothing share a cause: GCN-style averaging pulls nodes towards neighbours of other classes (heterophily) or makes all nodes alike as depth grows (oversmoothing). It changes each graph convolution in three ways:

- **Signed messages:** every edge `(i, j)` is scored by the cosine similarity of the transformed endpoints `W h_i`, `W h_j`. Positive (similar) and negative (dissimilar) parts of the attention-weighted adjacency are propagated separately, and mixed with the node's own message by learnable softmax weights. A learnable softplus factor scales the result. Dissimilar (heterophilous) neighbours can therefore push a node away instead of pulling it closer.
- **Degree correction:** edge weights of the symmetric normalised adjacency `D^-1/2 (A + I) D^-1/2` are multiplied by `softplus(a * (A_ii / A_ij - 1) + b)` (that is, `sqrt(d_j / d_i) - 1` inside), with learnable `a, b` per layer. This compensates the different effect of low- and high-degree neighbours.
- **Decaying aggregation:** layer outputs are accumulated as `H <- H + log(decay_rate / (l + 2)^exponent + 1) * layer_l`, starting from `ELU(W_0 x)`. Deep layers contribute less, which limits oversmoothing.

Training is full-batch cross-entropy.

This port follows `reference/Heterophily_and_oversmoothing-main` (official code, `model.py`: `GGCN`, `GGCNlayer`, `GGCNlayer_SP`; `full-supervised.py`; `process.py`). The reference runs dense `N x N` matrices (or COO products for Pubmed), which cannot fit fraud graphs. Here every layer works on the edge list:
- edge cosine similarities are a chunked sampled dense-dense product;
- propagation is a CSR sparse product with learnable edge values;
- both have custom backward passes (CSR products with the transposed layout), so memory stays linear in the number of edges.

On a random two-relation graph with an isolated node, it reproduces the reference's dense `GGCN` outputs and gradients to float precision. This holds for the default setting, deeper decaying stacks, and with signed messages, degree correction or decay switched off.

## Run

From `algorithms/GGCN/`:

```powershell
python GGCN_main.py --dataset yelpchi --gpu 0
python GGCN_main.py --dataset amazon --gpu 0
python GGCN_main.py --dataset elliptic --gpu 0
python GGCN_main.py --dataset tfinance --gpu 0
python GGCN_main.py --dataset tsocial --gpu 0
python GGCN_main.py --dataset comp --gpu 0
```

`--train_ratio` (default 0.4) shrinks only the train pool; valid/test stay 20/40. `--no_sign`, `--no_degree` and `--no_decay` turn off the three components (ablations as in the paper).

## Note

- Reference hyperparameters (`full-supervised.py`): Adam lr `0.01`, up to 1500 epochs with patience 100, `decay_rate 1.0`, `exponent 3.0`, degree correction and signed messages on. The reference tunes layers / hidden size / dropout / weight decay per benchmark (`table_1_GGCN.sh`). All fraud datasets use its Squirrel setting (2 layers, hidden 32, dropout 0.5, weight decay `4e-3`), the largest heterophilous benchmark in the paper. T-Social uses 500 epochs with patience 50 for full-batch training on 5.8M nodes.
- GGCN is a homogeneous model. Every dataset uses the binary union of its relations (`--relation` selects a single one), with self-loops added by the normalisation, as in `sys_normalized_adjacency`. Signed attention skips self-loops, as in the reference (`e - diag(e)`). The node's own message enters through the third mixing weight.
- The reference row-normalises features (`preprocess_features`), which suits its bag-of-words benchmarks. On the fraud datasets it destroys the scale of numeric features (YelpChi valid AUROC about 0.6 versus about 0.8 with standardisation), so features are standardised by default. `--feat_norm row` restores the reference behaviour and falls back to standard scaling when features are negative.
- The reference minimises NLL without class weights and keeps the epoch with the lowest validation loss. For consistency with the rest of the toolbox, early stopping and checkpointing use validation AUROC, and predictions use argmax. The loss is unweighted by default. `--loss_weight balanced` adds inverse-frequency class weights, which can raise G-mean / F1 at the default threshold.
- The reference's row-normalised adjacency option (`--row_normalized_adj`) is not ported. The reference disables degree correction in that mode anyway.
- `--use_bn` / `--use_ln` add batch / layer norm between layers only when `--no_decay` is set, as in the reference.
- The reference computes cosine similarity as `<w_i, w_j> / max(|w_i| |w_j|, 1e-9)`; the port uses the same formula.
- Amazon splits only labeled nodes (`index >= 3305`); unlabeled users stay in the graph.
- `config.json` records the learned per-layer mixing weights (positive / negative / self), scales and degree coefficients.
- No DGL or PyG import.
