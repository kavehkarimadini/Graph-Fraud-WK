# FRAUDRE

## Paper
The FRAUDRE model is proposed by the [paper](https://ieeexplore.ieee.org/document/9679178) below:
```bibtex
@inproceedings{zhang2021fraudre,
  title={FRAUDRE: Fraud Detection Dual-Resistant to Graph Inconsistency and Imbalance},
  author={Zhang, Ge and Wu, Jia and Yang, Jian and Beheshti, Amin and Xue, Shan and Zhou, Chuan and Sheng, Quan Z},
  booktitle={2021 IEEE International Conference on Data Mining (ICDM)},
  pages={867--876},
  year={2021}
}
```

## Brief Introduction

FRAUDRE targets two problems of graph fraud detection: inconsistency (camouflaged fraudsters whose features, topology and relations disagree with their neighbours) and class imbalance.

- **Ego-feature embedding:** `h0 = relu(W x)` keeps each node's own features separate from its neighbourhood.
- **Fraud-aware convolution:** under every relation a node takes the mean of its neighbours and the difference `self - mean` (so inconsistent neighbourhoods stay visible instead of being smoothed away). Relations are mixed with learnable per-dimension softmax weights, and the result is concatenated to the node's representation. Two layers give `d -> 3d -> 7d`; each layer aggregates only the part produced by the previous convolution.
- **Imbalance-aware classifier:** a two-layer head (`7d -> 64 -> 2`, LeakyReLU 0.3) trained with cross-entropy on logits shifted by `log(prior)` (class frequencies of the train set); inference uses the raw logits.

This port follows `reference/FRAUDRE-main` (official code). Neighbour lists are CSR arrays and aggregation uses vectorised gathers / `index_add_` over each batch's K-hop sampled neighbourhood instead of Python sets, so any number of relations (one for Elliptic / T-Finance / T-Social) and layers works. On a random graph with copied weights it reproduces the reference `InterAgg` / `MODEL` outputs to float precision.

## Run

From `algorithms/FRAUDRE/`:

```powershell
python FRAUDRE_main.py --dataset yelpchi --gpu 0
python FRAUDRE_main.py --dataset amazon --gpu 0
python FRAUDRE_main.py --dataset elliptic --gpu 0
python FRAUDRE_main.py --dataset tfinance --gpu 0
python FRAUDRE_main.py --dataset tsocial --gpu 0
python FRAUDRE_main.py --dataset comp --gpu 0
```

`--train_ratio` (default 0.4) shrinks only the train pool; valid/test stay 20/40. `--n_layers` sets the number of convolutions (reference: 2).

## Note

- Reference hyperparameters: embedding 64, classifier hidden 64, Adam with weight decay `1e-4`, lr `0.1` / batch 256 on Amazon and lr `0.001` / batch 1024 on YelpChi, 61 epochs. Elliptic / T-Finance / T-Social use lr `0.01`, batch 1024; FDCompCN lr `0.01`, batch 256.
- Each relation is symmetrised and given self-loops; every node then keeps a fixed random sample of at most `--max_neighbors` (10) neighbours, drawn once before training as in the reference `sparse_to_adjlist` (seeded per relation).
- Training iterates over all train nodes in shuffled mini-batches (no under-sampling; the prior shift handles imbalance). The reference evaluated the test set every 10 epochs with a 20/80 (YelpChi) or 10/90 (Amazon) split; here the toolbox 40/20/40 split is used with valid-AUC checkpointing and early stopping.
- Probabilities are `softmax(logits)` (the reference reported `sigmoid` of the fraud logit); hard predictions are identical.
- Features are row-normalised (`x / rowsum`) as in the reference whenever every row sums to a positive value (YelpChi, Amazon); Elliptic / T-Finance / T-Social / FDCompCN default to standard scaling. Override with `--feat_norm`.
- The reference's unused auxiliary MLP head (`weight_mlp`, its loss is commented out) is omitted.
- Amazon splits only labeled nodes (`index >= 3305`); unlabeled users stay in the graph as neighbours.
- `config.json` records the train-set class prior (`train_prior`).
- No DGL import.
