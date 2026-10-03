# H2-FDetector

## Paper
The H2-FDetector model is proposed by the [paper](https://dl.acm.org/doi/10.1145/3485447.3512195) below:
```bibtex
@inproceedings{h2fdetector_webconf22,
  title={H2-FDetector: A GNN-based Fraud Detector with Homophilic and Heterophilic Connections},
  author={Shi, Fengzhao and Cao, Yanan and Shang, Yanmin and Zhou, Yuchen and Zhou, Chuan and Wu, Jia},
  booktitle={Proceedings of the ACM Web Conference 2022 (TheWebConf)},
  pages={1486--1494},
  year={2022}
}
```

## Brief Introduction

H2-FDetector separates homophilic connections (same class) from heterophilic ones (fraud linked to benign). A relation-aware scorer `tanh(f([d(h_u), d(h_v), d(h_u) - d(h_v)]))` is trained with a hinge loss to output +1 on train edges whose endpoints share a label and -1 otherwise. During aggregation each message is multiplied by the sign of its edge score, so homophilic neighbours are attracted and heterophilic ones repelled; messages are combined with per-relation multi-head attention. Relation outputs are concatenated and mixed by a linear layer. A prototype loss pulls every train embedding towards the mean embedding of its class at each layer.

Loss: `CE(balanced train sample) + gamma1 * edge hinge loss + gamma2 * prototype loss`, summed over layers for the two auxiliary terms.

This port follows `reference/H2-FDetector-master` (official DGL code) with plain `torch` gather / scatter ops:

- Attention `atten([sign * h_src, h_dst])` is split into per-node source / destination terms (exact for the linear `atten`), followed by a segment softmax over each destination's incoming edges.
- Edge signs carry no gradient, so they are computed under `no_grad`: per edge in chunks while dropout is active (as in the reference), and from two per-node scalars at evaluation time.
- The hinge loss only scores the sampled train edges (all heterophilic train edges plus as many homophilic ones drawn with replacement), which is the subset the reference uses after scoring every `homo` edge.

## Run

From `algorithms/H2FDetector/`:

```powershell
python H2FDetector_main.py --dataset yelpchi --gpu 0
python H2FDetector_main.py --dataset amazon --gpu 0
python H2FDetector_main.py --dataset elliptic --gpu 0
python H2FDetector_main.py --dataset tfinance --gpu 0
python H2FDetector_main.py --dataset tsocial --gpu 0
python H2FDetector_main.py --dataset comp --gpu 0
```

`--train_ratio` (default 0.4) shrinks only the train pool; valid/test stay 20/40. `--n_layer 1` gives a single summed layer straight to the class logits, as in the reference.

## Note

- Paper / reference hyperparameters: full-batch Adam `lr=0.1`, weight decay `5e-5`, `intra_dim=8`, 4 heads, 2 layers, dropout 0.1, up to 1000 epochs. YelpChi uses `gamma1=1.2`, `gamma2=2`, early stop 50; Amazon uses `gamma1=0.4`, `gamma2=1.4`, early stop 200. Elliptic / T-Finance / FDCompCN reuse the YelpChi weights; T-Social is capped at 200 epochs (early stop 20) because of its size.
- The reference split is 40/20/40 stratified, the same as the toolbox protocol; selection uses the best valid AUC.
- Each relation is symmetrised, de-duplicated and given one self-loop per node (reference `data_preprocess.py`). The `homo` graph used for edge labels is the union of all relations.
- Features are row-normalised (`x / (rowsum + 0.01)`) as in the reference whenever every row sums to a positive value (YelpChi, Amazon); otherwise they are standardised. Elliptic / T-Finance / T-Social / FDCompCN default to standard scaling. Override with `--feat_norm`.
- Amazon splits only labeled nodes (`index >= 3305`); unlabeled users stay in the graph as neighbours.
- Elliptic / T-Finance / T-Social have a single relation, so the model reduces to one signed-attention aggregator per layer.
- Training on CPU is slow on large graphs (per-edge dropout in the sign scorer); use a GPU.
- `config.json` records the number of homophilic / heterophilic train edges (`train_edges_pos` / `train_edges_neg`).
- No DGL import.
