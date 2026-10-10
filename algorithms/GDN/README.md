# GDN

## Paper
The GDN model is proposed by the [paper](https://doi.org/10.1145/3539597.3570377) below:
```bibtex
@inproceedings{gao2023gdn,
  title={Alleviating Structural Distribution Shift in Graph Anomaly Detection},
  author={Gao, Yuan and Wang, Xiang and He, Xiangnan and Liu, Zhenguang and Feng, Huamin and Zhang, Yongdong},
  booktitle={Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining (WSDM)},
  pages={357--365},
  year={2023}
}
```

## Brief Introduction

GDN (Graph Decomposition Network) targets **structural distribution shift**. Anomalies are a heterophilous minority, and how heterophilous their neighbourhoods are can differ between training and test nodes, so a GNN that absorbs neighbour signals generalises poorly to test anomalies. GDN splits the node features into two groups and constrains each group differently:

- **Backbone (multi-relation GNN):** a trainable node-feature table `X`. Each relation applies `relu([x_v, mean(x_N(v))] W_r)`, with the node itself among its neighbours. These are combined as `relu([x_v, h_1, ..., h_R] W)`, followed by a linear classifier trained with cross-entropy.
- **Feature separation:** the `topk` feature dimensions with the largest mean absolute cross-entropy gradient on `X` are the **anomaly (class) features**, recomputed every mini-batch. The rest are the **surrounding (normal) features**.
- **Anomaly-feature constraint:** on the anomaly dimensions, train fraud nodes are pulled towards a fraud prototype and pushed away from a normal prototype, using `beta * exp(KL(p_fraud || x) - KL(p_normal || x))`. The goal is anomaly features that stay stable regardless of heterophilous neighbours. Each prototype is the weighted mean of the train fraud (or normal) features, re-weighted every step by `softmax(cos(prototype, x) / 5)`.
- **Surrounding-feature constraint:** on the remaining dimensions, every fraud node in the batch is pulled towards its neighbourhood mean and pushed away from a random non-neighbour, in each relation. The non-neighbour is drawn from the nodes seen in the batch. This keeps the homophily signal for normal nodes.

Both constraints update only the feature table, which has its own small learning rate (`1e-5`) and strong L2 penalty (`0.1`).

This port follows `reference/wsdm_GDN-main`: `models/model.py`, `models/layers.py` (`InterAgg`, `IntraAgg`) and `model_handler.py`. The reference builds Python sets per node and a dense batch-by-unique-nodes mask for every relation. Here neighbour lists are CSR gathers with `index_add_` means, and negatives are drawn by vectorised rejection sampling over the same candidate set.

Tested against the reference modules on a random three-relation graph, one full training step reproduces the reference to float32 precision:
- cross-entropy loss;
- prototype updates (both the initial mean and the softmax re-weighting);
- selected anomaly dimensions;
- every gradient, including the feature table's.

The negative sampler's support equals the reference's candidate set.

## Run

From `algorithms/GDN/`:

```powershell
python GDN_main.py --dataset yelpchi --gpu 0
python GDN_main.py --dataset amazon --gpu 0
python GDN_main.py --dataset elliptic --gpu 0
python GDN_main.py --dataset tfinance --gpu 0
python GDN_main.py --dataset tsocial --gpu 0
python GDN_main.py --dataset comp --gpu 0
```

`--train_ratio` (default 0.4) shrinks only the train pool; valid/test stay 20/40. `--add_constraint 0` trains the backbone alone (ablation), and `--topk` / `--beta` set the feature split and the constraint weight.

## Note

- **Reference hyperparameters** (`config/gdn_yelpchi.yml`):
  - network: embedding 64, batch 1024, Adam with lr `1e-3` and L2 `1e-3`;
  - feature table: lr `1e-5`, L2 `0.1`;
  - constraints: `Beta 0.1`, `topk 10`;
  - decision threshold `0.2`; 101 epochs.

  The paper tunes `topk` over {6, 12, 18, 24} on YelpChi and {5, 10, 15, 20} on Amazon, and `beta` over {0.01, 0.1, 0.5, 1}. All datasets use the config values, except T-Finance and T-Social: they have only 10 features, so `topk` is 3 to keep both feature groups non-empty. `topk` is always capped at `features - 1`.
- **Features:** raw on YelpChi and row-normalised (`x / (sum + 0.01)`) on Amazon, as in `model_handler.py`. Elliptic, T-Finance, T-Social and FDCompCN are standardised.
- **Training schedule:** the reference iterates the train nodes in a fixed order each epoch, validates every 5 epochs and restores the best validation-AUROC checkpoint. Here the train nodes are permuted once with the seed and then kept in a fixed order. Validation runs every epoch, with early stopping patience 20 over at most 100 epochs (T-Social: 50 / 10), and the best checkpoint is restored.
- **Relations:** the reference hard-codes three relations. The port takes any number, so single-relation datasets use one intra-relation aggregator. Each relation is symmetrised and gets self-loops, as in `sparse_to_adjlist`.
- **Prototype updates:** the reference updates the prototypes in every forward pass, including validation and test passes. The port updates them once per training step, before the constraints. Predictions do not depend on the prototypes.
- **Dropped unused reference pieces:** `label_clf` and `cal_simi_scores` in `InterAgg` are never used, so they are omitted.
- **Batches without fraud nodes** skip the surrounding-feature constraint. The reference would produce an undefined (NaN) loss there.
- **Predictions:** probabilities are the softmax output. Hard predictions use `--thres` (default 0.2, reference).
- **`config.json`** records how often each feature dimension was selected as an anomaly feature (`anomaly_feature_counts`).
- **Reported results** (paper Table 3, AUC): YelpChi 0.9034, Amazon 0.9709.
- Amazon splits only labeled nodes (`index >= 3305`); unlabeled users stay in the graph.
- No DGL or PyG import.
