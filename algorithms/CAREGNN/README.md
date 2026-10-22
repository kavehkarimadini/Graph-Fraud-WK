# CARE-GNN

## Paper
The CARE-GNN model is proposed by the [paper](https://arxiv.org/pdf/2008.08692.pdf) below:
```bibtex
@inproceedings{dou2020enhancing,
  title={Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters},
  author={Dou, Yingtong and Liu, Zhiwei and Sun, Li and Deng, Yutong and Peng, Hao and Yu, Philip S},
  booktitle={Proceedings of the 29th ACM International Conference on Information and Knowledge Management (CIKM'20)},
  year={2020}
}
```

## Brief Introduction

CAmouflage-REsistant GNN (CARE-GNN) targets fraudsters who hide by connecting to benign entities. A linear label predictor scores every node; neighbours are ranked by the L1 distance between their score and the centre node's score, and only the closest `ceil(deg * p_r)` are kept under each relation. The per-relation thresholds `p_r` are adjusted by a reinforcement-learning rule (+/- `step_size` each epoch depending on whether the average neighbour distance of fraud nodes went down). Kept neighbours are mean-aggregated per relation, and relations are combined with `p_r` as weights. The loss is GNN cross-entropy plus `lambda_1` times the label-predictor cross-entropy, trained on under-sampled (balanced) batches.

This port follows `reference/CARE-GNN-master` (official code). Neighbour filtering is vectorised over CSR adjacency instead of per-node Python loops, so it scales to T-Finance / Elliptic; the selection rule is unchanged. Training uses the toolbox 40/20/40 split with valid-AUC checkpointing (the reference used a 40/60 train/test split and fixed 31 epochs).

## Run

From `algorithms/CAREGNN/`:

```powershell
python CAREGNN_main.py --dataset yelpchi --gpu 0
python CAREGNN_main.py --dataset amazon --gpu 0
python CAREGNN_main.py --dataset elliptic --gpu 0
python CAREGNN_main.py --dataset tfinance --gpu 0
python CAREGNN_main.py --dataset tsocial --gpu 0
python CAREGNN_main.py --dataset comp --gpu 0
```

`--inter` selects the inter-relation aggregator: `GNN` (CARE-GNN, default), `Mean`, `Weight` or `Att` (the paper's variants). `--no_rl` freezes the thresholds at `--init_threshold`. `--train_ratio` (default 0.4) shrinks only the train pool; valid/test stay 20/40.

## Note

- Paper hyperparameters: embedding 64, Adam `lr=0.01`, `lambda_1=2`, weight decay `1e-3`, RL step `0.02`, batch 1024 (YelpChi) / 256 (Amazon), under-sampling scale 1.
- YelpChi / Amazon / FDCompCN use their three relations; Elliptic / T-Finance / T-Social have a single relation, so CARE-GNN reduces to one filtered aggregator there.
- Each relation is symmetrised and given self-loops, as in the reference `sparse_to_adjlist`, so isolated nodes (common in FDCompCN) still aggregate themselves.
- Features are row-normalised on YelpChi as in the reference; datasets with signed features (Amazon, Elliptic, T-Finance, T-Social, FDCompCN) are standardised instead. Override with `--feat_norm`.
- Amazon splits only labeled nodes (`index >= 3305`); unlabeled users stay in the graph as neighbours.
- `config.json` records `final_thresholds` and the per-epoch `thresholds_log`.
- No DGL import.
