
# DGA-GNN

## Paper
The DGA-GNN model is proposed by the [paper](https://ojs.aaai.org/index.php/AAAI/article/view/29067) below:
```bibtex
@inproceedings{duan2024dgagnn,
  title={DGA-GNN: Dynamic Grouping Aggregation GNN for Fraud Detection},
  author={Duan, Mingjiang and Zheng, Tongya and Gao, Yang and Feng, Zunlei and Wang, Xinyu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2024}
}
```

## Brief Introduction

Dynamic Grouping Aggregation GNN (DGA-GNN) bins non-additive attributes with decision trees and aggregates neighbors in two feedback groups plus the full neighborhood, then fuses those messages with attention. This port follows the official AAAI 2024 code; message passing uses scatter-mean instead of DGL, and training uses the toolbox 40/20/40 split (no Lightning / wandb / Hydra).

## Run

From `algorithms/DGAGNN/`:

```powershell
python DGAGNN_main.py --dataset yelpchi --gpu 0
python DGAGNN_main.py --dataset amazon --gpu 0
python DGAGNN_main.py --dataset elliptic --gpu 0
python DGAGNN_main.py --dataset tfinance --gpu 0
python DGAGNN_main.py --dataset tsocial --gpu 0
python DGAGNN_main.py --dataset comp --gpu 0
```

Dataset defaults follow the official yaml files (`n_hidden`, `n_head`, bins `k`, grouping threshold `z`). Learning rate defaults to Adam `0.01`. `--train_ratio` (default 0.4) shrinks only the train pool; valid/test stay 20/40.

## Note

- Decision-tree binning uses scikit-learn (the official repo used `toad`). Pass `--no_bin_encoding` to skip it.
- T-Social (`--dataset tsocial`) uses the same loader as BWGNN (`dataset/tsocial`). This port runs full-graph DGA (no neighbor sampling), so memory is much higher than the official batched code. Defaults are reduced (`n_hidden=32`, `n_head=1`, `n_bins=8`, 100 epochs) versus the paper yaml (`128` / `2` / `32`).
- No DGL import.
