# BSL

## Paper
The BSL model is proposed by the [paper](https://ojs.aaai.org/index.php/AAAI/article/view/29593) below:
```bibtex
@inproceedings{yu2024bsl,
  title={Barely Supervised Learning for Graph-Based Fraud Detection},
  author={Yu, Hang and Liu, Zhengyang and Luo, Xiangfeng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  pages={16548--16557},
  year={2024}
}
```

## Brief Introduction

Barely Supervised Learning (BSL) encodes the graph with a GCN, splits the latent vector into NA / AA / NN subspaces, and trains edge-type classification plus an attention contrast so same-class subspaces dominate. Unlabeled nodes are then weakly and strongly augmented with labeled normal/fraud prototypes; consistency and inconsistency losses use those views. This port follows `reference/BSL/BSL.md`; GCN uses `torch.sparse` instead of DGL.

Default training uses the toolbox 40/20/40 split (valid-AUC checkpoint). Pass `--paper_split` for the paper's 1% / 10% / 89% setting (0.01% / 0.1% / rest on T-Social), which checkpoints on train loss because there is no validation set.

## Run

From `algorithms/BSL/`:

```powershell
python BSL_main.py --dataset yelpchi --gpu 0
python BSL_main.py --dataset amazon --gpu 0
python BSL_main.py --dataset elliptic --gpu 0
python BSL_main.py --dataset tfinance --gpu 0
python BSL_main.py --dataset tsocial --gpu 0
python BSL_main.py --dataset comp --gpu 0
python BSL_main.py --dataset yelpchi --paper_split --gpu 0
```

`--train_ratio` (default 0.4) shrinks only the train pool in the toolbox protocol. `--alpha` / `--beta` weight `L_D` and `L_bsl` (paper grid: 0.4 / 0.8).

## Note

- Paper hyperparameters: embedding 96, Adam `lr=0.01`, `weight_decay=1e-5`, unlabeled batch 256.
- Relations are unioned into one homogeneous graph (paper setting). Amazon still splits only labeled nodes (`index >= 3305`).
- T-Social uses a smaller hidden size (48) because this port is full-graph.
- No DGL import.
