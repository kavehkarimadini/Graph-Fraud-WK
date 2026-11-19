
# BWGNN

## Paper
The BWGNN model is proposed by the [paper](https://proceedings.mlr.press/v162/tang22b.html) below:
```bibtex
@InProceedings{tang2022rethinking,
  title = {Rethinking Graph Neural Networks for Anomaly Detection},
  author = {Tang, Jianheng and Li, Jiajin and Gao, Ziqi and Li, Jia},
  booktitle = {International Conference on Machine Learning},
  year = {2022}
}
```

## Brief Introduction

Beta Wavelet Graph Neural Network (BWGNN) uses beta-kernel spectral filters to better capture high-frequency anomaly signals on graphs. This port follows the official ICML 2022 implementation; message passing uses `torch.sparse` instead of DGL.

## Run

From `algorithms/BWGNN/` after unzipping `dataset/tfinance.zip` / `dataset/tsocial.zip` and installing sympy (see the repo README uv section):

```powershell
python BWGNN_main.py --dataset tfinance --hid_dim 64 --order 2 --epochs 100 --gpu 0
python BWGNN_main.py --dataset tsocial --epochs 100 --gpu 0   # paper: h 64, C 5 (add --hid_dim 10 if memory is tight)
python BWGNN_main.py --dataset elliptic --hid_dim 64 --order 2 --epochs 100 --gpu 0
python BWGNN_main.py --dataset comp --hid_dim 64 --order 2 --epochs 100 --gpu 0
```

`--train_ratio 0.4` matches the paper (40% train). Per-dataset defaults follow paper Sec. 4.1 (`h = 64`; `C = 2`, or `C = 5` on T-Social). Elliptic and FDCompCN are not in the paper and use `configs/tuned/BWGNN/<dataset>.json` (root README, Hyperparameter Policy). T-Social is large; pass `--hid_dim 10` if the paper's `h = 64` does not fit in GPU memory.

## Note

- Requires `sympy`. T-Finance / T-Social DGL v2 binaries are read without importing DGL.
- Files are named `dataset/tfinance` and `dataset/tsocial` (no file extension). FDCompCN is `dataset/comp.dgl`.
