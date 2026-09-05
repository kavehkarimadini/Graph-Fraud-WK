
# SplitGNN

## Paper
The SplitGNN model is proposed by the [paper](https://dl.acm.org/doi/10.1145/3583780.3615067) below:
```bibtex
@inproceedings{10.1145/3583780.3615067,
  author = {Wu, Bin and Yao, Xinyu and Zhang, Boyan and Chao, Kuo-Ming and Li, Yinsheng},
  title = {SplitGNN: Spectral Graph Neural Network for Fraud Detection against Heterophily},
  year = {2023},
  url = {https://doi.org/10.1145/3583780.3615067},
  booktitle = {Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
  series = {CIKM '23}
}
```

## Brief Introduction

SplitGNN splits signed multi-relation edges and applies beta-wavelet filters on the full / positive / negative graphs. This port follows the official CIKM 2023 implementation; message passing uses `torch.sparse` instead of DGL.

## Run

From `algorithms/SplitGNN/` after placing `dataset/comp.dgl` (and optionally unzipping YelpChi / Amazon) and installing sympy:

```powershell
python SplitGNN_main.py --dataset comp --gpu 0
python SplitGNN_main.py --dataset yelpchi --gpu 0
python SplitGNN_main.py --dataset amazon --gpu 0
```

Defaults match the paper table (FDCompCN: `lr=0.001`, `gamma=0.8`, `C=3`, `K=2`, `intra_dim=8`, early stop 100). `comp.dgl` train/valid/test masks are used unless you pass `--ignore_file_masks`.

## Note

- Requires `sympy`. FDCompCN is read from `dataset/comp.dgl` without importing DGL.
- Amazon drops feature column 19 and trains only on labeled nodes (index >= 3305), matching the official script.
