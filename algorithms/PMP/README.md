# PMP

## Paper
The PMP model is proposed by the [paper](https://github.com/Xtra-Computing/PMP) below (PDF: `paper/papers-to-read/PMP.pdf`):
```bibtex
@inproceedings{zhuo2024partitioning,
  title = {Partitioning Message Passing for Graph Fraud Detection},
  author = {Zhuo, Wei and Liu, Zemin and Hooi, Bryan and He, Bingsheng and Tan, Guang and Fathony, Rizal and Chen, Jia},
  booktitle = {International Conference on Learning Representations},
  year = {2024}
}
```

## Brief Introduction

Partitioning Message Passing (PMP) aggregates a node's neighbours with different functions depending on their class: fraud and benign training neighbours each get their own weight matrix. Unlabelled neighbours get a node-specific mix of the two, `alpha_i W_fr + (1 - alpha_i) W_be` with `alpha_i = sigmoid(Phi(h_i))`. Both weight matrices are generated per centre node, `W_fr,i = Psi_fr(h_i)`. Each relation is propagated separately; the outputs are concatenated and mixed by an MLP.

This implementation follows the paper (Eqs. 3–5, Algorithm 1, App. D.4). Where the paper is silent, it takes the details from the official code (`LA-SAGE-S`):
- neighbour messages are summed;
- the generated weights are implemented as `(h_j * g(h_i)) W`;
- the layer's parameters are shared across relations;
- the official feature normalisation `x / (sum(x) + 0.01)` is used;
- the output layer is dropout followed by a 2-class linear head.

`alpha_i` uses the paper's single-layer `Phi`.

It is pure PyTorch, with no DGL. With the paper's single layer, the label-partitioned neighbour sums depend only on the input features and the training labels. They are therefore precomputed once with `scipy.sparse`, and training runs on mini-batches of nodes, so it scales to T-Social. With `--num_layers` > 1 the model propagates over the full graph instead.

## Run

From the repo root:

```bash
python algorithms/PMP/PMP_main.py --dataset amazon --data_prefix dataset/ --seed 123 --train_ratio 0.4 --gpu 0
python algorithms/PMP/PMP_main.py --dataset tsocial --data_prefix dataset/ --seed 123 --train_ratio 0.4 --gpu 0
```

Hyperparameters follow the root README's Hyperparameter Policy:
- **YelpChi, Amazon, T-Finance, T-Social:** the paper's App. D.4 Table 5 values (Adam, L = 1).

  | Dataset | lr | wd | dropout | batch | hidden |
  | --- | --- | --- | --- | --- | --- |
  | YelpChi | 0.01 | 0 | 0 | 512 | 256 |
  | Amazon | 0.01 | 0 | 0.6 | 128 | 128 |
  | T-Finance | 0.01 | 0 | 0.4 | 256 | 64 |
  | T-Social | 0.001 | 0 | 0 | 2^17 | 128 |
- **Elliptic, FDCompCN:** not in the paper, so they use `configs/tuned/PMP/<dataset>.json` from `python tune_hparams.py --model PMP --dataset elliptic,comp`. The search ranges are the paper's App. D.4 ranges.

The paper gives no epoch budget. Training stops early on validation AUROC (patience 25), with at most 500 epochs.

Sanity check (seed 123, 40% train, test AUROC): Amazon 0.980 (paper 0.9757), YelpChi 0.934 (paper 0.9397).

## Note

The older DGL runner `PMP_benchmark.py` and its modules (`LASAGE_S.py`, `SAGE.py`, `base.py`, `pmp_config.py`, `pmp_data.py`) are kept only to reproduce earlier results in `results/PMP/<dataset>/seed_<seed>/train_<pct>/`. They are not part of the benchmark.
