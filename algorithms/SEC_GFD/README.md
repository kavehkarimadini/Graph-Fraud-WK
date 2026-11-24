# SEC-GFD

## Paper
SEC-GFD is proposed by the paper below. The released code is in `reference/SEC-GFD-main.zip`; the PDF is not in `paper/papers-to-read`.
```bibtex
@inproceedings{xu2024revisiting,
  title={Revisiting graph-based fraud detection in sight of heterophily and spectrum},
  author={Xu, Fan and Wang, Nan and Wu, Hao and Wen, Xuezhi and Zhao, Xibin and Wan, Hai},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={38},
  number={8},
  pages={9214--9222},
  year={2024}
}
```

## Brief Introduction

SEC-GFD has two parts:
- **Hybrid filtering module:** a two-layer MLP encoder, then Beta-wavelet band-pass filters of order C (as in BWGNN) and `high_order` high-pass filters `L, L^2, ...` with `L = I - D^-1/2 A D^-1/2`. The filter outputs are concatenated and classified by a two-layer MLP.
- **Local environmental constraint:** a 2-layer GCN, run on the graph without self-loops, maps the features to an embedding of the same size. An auxiliary loss, weighted by `lemda`, pushes the feature/embedding cosine similarity up for benign training nodes and down for fraud training nodes.

This is a pure-PyTorch (`torch.sparse`) re-implementation of the released code, with no DGL. It builds the same graphs:
- **YelpChi, Amazon:** all relations merged with duplicate edges kept, plus self-loops.
- **T-Finance, T-Social, Elliptic:** the stored graph as is.
- **FDCompCN:** its `homo` relation.

It uses the same model, the same class-weighted cross-entropy and Adam. All filters share one sequence of Laplacian powers, which gives the same outputs as the released code at a fraction of the cost.

Changes from the released code:
- **Environmental loss indexing:** the loss uses global training-node ids. The released `cal_nceloss` indexes by position inside `idx_train`, which is a bug.
- **Environmental loss averaging:** it averages `exp(cos)` as in the paper's equation, as the repo's earlier SEC-GFD port also did. Averaging raw cosines can make the log's argument negative.
- **Stopping rule:** instead of reporting the best *test* epoch of a fixed 100-epoch run, training stops early on validation AUPRC (patience 25), with at most 500 epochs.

## Run

From the repo root:

```bash
python algorithms/SEC_GFD/SEC_GFD_main.py --dataset amazon --data_prefix dataset/ --seed 123 --train_ratio 0.4 --gpu 0
python algorithms/SEC_GFD/SEC_GFD_main.py --dataset tsocial --data_prefix dataset/ --seed 123 --train_ratio 0.4 --gpu 0
```

Hyperparameters follow the root README's Hyperparameter Policy:
- **YelpChi, Amazon, T-Finance, T-Social:** the released code's values (Adam lr 0.01, weight decay 0, hidden 64, C = 2, `high_order` 2, `lemda` 0.2). T-Social uses hidden 16 and C = 5, kept from the repo's earlier SEC-GFD configuration.
- **Elliptic, FDCompCN:** use `configs/tuned/SEC_GFD/<dataset>.json` from `python tune_hparams.py --model SEC_GFD --dataset elliptic,comp`.

Sanity check (seed 123, 40% train, test AUROC):
- **Amazon:** 0.973; the earlier DGL port gave 0.970 ± 0.006 over 5 seeds.
- **YelpChi:** 0.857; the earlier port gave 0.849 ± 0.011.

## Note

The older DGL runner `SEC_GFD_benchmark.py` and its modules (`main.py`, `dataset.py`, `model/`, `utils.py`, `secgfd_config.py`, `secgfd_data.py`) are kept only to reproduce earlier results in `results/SEC_GFD/<dataset>/seed_<seed>/train_<pct>/`. They are not part of the benchmark.
