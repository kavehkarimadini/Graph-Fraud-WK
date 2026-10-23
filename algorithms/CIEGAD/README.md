# CIE-GAD

## Paper
The CIE-GAD model is proposed by the [paper](https://doi.org/10.1109/TCYB.2025.3558941) below:
```bibtex
@article{huang2025correlation,
  title={Correlation Information Enhanced Graph Anomaly Detection via Hypergraph Transformation},
  author={Huang, Changqin and Gao, Chengling and Li, Ming and Li, Yongzhi and Wang, Xizhe and Jiang, Yunliang and Huang, Xiaodi},
  journal={IEEE Transactions on Cybernetics},
  volume={55},
  number={6},
  pages={2865--2878},
  year={2025},
  publisher={IEEE}
}
```

## Brief Introduction

CIE-GAD tackles two GAD issues—local heterophily and anomaly camouflage—with a spectral module and a hypergraph module on a homogeneous (union) graph:

- **Preliminary MLP** maps node features to embedding size `C`.
- **SCNAF** (Eqs. 8–15): Jacobi polynomial filters of order `0..K` on the symmetrically normalised adjacency `A_hat = D^{-1/2} A D^{-1/2}` produce `K+1` feature banks. A **node-level attention** mixes them: for each bank `k`, `w_k = tanh((Z_k W_k) · mean(Z_k))`, then `α^i = softmax_k(w_k^i)` and `Z̃ = ReLU(sum_k α_k ⊙ (Z_k W_k))`.
- **HCRE** (Eqs. 16–17): dual hypergraph transformation (original nodes → hyperedges, original edges → vertices) with equal vertex weights `Υ = I`. The hyperedge-centric convolution is
  `S = ReLU(D_e^{-1/2} H D_v^{-1/2} H^T Z̃ Θ)`.
  On an undirected graph without self-loops this operator equals `D^{-1/2}(A + D) / √2` and is applied with a shared CSR `spmm` (the `N × M` incidence matrix is never built).
- **Deep supervision** (Eqs. 18–21): separate linear classifiers on `S` (primary) and on `Z̃`, with `L = L_0 + γ L_1`.

**No official code was available.** This implementation follows the paper text (`reference/CIE-GAD.md`). Against a naive dense / per-edge implementation of the Jacobi banks, the node-level attention, and the dual-hypergraph operator (including the closed form `D^{-1/2}(A+D)/√2`), it matches `Z̃`, attention weights, `S`, and logits to float32 precision.

## Run

From `algorithms/CIEGAD/`:

```powershell
python CIEGAD_main.py --dataset yelpchi --gpu 0
python CIEGAD_main.py --dataset amazon --gpu 0
python CIEGAD_main.py --dataset elliptic --gpu 0
python CIEGAD_main.py --dataset tfinance --gpu 0
python CIEGAD_main.py --dataset tsocial --gpu 0
python CIEGAD_main.py --dataset comp --gpu 0
```

`--train_ratio` (default 0.4) shrinks only the train pool; valid/test stay 20/40.

Ablations as in the paper:
- `--no_naf` uses one scalar `α_k` per filter (w/o NAF).
- `--no_scnaf` drops Jacobi filtering and fuses a single projected MLP bank (w/o SCNAF).
- `--no_hcre` classifies from `Z̃` only (w/o HCRE).
- `--gamma 0` turns off deep supervision (w/o DS).

## Note

- **Paper settings** (Sec. V-B): embedding `C = 32`, up to 1000 epochs with early stopping, `γ = 1`, Adam, Optuna for the remaining hyperparameters; Jacobi order `K = 12` on T-Finance and `K = 6` elsewhere. Relations of YelpChi / Amazon are merged into one homogeneous graph.
- **Unspecified in the paper** (documented defaults):
  - Learning rate / weight decay: `1e-2` / `5e-5` (typical for full-batch spectral GAD models here). Early-stop patience is 100 (T-Social: 200 epochs / patience 20).
  - Jacobi weights `a, b`: default `1.0`, **learnable** and clamped to `> -1` (`--learn_jacobi 0` freezes them; `--jacobi_a` / `--jacobi_b` set the init / fixed values).
  - Preliminary MLP: `Linear-ReLU-Linear-ReLU` into `C`.
  - No dropout by default (`--dropout`).
- **Training** is full-batch. Validation AUROC selects the checkpoint; predictions use threshold `0.5` (`--thres`).
- **HCRE memory:** only the sparse adjacency is kept. Degrees come from `A 1`; isolated nodes produce a zero hyperedge embedding.
- **Smoke tests** (CPU): Amazon reached test AUROC **0.975** / AUPRC **0.875** / Macro-F1 **0.921** after 80 epochs (paper Table III: 98.69 / 92.84 / 93.33). YelpChi reached test AUROC **0.864** / AUPRC **0.560** / Macro-F1 **0.725** after 200 epochs (best valid @ 160; paper: 88.32 / 64.98 / 75.31)—still short of a full 1000-epoch Optuna run.
- Amazon splits only labeled nodes (`index >= 3305`); unlabeled users stay in the graph.
- No DGL or PyG import.
