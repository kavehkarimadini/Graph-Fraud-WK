# COFRAUD

## Paper
The COFRAUD model is proposed by the [paper](https://doi.org/10.24963/ijcai.2023/551) below:
```bibtex
@inproceedings{zang2023cofraud,
  title={Don't Ignore Alienation and Marginalization: Correlating Fraud Detection},
  author={Zang, Yilong and Hu, Ruimin and Wang, Zheng and Xu, Danni and Wu, Jia and Li, Dengshi and Wu, Junhang and Ren, Lingfei},
  booktitle={Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence (IJCAI)},
  pages={4959--4966},
  year={2023},
  doi={10.24963/ijcai.2023/551}
}
```

## Brief Introduction

COFRAUD targets **synergistic camouflage**: fraudsters combine multi-relation behaviours (alienation of cross-relation neighbours + marginalization on each subgraph). The model has three modules on a multi-relation graph:

- **Intra-relation** (Eqs. 1–6): on each relation subgraph, project nodes / relations (`z_v = W_v h`, `z_r = W_r h_r` with `dim(z_r)=2·dim(z_v)`), learn frequency attention `α_ij = tanh(z_r^T [z_i ‖ z_j]) ∈ [-1,1]`, and aggregate with degree normalisation and a trainable residual `λ`:
  `c = (1-λ) W_align h + λ Σ_j α_ij/√(d_i d_j) z_j`. Relation features are updated by an MLP.
- **Inter-relation** (Eqs. 7–8): for each target relation `r2`, mix the relation-specific `c`'s with `β = softmax(LeakyReLU(q_{r2}^T c_{ri}))`.
- **Fusion** (Eqs. 9–10): attention over relations that also uses relation semantics `γ ∝ LeakyReLU((V_r h')^T U_r h'_r)`, then `h_final = Σ γ V_r h'`. A linear classifier is trained with cross-entropy.

Two stacked intra+inter layers; paper Table 3 best setting is hidden 32 / 2 layers.

This port follows `reference/COFRAUD.md`. Hyperparameters not stated in the paper (heads, dropout, lr, epochs, relation hidden size) are taken from the authors' released training script ([zyl199710/COFRAUD](https://github.com/zyl199710/COFRAUD)). Intra attention uses the **paper tanh** form; the released `RelationGraphConv` uses LeakyReLU+softmax instead. Against a naive dense / per-edge implementation of Eqs. 1–10 it matches logits to float32 precision. No DGL or PyG import.

## Run

From `algorithms/COFRAUD/`:

```powershell
python COFRAUD_main.py --dataset yelpchi --gpu 0
python COFRAUD_main.py --dataset amazon --gpu 0
python COFRAUD_main.py --dataset elliptic --gpu 0
python COFRAUD_main.py --dataset tfinance --gpu 0
python COFRAUD_main.py --dataset tsocial --gpu 0
python COFRAUD_main.py --dataset comp --gpu 0
```

`--train_ratio` (default 0.4) shrinks only the train pool; valid/test stay 20/40 (matches the paper's 40% setup with remaining valid:test = 1:2).

Ablations as in the paper:
- `--no_inter` skips inter-relation crossing (w/o inter).
- `--no_fusion` replaces Eqs. 9–10 with a uniform mean of `V_r h'_r` (w/o fusion).
- `--residual 0` drops the Eq. 5 residual term.

## Note

- **Paper settings** (Sec. 4): training ratios 10% / 40%; remaining split valid:test 1:2; select by max valid **F1** in the paper (this toolbox selects by valid **AUROC**); metrics AUC / macro-Rec / macro-F1; 2 layers, hidden 32; Amazon and Yelp only in the paper; CE loss; batch 1024 mentioned for timing.
- **From the official script** (documented defaults): 8 heads, relation embedding / hidden 16, dropout 0.5, residual on, Adam lr `0.02`, weight decay `0`, 500 epochs, patience 50. T-Social uses 4 heads / 200 epochs / patience 20.
- **Training** is full-batch over all relations (official uses neighbour sampling + batch 1280). Validation AUROC selects the checkpoint; predictions use threshold `0.5` (`--thres`).
- Relation embeddings are learnable vectors of size `--relation_dim` (default 16), not fixed one-hots; the paper initialises with one-hot encodings of relation type.
- **Smoke tests** (CPU, reduced `hidden=16` / `n_heads=2`): Amazon reached test AUROC **0.963** / AUPRC **0.876** / Macro-F1 **0.909** after 10 epochs (paper Table 2 @ 40%: 97.21 / Rec 89.08 / F1 91.53). YelpChi reached test AUROC **0.810** / AUPRC **0.489** / Macro-F1 **0.661** after 15 epochs and was still climbing (paper: 91.52 / 79.70 / 79.71)—short of a full 500-epoch 8-head run.
- Amazon splits only labeled nodes (`index >= 3305`); unlabeled users stay in the graph.
- No DGL or PyG import.
