# H2IDE

## Paper
The H2IDE (H²IDE) model is proposed by the [paper](https://doi.org/10.1109/TKDE.2024.3523107) below:
```bibtex
@article{fu2025nowhere,
  title={Nowhere to H$^2$IDE: Fraud Detection From Multi-Relation Graphs via Disentangled Homophily and Heterophily Identification},
  author={Fu, Chao and Liu, Guannan and Yuan, Kun and Wu, Junjie},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  volume={37},
  number={3},
  pages={1380--1393},
  year={2025},
  publisher={IEEE}
}
```

## Brief Introduction

H2IDE extends H2-FDetector's idea of separating homophilic from heterophilic edges to the level of latent factors: normal users connect for specific reasons (shared interests, similar ratings), while fraudsters' camouflage links follow no factor. It has four parts:

- **Disentangled transformation** (Eq. 2): every node gets `K` factor embeddings, `h_{u,k} = norm(LeakyReLU(W_k x_u + b_k))`.
- **Homophily and heterophily identification** (Eqs. 3–6):
  - the factor-level edge likelihood is `q(v|u,k) = cos(h_{u,k}, h_{v,k}) / tau`;
  - the factor distribution `p(k|u) = softmax_k |phi(h_{u,k}, c_{u,k})|` compares each factor with the neighbour prototype `c_{u,k}` (mean over all relations);
  - `Q(v|u) = sum_k p(k|u) q(v|u,k)` is trained with a hinge loss on edges whose two endpoints are labeled (+1 same class, -1 different class).
- **Relation-aware aggregation** (Eqs. 7–14):
  - per relation and factor, `m = sign(q(v|u,k))` marks a neighbour as homophilic or heterophilic;
  - signed GAT attention `a [W h_u || W m h_v]` (softmax over the relation neighbourhood) gives `h^r_{u,k} = LeakyReLU(sum_v alpha m h_{v,k})`, so heterophilic neighbours push instead of pull;
  - relations are mixed by `h_{u,k} = sum_r beta_{r,k} * s(k|u,r) * h^r_{u,k}`, where `s(k|u,r)` is a softmax over factors comparing `h^r_{u,k}` with the signed prototype of the relation neighbours.
- **Independence constraint** (Eqs. 15–18): the CLUB upper bound of the mutual information between every ordered pair of factors, with a Gaussian variational network `o(h_i | h_j)` per pair.

The loss is `CE + alpha1 * L_Disen + alpha2 * L_MI + alpha3 * ||Theta||^2` (Eq. 21), with the normal class under-sampled to the size of the fraud class.

**No official code was available.** This implementation follows the paper text (`reference/H2IDE.md`). Against a naive per-node implementation of Eqs. 2–14 (Python neighbour sets and loops over relations, factors and neighbours), the mini-batch path with exact two-hop neighbourhoods and the chunked full-graph inference both match the logits to float64 precision. The identification score `Q` and the CLUB estimate (against an explicit `N x N` negative term) match as well.

## Run

From `algorithms/H2IDE/`:

```powershell
python H2IDE_main.py --dataset yelpchi --gpu 0
python H2IDE_main.py --dataset amazon --gpu 0
python H2IDE_main.py --dataset elliptic --gpu 0
python H2IDE_main.py --dataset tfinance --gpu 0
python H2IDE_main.py --dataset tsocial --gpu 0
python H2IDE_main.py --dataset comp --gpu 0
```

`--train_ratio` (default 0.4) shrinks only the train pool; valid/test stay 20/40.

Ablations as in the paper: `--alpha1 0` drops the identification loss (H2IDE w/o HI keeps `m = sign(q)` from untrained factors, so it is only an approximation), `--alpha2 0` drops the independence constraint (w/o IC), and `--n_factors 1` approximates w/o DT.

## Note

- **Paper settings** (Sec. V-A4), used unchanged:
  - Adam with lr `0.005` and weight decay `5e-5`; one layer; 500 epochs; factor dimension 8; dropout 0.1; `beta = 1`; `alpha3 = 1e-4`.
  - YelpChi: `alpha1 1.2`, `alpha2 0.001`, batch 256, `K = 12`. Amazon: `alpha1 0.8`, `alpha2 0.01`, batch 1024, `K = 8`. T-Finance / T-Social: `alpha1 1.2`, `alpha2 0.01`, batch 1024, `K = 8`.
  - Elliptic uses the T-Finance setting and FDCompCN the Amazon setting.
- **Training loop:** each epoch draws all train frauds plus an equal number of train normals and splits them into batches. On Amazon that is a single step per epoch, so short runs are undertrained: validation AUROC sat at 0.90 for the first 30 epochs and reached 0.959 by epoch 100. Training validates every epoch, stops after 100 epochs without improvement (T-Social: 50 epochs, patience 10), and restores the best validation-AUROC checkpoint. Predictions use threshold 0.5 (`--thres`).
- **Neighbour sampling:** Eqs. 11–12 need the relation embeddings of every neighbour, so each batch needs a two-hop computation graph. Training samples up to 25 neighbours per relation at hop 1 and 10 at hop 2 (`--fanouts`, GraphSAGE-style draws with replacement; `0` keeps all). Evaluation is exact: relation embeddings of all nodes are computed layer-wise with full neighbourhoods, in chunks bounded by `--eval_edge_budget` edges.
- **Choices the paper leaves open:**
  - `tau` is not given; the default is 0.5 (`--tau`). It does not change the signs `m`, only the scale of `Q` against the hinge margin and the sharpness of `p(k|u)` and `s(k|u,r)`.
  - The activation in Eqs. 2 and 10 is LeakyReLU (slope 0.2), so negative signed sums survive.
  - Every relation keeps a self-loop in the attention (`q(u|u,k) > 0`, so it is homophilic). Prototypes (Eqs. 4 and 12) exclude the centre.
  - `N(u)` in Eq. 4 is the multiset of neighbours over all relations, since the paper defines a multi-edged graph.
  - Eq. 12's sign `m^{l+1}` is taken between the relation embeddings `h^r_u` and `h^r_v`, which avoids the circular dependence on `h^{l+1}`.
  - `beta_{r,k}` starts at 1. The MLP is `Linear-ReLU-Linear` with dropout on its input; dropout is also applied to the raw features.
  - The CLUB networks follow the official CLUB code: two-layer MLPs with hidden size 16 (`--mi_hidden`) for the mean and a `tanh`-bounded log-variance. Its `N x N` negative term is computed in closed form.
  - The paper updates the estimator and the model "iteratively". Each step first fits the estimators (separate Adam) on detached embeddings by maximising `beta * F` (Eq. 17), then the model minimises `alpha2` times the CLUB estimate with the estimators fixed.
  - Eq. 21's `||Theta||_2` is applied as a squared L2 term on the model parameters (not the estimators), on top of Adam's weight decay, as the paper lists both.
  - The hinge loss is averaged over all labeled edges of the batch, without balancing homophilic and heterophilic edges.
- **`--residual 1`** adds the centre's factor embedding to its aggregated message (`LeakyReLU(h_u + sum alpha m h_v)`, as in DisenGCN). Eq. 10 has no such term, so the centre enters only through its self-loop, whose attention share is about `1/d` on dense relations such as Amazon's U-S-U. It is off by default: in 100-epoch Amazon runs both reached validation AUROC 0.959 (test 0.965 with it, 0.959 without).
- **CLUB saturation:** the estimator's log-likelihood approaches its ceiling (log-variance at the `tanh` bound of -1) within a few epochs, so the MI term mostly acts on factors whose spread across nodes is still large.
- **Features** are standardised on every dataset (`--feat_norm`).
- **Reported results** (paper Tables II–IV, 40% training, AUC / F1-macro / GMean / PR-AUC): YelpChi 91.04 / 74.88 / 83.36 / 64.74, Amazon 98.05 / 88.72 / 93.75 / 91.69, T-Finance 96.54 / 88.15 / 92.17 / 86.32, T-Social 96.76 / 87.85 / 90.38 / 61.15.
- Amazon splits only labeled nodes (`index >= 3305`); unlabeled users stay in the graph.
- No DGL or PyG import.
