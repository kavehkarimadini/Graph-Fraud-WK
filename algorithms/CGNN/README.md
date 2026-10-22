# CGNN

## Paper
The CGNN model is proposed by the [paper](https://doi.org/10.1609/aaai.v39i11.33319) below:
```bibtex
@inproceedings{li2025context,
  title={Context-aware Graph Neural Network for Graph-based Fraud Detection with Extremely Limited Labels},
  author={Li, Pengbo and Yu, Hang and Luo, Xiangfeng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={11},
  pages={12112--12120},
  year={2025}
}
```

## Brief Introduction

CGNN targets fraud detection when only a handful of nodes are labeled (1 or 5 frauds in the paper) and most nodes are unlabeled "gray" samples. It has four parts:

- **Latent mapping** (Eq. 1): `h = LeakyReLU(FC(x))` for every node.
- **Category semantic decomposition** (Eqs. 2–3): each neighbour's latent vector is cut into halves `h[:m]` and `h[m:]` (`m = d // 2`). Two linear layers project them into a normal component `x^0` and an abnormal component `x^1`. A supervised contrastive loss `L_CSD`, whose labels are the two subspaces, pulls normal components together and pushes them away from abnormal ones.
- **Denoising attention and update** (Eqs. 4–5):
  - `alpha_j = softmax(W_att tanh(W (x^0_j + x^1_j) + b))` weighs the two components of neighbour `j`, and the message is `m_j = alpha_j x^0_j + (1 - alpha_j) x^1_j`.
  - The centre is updated as `z_i = FC(h_i + sum_j m_j / sqrt(d_i))`, followed by an MLP classifier trained with class-weighted cross-entropy (Eq. 6).
- **Consistency training on gray samples** (Eqs. 9–10): for unlabeled centres, the normal components (and, in a second view, the abnormal components) are randomly permuted among the centre's neighbours, while each neighbour keeps its own `alpha_j`. The clean prediction `y` and the two perturbed predictions are tied by `L_CL = -mean(cos(y, y^0) * log cos(y, y^1))`.

The loss is `L_CE + lambda * L_CSD + beta * L_CL` (Eq. 11). No perturbation is applied at test time.

**No official code was available.** This implementation follows the paper text (`reference/CGNN.md`). Against a naive per-node implementation of Eqs. 1–5, 3, 9 and 10 (Python neighbour sets, per-centre loops, explicit per-centre permutations), it matches the logits of the clean and both perturbed views, and both losses, to float64 precision.

## Run

From `algorithms/CGNN/`:

```powershell
python CGNN_main.py --dataset yelpchi --gpu 0
python CGNN_main.py --dataset amazon --gpu 0
python CGNN_main.py --dataset elliptic --gpu 0
python CGNN_main.py --dataset tfinance --gpu 0
python CGNN_main.py --dataset tsocial --gpu 0
python CGNN_main.py --dataset comp --gpu 0
python CGNN_main.py --dataset amazon --n_black 5 --gpu 0   # paper few-label split
python CGNN_main.py --dataset amazon --n_black 1 --gpu 0
```

There are two split modes:

- **Toolbox protocol** (default): the 40/20/40 stratified split shared by the other models. `--train_ratio` shrinks only the train pool. The gray samples for `L_CL` are the discarded part of the train pool plus nodes outside the labeled range (Amazon users below index 3305). When both are empty, the validation nodes are used without their labels, as BSL does.
- **Paper few-label split** (`--n_black 1` or `5`): a stratified 10/30/60 train/valid/test split. From the 10% train pool, `n_black` frauds and `round(eta * n_black)` normals are labeled, where `eta` is the dataset's normal-to-fraud ratio. The rest of the pool is gray. Runs are tagged `black<n>` unless `--run_tag` is set, and the `split` block of `metrics.json` records the labeled counts, `eta` and the gray-set size.

Ablations as in the paper: `--lam 0` removes `L_CSD` (CGNN w/o CSD), and `--beta 0` removes consistency training (CGNN w/o CL).

## Note

- **Paper settings** (Sec. 5.1): Adam with lr `0.01` and weight decay `1e-4`, embedding size 64, `lambda = beta = 1`, and batch size 16. Batch 16 and the paper temperatures (`tau = 0.8` on T-Finance, `0.01` elsewhere) are the defaults with `--n_black`.
- **Toolbox-split defaults differ from the paper** because the paper only trains with a few dozen labels:
  - Batch size is 256 (1024 on T-Social). With 40% of nodes labeled, batch 16 would mean hundreds of steps per epoch.
  - `tau` is 0.5 (T-Finance stays at 0.8). At `tau = 0.01` the `L_CSD` gradients are about 100 times larger than those of the cross-entropy. In smoke runs YelpChi then stayed at chance (test AUROC 0.65 after 10 epochs, versus 0.86 with `tau 0.5` after 6 epochs), and Amazon reached 0.954 versus 0.976 after 15 epochs. `--tau` overrides either default.
- **Training loop:** each step takes `batch_size` labeled centres and, when consistency training is on, `batch_size` gray centres. An epoch is `ceil(max(#labeled, #gray) / batch_size)` steps; the smaller pool is cycled with reshuffling. Training validates every epoch, stops after 25 epochs without improvement (T-Social: 30 epochs, patience 10), and restores the best validation-AUROC checkpoint. Predictions use threshold `0.5` on the fraud probability (`--thres`).
- **Choices the paper leaves open:**
  - The graph is the de-duplicated, symmetrised union of all relations without self-loops. Eq. 5 adds the centre's own `h_i` separately, and `d_i` is the centre's degree (isolated nodes use `d_i = 1` with an empty sum).
  - The softmax of Eq. 4 is taken over the two components of each neighbour (equivalently a sigmoid of a logit difference). A softmax across the neighbourhood would make the paper's filter coefficient `gamma_i = mean_j alpha_j` the constant `1/d_i`, contradicting Theorem 1.
  - `W_nor` and `W_abnor` map the two halves back to the full embedding size, so messages can be added to `h_i` in Eq. 5. The MLP classifier is `Linear-LeakyReLU-Linear`.
  - `L_CSD` follows Khosla et al. (cited by the paper): the denominator runs over all other components, log-probabilities are averaged over positives and then over anchors. Each step uses the components of up to `--csd_nodes` (256) distinct neighbours, which bounds the similarity matrix on dense graphs such as Amazon.
  - SFP permutes components within each centre's own neighbourhood; a centre with one neighbour gets no perturbation. `L_CL` follows Eq. 10 literally, without stopping gradients through the clean prediction.
  - Imbalance weights `eta^c` are inverse class frequencies of the labeled training nodes.
- **Collapse under `L_CSD`:** with labels given by the subspace, the loss is minimised when all normal components share one direction and all abnormal components share another. In smoke runs `L_CSD` sits near its floor `log(255)` after a few epochs, so node-specific information in messages is carried mainly by component norms and the attention weights.
- **Few-label smoke runs** (after 15 and 6 epochs): Amazon with 5 labeled frauds reached test Macro-F1 77.6 (AUROC 0.875), and YelpChi with 5 frauds reached Macro-F1 53.8 (AUROC 0.58). The labeled set is memorised within a few epochs, so early stopping on validation matters.
- **Reported results** (paper Table 2, Macro-F1 with 1 / 5 labeled frauds): Amazon 85.56 / 90.08, Elliptic 76.26 / 82.79, T-Finance 86.36 / 90.10, YelpChi 52.71 / 56.08.
- Amazon splits only labeled nodes (`index >= 3305`); unlabeled users stay in the graph.
- No DGL or PyG import.
