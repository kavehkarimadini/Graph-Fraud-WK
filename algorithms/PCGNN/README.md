# PC-GNN

## Paper
The PC-GNN model is proposed by the [paper](https://dl.acm.org/doi/abs/10.1145/3442381.3449989) below:
```bibtex
@inproceedings{liu2021pick,
  title={Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection},
  author={Liu, Yang and Ao, Xiang and Qin, Zidi and Chi, Jianfeng and Feng, Jinghua and Yang, Hao and He, Qing},
  booktitle={Proceedings of the Web Conference 2021},
  pages={3168--3177},
  year={2021}
}
```

## Brief Introduction

Pick and Choose GNN (PC-GNN) handles class imbalance in graph fraud detection at two levels:

- **Pick (node sampler):** every epoch draws `2 * |train fraud|` training nodes with replacement, with probability `degree / |class|` (degree on the union graph), so the mini-batches are roughly label-balanced and favour well-connected nodes.
- **Choose (neighbour sampler):** a linear label predictor scores every node; under each relation a centre keeps its `ceil(deg * 0.5)` neighbours closest in score (under-sampling the dissimilar majority) and, during training, fraud centres additionally receive the `int(keep * rho)` most similar train fraud nodes as neighbours (over-sampling the minority).
- **Aggregate:** `relu([x_v, mean(x_N)] W_r)` per relation, `relu([x_v, h_1, ..., h_R] W)` across relations, then a linear classifier.

Loss: `CE(gnn) + alpha * CE(label predictor)`. Fraud probability is `sigmoid` of the fraud logit, and predictions use a decision threshold `thres` (0.4 on Amazon, 0.5 elsewhere), as in the reference.

This port follows `reference/PC-GNN-main` (official code, itself derived from CARE-GNN). Neighbour selection and minority over-sampling are vectorised over CSR adjacency (top-k per centre with stable sorts, `index_add_` mean aggregation), so any number of relations works. On a random graph with copied weights it reproduces the reference `PCALayer` outputs to float precision in both train mode (with over-sampling) and test mode.

## Run

From `algorithms/PCGNN/`:

```powershell
python PCGNN_main.py --dataset yelpchi --gpu 0
python PCGNN_main.py --dataset amazon --gpu 0
python PCGNN_main.py --dataset elliptic --gpu 0
python PCGNN_main.py --dataset tfinance --gpu 0
python PCGNN_main.py --dataset tsocial --gpu 0
python PCGNN_main.py --dataset comp --gpu 0
```

`--train_ratio` (default 0.4) shrinks only the train pool; valid/test stay 20/40. `--rho 0` disables minority over-sampling; `--threshold` changes the kept-neighbour fraction.

## Note

- Reference hyperparameters: embedding 64, `rho=0.5`, `alpha=2`, kept-neighbour fraction 0.5, 51 epochs. YelpChi: lr `0.01`, weight decay `1e-3`, batch 1024, `thres=0.5`. Amazon: lr `0.005`, weight decay `5e-4`, batch 256, `thres=0.4`. Elliptic / T-Finance / T-Social reuse the YelpChi settings; FDCompCN uses the Amazon optimiser settings with `thres=0.5`.
- The reference split is 40/20/40 stratified (`train_ratio=0.4`, `test_ratio=0.67`), the same as the toolbox protocol. The reference validated every 5 epochs; here validation runs every epoch with early stopping, and the checkpoint with the best valid AUC is restored.
- Each relation is symmetrised and given self-loops (reference `sparse_to_adjlist`); the Pick degree uses the union of all relations.
- Features are row-normalised (`x / (rowsum + 0.01)`) as in the reference whenever every row sums to a positive value (YelpChi, Amazon); Elliptic / T-Finance / T-Social / FDCompCN default to standard scaling. Override with `--feat_norm`.
- Amazon splits only labeled nodes (`index >= 3305`); unlabeled users stay in the graph as neighbours.
- The saved `probabilities` are `[1 - p, p]` with `p = sigmoid(fraud logit)`; `predictions` use `p >= thres`.
- Evaluation batches default to 1024 nodes (`--eval_batch_size`); lower it on graphs with very high-degree nodes.
- No DGL import.
