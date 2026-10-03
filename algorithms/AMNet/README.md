# AMNet

## Paper
The AMNet model is proposed by the [paper](https://www.ijcai.org/proceedings/2022/270) below:
```bibtex
@inproceedings{chai2022can,
  title={Can Abnormality be Detected by Graph Neural Networks?},
  author={Chai, Ziwei and You, Siqi and Yang, Yang and Pu, Shiliang and Xu, Jiarong and Cai, Haoyang and Jiang, Weihao},
  booktitle={Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, IJCAI},
  year={2022}
}
```

## Brief Introduction

AMNet (Adaptive Multi-frequency graph neural Network) starts from the observation that standard GNNs are low-pass filters that smooth neighbouring signals, while anomalies tend to differ from their (mostly normal) neighbours and therefore live partly in the high-frequency band.

- **Input MLP:** `Linear -> ReLU -> Linear` maps node features to the hidden space.
- **Bernstein filters:** each of `K` filters (default 2) is a Bernstein polynomial of order `M` of the scaled Laplacian `P = L / 2 = (I - D^-1/2 A D^-1/2) / 2`, whose spectrum lies in `[0, 1]`: `h_f = sum_k sigmoid(w_fk) * C(M, k) P^k (I - P)^(M-k) x`. The `M + 1` learnable coefficients shape each filter's frequency response anywhere between low-pass and high-pass.
- **Adaptive fusion:** per node, filter outputs are weighted by `softmax_f(tanh(W_f h_f) . tanh(W_x x))` and summed, followed by dropout and a linear classifier.
- **Marginal loss:** `mean relu(s1 - s0)` over normal training nodes plus `mean relu(s0 - s1)` over anomalous training nodes, weighted by `beta`. Normal nodes are pushed towards filter 0 and anomalies towards filter 1, so the filters specialise.

Training is full-batch cross-entropy plus the marginal loss. Filter coefficients use their own learning rate (`5e-2`, no weight decay).

This port follows `reference/AMNet-main` (official PyG code, `model.py`, `bernconv.py`, `train.py`, `config.py`). The scaled Laplacian is a constant `torch.sparse` CSR matrix with a symmetric custom autograd function, and the powers `P^i x` are computed once and shared by all filters. Filters in the reference all receive the same input, so this is exact and avoids `K - 1` redundant propagations. On a random multi-relation graph with an isolated node, it reproduces a literal dense transcription of the reference forward pass, marginal loss and gradients to float precision. Bernstein coefficients match the reference's `numpy.polynomial` construction.

## Run

From `algorithms/AMNet/`:

```powershell
python AMNet_main.py --dataset yelpchi --gpu 0
python AMNet_main.py --dataset amazon --gpu 0
python AMNet_main.py --dataset elliptic --gpu 0
python AMNet_main.py --dataset tfinance --gpu 0
python AMNet_main.py --dataset tsocial --gpu 0
python AMNet_main.py --dataset comp --gpu 0
```

`--train_ratio` (default 0.4) shrinks only the train pool; valid/test stay 20/40.

## Note

- Reference hyperparameters (`config.py`): 2 filters, filter lr `5e-2`, dropout 0.3, up to 2000 epochs with patience 200.
  - Elliptic: `elliptic` profile (`M=4`, hidden 128, lr `1e-4`, weight decay `1e-5`, `beta=0.5`).
  - YelpChi: `yelp` profile (`M=5`, hidden 64, lr `5e-4`, weight decay `1e-5`, `beta=1`).
  - T-Finance: `FinV` profile (`M=7`, hidden 128, lr `5e-4`).
  - Amazon and FDCompCN have no reference profile and use the `yelp` settings.
  - T-Social uses the `yelp` settings with hidden 32 and 500 epochs / patience 50, so full-batch training fits in memory on a 5.8M-node graph.
- AMNet is a homogeneous model. The reference evaluates "a homophilous version of Yelp", so YelpChi defaults to the R-U-R relation (`--relation rur`, as BWGNN does in this repo). The other datasets use the union of all relations. The graph is made undirected, binary and free of self-loops before building the symmetric normalised Laplacian (PyG `remove_self_loops` + `get_laplacian(..., 'sym')`, `lambda_max = 2`). Isolated nodes keep `P_ii = 1/2`. Use `--relation all` (or another relation name) to change this.
- The reference selects the checkpoint with the best validation AUC-PR. For consistency with the rest of the toolbox, the default is validation AUROC with early stopping. `--select_metric auprc` restores the reference behaviour.
- The cross-entropy is unweighted as in the reference (`weights = [1, 1]`), and predictions use argmax. `--loss_weight balanced` adds inverse-frequency class weights, which can raise G-mean / F1 at the default threshold.
- The reference datasets ship pre-processed and their feature scaling is not documented, so features are standardised by default (`--feat_norm` to override). The reference Elliptic split is temporal. Here every dataset uses the toolbox's stratified 40/20/40 protocol.
- `BernConv` in the reference registers a bias that is never added in `forward`. It is omitted here, so the parameter count is `hidden` smaller per filter.
- Amazon splits only labeled nodes (`index >= 3305`); unlabeled users stay in the graph.
- `config.json` records the learned filter coefficients (`sigmoid(w)`) and the mean filter attention of normal and anomalous test nodes.
- No DGL or PyG import.
