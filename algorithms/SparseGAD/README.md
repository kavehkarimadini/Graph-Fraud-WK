# SparseGAD

## Paper
The SparseGAD model is proposed by the [paper](https://www.ijcai.org/proceedings/2023/234) below:
```bibtex
@inproceedings{gong2023beyond,
  title={Beyond Homophily: Robust Graph Anomaly Detection via Neural Sparsification},
  author={Gong, Zheng and Wang, Guifeng and Sun, Ying and Liu, Qi and Ning, Yuting and Xiong, Hui and Peng, Jingyu},
  booktitle={Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, IJCAI},
  pages={2104--2113},
  year={2023}
}
```

## Brief Introduction

SparseGAD learns which edges to trust on heterophilic fraud graphs instead of aggregating every neighbour equally.

- **GPR encoder:** an input projection followed by a generalised-PageRank GCN; the outputs of every hop (`relu(A_w (H W_l))`) are summed with learnable coefficients initialised to `0.1 * 0.9**k`.
- **Neural sparsification:** an extractor MLP maps node embeddings to a space where each edge is scored by the cosine similarity of its endpoints. The GCN is run a second time with edge weights multiplied by this signed score, so dissimilar (heterophilic) edges are down-weighted or turned into negative messages.
- **Graph augmentation:** every 200 epochs, a cosine kNN graph (k = 10, self included) built from the extractor embeddings is added to the current graph, which is then re-normalised.

Training is full-batch cross-entropy on the train nodes.

This port follows `reference/SparseGAD-main` (official DGL code, model `gpr_att` as run by `main.py`). Message passing uses `torch` gather / `index_add_` and the kNN graph is a chunked brute-force cosine top-k, so no DGL is needed. On a random graph it reproduces a literal dense transcription of the reference forward pass to float precision, and its kNN neighbour sets match scikit-learn's cosine neighbours.

## Run

From `algorithms/SparseGAD/`:

```powershell
python SparseGAD_main.py --dataset yelpchi --gpu 0
python SparseGAD_main.py --dataset amazon --gpu 0
python SparseGAD_main.py --dataset elliptic --gpu 0
python SparseGAD_main.py --dataset tfinance --gpu 0
python SparseGAD_main.py --dataset tsocial --gpu 0
python SparseGAD_main.py --dataset comp --gpu 0
```

`--train_ratio` (default 0.4) shrinks only the train pool; valid/test stay 20/40. `--update_adj_epoch 0` disables the kNN graph updates.

## Note

- Reference hyperparameters (`scripts/*_train_0.4.sh`): Adam lr `0.01`, weight decay `5e-4`, 2 layers, `k=10`, graph update every 200 epochs, up to 1000 epochs. YelpChi: hidden 32, dropout 0.1, edge dropout 0.25. Amazon: hidden 16, dropout 0.2, edge dropout 0.1. Elliptic / T-Finance / FDCompCN use the YelpChi settings. T-Social defaults to 300 epochs with the kNN update disabled, because brute-force kNN over 5.8M nodes is quadratic.
- The graph is the union of all relations plus one self-loop per node. An edge that appears in several relations carries one unit of weight per relation (DGL `to_homogeneous` + `add_self_loop` + `adj()`), followed by symmetric normalisation with source degrees. After a kNN update the graph is de-duplicated and binary before re-normalisation (`dgl.to_simple`). kNN edges point from each neighbour to the node.
- As in the reference, edge dropout in the second GCN pass is applied on top of the first pass's dropped weights.
- The reference split is 40/20/40 stratified, the same as the toolbox protocol. The reference trained for all 1000 epochs and kept the test score of the best-validation epoch. Here early stopping on valid AUC uses a patience of 200 epochs (50 on T-Social), so at least one graph update happens before stopping. The graph of the best epoch is restored together with its weights for the final evaluation.
- Features follow the reference: min-max per column on YelpChi and row normalisation (`x / (rowsum + 0.01)`) on Amazon. The other datasets are standardised. Override with `--feat_norm`.
- The extractor MLP applies its dropout once per node rather than once per edge endpoint, which keeps memory linear in the number of nodes. Evaluation is unaffected.
- The reference's unused `sparsify()` / `threshold` path (attention-threshold edge filtering, never called by `main.py`) is not ported.
- Amazon splits only labeled nodes (`index >= 3305`); unlabeled users stay in the graph.
- Full-batch training keeps one `edges x hidden` tensor per propagation for the backward pass, so very large graphs (T-Social) need a large GPU.
- `config.json` records the edge counts of the initial graph, the best-epoch graph, and every update (`graph_updates`).
- No DGL import.
