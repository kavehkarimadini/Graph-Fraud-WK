# GLHN (GLHAD)

## Paper
The Graph Local Homophily Network is proposed as **GLHAD** in the [paper](https://doi.org/10.1145/3627673.3679785) below:
```bibtex
@inproceedings{guo2024glhad,
  title={Graph Local Homophily Network for Anomaly Detection},
  author={Guo, Ronghui and Zou, Minghui and Zhang, Sai and Zhang, Xiaowang and Yu, Zhizhi and Feng, Zhiyong},
  booktitle={Proceedings of the 33rd ACM International Conference on Information and Knowledge Management (CIKM '24)},
  year={2024}
}
```

## Brief Introduction

GLHN predicts node-level local homophily from projected center/neighbor features, then mixes a low-pass filter (`I-L`) with a high-pass filter (`L`) using that score as the weight. Two InfoNCE losses (`L_k` on labeled centers, `L_u` on unlabeled centers) resist feature camouflage. This port follows `reference/GLHN/GLHN.md`; spectral ops use `torch.sparse` instead of DGL, and training uses the toolbox 40/20/40 split.

## Run

From `algorithms/GLHN/`:

```powershell
python GLHN_main.py --dataset yelpchi --gpu 0
python GLHN_main.py --dataset amazon --gpu 0
python GLHN_main.py --dataset elliptic --gpu 0
python GLHN_main.py --dataset tfinance --gpu 0
python GLHN_main.py --dataset tsocial --gpu 0
python GLHN_main.py --dataset comp --gpu 0
```

`--graph_type hetero` (default on YelpChi / Amazon / FDCompCN) runs each relation then mean-pools. `--graph_type homo` unions edges. `--train_ratio` (default 0.4) shrinks only the train pool; valid/test stay 20/40.

## Note

- Paper hyperparameters: hidden 32 (10 on T-Social), 2 layers, `τ=0.1`, Adam `lr=0.01`, `weight_decay=1e-5`. `γ1`/`γ2` follow the paper sensitivity study on YelpChi / Amazon.
- Contrastive losses use **train** labels only (no valid/test leakage). Amazon still splits only labeled nodes (`index >= 3305`).
- T-Social is full-graph (no mini-batch neighbor sampling from the paper), so memory is higher.
- No DGL import.
