
# GraphConsis

## Paper
The GraphConsis model is proposed by the [paper](https://arxiv.org/abs/2005.00625) below:
```bibtex
@inproceedings{liu2020alleviating,
  title={Alleviating the Inconsistency Problem of Applying Graph Neural Network to Fraud Detection},
  author={Liu, Zhiwei and Dou, Yingtong and Yu, Philip S. and Deng, Yutong and Peng, Hao},
  booktitle={Proceedings of the 43nd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2020}
}
```


## Brief Introduction

Paper GraphConsis (SIGIR 2020): trainable context embeddings concatenated with node features, consistency-score neighbor sampling (Eq. 3–4), and relation attention over the Q samples (Eq. 5–6). Two layers use hidden sizes 200 then 100 and sample 10 then 5 neighbors. Training is minibatch and DGL-free.

Toolbox knobs that stay at script defaults (not paper): `--train_ratio 0.4`, `--epochs 10`, `--batch_size 512`, Adam `--learning_rate 0.01`.


## Run the code
Go to `algorithms/GraphConsis/`, and run the following command in the terminal:

`python supervised_train.py --train_prefix ../../dataset/ --file_name YelpChi.mat --sigmoid --gpu 0`

or run `supervised_train.py` in your IDEs.


## Meaning of the arguments
```
--samples_1 --samples_2: neighbors sampled at each layer (paper: 10 then 5)
--dim_1 --dim_2: hidden sizes (paper: 200 then 100)
--context_dim: trainable context embedding size (paper / official code: 128; 0 disables)
--consis_eps: consistency threshold ε (default 0 keeps all scored neighbors)
--max_neighbors: cap on neighbors scored per node (high-degree relations)
--train_ratio: fraction of labeled nodes used for train (max 0.4; valid/test stay 20/40)
```
For more information about the arguments, please refer to `supervised_train.py`.

## Note
- Before running the code, please remember unzip the given dataset.
- T-Finance / T-Social: `python supervised_train.py --train_prefix ../../dataset/ --file_name tfinance --sigmoid --gpu 0`
- FDCompCN: `python supervised_train.py --train_prefix ../../dataset/ --file_name comp --sigmoid --gpu 0`
