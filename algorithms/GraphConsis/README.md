
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

It is revised based on the [GraphSage](https://github.com/williamleif/GraphSAGE/tree/master/graphsage) model. We support multiple relations and distance sampling as mentioned in [our paper](https://arxiv.org/pdf/2005.00625.pdf).


## Run the code
Go to `algorithms/GraphConsis/`, and run the following command in the terminal:

`python supervised_train.py --train_prefix ../../dataset/ --file_name YelpChi.mat --model graphsage_mean --sigmoid --epochs 3 --samples_1 10 --samples_2 5 --context_dim 128 --train_perc 1. --gpu 0`

or run `supervised_train.py` in your IDEs.


## Meaning of the arguments
```
--samples_1 -samples_2: the number of samples using at difference layers
--context_dim: the dimension of context embeddings
--train_perc: the percentage of training data used to train the model, 1. represents using 80% training and 20% testing, 0.5 will use 40% as training and the same 20% as testing
```
For more information about the arguments, please refer to `supervised_train.py`.

## Note
- GraphSAGE and GraphConsis now use PyTorch Geometric `SAGEConv` (and relation-level attention for GraphConsis).

- Before running the code, please remember unzip the given dataset.
- T-Finance / T-Social: `python supervised_train.py --train_prefix ../../dataset/ --file_name tfinance --sigmoid --gpu 0`
- FDCompCN: `python supervised_train.py --train_prefix ../../dataset/ --file_name comp --sigmoid --gpu 0` 
