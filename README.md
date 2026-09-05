A Deep Graph-based Toolbox for Fraud Detection

**Introduction**

**Update:** DGFraud now uses **PyTorch** and **PyTorch Geometric** (`GCNConv`, `GATConv`, `SAGEConv`). The previous TensorFlow code lives in [DGFraud-TF2](https://github.com/safe-graph/DGFraud-TF2).

**DGFraud** is a Graph Neural Network (GNN) toolbox for fraud detection. It implements and compares several GNN-based fraud detectors. Model papers are listed under [Implemented Models](#implemented-models).

We welcome contributions on adding new fraud detectors and extending the toolbox. Planned items are in [TODO list](#todo-list).

If you use this toolbox, please cite one of the papers below and the [algorithms](#implemented-models) you used:

CIKM'20 ([PDF](https://arxiv.org/pdf/2008.08692.pdf))

```bibtex
@inproceedings{dou2020enhancing,
  title={Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters},
  author={Dou, Yingtong and Liu, Zhiwei and Sun, Li and Deng, Yutong and Peng, Hao and Yu, Philip S},
  booktitle={Proceedings of the 29th ACM International Conference on Information and Knowledge Management (CIKM'20)},
  year={2020}
}
```

SIGIR'20 ([PDF](https://arxiv.org/pdf/2005.00625.pdf))

```bibtex
@inproceedings{liu2020alleviating,
  title={Alleviating the Inconsistency Problem of Applying Graph Neural Network to Fraud Detection},
  author={Liu, Zhiwei and Dou, Yingtong and Yu, Philip S. and Deng, Yutong and Peng, Hao},
  booktitle={Proceedings of the 43nd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2020}
}
```

**Useful Resources**

- [PyGOD: A Python Library for Graph Outlier Detection (Anomaly Detection)](https://github.com/pygod-team/pygod)
- [UGFraud: An Unsupervised Graph-based Toolbox for Fraud Detection](https://github.com/safe-graph/UGFraud)
- [Graph-based Fraud Detection Paper List](https://github.com/safe-graph/graph-fraud-detection-papers)
- [Awesome Fraud Detection Papers](https://github.com/benedekrozemberczki/awesome-fraud-detection-papers)
- [Attack and Defense Papers on Graph Data](https://github.com/safe-graph/graph-adversarial-learning-literature)
- [PyOD: A Python Toolbox for Scalable Outlier Detection (Anomaly Detection)](https://github.com/yzhao062/pyod)
- [PyODD: An End-to-end Outlier Detection System](https://github.com/datamllab/pyodds)
- [DGL: Deep Graph Library](https://github.com/dmlc/dgl)
- [Outlier Detection DataSets (ODDS)](http://odds.cs.stonybrook.edu/)

**Table of Contents**

- [Installation](#installation)
- [Datasets](#datasets)
- [User Guide](#user-guide)
- [Command Examples](#command-examples)
- [Evaluation & Saved Outputs](#evaluation--saved-outputs)
- [Training Arguments](#training-arguments)
- [Implemented Models](#implemented-models)
- [Model Comparison](#model-comparison)
- [Repository Structure](#repository-structure)
- [TODO List](#todo-list)
- [How to Contribute](#how-to-contribute)

---



## Installation



### Requirements

- Python **3.10–3.12** recommended (3.13 may lack CUDA wheels for some torch builds)
- NVIDIA GPU + recent driver (optional; CPU works too)
- `torch` (CUDA build for GPU)
- `torch-geometric`
- `numpy`, `scipy`, `scikit-learn`
- **BWGNN / SplitGNN:** `sympy` (see uv step 4). T-Finance / T-Social / FDCompCN binaries are read without DGL.



### Setup with `uv` (recommended, GPU)

From the repo root:

```powershell
# 1) Create and activate a virtual environment
uv venv
.venv\Scripts\activate

# 2) Install CUDA PyTorch FIRST (pick a CUDA index that matches your driver)
#    Check driver CUDA with: nvidia-smi
#    Example for driver reporting CUDA 12.8:
uv pip install torch --index-url https://download.pytorch.org/whl/cu128

#    If cu128 has no wheel for your Python, try cu126:
#    uv pip install torch --index-url https://download.pytorch.org/whl/cu126

# 3) Install the remaining packages (do not reinstall torch from PyPI after this)
uv pip install torch-geometric numpy scipy scikit-learn

# 4) BWGNN extra (beta-wavelet coefficients)
uv pip install sympy
# Do not `uv pip install dgl` on Python 3.13 / recent Windows CUDA: PyPI ships
# dgl==0.1.3 whose libdgl.dll fails to load. T-Finance/T-Social are loaded
# from the DGL v2 binary without importing DGL.
```

**Important:** Do not run a plain `uv pip install -r requirements.txt` after the CUDA install unless you keep the CUDA index. `requirements.txt` only pins `torch>=...`, and PyPI often installs the **CPU** wheel and overwrites the GPU build.

Verify GPU PyTorch:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

You want a version like `2.x.x+cu128` and `True`.

If CUDA is unavailable, common causes are:

- CPU `torch` was installed again from PyPI
- Python version has no CUDA wheel → recreate with `uv venv --python 3.12`



### Setup with `pip` (alternative)

```bash
git clone https://github.com/safe-graph/DGFraud.git
cd DGFraud
# Install CUDA torch from https://pytorch.org/ first, then:
pip install torch-geometric
pip install -r requirements.txt
# BWGNN: pip install sympy
# or: python setup.py install
```

---



## Datasets



### DBLP

Pre-processed DBLP from [Jhy1993/HAN](https://github.com/Jhy1993/HAN). Used by **FdGars**, **Player2Vec**, **GeniePath**, **GEM**, and **HACUD**.

Extract the archive first.

**Windows PowerShell** (`unzip` is not available by default):

```powershell
cd dataset
Expand-Archive -Path DBLP4057_GAT_with_idx_tra200_val_800.zip -DestinationPath . -Force
```

**Linux / macOS:**

```bash
cd dataset
unzip DBLP4057_GAT_with_idx_tra200_val_800.zip
```

You should see `DBLP4057_GAT_with_idx_tra200_val_800.mat`.

### Example graphs

Small synthetic graphs for **SemiGNN**, **GAS**, and **GEM** are built in `utils/data_loader.py` (`load_example_semi`, `load_data_gas`, `load_example_gem`). Use these for quick smoke tests without DBLP/Yelp.

### YelpChi

For **GraphConsis** / **GraphSAGE**, unzip `dataset/YelpChi.zip`. The `.mat` file includes:

- `net_rur`, `net_rtr`, `net_rsr`: sparse adjacency matrices for three relations
- `features`: 32-d handcrafted features
- `label`: `1` = spam, `0` = benign

Preprocessing details: [CIKM'20](https://arxiv.org/pdf/2008.08692.pdf). For full Yelp metadata, contact [ytongdou@gmail.com](mailto:ytongdou@gmail.com).

### T-Finance and T-Social

DGL v2 binaries from [Tang et al., ICML 2022](https://proceedings.mlr.press/v162/tang22b.html) (BWGNN). Unzip into `dataset/` so you have extension-less files `dataset/tfinance` and `dataset/tsocial`. They are loaded without importing DGL (see `utils/dgl_bin.py`).

**Windows PowerShell:**

```powershell
cd dataset
Expand-Archive -Path tfinance.zip -DestinationPath . -Force
Expand-Archive -Path tsocial.zip -DestinationPath . -Force
cd ..
```

Used by **BWGNN** and **GraphConsis** (`--file_name tfinance` or `tsocial`). T-Social is large (multi-GB); BWGNN uses a smaller hidden size (`--hid_dim 10 --order 5`).

### Elliptic Bitcoin

Kaggle / [Elliptic](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) transaction graph. Place the three CSVs in `dataset/elliptic_bitcoin_dataset/`:

- `elliptic_txs_classes.csv` — `txId`, `class` (`1` illicit, `2` licit, `unknown`)
- `elliptic_txs_edgelist.csv` — payment edges `txId1` → `txId2`
- `elliptic_txs_features.csv` — no header; 167 columns (`txId`, timestep, 165 features)

Loaders keep **labeled** nodes only (~46.5k: 4,545 illicit / 42,019 licit), induce the subgraph, and map illicit→`1`, licit→`0`. Used by **GraphConsis** (`--file_name elliptic`), **BWGNN** (`--dataset elliptic`), and **GraphSAGE**.

### FDCompCN (`comp.dgl`)

Multi-relation company graph from [Wu et al., CIKM 2023](https://dl.acm.org/doi/10.1145/3583780.3615067) (SplitGNN). Place `dataset/comp.dgl` (5,317 nodes, 57-d features, 559 fraud / 10.5%). Relations: C-I-C, C-S-C, C-P-C, plus a homogeneous union.

Used by **GraphConsis** (`--file_name comp`, three non-homo relations), **BWGNN** (`--dataset comp`, homo edges), **SplitGNN** (`--dataset comp`, all relations + signed homo edges), **DGA-GNN**, **GLHN**, and **BSL**. Loaded without importing DGL.

### Train / valid / test protocol (GraphConsis, BWGNN, SplitGNN, DGA-GNN, GLHN, BSL)

All three models use the same stratified split of **labeled** nodes:

| Split | Fraction of labeled nodes | Role |
| ----- | ------------------------- | ---- |
| Train | `--train_ratio` (default **0.4**, max 0.4) | Fit the model |
| Valid | **0.20** (fixed) | Checkpoint selection / hyperparameter tuning |
| Test | **0.40** (fixed) | Final metrics only |

If you lower `--train_ratio` (for example `0.2`), valid and test stay at 20% / 40%. The unused part of the 40% train pool is **discarded**, not moved into valid or test. Training restores the best valid-AUC checkpoint before reporting test metrics.

---



## User Guide



### Quick start (unified entry: `main.py`)

From the **repo root**, with the venv activated:

```powershell
# Example / toy graphs (default dataset_str=example)
python main.py --model GAS --gpu 0
python main.py --model GEM --gpu 0
python main.py --model SemiGNN --batch_size 4 --gpu 0

# DBLP (after extracting the .mat)
python main.py --model Player2Vec --dataset_str dblp --gpu 0
python main.py --model FdGars --dataset_str dblp --gpu 0
python main.py --model GeniePath --dataset_str dblp --gpu 0
python main.py --model GEM --dataset_str dblp --gpu 0
```

`--gpu 0` selects CUDA device 0 when `torch.cuda.is_available()` is true; otherwise training falls back to CPU.

Models available via `main.py`: `Player2Vec`, `FdGars`, `GEM`, `SemiGNN`, `GAS`, `GeniePath`.

### Per-model scripts

Each algorithm folder has its own `*_main.py` (or `supervised_train.py` / `main.py`):


| Model       | Command                                                                                                                                          |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Player2Vec  | `cd algorithms/Player2Vec` then `python Player2Vec_main.py --gpu 0`                                                                              |
| FdGars      | `python FdGars_main.py --gpu 0`                                                                                                                  |
| GAS         | `python GAS_main.py --gpu 0`                                                                                                                     |
| GEM         | `python GEM_main.py --dataset_str example --gpu 0`                                                                                               |
| SemiGNN     | `python SemiGNN_main.py --batch_size 4 --gpu 0`                                                                                                  |
| GeniePath   | `python GeniePath_main.py --gpu 0`                                                                                                               |
| GraphSAGE   | `cd algorithms/GraphSage` then `python supervised_train.py --train_prefix ../../dataset --file_name YelpChi.mat --sigmoid --gpu 0`               |
| GraphConsis | `cd algorithms/GraphConsis` then `python supervised_train.py --train_prefix ../../dataset/ --file_name YelpChi.mat --sigmoid --epochs 3 --gpu 0` (also `Amazon.mat`, `tfinance`, `tsocial`, `elliptic`, `comp`) |
| BWGNN       | `cd algorithms/BWGNN` then `python BWGNN_main.py --dataset tfinance --gpu 0` (also `tsocial`, `yelpchi`, `amazon`, `elliptic`, `comp`)                                                                                                              |
| SplitGNN    | `cd algorithms/SplitGNN` then `python SplitGNN_main.py --dataset comp --gpu 0` (also `yelpchi`, `amazon`, `elliptic`)                                                                                              |
| DGA-GNN     | `cd algorithms/DGAGNN` then `python DGAGNN_main.py --dataset yelpchi --gpu 0` (also `amazon`, `elliptic`, `tfinance`, `tsocial`, `comp`)                                                                           |
| GLHN        | `cd algorithms/GLHN` then `python GLHN_main.py --dataset yelpchi --gpu 0` (also `amazon`, `elliptic`, `tfinance`, `tsocial`, `comp`)                                                                               |
| BSL         | `cd algorithms/BSL` then `python BSL_main.py --dataset yelpchi --gpu 0` (also `amazon`, `elliptic`, `tfinance`, `tsocial`, `comp`)                                                                                 |
| HACUD       | `cd algorithms/HACUD` then `python main.py --dataset dblp --gpu 0 --epoch 100 --embed_size 16 --batch_size 64 --lr 1e-4`                         |




### Running on your own data

See `load_data_dblp()` in `utils/data_loader.py` as a template. You need:

- adjacency matrix / matrices (or adjlists for GAS)
- feature matrix
- label matrix
- train / test index split

Then point `--dataset_str` / paths at your files, or edit the corresponding `*_main.py`.

### Training API (for developers)

Models are `torch.nn.Module` subclasses. Use:

- `fit_batch(...)` for a training step (do not confuse with `nn.Module.train()`, which only sets train mode)
- `evaluate(...)` for evaluation (returns `acc, pred, prob, tags`)
- `save(path)` / `load(path)` for checkpoints (legacy single-file `.pt`; unified runs use `checkpoint.pt` under `results/<Model>/<dataset>/seed<seed>/`)

After `main.py` training, evaluation artifacts are written via `utils.training.finalize_run` / `utils.evaluation.save_run_outputs` (see [Evaluation & Saved Outputs](#evaluation--saved-outputs)).

---



## Command Examples

All examples assume you are at the **repo root** with the virtual environment activated:

```powershell
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
```

Use `#` comments in PowerShell/bash blocks below as inline documentation — do not paste the comment lines as separate commands unless shown on their own line.

### Environment setup

```powershell
# Create venv with a Python version that has CUDA wheels (recommended)
uv venv --python 3.12
.venv\Scripts\activate

# Install GPU PyTorch (match cu128/cu126 to your nvidia-smi CUDA version)
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install torch-geometric numpy scipy scikit-learn

# Confirm GPU is visible to PyTorch
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

```powershell
# CPU-only fallback (no NVIDIA GPU)
uv pip install torch torch-geometric numpy scipy scikit-learn
python main.py --model GAS --gpu -1
```



### Dataset preparation

```powershell
# DBLP — Windows
cd dataset
Expand-Archive -Path DBLP4057_GAT_with_idx_tra200_val_800.zip -DestinationPath . -Force
cd ..

# YelpChi — Windows (for GraphSAGE / GraphConsis)
cd dataset
Expand-Archive -Path YelpChi.zip -DestinationPath . -Force
cd ..
```

```bash
# DBLP — Linux / macOS
cd dataset && unzip DBLP4057_GAT_with_idx_tra200_val_800.zip && cd ..

# YelpChi — Linux / macOS
cd dataset && unzip YelpChi.zip && cd ..
```

---



### `main.py` — unified trainer

`main.py` supports: `Player2Vec`, `FdGars`, `GEM`, `SemiGNN`, `GAS`, `GeniePath`.

#### Smoke tests (example graphs, fast)

```powershell
# GAS — default example U-E-I review graph, 2 epochs, small batch
python main.py --model GAS --dataset_str example --epoch_num 2 --batch_size 4 --gpu 0

# GEM — example heterogeneous device graph
python main.py --model GEM --dataset_str example --epoch_num 2 --batch_size 4 --hop 1 --k 16 --gpu 0

# SemiGNN — example semi-supervised graph (batch_size must be small on toy data)
python main.py --model SemiGNN --dataset_str example --epoch_num 2 --batch_size 4 `
  --init_emb_size 4 --semi_encoding1 3 --semi_encoding2 2 --semi_encoding3 4 `
  --alpha 0.5 --lamtha 0.5 --gpu 0
```



#### DBLP training (requires extracted `.mat`)

```powershell
# Player2Vec — meta-path GCN + attention, DBLP
python main.py --model Player2Vec --dataset_str dblp --gpu 0 `
  --epoch_num 30 --batch_size 1000 --learning_rate 0.001 `
  --hidden1 16 --gcn_output 4 --seed 123

# FdGars — homogeneous GCN on DBLP
python main.py --model FdGars --dataset_str dblp --gpu 0 `
  --epoch_num 30 --batch_size 1000 --learning_rate 0.001 `
  --hidden1 16 --hidden2 16 --gcn_output 4

# GeniePath — GAT + LSTM on DBLP (reduce dim/layers if GPU memory is limited)
python main.py --model GeniePath --dataset_str dblp --gpu 0 `
  --epoch_num 30 --batch_size 512 --learning_rate 0.001 `
  --dim 128 --lstm_hidden 128 --heads 1 --layer_num 4

# GEM on DBLP (multi-device heterogeneous graph)
python main.py --model GEM --dataset_str dblp --gpu 0 `
  --epoch_num 30 --batch_size 1000 --hop 2 --k 32
```



#### Tuning common hyperparameters via `main.py`

```powershell
# Lower learning rate, more epochs, fixed seed for reproducibility
python main.py --model GAS --dataset_str example --gpu 0 `
  --learning_rate 0.0005   --seed 42

# Larger GAS encoder dimensions
python main.py --model GAS --dataset_str example --gpu 0 `
  --encoding1 128 --encoding2 128 --encoding3 128 --gcn_dim 8

# SemiGNN — emphasize supervised loss over graph (random-walk) loss
python main.py --model SemiGNN --dataset_str example --gpu 0 `
  --batch_size 4 --alpha 0.8 --lamtha 0.1 `
  --init_emb_size 8 --semi_encoding3 8

# GeniePath — lighter model for 6 GB GPUs (e.g. RTX 3050)
python main.py --model GeniePath --dataset_str dblp --gpu 0 `
  --dim 64 --lstm_hidden 64 --layer_num 2 --batch_size 256
```

---



### Per-model scripts

Run from each model folder **or** pass the full path from repo root.

#### Player2Vec (`algorithms/Player2Vec/Player2Vec_main.py`)

```powershell
cd algorithms/Player2Vec

# Default DBLP run on GPU 0
python Player2Vec_main.py --gpu 0

# Custom GCN width and embedding size
python Player2Vec_main.py --gpu 0 --hidden1 32 --gcn_output 8 --learning_rate 0.0005

# Reproducible run
python Player2Vec_main.py --gpu 0 --seed 42 --epoch_num 50 --batch_size 512

cd ../..
```



#### FdGars (`algorithms/FdGars/FdGars_main.py`)

```powershell
cd algorithms/FdGars

# Default DBLP
python FdGars_main.py --gpu 0

# Wider GCN, smaller batches (helps on low VRAM)
python FdGars_main.py --gpu 0 --hidden1 32 --hidden2 32 --gcn_output 8 --batch_size 256

cd ../..
```



#### GAS (`algorithms/GAS/GAS_main.py`)

```powershell
cd algorithms/GAS

# Example review graph (default)
python GAS_main.py --gpu 0 --dataset_str example

# Larger aggregators, more training
python GAS_main.py --gpu 0 --dataset_str example `
  --encoding1 128 --encoding2 64 --encoding3 128 `
  --epoch_num 50 --learning_rate 0.0005

cd ../..
```



#### GEM (`algorithms/GEM/GEM_main.py`)

```powershell
cd algorithms/GEM

# Toy example graph
python GEM_main.py --dataset_str example --gpu 0 --hop 1 --k 16

# DBLP with deeper GEM stack
python GEM_main.py --dataset_str dblp --gpu 0 --hop 2 --k 32 --epoch_num 50

cd ../..
```



#### SemiGNN (`algorithms/SemiGNN/SemiGNN_main.py`)

```powershell
cd algorithms/SemiGNN

# Default example (batch_size=4 is required on small graphs)
python SemiGNN_main.py --gpu 0 --batch_size 4

# Tune view-attention and loss balance
python SemiGNN_main.py --gpu 0 --batch_size 4 `
  --init_emb_size 8 --semi_encoding1 4 --semi_encoding2 4 --semi_encoding3 8 `
  --alpha 0.7 --lamtha 0.3 --learning_rate 0.001 --epoch_num 30

cd ../..
```



#### GeniePath (`algorithms/GeniePath/GeniePath_main.py`)

```powershell
cd algorithms/GeniePath

# Default DBLP
python GeniePath_main.py --gpu 0

# Multi-head GAT, fewer layers
python GeniePath_main.py --gpu 0 --heads 4 --layer_num 2 --dim 64 --batch_size 256

cd ../..
```



#### GraphSAGE (`algorithms/GraphSage/supervised_train.py`)

Requires extracted `dataset/YelpChi.mat`.

```powershell
cd algorithms/GraphSage

# Baseline mean aggregator, sigmoid loss (Yelp spam detection)
python supervised_train.py `
  --train_prefix ../../dataset `
  --file_name YelpChi.mat `
  --model graphsage_mean `
  --sigmoid `
  --gpu 0

# Fewer neighbor samples — faster, less memory
python supervised_train.py `
  --train_prefix ../../dataset `
  --file_name YelpChi.mat `
  --sigmoid `
  --samples_1 10 --samples_2 5 `
  --dim_1 64 --batch_size 256 `
  --epochs 20 --learning_rate 0.005 `
  --dropout 0.2 --weight_decay 1e-4 `
  --gpu 0

cd ../..
```



#### GraphConsis (`algorithms/GraphConsis/supervised_train.py`)

Uses Yelp relations (`net_rur`, `net_rtr`, `net_rsr`), Amazon relations (`net_upu`, `net_usu`, `net_uvu`), a single T-Finance / T-Social adjacency, or the labeled Elliptic payment graph.

```powershell
cd algorithms/GraphConsis

# Quick 3-epoch run with context embeddings
python supervised_train.py `
  --train_prefix ../../dataset/ `
  --file_name YelpChi.mat `
  --sigmoid `
  --epochs 3 `
  --context_dim 128 `
  --samples_1 10 --samples_2 5 `
  --dim_1 128 --batch_size 512 `
  --gpu 0

# T-Finance / T-Social (DGL binaries; unzip dataset/tfinance.zip and tsocial.zip first)
python supervised_train.py --train_prefix ../../dataset/ --file_name tfinance --sigmoid --epochs 50 --gpu 0
python supervised_train.py --train_prefix ../../dataset/ --file_name tsocial --sigmoid --epochs 20 --batch_size 256 --gpu 0
python supervised_train.py --train_prefix ../../dataset/ --file_name elliptic --sigmoid --epochs 50 --gpu 0
python supervised_train.py --train_prefix ../../dataset/ --file_name comp --sigmoid --epochs 50 --gpu 0

cd ../..
```



#### BWGNN (`algorithms/BWGNN/BWGNN_main.py`)

Requires sympy (README uv step 4) and unzipped `dataset/tfinance` / `dataset/tsocial`.

```powershell
cd algorithms/BWGNN

# Paper defaults: 40% train, hid 64, wavelet order 2
python BWGNN_main.py --dataset tfinance --train_ratio 0.4 --hid_dim 64 --order 2 --epochs 100 --gpu 0

# T-Social is much larger — smaller hidden size and higher order (as in the paper)
python BWGNN_main.py --dataset tsocial --train_ratio 0.4 --hid_dim 10 --order 5 --epochs 100 --gpu 0

# Homogeneous Amazon / YelpChi .mat
python BWGNN_main.py --dataset amazon --gpu 0
python BWGNN_main.py --dataset yelpchi --gpu 0
python BWGNN_main.py --dataset elliptic --train_ratio 0.4 --hid_dim 64 --order 2 --epochs 100 --gpu 0
python BWGNN_main.py --dataset comp --train_ratio 0.4 --hid_dim 64 --order 2 --epochs 100 --gpu 0

cd ../..
```



#### SplitGNN (`algorithms/SplitGNN/SplitGNN_main.py`)

Requires sympy and `dataset/comp.dgl` (YelpChi / Amazon `.mat` also work). Defaults follow the paper table.

```powershell
cd algorithms/SplitGNN
python SplitGNN_main.py --dataset comp --gpu 0
python SplitGNN_main.py --dataset yelpchi --gpu 0
python SplitGNN_main.py --dataset amazon --gpu 0
python SplitGNN_main.py --dataset elliptic --gpu 0
cd ../..
```



#### DGA-GNN (`algorithms/DGAGNN/DGAGNN_main.py`)

Decision-tree binning + dynamic grouping aggregation (AAAI 2024). Same 40/20/40 protocol.

```powershell
cd algorithms/DGAGNN
python DGAGNN_main.py --dataset yelpchi --gpu 0
python DGAGNN_main.py --dataset amazon --gpu 0
python DGAGNN_main.py --dataset elliptic --gpu 0
python DGAGNN_main.py --dataset tfinance --gpu 0
python DGAGNN_main.py --dataset tsocial --gpu 0
python DGAGNN_main.py --dataset comp --gpu 0
cd ../..
```



#### GLHN (`algorithms/GLHN/GLHN_main.py`)

Local-homophily-guided low/high-pass fusion (CIKM 2024; paper name GLHAD). Same 40/20/40 protocol.

```powershell
cd algorithms/GLHN
python GLHN_main.py --dataset yelpchi --gpu 0
python GLHN_main.py --dataset amazon --gpu 0
python GLHN_main.py --dataset elliptic --gpu 0
python GLHN_main.py --dataset tfinance --gpu 0
python GLHN_main.py --dataset tsocial --gpu 0
python GLHN_main.py --dataset comp --gpu 0
cd ../..
```



#### BSL (`algorithms/BSL/BSL_main.py`)

Disentangled GCN + barely-supervised consistency (AAAI 2024). Default 40/20/40; `--paper_split` uses the paper 1%/10%/89% setting.

```powershell
cd algorithms/BSL
python BSL_main.py --dataset yelpchi --gpu 0
python BSL_main.py --dataset amazon --gpu 0
python BSL_main.py --dataset elliptic --gpu 0
python BSL_main.py --dataset tfinance --gpu 0
python BSL_main.py --dataset tsocial --gpu 0
python BSL_main.py --dataset comp --gpu 0
python BSL_main.py --dataset yelpchi --paper_split --gpu 0
cd ../..
```



#### HACUD (`algorithms/HACUD/main.py`)

Requires extracted DBLP `.mat` (see `get_data.py` path).

```powershell
cd algorithms/HACUD

# Recommended starter command from HACUD README
python main.py --dataset dblp --gpu 0 --epoch 100 --embed_size 16 --batch_size 64 --lr 1e-4

# Normalized adjacency (default), stronger regularization
python main.py --dataset dblp --gpu 0 `
  --epoch 200 --embed_size 32 --batch_size 128 --lr 5e-4 `
  --adj_type norm --reg 1e-2 --n_fc 4 --fc "[32,16,8,4]"

# Plain (unnormalized) adjacency matrix
python main.py --dataset dblp --gpu 0 --adj_type plain --epoch 50 --embed_size 16

cd ../..
```

---



### Quick reference — which script for which dataset?


| Goal                                | Command                                                                                                                            |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Fast sanity check, no data download | `python main.py --model GAS --epoch_num 2 --batch_size 4 --gpu 0`                                                                  |
| DBLP + Player2Vec                   | `python main.py --model Player2Vec --dataset_str dblp --gpu 0`                                                                     |
| DBLP + FdGars                       | `python main.py --model FdGars --dataset_str dblp --gpu 0`                                                                         |
| YelpChi + GraphSAGE                 | `cd algorithms/GraphSage` → `python supervised_train.py --train_prefix ../../dataset --file_name YelpChi.mat --sigmoid --gpu 0`    |
| YelpChi + GraphConsis               | `cd algorithms/GraphConsis` → `python supervised_train.py --train_prefix ../../dataset/ --file_name YelpChi.mat --sigmoid --gpu 0` |
| DBLP + HACUD                        | `cd algorithms/HACUD` → `python main.py --dataset dblp --gpu 0 --epoch 100 --embed_size 16 --batch_size 64 --lr 1e-4`              |


---



## Evaluation & Saved Outputs

Every training entry point (`main.py`, `*_main.py`, GraphSAGE/GraphConsis `supervised_train.py`, HACUD `main.py`) now:

1. **Splits data with stratified train/test** (preserves class balance for imbalanced fraud labels).
2. **Evaluates train and test** after the last epoch.
3. **Writes unified artifacts** under `results/` (unless `--no_save`).



### Unified run layout

```
results/<Model>/<dataset>/seed<seed>/
  config.json       # full CLI / hyperparameter snapshot
  metrics.json      # train + test metrics, split counts, train time, parameter count
  split.npz         # train_idx, test_idx, y_train, y_test
  predictions.npz   # test predictions + probabilities
  checkpoint.pt     # {state_dict, config, model_name}
```


| File              | Contents                                                                                                                |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `config.json`     | All training arguments (model, seed, test_size, lr, …)                                                                  |
| `metrics.json`    | `train` / `test` metrics, `split` report, and `complexity` (`train_time_sec`, `num_parameters`, `num_trainable_parameters`) |
| `split.npz`       | Stratified indices and labels for reproducible re-evaluation                                                            |
| `predictions.npz` | Held-out test predictions and probabilities                                                                             |
| `checkpoint.pt`   | PyTorch checkpoint for loading the trained model                                                                        |


Examples:

- `results/GAS/example/seed123/metrics.json`
- `results/Player2Vec/dblp/seed123/checkpoint.pt`
- `results/GraphSage/YelpChi/seed123/predictions.npz`
- `results/HACUD/dblp/seed123/split.npz`



### Stratified split defaults


| Context                                | Default `--test_size`              |
| -------------------------------------- | ---------------------------------- |
| DBLP (`dataset_str=dblp`, HACUD)       | `0.4`                              |
| Example toy graphs (GAS, GEM, SemiGNN) | `0.4` (GAS) / `0.2` (GEM, SemiGNN) |
| YelpChi (GraphSAGE, GraphConsis)       | `0.2`                              |


Override with `--test_size 0.3 --seed 42` on any script that supports `add_save_args`.

### Train + evaluate (writes `results/`)

```powershell
# From repo root — unified artifacts under results/<Model>/<dataset>/seed<seed>/
python main.py --model GAS --dataset_str example --epoch_num 10 --batch_size 4 --gpu 0
# → results/GAS/example/seed123/

python main.py --model GEM --dataset_str example --epoch_num 10 --batch_size 4 --gpu 0 `
  --save_dir results --test_size 0.2 --seed 42

# DBLP models
python main.py --model Player2Vec --dataset_str dblp --gpu 0
python main.py --model FdGars --dataset_str dblp --gpu 0
python main.py --model GeniePath --dataset_str dblp --gpu 0 --dim 64 --layer_num 2

# Per-model scripts (same artifact layout)
cd algorithms/GAS
python GAS_main.py --gpu 0 --seed 123

cd ../Player2Vec
python Player2Vec_main.py --gpu 0 --dataset_str dblp

cd ../HACUD
python main.py --dataset dblp --gpu 0 --epoch 10 --embed_size 16 --batch_size 64 --lr 1e-4 --seed 123

# Skip writing files (console metrics only)
python main.py --model GAS --dataset_str example --batch_size 4 --gpu 0 --no_save
```

```powershell
# GraphSAGE / GraphConsis (stratified split on YelpChi labels)
cd algorithms/GraphSage
python supervised_train.py `
  --train_prefix ../../dataset --file_name YelpChi.mat `
  --sigmoid --epochs 10 --gpu 0 --seed 123
# → results/GraphSage/YelpChi/seed123/

cd ../GraphConsis
python supervised_train.py `
  --train_prefix ../../dataset/ --file_name YelpChi.mat `
  --sigmoid --epochs 3 --context_dim 128 --gpu 0 --seed 123

cd ../..
```



### Inspect saved evaluation

```powershell
# Unified inspector (prints metrics, split info, file checklist)
python evaluate.py --run_dir results/GAS/example/seed123

# Verify saved predictions match recorded metrics
python evaluate.py --run_dir results/GAS/example/seed123 --verify

# Manual inspection
python -c "import json; print(json.load(open('results/GAS/example/seed123/metrics.json', encoding='utf-8')))"
python -c "import numpy as np; s=np.load('results/GAS/example/seed123/split.npz'); print(s.files, s['train_idx'].shape, s['test_idx'].shape)"
```



### Load a checkpoint and re-evaluate

Checkpoints are at `results/<Model>/<dataset>/seed<seed>/checkpoint.pt`. Use the same `test_size` and `seed` from `config.json` when reloading data so the split matches `split.npz`:

```python
import torch
from utils.evaluation import load_checkpoint

payload = load_checkpoint('results/GAS/example/seed123/checkpoint.pt')
config = payload['config']
state_dict = payload['state_dict']
# Rebuild model from config, load state_dict, evaluate on split.npz test indices
```

```powershell
# List everything written for a run
Get-ChildItem -Recurse results   # Windows PowerShell
```

---



## Training Arguments



### Shared args (`main.py` and most `*_main.py`)


| Argument          | Default                                | Description                                     |
| ----------------- | -------------------------------------- | ----------------------------------------------- |
| `--seed`          | `123`                                  | Random seed                                     |
| `--dataset_str`   | `example` or `dblp`                    | Dataset key (`example` / `dblp`)                |
| `--epoch_num`     | `30`                                   | Training epochs                                 |
| `--batch_size`    | `1000` (SemiGNN scripts often use `4`) | Batch size                                      |
| `--learning_rate` | `0.001`                                | Adam learning rate                              |
| `--gpu`           | `0`                                    | CUDA device id (unavailable → CPU)              |
| `--momentum`      | `0.9`                                  | Kept for API compatibility                      |
| `--save_dir`      | `results`                              | Root folder for eval outputs (`main.py`)        |
| `--no_save`       | flag                                   | Do not write metrics / predictions / checkpoint |




### Player2Vec / FdGars (GCN)


| Argument       | Default | Description                |
| -------------- | ------- | -------------------------- |
| `--hidden1`    | `16`    | GCN hidden units (layer 1) |
| `--hidden2`    | `16`    | GCN hidden units (FdGars)  |
| `--gcn_output` | `4`     | Embedding / encoding size  |




### GAS


| Argument      | Default | Description                     |
| ------------- | ------- | ------------------------------- |
| `--gcn_dim`   | `5`     | Homogeneous GCN dim (legacy)    |
| `--encoding1` | `64`    | Review aggregator output        |
| `--encoding2` | `64`    | Attention hidden size           |
| `--encoding3` | `64`    | User/item aggregator output     |
| `--encoding4` | `64`    | Comment-graph encoding (legacy) |




### SemiGNN


| Argument           | Default | Description                     |
| ------------------ | ------- | ------------------------------- |
| `--init_emb_size`  | `4`     | Initial node embedding dim      |
| `--semi_encoding1` | `3`     | First view-attention width      |
| `--semi_encoding2` | `2`     | Second view-attention width     |
| `--semi_encoding3` | `4`     | MLP units                       |
| `--alpha`          | `0.5`   | Supervised vs graph loss weight |
| `--lamtha`         | `0.5`   | L2 weight multiplier            |




### GEM


| Argument | Default | Description            |
| -------- | ------- | ---------------------- |
| `--hop`  | `1`     | Number of GEM layers   |
| `--k`    | `16`    | Encoding / hidden size |




### GeniePath


| Argument        | Default | Description                |
| --------------- | ------- | -------------------------- |
| `--dim`         | `128`   | Breadth (GAT) width        |
| `--lstm_hidden` | `128`   | Depth LSTM size            |
| `--heads`       | `1`     | GAT heads                  |
| `--layer_num`   | `4`     | Number of GeniePath layers |




### GraphSAGE (`algorithms/GraphSage/supervised_train.py`)


| Argument                      | Default                   | Description                          |
| ----------------------------- | ------------------------- | ------------------------------------ |
| `--train_prefix`              | `../../dataset/`          | Dataset directory                    |
| `--file_name`                 | `YelpChi.mat`             | `.mat` file name                     |
| `--epochs`                    | `10`                      | Epochs                               |
| `--learning_rate`             | `0.01`                    | Learning rate                        |
| `--dim_1`                     | `128`                     | Hidden size                          |
| `--samples_1` / `--samples_2` | `25` / `10`               | Neighbor samples per layer           |
| `--batch_size`                | `512`                     | Batch size                           |
| `--dropout`                   | `0.0`                     | Dropout                              |
| `--weight_decay`              | `0.0`                     | Weight decay                         |
| `--sigmoid`                   | flag                      | Use sigmoid (multi-label style) loss |
| `--gpu`                       | `0`                       | Device                               |
| `--save_dir`                  | `../../results/GraphSage` | Eval output directory                |
| `--no_save`                   | flag                      | Skip writing eval outputs            |




### GraphConsis (`algorithms/GraphConsis/supervised_train.py`)

Same as GraphSAGE, plus:


| Argument        | Default                     | Description                            |
| --------------- | --------------------------- | -------------------------------------- |
| `--context_dim` | `0`                         | Context embedding dim (`0` disables)   |
| `--train_perc`  | `1.`                        | Fraction of the 80% train split to use |
| `--save_dir`    | `../../results/GraphConsis` | Eval output directory                  |
| `--no_save`     | flag                        | Skip writing eval outputs              |




### BWGNN (`algorithms/BWGNN/BWGNN_main.py`)


| Argument          | Default    | Description                                         |
| ----------------- | ---------- | --------------------------------------------------- |
| `--dataset`       | `tfinance` | `tfinance` / `tsocial` / `yelpchi` / `amazon` / `elliptic` / `comp` |
| `--train_ratio`   | `0.4`      | Train fraction of labeled nodes (max `0.4`; valid/test fixed at 20/40) |
| `--hid_dim`       | `64`       | Hidden size (`10` recommended for T-Social)         |
| `--order`         | `2`        | Beta wavelet order C (`5` for T-Social)             |
| `--epochs`        | `100`      | Epochs                                              |
| `--learning_rate` | `0.01`     | Learning rate                                       |
| `--gpu`           | `0`        | Device                                              |




### SplitGNN (`algorithms/SplitGNN/SplitGNN_main.py`)


| Argument          | Default     | Description                                              |
| ----------------- | ----------- | -------------------------------------------------------- |
| `--dataset`       | `comp`      | `comp` / `yelpchi` / `amazon` / `elliptic`               |
| `--train_ratio`   | `0.4`       | Train fraction of labeled nodes (max `0.4`; valid/test fixed at 20/40) |
| `--intra_dim`     | paper table | Hidden size (`8`)                                        |
| `--C` / `--K`     | paper table | Wavelet order / positive-band cutoff (`3` / `2` on comp) |
| `--gamma`         | paper table | Homo-edge hinge weight (`0.8` on comp)                   |
| `--epochs`        | `1000`      | Max epochs (early stop on valid AUC)                     |
| `--early_stop`    | paper table | Patience (`100` on comp; `0` disables)                   |
| `--learning_rate` | paper table | `0.001` (comp), `0.01` (yelpchi), `0.1` (amazon)         |
| `--gpu`           | `0`         | Device                                                   |


### DGA-GNN (`algorithms/DGAGNN/DGAGNN_main.py`)


| Argument          | Default     | Description                                              |
| ----------------- | ----------- | -------------------------------------------------------- |
| `--dataset`       | `yelpchi`   | `yelpchi` / `amazon` / `elliptic` / `tfinance` / `tsocial` / `comp` |
| `--train_ratio`   | `0.4`       | Train fraction of labeled nodes (valid/test fixed 20/40) |
| `--n_hidden`      | paper yaml  | Hidden size                                              |
| `--n_head`        | paper yaml  | Attention heads                                          |
| `--n_bins`        | paper yaml  | Decision-tree bins per feature (`k`)                     |
| `--z`             | paper yaml  | Grouping threshold                                       |
| `--epochs`        | `200`       | Max epochs (early stop on valid AUC)                     |
| `--learning_rate` | `0.001`     | Learning rate                                            |
| `--gpu`           | `0`         | Device                                                   |


### GLHN (`algorithms/GLHN/GLHN_main.py`)


| Argument          | Default     | Description                                              |
| ----------------- | ----------- | -------------------------------------------------------- |
| `--dataset`       | `yelpchi`   | `yelpchi` / `amazon` / `elliptic` / `tfinance` / `tsocial` / `comp` |
| `--train_ratio`   | `0.4`       | Train fraction of labeled nodes (valid/test fixed 20/40) |
| `--hid_dim`       | `32`        | Hidden size (`10` on T-Social)                           |
| `--n_layers`      | `2`         | GLHN layers                                              |
| `--tau`           | `0.1`       | InfoNCE temperature                                      |
| `--gamma1` / `--gamma2` | paper | Weights of `L_k` / `L_u` (0.8/1.0 on YelpChi)      |
| `--graph_type`    | dataset     | `hetero` (mean-pool relations) or `homo`                 |
| `--epochs`        | `100`       | Max epochs (early stop on valid AUC)                     |
| `--learning_rate` | `0.01`      | Learning rate                                            |
| `--gpu`           | `0`         | Device                                                   |


### BSL (`algorithms/BSL/BSL_main.py`)


| Argument          | Default     | Description                                              |
| ----------------- | ----------- | -------------------------------------------------------- |
| `--dataset`       | `yelpchi`   | `yelpchi` / `amazon` / `elliptic` / `tfinance` / `tsocial` / `comp` |
| `--train_ratio`   | `0.4`       | Train fraction of labeled nodes (valid/test fixed 20/40) |
| `--paper_split`   | off         | Paper 1%/10%/89% split (0.01%/0.1% on T-Social)          |
| `--hid_dim`       | `96`        | Latent size, divisible by 3 (`48` on T-Social)           |
| `--alpha` / `--beta` | `0.4` / `0.8` | Weights of `L_D` / `L_bsl`                         |
| `--batch_size`    | `256`       | Unlabeled aux batch for `L_bsl`                          |
| `--epochs`        | `100`       | Max epochs (early stop on valid AUC)                     |
| `--learning_rate` | `0.01`      | Learning rate                                            |
| `--gpu`           | `0`         | Device                                                   |


### HACUD (`algorithms/HACUD/main.py`)


| Argument       | Default       | Description               |
| -------------- | ------------- | ------------------------- |
| `--dataset`    | `dblp`        | Dataset name              |
| `--epoch`      | `500`         | Epochs                    |
| `--embed_size` | `64`          | Node embedding size       |
| `--batch_size` | `64`          | Batch size                |
| `--lr`         | `0.001`       | Learning rate             |
| `--reg`        | `1e-3`        | L2 regularization         |
| `--n_fc`       | `4`           | Number of FC layers       |
| `--fc`         | `[32,16,8,4]` | FC widths                 |
| `--adj_type`   | `norm`        | `plain` / `norm` / `mean` |
| `--gpu`        | `0`           | Device                    |


---



## Implemented Models


| Model           | Paper                                                                                                                                                                                    | Venue      | Reference                                                                             |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------- |
| **SemiGNN**     | [A Semi-supervised Graph Attentive Network for Financial Fraud Detection](https://arxiv.org/pdf/2003.01171)                                                                              | ICDM 2019  | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/semignn.txt)     |
| **Player2Vec**  | [Key Player Identification in Underground Forums over Attributed Heterogeneous Information Network Embedding Framework](http://mason.gmu.edu/~lzhao9/materials/papers/lp0110-zhangA.pdf) | CIKM 2019  | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/player2vec.txt)  |
| **GAS**         | [Spam Review Detection with Graph Convolutional Networks](https://arxiv.org/abs/1908.10679)                                                                                              | CIKM 2019  | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/gas.txt)         |
| **FdGars**      | [FdGars: Fraudster Detection via Graph Convolutional Networks in Online App Review System](https://dl.acm.org/citation.cfm?id=3316586)                                                   | WWW 2019   | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/fdgars.txt)      |
| **GeniePath**   | [GeniePath: Graph Neural Networks with Adaptive Receptive Paths](https://arxiv.org/abs/1802.00910)                                                                                       | AAAI 2019  | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/geniepath.txt)   |
| **GEM**         | [Heterogeneous Graph Neural Networks for Malicious Account Detection](https://arxiv.org/pdf/2002.12307.pdf)                                                                              | CIKM 2018  | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/gem.txt)         |
| **GraphSAGE**   | [Inductive Representation Learning on Large Graphs](https://arxiv.org/pdf/1706.02216.pdf)                                                                                                | NIPS 2017  | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/graphsage.txt)   |
| **GraphConsis** | [Alleviating the Inconsistency Problem of Applying Graph Neural Network to Fraud Detection](https://arxiv.org/pdf/2005.00625.pdf)                                                        | SIGIR 2020 | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/graphconsis.txt) |
| **HACUD**       | [Cash-Out User Detection Based on Attributed Heterogeneous Information Network with a Hierarchical Attention Mechanism](https://aaai.org/ojs/index.php/AAAI/article/view/3884)           | AAAI 2019  | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/hacud.txt)       |
| **BWGNN**       | [Rethinking Graph Neural Networks for Anomaly Detection](https://proceedings.mlr.press/v162/tang22b.html)                                                                                | ICML 2022  | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/bwgnn.txt)       |
| **SplitGNN**    | [SplitGNN: Spectral Graph Neural Network for Fraud Detection against Heterophily](https://dl.acm.org/doi/10.1145/3583780.3615067)                                                         | CIKM 2023  | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/splitgnn.txt)    |
| **DGA-GNN**     | [DGA-GNN: Dynamic Grouping Aggregation GNN for Fraud Detection](https://ojs.aaai.org/index.php/AAAI/article/view/29067)                                                                   | AAAI 2024  | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/dgagnn.txt)      |
| **GLHN**        | [Graph Local Homophily Network for Anomaly Detection](https://doi.org/10.1145/3627673.3679785)                                                                                             | CIKM 2024  | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/glhn.txt)        |
| **BSL**         | [Barely Supervised Learning for Graph-Based Fraud Detection](https://ojs.aaai.org/index.php/AAAI/article/view/29593)                                                                       | AAAI 2024  | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/bsl.txt)         |




### Backend notes (PyTorch Geometric)

- **Player2Vec / FdGars:** multi-layer GCN (`GCNConv`) (+ meta-path attention for Player2Vec)
- **GeniePath:** GAT breadth (`GATConv`) + LSTM depth
- **GraphSAGE / GraphConsis:** `SAGEConv` (GraphConsis adds multi-relation attention)
- **BWGNN:** beta-wavelet polynomial filters (`PolyConv`) with `torch.sparse`
- **SplitGNN:** signed multi-relation beta-wavelet filters (`PolyConv` + `RelationAware`) with `torch.sparse`
- **DGA-GNN:** decision-tree binning + grouped mean aggregation + attention (no DGL)
- **GLHN:** local-homophily prediction + `I-L` / `L` mix + InfoNCE (no DGL)
- **BSL:** GCN + NA/AA/NN disentanglement + weak/strong unlabeled augmentations (no DGL)
- **GAS / SemiGNN / GEM / HACUD:** custom PyTorch modules with sparse / dense graph ops

---



## Model Comparison


| Model           | Application     | Graph Type    | Base Model          |
| --------------- | --------------- | ------------- | ------------------- |
| **SemiGNN**     | Financial Fraud | Heterogeneous | GAT, LINE, DeepWalk |
| **Player2Vec**  | Cyber Criminal  | Heterogeneous | GAT, GCN            |
| **GAS**         | Opinion Fraud   | Heterogeneous | GCN, GAT            |
| **FdGars**      | Opinion Fraud   | Homogeneous   | GCN                 |
| **GeniePath**   | Financial Fraud | Homogeneous   | GAT                 |
| **GEM**         | Financial Fraud | Heterogeneous | GCN                 |
| **GraphSAGE**   | Opinion Fraud   | Homogeneous   | GraphSAGE           |
| **GraphConsis** | Opinion Fraud   | Heterogeneous | GraphSAGE           |
| **HACUD**       | Financial Fraud | Heterogeneous | GAT                 |
| **BWGNN**       | Graph Anomaly   | Homogeneous   | Beta wavelet GNN    |
| **SplitGNN**    | Financial Fraud | Heterogeneous | Split beta wavelet  |
| **DGA-GNN**     | Financial Fraud | Heterogeneous | Grouped GraphSAGE   |
| **GLHN**        | Graph Anomaly   | Heterogeneous | Homophily mix GNN   |
| **BSL**         | Graph Anomaly   | Homogeneous   | Disentangled GCN    |


---



## Repository Structure

- `algorithms/` — model implementations and training scripts
- `base_models/` — shared GCN / attention / GeniePath / GAS / GEM layers (PyTorch + PyG)
- `dataset/` — zipped datasets (extract before use)
- `utils/`
  - `data_loader.py` — dataset loaders
  - `utils.py` — graph helpers (normalize adj, walks, negative sampling)
  - `torch_helpers.py` — device selection, tensors, edge_index conversion, `save_evaluation`
- `main.py` — unified trainer for Player2Vec, FdGars, GAS, GEM, SemiGNN, GeniePath
- `results/` — evaluation outputs after training (`metrics.json`, `predictions.npz`, `*.pt`)

---



## TODO List

- The log loss for GEM model
- Time-based sampling for GEM
- Add sampling methods
- Benchmarking SOTA models
- Scalable implementation

----------------------------------
cd algorithms/GraphSage

python supervised_train.py `
  --model graphsage_mean `
  --learning_rate 0.01 `
  --train_prefix ../../dataset/ `
  --file_name YelpChi.mat `
  --epochs 100 `
  --dropout 0.0 `
  --weight_decay 0.0 `
  --dim_1 128 `
  --batch_size 512 `
  --sigmoid `
  --gpu 0 `
  --seed 123 `
  --samples_1 25 `
  --samples_2 10 `
  --save_dir results `
  --test_size 0.2
-------------------------------------
cd algorithms/FdGars
python FdGars_main.py --dataset_str yelpchi --epoch_num 100 --batch_size 512 --gpu 0 --seed 123
--------------------------------------------

From `algorithms/GraphConsis`. GraphConsis now picks Yelp relations (`net_rur`, `net_rtr`, `net_rsr`) or Amazon relations (`net_upu`, `net_usu`, `net_uvu`) from the file name.

**YelpChi**

```powershell
cd algorithms/GraphConsis

python supervised_train.py `
  --train_prefix ../../dataset/ `
  --file_name YelpChi.mat `
  --sigmoid `
  --epochs 50 `
  --learning_rate 0.01 `
  --dim_1 128 `
  --batch_size 512 `
  --dropout 0.0 `
  --weight_decay 0.0 `
  --context_dim 0 `
  --samples_1 25 `
  --samples_2 10 `
  --test_size 0.2 `
  --seed 123 `
  --gpu 0
```

**Amazon**

```powershell
python supervised_train.py `
  --train_prefix ../../dataset/ `
  --file_name Amazon.mat `
  --sigmoid `
  --epochs 50 `
  --learning_rate 0.01 `
  --dim_1 128 `
  --batch_size 512 `
  --dropout 0.0 `
  --weight_decay 0.0 `
  --context_dim 0 `
  --samples_1 25 `
  --samples_2 10 `
  --test_size 0.2 `
  --seed 123 `
  --gpu 0
```

Outputs: `algorithms/GraphConsis/results/GraphConsis/YelpChi/seed123/` and `.../Amazon/seed123/`. Amazon’s `net_usu` is dense, so the first epoch can take a while to load.

---------------------------------------
BWGNN is in `algorithms/BWGNN`, and T-Finance / T-Social loaders are in `utils/data_loader.py` (`load_tgraph_dgl`, `load_tgraph_pyg`, `load_data_tfinance`, `load_data_tsocial`) plus `utils/dgl_bin.py`. GraphConsis accepts `--file_name tfinance` or `tsocial`.

**Install (after your CUDA torch step)** — also in the README uv section:

```powershell
uv pip install sympy
```

Do not `uv pip install dgl` on Python 3.13 / recent Windows CUDA: PyPI’s `dgl==0.1.3` fails to load `libdgl.dll`. Unzip `dataset/tfinance.zip` and `dataset/tsocial.zip` so you have extension-less `dataset/tfinance` and `dataset/tsocial`.

**GraphConsis**

```powershell
cd algorithms/GraphConsis
python supervised_train.py --train_prefix ../../dataset/ --file_name tfinance --sigmoid --epochs 50 --gpu 0

cd algorithms\GraphConsis
python supervised_train.py `
  --train_prefix ../../dataset/ `
  --file_name tfinance `
  --sigmoid `
  --epochs 50 `
  --learning_rate 0.01 `
  --dim_1 128 `
  --batch_size 16 `
  --dropout 0.0 `
  --weight_decay 0.0 `
  --context_dim 0 `
  --samples_1 25 `
  --samples_2 10 `
  --test_size 0.6 `
  --seed 123 `
  --gpu 0

python supervised_train.py --train_prefix ../../dataset/ --file_name tsocial --sigmoid --epochs 20 --batch_size 256 --gpu 0
```

**BWGNN** (paper settings: 40% train)

```powershell
cd algorithms/BWGNN
python BWGNN_main.py --dataset tfinance --train_ratio 0.4 --hid_dim 64 --order 2 --epochs 100 --gpu 0
python BWGNN_main.py --dataset tsocial --train_ratio 0.4 --hid_dim 10 --order 5 --epochs 100 --gpu 0
```

T-Social is multi-GB; keep `--hid_dim 10`. BWGNN also takes `--dataset amazon` or `yelpchi`.
----------------------------------------------

Elliptic is wired in as a fifth dataset for GraphConsis, BWGNN, and GraphSAGE.

The three CSVs in dataset/elliptic_bitcoin_dataset/ are loaded as a labeled-only payment graph: unknown txs are dropped, class 1 → illicit (1), class 2 → licit (0). That gives 46,564 nodes (4,545 illicit / 42,019 licit), 36,624 edges, and 166 features (timestep + 165 published features).

GraphConsis

cd algorithms\GraphConsis
python supervised_train.py --train_prefix ../../dataset/ --file_name elliptic --sigmoid --epochs 50 --gpu 0
BWGNN

cd algorithms\BWGNN
python BWGNN_main.py --dataset elliptic --train_ratio 0.4 --hid_dim 64 --order 2 --epochs 50 --gpu 0
The Colab notebook has matching GraphConsis and BWGNN Elliptic cells. Keep the folder name elliptic_bitcoin_dataset with those three CSV filenames.

FDCompCN (`dataset/comp.dgl`) is wired in for GraphConsis, BWGNN, and SplitGNN.

```powershell
cd algorithms\GraphConsis
python supervised_train.py --train_prefix ../../dataset/ --file_name comp --sigmoid --epochs 50 --gpu 0

cd algorithms\BWGNN
python BWGNN_main.py --dataset comp --train_ratio 0.4 --hid_dim 64 --order 2 --epochs 100 --gpu 0

cd algorithms\SplitGNN
python SplitGNN_main.py --dataset comp --gpu 0

cd algorithms\DGAGNN
python DGAGNN_main.py --dataset comp --gpu 0

cd algorithms\GLHN
python GLHN_main.py --dataset comp --gpu 0

cd algorithms\BSL
python BSL_main.py --dataset comp --gpu 0
```
--------------------------------------------------
FDCompCN (`dataset/comp.dgl`) is now a first-class dataset for GraphConsis and BWGNN, and **SplitGNN** is in the toolbox as a DGL-free port of `reference/SplitGNN-master`.

**What `comp.dgl` contains:** 5,317 nodes, 57-d features, 559 fraud labels, four directed relations (`homo` 10,059 / `cic` 8,505 / `csc` 5,944 / `cpc` 6,244), plus the official train/valid/test masks.

**How each model uses it**
- **GraphConsis** — the three non-homo relations (C-I-C / C-S-C / C-P-C), toolbox stratified split
- **BWGNN** — homogeneous union only
- **SplitGNN** — all relations; signed homo edges + hinge loss; official file masks by default
- **DGA-GNN / GLHN** — the three non-homo relations; toolbox 40/20/40 split
- **BSL** — homogeneous union of the three non-homo relations; toolbox 40/20/40 (or `--paper_split`)

```powershell
cd algorithms\GraphConsis
python supervised_train.py --train_prefix ../../dataset/ --file_name comp --sigmoid --epochs 50 --gpu 0

cd algorithms\BWGNN
python BWGNN_main.py --dataset comp --train_ratio 0.4 --hid_dim 64 --order 2 --epochs 50 --gpu 0

cd algorithms\SplitGNN
python SplitGNN_main.py --dataset comp --gpu 0
python SplitGNN_main.py --dataset yelpchi --gpu 0
python SplitGNN_main.py --dataset amazon --gpu 0
python SplitGNN_main.py --dataset elliptic --gpu 0 --epochs 50
python SplitGNN_main.py --dataset tsocial --gpu 0 --epochs 50
python SplitGNN_main.py --dataset tfinance --gpu 0 --epochs 50
```

SplitGNN defaults follow the paper table for FDCompCN (`lr=0.001`, `gamma=0.8`, `C=3`, `K=2`, `intra_dim=8`, early stop 100). Pass `--ignore_file_masks` to use a fresh 40/20/40 split instead of the masks in `comp.dgl`. Amazon still drops feature column 19 and trains only on labeled nodes (`index >= 3305`).

A 2-epoch smoke run on `comp` completed (forward, backward, and the train loop). Still no DGL import — the `.dgl` file is read by `utils/dgl_bin.py`.
----------------------------------

GraphConsis, BWGNN, and SplitGNN now share one stratified split of **labeled** nodes:

| Split | Fraction | Role |
| ----- | -------- | ---- |
| **Train** | `--train_ratio` (default **40%**, max 40%) | Fit the model |
| **Valid** | **20%** (fixed) | Pick the checkpoint / tune |
| **Test** | **40%** (fixed) | Final metrics only |

If you set `--train_ratio 0.2`, valid and test stay 20/40. The unused half of the 40% train pool is **discarded**, not moved into valid or test. Same seed → same valid and test nodes.

After training, each model reloads the **best valid-AUC** checkpoint, then reports train / valid / test. `--early_stop N` stops if valid AUC does not improve for N epochs (`0` trains all epochs).

```powershell
# default 40 / 20 / 40
python supervised_train.py --train_prefix ../../dataset/ --file_name comp --sigmoid --epochs 50 --gpu 0

# 20% train; valid/test unchanged
python BWGNN_main.py --dataset elliptic --train_ratio 0.2 --gpu 0
python SplitGNN_main.py --dataset comp --train_ratio 0.4 --gpu 0
```

`comp.dgl` file masks are no longer used unless you pass `--use_file_masks` on SplitGNN. Amazon still splits only labeled nodes (`index >= 3305`); the 40/20/40 ratios apply to that labeled set.
------------------------------------------

**DGA-GNN** (Duan et al., AAAI 2024) is in the toolbox as a DGL-free port of `reference/DGA-GNN-main`.

It follows the official pieces: decision-tree binning of attributes, mean aggregation on the full neighborhood plus two feedback groups, then attention fusion. Training uses this repo’s 40/20/40 split, valid-AUC checkpointing, and the same `metrics.json` / `checkpoint.pt` artifacts. No DGL, Lightning, Hydra, wandb, or `toad` (binning is scikit-learn).

A 2-epoch run on FDCompCN completed (57-d features → 265 after binning).

```powershell
cd algorithms\DGAGNN
python DGAGNN_main.py --dataset yelpchi --gpu 0
python DGAGNN_main.py --dataset amazon --gpu 0
python DGAGNN_main.py --dataset elliptic --gpu 0
python DGAGNN_main.py --dataset tfinance --gpu 0
python DGAGNN_main.py --dataset comp --gpu 0
```

Hyperparameters default to the official yaml per dataset (`n_hidden`, heads, bins `k`, grouping threshold `z`). `--train_ratio` still only shrinks the 40% train pool. `--no_bin_encoding` skips DT binning. T-Social uses a smaller hidden size than the paper yaml because that graph is large.

**GLHN** (Guo et al., CIKM 2024; paper name GLHAD) is in the toolbox as a DGL-free port of `reference/GLHN/GLHN.md`.

It predicts node-level local homophily, mixes low-pass (`I-L`) and high-pass (`L`) signals with that weight, and adds the two camouflage-resistant InfoNCE losses. Training uses this repo’s 40/20/40 split, valid-AUC checkpointing, and the same `metrics.json` / `checkpoint.pt` artifacts. No DGL.

```powershell
cd algorithms\GLHN
python GLHN_main.py --dataset yelpchi --gpu 0
python GLHN_main.py --dataset amazon --gpu 0
python GLHN_main.py --dataset elliptic --gpu 0
python GLHN_main.py --dataset tfinance --gpu 0
python GLHN_main.py --dataset comp --gpu 0
```

Paper defaults: hidden 32 (10 on T-Social), 2 layers, `τ=0.1`, Adam `lr=0.01`. `--graph_type hetero` mean-pools per-relation filters on YelpChi / Amazon / FDCompCN. `--train_ratio` still only shrinks the 40% train pool. T-Social is full-graph, so memory is higher than the paper's mini-batch setup.

**BSL** (Yu, Liu, Luo, AAAI 2024) is in the toolbox as a DGL-free port of `reference/BSL/BSL.md`.

A GCN encoder splits the latent vector into NA / AA / NN subspaces, trains edge-type classification and same-class attention, then weakly/strongly augments unlabeled nodes with labeled prototypes. Default training is the toolbox 40/20/40 split; `--paper_split` uses the paper 1%/10%/89% setting (train-loss checkpoint). No DGL.

```powershell
cd algorithms\BSL
python BSL_main.py --dataset yelpchi --gpu 0
python BSL_main.py --dataset amazon --gpu 0
python BSL_main.py --dataset elliptic --gpu 0
python BSL_main.py --dataset tfinance --gpu 0
python BSL_main.py --dataset comp --gpu 0
python BSL_main.py --dataset yelpchi --paper_split --gpu 0
```
--------------------------------------------------------
