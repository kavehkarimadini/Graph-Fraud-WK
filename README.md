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
- [Hyperparameter Policy](#hyperparameter-policy)
- [Running the Full Benchmark](#running-the-full-benchmark)
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

**RunPod (GPU pod, Google Drive zip →** `gdown` **→** `uv sync` **→ COFRAUD):** see [`docs/RUNPOD.md`](docs/RUNPOD.md).

`pyproject.toml` pins **CUDA PyTorch** (not the CPU wheel). From the repo root:

```powershell
# Windows / x86_64 Linux GPU (CUDA 12.8 wheels)
uv sync --extra gpu

# NVIDIA GB10 / DGX Spark / Vast ARM (linux aarch64, CUDA 13.0 wheels)
uv sync --extra gb10
```

`uv sync` with no extra still picks **cu130** on `linux aarch64` and **cu128** on Windows / x86_64 Linux. Do not `uv pip install torch` from PyPI after this or you can overwrite the GPU build. Do not install DGL.

Manual equivalent (if you are not using `uv sync`):

```powershell
uv venv
.venv\Scripts\activate

# GPU (x86 / Windows): CUDA 12.8
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
# GB10 (linux aarch64): CUDA 13.0
# uv pip install torch --index-url https://download.pytorch.org/whl/cu130

uv pip install torch-geometric numpy scipy scikit-learn sympy
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

Used by **BWGNN** and **GraphConsis** (`--file_name tfinance` or `tsocial`). T-Social is large (multi-GB); BWGNN defaults to the paper's T-Social setting (`h = 64`, `C = 5`); pass `--hid_dim 10` if GPU memory is tight.

### Elliptic Bitcoin

Kaggle / [Elliptic](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) transaction graph. Place the three CSVs in `dataset/elliptic_bitcoin_dataset/`:

- `elliptic_txs_classes.csv` — `txId`, `class` (`1` illicit, `2` licit, `unknown`)
- `elliptic_txs_edgelist.csv` — payment edges `txId1` → `txId2`
- `elliptic_txs_features.csv` — no header; 167 columns (`txId`, timestep, 165 features)

Loaders keep **labeled** nodes only (~46.5k: 4,545 illicit / 42,019 licit), induce the subgraph, and map illicit→`1`, licit→`0`. Used by **GraphConsis** (`--file_name elliptic`), **BWGNN** (`--dataset elliptic`), and **GraphSAGE**.

### FDCompCN (`comp.dgl`)

Multi-relation company graph from [Wu et al., CIKM 2023](https://dl.acm.org/doi/10.1145/3583780.3615067) (SplitGNN). Place `dataset/comp.dgl` (5,317 nodes, 57-d features, 559 fraud / 10.5%). Relations: C-I-C, C-S-C, C-P-C, plus a homogeneous union.

Used by **GraphConsis** (`--file_name comp`, three non-homo relations), **BWGNN** (`--dataset comp`, homo edges), **SplitGNN** (`--dataset comp`, all relations + signed homo edges), **DGA-GNN**, **GLHN**, **BSL**, **CARE-GNN**, **H2-FDetector**, **FRAUDRE**, **PC-GNN**, **SparseGAD**, **AMNet**, **GHRN**, **GGCN**, **TAM**, **GDN**, **ImGSL**, **CGNN**, **H2IDE**, **CIE-GAD**, and **COFRAUD**. Loaded without importing DGL.

### Train / valid / test protocol (GraphConsis, BWGNN, SplitGNN, DGA-GNN, GLHN, BSL, CARE-GNN, H2-FDetector, FRAUDRE, PC-GNN, SparseGAD, AMNet, GHRN, GGCN, TAM, GDN, ImGSL, CGNN, H2IDE, CIE-GAD, COFRAUD, PMP, SEC-GFD)

All three models use the same stratified split of **labeled** nodes:

| Split | Fraction of labeled nodes                  | Role                                         |
| ----- | ------------------------------------------ | -------------------------------------------- |
| Train | `--train_ratio` (default **0.4**, max 0.4) | Fit the model                                |
| Valid | **0.20** (fixed)                           | Checkpoint selection / hyperparameter tuning |
| Test  | **0.40** (fixed)                           | Final metrics only                           |

If you lower `--train_ratio` (for example `0.2`), valid and test stay at 20% / 40%. The unused part of the 40% train pool is **discarded**, not moved into valid or test. Training restores the best valid-AUC checkpoint before reporting test metrics. `--early_stop` defaults to **25** (pass `0` to train every epoch).

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

| Model       | Command                                                                                                                                                                                                         |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Player2Vec  | `cd algorithms/Player2Vec` then `python Player2Vec_main.py --gpu 0`                                                                                                                                             |
| FdGars      | `python FdGars_main.py --gpu 0`                                                                                                                                                                                 |
| GAS         | `python GAS_main.py --gpu 0`                                                                                                                                                                                    |
| GEM         | `python GEM_main.py --dataset_str example --gpu 0`                                                                                                                                                              |
| SemiGNN     | `python SemiGNN_main.py --batch_size 4 --gpu 0`                                                                                                                                                                 |
| GeniePath   | `python GeniePath_main.py --gpu 0`                                                                                                                                                                              |
| GraphSAGE   | `cd algorithms/GraphSage` then `python supervised_train.py --train_prefix ../../dataset --file_name YelpChi.mat --sigmoid --gpu 0`                                                                              |
| GraphConsis | `cd algorithms/GraphConsis` then `python supervised_train.py --train_prefix ../../dataset/ --file_name YelpChi.mat --sigmoid --epochs 3 --gpu 0` (also `Amazon.mat`, `tfinance`, `tsocial`, `elliptic`, `comp`) |
| BWGNN       | `cd algorithms/BWGNN` then `python BWGNN_main.py --dataset tfinance --gpu 0` (also `tsocial`, `yelpchi`, `amazon`, `elliptic`, `comp`)                                                                          |
| SplitGNN    | `cd algorithms/SplitGNN` then `python SplitGNN_main.py --dataset comp --gpu 0` (also `yelpchi`, `amazon`, `elliptic`)                                                                                           |
| DGA-GNN     | `cd algorithms/DGAGNN` then `python DGAGNN_main.py --dataset yelpchi --gpu 0` (also `amazon`, `elliptic`, `tfinance`, `tsocial`, `comp`)                                                                        |
| GLHN        | `cd algorithms/GLHN` then `python GLHN_main.py --dataset yelpchi --gpu 0` (also `amazon`, `elliptic`, `tfinance`, `tsocial`, `comp`)                                                                            |
| BSL         | `cd algorithms/BSL` then `python BSL_main.py --dataset yelpchi --gpu 0` (also `amazon`, `elliptic`, `tfinance`, `tsocial`, `comp`)                                                                              |
| CARE-GNN    | `cd algorithms/CAREGNN` then `python CAREGNN_main.py --dataset yelpchi --gpu 0` (also `amazon`, `elliptic`, `tfinance`, `tsocial`, `comp`)                                                                      |
| H2-FDetector | `cd algorithms/H2FDetector` then `python H2FDetector_main.py --dataset yelpchi --gpu 0` (also `amazon`, `elliptic`, `tfinance`, `tsocial`, `comp`)                                                             |
| FRAUDRE     | `cd algorithms/FRAUDRE` then `python FRAUDRE_main.py --dataset yelpchi --gpu 0` (also `amazon`, `elliptic`, `tfinance`, `tsocial`, `comp`)                                                                      |
| PC-GNN      | `cd algorithms/PCGNN` then `python PCGNN_main.py --dataset yelpchi --gpu 0` (also `amazon`, `elliptic`, `tfinance`, `tsocial`, `comp`)                                                                          |
| SparseGAD   | `cd algorithms/SparseGAD` then `python SparseGAD_main.py --dataset yelpchi --gpu 0` (also `amazon`, `elliptic`, `tfinance`, `tsocial`, `comp`)                                                                  |
| AMNet       | `cd algorithms/AMNet` then `python AMNet_main.py --dataset yelpchi --gpu 0` (also `amazon`, `elliptic`, `tfinance`, `tsocial`, `comp`)                                                                          |
| GHRN        | `cd algorithms/GHRN` then `python GHRN_main.py --dataset yelpchi --gpu 0` (also `amazon`, `elliptic`, `tfinance`, `tsocial`, `comp`)                                                                            |
| GGCN        | `cd algorithms/GGCN` then `python GGCN_main.py --dataset yelpchi --gpu 0` (also `amazon`, `elliptic`, `tfinance`, `tsocial`, `comp`)                                                                            |
| TAM         | `cd algorithms/TAM` then `python TAM_main.py --dataset yelpchi --gpu 0` (also `amazon`, `elliptic`, `tfinance`, `tsocial`, `comp`)                                                                              |
| GDN         | `cd algorithms/GDN` then `python GDN_main.py --dataset yelpchi --gpu 0` (also `amazon`, `elliptic`, `tfinance`, `tsocial`, `comp`)                                                                              |
| ImGSL       | `cd algorithms/ImGSL` then `python ImGSL_main.py --dataset yelpchi --gpu 0` (also `amazon`, `elliptic`, `tfinance`, `tsocial`, `comp`)                                                                          |
| CGNN        | `cd algorithms/CGNN` then `python CGNN_main.py --dataset yelpchi --gpu 0` (also `amazon`, `elliptic`, `tfinance`, `tsocial`, `comp`; `--n_black 1` / `5` for the paper few-label split)                         |
| H2IDE       | `cd algorithms/H2IDE` then `python H2IDE_main.py --dataset yelpchi --gpu 0` (also `amazon`, `elliptic`, `tfinance`, `tsocial`, `comp`)                                                                          |
| CIE-GAD     | `cd algorithms/CIEGAD` then `python CIEGAD_main.py --dataset yelpchi --gpu 0` (also `amazon`, `elliptic`, `tfinance`, `tsocial`, `comp`)                                                                        |
| COFRAUD     | `cd algorithms/COFRAUD` then `python COFRAUD_main.py --dataset yelpchi --gpu 0` (also `amazon`, `elliptic`, `tfinance`, `tsocial`, `comp`)                                                                      |
| PMP         | `cd algorithms/PMP` then `python PMP_main.py --dataset amazon --gpu 0` (also `yelpchi`, `elliptic`, `tfinance`, `tsocial`, `comp`)                                                                              |
| SEC-GFD     | `cd algorithms/SEC_GFD` then `python SEC_GFD_main.py --dataset amazon --gpu 0` (also `yelpchi`, `elliptic`, `tfinance`, `tsocial`, `comp`)                                                                      |
| HACUD       | `cd algorithms/HACUD` then `python main.py --dataset dblp --gpu 0 --epoch 100 --embed_size 16 --batch_size 64 --lr 1e-4`                                                                                        |

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
- `save(path)` / `load(path)` for checkpoints (legacy single-file `.pt`; unified runs use `checkpoint.pt` under `results/<Model>/<dataset>/seed<seed>_train<pct>/`)

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

# T-Social: paper setting is h = 64, C = 5 (the default); add --hid_dim 10 if GPU memory is tight
python BWGNN_main.py --dataset tsocial --train_ratio 0.4 --epochs 100 --gpu 0

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

#### CARE-GNN (`algorithms/CAREGNN/CAREGNN_main.py`)

Label-aware neighbour filtering with RL-tuned per-relation thresholds (CIKM 2020). Same 40/20/40 protocol.

```powershell
cd algorithms/CAREGNN
python CAREGNN_main.py --dataset yelpchi --gpu 0
python CAREGNN_main.py --dataset amazon --gpu 0
python CAREGNN_main.py --dataset elliptic --gpu 0
python CAREGNN_main.py --dataset tfinance --gpu 0
python CAREGNN_main.py --dataset tsocial --gpu 0
python CAREGNN_main.py --dataset comp --gpu 0
cd ../..
```

#### H2-FDetector (`algorithms/H2FDetector/H2FDetector_main.py`)

Signed (homophilic / heterophilic) multi-relation attention with edge hinge and prototype losses (WWW 2022). Full-batch, same 40/20/40 protocol.

```powershell
cd algorithms/H2FDetector
python H2FDetector_main.py --dataset yelpchi --gpu 0
python H2FDetector_main.py --dataset amazon --gpu 0
python H2FDetector_main.py --dataset elliptic --gpu 0
python H2FDetector_main.py --dataset tfinance --gpu 0
python H2FDetector_main.py --dataset tsocial --gpu 0
python H2FDetector_main.py --dataset comp --gpu 0
cd ../..
```

#### FRAUDRE (`algorithms/FRAUDRE/FRAUDRE_main.py`)

Ego-feature MLP + fraud-aware (mean and difference) multi-relation convolutions + prior-adjusted loss (ICDM 2021). Same 40/20/40 protocol.

```powershell
cd algorithms/FRAUDRE
python FRAUDRE_main.py --dataset yelpchi --gpu 0
python FRAUDRE_main.py --dataset amazon --gpu 0
python FRAUDRE_main.py --dataset elliptic --gpu 0
python FRAUDRE_main.py --dataset tfinance --gpu 0
python FRAUDRE_main.py --dataset tsocial --gpu 0
python FRAUDRE_main.py --dataset comp --gpu 0
cd ../..
```

#### PC-GNN (`algorithms/PCGNN/PCGNN_main.py`)

Label-balanced node picking + label-aware neighbour choosing with minority over-sampling (WWW 2021). Same 40/20/40 protocol.

```powershell
cd algorithms/PCGNN
python PCGNN_main.py --dataset yelpchi --gpu 0
python PCGNN_main.py --dataset amazon --gpu 0
python PCGNN_main.py --dataset elliptic --gpu 0
python PCGNN_main.py --dataset tfinance --gpu 0
python PCGNN_main.py --dataset tsocial --gpu 0
python PCGNN_main.py --dataset comp --gpu 0
cd ../..
```

#### SparseGAD (`algorithms/SparseGAD/SparseGAD_main.py`)

GPR GCN with learned cosine edge attention (neural sparsification) and periodic kNN graph augmentation (IJCAI 2023). Full-batch, same 40/20/40 protocol.

```powershell
cd algorithms/SparseGAD
python SparseGAD_main.py --dataset yelpchi --gpu 0
python SparseGAD_main.py --dataset amazon --gpu 0
python SparseGAD_main.py --dataset elliptic --gpu 0
python SparseGAD_main.py --dataset tfinance --gpu 0
python SparseGAD_main.py --dataset tsocial --gpu 0
python SparseGAD_main.py --dataset comp --gpu 0
cd ../..
```

#### AMNet (`algorithms/AMNet/AMNet_main.py`)

Adaptive multi-frequency GNN: learnable Bernstein spectral filters mixed by per-node attention, with a marginal loss that separates normal and anomalous nodes across filters (IJCAI 2022). Full-batch, same 40/20/40 protocol.

```powershell
cd algorithms/AMNet
python AMNet_main.py --dataset yelpchi --gpu 0
python AMNet_main.py --dataset amazon --gpu 0
python AMNet_main.py --dataset elliptic --gpu 0
python AMNet_main.py --dataset tfinance --gpu 0
python AMNet_main.py --dataset tsocial --gpu 0
python AMNet_main.py --dataset comp --gpu 0
cd ../..
```

#### GHRN (`algorithms/GHRN/GHRN_main.py`)

Heterophily-reducing edge pruning (WWW 2023): BWGNN is trained, its predictions score every edge with a graph-Laplacian high-frequency indicator, the most heterophilous edges are deleted, and a fresh BWGNN is trained on the pruned graph. Both stages run in one command; same 40/20/40 protocol.

```powershell
cd algorithms/GHRN
python GHRN_main.py --dataset yelpchi --gpu 0
python GHRN_main.py --dataset amazon --gpu 0
python GHRN_main.py --dataset elliptic --gpu 0
python GHRN_main.py --dataset tfinance --gpu 0
python GHRN_main.py --dataset tsocial --gpu 0
python GHRN_main.py --dataset comp --gpu 0
python GHRN_main.py --dataset yelpchi --homo 0 --gpu 0   # per-relation BWGNN
cd ../..
```

#### GGCN (`algorithms/GGCN/GGCN_main.py`)

Heterophily- and oversmoothing-aware GCN (ICDM 2022): signed cosine-similarity messages, learnable degree correction, and decaying residual aggregation. Full-batch on the union graph with sparse edge kernels; same 40/20/40 protocol.

```powershell
cd algorithms/GGCN
python GGCN_main.py --dataset yelpchi --gpu 0
python GGCN_main.py --dataset amazon --gpu 0
python GGCN_main.py --dataset elliptic --gpu 0
python GGCN_main.py --dataset tfinance --gpu 0
python GGCN_main.py --dataset tsocial --gpu 0
python GGCN_main.py --dataset comp --gpu 0
cd ../..
```

#### TAM (`algorithms/TAM/TAM_main.py`)

Truncated Affinity Maximization (NeurIPS 2023), an **unsupervised** detector: GCNs trained without labels to maximise neighbour affinity on graphs with dissimilar edges repeatedly truncated; the anomaly score is low local affinity. Labels are used only for evaluation on the 40/20/40 split and for the decision threshold.

```powershell
cd algorithms/TAM
python TAM_main.py --dataset yelpchi --gpu 0
python TAM_main.py --dataset amazon --gpu 0
python TAM_main.py --dataset elliptic --gpu 0
python TAM_main.py --dataset tfinance --gpu 0
python TAM_main.py --dataset tsocial --gpu 0
python TAM_main.py --dataset comp --gpu 0
python TAM_main.py --dataset amazon --relation upu --gpu 0   # single relation (K=4, T=3)
cd ../..
```

#### GDN (`algorithms/GDN/GDN_main.py`)

Graph Decomposition Network (WSDM 2023) against structural distribution shift: a multi-relation mean-aggregation GNN on a trainable feature table, whose features are split by CE-gradient magnitude into anomaly features (constrained towards class prototypes) and surrounding features (constrained towards the neighbourhood). Mini-batch training; same 40/20/40 protocol.

```powershell
cd algorithms/GDN
python GDN_main.py --dataset yelpchi --gpu 0
python GDN_main.py --dataset amazon --gpu 0
python GDN_main.py --dataset elliptic --gpu 0
python GDN_main.py --dataset tfinance --gpu 0
python GDN_main.py --dataset tsocial --gpu 0
python GDN_main.py --dataset comp --gpu 0
python GDN_main.py --dataset yelpchi --add_constraint 0 --gpu 0   # backbone only (ablation)
cd ../..
```

#### ImGSL (`algorithms/ImGSL/ImGSL_main.py`)

Imbalanced Graph Structure Learning (IGSL, Machine Learning 2024), implemented from the paper (no official code): multi-relational class-balanced node sampler, label-predictor edge pruning, and per-relation iterative node-anchor graph learning (IDGL-style) mixed with the refined input graph. Mini-batch training; same 40/20/40 protocol.

```powershell
cd algorithms/ImGSL
python ImGSL_main.py --dataset yelpchi --gpu 0
python ImGSL_main.py --dataset amazon --gpu 0
python ImGSL_main.py --dataset elliptic --gpu 0
python ImGSL_main.py --dataset tfinance --gpu 0
python ImGSL_main.py --dataset tsocial --gpu 0
python ImGSL_main.py --dataset comp --gpu 0
python ImGSL_main.py --dataset amazon --remove_hetero 1 --oversample_ratio 0.5 --gpu 0   # paper-literal Eq. 9
cd ../..
```

#### CGNN (`algorithms/CGNN/CGNN_main.py`)

Context-aware GNN for fraud detection with extremely limited labels (AAAI 2025), implemented from the paper (no official code): each neighbour is decomposed into normal / abnormal subspace components (supervised contrastive loss), mixed by a per-neighbour denoising attention, and aggregated with `1/sqrt(d_i)` scaling; unlabeled gray centres get consistency training on subspace-permuted neighbour messages. Mini-batch training; same 40/20/40 protocol by default, or the paper's 10/30/60 split with 1 or 5 labeled frauds (`--n_black`).

```powershell
cd algorithms/CGNN
python CGNN_main.py --dataset yelpchi --gpu 0
python CGNN_main.py --dataset amazon --gpu 0
python CGNN_main.py --dataset elliptic --gpu 0
python CGNN_main.py --dataset tfinance --gpu 0
python CGNN_main.py --dataset tsocial --gpu 0
python CGNN_main.py --dataset comp --gpu 0
python CGNN_main.py --dataset amazon --n_black 5 --gpu 0   # paper few-label split
cd ../..
```

#### H2IDE (`algorithms/H2IDE/H2IDE_main.py`)

Homophily and Heterophily Identification with Disentangled Embeddings (IEEE TKDE 2025), implemented from the paper (no official code): `K` factor embeddings per node, factor-level edge homophily trained with a hinge loss on labeled edges, signed per-relation per-factor attention, factor-aware relation mixing, and a CLUB mutual-information penalty between factors. Mini-batch training with two-hop neighbour sampling and exact layer-wise inference; same 40/20/40 protocol.

```powershell
cd algorithms/H2IDE
python H2IDE_main.py --dataset yelpchi --gpu 0
python H2IDE_main.py --dataset amazon --gpu 0
python H2IDE_main.py --dataset elliptic --gpu 0
python H2IDE_main.py --dataset tfinance --gpu 0
python H2IDE_main.py --dataset tsocial --gpu 0
python H2IDE_main.py --dataset comp --gpu 0
cd ../..
```

#### CIE-GAD (`algorithms/CIEGAD/CIEGAD_main.py`)

Correlation Information Enhanced GAD via Hypergraph Transformation (IEEE TCYB 2025), implemented from the paper (no official code): Jacobi spectral filters with node-level attention fusion (SCNAF), dual-hypergraph co-occurrence convolution (HCRE), and deep supervision. Full-batch training on the homogeneous union graph; same 40/20/40 protocol.

```powershell
cd algorithms/CIEGAD
python CIEGAD_main.py --dataset yelpchi --gpu 0
python CIEGAD_main.py --dataset amazon --gpu 0
python CIEGAD_main.py --dataset elliptic --gpu 0
python CIEGAD_main.py --dataset tfinance --gpu 0
python CIEGAD_main.py --dataset tsocial --gpu 0
python CIEGAD_main.py --dataset comp --gpu 0
cd ../..
```

#### COFRAUD (`algorithms/COFRAUD/COFRAUD_main.py`)

Correlation-aware fraud detection for synergistic camouflage (IJCAI 2023): intra-relation tanh frequency attention with degree-normalised residual aggregation, inter-relation crossing, and relation-semantic fusion. Full-batch multi-relation CSR; same 40/20/40 protocol.

```powershell
cd algorithms/COFRAUD
python COFRAUD_main.py --dataset yelpchi --gpu 0
python COFRAUD_main.py --dataset amazon --gpu 0
python COFRAUD_main.py --dataset elliptic --gpu 0
python COFRAUD_main.py --dataset tfinance --gpu 0
python COFRAUD_main.py --dataset tsocial --gpu 0
python COFRAUD_main.py --dataset comp --gpu 0
cd ../..
```

#### PMP (`algorithms/PMP/PMP_main.py`)

Partitioning Message Passing (ICLR 2024), implemented from `paper/papers-to-read/PMP.pdf` with the official code's details: neighbours are summed separately by their training label (fraud / benign / unlabelled), with root-specific weights, an adaptive fraud/benign mix for unlabelled neighbours, and one layer per relation followed by concatenation and an MLP. With one layer (the paper's setting) the partitioned neighbour sums are precomputed once, so training is mini-batched over nodes and scales to T-Social. No DGL; same 40/20/40 protocol.

```powershell
cd algorithms/PMP
python PMP_main.py --dataset yelpchi --gpu 0
python PMP_main.py --dataset amazon --gpu 0
python PMP_main.py --dataset elliptic --gpu 0
python PMP_main.py --dataset tfinance --gpu 0
python PMP_main.py --dataset tsocial --gpu 0
python PMP_main.py --dataset comp --gpu 0
cd ../..
```

#### SEC-GFD (`algorithms/SEC_GFD/SEC_GFD_main.py`)

Spectrum-based fraud detector (AAAI 2024), re-implemented from the released code in `reference/SEC-GFD-main.zip`. It combines Beta-wavelet band-pass filters with high-pass Laplacian filters (the hybrid filtering module), and a GCN-based local environmental constraint loss. Full-batch on `torch.sparse`; no DGL; same 40/20/40 protocol.

```powershell
cd algorithms/SEC_GFD
python SEC_GFD_main.py --dataset yelpchi --gpu 0
python SEC_GFD_main.py --dataset amazon --gpu 0
python SEC_GFD_main.py --dataset elliptic --gpu 0
python SEC_GFD_main.py --dataset tfinance --gpu 0
python SEC_GFD_main.py --dataset tsocial --gpu 0
python SEC_GFD_main.py --dataset comp --gpu 0
cd ../..
```

The older DGL runners (`algorithms/PMP/PMP_benchmark.py`, `algorithms/SEC_GFD/SEC_GFD_benchmark.py`) remain only to reproduce earlier results in `results/<Model>/<dataset>/seed_<seed>/train_<pct>/`. They are not part of the benchmark.

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
results/<Model>/<dataset>/seed<seed>_train<pct>/   # benchmark models (legacy models: seed<seed>/)
  config.json       # full CLI / hyperparameter snapshot
  metrics.json      # train + test metrics, split counts, train time, parameter count
  split.npz         # train_idx, test_idx, y_train, y_test
  predictions.npz   # test predictions + probabilities
  checkpoint.pt     # {state_dict, config, model_name}
  OOT.txt / OOM.txt # only when the run was stopped by the 4 h / 100 GB limit (then no other files)
```

| File              | Contents                                                                                                                    |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `config.json`     | All training arguments (model, seed, test_size, lr, …)                                                                      |
| `metrics.json`    | `train` / `test` metrics, `split` report, and `complexity` (`train_time_sec`, `num_parameters`, `num_trainable_parameters`) |
| `split.npz`       | Stratified indices and labels for reproducible re-evaluation                                                                |
| `predictions.npz` | Held-out test predictions and probabilities                                                                                 |
| `checkpoint.pt`   | PyTorch checkpoint for loading the trained model                                                                            |

Examples:

- `results/GAS/example/seed123/metrics.json`
- `results/Player2Vec/dblp/seed123/checkpoint.pt`
- `results/GraphSage/yelpchi/seed123/predictions.npz`
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
# → results/GraphSage/yelpchi/seed123/

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

## Hyperparameter Policy

Each benchmark model (`algorithms/<Model>`) runs with one of two kinds of hyperparameters per dataset:

- **Paper**: the model's original paper (`paper/papers-to-read`) reports results on the dataset, so the paper's hyperparameters are used (the per-dataset `DATASET_DEFAULTS` in each `*_main.py`; values the paper omits come from the official code).
- **Tuned**: otherwise the hyperparameters are chosen by `tune_hparams.py`, a validation-only random search on the 40/20/40 split (train ratio 0.4, the five benchmark seeds `123, 1234, 12345, 123456, 1234567`, ranked by mean validation AUROC; the test split is never used for selection). Trial 0 is always the previous default. The winner is written to `configs/tuned/<Model>/<dataset>.json` and loaded automatically by the `*_main.py` scripts and `utils.notebook_runner`.

**Optimizer:** every model trains with Adam (`torch.optim.Adam`); no other optimizer is used. Its learning rate and weight decay are hyperparameters like any other: paper values on the paper's datasets, tuned values (both are in every search space) elsewhere.

Explicit command-line flags always win over tuned values. Every run prints, and saves in `config.json`, its `hparam_source` (`paper`, `tuned`, or `untuned_default` when a pair still needs tuning).

| Model       | YelpChi | Amazon | T-Finance | T-Social | Elliptic | FDCompCN |
| ----------- | :-----: | :----: | :-------: | :------: | :------: | :------: |
| AMNet       | paper   | tune   | tune      | tune     | paper    | tune     |
| BSL         | paper   | paper  | paper     | paper    | paper    | tune     |
| BWGNN       | paper   | paper  | paper     | paper    | tune     | tune     |
| CARE-GNN    | paper   | paper  | tune      | tune     | tune     | tune     |
| CGNN        | paper   | paper  | paper     | tune     | paper    | tune     |
| CIE-GAD     | paper   | paper  | paper     | tune     | paper    | tune     |
| COFRAUD     | paper   | paper  | tune      | tune     | tune     | tune     |
| DGA-GNN     | paper   | paper  | paper     | paper    | paper    | tune     |
| FRAUDRE     | paper   | paper  | tune      | tune     | tune     | tune     |
| GDN ¹       | tune    | tune   | tune      | tune     | tune     | tune     |
| GGCN        | tune    | tune   | tune      | tune     | tune     | tune     |
| GHRN ¹      | tune    | tune   | tune      | tune     | tune     | tune     |
| GLHN        | paper   | paper  | paper     | paper    | tune     | tune     |
| GraphConsis | paper   | tune   | tune      | tune     | tune     | tune     |
| H2-FDetector| paper   | paper  | tune      | tune     | tune     | tune     |
| H2IDE       | paper   | paper  | paper     | paper    | tune     | tune     |
| ImGSL       | paper   | paper  | tune      | tune     | tune     | tune     |
| PC-GNN      | paper   | paper  | tune      | tune     | tune     | tune     |
| PMP         | paper   | paper  | paper     | paper    | tune     | tune     |
| SEC-GFD ²   | paper   | paper  | paper     | paper    | tune     | tune     |
| SparseGAD   | paper   | paper  | tune      | tune     | tune     | tune     |
| SplitGNN    | paper   | paper  | tune      | tune     | tune     | paper    |
| TAM         | paper   | paper  | paper     | tune     | tune     | tune     |

¹ Paper not in `paper/papers-to-read`, so every dataset is tuned. Add the PDF and list its datasets in `utils/hparams.py` (`PAPER_EVALUATED`) to switch to paper settings.

² The SEC-GFD paper is not in `paper/papers-to-read`. Its four datasets and settings come from the released code (`reference/SEC-GFD-main.zip`).

The registry (`utils/hparams.py`) cites the paper section behind every "paper" cell; the search spaces (`utils/hparam_spaces.py`) follow GADBench App. C, GLHAD §4.1.4, PMP App. D.4 and each paper's sensitivity study.

```powershell
python tune_hparams.py --list                                   # matrix above + which pairs are already tuned
python tune_hparams.py --model BWGNN --dataset elliptic --gpu 0 # one pair (20 trials x 5 seeds by default)
python tune_hparams.py --all --skip_dataset tsocial --gpu 0     # every pair still marked "tune"
python tune_hparams.py --all --resume --gpu 0                   # continue an interrupted sweep
# Budget knobs: --n_trials, --seeds 123,1234, --max_epochs, --prune_margin 0.05
```

Per-trial training output goes to `tuning_logs/<Model>/<dataset>.log` (git-ignored); the JSON keeps every trial's per-seed validation AUROC.

---

## Running the Full Benchmark

The benchmark is **23 models × 6 datasets × 4 training ratios × 5 seeds** (2,760 runs). Training does **not** tune automatically: a run on a dataset its paper did not evaluate uses `configs/tuned/<Model>/<dataset>.json` if it exists, and otherwise falls back to untuned defaults with a warning. So the order matters: **tune first, then train**.

Every command below runs from the **repo root** with the venv active. On RunPod (Linux) run them inside `tmux` (see [`docs/RUNPOD.md`](docs/RUNPOD.md)). The complete explicit list, one line per run, is in **[`docs/RUN_COMMANDS.md`](docs/RUN_COMMANDS.md)**. Regenerate it with `python docs/make_run_commands.py`.

| Setting        | Values |
| -------------- | ------ |
| Models         | AMNet, BSL, BWGNN, CAREGNN, CGNN, CIEGAD, COFRAUD, DGAGNN, FRAUDRE, GDN, GGCN, GHRN, GLHN, GraphConsis, H2FDetector, H2IDE, ImGSL, PCGNN, PMP, SEC_GFD, SparseGAD, SplitGNN, TAM |
| Datasets       | `yelpchi`, `amazon`, `tfinance`, `tsocial`, `elliptic`, `comp` (FDCompCN) |
| Training ratio | `0.1`, `0.2`, `0.3`, `0.4` (valid 20% / test 40% fixed) |
| Seeds          | `123`, `1234`, `12345`, `123456`, `1234567` |
| Output folder  | `results/<Model>/<dataset>/seed<seed>_train<pct>/`, for example `results/BWGNN/comp/seed123_train10/` |
| Limits per run | 4 hours (otherwise `OOT.txt`) and 100 GB of GPU memory (otherwise `OOM.txt`) |

**Folder names are built automatically** from the model, dataset, seed and training ratio, so no two commands in the benchmark write to the same folder, across models too. You can run the command list from top to bottom without anything being overwritten. Dataset names are normalised (`YelpChi.mat` / `yelp` → `yelpchi`, `fdcompcn` → `comp`), so every model uses the same six dataset folder names. You don't need `--run_tag`. If you add one, it is appended (`seed123_train10_<tag>`). Running the *same* command twice reuses its folder, which the loops below avoid by skipping finished runs.

**Do not pass hyperparameter flags** (`--learning_rate`, `--weight_decay`, `--hid_dim`, `--epochs` for most models, ...) in benchmark runs. Explicit flags override both the paper and the tuned values. The one exception is GraphConsis: it gets `--sigmoid --epochs 500`, because its CLI default is only 10 epochs. Early stopping (patience 25) still ends it sooner.

### Time and memory limits (OOT / OOM)

Every run is checked against two limits (`utils/run_guard.py`):

| Limit | What happens | Exit code |
| --- | --- | --- |
| **OOT**: the run takes more than **4 hours** (wall clock from start, including data loading) | The run is stopped and `OOT.txt` is written to its run folder | `124` |
| **OOM**: the run needs more than **100 GB of GPU memory** | PyTorch's allocator is capped at 100 GB (GiB). The first allocation past the cap stops the run and writes `OOM.txt` | `125` |

- The status file's first line is just `OOT` or `OOM`. After it come the model, dataset, seed, training ratio, elapsed time, GPU, the error message (OOM) and the exact command.
- A stopped run writes no other result files.
- The loops below don't retry stopped runs. To retry one, delete its `OOT.txt` / `OOM.txt`. Starting the same run again also removes the old status file.
- **GPU size matters for OOM.** The 100 GB criterion is exact only on a GPU with more than 100 GB, such as an H200 (141 GB) or B200 (180 GB). On a smaller GPU, for example an 80 GB A100/H100, the device itself is the limit: a run that needs 85 GB is marked OOM, and `OOM.txt` notes this.
- A process killed by the operating system for using too much **CPU RAM** cannot write a file. It shows up as `FAILED` in the loop.
- For a quick test of the mechanism you can lower the limits, e.g. `GFW_TIME_LIMIT_HOURS=0.01` or `GFW_VRAM_LIMIT_GB=1`. Leave them unset for the benchmark.

### Step 0: datasets

`dataset/` must contain `YelpChi.mat`, `Amazon.mat`, `tfinance`, `tsocial`, `elliptic_bitcoin_dataset/` (three CSVs) and `comp.dgl` (see [Datasets](#datasets)).

```bash
ls dataset/YelpChi.mat dataset/Amazon.mat dataset/tfinance dataset/tsocial dataset/comp.dgl dataset/elliptic_bitcoin_dataset/
```

### Step 1: tune the pairs without paper settings (77 pairs, once)

```bash
python tune_hparams.py --list                 # TUNE = still to do, tuned = done, paper = never tuned
python tune_hparams.py --all --gpu 0          # all 77 pairs: 20 trials x 5 seeds each on the 40% split
python tune_hparams.py --all --resume --gpu 0 # after an interruption: finished pairs are skipped, the open one resumes
python tune_hparams.py --list                 # repeat until no TUNE cell is left
```

Per-pair commands, to split the work across pods or run small datasets first, are in [`docs/RUN_COMMANDS.md`](docs/RUN_COMMANDS.md#step-1---hyperparameter-tuning-pairs-not-evaluated-in-the-original-paper). An example:

```bash
python tune_hparams.py --model GGCN --dataset comp --gpu 0
python tune_hparams.py --all --dataset comp,elliptic --gpu 0    # every model, two datasets
python tune_hparams.py --all --skip_dataset tsocial --gpu 0     # everything except T-Social
```

Commit the resulting `configs/tuned/**/*.json` (or copy them to every machine that trains), so training runs pick them up.

### Step 2: train every model × dataset × ratio × seed

**bash (RunPod / Linux):** finished runs are skipped: those with `metrics.json` (done), `OOT.txt` or `OOM.txt` (stopped). So the loop can be restarted at any time.

```bash
mkdir -p logs
MODELS="AMNet BSL BWGNN CAREGNN CGNN CIEGAD COFRAUD DGAGNN FRAUDRE GDN GGCN GHRN GLHN H2FDetector H2IDE ImGSL PCGNN PMP SEC_GFD SparseGAD SplitGNN TAM"
DATASETS="yelpchi amazon tfinance tsocial elliptic comp"
RATIOS="0.1 0.2 0.3 0.4"
SEEDS="123 1234 12345 123456 1234567"
declare -A GC_FILE=([yelpchi]=YelpChi.mat [amazon]=Amazon.mat [tfinance]=tfinance [tsocial]=tsocial [elliptic]=elliptic [comp]=comp)

for M in $MODELS GraphConsis; do for D in $DATASETS; do for R in $RATIOS; do for S in $SEEDS; do
  P=$(awk "BEGIN{print int($R*100+0.5)}")
  RUN="results/$M/$D/seed${S}_train$P"
  LOG="logs/${M}_${D}_train${P}_seed${S}.log"
  if [ -f "$RUN/metrics.json" ] || [ -f "$RUN/OOT.txt" ] || [ -f "$RUN/OOM.txt" ]; then continue; fi
  echo ">>> $M $D train=$R seed=$S"
  if [ "$M" = "GraphConsis" ]; then
    python algorithms/GraphConsis/supervised_train.py --train_prefix dataset/ --file_name "${GC_FILE[$D]}" \
      --sigmoid --epochs 500 --seed $S --train_ratio $R --gpu 0 > "$LOG" 2>&1
  else
    python algorithms/$M/${M}_main.py --dataset $D --data_prefix dataset/ \
      --seed $S --train_ratio $R --gpu 0 > "$LOG" 2>&1
  fi
  case $? in
    0)   ;;
    124) echo "OOT: $M $D train=$R seed=$S (over 4 h, see $RUN/OOT.txt)" ;;
    125) echo "OOM: $M $D train=$R seed=$S (over 100 GB GPU memory, see $RUN/OOM.txt)" ;;
    *)   echo "FAILED: $M $D train=$R seed=$S (see $LOG)" ;;
  esac
done; done; done; done
```

To run a subset, shorten `MODELS` / `DATASETS` (e.g. `MODELS="BWGNN"`, `DATASETS="comp"`).

**PowerShell (Windows):**

```powershell
New-Item -ItemType Directory -Force logs | Out-Null
$Models   = 'AMNet','BSL','BWGNN','CAREGNN','CGNN','CIEGAD','COFRAUD','DGAGNN','FRAUDRE','GDN','GGCN','GHRN','GLHN','H2FDetector','H2IDE','ImGSL','PCGNN','PMP','SEC_GFD','SparseGAD','SplitGNN','TAM','GraphConsis'
$Datasets = 'yelpchi','amazon','tfinance','tsocial','elliptic','comp'
$Ratios   = 0.1,0.2,0.3,0.4
$Seeds    = 123,1234,12345,123456,1234567
$GcFile   = @{ yelpchi='YelpChi.mat'; amazon='Amazon.mat'; tfinance='tfinance'; tsocial='tsocial'; elliptic='elliptic'; comp='comp' }

foreach ($M in $Models) { foreach ($D in $Datasets) { foreach ($R in $Ratios) { foreach ($S in $Seeds) {
  $P   = [int][math]::Round($R * 100)
  $run = "results/$M/$D/seed${S}_train$P"
  $log = "logs/${M}_${D}_train${P}_seed${S}.log"
  if ((Test-Path "$run/metrics.json") -or (Test-Path "$run/OOT.txt") -or (Test-Path "$run/OOM.txt")) { continue }
  Write-Host ">>> $M $D train=$R seed=$S"
  if ($M -eq 'GraphConsis') {
    python algorithms/GraphConsis/supervised_train.py --train_prefix dataset/ --file_name $GcFile[$D] --sigmoid --epochs 500 --seed $S --train_ratio $R --gpu 0 *> $log
  } else {
    python "algorithms/$M/${M}_main.py" --dataset $D --data_prefix dataset/ --seed $S --train_ratio $R --gpu 0 *> $log
  }
  $code = $LASTEXITCODE
  if ($code -eq 124)     { Write-Host "OOT: $M $D train=$R seed=$S (over 4 h, see $run/OOT.txt)" }
  elseif ($code -eq 125) { Write-Host "OOM: $M $D train=$R seed=$S (over 100 GB GPU memory, see $run/OOM.txt)" }
  elseif ($code -ne 0)   { Write-Host "FAILED: $M $D train=$R seed=$S (see $log)" }
}}}}
```

A single run looks like this (the full list is in [`docs/RUN_COMMANDS.md`](docs/RUN_COMMANDS.md#step-2---training-every-model-x-dataset-x-training-ratio-x-seed)):

```bash
python algorithms/BWGNN/BWGNN_main.py --dataset elliptic --data_prefix dataset/ --seed 123 --train_ratio 0.4 --gpu 0
python algorithms/GraphConsis/supervised_train.py --train_prefix dataset/ --file_name YelpChi.mat --sigmoid --epochs 500 --seed 123 --train_ratio 0.4 --gpu 0
```

### Step 3: check the runs

```bash
# Per model: done / OOT / OOM (total should be 120 = 6 datasets x 4 ratios x 5 seeds)
python -c "import glob,collections,pathlib as P; K={'metrics.json':'done','OOT.txt':'OOT','OOM.txt':'OOM'}; c=collections.defaultdict(collections.Counter); fs=glob.glob('results/*/*/seed*_train*/*'); [c[P.Path(f).parts[1]].update([K[P.Path(f).name]]) for f in fs if P.Path(f).name in K]; [print(m, dict(v), 'total', sum(v.values())) for m,v in sorted(c.items())]"
# List every stopped run
python -c "import glob; print(*sorted(glob.glob('results/*/*/seed*_train*/OO[TM].txt')), sep='\n')"
# Runs that trained without a tuned config (should print 0: re-tune that pair and re-run)
python -c "import glob,json; b=[p for p in glob.glob('results/*/*/seed*_train*/config.json') if json.load(open(p)).get('hparam_source')=='untuned_default']; print(len(b)); print(*b[:20], sep='\n')"
```

All three commands work on Linux and Windows. Download `results/` as described in [`docs/RUNPOD.md`](docs/RUNPOD.md#9-zip-results-on-the-pod-and-download-to-your-pc).

---

## Training Arguments

### Shared args (`main.py` and most `*_main.py`)

| Argument          | Default                                | Description                                     |
| ----------------- | -------------------------------------- | ----------------------------------------------- |
| `--seed`          | `123`                                  | Random seed                                     |
| `--dataset_str`   | `example` or `dblp`                    | Dataset key (`example` / `dblp`)                |
| `--epoch_num`     | `30`                                   | Training epochs                                 |
| `--batch_size`    | `1000` (SemiGNN scripts often use `4`) | Batch size                                      |
| `--learning_rate` | `0.01`                                 | Adam learning rate                              |
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

| Argument        | Default                     | Description                                |
| --------------- | --------------------------- | ------------------------------------------ |
| `--context_dim` | `0`                         | Context embedding dim (`0` disables)       |
| `--train_perc`  | `1.`                        | Fraction of the 80% train split to use     |
| `--early_stop`  | `25`                        | Valid-AUC patience (`0` trains all epochs) |
| `--save_dir`    | `../../results/GraphConsis` | Eval output directory                      |
| `--no_save`     | flag                        | Skip writing eval outputs                  |

### BWGNN (`algorithms/BWGNN/BWGNN_main.py`)

| Argument          | Default    | Description                                                            |
| ----------------- | ---------- | ---------------------------------------------------------------------- |
| `--dataset`       | `tfinance` | `tfinance` / `tsocial` / `yelpchi` / `amazon` / `elliptic` / `comp`    |
| `--train_ratio`   | `0.4`      | Train fraction of labeled nodes (max `0.4`; valid/test fixed at 20/40) |
| `--hid_dim`       | `64`       | Hidden size (`10` recommended for T-Social)                            |
| `--order`         | `2`        | Beta wavelet order C (`5` for T-Social)                                |
| `--epochs`        | `100`      | Epochs                                                                 |
| `--early_stop`    | `25`       | Valid-AUC patience (`0` trains all epochs)                             |
| `--learning_rate` | `0.01`     | Learning rate                                                          |
| `--gpu`           | `0`        | Device                                                                 |

### SplitGNN (`algorithms/SplitGNN/SplitGNN_main.py`)

| Argument          | Default     | Description                                                            |
| ----------------- | ----------- | ---------------------------------------------------------------------- |
| `--dataset`       | `comp`      | `comp` / `yelpchi` / `amazon` / `elliptic`                             |
| `--train_ratio`   | `0.4`       | Train fraction of labeled nodes (max `0.4`; valid/test fixed at 20/40) |
| `--intra_dim`     | paper table | Hidden size (`8`)                                                      |
| `--C` / `--K`     | paper table | Wavelet order / positive-band cutoff (`3` / `2` on comp)               |
| `--gamma`         | paper table | Homo-edge hinge weight (`0.8` on comp)                                 |
| `--epochs`        | `1000`      | Max epochs (early stop on valid AUC)                                   |
| `--early_stop`    | paper table | Patience (`100` on comp; `0` disables)                                 |
| `--learning_rate` | `0.01`      | Adam learning rate (same default on every dataset)                     |
| `--gpu`           | `0`         | Device                                                                 |

### DGA-GNN (`algorithms/DGAGNN/DGAGNN_main.py`)

| Argument          | Default    | Description                                                         |
| ----------------- | ---------- | ------------------------------------------------------------------- |
| `--dataset`       | `yelpchi`  | `yelpchi` / `amazon` / `elliptic` / `tfinance` / `tsocial` / `comp` |
| `--train_ratio`   | `0.4`      | Train fraction of labeled nodes (valid/test fixed 20/40)            |
| `--n_hidden`      | paper yaml | Hidden size                                                         |
| `--n_head`        | paper yaml | Attention heads                                                     |
| `--n_bins`        | paper yaml | Decision-tree bins per feature (`k`)                                |
| `--z`             | paper yaml | Grouping threshold                                                  |
| `--epochs`        | `200`      | Max epochs (early stop on valid AUC)                                |
| `--learning_rate` | `0.01`     | Learning rate                                                       |
| `--gpu`           | `0`        | Device                                                              |

### GLHN (`algorithms/GLHN/GLHN_main.py`)

| Argument                | Default   | Description                                                         |
| ----------------------- | --------- | ------------------------------------------------------------------- |
| `--dataset`             | `yelpchi` | `yelpchi` / `amazon` / `elliptic` / `tfinance` / `tsocial` / `comp` |
| `--train_ratio`         | `0.4`     | Train fraction of labeled nodes (valid/test fixed 20/40)            |
| `--hid_dim`             | `32`      | Hidden size (`10` on T-Social)                                      |
| `--n_layers`            | `2`       | GLHN layers                                                         |
| `--tau`                 | `0.1`     | InfoNCE temperature                                                 |
| `--gamma1` / `--gamma2` | paper     | Weights of `L_k` / `L_u` (0.8/1.0 on YelpChi)                       |
| `--graph_type`          | dataset   | `hetero` (mean-pool relations) or `homo`                            |
| `--epochs`              | `100`     | Max epochs (early stop on valid AUC)                                |
| `--learning_rate`       | `0.01`    | Learning rate                                                       |
| `--gpu`                 | `0`       | Device                                                              |

### BSL (`algorithms/BSL/BSL_main.py`)

| Argument             | Default       | Description                                                         |
| -------------------- | ------------- | ------------------------------------------------------------------- |
| `--dataset`          | `yelpchi`     | `yelpchi` / `amazon` / `elliptic` / `tfinance` / `tsocial` / `comp` |
| `--train_ratio`      | `0.4`         | Train fraction of labeled nodes (valid/test fixed 20/40)            |
| `--paper_split`      | off           | Paper 1%/10%/89% split (0.01%/0.1% on T-Social)                     |
| `--hid_dim`          | `96`          | Latent size, divisible by 3 (`48` on T-Social)                      |
| `--alpha` / `--beta` | `0.4` / `0.8` | Weights of `L_D` / `L_bsl`                                          |
| `--batch_size`       | `256`         | Unlabeled aux batch for `L_bsl`                                     |
| `--epochs`           | `100`         | Max epochs (early stop on valid AUC)                                |
| `--learning_rate`    | `0.01`        | Learning rate                                                       |
| `--gpu`              | `0`           | Device                                                              |

### CARE-GNN (`algorithms/CAREGNN/CAREGNN_main.py`)

| Argument           | Default   | Description                                                         |
| ------------------ | --------- | ------------------------------------------------------------------- |
| `--dataset`        | `yelpchi` | `yelpchi` / `amazon` / `elliptic` / `tfinance` / `tsocial` / `comp` |
| `--train_ratio`    | `0.4`     | Train fraction of labeled nodes (valid/test fixed 20/40)            |
| `--inter`          | `GNN`     | Inter-relation aggregator: `GNN` / `Mean` / `Weight` / `Att`        |
| `--emb_size`       | `64`      | Embedding size                                                      |
| `--batch_size`     | dataset   | 1024 (YelpChi, Elliptic, T-Finance, T-Social) / 256 (Amazon, comp)  |
| `--lambda_1`       | `2.0`     | Weight of the label-predictor (similarity) loss                     |
| `--weight_decay`   | `1e-3`    | L2 weight (`lambda_2`)                                              |
| `--step_size`      | `0.02`    | RL threshold step                                                   |
| `--under_sample`   | `1.0`     | Negatives drawn per positive each epoch                             |
| `--no_rl`          | off       | Freeze thresholds at `--init_threshold` (0.5)                       |
| `--epochs`         | `100`     | Max epochs (early stop on valid AUC)                                |
| `--learning_rate`  | `0.01`    | Learning rate                                                       |
| `--gpu`            | `0`       | Device                                                              |

### H2-FDetector (`algorithms/H2FDetector/H2FDetector_main.py`)

| Argument          | Default   | Description                                                          |
| ----------------- | --------- | -------------------------------------------------------------------- |
| `--dataset`       | `yelpchi` | `yelpchi` / `amazon` / `elliptic` / `tfinance` / `tsocial` / `comp`  |
| `--train_ratio`   | `0.4`     | Train fraction of labeled nodes (valid/test fixed 20/40)             |
| `--intra_dim`     | `8`       | Hidden size per head                                                 |
| `--head`          | `4`       | Attention heads                                                      |
| `--n_layer`       | `2`       | H2-FDetector layers                                                  |
| `--dropout`       | `0.1`     | Dropout (edge scorer, relation concat, between layers)               |
| `--gamma1`        | dataset   | Edge hinge-loss weight: 1.2 (YelpChi and others) / 0.4 (Amazon)      |
| `--gamma2`        | dataset   | Prototype-loss weight: 2.0 (YelpChi and others) / 1.4 (Amazon)       |
| `--feat_norm`     | dataset   | `row` (YelpChi, Amazon) / `standard` (others) / `none`               |
| `--epochs`        | `1000`    | Max epochs, early stop on valid AUC (T-Social: 200)                  |
| `--learning_rate` | `0.1`     | Learning rate                                                        |
| `--weight_decay`  | `5e-5`    | L2 weight                                                            |
| `--gpu`           | `0`       | Device                                                               |

### FRAUDRE (`algorithms/FRAUDRE/FRAUDRE_main.py`)

| Argument            | Default   | Description                                                          |
| ------------------- | --------- | -------------------------------------------------------------------- |
| `--dataset`         | `yelpchi` | `yelpchi` / `amazon` / `elliptic` / `tfinance` / `tsocial` / `comp`  |
| `--train_ratio`     | `0.4`     | Train fraction of labeled nodes (valid/test fixed 20/40)             |
| `--embed_dim`       | `64`      | Ego-feature embedding size `d` (final embedding `7d` for 2 layers)   |
| `--n_layers`        | `2`       | Fraud-aware convolution layers                                       |
| `--hidden_dim`      | `64`      | Classifier hidden size                                               |
| `--max_neighbors`   | `10`      | Fixed neighbour sample per node and relation (0 keeps all)           |
| `--batch_size`      | dataset   | 1024 (YelpChi, Elliptic, T-Finance, T-Social) / 256 (Amazon, comp)   |
| `--learning_rate`   | dataset   | 0.001 (YelpChi) / 0.1 (Amazon) / 0.01 (others)                       |
| `--weight_decay`    | `1e-4`    | L2 weight                                                            |
| `--feat_norm`       | dataset   | `row` (YelpChi, Amazon) / `standard` (others) / `none`               |
| `--epochs`          | `100`     | Max epochs, early stop on valid AUC (T-Social: 50)                   |
| `--gpu`             | `0`       | Device                                                               |

### PC-GNN (`algorithms/PCGNN/PCGNN_main.py`)

| Argument            | Default   | Description                                                          |
| ------------------- | --------- | -------------------------------------------------------------------- |
| `--dataset`         | `yelpchi` | `yelpchi` / `amazon` / `elliptic` / `tfinance` / `tsocial` / `comp`  |
| `--train_ratio`     | `0.4`     | Train fraction of labeled nodes (valid/test fixed 20/40)             |
| `--emb_size`        | `64`      | Embedding size                                                       |
| `--rho`             | `0.5`     | Oversampled fraud neighbours per fraud centre (fraction of kept)     |
| `--alpha`           | `2.0`     | Label-predictor loss weight                                          |
| `--threshold`       | `0.5`     | Fraction of neighbours kept per relation                             |
| `--thres`           | dataset   | Decision threshold on the fraud probability: 0.4 (Amazon) / 0.5      |
| `--batch_size`      | dataset   | 1024 (YelpChi, Elliptic, T-Finance, T-Social) / 256 (Amazon, comp)   |
| `--learning_rate`   | dataset   | 0.01 (YelpChi and large graphs) / 0.005 (Amazon, comp)               |
| `--weight_decay`    | dataset   | 1e-3 (YelpChi and large graphs) / 5e-4 (Amazon, comp)                |
| `--feat_norm`       | dataset   | `row` (YelpChi, Amazon) / `standard` (others) / `none`               |
| `--epochs`          | `100`     | Max epochs, early stop on valid AUC (T-Social: 50)                   |
| `--gpu`             | `0`       | Device                                                               |

### SparseGAD (`algorithms/SparseGAD/SparseGAD_main.py`)

| Argument             | Default   | Description                                                          |
| -------------------- | --------- | -------------------------------------------------------------------- |
| `--dataset`          | `yelpchi` | `yelpchi` / `amazon` / `elliptic` / `tfinance` / `tsocial` / `comp`  |
| `--train_ratio`      | `0.4`     | Train fraction of labeled nodes (valid/test fixed 20/40)             |
| `--hidden`           | dataset   | 32 (YelpChi and others) / 16 (Amazon)                                |
| `--nlayers`          | `2`       | GPR propagation layers                                               |
| `--dropout`          | dataset   | Feature dropout: 0.1 (YelpChi and others) / 0.2 (Amazon)             |
| `--dropout_adj`      | dataset   | Edge-weight dropout: 0.25 (YelpChi and others) / 0.1 (Amazon)        |
| `--k`                | `10`      | Neighbours per node in the kNN graph update                          |
| `--update_adj_epoch` | dataset   | kNN graph update interval: 200 (T-Social: 0 = off)                   |
| `--feat_norm`        | dataset   | `minmax` (YelpChi) / `row` (Amazon) / `standard` (others) / `none`   |
| `--epochs`           | `1000`    | Max epochs, early stop on valid AUC with patience 200 (T-Social: 300 / 50) |
| `--learning_rate`    | `0.01`    | Learning rate                                                        |
| `--weight_decay`     | `5e-4`    | L2 weight                                                            |
| `--gpu`              | `0`       | Device                                                               |

### AMNet (`algorithms/AMNet/AMNet_main.py`)

| Argument          | Default    | Description                                                                 |
| ----------------- | ---------- | --------------------------------------------------------------------------- |
| `--dataset`       | `yelpchi`  | `yelpchi` / `amazon` / `elliptic` / `tfinance` / `tsocial` / `comp`         |
| `--train_ratio`   | `0.4`      | Train fraction of labeled nodes (valid/test fixed 20/40)                    |
| `--relation`      | dataset    | `rur` (YelpChi) / `all` = union of relations (others)                       |
| `--hidden`        | dataset    | 64 (YelpChi, Amazon, FDCompCN) / 128 (Elliptic, T-Finance) / 32 (T-Social)  |
| `--order`         | dataset    | Bernstein order M: 5 / 4 (Elliptic) / 7 (T-Finance)                         |
| `--n_filters`     | `2`        | Number of Bernstein filters K                                               |
| `--beta`          | dataset    | Marginal loss weight: 1.0 / 0.5 (Elliptic)                                  |
| `--learning_rate` | dataset    | MLP / attention lr: `5e-4` / `1e-4` (Elliptic)                              |
| `--lr_filter`     | `5e-2`     | Filter coefficient lr (no weight decay)                                     |
| `--weight_decay`  | `1e-5`     | L2 weight of non-filter parameters                                          |
| `--dropout`       | `0.3`      | Dropout before the classifier                                               |
| `--feat_norm`     | `standard` | `row` / `minmax` / `standard` / `none`                                      |
| `--loss_weight`   | `none`     | `none` (reference) / `balanced` CE class weights                            |
| `--select_metric` | `auroc`    | Validation metric for checkpointing / early stopping (`auprc` = reference) |
| `--epochs`        | `2000`     | Max epochs, early stop with patience 200 (T-Social: 500 / 50)               |
| `--gpu`           | `0`        | Device                                                                      |

### GHRN (`algorithms/GHRN/GHRN_main.py`)

| Argument          | Default   | Description                                                                         |
| ----------------- | --------- | ----------------------------------------------------------------------------------- |
| `--dataset`       | `yelpchi` | `yelpchi` / `amazon` / `elliptic` / `tfinance` / `tsocial` / `comp`                 |
| `--train_ratio`   | `0.4`     | Train fraction of labeled nodes (valid/test fixed 20/40)                            |
| `--homo`          | `1`       | `1`: BWGNN on the union graph; `0`: per-relation BWGNN                              |
| `--hid_dim`       | dataset   | 64 / 10 (T-Social)                                                                  |
| `--order`         | dataset   | Beta wavelet order: 2 / 5 (T-Social)                                                |
| `--del_ratio`     | dataset   | Fraction of edges deleted: 0.1 (YelpChi) / 0.015 (others); `0` = plain BWGNN        |
| `--adj_type`      | `sym`     | Indicator normalisation: `sym` / `rw`                                               |
| `--self_loop`     | dataset   | Self-loops on the union graph: 1 (YelpChi, Amazon, FDCompCN) / 0 (others)           |
| `--feat_norm`     | dataset   | `none` (YelpChi, Amazon, T-Finance, T-Social) / `standard` (Elliptic, FDCompCN)     |
| `--epochs`        | `100`     | Max epochs per stage, early stop on valid AUC (patience 25)                         |
| `--learning_rate` | `0.01`    | Learning rate                                                                       |
| `--weight_decay`  | `0.0`     | L2 weight                                                                           |
| `--gpu`           | `0`       | Device                                                                              |

### GGCN (`algorithms/GGCN/GGCN_main.py`)

| Argument          | Default    | Description                                                                   |
| ----------------- | ---------- | ----------------------------------------------------------------------------- |
| `--dataset`       | `yelpchi`  | `yelpchi` / `amazon` / `elliptic` / `tfinance` / `tsocial` / `comp`           |
| `--train_ratio`   | `0.4`      | Train fraction of labeled nodes (valid/test fixed 20/40)                      |
| `--relation`      | `all`      | Relation to use (`all` = union of relations)                                  |
| `--nlayers`       | `2`        | GGCN layers (>= 2)                                                            |
| `--hidden`        | `32`       | Hidden size                                                                   |
| `--dropout`       | `0.5`      | Dropout                                                                       |
| `--decay_rate`    | `1.0`      | Decay rate of the residual weights                                            |
| `--exponent`      | `3.0`      | Exponent of the residual decay                                                |
| `--no_sign` / `--no_degree` / `--no_decay` | off | Disable signed messages / degree correction / decaying aggregation |
| `--feat_norm`     | `standard` | `row` (reference) / `minmax` / `standard` / `none`                            |
| `--loss_weight`   | `none`     | `none` (reference) / `balanced` CE class weights                              |
| `--epochs`        | `1500`     | Max epochs, early stop on valid AUC with patience 100 (T-Social: 500 / 50)    |
| `--learning_rate` | `0.01`     | Learning rate                                                                 |
| `--weight_decay`  | `4e-3`     | L2 weight                                                                     |
| `--gpu`           | `0`        | Device                                                                        |

### TAM (`algorithms/TAM/TAM_main.py`)

| Argument          | Default   | Description                                                                         |
| ----------------- | --------- | ----------------------------------------------------------------------------------- |
| `--dataset`       | `yelpchi` | `yelpchi` / `amazon` / `elliptic` / `tfinance` / `tsocial` / `comp`                 |
| `--train_ratio`   | `0.4`     | Train fraction of labeled nodes; only sets the anomaly-rate estimate for the threshold (unsupervised) |
| `--relation`      | `all`     | Relation to use (`all` = union of relations)                                        |
| `--embedding_dim` | `128`     | LAMNet embedding size                                                               |
| `--cutting`       | auto      | Truncations K (union graph `7`, single relation `4`)                                |
| `--n_tree`        | auto      | Truncation sequences T (union graph `1`, single relation `3`)                       |
| `--minmax_loss`   | auto      | Min-max normalise the affinity in the loss (union graph `1`, single relation `0`)   |
| `--lamda`         | `0`       | Weight of the non-neighbour similarity regulariser                                 |
| `--feat_norm`     | per dataset | `row` (YelpChi / Amazon, reference) / `minmax` / `standard` (Elliptic / FDCompCN) / `none` (T-Finance / T-Social) |
| `--epochs`        | `500`     | Epochs per LAMNet (no early stopping)                                               |
| `--learning_rate` | `1e-5`    | Learning rate                                                                       |
| `--weight_decay`  | `0`       | L2 weight                                                                           |
| `--gpu`           | `0`       | Device                                                                              |

### GDN (`algorithms/GDN/GDN_main.py`)

| Argument               | Default     | Description                                                                 |
| ---------------------- | ----------- | --------------------------------------------------------------------------- |
| `--dataset`            | `yelpchi`   | `yelpchi` / `amazon` / `elliptic` / `tfinance` / `tsocial` / `comp`         |
| `--train_ratio`        | `0.4`       | Train fraction of labeled nodes (valid/test fixed 20/40)                    |
| `--emb_size`           | `64`        | Embedding size                                                              |
| `--topk`               | per dataset | Anomaly feature dimensions (`10`; T-Finance / T-Social `3`)                 |
| `--beta`               | `0.1`       | Weight of the two feature constraints                                       |
| `--add_constraint`     | `1`         | `0` trains the backbone only                                                |
| `--thres`              | `0.2`       | Decision threshold on the fraud probability                                 |
| `--batch_size`         | `1024`      | Mini-batch size                                                             |
| `--learning_rate`      | `1e-3`      | Network learning rate                                                       |
| `--weight_decay`       | `1e-3`      | Network L2 weight                                                           |
| `--feat_learning_rate` | `1e-5`      | Learning rate of the trainable feature table                                |
| `--feat_weight_decay`  | `0.1`       | L2 weight of the trainable feature table                                    |
| `--feat_norm`          | per dataset | `none` (YelpChi) / `row` (Amazon) / `standard` (others)                     |
| `--epochs`             | `100`       | Max epochs, early stop on valid AUC with patience 20 (T-Social: 50 / 10)    |
| `--gpu`                | `0`         | Device                                                                      |

### ImGSL (`algorithms/ImGSL/ImGSL_main.py`)

| Argument             | Default     | Description                                                                         |
| -------------------- | ----------- | ----------------------------------------------------------------------------------- |
| `--dataset`          | `yelpchi`   | `yelpchi` / `amazon` / `elliptic` / `tfinance` / `tsocial` / `comp`                 |
| `--train_ratio`      | `0.4`       | Train fraction of labeled nodes (valid/test fixed 20/40)                            |
| `--emb_size`         | `64`        | Embedding size                                                                      |
| `--beta`             | `0.5`       | Weight of the refined input graph versus the learned graph (`1` = no learned graph) |
| `--epsilon`          | per dataset | Sparsity threshold of the learned node-anchor graph (Amazon `0.5`, others `0`)      |
| `--tau_minus`        | `0.2`       | Edge pruning threshold on label-predictor similarity                                |
| `--remove_hetero`    | `0`         | `1` = drop edges between train nodes with different labels (paper Eq. 9)            |
| `--oversample_ratio` | `0`         | Fraud oversampling `k_r = ratio * avg fraud degree` (paper `0.5`)                   |
| `--alpha`            | `2`         | Label-predictor loss weight                                                         |
| `--num_anchors`      | `512`       | Anchors of the node-anchor graph                                                    |
| `--heads`            | `4`         | Heads of the weighted cosine                                                        |
| `--max_iter` / `--delta` | `3` / `1e-2` | Maximum structure-learning iterations / dynamic stopping threshold           |
| `--no_sampler`       | off         | Uniform node sampling instead of the class-balanced sampler                         |
| `--thres`            | `0.5`       | Decision threshold on the fraud probability                                         |
| `--batch_size`       | per dataset | Amazon / FDCompCN `256`, others `1024`                                              |
| `--learning_rate`    | `0.01`      | Learning rate                                                                       |
| `--weight_decay`     | `1e-3`      | L2 weight                                                                           |
| `--feat_norm`        | `standard`  | `row` / `standard` / `none`                                                         |
| `--epochs`           | `150`       | Max epochs, early stop on valid AUC with patience 30 (T-Social: 50 / 10)            |
| `--gpu`              | `0`         | Device                                                                              |

### CGNN (`algorithms/CGNN/CGNN_main.py`)

| Argument            | Default     | Description                                                                                       |
| ------------------- | ----------- | ------------------------------------------------------------------------------------------------- |
| `--dataset`         | `yelpchi`   | `yelpchi` / `amazon` / `elliptic` / `tfinance` / `tsocial` / `comp`                               |
| `--train_ratio`     | `0.4`       | Train fraction of labeled nodes (valid/test fixed 20/40; ignored with `--n_black`)                |
| `--n_black`         | `0`         | `1` / `5` = paper split: 10/30/60, that many labeled frauds plus `eta`-times as many normals      |
| `--emb_size`        | `64`        | Embedding size                                                                                    |
| `--lam`             | `1`         | Weight of the category semantic decomposition loss (`0` = w/o CSD)                                |
| `--beta`            | `1`         | Weight of the consistency loss on gray samples (`0` = w/o CL)                                     |
| `--tau`             | per mode    | Contrastive temperature: paper `0.01` (T-Finance `0.8`) with `--n_black`, else `0.5` (T-Finance `0.8`) |
| `--csd_nodes`       | `256`       | Max distinct neighbours per step whose components enter the contrastive loss                     |
| `--batch_size`      | per mode    | Labeled (and gray) centres per step: `16` with `--n_black`, else `256` (T-Social `1024`)          |
| `--thres`           | `0.5`       | Decision threshold on the fraud probability                                                       |
| `--learning_rate`   | `0.01`      | Learning rate                                                                                     |
| `--weight_decay`    | `1e-4`      | L2 weight                                                                                         |
| `--feat_norm`       | `standard`  | `row` / `standard` / `none`                                                                       |
| `--epochs`          | `100`       | Max epochs, early stop on valid AUC with patience 25 (T-Social: 30 / 10)                          |
| `--gpu`             | `0`         | Device                                                                                            |

### H2IDE (`algorithms/H2IDE/H2IDE_main.py`)

| Argument             | Default     | Description                                                                                  |
| -------------------- | ----------- | -------------------------------------------------------------------------------------------- |
| `--dataset`          | `yelpchi`   | `yelpchi` / `amazon` / `elliptic` / `tfinance` / `tsocial` / `comp`                          |
| `--train_ratio`      | `0.4`       | Train fraction of labeled nodes (valid/test fixed 20/40)                                     |
| `--n_factors`        | per dataset | Latent factors `K` (YelpChi `12`, others `8`)                                                |
| `--factor_dim`       | `8`         | Embedding size per factor                                                                    |
| `--tau`              | `0.5`       | Temperature of the cosine proximity (not given in the paper)                                 |
| `--alpha1`           | per dataset | Identification hinge-loss weight (Amazon / FDCompCN `0.8`, others `1.2`)                     |
| `--alpha2`           | per dataset | MI independence-loss weight (YelpChi `0.001`, others `0.01`)                                 |
| `--alpha3`           | `1e-4`      | Explicit squared-L2 weight                                                                   |
| `--mi_beta`          | `1`         | Weight of the variational log-likelihood fitted by the CLUB estimators                       |
| `--mi_hidden`        | `16`        | Hidden size of the CLUB networks                                                             |
| `--residual`         | `0`         | `1` = add the centre factor embedding to its aggregated message                              |
| `--dropout`          | `0.1`       | Dropout on features and before the MLP                                                       |
| `--fanouts`          | `25 10`     | Training neighbours per relation at hop 1 / hop 2 (`0` = all); evaluation is exact           |
| `--batch_size`       | per dataset | YelpChi `256`, others `1024` (balanced, under-sampled normals)                               |
| `--thres`            | `0.5`       | Decision threshold on the fraud probability                                                  |
| `--learning_rate`    | `0.005`     | Learning rate (also used by the CLUB estimators)                                             |
| `--weight_decay`     | `5e-5`      | Adam weight decay                                                                            |
| `--feat_norm`        | `standard`  | `row` / `standard` / `none`                                                                  |
| `--epochs`           | `500`       | Max epochs, early stop on valid AUC with patience 100 (T-Social: 50 / 10)                    |
| `--gpu`              | `0`         | Device                                                                                       |

### CIE-GAD (`algorithms/CIEGAD/CIEGAD_main.py`)

| Argument            | Default     | Description                                                                                 |
| ------------------- | ----------- | ------------------------------------------------------------------------------------------- |
| `--dataset`         | `yelpchi`   | `yelpchi` / `amazon` / `elliptic` / `tfinance` / `tsocial` / `comp`                         |
| `--train_ratio`     | `0.4`       | Train fraction of labeled nodes (valid/test fixed 20/40)                                    |
| `--hidden`          | `32`        | Embedding size `C`                                                                          |
| `--order`           | per dataset | Jacobi order `K` (T-Finance `12`, others `6`)                                               |
| `--gamma`           | `1`         | Deep-supervision weight (`0` = w/o DS)                                                      |
| `--jacobi_a` / `--jacobi_b` | `1` / `1` | Jacobi weights (paper leaves values open; learnable by default)                       |
| `--learn_jacobi`    | `1`         | `1` = learn `a,b` (clamped `> -1`); `0` = keep them fixed                                   |
| `--no_scnaf`        | off         | Drop Jacobi filters (w/o SCNAF)                                                             |
| `--no_naf`          | off         | One scalar `α_k` per filter (w/o NAF)                                                       |
| `--no_hcre`         | off         | Classify from `Z̃` only (w/o HCRE)                                                          |
| `--thres`           | `0.5`       | Decision threshold on the fraud probability                                                 |
| `--learning_rate`   | `0.01`      | Adam learning rate (not reported in the paper)                                              |
| `--weight_decay`    | `5e-5`      | Adam weight decay                                                                           |
| `--feat_norm`       | `standard`  | `row` / `standard` / `none`                                                                 |
| `--epochs`          | `1000`      | Max epochs, early stop on valid AUC with patience 100 (T-Social: 200 / 20)                  |
| `--gpu`             | `0`         | Device                                                                                      |

### COFRAUD (`algorithms/COFRAUD/COFRAUD_main.py`)

| Argument            | Default     | Description                                                                                 |
| ------------------- | ----------- | ------------------------------------------------------------------------------------------- |
| `--dataset`         | `yelpchi`   | `yelpchi` / `amazon` / `elliptic` / `tfinance` / `tsocial` / `comp`                         |
| `--train_ratio`     | `0.4`       | Train fraction of labeled nodes (valid/test fixed 20/40)                                    |
| `--hidden`          | `32`        | Per-head hidden size (paper Table 3)                                                        |
| `--n_layers`        | `2`         | Intra/inter module layers                                                                   |
| `--n_heads`         | `8`         | Attention heads (official; T-Social default `4`)                                            |
| `--relation_dim`    | `16`        | Relation embedding size                                                                     |
| `--relation_hidden` | `16`        | Relation MLP hidden size                                                                    |
| `--dropout`         | `0.5`       | Dropout                                                                                     |
| `--residual`        | `1`         | Eq. 5 residual with trainable `λ`                                                           |
| `--no_inter`        | off         | Skip inter-relation crossing (w/o inter)                                                    |
| `--no_fusion`       | off         | Mean of `V_r h'` instead of Eqs. 9–10 (w/o fusion)                                          |
| `--thres`           | `0.5`       | Decision threshold on the fraud probability                                                 |
| `--learning_rate`   | `0.02`      | Adam learning rate (official)                                                               |
| `--weight_decay`    | `0`         | Adam weight decay                                                                           |
| `--feat_norm`       | `standard`  | `row` / `standard` / `none`                                                                 |
| `--epochs`          | `500`       | Max epochs, early stop on valid AUC with patience 50 (T-Social: 200 / 20)                   |
| `--gpu`             | `0`         | Device                                                                                      |

### PMP (`algorithms/PMP/PMP_main.py`)

Defaults are the paper's App. D.4 Table 5 values on YelpChi / Amazon / T-Finance / T-Social; on Elliptic / FDCompCN they come from `configs/tuned/PMP/<dataset>.json`.

| Argument            | Default        | Description                                                                 |
| ------------------- | -------------- | --------------------------------------------------------------------------- |
| `--dataset`         | `amazon`       | `yelpchi` / `amazon` / `tfinance` / `tsocial` / `elliptic` / `comp`         |
| `--train_ratio`     | `0.4`          | Train fraction of labeled nodes (valid/test fixed 20/40)                    |
| `--hid_dim`         | per dataset    | Hidden size d' (256 / 128 / 64 / 128 on Yelp / Amazon / T-Fin. / T-Soc.)    |
| `--num_layers`      | `1`            | PMP layers L (paper: 1 on every dataset; >1 propagates over the full graph) |
| `--dropout`         | per dataset    | Dropout before the output layer (0 / 0.6 / 0.4 / 0)                         |
| `--batch_size`      | per dataset    | Training nodes per mini-batch (512 / 128 / 256 / 2^17)                      |
| `--learning_rate`   | per dataset    | Adam learning rate (0.01; T-Social 0.001)                                   |
| `--weight_decay`    | `0`            | Adam weight decay                                                           |
| `--epochs`          | `500`          | Max epochs, early stop on valid AUC (`--early_stop`, patience 25)           |
| `--eval_batch_size` | `65536`        | Nodes per forward pass at evaluation (memory only)                          |
| `--gpu`             | `0`            | Device                                                                      |

### SEC-GFD (`algorithms/SEC_GFD/SEC_GFD_main.py`)

Defaults are the released code's values on YelpChi / Amazon / T-Finance / T-Social; on Elliptic / FDCompCN they come from `configs/tuned/SEC_GFD/<dataset>.json`.

| Argument          | Default      | Description                                                              |
| ----------------- | ------------ | ------------------------------------------------------------------------ |
| `--dataset`       | `tfinance`   | `yelpchi` / `amazon` / `tfinance` / `tsocial` / `elliptic` / `comp`      |
| `--train_ratio`   | `0.4`        | Train fraction of labeled nodes (valid/test fixed 20/40)                 |
| `--hid_dim`       | `64`         | Hidden size (T-Social `16`)                                              |
| `--order`         | `2`          | Beta-wavelet order C (T-Social `5`)                                      |
| `--high_order`    | `2`          | Number of high-pass filters `L, L^2, ...`                                |
| `--lemda`         | `0.2`        | Weight of the local environmental constraint loss                        |
| `--learning_rate` | `0.01`       | Adam learning rate                                                       |
| `--weight_decay`  | `0`          | Adam weight decay                                                        |
| `--epochs`        | `500`        | Max epochs, early stop on valid AUC (`--early_stop`, patience 25)        |
| `--gpu`           | `0`          | Device                                                                   |

### HACUD (`algorithms/HACUD/main.py`)

| Argument       | Default       | Description               |
| -------------- | ------------- | ------------------------- |
| `--dataset`    | `dblp`        | Dataset name              |
| `--epoch`      | `500`         | Epochs                    |
| `--embed_size` | `64`          | Node embedding size       |
| `--batch_size` | `64`          | Batch size                |
| `--lr`         | `0.01`        | Learning rate             |
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
| **SplitGNN**    | [SplitGNN: Spectral Graph Neural Network for Fraud Detection against Heterophily](https://dl.acm.org/doi/10.1145/3583780.3615067)                                                        | CIKM 2023  | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/splitgnn.txt)    |
| **DGA-GNN**     | [DGA-GNN: Dynamic Grouping Aggregation GNN for Fraud Detection](https://ojs.aaai.org/index.php/AAAI/article/view/29067)                                                                  | AAAI 2024  | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/dgagnn.txt)      |
| **GLHN**        | [Graph Local Homophily Network for Anomaly Detection](https://doi.org/10.1145/3627673.3679785)                                                                                           | CIKM 2024  | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/glhn.txt)        |
| **BSL**         | [Barely Supervised Learning for Graph-Based Fraud Detection](https://ojs.aaai.org/index.php/AAAI/article/view/29593)                                                                     | AAAI 2024  | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/bsl.txt)         |
| **CARE-GNN**    | [Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters](https://arxiv.org/pdf/2008.08692.pdf)                                                              | CIKM 2020  | [BibTex](https://github.com/YingtongDou/CARE-GNN#citation)                            |
| **H2-FDetector** | [H2-FDetector: A GNN-based Fraud Detector with Homophilic and Heterophilic Connections](https://dl.acm.org/doi/10.1145/3485447.3512195)                                               | WWW 2022   | [BibTex](https://github.com/shifengzhao/H2-FDetector#citation)                        |
| **FRAUDRE**     | [FRAUDRE: Fraud Detection Dual-Resistant to Graph Inconsistency and Imbalance](https://ieeexplore.ieee.org/document/9679178)                                                             | ICDM 2021  | [BibTex](algorithms/FRAUDRE/README.md#paper)                                          |
| **PC-GNN**      | [Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection](https://dl.acm.org/doi/abs/10.1145/3442381.3449989)                                                     | WWW 2021   | [BibTex](https://github.com/PonderLY/PC-GNN#citation)                                 |
| **SparseGAD**   | [Beyond Homophily: Robust Graph Anomaly Detection via Neural Sparsification](https://www.ijcai.org/proceedings/2023/234)                                                                 | IJCAI 2023 | [BibTex](algorithms/SparseGAD/README.md#paper)                                        |
| **AMNet**       | [Can Abnormality be Detected by Graph Neural Networks?](https://www.ijcai.org/proceedings/2022/270)                                                                                      | IJCAI 2022 | [BibTex](algorithms/AMNet/README.md#paper)                                            |
| **GHRN**        | [Addressing Heterophily in Graph Anomaly Detection: A Perspective of Graph Spectrum](https://dl.acm.org/doi/10.1145/3543507.3583268)                                                     | WWW 2023   | [BibTex](algorithms/GHRN/README.md#paper)                                             |
| **GGCN**        | [Two Sides of the Same Coin: Heterophily and Oversmoothing in Graph Convolutional Neural Networks](https://arxiv.org/abs/2102.06462)                                                    | ICDM 2022  | [BibTex](algorithms/GGCN/README.md#paper)                                             |
| **TAM**         | [Truncated Affinity Maximization: One-class Homophily Modeling for Graph Anomaly Detection](https://arxiv.org/abs/2306.00006)                                                           | NeurIPS 2023 | [BibTex](algorithms/TAM/README.md#paper)                                            |
| **GDN**         | [Alleviating Structural Distribution Shift in Graph Anomaly Detection](https://doi.org/10.1145/3539597.3570377)                                                                          | WSDM 2023  | [BibTex](algorithms/GDN/README.md#paper)                                              |
| **ImGSL**       | [Improving fraud detection via imbalanced graph structure learning](https://doi.org/10.1007/s10994-023-06464-0)                                                                          | Machine Learning 2024 | [BibTex](algorithms/ImGSL/README.md#paper)                                 |
| **CGNN**        | [Context-aware Graph Neural Network for Graph-based Fraud Detection with Extremely Limited Labels](https://doi.org/10.1609/aaai.v39i11.33319)                                            | AAAI 2025  | [BibTex](algorithms/CGNN/README.md#paper)                                             |
| **H2IDE**       | [Nowhere to H²IDE: Fraud Detection From Multi-Relation Graphs via Disentangled Homophily and Heterophily Identification](https://doi.org/10.1109/TKDE.2024.3523107)                 | TKDE 2025  | [BibTex](algorithms/H2IDE/README.md#paper)                                            |
| **CIE-GAD**     | [Correlation Information Enhanced Graph Anomaly Detection via Hypergraph Transformation](https://doi.org/10.1109/TCYB.2025.3558941)                                                  | TCYB 2025  | [BibTex](algorithms/CIEGAD/README.md#paper)                                           |
| **COFRAUD**     | [Don't Ignore Alienation and Marginalization: Correlating Fraud Detection](https://doi.org/10.24963/ijcai.2023/551)                                                                  | IJCAI 2023 | [BibTex](algorithms/COFRAUD/README.md#paper)                                          |
| **PMP**         | [Partitioning Message Passing for Graph Fraud Detection](https://github.com/Xtra-Computing/PMP) | ICLR 2024 | [BibTex](algorithms/PMP/README.md#paper) |
| **SEC-GFD**     | Revisiting Graph-based Fraud Detection in Sight of Heterophily and Spectrum | AAAI 2024 | [BibTex](algorithms/SEC_GFD/README.md#paper) |

### Backend notes (PyTorch Geometric)

- **Player2Vec / FdGars:** multi-layer GCN (`GCNConv`) (+ meta-path attention for Player2Vec)
- **GeniePath:** GAT breadth (`GATConv`) + LSTM depth
- **GraphSAGE / GraphConsis:** `SAGEConv` (GraphConsis adds multi-relation attention)
- **BWGNN:** beta-wavelet polynomial filters (`PolyConv`) with `torch.sparse`
- **SplitGNN:** signed multi-relation beta-wavelet filters (`PolyConv` + `RelationAware`) with `torch.sparse`
- **DGA-GNN:** decision-tree binning + grouped mean aggregation + attention (no DGL)
- **GLHN:** local-homophily prediction + `I-L` / `L` mix + InfoNCE (no DGL)
- **BSL:** GCN + NA/AA/NN disentanglement + weak/strong unlabeled augmentations (no DGL)
- **CARE-GNN:** label-aware top-p neighbour filtering (vectorised over CSR) + RL thresholds + threshold-weighted relation sum (no DGL)
- **H2-FDetector:** relation-aware edge-sign scorer + signed multi-head attention (gather / segment softmax) + prototype loss (no DGL)
- **FRAUDRE:** ego MLP + mean / difference aggregation over fixed sampled neighbours (vectorised K-hop gathers) + softmax relation weights + prior-adjusted CE (no DGL)
- **PC-GNN:** degree / class-size node picking + label-aware top-p neighbour choosing with vectorised minority over-sampling + per-relation SAGE-style aggregation (no DGL)
- **SparseGAD:** GPR GCN (gather / `index_add_`) + cosine edge attention from an extractor MLP + chunked brute-force kNN graph augmentation (no DGL)
- **AMNet:** Bernstein polynomial filters on the scaled Laplacian (shared `torch.sparse` CSR powers) + per-node filter attention + marginal loss (no DGL / PyG)
- **GHRN:** BWGNN (shared `torch.sparse` CSR Laplacian powers, union or per-relation) + graph-Laplacian edge indicator on predictions + edge pruning and retraining (no DGL)
- **GGCN:** signed cosine-similarity messages + learnable degree correction + decaying residuals, on edge lists with custom-backward CSR kernels (sampled dot products and value-weighted sparse products; no dense `N x N`, no DGL / PyG)
- **TAM:** unsupervised; two-layer GCN encoders (`torch.sparse` CSR) maximising local affinity `z_i . (A z)_i / d_i` on the original graph, trained on graphs truncated by vectorised per-node distance thresholds (edge masks + reverse-edge indices; no dense `N x N`, no DGL / PyG)
- **GDN:** per-relation mean aggregation (CSR gathers + `index_add_`) on a trainable feature table + gradient-based feature split + prototype / neighbourhood KL constraints with vectorised non-neighbour sampling (no DGL / PyG)
- **ImGSL:** degree / class-frequency sampler per relation + label-predictor edge pruning (CSR gathers) + K-head weighted-cosine node-anchor graphs with two-step anchor message passing and dynamic stopping (no dense `N x N`, no DGL / PyG)
- **CGNN:** per-neighbour normal / abnormal subspace projections + supervised contrastive loss + two-way denoising attention, aggregated over CSR gathers with `index_add_`; consistency training permutes subspace components within each gray centre's neighbourhood (segment-wise random sort; no DGL / PyG)
- **H2IDE:** factor-wise embeddings + cosine edge homophily with prototype-weighted factor mixing (hinge loss on labeled edges) + signed per-relation per-factor attention (segment softmax over CSR gathers) + CLUB pairwise MI penalty; two-hop GraphSAGE-style sampling for training and chunked exact layer-wise inference (no DGL / PyG)
- **CIE-GAD:** Jacobi spectral banks on shared CSR `A_hat` + node-level attention fusion + dual-hypergraph HCRE as `D^{-1/2}(A+D)/√2` (no dense incidence matrix) + deep supervision; full-batch (no DGL / PyG)
- **COFRAUD:** per-relation tanh frequency attention with degree-normalised residual aggregation (CSR gathers) + inter-relation crossing + relation-semantic fusion; full-batch multi-relation (no DGL / PyG)
- **PMP:** label-partitioned neighbour sums (fraud / benign / unlabelled) with root-specific weight generators and an adaptive unlabelled mix, one layer per relation + concatenation MLP; one-layer sums precomputed with `scipy.sparse`, so training is mini-batched over nodes (no DGL)
- **SEC-GFD:** Beta-wavelet band-pass + high-pass Laplacian filters over shared `torch.sparse` Laplacian powers + GCN-based local environmental constraint loss; full-batch (no DGL)
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
| **CARE-GNN**    | Opinion Fraud   | Heterogeneous | GraphSAGE + RL      |
| **H2-FDetector** | Opinion Fraud  | Heterogeneous | Signed GAT          |
| **FRAUDRE**     | Opinion Fraud   | Heterogeneous | GraphSAGE (mean + difference) |
| **PC-GNN**      | Opinion Fraud   | Heterogeneous | GraphSAGE + resampling |
| **SparseGAD**   | Graph Anomaly   | Homogeneous   | GPR-GNN + edge sparsification |
| **AMNet**       | Graph Anomaly   | Homogeneous   | Bernstein spectral GNN |
| **GHRN**        | Graph Anomaly   | Homogeneous   | Beta wavelet GNN + edge pruning |
| **GGCN**        | Heterophilous Graphs | Homogeneous | Signed GCN + degree correction |
| **TAM**         | Graph Anomaly (unsupervised) | Homogeneous | GCN + affinity maximisation + edge truncation |
| **GDN**         | Opinion Fraud   | Heterogeneous | RGCN + feature decomposition |
| **ImGSL**       | Opinion Fraud   | Heterogeneous | GCN + graph structure learning |
| **CGNN**        | Graph Fraud (few labels) | Homogeneous | Subspace-decomposed GNN + consistency training |
| **H2IDE**       | Opinion / Financial Fraud | Heterogeneous | Disentangled signed GAT + MI independence |
| **CIE-GAD**     | Graph Anomaly   | Homogeneous   | Jacobi spectral GNN + hypergraph HCRE |
| **COFRAUD**     | Opinion Fraud   | Heterogeneous | Intra/inter relation attention + fusion |
| **PMP**         | Opinion / Financial Fraud | Heterogeneous | Label-partitioned message passing |
| **SEC-GFD**     | Opinion / Financial Fraud | Homogeneous   | Hybrid spectral filters + environment loss |

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

---

cd algorithms/GraphSage

## python supervised_train.py   `--model graphsage_mean`

  --learning_rate 0.01   `--train_prefix ../../dataset/`
  --file_name YelpChi.mat   `--epochs 100`
  --dropout 0.0   `--weight_decay 0.0`
  --dim_1 128   `--batch_size 512`
  --sigmoid   `--gpu 0`
  --seed 123   `--samples_1 25`
  --samples_2 10   `--save_dir results`
  --test_size 0.2

## cd algorithms/FdGars

python FdGars_main.py --dataset_str yelpchi --epoch_num 100 --batch_size 512 --gpu 0 --seed 123

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
  --seed 123 `
  --gpu 0
```

Outputs: `algorithms/GraphConsis/results/GraphConsis/yelpchi/seed123_train40/` and `.../amazon/seed123_train40/`. Amazon’s `net_usu` is dense, so the first epoch can take a while to load.

---

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
  --seed 123 `
  --gpu 0

python supervised_train.py --train_prefix ../../dataset/ --file_name tsocial --sigmoid --epochs 20 --batch_size 256 --gpu 0
```

**BWGNN** (paper settings: 40% train)

```powershell
cd algorithms/BWGNN
python BWGNN_main.py --dataset tfinance --train_ratio 0.4 --hid_dim 64 --order 2 --epochs 100 --gpu 0
python BWGNN_main.py --dataset tsocial --train_ratio 0.4 --epochs 100 --gpu 0
```

T-Social is multi-GB; the paper setting is `h = 64`, `C = 5` (pass `--hid_dim 10` if GPU memory is tight). BWGNN also takes `--dataset amazon` or `yelpchi`.

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

---

FDCompCN (`dataset/comp.dgl`) is now a first-class dataset for GraphConsis and BWGNN, and **SplitGNN** is in the toolbox as a DGL-free port of `reference/SplitGNN-master`.

**What** `comp.dgl` **contains:** 5,317 nodes, 57-d features, 559 fraud labels, four directed relations (`homo` 10,059 / `cic` 8,505 / `csc` 5,944 / `cpc` 6,244), plus the official train/valid/test masks.

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

SplitGNN defaults follow the paper table for FDCompCN architecture (`gamma=0.8`, `C=3`, `K=2`, `intra_dim=8`, early stop 100). Learning rate defaults to Adam `0.01`. Pass `--ignore_file_masks` to use a fresh 40/20/40 split instead of the masks in `comp.dgl`. Amazon still drops feature column 19 and trains only on labeled nodes (`index >= 3305`).

## A 2-epoch smoke run on `comp` completed (forward, backward, and the train loop). Still no DGL import — the `.dgl` file is read by `utils/dgl_bin.py`.

GraphConsis, BWGNN, and SplitGNN now share one stratified split of **labeled** nodes:

| Split     | Fraction                                   | Role                       |
| --------- | ------------------------------------------ | -------------------------- |
| **Train** | `--train_ratio` (default **40%**, max 40%) | Fit the model              |
| **Valid** | **20%** (fixed)                            | Pick the checkpoint / tune |
| **Test**  | **40%** (fixed)                            | Final metrics only         |

If you set `--train_ratio 0.2`, valid and test stay 20/40. The unused half of the 40% train pool is **discarded**, not moved into valid or test. Same seed → same valid and test nodes.

After training, each model reloads the **best valid-AUC** checkpoint, then reports train / valid / test. `--early_stop` defaults to **25** (stop if valid AUC does not improve for 25 epochs). Pass `--early_stop 0` to train all epochs.

```powershell
# default 40 / 20 / 40
python supervised_train.py --train_prefix ../../dataset/ --file_name comp --sigmoid --epochs 50 --gpu 0

# 20% train; valid/test unchanged
python BWGNN_main.py --dataset elliptic --train_ratio 0.2 --gpu 0
python SplitGNN_main.py --dataset comp --train_ratio 0.4 --gpu 0
```

## `comp.dgl` file masks are no longer used unless you pass `--use_file_masks` on SplitGNN. Amazon still splits only labeled nodes (`index >= 3305`); the 40/20/40 ratios apply to that labeled set.

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

Hyperparameters default to the official yaml per dataset (`n_hidden`, heads, bins `k`, grouping threshold `z`) with the paper's optimiser settings (Adam lr `1e-3`, weight decay `1e-3`, up to 1000 epochs) on its five datasets; FDCompCN is tuned (see [Hyperparameter Policy](#hyperparameter-policy)). `--train_ratio` still only shrinks the 40% train pool. `--no_bin_encoding` skips DT binning. T-Social uses a smaller hidden size than the paper yaml because that graph is large.

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

---

Early stopping is now **on by default** for GraphConsis and BWGNN.

`--early_stop` defaults to **25**: training stops if valid AUC does not improve for 25 epochs. The best valid-AUC checkpoint is still restored before test metrics.

Pass `--early_stop 0` if you want to run every epoch. SplitGNN / DGA-GNN / GLHN / BSL keep their own per-dataset patience values.

---

---

Fill these in once, then paste them into every command. **None of these six scripts expose an optimizer flag** — they all use **Adam**. You only set seed, learning rate, and train ratio.

```bash
SEED=        # e.g. 123
LR=          # e.g. 0.01
TRAIN_RATIO= # e.g. 0.4   (max 0.4; valid=0.20 and test=0.40 stay fixed)
```

Validation is already inside each run: train on train, checkpoint on **valid AUC** (`--early_stop` defaults on), then print and save **test** metrics under `results/` (or `--save_dir`). You do not run a separate validate job.

Work from the **repo root** with the venv activated and datasets already in `dataset/` (`YelpChi.mat`, `Amazon.mat`, `tfinance`, `tsocial`, `elliptic` folder, `comp.dgl`).

---

### GraphConsis

Uses `--file_name`, not `--dataset`. Keep `--sigmoid`. Default epochs in the script is `10` (short). Other knobs stay at script defaults (`dim_1=128`, `batch_size=512`, `samples_1=25`, `samples_2=10`).

```bash
cd algorithms/GraphConsis

uv run python supervised_train.py --train_prefix ../../dataset/ --file_name YelpChi.mat --sigmoid --gpu 0 --seed 123 --learning_rate 0.01 --train_ratio 0.4 --epochs 500
uv run python supervised_train.py --train_prefix ../../dataset/ --file_name YelpChi.mat --sigmoid --gpu 0 --seed 1234 --learning_rate 0.01 --train_ratio 0.4 --epochs 500 --batch_size 100000
uv run python supervised_train.py --train_prefix ../../dataset/ --file_name YelpChi.mat --sigmoid --gpu 0 --seed 12345 --learning_rate 0.01 --train_ratio 0.4 --epochs 500 --batch_size 100000
uv run python supervised_train.py --train_prefix ../../dataset/ --file_name YelpChi.mat --sigmoid --gpu 0 --seed 123456 --learning_rate 0.01 --train_ratio 0.4 --epochs 500 --batch_size 100000
uv run python supervised_train.py --train_prefix ../../dataset/ --file_name YelpChi.mat --sigmoid --gpu 0 --seed 1234567 --learning_rate 0.01 --train_ratio 0.4 --epochs 500 --batch_size 100000
uv run python supervised_train.py --train_prefix ../../dataset/ --file_name Amazon.mat --sigmoid --gpu 0 --seed 123 --learning_rate 0.01 --train_ratio 0.4 --epochs 500
python supervised_train.py --train_prefix ../../dataset/ --file_name elliptic --sigmoid --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python supervised_train.py --train_prefix ../../dataset/ --file_name tfinance --sigmoid --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python supervised_train.py --train_prefix ../../dataset/ --file_name tsocial --sigmoid --gpu 0 --seed 123 --learning_rate 0.01 --train_ratio 0.4 --epochs 500 --batch_size 100000
python supervised_train.py --train_prefix ../../dataset/ --file_name comp --sigmoid --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO

cd ../..
```

On T-Social, add `--batch_size 256` if you hit memory issues (that is the usual toolbox default for that graph).

---

### BWGNN

Dataset defaults: `hid_dim=64`, `order=2`, `epochs=100`. T-Social is the exception — use `hid_dim=10` and `order=5` or a 5.8M-node run with 64/2 will not fit.

```bash
cd algorithms/BWGNN

uv run python BWGNN_main.py --dataset yelpchi --gpu 0 --seed 123 --learning_rate 0.01 --train_ratio 0.4 --epochs 500
uv run python BWGNN_main.py --dataset yelpchi --gpu 0 --seed 1234 --learning_rate 0.01 --train_ratio 0.4 --epochs 500
uv run python BWGNN_main.py --dataset yelpchi --gpu 0 --seed 12345 --learning_rate 0.01 --train_ratio 0.4 --epochs 500
uv run python BWGNN_main.py --dataset yelpchi --gpu 0 --seed 123456 --learning_rate 0.01 --train_ratio 0.4 --epochs 500
uv run python BWGNN_main.py --dataset yelpchi --gpu 0 --seed 1234567 --learning_rate 0.01 --train_ratio 0.4 --epochs 500
uv run python BWGNN_main.py --dataset amazon --gpu 0 --seed 1234567 --learning_rate 0.01 --train_ratio 0.4 --epochs 500
uv run python BWGNN_main.py --dataset amazon --gpu 0 --seed 123456 --learning_rate 0.01 --train_ratio 0.4 --epochs 500
uv run python BWGNN_main.py --dataset amazon --gpu 0 --seed 12345 --learning_rate 0.01 --train_ratio 0.4 --epochs 500
uv run python BWGNN_main.py --dataset amazon --gpu 0 --seed 1234 --learning_rate 0.01 --train_ratio 0.4 --epochs 500
uv run python BWGNN_main.py --dataset amazon --gpu 0 --seed 123 --learning_rate 0.01 --train_ratio 0.4 --epochs 500
uv run python BWGNN_main.py --dataset elliptic --gpu 0 --seed 1234567 --learning_rate 0.01 --train_ratio 0.4 --epochs 500
uv run python BWGNN_main.py --dataset elliptic --gpu 0 --seed 123456 --learning_rate 0.01 --train_ratio 0.4 --epochs 500
uv run python BWGNN_main.py --dataset elliptic --gpu 0 --seed 12345 --learning_rate 0.01 --train_ratio 0.4 --epochs 500
uv run python BWGNN_main.py --dataset elliptic --gpu 0 --seed 1234 --learning_rate 0.01 --train_ratio 0.4 --epochs 500
uv run python BWGNN_main.py --dataset elliptic --gpu 0 --seed 123 --learning_rate 0.01 --train_ratio 0.4 --epochs 500
uv run python BWGNN_main.py --dataset tfinance --gpu 0 --seed 123 --learning_rate 0.01 --train_ratio 0.4 --epochs 500
uv run python BWGNN_main.py --dataset tfinance --gpu 0 --seed 1234 --learning_rate 0.01 --train_ratio 0.4 --epochs 500
uv run python BWGNN_main.py --dataset tfinance --gpu 0 --seed 12345 --learning_rate 0.01 --train_ratio 0.4 --epochs 500
uv run python BWGNN_main.py --dataset tfinance --gpu 0 --seed 123456 --learning_rate 0.01 --train_ratio 0.4 --epochs 500
uv run python BWGNN_main.py --dataset tfinance --gpu 0 --seed 1234567 --learning_rate 0.01 --train_ratio 0.4 --epochs 500
python BWGNN_main.py --dataset tsocial --hid_dim 10 --order 5 --gpu 0 --seed 1234567 --learning_rate 0.01 --train_ratio 0.4 --epochs 500 --batch_size 10000
uv run python BWGNN_main.py --dataset comp --gpu 0 --seed 1234567 --learning_rate 0.01 --train_ratio 0.4 --epochs 500
uv run python BWGNN_main.py --dataset comp --gpu 0 --seed 123456 --learning_rate 0.01 --train_ratio 0.4 --epochs 500
uv run python BWGNN_main.py --dataset comp --gpu 0 --seed 12345 --learning_rate 0.01 --train_ratio 0.4 --epochs 500
uv run python BWGNN_main.py --dataset comp --gpu 0 --seed 1234 --learning_rate 0.01 --train_ratio 0.4 --epochs 500
uv run python BWGNN_main.py --dataset comp --gpu 0 --seed 123 --learning_rate 0.01 --train_ratio 0.4 --epochs 500

cd ../..
```

---

### SplitGNN

Default learning rate is Adam `0.01` on every dataset. Other defaults (`C`, `K`, `gamma`, `intra_dim`, `epochs`, `early_stop`) stay dataset-specific.

```bash
cd algorithms/SplitGNN

uv run python SplitGNN_main.py --dataset yelpchi --gpu 0 --seed 123 --learning_rate 0.01 --train_ratio 0.2 --epochs 500
uv run python SplitGNN_main.py --dataset yelpchi --gpu 0 --seed 1234 --learning_rate 0.01 --train_ratio 0.2 --epochs 500
uv run python SplitGNN_main.py --dataset yelpchi --gpu 0 --seed 12345 --learning_rate 0.01 --train_ratio 0.2 --epochs 500
uv run python SplitGNN_main.py --dataset yelpchi --gpu 0 --seed 123456 --learning_rate 0.01 --train_ratio 0.2 --epochs 500
uv run python SplitGNN_main.py --dataset yelpchi --gpu 0 --seed 1234567 --learning_rate 0.01 --train_ratio 0.2 --epochs 500
uv run python SplitGNN_main.py --dataset amazon --gpu 0 --gpu 0 --seed 123 --learning_rate 0.1 --train_ratio 0.2 --epochs 500
uv run python SplitGNN_main.py --dataset amazon --gpu 0 --seed 1234 --learning_rate 0.1 --train_ratio 0.2 --epochs 500
uv run python SplitGNN_main.py --dataset amazon --gpu 0 --seed 12345 --learning_rate 0.1 --train_ratio 0.2 --epochs 500
uv run python SplitGNN_main.py --dataset amazon --gpu 0 --seed 123456 --learning_rate 0.1 --train_ratio 0.2 --epochs 500
uv run python SplitGNN_main.py --dataset amazon --gpu 0 --seed 1234567 --learning_rate 0.1 --train_ratio 0.2 --epochs 500
uv run python SplitGNN_main.py --dataset elliptic --gpu 0 --seed 1234567 --learning_rate 0.01 --train_ratio 0.2 --epochs 500
uv run python SplitGNN_main.py --dataset elliptic --gpu 0 --seed 123456 --learning_rate 0.01 --train_ratio 0.2 --epochs 500
uv run python SplitGNN_main.py --dataset elliptic --gpu 0 --seed 12345 --learning_rate 0.01 --train_ratio 0.2 --epochs 500
uv run python SplitGNN_main.py --dataset elliptic --gpu 0 --seed 1234 --learning_rate 0.01 --train_ratio 0.2 --epochs 500
uv run python SplitGNN_main.py --dataset elliptic --gpu 0 --seed 123 --learning_rate 0.01 --train_ratio 0.2 --epochs 500
uv run python SplitGNN_main.py --dataset tfinance --gpu 0 --seed 1234567 --learning_rate 0.01 --train_ratio 0.2 --epochs 500
uv run python SplitGNN_main.py --dataset tfinance --gpu 0 --seed 123456 --learning_rate 0.01 --train_ratio 0.2 --epochs 500
uv run python SplitGNN_main.py --dataset tfinance --gpu 0 --seed 12345 --learning_rate 0.01 --train_ratio 0.2 --epochs 500
uv run python SplitGNN_main.py --dataset tfinance --gpu 0 --seed 1234 --learning_rate 0.01 --train_ratio 0.2 --epochs 500
uv run python SplitGNN_main.py --dataset tfinance --gpu 0 --seed 123 --learning_rate 0.01 --train_ratio 0.2 --epochs 500
uv run python SplitGNN_main.py --dataset tsocial --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
uv run python SplitGNN_main.py --dataset comp --gpu 0 --seed 1234567 --learning_rate 0.001 --train_ratio 0.2 --epochs 500
uv run python SplitGNN_main.py --dataset comp --gpu 0 --seed 123456 --learning_rate 0.001 --train_ratio 0.2 --epochs 500
uv run python SplitGNN_main.py --dataset comp --gpu 0 --seed 12345 --learning_rate 0.001 --train_ratio 0.2 --epochs 500
uv run python SplitGNN_main.py --dataset comp --gpu 0 --seed 1234 --learning_rate 0.001 --train_ratio 0.2 --epochs 500
uv run python SplitGNN_main.py --dataset comp --gpu 0 --seed 123 --learning_rate 0.001 --train_ratio 0.2 --epochs 500

cd ../..
```

---

### DGA-GNN

Hidden size, heads, bins `k`, `z`, dropout, epochs, and early-stop come from the official yaml per dataset. Script default LR is Adam `0.01` if you omit `--learning_rate`.

```bash
cd algorithms/DGAGNN

python DGAGNN_main.py --dataset yelpchi --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python DGAGNN_main.py --dataset amazon --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python DGAGNN_main.py --dataset elliptic --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python DGAGNN_main.py --dataset tfinance --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python DGAGNN_main.py --dataset tsocial --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python DGAGNN_main.py --dataset comp --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO

cd ../..
```

---

### GLHN

Per-dataset defaults: hidden 32 (10 on T-Social), 2 layers, `τ=0.1`, hetero on Yelp/Amazon/comp, homo on elliptic/tfinance/tsocial.

```bash
cd algorithms/GLHN

python GLHN_main.py --dataset yelpchi --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python GLHN_main.py --dataset amazon --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python GLHN_main.py --dataset elliptic --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python GLHN_main.py --dataset tfinance --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python GLHN_main.py --dataset tsocial --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python GLHN_main.py --dataset comp --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO

cd ../..
```

---

### BSL

Toolbox 40/20/40 protocol (not `--paper_split`). Hidden 96 (48 on T-Social), `α=0.4`, `β=0.8`.

```bash
cd algorithms/BSL

python BSL_main.py --dataset yelpchi --gpu 0 --seed 123 --learning_rate 0.01 --train_ratio 40
python BSL_main.py --dataset amazon --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python BSL_main.py --dataset elliptic --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python BSL_main.py --dataset tfinance --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python BSL_main.py --dataset tsocial --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python BSL_main.py --dataset comp --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO

cd ../..
```

---

### CARE-GNN

Paper defaults: embedding 64, `lambda_1=2`, weight decay `1e-3`, RL step `0.02`, batch 1024 (256 on Amazon / comp).

```bash
cd algorithms/CAREGNN

python CAREGNN_main.py --dataset yelpchi --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python CAREGNN_main.py --dataset amazon --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python CAREGNN_main.py --dataset elliptic --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python CAREGNN_main.py --dataset tfinance --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python CAREGNN_main.py --dataset tsocial --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python CAREGNN_main.py --dataset comp --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO

cd ../..
```

---

### H2-FDetector

Paper defaults: full-batch, `lr=0.1` (set `LR=0.1` or drop the flag), weight decay `5e-5`, `intra_dim=8`, 4 heads, 2 layers; loss weights per dataset (`gamma1` / `gamma2`).

```bash
cd algorithms/H2FDetector

python H2FDetector_main.py --dataset yelpchi --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python H2FDetector_main.py --dataset amazon --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python H2FDetector_main.py --dataset elliptic --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python H2FDetector_main.py --dataset tfinance --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python H2FDetector_main.py --dataset tsocial --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python H2FDetector_main.py --dataset comp --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO

cd ../..
```

---

### FRAUDRE

Reference defaults: embedding 64, weight decay `1e-4`, 2 convolution layers, 10 sampled neighbours. Learning rate differs per dataset (0.001 YelpChi, 0.1 Amazon, 0.01 others); drop `--learning_rate` to use these.

```bash
cd algorithms/FRAUDRE

python FRAUDRE_main.py --dataset yelpchi --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python FRAUDRE_main.py --dataset amazon --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python FRAUDRE_main.py --dataset elliptic --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python FRAUDRE_main.py --dataset tfinance --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python FRAUDRE_main.py --dataset tsocial --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python FRAUDRE_main.py --dataset comp --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO

cd ../..
```

---

### PC-GNN

Reference defaults: embedding 64, `rho=0.5`, `alpha=2`, kept-neighbour fraction 0.5. Learning rate / weight decay / decision threshold differ per dataset (0.01 / 1e-3 / 0.5 on YelpChi, 0.005 / 5e-4 / 0.4 on Amazon); drop `--learning_rate` to use these.

```bash
cd algorithms/PCGNN

python PCGNN_main.py --dataset yelpchi --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python PCGNN_main.py --dataset amazon --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python PCGNN_main.py --dataset elliptic --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python PCGNN_main.py --dataset tfinance --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python PCGNN_main.py --dataset tsocial --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python PCGNN_main.py --dataset comp --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO

cd ../..
```

---

### SparseGAD

Reference defaults: full-batch, lr `0.01`, weight decay `5e-4`, 2 layers, kNN `k=10` added every 200 epochs; hidden size and dropouts per dataset.

```bash
cd algorithms/SparseGAD

python SparseGAD_main.py --dataset yelpchi --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python SparseGAD_main.py --dataset amazon --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python SparseGAD_main.py --dataset elliptic --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python SparseGAD_main.py --dataset tfinance --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python SparseGAD_main.py --dataset tsocial --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python SparseGAD_main.py --dataset comp --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO

cd ../..
```

---

### AMNet

Reference defaults: full-batch, 2 Bernstein filters with filter lr `5e-2`, up to 2000 epochs with patience 200; order, hidden size, lr and marginal-loss weight per dataset. `--learning_rate` is omitted so the per-dataset rates apply.

```bash
cd algorithms/AMNet

python AMNet_main.py --dataset yelpchi --gpu 0 --seed $SEED --train_ratio $TRAIN_RATIO
python AMNet_main.py --dataset amazon --gpu 0 --seed $SEED --train_ratio $TRAIN_RATIO
python AMNet_main.py --dataset elliptic --gpu 0 --seed $SEED --train_ratio $TRAIN_RATIO
python AMNet_main.py --dataset tfinance --gpu 0 --seed $SEED --train_ratio $TRAIN_RATIO
python AMNet_main.py --dataset tsocial --gpu 0 --seed $SEED --train_ratio $TRAIN_RATIO
python AMNet_main.py --dataset comp --gpu 0 --seed $SEED --train_ratio $TRAIN_RATIO

cd ../..
```

---

### GHRN

Reference defaults: BWGNN hidden 64 / order 2 (T-Social 10 / 5), lr `0.01`, 100 epochs per stage, class-weighted CE; deleting ratio 0.1 on YelpChi and 0.015 elsewhere. Each run trains BWGNN, prunes, and retrains.

```bash
cd algorithms/GHRN

python GHRN_main.py --dataset yelpchi --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python GHRN_main.py --dataset amazon --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python GHRN_main.py --dataset elliptic --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python GHRN_main.py --dataset tfinance --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python GHRN_main.py --dataset tsocial --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python GHRN_main.py --dataset comp --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO

cd ../..
```

---

### GGCN

Reference defaults: full-batch, lr `0.01`, 2 layers, hidden 32, dropout 0.5, weight decay `4e-3`, decaying residuals (`decay_rate 1.0`, `exponent 3.0`), up to 1500 epochs with patience 100.

```bash
cd algorithms/GGCN

python GGCN_main.py --dataset yelpchi --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python GGCN_main.py --dataset amazon --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python GGCN_main.py --dataset elliptic --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python GGCN_main.py --dataset tfinance --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python GGCN_main.py --dataset tsocial --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python GGCN_main.py --dataset comp --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO

cd ../..
```

---

### TAM

Reference defaults: unsupervised, embedding 128, lr `1e-5` (kept, so `$LR` is not passed), 500 epochs per LAMNet. Union graphs use K=7 truncations, T=1 and the min-max loss; single relations use K=4 and T=3. The seed changes initialisation and truncation thresholds; `$TRAIN_RATIO` only affects the decision threshold.

```bash
cd algorithms/TAM

python TAM_main.py --dataset yelpchi --gpu 0 --seed $SEED --train_ratio $TRAIN_RATIO
python TAM_main.py --dataset amazon --gpu 0 --seed $SEED --train_ratio $TRAIN_RATIO
python TAM_main.py --dataset elliptic --gpu 0 --seed $SEED --train_ratio $TRAIN_RATIO
python TAM_main.py --dataset tfinance --gpu 0 --seed $SEED --train_ratio $TRAIN_RATIO
python TAM_main.py --dataset tsocial --gpu 0 --seed $SEED --train_ratio $TRAIN_RATIO
python TAM_main.py --dataset comp --gpu 0 --seed $SEED --train_ratio $TRAIN_RATIO

cd ../..
```

---

### GDN

Reference defaults: emb 64, batch 1024, Adam lr `1e-3` / L2 `1e-3` (network) and lr `1e-5` / L2 `0.1` (feature table), Beta 0.1, topk 10 (3 on 10-feature datasets), threshold 0.2; up to 100 epochs with patience 20.

```bash
cd algorithms/GDN

python GDN_main.py --dataset yelpchi --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python GDN_main.py --dataset amazon --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python GDN_main.py --dataset elliptic --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python GDN_main.py --dataset tfinance --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python GDN_main.py --dataset tsocial --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python GDN_main.py --dataset comp --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO

cd ../..
```

---

### ImGSL

Paper defaults: lr `0.01`, weight decay `1e-3`, `beta 0.5`, `tau- 0.2`, `alpha 2`, 512 anchors; batch 256 / `epsilon 0.5` on Amazon, 1024 / `0` elsewhere; up to 150 epochs with patience 30. Label-dependent Eq. 9 edits off (see `algorithms/ImGSL/README.md`).

```bash
cd algorithms/ImGSL

python ImGSL_main.py --dataset yelpchi --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python ImGSL_main.py --dataset amazon --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python ImGSL_main.py --dataset elliptic --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python ImGSL_main.py --dataset tfinance --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python ImGSL_main.py --dataset tsocial --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python ImGSL_main.py --dataset comp --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO

cd ../..
```

---

### CGNN

Paper defaults: lr `0.01`, weight decay `1e-4`, embedding 64, `lambda = beta = 1`; up to 100 epochs with patience 25. The toolbox split uses batch 256 and `tau 0.5` (T-Finance `0.8`); `--n_black 1` / `5` switches to the paper's few-label split with batch 16 and the paper temperatures (see `algorithms/CGNN/README.md`).

```bash
cd algorithms/CGNN

python CGNN_main.py --dataset yelpchi --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python CGNN_main.py --dataset amazon --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python CGNN_main.py --dataset elliptic --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python CGNN_main.py --dataset tfinance --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python CGNN_main.py --dataset tsocial --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python CGNN_main.py --dataset comp --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python CGNN_main.py --dataset amazon --gpu 0 --seed $SEED --learning_rate $LR --n_black 5   # paper few-label split

cd ../..
```

---

### H2IDE

Paper defaults: lr `0.005`, weight decay `5e-5`, one layer, factor dimension 8, dropout 0.1, `alpha3 1e-4`; YelpChi `K 12`, `alpha1 1.2`, `alpha2 0.001`, batch 256; Amazon `K 8`, `alpha1 0.8`, `alpha2 0.01`, batch 1024; others `K 8`, `alpha1 1.2`, `alpha2 0.01`, batch 1024; up to 500 epochs with patience 100 (see `algorithms/H2IDE/README.md`).

```bash
cd algorithms/H2IDE

python H2IDE_main.py --dataset yelpchi --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python H2IDE_main.py --dataset amazon --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python H2IDE_main.py --dataset elliptic --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python H2IDE_main.py --dataset tfinance --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python H2IDE_main.py --dataset tsocial --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO
python H2IDE_main.py --dataset comp --gpu 0 --seed $SEED --learning_rate $LR --train_ratio $TRAIN_RATIO

cd ../..
```

---

### CIE-GAD

Default learning rate is Adam `0.01` on every dataset. Other defaults (embedding `32`, `γ = 1`, Jacobi order `K = 12` on T-Finance / `K = 6` elsewhere, weight decay `5e-5`, epochs, early stop) stay dataset-specific (see `algorithms/CIEGAD/README.md`).

```bash
cd algorithms/CIEGAD

uv run python CIEGAD_main.py --dataset yelpchi --gpu 0 --seed 123 --train_ratio 0.1 --epochs 500
uv run python CIEGAD_main.py --dataset yelpchi --gpu 0 --seed 1234 --train_ratio 0.1 --epochs 500
uv run python CIEGAD_main.py --dataset yelpchi --gpu 0 --seed 12345 --train_ratio 0.1 --epochs 500
uv run python CIEGAD_main.py --dataset yelpchi --gpu 0 --seed 123456 --train_ratio 0.1 --epochs 500
uv run python CIEGAD_main.py --dataset yelpchi --gpu 0 --seed 1234567 --train_ratio 0.1 --epochs 500
uv run python CIEGAD_main.py --dataset amazon --gpu 0 --seed 123 --train_ratio 0.1 --epochs 500
uv run python CIEGAD_main.py --dataset amazon --gpu 0 --seed 1234 --train_ratio 0.1 --epochs 500
uv run python CIEGAD_main.py --dataset amazon --gpu 0 --seed 12345 --train_ratio 0.1 --epochs 500
uv run python CIEGAD_main.py --dataset amazon --gpu 0 --seed 123456 --train_ratio 0.1 --epochs 500
uv run python CIEGAD_main.py --dataset amazon --gpu 0 --seed 1234567 --train_ratio 0.1 --epochs 500
uv run python CIEGAD_main.py --dataset elliptic --gpu 0 --seed 123 --train_ratio 0.1 --epochs 500
uv run python CIEGAD_main.py --dataset elliptic --gpu 0 --seed 1234 --train_ratio 0.1 --epochs 500
uv run python CIEGAD_main.py --dataset elliptic --gpu 0 --seed 12345 --train_ratio 0.1 --epochs 500
uv run python CIEGAD_main.py --dataset elliptic --gpu 0 --seed 123456 --train_ratio 0.1 --epochs 500
uv run python CIEGAD_main.py --dataset elliptic --gpu 0 --seed 1234567 --train_ratio 0.1 --epochs 500
uv run python CIEGAD_main.py --dataset tfinance --gpu 0 --seed 123 --train_ratio 0.1 --epochs 500
uv run python CIEGAD_main.py --dataset tfinance --gpu 0 --seed 1234 --train_ratio 0.1 --epochs 500
uv run python CIEGAD_main.py --dataset tfinance --gpu 0 --seed 12345 --train_ratio 0.1 --epochs 500
uv run python CIEGAD_main.py --dataset tfinance --gpu 0 --seed 123456 --train_ratio 0.1 --epochs 500
uv run python CIEGAD_main.py --dataset tfinance --gpu 0 --seed 1234567 --train_ratio 0.1 --epochs 500
uv run python CIEGAD_main.py --dataset tsocial --gpu 0 --seed 123 --learning_rate 0.01 --train_ratio 0.4 --epochs 200
uv run python CIEGAD_main.py --dataset tsocial --gpu 0 --seed 1234 --learning_rate 0.01 --train_ratio 0.4 --epochs 200
uv run python CIEGAD_main.py --dataset tsocial --gpu 0 --seed 12345 --learning_rate 0.01 --train_ratio 0.4 --epochs 200
uv run python CIEGAD_main.py --dataset tsocial --gpu 0 --seed 123456 --learning_rate 0.01 --train_ratio 0.4 --epochs 200
uv run python CIEGAD_main.py --dataset tsocial --gpu 0 --seed 1234567 --learning_rate 0.01 --train_ratio 0.4 --epochs 200
uv run python CIEGAD_main.py --dataset comp --gpu 0 --seed 123 --train_ratio 0.1 --epochs 500
uv run python CIEGAD_main.py --dataset comp --gpu 0 --seed 1234 --train_ratio 0.1 --epochs 500
uv run python CIEGAD_main.py --dataset comp --gpu 0 --seed 12345 --train_ratio 0.1 --epochs 500
uv run python CIEGAD_main.py --dataset comp --gpu 0 --seed 123456 --train_ratio 0.1 --epochs 500
uv run python CIEGAD_main.py --dataset comp --gpu 0 --seed 1234567 --train_ratio 0.1 --epochs 500

cd ../..
```

---

### COFRAUD

Default learning rate is Adam `0.02` on every dataset. Other defaults (2 layers, hidden 32, 8 heads, relation hidden 16, dropout 0.5, weight decay `0`, epochs, early stop) stay dataset-specific; T-Social uses 4 heads / 200 epochs (see `algorithms/COFRAUD/README.md`).

```bash
cd algorithms/COFRAUD

uv run python COFRAUD_main.py --dataset yelpchi --gpu 0 --seed 123 --train_ratio 0.1 --epochs 500
uv run python COFRAUD_main.py --dataset yelpchi --gpu 0 --seed 1234 --train_ratio 0.1 --epochs 500
uv run python COFRAUD_main.py --dataset yelpchi --gpu 0 --seed 12345 --train_ratio 0.1 --epochs 500
uv run python COFRAUD_main.py --dataset yelpchi --gpu 0 --seed 123456 --train_ratio 0.1 --epochs 500
uv run python COFRAUD_main.py --dataset yelpchi --gpu 0 --seed 1234567 --train_ratio 0.1 --epochs 500
uv run python COFRAUD_main.py --dataset amazon --gpu 0 --seed 123 --train_ratio 0.1 --epochs 500
uv run python COFRAUD_main.py --dataset amazon --gpu 0 --seed 1234 --train_ratio 0.1 --epochs 500
uv run python COFRAUD_main.py --dataset amazon --gpu 0 --seed 12345 --train_ratio 0.1 --epochs 500
uv run python COFRAUD_main.py --dataset amazon --gpu 0 --seed 123456 --train_ratio 0.1 --epochs 500
uv run python COFRAUD_main.py --dataset amazon --gpu 0 --seed 1234567 --train_ratio 0.1 --epochs 500
uv run python COFRAUD_main.py --dataset elliptic --gpu 0 --seed 123 --train_ratio 0.1 --epochs 500
uv run python COFRAUD_main.py --dataset elliptic --gpu 0 --seed 1234 --train_ratio 0.1 --epochs 500
uv run python COFRAUD_main.py --dataset elliptic --gpu 0 --seed 12345 --train_ratio 0.1 --epochs 500
uv run python COFRAUD_main.py --dataset elliptic --gpu 0 --seed 123456 --train_ratio 0.1 --epochs 500
uv run python COFRAUD_main.py --dataset elliptic --gpu 0 --seed 1234567 --train_ratio 0.1 --epochs 500
uv run python COFRAUD_main.py --dataset tfinance --gpu 0 --seed 123 --train_ratio 0.4 --epochs 500
uv run python COFRAUD_main.py --dataset tfinance --gpu 0 --seed 1234 --learning_rate 0.02 --train_ratio 0.4 --epochs 500
uv run python COFRAUD_main.py --dataset tfinance --gpu 0 --seed 12345 --learning_rate 0.02 --train_ratio 0.4 --epochs 500
uv run python COFRAUD_main.py --dataset tfinance --gpu 0 --seed 123456 --learning_rate 0.02 --train_ratio 0.4 --epochs 500
uv run python COFRAUD_main.py --dataset tfinance --gpu 0 --seed 1234567 --learning_rate 0.02 --train_ratio 0.4 --epochs 500
uv run python COFRAUD_main.py --dataset tsocial --gpu 0 --seed 123 --learning_rate 0.02 --train_ratio 0.4 --epochs 200
uv run python COFRAUD_main.py --dataset tsocial --gpu 0 --seed 1234 --learning_rate 0.02 --train_ratio 0.4 --epochs 200
uv run python COFRAUD_main.py --dataset tsocial --gpu 0 --seed 12345 --learning_rate 0.02 --train_ratio 0.4 --epochs 200
uv run python COFRAUD_main.py --dataset tsocial --gpu 0 --seed 123456 --learning_rate 0.02 --train_ratio 0.4 --epochs 200
uv run python COFRAUD_main.py --dataset tsocial --gpu 0 --seed 1234567 --learning_rate 0.02 --train_ratio 0.4 --epochs 200
uv run python COFRAUD_main.py --dataset comp --gpu 0 --seed 123 --train_ratio 0.1 --epochs 500
uv run python COFRAUD_main.py --dataset comp --gpu 0 --seed 1234 --train_ratio 0.1 --epochs 500
uv run python COFRAUD_main.py --dataset comp --gpu 0 --seed 12345 --train_ratio 0.1 --epochs 500
uv run python COFRAUD_main.py --dataset comp --gpu 0 --seed 123456 --train_ratio 0.1 --epochs 500
uv run python COFRAUD_main.py --dataset comp --gpu 0 --seed 1234567 --train_ratio 0.1 --epochs 500

cd ../..
```

---

### How to read validation

Each command prints train / valid / test metrics. Artifacts:

`algorithms/<MODEL>/results/<MODEL>/<dataset>/seed<SEED>_train<pct>/`

- `metrics.json` — test metrics plus `valid_auc`
- `checkpoint.pt` — best valid-AUC weights
- predictions / split indices

On PowerShell, either `set SEED=123` and use `%SEED%`, or type the numbers in place of `$SEED`, `$LR`, `$TRAIN_RATIO`. Run T-Social last and one model at a time; it is the only graph that commonly OOMs.