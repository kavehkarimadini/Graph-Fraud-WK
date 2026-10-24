# Run Graph-Fraud-WK on RunPod (COFRAUD)

End-to-end: SSH into a GPU pod → download the project zip from Google Drive with `gdown` → unzip → `uv sync` → train **COFRAUD**.

## 1. Prepare the Google Drive zip

1. Zip the repo on your machine (include `dataset/` with `.mat` / `.dgl` files you need).
2. Upload the zip to Google Drive.
3. Share the file: **Anyone with the link** → Viewer.
4. Copy the link. It looks like:
   ```text
   https://drive.google.com/file/d/FILE_ID/view?usp=sharing
   ```
   You only need `FILE_ID` for `gdown`.

Large zips (especially with Amazon / YelpChi mats) can take a few minutes to download on the pod.

## 2. Create / start a GPU pod

In the [RunPod console](https://www.runpod.io/console/pods):

1. **Deploy** a GPU pod (PyTorch / CUDA template is fine).
2. Prefer **Python 3.10–3.12** (matches this repo).
3. Give the volume enough space for the zip + unzipped tree + `results/` (tens of GB if datasets are included).
4. Start the pod and open either:
   - **Connect → SSH**, or
   - **Connect → Web terminal**

The rest of this guide assumes a Linux shell on the pod. The usual working directory is `/workspace`.

## 3. SSH from your laptop (optional)

From the RunPod UI, copy the SSH command (example shape):

```bash
ssh root@IP -p PORT -i ~/.ssh/id_ed25519
```

Then:

```bash
cd /workspace
```

## 4. Install tools on the pod

```bash
# uv (recommended package manager for this repo)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env   # or: export PATH="$HOME/.local/bin:$PATH"

# gdown for Google Drive
pip install -U gdown
# or: uv tool install gdown
```

Check GPU:

```bash
nvidia-smi
```

## 5. Download and unzip the project

Replace `FILE_ID` with your Drive file id:

```bash
cd /workspace

gdown "https://drive.google.com/uc?id=FILE_ID" -O Graph-Fraud-WK.zip

# If gdown says the file is too large / needs confirm:
# gdown --fuzzy "https://drive.google.com/file/d/FILE_ID/view?usp=sharing" -O Graph-Fraud-WK.zip

unzip -q Graph-Fraud-WK.zip -d Graph-Fraud-WK
# If the zip already contains a top-level folder, you may get
# /workspace/Graph-Fraud-WK/Graph-Fraud-WK-... — cd into the folder that has pyproject.toml:
cd Graph-Fraud-WK
# or: cd Graph-Fraud-WK/*/   after locating pyproject.toml
ls pyproject.toml algorithms/COFRAUD dataset
```

Find the repo root if the layout is nested:

```bash
find /workspace -name pyproject.toml -path '*/Graph-Fraud*' 2>/dev/null | head
cd "$(dirname "$(find /workspace -name pyproject.toml | head -1)")"
```

Confirm datasets you care about exist, e.g.:

```bash
ls -lh dataset/Amazon.mat dataset/YelpChi.mat 2>/dev/null || ls dataset/
```

## 6. Environment: `uv sync` (GPU)

From the **repo root** (directory with `pyproject.toml`):

```bash
# x86_64 Linux GPU pods (most RunPod instances): CUDA 12.8 wheels
uv sync --extra gpu

# Activate the venv
source .venv/bin/activate
```

Verify CUDA PyTorch (you want `True` and a `+cu*` build):

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

Do **not** run a plain `pip install torch` from PyPI afterward — it can replace the GPU wheel with CPU.

## 7. Run COFRAUD

Still from the repo root, with the venv active:

```bash
cd algorithms/COFRAUD

# Quick check (Amazon)
uv run python COFRAUD_main.py --dataset amazon --gpu 0 --seed 123 --learning_rate 0.02 --train_ratio 0.4 --epochs 500

# YelpChi
uv run python COFRAUD_main.py --dataset yelpchi --gpu 0 --seed 123 --learning_rate 0.02 --train_ratio 0.4 --epochs 500
```

Or with the activated venv:

```bash
source ../../.venv/bin/activate   # if you left the repo root
python COFRAUD_main.py --dataset amazon --gpu 0 --seed 123 --learning_rate 0.02 --train_ratio 0.4 --epochs 500
```

Other datasets: `elliptic`, `tfinance`, `tsocial`, `comp` (need the matching files under `dataset/`).

Artifacts land under something like:

```text
algorithms/COFRAUD/results/COFRAUD/<dataset>/seed<SEED>/
```

(or `--save_dir` if you set one). Useful files: `metrics.json`, `checkpoint.pt`.

### Multi-seed example (Amazon)

```bash
cd /workspace/.../algorithms/COFRAUD   # your repo path

for SEED in 123 1234 12345 123456 1234567; do
  uv run python COFRAUD_main.py \
    --dataset amazon --gpu 0 \
    --seed $SEED --learning_rate 0.02 --train_ratio 0.4 --epochs 500
done
```

### Detach with `tmux` (survive browser disconnects)

```bash
tmux new -s cofraud
# ... run training ...
# Detach: Ctrl-b then d
# Reattach later:
tmux attach -t cofraud
```

## 8. Copy results back (optional)

From your **laptop**:

```bash
# Replace IP, PORT, and remote path
scp -P PORT -i ~/.ssh/id_ed25519 \
  -r root@IP:/workspace/.../algorithms/COFRAUD/results \
  ./cofraud_results
```

Or zip on the pod and `gdown`/`scp` the archive:

```bash
cd /workspace/.../algorithms/COFRAUD
zip -r /workspace/cofraud_results.zip results
```

## Troubleshooting

| Issue | What to try |
| --- | --- |
| `gdown` fails / “Cannot retrieve the public link” | Re-share as **Anyone with the link**; use `--fuzzy` with the full Drive URL; or download via a Drive API token. |
| Zip extracts to nested folders | `find` for `pyproject.toml` and `cd` there before `uv sync`. |
| `torch.cuda.is_available()` is `False` | Re-run `uv sync --extra gpu`; ensure you did not overwrite with CPU torch; confirm `nvidia-smi` works. |
| `Amazon.mat not found` | Dataset was not in the zip; put mats under `dataset/` on the pod (or re-zip including them). |
| OOM on T-Social | Run that dataset alone; lower heads (`--n_heads 4` is already the T-Social default) or use a larger GPU. |
| Pod stopped / disk wiped | Ephemeral pods lose `/workspace` unless you used a **network volume**; keep zips/results on Drive or download before stopping. |

## Minimal copy-paste block

After SSH, with your `FILE_ID` filled in:

```bash
cd /workspace
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
pip install -U gdown

gdown "https://drive.google.com/uc?id=FILE_ID" -O Graph-Fraud-WK.zip
unzip -q Graph-Fraud-WK.zip -d Graph-Fraud-WK
cd Graph-Fraud-WK
# if nested: cd into the directory that contains pyproject.toml

uv sync --extra gpu
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

cd algorithms/COFRAUD
uv run python COFRAUD_main.py --dataset amazon --gpu 0 --seed 123 --learning_rate 0.02 --train_ratio 0.4 --epochs 500
```

More COFRAUD flags and multi-seed commands: root `README.md` (COFRAUD section) and `algorithms/COFRAUD/README.md`.
