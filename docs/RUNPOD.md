# Run Graph-Fraud-WK on RunPod (COFRAUD)

End-to-end: SSH into a GPU pod → download the project zip from Google Drive with `gdown` → unzip → `uv sync` → train **COFRAUD**.

## 1. Prepare the Google Drive zip

1. Zip the repo on your machine (include `dataset/` with `.mat` / `.dgl` files you need).
2. Upload the zip to Google Drive.
3. Share the file: **Anyone with the link** → Viewer.
4. Copy the link. It looks like:
   ```text
       Compress-Archive -Path Graph-Fraud-WK\* -DestinationPath Graph-Fraud-WK.zip

       https://drive.google.com/file/d/FILE_ID/view?usp=sharing
       https://drive.google.com/file/d/1N8O9l-4mFKWCAKaTwnikF50mTPN8Zvhi/view?usp=sharing
   ```

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
ssh root@103.196.86.173 -p 19671 -i ~/.ssh/id_ed25519
copy your public key:

Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub
RunPod → Settings → SSH Public Keys → paste that whole line → save.
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

gdown "https://drive.google.com/uc?id=1N8O9l-4mFKWCAKaTwnikF50mTPN8Zvhi" -O Graph-Fraud-WK.zip

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

cd /workspace/Graph-Fraud-WK/Graph-Fraud-WK
rm -rf .venv

export UV_CACHE_DIR=/tmp/uv-cache
export UV_PROJECT_ENVIRONMENT=/tmp/gf-venv
mkdir -p "$UV_CACHE_DIR"

uv sync --extra gpu
source /tmp/gf-venv/bin/activate

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

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

## 8. Zip results on the pod and download to your PC

COFRAUD writes under `algorithms/COFRAUD/results/` (or `--save_dir`). Zip that folder on the pod, then pull the archive with `scp` from your laptop.

### 8.1 On the pod — create the zip

```bash
# Go to the COFRAUD algorithm folder (adjust nested path if needed)
cd /workspace/Graph-Fraud-WK/Graph-Fraud-WK/algorithms/COFRAUD
ls results

# Zip the whole results tree into /workspace (easy to find / scp)
zip -r /workspace/cofraud_results_4.zip results

# Confirm size
ls -lh /workspace/cofraud_results.zip
```

Optional: zip only one dataset / seed:

```bash
zip -r /workspace/cofraud_amazon_seed123.zip results/COFRAUD/amazon/seed123
```

If `zip` is missing:

```bash
apt-get update && apt-get install -y zip
# or use tar instead:
tar -czf /workspace/cofraud_results.tar.gz -C algorithms/COFRAUD results
```

Leave the SSH session open or note the pod **IP** and **PORT** from **Connect → SSH** (e.g. `root@157.157.221.29 -p 50642`).

### 8.2 On your PC — download with `scp`

**PowerShell** (Windows), from any folder you want the file in:

```powershell
# Direct TCP SSH (IP + port from RunPod Connect)
scp -P 50642 -i $env:USERPROFILE\.ssh\id_ed25519 `
  root@157.157.221.29:/workspace/cofraud_results_4.zip `
  .\cofraud_results_4.zip
```

Replace `50642` / `157.157.221.29` with **your** pod’s port and IP. To put it somewhere specific:

```powershell
scp -P 50642 -i $env:USERPROFILE\.ssh\id_ed25519 `
  root@IP:/workspace/cofraud_results.zip `
  C:\Users\decis\Documents\GitHub\Graph-Fraud-WK\cofraud_results.zip
```

If you connect via RunPod’s SSH proxy instead (`user@ssh.runpod.io`), use that host and the key the UI shows (often no custom `-P`, or port `22`):

```powershell
scp -i $env:USERPROFILE\.ssh\id_ed25519 `
  PODUSER@ssh.runpod.io:/workspace/cofraud_results.zip `
  .\cofraud_results.zip
```

**Bash / macOS / WSL:**

```bash
scp -P PORT -i ~/.ssh/id_ed25519 \
  root@IP:/workspace/cofraud_results.zip \
  ./cofraud_results.zip
```

### 8.3 On your PC — unzip

```powershell
Expand-Archive -Path .\cofraud_results.zip -DestinationPath .\cofraud_results -Force
# or 7-Zip / Explorer: right-click → Extract
```

You should see paths like `cofraud_results\results\COFRAUD\<dataset>\seed<SEED>\metrics.json`.

### 8.4 Alternative: copy the folder without zipping

Slower for many small files, but fine for one run:

```powershell
scp -P PORT -i $env:USERPROFILE\.ssh\id_ed25519 -r `
  root@IP:/workspace/.../algorithms/COFRAUD/results `
  .\cofraud_results
```

### 8.5 Alternative: upload the zip to Google Drive from the pod

If `scp` is awkward, upload from the pod (needs Drive credentials / `rclone` or a shared upload tool). Simplest path for this project is still **zip +** `scp`.

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
