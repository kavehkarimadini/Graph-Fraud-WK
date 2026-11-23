# Run Graph-Fraud-WK on RunPod

End-to-end: SSH into a GPU pod → download the project zip from Google Drive with `gdown` → unzip → `uv sync` → **tune, then train every model** (section 7) or a single model such as **COFRAUD** (section 8) → download `results/` (section 9).

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

## 7. Full benchmark: tune, then train every model

The benchmark is 23 models (including PMP and SEC-GFD) × 6 datasets × 4 training ratios (`0.1 0.2 0.3 0.4`) × 5 seeds (`123 1234 12345 123456 1234567`), 2,760 runs. Training does **not** tune by itself: a model on a dataset its paper did not evaluate needs `configs/tuned/<Model>/<dataset>.json`, otherwise it trains with untuned defaults (and warns). So run step 7.2 before 7.3. Every command below runs from the **repo root** with the venv active (section 6). The full explicit list (one line per run) is in [`docs/RUN_COMMANDS.md`](RUN_COMMANDS.md); background on the policy is in the root `README.md` ("Hyperparameter Policy", "Running the Full Benchmark").

**Do not add hyperparameter flags** (`--learning_rate`, `--hid_dim`, ...): explicit flags override the paper / tuned values.

**Limits per run.**
- A run that takes more than **4 hours** is stopped and leaves `OOT.txt` in its run folder.
- A run that needs more than **100 GB of GPU memory** is stopped and leaves `OOM.txt` there.
- Each file's first line is `OOT` or `OOM`, followed by the details; a stopped run writes no other result files.
- **Pick a GPU with more than 100 GB** (H200 141 GB, B200 180 GB) if the OOM column should mean exactly "more than 100 GB". On an 80 GB A100/H100 a run is already marked OOM when it needs more than 80 GB.

Details are in the root `README.md` ("Time and memory limits").

**Folder names.** Every run writes to `results/<Model>/<dataset>/seed<seed>_train<pct>/`, built from its arguments. All 2,760 commands map to different folders, so running them from top to bottom, model after model, never overwrites anything.

### 7.1 Check data and start tmux

```bash
cd /workspace/Graph-Fraud-WK            # the folder with pyproject.toml
source .venv/bin/activate               # or: source /tmp/gf-venv/bin/activate (section 6)
ls dataset/YelpChi.mat dataset/Amazon.mat dataset/tfinance dataset/tsocial dataset/comp.dgl dataset/elliptic_bitcoin_dataset/
tmux new -s bench                       # detach: Ctrl-b d, reattach: tmux attach -t bench
```

T-Social (5.8M nodes) needs a large-memory GPU; run it on its own pod if needed (`--skip_dataset tsocial` / `DATASETS=...` below).

### 7.2 Hyperparameter tuning (77 pairs, once)

```bash
python tune_hparams.py --list                    # TUNE = still to do
python tune_hparams.py --all --gpu 0             # 20 trials x 5 seeds per pair, 40% split, validation AUROC only
python tune_hparams.py --all --resume --gpu 0    # after a disconnect / pod restart
python tune_hparams.py --list                    # done when no TUNE cell is left
```

Split across pods, e.g. small datasets on one pod and T-Social on another:

```bash
python tune_hparams.py --all --dataset comp,elliptic,amazon --gpu 0
python tune_hparams.py --all --dataset yelpchi,tfinance --gpu 0
python tune_hparams.py --all --dataset tsocial --gpu 0
```

Every training pod needs the tuned configs. Download them (`zip -r /workspace/tuned.zip configs/tuned`, then `scp` as in section 9) and either commit them to the repo or copy them into `configs/tuned/` on the other pods.

### 7.3 Train every model × dataset × ratio × seed

Finished runs (`metrics.json`) and stopped runs (`OOT.txt` / `OOM.txt`) are skipped, so re-running the loop after an interruption continues where it stopped. Per-run logs go to `logs/`.

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

Run a subset by shortening the lists, e.g. one model on all datasets: `MODELS="BWGNN"`, or skip T-Social: `DATASETS="yelpchi amazon tfinance elliptic comp"`. Follow a run live with `tail -f logs/<file>.log`.

Results: `results/<Model>/<dataset>/seed<seed>_train<pct>/`. A finished run has `config.json`, `metrics.json`, `predictions.npz`, `split.npz` and `checkpoint.pt`; a stopped run has only `OOT.txt` or `OOM.txt`.

### 7.4 Check, then download

```bash
# Per model: done / OOT / OOM (total should be 120 = 6 datasets x 4 ratios x 5 seeds)
python -c "import glob,collections,pathlib as P; K={'metrics.json':'done','OOT.txt':'OOT','OOM.txt':'OOM'}; c=collections.defaultdict(collections.Counter); fs=glob.glob('results/*/*/seed*_train*/*'); [c[P.Path(f).parts[1]].update([K[P.Path(f).name]]) for f in fs if P.Path(f).name in K]; [print(m, dict(v), 'total', sum(v.values())) for m,v in sorted(c.items())]"
# List every stopped run
python -c "import glob; print(*sorted(glob.glob('results/*/*/seed*_train*/OO[TM].txt')), sep='\n')"
# Runs that trained without a tuned config (should print 0: re-tune that pair and re-run)
python -c "import glob,json; b=[p for p in glob.glob('results/*/*/seed*_train*/config.json') if json.load(open(p)).get('hparam_source')=='untuned_default']; print(len(b)); print(*b[:20], sep='\n')"
grep -l "FAILED\|Traceback" logs/*.log | head      # crashed runs, if any (OOT / OOM runs are listed above)

zip -r /workspace/benchmark_results.zip results configs/tuned
```

Then download `/workspace/benchmark_results.zip` with `scp` (section 9.2).

## 8. Run a single model (COFRAUD example)

Still from the repo root, with the venv active. On Amazon / YelpChi (COFRAUD's paper datasets) `--learning_rate 0.02 --epochs 500` equal the paper defaults; on other datasets **leave hyperparameter flags out** so the tuned values from section 7 apply.

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
algorithms/COFRAUD/results/COFRAUD/<dataset>/seed<SEED>_train40/
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

## 9. Zip results on the pod and download to your PC

COFRAUD writes under `algorithms/COFRAUD/results/` (or `--save_dir`). Zip that folder on the pod, then pull the archive with `scp` from your laptop.

### 9.1 On the pod — create the zip

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

### 9.2 On your PC — download with `scp`

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

### 9.3 On your PC — unzip

```powershell
Expand-Archive -Path .\cofraud_results.zip -DestinationPath .\cofraud_results -Force
# or 7-Zip / Explorer: right-click → Extract
```

You should see paths like `cofraud_results\results\COFRAUD\<dataset>\seed<SEED>\metrics.json`.

### 9.4 Alternative: copy the folder without zipping

Slower for many small files, but fine for one run:

```powershell
scp -P PORT -i $env:USERPROFILE\.ssh\id_ed25519 -r `
  root@IP:/workspace/.../algorithms/COFRAUD/results `
  .\cofraud_results
```

### 9.5 Alternative: upload the zip to Google Drive from the pod

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

For the full benchmark instead of the last two lines, stay in the repo root and follow section 7 (`python tune_hparams.py --all --gpu 0`, then the 7.3 loop).

More COFRAUD flags and multi-seed commands: root `README.md` (COFRAUD section) and `algorithms/COFRAUD/README.md`.
