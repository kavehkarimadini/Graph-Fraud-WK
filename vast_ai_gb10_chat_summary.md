# Vast.ai GB10 Setup and Troubleshooting Summary

This document summarizes the Vast.ai-related guidance and troubleshooting from this chat for using an NVIDIA GB10 instance from a local Windows computer through Cursor/PowerShell.

---

## 1. Intended workflow

The goal is to:

1. Rent an NVIDIA GB10 GPU on Vast.ai.
2. Work from a local Windows computer using Cursor.
3. Connect to the Vast.ai instance through SSH.
4. Transfer the local repository directly to Vast.ai without requiring GitHub.
5. Recreate the Python environment on the remote GB10 using `uv`.
6. Run training/inference remotely on the GB10.
7. Copy results back to the local computer when finished.

The important architecture distinction is:

- Local machine: Windows, typically x86-64 / AMD64.
- Vast.ai GB10: Linux, ARM64 / `aarch64`.
- GB10 memory: unified CPU/GPU memory rather than a conventional discrete-GPU-only VRAM layout.

Because of this, a local Windows `.venv` should **not** be copied and reused on the GB10.

---



## 2. Recommended Vast.ai GB10 configuration

Suggested starting configuration:


| Setting          | Recommendation                                    |
| ---------------- | ------------------------------------------------- |
| GPU              | NVIDIA GB10                                       |
| GPU count        | 1                                                 |
| CPU architecture | ARM64 / `aarch64`                                 |
| Instance type    | On-demand while developing                        |
| Reliability      | Preferably 98–99%+                                |
| Verified host    | Yes                                               |
| Disk             | Around 150 GB initially; increase if needed       |
| SSH              | Enabled                                           |
| Direct SSH       | Prefer if available                               |
| Template         | Vast recommended PyTorch / ARM64-compatible image |


Important points:

- GB10 is ARM64, so packages with only x86-64 binaries may fail.
- Prefer an ARM64-compatible Vast/PyTorch image.
- Do not blindly use an arbitrary x86-only Docker image.
- Avoid manually replacing NVIDIA drivers or CUDA before first checking the preinstalled environment.

---



## 3. Installing the Vast.ai CLI locally

In the local Cursor PowerShell terminal:

```powershell
pip install -U vastai
```

Then verify:

```powershell
vastai --help
```

If using `uv`, the Vast CLI can also be kept outside the project environment:

```powershell
uv tool install vastai
```

---



## 4. Vast.ai API key

Create an API key from the Vast.ai account dashboard and configure it locally:

```powershell
vastai set api-key YOUR_API_KEY
```

Test:

```powershell
vastai show user
```

Do not commit the API key into the repository.

---



## 5. Creating the SSH directory and key on Windows

An early error occurred because:

```text
C:\Users\decis\.ssh
```

did not exist.

Create it first:

```powershell
New-Item -ItemType Directory -Force "$HOME\.ssh"
```

Then generate an Ed25519 key:

```powershell
ssh-keygen -t ed25519 -C "vast-ai" -f "$HOME\.ssh\id_ed25519"
```

This creates:

```text
id_ed25519       # private key
id_ed25519.pub   # public key
```

Verify:

```powershell
Get-ChildItem "$HOME\.ssh"
```

Display the public key:

```powershell
Get-Content "$HOME\.ssh\id_ed25519.pub"
```

The public key should look like:

```text
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAA... vast-ai
```

Never upload or share the private key:

```text
id_ed25519
```

---



## 6. Problem: Vast stored the public-key file path instead of the key

The command:

```powershell
vastai show ssh-keys
```

showed:

```text
'public_key': 'C:\\Users\\decis\\.ssh\\id_ed25519.pub'
```

That is incorrect.

Vast should store the **contents** of the public key, not the Windows path.

A safer PowerShell approach is:

```powershell
$pubkey = (Get-Content "$HOME\.ssh\id_ed25519.pub" -Raw).Trim()
```

Verify:

```powershell
$pubkey
```

It should begin with:

```text
ssh-ed25519 AAAA...
```

Then use the actual public key text when updating/attaching the SSH key.

Example shown in the chat:

```powershell
vastai update ssh-key SSH_KEY_ID "$pubkey"
```

For an already-running instance:

```powershell
vastai attach ssh INSTANCE_ID "$pubkey"
```

Then verify:

```powershell
vastai show ssh-keys
```

---



## 7. Connecting to the Vast.ai GB10

Example Vast proxy SSH connection:

```powershell
ssh -p 14531 root@ssh3.vast.ai
```

A more explicit and safer form is:

```powershell
ssh -i "$HOME\.ssh\id_ed25519" -o IdentitiesOnly=yes -p 14531 root@ssh3.vast.ai
```

If port forwarding is needed for a service on port 8080:

```powershell
ssh -i "$HOME\.ssh\id_ed25519" -o IdentitiesOnly=yes -p 14531 root@ssh3.vast.ai -L 8080:localhost:8080
```

Notes:

- `ssh` uses lowercase `-p`.
- `scp` uses uppercase `-P`.
- `-i` explicitly selects the private key.
- `-o IdentitiesOnly=yes` prevents SSH from trying unrelated keys first.

---



## 8. First-time SSH host fingerprint prompt

The first connection may show:

```text
The authenticity of host ... can't be established.
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

If the SSH host and port came directly from the Vast.ai instance connection details, type:

```text
yes
```

The host key is then stored locally in:

```text
C:\Users\decis\.ssh\known_hosts
```

---



## 9. `Permission denied (publickey)`

This means:

- The server is reachable.
- The port is correct.
- SSH authentication failed.

Likely causes discussed:

1. The public key was not added correctly to Vast.
2. The public key was added after the instance was created and not attached to the running instance.
3. The wrong private key was being used.
4. Vast had stored the path to the `.pub` file instead of its contents.

Useful diagnostics:

```powershell
vastai show ssh-keys
```

```powershell
vastai show instances
```

Verbose SSH:

```powershell
ssh -vv -i "$HOME\.ssh\id_ed25519" -o IdentitiesOnly=yes -p PORT root@HOST
```

Look for lines similar to:

```text
Offering public key:
```

and ideally:

```text
Server accepts key:
```

---



## 10. Verifying the GB10 after login

Once connected:

```bash
uname -m
```

Expected:

```text
aarch64
```

Check the GPU:

```bash
nvidia-smi
```

Test PyTorch:

```bash
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

Also inspect CUDA before installing or replacing anything:

```bash
nvcc --version
```

```bash
python -c "import torch; print(torch.version.cuda)"
```

---



## 11. Vast.ai instance guide

The instance displayed this notice:

```text
AI agents: READ /etc/vast-agents-guide.md
```

Before automating or describing instance-specific behavior from inside the instance, inspect:

```bash
cat /etc/vast-agents-guide.md
```

or the linked:

```bash
cat ./AGENTS.md
```

---



## 12. Using `uv` on the GB10

`uv` is suitable for the ARM64 GB10 and was recommended for the project.

Install on the remote machine:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Reload shell:

```bash
source ~/.bashrc
```

Verify:

```bash
uv --version
```

If the repo contains:

```text
pyproject.toml
uv.lock
```

run:

```bash
uv sync
```

Then execute code with:

```bash
uv run python main.py
```

or:

```bash
uv run python train.py
```

If only `requirements.txt` exists:

```bash
uv venv
uv pip install -r requirements.txt
```

Important:

Do not copy the local Windows `.venv` to the remote GB10.

Reasons:

- Windows binaries are incompatible with Linux.
- x86-64 binaries are incompatible with ARM64.
- CUDA-related packages may be platform-specific.

---



## 13. ARM64 dependency warning

`uv` itself is not the main concern; individual dependencies are.

Potential problems include packages that only ship:

```text
manylinux_x86_64
```

wheels.

Typical error patterns:

```text
No matching distribution found
```

```text
unsupported platform
```

or references to:

```text
x86_64
```

If this happens, check ARM64/aarch64 support before changing Python versions.

---



## 14. Directly copying the repo without GitHub

GitHub is **not required**.

The repository can be copied directly from the local Windows computer to Vast.ai over SSH/SCP.

Local repo:

```text
C:\Users\decis\Documents\GitHub\Graph-Fraud-WK
```

Remote target:

```text
/workspace/Graph-Fraud-WK
```

---



## 15. Direct SCP method

If currently inside the repo locally:

```text
C:\Users\decis\Documents\GitHub\Graph-Fraud-WK
```

one approach is:

```powershell
scp -i "$HOME\.ssh\id_ed25519" -P PORT -r . root@HOST:/workspace/Graph-Fraud-WK/
```

However, the destination directory may need to exist first, depending on the SCP/SFTP behavior.

A cleaner method is to move up one local directory:

```powershell
cd ..
```

Then copy the entire folder to `/workspace`:

```powershell
scp -i "$HOME\.ssh\id_ed25519" -P PORT -r Graph-Fraud-WK root@HOST:/workspace/
```

This creates:

```text
/workspace/Graph-Fraud-WK
```

automatically.

---



## 16. SCP destination error

The error:

```text
realpath /workspace/Graph-Fraud-WK/: No such file
path canonicalization failed
failed to upload directory . to /workspace/Graph-Fraud-WK/
```

meant the target path did not exist.

Possible fixes:

### Option A: create it remotely first

```bash
mkdir -p /workspace/Graph-Fraud-WK
```



### Option B: preferred

From the parent local directory:

```powershell
scp -i "$HOME\.ssh\id_ed25519" -P PORT -r Graph-Fraud-WK root@HOST:/workspace/
```

This avoids needing to pre-create the repo directory.

---



## 17. `mkdir` over SSH appeared to hang

This command:

```powershell
ssh -i "$HOME\.ssh\id_ed25519" -p 14531 root@ssh3.vast.ai "mkdir -p /workspace/Graph-Fraud-WK"
```

should normally finish almost instantly.

If it hangs for several minutes:

1. Cancel with:

```text
Ctrl + C
```

1. Connect interactively:

```powershell
ssh -i "$HOME\.ssh\id_ed25519" -p PORT root@HOST
```

1. Then run:

```bash
cd /workspace
mkdir -p Graph-Fraud-WK
ls -la
```

Alternatively, avoid the manual `mkdir` entirely by SCPing the whole folder to `/workspace/`.

---



## 18. Recommended archive method instead of raw `scp -r`

A compressed archive is preferable for ML repos because it:

- avoids transferring thousands of small files individually;
- lets you exclude `.venv`;
- can exclude `.git`;
- can exclude caches;
- is often cleaner and faster.

From:

```text
C:\Users\decis\Documents\GitHub
```

create an archive:

```powershell
tar --exclude=Graph-Fraud-WK/.venv --exclude=Graph-Fraud-WK/venv --exclude=Graph-Fraud-WK/__pycache__ --exclude=Graph-Fraud-WK/.pytest_cache --exclude=Graph-Fraud-WK/.git -czf Graph-Fraud-WK.tar.gz Graph-Fraud-WK
```

Upload:

```powershell
scp -i "$HOME\.ssh\id_ed25519" -P PORT Graph-Fraud-WK.tar.gz root@HOST:/workspace/
```

Connect:

```powershell
ssh -i "$HOME\.ssh\id_ed25519" -p PORT root@HOST
```

Extract:

```bash
cd /workspace
tar -xzf Graph-Fraud-WK.tar.gz
cd Graph-Fraud-WK
ls -la
```

Then recreate the environment:

```bash
uv sync
```

---



## 19. Files/folders that should generally not be uploaded

Avoid copying:

```text
.venv/
venv/
__pycache__/
.pytest_cache/
*.pyc
```

Often also unnecessary:

```text
.git/
```

unless you specifically want Git history on the remote machine.

A typical `.gitignore` includes:

```gitignore
.venv/
venv/
__pycache__/
.pytest_cache/
.env
*.pyc
```

---



## 20. Removing uploaded files from [Vast.ai](http://Vast.ai)

Connect to the instance:

```powershell
ssh -i "$HOME\.ssh\id_ed25519" -p PORT root@HOST
```

Inspect:

```bash
cd /workspace
ls -la
```

Remove the whole uploaded repo:

```bash
rm -rf /workspace/Graph-Fraud-WK
```

Remove one file:

```bash
rm filename
```

Remove one directory:

```bash
rm -rf foldername
```

Warning:

```bash
rm -rf
```

is destructive and cannot normally be undone.

---



## 21. Disconnecting from the GB10

To disconnect the SSH session:

```bash
exit
```

or:

```text
Ctrl + D
```

This only disconnects the terminal.

It does **not** stop the Vast.ai instance.

---



## 22. Stopping versus destroying an instance

Check instances:

```powershell
vastai show instances
```

Stop an instance:

```powershell
vastai stop instance INSTANCE_ID
```

Destroy an instance:

```powershell
vastai destroy instance INSTANCE_ID
```

Important distinction:

- Disconnect SSH: terminal disconnect only.
- Stop: compute is stopped, but storage/data may remain.
- Destroy: instance/data are removed permanently.

---



## 23. Important marketplace behavior after stopping a GB10

A major issue encountered:

```text
Attempting to schedule your instance. Your GPU is currently in use...
```

This means the stopped instance's physical GPU was rented by someone else.

Vast.ai is a marketplace, so stopping a machine can release the GPU.

When trying to restart:

- the instance may enter `scheduling`;
- the same GPU may not be available;
- waiting can take an unpredictable amount of time.

If the original data is important, Vast may allow copying the data directory to another running instance.

For development, the safer mental model is:

```text
Disconnect SSH != Stop instance
```

If you only want to close the terminal temporarily, use:

```bash
exit
```

rather than stopping the instance.

---



## 24. `Connection refused`

Example:

```text
ssh: connect to host ssh3.vast.ai port 14531: Connection refused
```

This is different from `Permission denied`.

It means:

- SSH authentication was not reached.
- Nothing is currently accepting connections on that host/port.

Possible causes:

- the instance is stopped;
- the instance is restarting;
- the previous SSH port is no longer valid;
- the Vast proxy mapping changed.

Check:

```powershell
vastai show instances
```

Then use the **current** SSH host and port shown by Vast.ai.

Do not assume an old port such as:

```text
14531
```

will remain valid forever.

---



## 25. `loading` status

`loading` means the instance is not yet ready for normal work.

Typically Vast is:

- downloading the container image;
- preparing the environment;
- starting the container.

Do not rely on SSH until the instance is fully running.

Check:

```powershell
vastai show instances
```

Typical progression:

```text
loading
  ->
running
```

---



## 26. `scheduling` status

`scheduling` means Vast is trying to obtain/allocate GPU resources for the instance.

This happened after a stopped GB10 could not reclaim its previous physical GPU.

If the GPU is occupied by another renter, the instance may remain in scheduling for an unpredictable period.

Practical options:

1. Wait for the original GPU.
2. Rent another available GB10.
3. Copy data from the old stopped instance to a new running instance if supported.

---



## 27. `kex_exchange_identification: Connection reset`

Example:

```text
kex_exchange_identification: read: Connection reset
Connection reset by ... port 19475
```

This means:

- the network connection reached the SSH endpoint;
- the connection was closed before SSH key exchange completed.

This is usually **not** a key-authentication problem.

Possible causes:

- instance still initializing;
- SSH service inside the instance is not ready;
- proxy mapping/backend is unhealthy;
- host/container issue;
- possibly IPv6 path/proxy behavior.

First simplify the connection and remove port forwarding:

```powershell
ssh -i "$HOME\.ssh\id_ed25519" -o IdentitiesOnly=yes -p 19475 root@ssh1.vast.ai
```

If the connection is using IPv6, try forcing IPv4:

```powershell
ssh -4 -i "$HOME\.ssh\id_ed25519" -o IdentitiesOnly=yes -p 19475 root@ssh1.vast.ai
```

Verbose diagnostic:

```powershell
ssh -vv -4 -i "$HOME\.ssh\id_ed25519" -o IdentitiesOnly=yes -p 19475 root@ssh1.vast.ai
```

Also verify:

```powershell
vastai show instances
```

and confirm the instance status is actually:

```text
running
```

---



## 28. Testing a Vast SSH port from PowerShell

Use:

```powershell
Test-NetConnection ssh1.vast.ai -Port 19475
```

If:

```text
TcpTestSucceeded : True
```

the TCP port is reachable.

If SSH still resets, the problem is likely in the Vast proxy/backend or instance SSH service.

If:

```text
TcpTestSucceeded : False
```

the host/port is not currently reachable.

---



## 29. Prefer Direct SSH if Vast offers it

Vast may provide:

- proxy SSH, such as:

```text
root@ssh1.vast.ai
```

- direct SSH to an IP address.

If a direct SSH option is available, it is often preferable for:

- reliability;
- SCP transfers;
- large files;
- reduced proxy complexity.

Example form:

```powershell
ssh -i "$HOME\.ssh\id_ed25519" -o IdentitiesOnly=yes -p DIRECT_PORT root@DIRECT_IP
```

For SCP:

```powershell
scp -i "$HOME\.ssh\id_ed25519" -P DIRECT_PORT FILE root@DIRECT_IP:/workspace/
```

---



## 30. GitHub was not required

At one point, a Git command behaved unexpectedly and appeared to route through Vast SSH settings.

However, since the desired workflow is to transfer the local repository directly, GitHub can be skipped completely.

Preferred workflow:

```text
Local Windows repo
        |
        | tar / SCP
        v
Vast.ai GB10
        |
        | uv sync
        v
Remote execution
```

No GitHub authentication is needed.

---



## 31. Recommended end-to-end workflow



### Local Windows

Repository:

```text
C:\Users\decis\Documents\GitHub\Graph-Fraud-WK
```

Check SSH key:

```powershell
Get-ChildItem "$HOME\.ssh\id_ed25519*"
```

Check instances:

```powershell
vastai show instances
```

Get the **current** SSH host and port from Vast.ai.

### Package the project

From:

```text
C:\Users\decis\Documents\GitHub
```

```
scp -i "$HOME\.ssh\id_ed25519" -P 22726 Graph-Fraud-WK.tar.gz root@217.154.252.24:/workspace/

scp -i "$env:USERPROFILE\.ssh\id_ed25519" -P 21827 Graph-Fraud-WK.tar.gz root@162.142.111.64:/workspace/
```

On the **Vast instance** (SSH session), from the folder you want to pack:

```bash
cd /path/to/that/folder   # or stay where you are
tar --exclude='.venv' --exclude='__pycache__' --exclude='.git' -czf /workspace/SplitGNN.tar.gz .
```

Or if you want the folder name inside the archive:

```bash
cd ..
tar --exclude='MyFolder/.venv' --exclude='MyFolder/__pycache__' --exclude='MyFolder/.git' -czf /workspace/MyFolder.tar.gz MyFolder
```

On your **local PC** (PowerShell), download it with the current Vast public IP and SSH port from the Connect UI:

```powershell
scp -i "$env:USERPROFILE\.ssh\id_ed25519" -P 36803 root@81.183.231.113:/workspace/SplitGNN.tar.gz "$HOME\Downloads\"
scp -i "$env:USERPROFILE\.ssh\id_ed25519" -P 22596 root@217.154.252.24:/workspace/SplitGNN.tar.gz "$HOME\Downloads\"
```

Example with values like before:

```powershell
scp -i "$env:USERPROFILE\.ssh\id_ed25519" -P 21827 root@162.142.111.64:/workspace/BWGNN.tar.gz "$HOME\Downloads\"
```

Then unzip locally:

```powershell
tar -xzf "$HOME\Downloads\my-folder.tar.gz" -C "$HOME\Downloads\my-folder"
```

If `scp` asks for a password, key auth failed — use the same `-i` key that works for `ssh` to that instance.

run:

```powershell
tar --exclude=Graph-Fraud-WK/.venv --exclude=Graph-Fraud-WK/venv --exclude=Graph-Fraud-WK/__pycache__ --exclude=Graph-Fraud-WK/.pytest_cache --exclude=Graph-Fraud-WK/.git -czf Graph-Fraud-WK.tar.gz Graph-Fraud-WK
```



### Upload

```powershell
scp -i "$HOME\.ssh\id_ed25519" -P PORT Graph-Fraud-WK.tar.gz root@HOST:/workspace/
```



### Connect

```powershell
ssh -4 -i "$HOME\.ssh\id_ed25519" -o IdentitiesOnly=yes -p PORT root@HOST
```



### On the GB10

Read instance guidance:

```bash
cat /etc/vast-agents-guide.md
```

Verify architecture:

```bash
uname -m
```

Expected:

```text
aarch64
```

Verify GPU:

```bash
nvidia-smi
```

Extract repo:

```bash
cd /workspace
tar -xzf Graph-Fraud-WK.tar.gz
cd Graph-Fraud-WK
```

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

Create/sync environment:

```bash
uv sync
```

Run:

```bash
uv run python main.py
```

or whichever project entrypoint is appropriate.

---



## 32. Recommended mental model

```text
LOCAL WINDOWS / CURSOR
C:\Users\decis\Documents\GitHub\Graph-Fraud-WK
        |
        | tar + SCP
        v
VAST.AI GB10
Linux / ARM64 / aarch64
/workspace/Graph-Fraud-WK
        |
        | uv sync
        v
Fresh Linux ARM64 .venv
        |
        | PyTorch / CUDA
        v
NVIDIA GB10
```

---



## 33. Most important lessons from the troubleshooting

1. **GB10 is ARM64**, so do not reuse your Windows/x86 `.venv`.
2. `uv` is a good choice for recreating the environment remotely.
3. GitHub is optional; direct SCP/tar transfer works.
4. Make sure Vast stores the actual `ssh-ed25519 ...` public key, not the `.pub` file path.
5. Use `-p` with SSH and `-P` with SCP.
6. Use the **current** Vast SSH port; ports can change.
7. `Permission denied (publickey)` means key authentication failed.
8. `Connection refused` means the SSH service/port is not accepting connections.
9. `Connection reset` during key exchange usually indicates an instance/proxy/backend readiness problem.
10. `loading` means the instance is still preparing.
11. `scheduling` means Vast is trying to reacquire/allocate a GPU.
12. Stopping a Vast marketplace instance can release the GPU to another user.
13. Use `exit` when you only want to disconnect without releasing the GPU.
14. Prefer direct SSH when Vast provides it.
15. For transferring the repo, a compressed archive excluding `.venv` and caches is usually cleaner than raw recursive SCP.

---



## 34. Useful command reference



### Local PowerShell

```powershell
vastai show instances
```

```powershell
vastai show ssh-keys
```

```powershell
$pubkey = (Get-Content "$HOME\.ssh\id_ed25519.pub" -Raw).Trim()
```

```powershell
ssh -4 -i "$HOME\.ssh\id_ed25519" -o IdentitiesOnly=yes -p PORT root@HOST
```

```powershell
scp -i "$HOME\.ssh\id_ed25519" -P PORT Graph-Fraud-WK.tar.gz root@HOST:/workspace/
```

```powershell
Test-NetConnection HOST -Port PORT
```



### Remote GB10

```bash
cat /etc/vast-agents-guide.md
```

```bash
uname -m
```

```bash
nvidia-smi
```

```bash
cd /workspace
```

```bash
tar -xzf Graph-Fraud-WK.tar.gz
```

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
source ~/.bashrc
```

```bash
uv sync
```

```bash
uv run python main.py
```

```bash
exit
```

---



## 35. Suggested workflow for this specific repository

For `Graph-Fraud-WK`, the cleanest workflow discussed is:

```text
1. Keep the canonical project files on the local Windows machine.
2. Rent a running GB10.
3. Wait until Vast reports it as running.
4. Verify the current SSH connection details.
5. Package Graph-Fraud-WK without .venv/.git/caches.
6. SCP the archive directly to /workspace.
7. SSH into the GB10.
8. Verify aarch64 and NVIDIA GB10.
9. Extract the repository.
10. Install uv.
11. Run uv sync.
12. Run the project.
13. Copy results/checkpoints back locally.
14. Use exit to disconnect if you intend to reconnect soon.
15. Destroy the instance only when all needed data has been copied elsewhere.
```

