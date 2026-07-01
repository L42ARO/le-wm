# Setup

This document records the commands needed to prepare this repo on a fresh machine,
including the fixes needed to make dataset download and HDF5-to-Lance conversion work.

## 1. Create the virtual environment

This repo needs Python 3.10.

If `python3.10` is already installed on the machine:

```bash
cd /home/ubuntu/le-wm
python3.10 -m venv .venv
source .venv/bin/activate
python --version
```

If `python3.10` is not on `PATH` but exists in another local environment, use that interpreter explicitly:

```bash
cd /home/ubuntu/le-wm
/path/to/python3.10 -m venv .venv
source .venv/bin/activate
python --version
```

Expected:

```bash
Python 3.10.x
```

## 2. Configure repo-local environment variables

The repo expects environment to be loaded from `.env`.

Set `.env` to:

```bash
STABLEWM_HOME=/home/ubuntu/le-wm/data
MPLCONFIGDIR=/tmp/matplotlib
```

Before running repo commands, load it like this:

```bash
cd /home/ubuntu/le-wm
source .venv/bin/activate
set -a
source .env
set +a
```

## 3. Install Python dependencies

Install the core training stack plus the packages needed by dataset download and conversion:

```bash
cd /home/ubuntu/le-wm
source .venv/bin/activate
pip install "stable-worldmodel[train,env]" huggingface_hub zstandard hdf5plugin
```

Notes:

- `hdf5plugin` is required because the downloaded `.h5` datasets use HDF5 compression plugins.
- This repo also uses a local converter at `data/hdf5_to_lance.py` to produce Lance datasets.

## 4. Pin the `lancedb` version

`stable-worldmodel` allows newer `lancedb` versions, but this repo was validated here with `lancedb==0.30.0`.

```bash
cd /home/ubuntu/le-wm
source .venv/bin/activate
pip install --force-reinstall "lancedb==0.30.0"
```

## 5. Verify the environment

```bash
cd /home/ubuntu/le-wm
source .venv/bin/activate
set -a
source .env
set +a
python - <<'PY'
import importlib.metadata as md
import torch
print("torch", md.version("torch"))
print("stable-worldmodel", md.version("stable-worldmodel"))
print("stable-pretraining", md.version("stable-pretraining"))
print("lancedb", md.version("lancedb"))
print("hdf5plugin", md.version("hdf5plugin"))
print("cuda_available", torch.cuda.is_available())
print("cuda_version", torch.version.cuda)
PY
python train.py --help
python eval.py --help
```

## 6. Download datasets

The repo includes an interactive downloader:

```bash
cd /home/ubuntu/le-wm
source .venv/bin/activate
set -a
source .env
set +a
./download_datasets.sh
```

Menu options currently include:

- `1` PushT HDF5
- `2` PushT Lance
- `3` TwoRoom HDF5
- `4` TwoRoom Lance
- `5` Cube HDF5
- `6` Cube Lance
- `7` Download/convert all missing

If you want the training-ready PushT dataset, choose `2`.

## 7. Optional Hugging Face auth

Unauthenticated downloads work, but you may see this warning:

```bash
Warning: You are sending unauthenticated requests to the HF Hub.
```

To avoid lower rate limits, set an HF token before running dataset downloads:

```bash
export HF_TOKEN=your_token_here
```

Or add it to your shell profile if you want it persistent.

## 8. If a dataset conversion partially failed earlier

Remove the incomplete Lance output and rerun the relevant menu option:

```bash
cd /home/ubuntu/le-wm
source .venv/bin/activate
set -a
source .env
set +a
rm -rf "$STABLEWM_HOME/datasets/pusht_expert_train.lance"
./download_datasets.sh
```

Then choose `2` again.

## 9. Start training

After the required dataset has been downloaded and converted:

```bash
cd /home/ubuntu/le-wm
source .venv/bin/activate
set -a
source .env
set +a
python train.py data=pusht
```

## Repo-specific fixes already captured in this tree

These local repo changes are part of the working setup:

- `download_datasets.sh` now works without `uv`
- `download_datasets.sh` installs `hdf5plugin`
- `data/hdf5_to_lance.py` converts the LeWM HDF5 datasets into Lance format
- `.gitignore` ignores `.venv/` and `__pycache__/`
