#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PYTHON="$ROOT_DIR/.venv/bin/python"
RUNS_DIR="${STABLE_PRETRAINING_RUNS_DIR:-$HOME/.cache/stable-pretraining/runs}"

DATA_CONFIG="pusht"
BATCH_SIZE="256"
NUM_WORKERS="8"
PREFETCH_FACTOR="1"
CKPT_PATH=""
DRY_RUN=0
PRINT_CKPT=0
AUTO_LATEST=0
LIST_COUNT="8"
EXTRA_OVERRIDES=()

usage() {
  cat <<'USAGE'
Usage: ./resume_training.sh [options] [extra Hydra overrides...]

Lists recent Stable Pretraining .ckpt files, asks which one to use, and resumes training.
The checkpoint is passed to train.py as resume_ckpt_path=..., not through .env.

Defaults:
  data=pusht
  loader.batch_size=256
  num_workers=8
  loader.prefetch_factor=1

Options:
  --batch-size N       Set loader.batch_size.
  --num-workers N      Set num_workers.
  --prefetch-factor N  Set loader.prefetch_factor.
  --data NAME          Set the Hydra data config.
  --ckpt PATH          Resume from this checkpoint instead of auto-detecting.
  --latest             Use the newest checkpoint without prompting for selection.
  --list-count N       Number of recent checkpoints to list. Default: 8.
  --print-ckpt         Print the selected checkpoint and exit.
  --dry-run            Print the command without starting training.
  -h, --help           Show this help.

Examples:
  ./resume_training.sh
  ./resume_training.sh --latest
  ./resume_training.sh --batch-size 224 --num-workers 6
  ./resume_training.sh trainer.max_epochs=120
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --batch-size)
      BATCH_SIZE="${2:?missing value for --batch-size}"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="${2:?missing value for --num-workers}"
      shift 2
      ;;
    --prefetch-factor)
      PREFETCH_FACTOR="${2:?missing value for --prefetch-factor}"
      shift 2
      ;;
    --data)
      DATA_CONFIG="${2:?missing value for --data}"
      shift 2
      ;;
    --ckpt)
      CKPT_PATH="${2:?missing value for --ckpt}"
      shift 2
      ;;
    --latest)
      AUTO_LATEST=1
      shift
      ;;
    --list-count)
      LIST_COUNT="${2:?missing value for --list-count}"
      shift 2
      ;;
    --print-ckpt)
      PRINT_CKPT=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_OVERRIDES+=("$@")
      break
      ;;
    *)
      EXTRA_OVERRIDES+=("$1")
      shift
      ;;
  esac
done

if [ ! -x "$PYTHON" ]; then
  echo "Error: expected venv python at $PYTHON" >&2
  exit 1
fi

if [ -z "$CKPT_PATH" ]; then
  if [ ! -d "$RUNS_DIR" ]; then
    echo "Error: Stable Pretraining runs directory not found: $RUNS_DIR" >&2
    exit 1
  fi

  mapfile -t ckpt_lines < <("$PYTHON" - "$RUNS_DIR" "$LIST_COUNT" <<'PY'
import sys
import time
from pathlib import Path

root = Path(sys.argv[1]).expanduser()
limit = int(sys.argv[2])
candidates = [p for p in root.rglob("*.ckpt") if p.is_file()]
if not candidates:
    raise SystemExit(f"No .ckpt files found under {root}")

def human_age(seconds: float) -> str:
    seconds = max(0, int(seconds))
    units = [
        ("d", 86400),
        ("h", 3600),
        ("m", 60),
        ("s", 1),
    ]
    for suffix, size in units:
        if seconds >= size or suffix == "s":
            value = seconds // size
            return f"{value}{suffix} ago"

def sort_key(path: Path):
    return path.stat().st_mtime

now = time.time()
for path in sorted(candidates, key=sort_key, reverse=True)[:limit]:
    stat = path.stat()
    age = human_age(now - stat.st_mtime)
    size_mib = stat.st_size / 1024 / 1024
    print(f"{path}\t{age}\t{size_mib:.1f} MiB")
PY
)

  if [ "${#ckpt_lines[@]}" -eq 0 ]; then
    echo "Error: no checkpoints found under $RUNS_DIR" >&2
    exit 1
  fi

  if [ "$PRINT_CKPT" -eq 1 ] || [ "$AUTO_LATEST" -eq 1 ]; then
    IFS=$'\t' read -r CKPT_PATH _age _size <<< "${ckpt_lines[0]}"
  else
    echo "Found checkpoints:"
    for i in "${!ckpt_lines[@]}"; do
      IFS=$'\t' read -r path age size <<< "${ckpt_lines[$i]}"
      printf '  %d. %s (%s, %s)\n' "$((i + 1))" "$path" "$age" "$size"
    done
    echo
    read -r -p "Choose checkpoint [1]: " choice
    choice="${choice:-1}"
    if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt "${#ckpt_lines[@]}" ]; then
      echo "Error: invalid checkpoint selection: $choice" >&2
      exit 1
    fi
    IFS=$'\t' read -r CKPT_PATH _age _size <<< "${ckpt_lines[$((choice - 1))]}"
  fi
fi

if [ ! -f "$CKPT_PATH" ]; then
  echo "Error: checkpoint not found: $CKPT_PATH" >&2
  exit 1
fi

if [ "$PRINT_CKPT" -eq 1 ]; then
  echo "$CKPT_PATH"
  exit 0
fi

cmd=(
  "$PYTHON"
  train.py
  "data=$DATA_CONFIG"
  "loader.batch_size=$BATCH_SIZE"
  "num_workers=$NUM_WORKERS"
  "loader.prefetch_factor=$PREFETCH_FACTOR"
  "resume_ckpt_path=$CKPT_PATH"
)

if [ "${#EXTRA_OVERRIDES[@]}" -gt 0 ]; then
  cmd+=("${EXTRA_OVERRIDES[@]}")
fi

echo "Using checkpoint: $CKPT_PATH"
echo "Running:"
printf '  %q' "${cmd[@]}"
echo

if [ "$DRY_RUN" -eq 1 ]; then
  exit 0
fi

exec "${cmd[@]}"
