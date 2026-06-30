#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DRY_RUN=1
INCLUDE_RUN_CACHE=0

usage() {
  cat <<'USAGE'
Usage: ./cleanup_training.sh [--apply] [--run-cache]

Removes local training artifacts while preserving datasets and .venv.

Default behavior is a dry run.

Options:
  --apply      Actually delete the listed paths.
  --run-cache  Also delete ~/.cache/stable-pretraining/runs.
  -h, --help   Show this help.
USAGE
}

for arg in "$@"; do
  case "$arg" in
    --apply) DRY_RUN=0 ;;
    --run-cache) INCLUDE_RUN_CACHE=1 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $arg" >&2
      usage >&2
      exit 1
      ;;
  esac
done

paths=(
  "$ROOT_DIR/outputs"
  "$ROOT_DIR/data/checkpoints"
  "$ROOT_DIR/__pycache__"
)

if [ "$INCLUDE_RUN_CACHE" -eq 1 ]; then
  paths+=("$HOME/.cache/stable-pretraining/runs")
fi

echo "Cleanup targets:"
for path in "${paths[@]}"; do
  if [ -e "$path" ]; then
    echo "  remove $path"
  else
    echo "  skip   $path (not found)"
  fi
done

if [ "$DRY_RUN" -eq 1 ]; then
  echo
  echo "Dry run only. Re-run with --apply to delete these paths."
  exit 0
fi

echo
for path in "${paths[@]}"; do
  if [ -e "$path" ]; then
    rm -rf "$path"
    echo "Removed $path"
  fi
done

echo "Cleanup complete. Datasets under $ROOT_DIR/data/datasets were not touched."
