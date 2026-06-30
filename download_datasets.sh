#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
else
  export STABLEWM_HOME="$ROOT_DIR/data"
fi

STABLEWM_HOME="${STABLEWM_HOME:-$ROOT_DIR/data}"
DATA_DIR="$STABLEWM_HOME"
DATASETS_DIR="$DATA_DIR/datasets"

PYTHON="$ROOT_DIR/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "Creating README-compatible local venv at .venv..."
  uv venv --python=3.10
fi

mkdir -p "$DATASETS_DIR"

echo "Installing dataset download dependencies..."
uv pip install --python "$PYTHON" huggingface_hub zstandard

download_hf_file() {
  local repo="$1"
  local filename="$2"
  local out_dir="$3"

  mkdir -p "$out_dir"
  "$PYTHON" - "$repo" "$filename" "$out_dir" <<'PY'
import sys
from huggingface_hub import hf_hub_download

repo, filename, out_dir = sys.argv[1:]
path = hf_hub_download(
    repo_id=repo,
    repo_type="dataset",
    filename=filename,
    local_dir=out_dir,
)
print(path)
PY
}

ensure_pusht() {
  local src_dir="$DATA_DIR/pusht"
  local archive="$src_dir/pusht_expert_train.h5.zst"
  local target="$DATASETS_DIR/pusht_expert_train.h5"

  if [ -f "$target" ]; then
    echo "PushT already ready: $target"
    return
  fi

  echo "Downloading PushT archive to $src_dir..."
  download_hf_file "quentinll/lewm-pusht" "pusht_expert_train.h5.zst" "$src_dir"

  echo "Decompressing PushT to $target..."
  zstd -d -f "$archive" -o "$target"
}

convert_to_lance() {
  local dataset_name="$1"
  local source_h5="$2"
  local target_lance="$3"

  if [ -d "$target_lance" ]; then
    echo "$dataset_name Lance already ready: $target_lance"
    return
  fi

  if [ ! -f "$source_h5" ]; then
    echo "Error: missing HDF5 source for $dataset_name: $source_h5"
    exit 1
  fi

  echo "Converting $dataset_name to Lance: $target_lance"
  "$PYTHON" - "$source_h5" "$target_lance" <<'PY'
import sys
from stable_worldmodel.data import convert

source, dest = sys.argv[1:]
convert(source, dest, source_format="hdf5", dest_format="lance", mode="overwrite")
PY
}

ensure_pusht_lance() {
  ensure_pusht
  convert_to_lance \
    "PushT" \
    "$DATASETS_DIR/pusht_expert_train.h5" \
    "$DATASETS_DIR/pusht_expert_train.lance"
}

extract_tar_dataset() {
  local dataset_name="$1"
  local repo="$2"
  local archive_name="$3"
  local src_dir="$4"
  local target_rel="$5"
  local target="$DATASETS_DIR/$target_rel"
  local tmp_dir="$src_dir/extracted"

  if [ -f "$target" ]; then
    echo "$dataset_name already ready: $target"
    return
  fi

  echo "Downloading $dataset_name archive to $src_dir..."
  download_hf_file "$repo" "$archive_name" "$src_dir"

  echo "Extracting $dataset_name archive..."
  rm -rf "$tmp_dir"
  mkdir -p "$tmp_dir" "$(dirname "$target")"
  tar --zstd -xf "$src_dir/$archive_name" -C "$tmp_dir"

  local extracted_h5
  extracted_h5="$(find "$tmp_dir" -type f -name '*.h5' | head -n 1)"
  if [ -z "$extracted_h5" ]; then
    echo "Error: no .h5 file found in $src_dir/$archive_name"
    exit 1
  fi

  mv "$extracted_h5" "$target"
  rm -rf "$tmp_dir"
  echo "$dataset_name ready: $target"
}

ensure_tworoom() {
  extract_tar_dataset \
    "TwoRoom" \
    "quentinll/lewm-tworooms" \
    "tworoom.tar.zst" \
    "$DATA_DIR/tworoom" \
    "tworoom.h5"
}

ensure_tworoom_lance() {
  ensure_tworoom
  convert_to_lance \
    "TwoRoom" \
    "$DATASETS_DIR/tworoom.h5" \
    "$DATASETS_DIR/tworoom.lance"
}

ensure_cube() {
  extract_tar_dataset \
    "Cube" \
    "quentinll/lewm-cube" \
    "cube_single_expert.tar.zst" \
    "$DATA_DIR/cube" \
    "ogbench/cube_single_expert.h5"
}

ensure_cube_lance() {
  ensure_cube
  convert_to_lance \
    "Cube" \
    "$DATASETS_DIR/ogbench/cube_single_expert.h5" \
    "$DATASETS_DIR/ogbench/cube_single_expert.lance"
}

print_status() {
  echo
  echo "Dataset root: $DATA_DIR"
  echo "Loader files: $DATASETS_DIR"
  echo
  [ -f "$DATASETS_DIR/pusht_expert_train.h5" ] && pusht_h5="h5" || pusht_h5="missing h5"
  [ -d "$DATASETS_DIR/pusht_expert_train.lance" ] && pusht_lance="lance" || pusht_lance="missing lance"
  [ -f "$DATASETS_DIR/tworoom.h5" ] && tworoom_h5="h5" || tworoom_h5="missing h5"
  [ -d "$DATASETS_DIR/tworoom.lance" ] && tworoom_lance="lance" || tworoom_lance="missing lance"
  [ -f "$DATASETS_DIR/ogbench/cube_single_expert.h5" ] && cube_h5="h5" || cube_h5="missing h5"
  [ -d "$DATASETS_DIR/ogbench/cube_single_expert.lance" ] && cube_lance="lance" || cube_lance="missing lance"

  echo "1) PushT HDF5       [$pusht_h5]"
  echo "2) PushT Lance      [$pusht_lance]"
  echo "3) TwoRoom HDF5     [$tworoom_h5]"
  echo "4) TwoRoom Lance    [$tworoom_lance]"
  echo "5) Cube HDF5        [$cube_h5]"
  echo "6) Cube Lance       [$cube_lance]"
  echo "7) Download/convert all missing"
  echo "q) Quit"
  echo
}

while true; do
  print_status
  read -r -p "Choose a dataset to download: " choice
  case "$choice" in
    1) ensure_pusht ;;
    2) ensure_pusht_lance ;;
    3) ensure_tworoom ;;
    4) ensure_tworoom_lance ;;
    5) ensure_cube ;;
    6) ensure_cube_lance ;;
    7) ensure_pusht_lance; ensure_tworoom_lance; ensure_cube_lance ;;
    q|Q) exit 0 ;;
    *) echo "Unknown choice: $choice" ;;
  esac
done
