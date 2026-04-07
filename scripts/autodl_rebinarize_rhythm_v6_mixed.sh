#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/project-1/conan-rhythm}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

for thread_var in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS TF_NUM_INTEROP_THREADS TF_NUM_INTRAOP_THREADS; do
  thread_value="${!thread_var:-}"
  if [[ -z "$thread_value" || ! "$thread_value" =~ ^[1-9][0-9]*$ ]]; then
    export "$thread_var"=1
  fi
done

CONDA_ENV="${CONDA_ENV:-conan}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/${CONDA_ENV}/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[rebinarize-v6] missing python: $PYTHON_BIN" >&2
  exit 1
fi
N_PROC="${N_PROC:-1}"
export N_PROC

TRAIN100_CFG="${TRAIN100_CFG:-egs/conan_emformer_rhythm_v2_teacher_offline_train100.yaml}"
TRAIN100_PROC="${TRAIN100_PROC:-data/processed/libritts_train100_formal}"
TRAIN100_BIN_OLD="${TRAIN100_BIN_OLD:-data/binary/libritts_train100_formal_rhythm_v5}"
TRAIN100_BIN="${TRAIN100_BIN:-data/binary/libritts_train100_formal_rhythm_v6}"

TRAIN360_CFG="${TRAIN360_CFG:-egs/conan_emformer_rhythm_v2_teacher_offline_train100_360.yaml}"
TRAIN360_PROC="${TRAIN360_PROC:-data/processed/libritts_train360_formal_trainset}"
TRAIN360_BIN_OLD="${TRAIN360_BIN_OLD:-data/binary/libritts_train360_formal_trainset_rhythm_v5}"
TRAIN360_BIN="${TRAIN360_BIN:-data/binary/libritts_train360_formal_trainset_rhythm_v6}"

STATUS_FILE="${STATUS_FILE:-logs/rebinarize_rhythm_v6_mixed.status}"
LOG_DIR="${LOG_DIR:-logs/rebinarize_rhythm_v6_mixed}"
mkdir -p "$LOG_DIR"

stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
set_status() { printf '%s %s\n' "$(stamp)" "$1" | tee "$STATUS_FILE"; }

require_dir() {
  local path="$1"
  [[ -d "$path" ]] || {
    echo "[rebinarize-v6] missing dir: $path" >&2
    exit 1
  }
}

ensure_target_dir_name() {
  local old_dir="$1"
  local new_dir="$2"
  if [[ -d "$new_dir" ]]; then
    return 0
  fi
  if [[ -d "$old_dir" ]]; then
    echo "[rebinarize-v6] renaming $old_dir -> $new_dir"
    mv "$old_dir" "$new_dir"
  else
    mkdir -p "$new_dir"
  fi
}

verify_split_version() {
  local split_path="$1"
  local label="$2"
  "$PYTHON_BIN" - "$REPO_ROOT" "$split_path" "$label" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
split_path = sys.argv[2]
label = sys.argv[3]
sys.path.insert(0, str(repo_root))

from utils.commons.indexed_datasets import IndexedDataset
from modules.Conan.rhythm.surface_metadata import RHYTHM_CACHE_VERSION
import numpy as np

dataset = IndexedDataset(split_path)
if len(dataset) <= 0:
    raise RuntimeError(f"{label}: empty indexed dataset at {split_path}")
item = dataset[0]
found = item.get("rhythm_cache_version", None)
if found is None:
    raise RuntimeError(f"{label}: missing rhythm_cache_version at {split_path}")
found = int(np.asarray(found).reshape(-1)[0])
expected = int(RHYTHM_CACHE_VERSION)
if found != expected:
    raise RuntimeError(f"{label}: found cache version {found}, expected {expected}")
print(f"[rebinarize-v6] {label}: cache version OK -> v{found}")
PY
}

rebinarize_train100() {
  require_dir "$TRAIN100_PROC"
  ensure_target_dir_name "$TRAIN100_BIN_OLD" "$TRAIN100_BIN"
  set_status "TRAIN100_BINARIZE_START"
  "$PYTHON_BIN" -m data_gen.tts.runs.binarize \
    --reset \
    --config "$TRAIN100_CFG" \
    --exp_name binarize_train100_formal_v6 \
    -hp "processed_data_dir='${TRAIN100_PROC}',binary_data_dir='${TRAIN100_BIN}'" \
    > "${LOG_DIR}/train100_binarize.log" 2>&1
  set_status "TRAIN100_BINARIZE_DONE"
  set_status "TRAIN100_PREFLIGHT_START"
  "$PYTHON_BIN" scripts/preflight_rhythm_v2.py \
    --config "$TRAIN100_CFG" \
    --binary_data_dir "$TRAIN100_BIN" \
    --processed_data_dir "$TRAIN100_PROC" \
    --splits train valid test \
    --inspect_items 2 \
    --model_dry_run \
    --strict_processed_data_dir \
    > "${LOG_DIR}/train100_preflight.log" 2>&1
  set_status "TRAIN100_PREFLIGHT_DONE"
  verify_split_version "${TRAIN100_BIN}/train" "train100/train"
  verify_split_version "${TRAIN100_BIN}/valid" "train100/valid"
  verify_split_version "${TRAIN100_BIN}/test" "train100/test"
}

rebinarize_train360() {
  require_dir "$TRAIN360_PROC"
  ensure_target_dir_name "$TRAIN360_BIN_OLD" "$TRAIN360_BIN"
  set_status "TRAIN360_BINARIZE_START"
  "$PYTHON_BIN" -m data_gen.tts.runs.binarize \
    --reset \
    --config "$TRAIN360_CFG" \
    --exp_name binarize_train360_formal_trainset_v6 \
    -hp "processed_data_dir='${TRAIN360_PROC}',binary_data_dir='${TRAIN360_BIN}',binarize_splits='train'" \
    > "${LOG_DIR}/train360_binarize.log" 2>&1
  set_status "TRAIN360_BINARIZE_DONE"
  set_status "TRAIN360_PREFLIGHT_START"
  "$PYTHON_BIN" scripts/preflight_rhythm_v2.py \
    --config "$TRAIN360_CFG" \
    --binary_data_dir "$TRAIN360_BIN" \
    --processed_data_dir "$TRAIN360_PROC" \
    --splits train \
    --inspect_items 2 \
    --model_dry_run \
    --strict_processed_data_dir \
    > "${LOG_DIR}/train360_preflight.log" 2>&1
  set_status "TRAIN360_PREFLIGHT_DONE"
  verify_split_version "${TRAIN360_BIN}/train" "train360/train"
}

set_status "REBINARY_V6_START"
echo "[rebinarize-v6] using N_PROC=${N_PROC}"
rebinarize_train100
rebinarize_train360
set_status "REBINARY_V6_DONE"
echo "[rebinarize-v6] done"
