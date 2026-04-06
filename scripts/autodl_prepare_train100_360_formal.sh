#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

for thread_var in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS TF_NUM_INTEROP_THREADS TF_NUM_INTRAOP_THREADS; do
  thread_value="${!thread_var:-}"
  if [[ -z "$thread_value" || ! "$thread_value" =~ ^[1-9][0-9]*$ ]]; then
    export "$thread_var"=1
  fi
done

CONDA_ENV="${CONDA_ENV:-conan}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/${CONDA_ENV}/bin/python}"
if [[ -x "$PYTHON_BIN" ]]; then
  PYTHON_RUNNER=("$PYTHON_BIN")
else
  PYTHON_RUNNER=(conda run -n "$CONDA_ENV" python)
fi
RAW_ROOT="${RAW_ROOT:-/root/autodl-tmp/data/LibriTTS}"
DEVICE="${DEVICE:-cpu}"
WITH_F0="${WITH_F0:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
CONFIG="egs/conan_emformer_rhythm_v2_teacher_offline_train100_360.yaml"
PROCESSED_DIR="data/processed/libritts_train100_360_formal"
BINARY_DIR="data/binary/libritts_train100_360_formal_rhythm_v5"

for split in train-clean-100 train-clean-360 dev-clean test-clean; do
  if [[ ! -d "$RAW_ROOT/$split" ]]; then
    echo "[autodl-prepare-train100-360] missing raw split: $RAW_ROOT/$split" >&2
    exit 1
  fi
done

rm -rf "$PROCESSED_DIR"

if [[ "$NUM_SHARDS" =~ ^[1-9][0-9]*$ ]] && [[ "$NUM_SHARDS" -gt 1 ]]; then
  SHARD_ROOT="${PROCESSED_DIR}_shards"
  LOG_DIR="logs/prepare_train100_360_shards"
  rm -rf "$SHARD_ROOT"
  mkdir -p "$SHARD_ROOT" "$LOG_DIR"
  pids=()
  shard_dirs=()
  for (( shard_idx=0; shard_idx<NUM_SHARDS; shard_idx++ )); do
    shard_dir="$SHARD_ROOT/shard_${shard_idx}"
    shard_log="$LOG_DIR/shard_${shard_idx}.log"
    shard_dirs+=("$shard_dir")
    rm -rf "$shard_dir"
    echo "[autodl-prepare-train100-360] launch shard $((shard_idx + 1))/$NUM_SHARDS -> $shard_dir"
    "${PYTHON_RUNNER[@]}" scripts/build_libritts_local_processed_metadata.py \
      --raw_root "$RAW_ROOT" \
      --processed_data_dir "$shard_dir" \
      --config "$CONFIG" \
      --exp_name "build_libritts_local_processed_metadata_train100_360_shard_${shard_idx}" \
      --train_split train-clean-100 \
      --train_split train-clean-360 \
      --valid_split dev-clean \
      --test_split test-clean \
      --content_source emformer \
      --device "$DEVICE" \
      --min_mel_frames 32 \
      --max_mel_frames 3000 \
      --num_shards "$NUM_SHARDS" \
      --shard_index "$shard_idx" > "$shard_log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  merge_args=()
  for shard_dir in "${shard_dirs[@]}"; do
    merge_args+=(--shard_dir "$shard_dir")
  done
  "${PYTHON_RUNNER[@]}" scripts/merge_processed_metadata_shards.py \
    --output_dir "$PROCESSED_DIR" \
    "${merge_args[@]}"
else
  "${PYTHON_RUNNER[@]}" scripts/build_libritts_local_processed_metadata.py \
    --raw_root "$RAW_ROOT" \
    --processed_data_dir "$PROCESSED_DIR" \
    --config "$CONFIG" \
    --exp_name "build_libritts_local_processed_metadata_train100_360" \
    --train_split train-clean-100 \
    --train_split train-clean-360 \
    --valid_split dev-clean \
    --test_split test-clean \
    --content_source emformer \
    --device "$DEVICE" \
    --min_mel_frames 32 \
    --max_mel_frames 3000
fi

if [[ "$WITH_F0" == "1" ]]; then
  "${PYTHON_RUNNER[@]}" utils/extract_f0_rmvpe.py \
    --config "$CONFIG" \
    --batch-size 4 \
    --max-tokens 40000
fi

"${PYTHON_RUNNER[@]}" -m data_gen.tts.runs.binarize \
  --config "$CONFIG"

"${PYTHON_RUNNER[@]}" scripts/preflight_rhythm_v2.py \
  --config "$CONFIG" \
  --binary_data_dir "$BINARY_DIR" \
  --processed_data_dir "$PROCESSED_DIR" \
  --splits train valid \
  --inspect_items 4 \
  --model_dry_run \
  --strict_processed_data_dir
