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
  echo "missing python: $PYTHON_BIN" >&2
  exit 1
fi

export N_PROC="${N_PROC:-12}"
SCRIPT_TAG="${SCRIPT_TAG:-stage1_recovery_mixed100360_v3_trainonly}"
STATUS_FILE="${STATUS_FILE:-logs/${SCRIPT_TAG}.status}"
PID_FILE="${PID_FILE:-logs/${SCRIPT_TAG}.pid}"
SHARD_LOG_DIR="${SHARD_LOG_DIR:-logs/prepare_train360_shards_trainonly}"

echo $$ > "$PID_FILE"
stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
set_status() { printf '%s %s\n' "$(stamp)" "$1" | tee "$STATUS_FILE"; }
trap 'rc=$?; printf "%s PIPELINE_FAILED rc=%s\n" "$(stamp)" "$rc" > "$STATUS_FILE"; exit $rc' ERR
trap 'rc=$?; rm -f "$PID_FILE"; exit $rc' EXIT

require_file() {
  local path="$1"
  [[ -f "$path" ]] || { echo "missing required file: $path" >&2; exit 1; }
}

require_dir() {
  local path="$1"
  [[ -d "$path" ]] || { echo "missing required dir: $path" >&2; exit 1; }
}

collect_conflicting_pids() {
  ps -eo pid=,cmd= | awk -v self_pid="$$" '
    {
      pid = $1
      $1 = ""
      cmd = $0
      if (pid == self_pid) next
      if (cmd ~ /awk/ || cmd ~ /ps -eo pid=/) next
      if ((cmd ~ /build_libritts_local_processed_metadata.py/ && cmd ~ /libritts_train360_formal_trainset/) ||
          (cmd ~ /stage1_recovery_mixed100360_v2/) ||
          (cmd ~ /merge_processed_metadata_shards.py/ && cmd ~ /libritts_train360_formal_trainset/)) {
        print pid
      }
    }
  ' | sort -u
}

handle_conflicting_pids() {
  local pids
  pids="$(collect_conflicting_pids || true)"
  if [[ -z "${pids// }" ]]; then
    return 0
  fi
  echo "[recovery-v3] conflicting train360-related pids detected: $pids" >&2
  if [[ "${KILL_CONFLICTING_PIDS:-0}" == "1" ]]; then
    echo "[recovery-v3] killing conflicting pids because KILL_CONFLICTING_PIDS=1" >&2
    kill $pids || true
    sleep 2
    return 0
  fi
  echo "[recovery-v3] refuse to start while conflicting pids exist." >&2
  echo "[recovery-v3] either wait for v2 to finish, or rerun with KILL_CONFLICTING_PIDS=1." >&2
  set_status "CONFLICTING_TRAIN360_PIDS"
  exit 2
}

validate_shard_outputs() {
  local shard_dir
  for shard_dir in "$@"; do
    [[ -f "$shard_dir/metadata.json" || -f "$shard_dir/metadata_vctk_librittsr_gt.json" ]] || {
      echo "missing shard metadata under $shard_dir" >&2
      exit 1
    }
    [[ -f "$shard_dir/build_summary.json" ]] || {
      echo "missing shard build_summary under $shard_dir" >&2
      exit 1
    }
  done
}

validate_processed_dir() {
  local processed_dir="$1"
  require_file "$processed_dir/metadata.json"
  require_file "$processed_dir/build_summary.json"
  require_file "$processed_dir/train_item_names.txt"
}

validate_binary_dir() {
  local binary_dir="$1"
  require_file "$binary_dir/train.data"
  require_file "$binary_dir/train.idx"
  require_file "$binary_dir/train_lengths.npy"
}

existing_train360_ready() {
  local processed_dir="$1"
  local binary_dir="$2"
  [[ -f "$processed_dir/metadata.json" ]] \
    && [[ -f "$processed_dir/build_summary.json" ]] \
    && [[ -f "$processed_dir/train_item_names.txt" ]] \
    && [[ -f "$binary_dir/train.data" ]] \
    && [[ -f "$binary_dir/train.idx" ]] \
    && [[ -f "$binary_dir/train_lengths.npy" ]]
}

maybe_delete_raw_dirs() {
  local flag="$1"
  shift
  if [[ "$flag" != "1" ]]; then
    return 0
  fi
  echo "[recovery-v3] deleting raw dirs: $*" >&2
  rm -rf "$@" || true
}

WARMUP_EXP="teacher_offline_train100_warmup"
WARMUP_CKPT_20K="checkpoints/${WARMUP_EXP}/model_ckpt_steps_20000.ckpt"
TRAIN100_PROC="data/processed/libritts_train100_formal"
TRAIN100_BIN="data/binary/libritts_train100_formal_rhythm_v5"
TRAIN360_PROC="data/processed/libritts_train360_formal_trainset"
TRAIN360_SHARD_ROOT="data/processed/libritts_train360_formal_trainset_shards"
TRAIN360_BIN="data/binary/libritts_train360_formal_trainset_rhythm_v5"
FORMAL_CFG="egs/conan_emformer_rhythm_v2_teacher_offline_train100_360.yaml"
MIXED_HP="binary_data_dir='${TRAIN100_BIN}',processed_data_dir='${TRAIN100_PROC}',train_sets='${TRAIN100_BIN}|${TRAIN360_BIN}',load_ckpt='${WARMUP_CKPT_20K}',load_ckpt_strict=True"
REAL_SMOKE_EXP="teacher_offline_train100_360_mixed_real_smoke"
FORMAL_EXP="teacher_offline_train100_360_stage1"
NUM_SHARDS="${NUM_SHARDS:-8}"
RAW_ROOT="${RAW_ROOT:-/root/autodl-tmp/data/LibriTTS}"
DELETE_RAW_TRAIN360_AFTER_BINARIZE="${DELETE_RAW_TRAIN360_AFTER_BINARIZE:-1}"
DELETE_RAW_DEV_TEST_AFTER_MIXED_PREFLIGHT="${DELETE_RAW_DEV_TEST_AFTER_MIXED_PREFLIGHT:-0}"
REUSE_EXISTING_TRAIN360_ARTIFACTS="${REUSE_EXISTING_TRAIN360_ARTIFACTS:-1}"

set_status "RECOVERY_V3_TRAINONLY_START"
require_file "$WARMUP_CKPT_20K"
require_dir "$TRAIN100_PROC"
require_dir "$TRAIN100_BIN"

handle_conflicting_pids

rm -rf artifacts/rhythm_teacher_export_student_kd checkpoints/${REAL_SMOKE_EXP} || true
if [[ "$REUSE_EXISTING_TRAIN360_ARTIFACTS" != "1" ]]; then
  rm -rf "$TRAIN360_PROC" "$TRAIN360_SHARD_ROOT" "$TRAIN360_BIN" "$SHARD_LOG_DIR" || true
fi
mkdir -p "$TRAIN360_SHARD_ROOT" "$SHARD_LOG_DIR"

df -h /root/autodl-tmp || true

if existing_train360_ready "$TRAIN360_PROC" "$TRAIN360_BIN" && [[ "$REUSE_EXISTING_TRAIN360_ARTIFACTS" == "1" ]]; then
  echo "[recovery-v3] reusing existing train360 processed/binary artifacts" >&2
  validate_processed_dir "$TRAIN360_PROC"
  validate_binary_dir "$TRAIN360_BIN"
  set_status "TRAIN360_REUSE_EXISTING_ARTIFACTS"
else
  require_dir "$RAW_ROOT/train-clean-360"
  set_status "TRAIN360_METADATA_SHARDED_START"
  pids=()
  shard_dirs=()
  for (( shard_idx=0; shard_idx<NUM_SHARDS; shard_idx++ )); do
    shard_dir="$TRAIN360_SHARD_ROOT/shard_${shard_idx}"
    shard_log="$SHARD_LOG_DIR/shard_${shard_idx}.log"
    shard_dirs+=("$shard_dir")
    mkdir -p "$shard_dir"
    echo "[recovery-v3] launch shard $((shard_idx+1))/$NUM_SHARDS -> $shard_dir"
    "$PYTHON_BIN" scripts/build_libritts_local_processed_metadata.py \
      --raw_root "$RAW_ROOT" \
      --processed_data_dir "$shard_dir" \
      --config "$FORMAL_CFG" \
      --exp_name "build_libritts_local_processed_metadata_train360_formal_trainset_shard_${shard_idx}" \
      --train_split train-clean-360 \
      --content_source emformer \
      --device cuda \
      --min_mel_frames 32 \
      --max_mel_frames 3000 \
      --num_shards "$NUM_SHARDS" \
      --shard_index "$shard_idx" \
      --train_only > "$shard_log" 2>&1 &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    wait "$pid"
  done

  validate_shard_outputs "${shard_dirs[@]}"

  merge_args=()
  for shard_dir in "${shard_dirs[@]}"; do
    merge_args+=(--shard_dir "$shard_dir")
  done
  "$PYTHON_BIN" scripts/merge_processed_metadata_shards.py --output_dir "$TRAIN360_PROC" "${merge_args[@]}"
  validate_processed_dir "$TRAIN360_PROC"
  rm -rf "$TRAIN360_SHARD_ROOT" "$SHARD_LOG_DIR" || true
  set_status "TRAIN360_METADATA_SHARDED_DONE"

  set_status "TRAIN360_BINARIZE_START"
  "$PYTHON_BIN" -m data_gen.tts.runs.binarize \
    --reset \
    --config "$FORMAL_CFG" \
    --exp_name binarize_train360_formal_trainset \
    -hp "processed_data_dir='${TRAIN360_PROC}',binary_data_dir='${TRAIN360_BIN}',binarize_splits='train'"
  validate_binary_dir "$TRAIN360_BIN"
  set_status "TRAIN360_BINARIZE_DONE"
fi

set_status "TRAIN360_PREFLIGHT_START"
"$PYTHON_BIN" scripts/preflight_rhythm_v2.py \
  --config "$FORMAL_CFG" \
  --binary_data_dir "$TRAIN360_BIN" \
  --processed_data_dir "$TRAIN360_PROC" \
  --inspect_items 2 \
  --splits train \
  --model_dry_run \
  --strict_processed_data_dir
set_status "TRAIN360_PREFLIGHT_DONE"

maybe_delete_raw_dirs "$DELETE_RAW_TRAIN360_AFTER_BINARIZE" \
  "$RAW_ROOT/train-clean-360"
df -h /root/autodl-tmp || true

set_status "MIXED_PREFLIGHT_START"
"$PYTHON_BIN" scripts/preflight_rhythm_v2.py \
  --config "$FORMAL_CFG" \
  --hparams "$MIXED_HP" \
  --inspect_items 4 \
  --splits train valid test \
  --model_dry_run \
  --strict_processed_data_dir
set_status "MIXED_PREFLIGHT_DONE"

maybe_delete_raw_dirs "$DELETE_RAW_DEV_TEST_AFTER_MIXED_PREFLIGHT" \
  "$RAW_ROOT/dev-clean" \
  "$RAW_ROOT/test-clean"
df -h /root/autodl-tmp || true

set_status "MIXED_REAL_SMOKE_START"
env RESET=1 MAX_UPDATES=100 VAL_CHECK_INTERVAL=100 CONDA_ENV="$CONDA_ENV" CUDA_VISIBLE_DEVICES=0 HP_EXTRA="$MIXED_HP" \
  bash scripts/autodl_train_stage1.sh "$FORMAL_CFG" "$REAL_SMOKE_EXP"
set_status "MIXED_REAL_SMOKE_DONE"
rm -rf checkpoints/${REAL_SMOKE_EXP} || true

set_status "FORMAL_STAGE1_RUN_START"
env RESET=1 MAX_UPDATES=180000 VAL_CHECK_INTERVAL=5000 CONDA_ENV="$CONDA_ENV" CUDA_VISIBLE_DEVICES=0 HP_EXTRA="$MIXED_HP" \
  bash scripts/autodl_train_stage1.sh "$FORMAL_CFG" "$FORMAL_EXP"
set_status "FORMAL_STAGE1_RUN_DONE"
set_status "PIPELINE_COMPLETED"
