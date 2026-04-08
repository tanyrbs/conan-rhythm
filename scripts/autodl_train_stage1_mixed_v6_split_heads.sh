#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/project/conan-rhythm}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

for thread_var in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS TF_NUM_INTEROP_THREADS TF_NUM_INTRAOP_THREADS; do
  thread_value="${!thread_var:-}"
  if [[ -z "$thread_value" || ! "$thread_value" =~ ^[1-9][0-9]*$ ]]; then
    export "$thread_var"=1
  fi
done

CONFIG="${CONFIG:-egs/conan_emformer_rhythm_v2_teacher_offline_train100_360_pause_split_heads_resume.yaml}"
EXP_NAME="${EXP_NAME:-teacher_offline_train100_360_v6_split_heads_warm17500}"
CONDA_ENV="${CONDA_ENV:-conan}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MAX_UPDATES="${MAX_UPDATES:-80000}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-5000}"
MAX_SENTENCES="${MAX_SENTENCES:-8}"
MAX_TOKENS="${MAX_TOKENS:-12000}"
DS_WORKERS="${DS_WORKERS:-4}"
RESET="${RESET:-1}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

TRAIN100_PROC="${TRAIN100_PROC:-data/processed/libritts_train100_formal}"
TRAIN100_BIN="${TRAIN100_BIN:-data/binary/libritts_train100_formal_rhythm_v6}"
TRAIN360_BIN="${TRAIN360_BIN:-data/binary/libritts_train360_formal_trainset_rhythm_v6}"
BOOTSTRAP_CKPT="${BOOTSTRAP_CKPT:-checkpoints/bootstrap/model_ckpt_steps_17500.ckpt}"

NUM_CKPT_KEEP="${NUM_CKPT_KEEP:-3}"
EXTRA_VALID_MONITOR_KEY="${EXTRA_VALID_MONITOR_KEY:-rhythm_metric_pause_event_f1}"
EXTRA_VALID_MONITOR_MODE="${EXTRA_VALID_MONITOR_MODE:-max}"
EXTRA_VALID_MONITOR_FILENAME="${EXTRA_VALID_MONITOR_FILENAME:-model_ckpt_pause_best.pt}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/${CONDA_ENV}/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[mixed-v6-split-heads] missing python: $PYTHON_BIN" >&2
  exit 1
fi

require_file() {
  local path="$1"
  [[ -f "$path" ]] || {
    echo "[mixed-v6-split-heads] missing required file: $path" >&2
    exit 1
  }
}

require_dir() {
  local path="$1"
  [[ -d "$path" ]] || {
    echo "[mixed-v6-split-heads] missing required dir: $path" >&2
    exit 1
  }
}

require_dir "$TRAIN100_PROC"
require_dir "$TRAIN100_BIN"
require_dir "$TRAIN360_BIN"
require_file "$TRAIN100_BIN/train_lengths.npy"
require_file "$TRAIN100_BIN/valid_lengths.npy"
require_file "$TRAIN360_BIN/train_lengths.npy"
require_file "$BOOTSTRAP_CKPT"

HP_EXTRA_BASE="binary_data_dir='${TRAIN100_BIN}',processed_data_dir='${TRAIN100_PROC}',train_sets='${TRAIN100_BIN}|${TRAIN360_BIN}',load_ckpt='${BOOTSTRAP_CKPT}',load_ckpt_strict=False,rhythm_cache_version=6,num_ckpt_keep=${NUM_CKPT_KEEP},save_best=True,extra_valid_monitor_key='${EXTRA_VALID_MONITOR_KEY}',extra_valid_monitor_mode='${EXTRA_VALID_MONITOR_MODE}',extra_valid_monitor_filename='${EXTRA_VALID_MONITOR_FILENAME}'"
HP_EXTRA="${HP_EXTRA:-$HP_EXTRA_BASE}"

echo "[mixed-v6-split-heads] repo       : $REPO_ROOT"
echo "[mixed-v6-split-heads] config     : $CONFIG"
echo "[mixed-v6-split-heads] exp        : $EXP_NAME"
echo "[mixed-v6-split-heads] bootstrap  : $BOOTSTRAP_CKPT"
echo "[mixed-v6-split-heads] max_updates: $MAX_UPDATES"
echo "[mixed-v6-split-heads] val every  : $VAL_CHECK_INTERVAL"
echo "[mixed-v6-split-heads] hp extra   : $HP_EXTRA"

if [[ "$SKIP_PREFLIGHT" != "1" ]]; then
  "$PYTHON_BIN" scripts/preflight_rhythm_v2.py \
    --config "$CONFIG" \
    --exp_name "$EXP_NAME" \
    --hparams "$HP_EXTRA" \
    --strict_processed_data_dir \
    --model_dry_run
fi

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  echo "[mixed-v6-split-heads] preflight only; not launching training."
  exit 0
fi

export CONDA_ENV
export CUDA_VISIBLE_DEVICES
export RESET
export MAX_UPDATES
export VAL_CHECK_INTERVAL
export MAX_SENTENCES
export MAX_TOKENS
export DS_WORKERS
export HP_EXTRA

bash scripts/autodl_train_stage1.sh "$CONFIG" "$EXP_NAME"
