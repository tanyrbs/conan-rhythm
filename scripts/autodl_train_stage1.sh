#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
for thread_var in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS TF_NUM_INTEROP_THREADS TF_NUM_INTRAOP_THREADS; do
  thread_value="${!thread_var:-}"
  if [[ -z "$thread_value" || ! "$thread_value" =~ ^[1-9][0-9]*$ ]]; then
    export "$thread_var"=1
  fi
done

CONFIG="${1:-egs/conan_emformer_rhythm_v2_teacher_offline_train100.yaml}"
EXP_NAME="${2:-teacher_offline_train100_warmup}"
CONDA_ENV="${CONDA_ENV:-conan}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/${CONDA_ENV}/bin/python}"
MAX_UPDATES="${MAX_UPDATES:-5000}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-1000}"
MAX_SENTENCES="${MAX_SENTENCES:-8}"
MAX_TOKENS="${MAX_TOKENS:-12000}"
DS_WORKERS="${DS_WORKERS:-4}"
RESET="${RESET:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
HP_EXTRA="${HP_EXTRA:-}"
NUM_WORKERS="${NUM_WORKERS:-$DS_WORKERS}"

if [[ -x "$PYTHON_BIN" ]]; then
  CMD=("$PYTHON_BIN" -m tasks.run --config "$CONFIG" --exp_name "$EXP_NAME")
else
  CMD=(conda run -n "$CONDA_ENV" python -m tasks.run --config "$CONFIG" --exp_name "$EXP_NAME")
fi
if [[ "$RESET" == "1" ]]; then
  CMD+=(--reset)
fi
CMD+=(-hp "max_updates=${MAX_UPDATES},val_check_interval=${VAL_CHECK_INTERVAL},valid_infer_interval=999999,num_valid_plots=0,num_sanity_val_steps=0,max_sentences=${MAX_SENTENCES},max_tokens=${MAX_TOKENS},ds_workers=${DS_WORKERS},save_best=True")
if [[ -n "$HP_EXTRA" ]]; then
  last_idx=$((${#CMD[@]} - 1))
  CMD[$last_idx]="${CMD[$last_idx]},${HP_EXTRA}"
fi

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" NUM_WORKERS="$NUM_WORKERS" "${CMD[@]}"
