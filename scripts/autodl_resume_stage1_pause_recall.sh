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

CONFIG="${CONFIG:-egs/conan_emformer_rhythm_v2_teacher_offline_train100_360_pause_recall.yaml}"
EXP_NAME="${EXP_NAME:-teacher_offline_train100_360_stage1}"
CONDA_ENV="${CONDA_ENV:-conan}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
STEP_DELTA="${STEP_DELTA:-5000}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-5000}"
MAX_SENTENCES="${MAX_SENTENCES:-8}"
MAX_TOKENS="${MAX_TOKENS:-12000}"
DS_WORKERS="${DS_WORKERS:-4}"
STOP_EXISTING="${STOP_EXISTING:-0}"
DRY_RUN="${DRY_RUN:-0}"
HP_EXTRA="${HP_EXTRA:-}"
PAUSE_TOPK_ANCHOR_STEP="${PAUSE_TOPK_ANCHOR_STEP:-}"

WORK_DIR="checkpoints/${EXP_NAME}"
CONFIG_PATH="${WORK_DIR}/config.yaml"
NOTE_DIR="${WORK_DIR}/resume_notes"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
TRAIN100_PROC="${TRAIN100_PROC:-data/processed/libritts_train100_formal}"
TRAIN100_BIN="${TRAIN100_BIN:-data/binary/libritts_train100_formal_rhythm_v6}"
TRAIN360_BIN="${TRAIN360_BIN:-data/binary/libritts_train360_formal_trainset_rhythm_v6}"
LEGACY_CACHE_VERSION="${LEGACY_CACHE_VERSION:-5}"
ALLOW_LEGACY_CACHE_RESUME="${ALLOW_LEGACY_CACHE_RESUME:-1}"

require_file() {
  local path="$1"
  [[ -f "$path" ]] || {
    echo "[pause-recall-resume] missing required file: $path" >&2
    exit 1
  }
}

require_dir() {
  local path="$1"
  [[ -d "$path" ]] || {
    echo "[pause-recall-resume] missing required dir: $path" >&2
    exit 1
  }
}

latest_step() {
  python - "$WORK_DIR" <<'PY'
import glob, os, re, sys
work_dir = sys.argv[1]
paths = sorted(glob.glob(os.path.join(work_dir, "model_ckpt_steps_*.ckpt")))
if not paths:
    print("")
    raise SystemExit(0)
steps = []
for path in paths:
    m = re.search(r"steps_(\d+)\.ckpt$", path)
    if m:
        steps.append(int(m.group(1)))
print(max(steps) if steps else "")
PY
}

collect_running_pids() {
  ps -eo pid=,cmd= | awk -v self_pid="$$" -v exp_name="$EXP_NAME" '
    {
      pid = $1
      $1 = ""
      cmd = $0
      if (pid == self_pid) next
      if (cmd ~ /awk/ || cmd ~ /ps -eo pid=/) next
      if (cmd ~ /tasks.run/ && cmd ~ exp_name) print pid
      else if (cmd ~ /autodl_train_stage1.sh/ && cmd ~ exp_name) print pid
    }
  ' | sort -u
}

ACTIVE_PIDS="$(collect_running_pids || true)"
if [[ -n "${ACTIVE_PIDS// }" ]]; then
  if [[ "$STOP_EXISTING" == "1" ]]; then
    echo "[pause-recall-resume] stopping existing pids: $ACTIVE_PIDS"
    kill $ACTIVE_PIDS || true
    sleep 3
  else
    echo "[pause-recall-resume] existing training pids detected: $ACTIVE_PIDS" >&2
    echo "[pause-recall-resume] refuse to start a second writer on ${WORK_DIR}." >&2
    echo "[pause-recall-resume] rerun with STOP_EXISTING=1 if you want to switch now." >&2
    exit 2
  fi
fi

LATEST_STEP="$(latest_step)"
if [[ -z "$LATEST_STEP" ]]; then
  echo "[pause-recall-resume] no checkpoint found under ${WORK_DIR}" >&2
  exit 1
fi

require_dir "$TRAIN100_PROC"
require_dir "$TRAIN100_BIN"
require_dir "$TRAIN360_BIN"
require_file "$TRAIN100_BIN/train_lengths.npy"
require_file "$TRAIN360_BIN/train_lengths.npy"

verify_binary_cache_version() {
  local split_path="$1"
  local label="$2"
  python - "$REPO_ROOT" "$split_path" "$label" <<'PY'
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
    raise RuntimeError(f"{label}: empty indexed dataset: {split_path}")
item = dataset[0]
found = item.get("rhythm_cache_version", None)
if found is None:
    raise RuntimeError(f"{label}: missing rhythm_cache_version in {split_path}")
found = int(np.asarray(found).reshape(-1)[0])
expected = int(RHYTHM_CACHE_VERSION)
if found != expected:
    raise RuntimeError(
        f"{label}: rhythm_cache_version mismatch at {split_path}: "
        f"found={found}, expected={expected}. Re-binarize before resume."
    )
print(f"[pause-recall-resume] {label}: cache version OK -> v{found}")
PY
}

verify_binary_cache_version "${TRAIN100_BIN}/train" "train100/train"
verify_binary_cache_version "${TRAIN100_BIN}/valid" "train100/valid"
verify_binary_cache_version "${TRAIN360_BIN}/train" "train360/train"

DEFAULT_HP_EXTRA="binary_data_dir='${TRAIN100_BIN}',processed_data_dir='${TRAIN100_PROC}',train_sets='${TRAIN100_BIN}|${TRAIN360_BIN}',load_ckpt=''"
if [[ "$ALLOW_LEGACY_CACHE_RESUME" == "1" ]]; then
  DEFAULT_HP_EXTRA="${DEFAULT_HP_EXTRA},rhythm_cache_version=${LEGACY_CACHE_VERSION},rhythm_allow_legacy_cache_resume=True"
fi
if [[ "$PAUSE_TOPK_ANCHOR_STEP" == "auto" ]]; then
  PAUSE_TOPK_ANCHOR_STEP="$LATEST_STEP"
fi
if [[ -n "$PAUSE_TOPK_ANCHOR_STEP" ]]; then
  if [[ ! "$PAUSE_TOPK_ANCHOR_STEP" =~ ^[0-9]+$ ]]; then
    echo "[pause-recall-resume] PAUSE_TOPK_ANCHOR_STEP must be an integer or 'auto', got: $PAUSE_TOPK_ANCHOR_STEP" >&2
    exit 1
  fi
  DEFAULT_HP_EXTRA="${DEFAULT_HP_EXTRA},rhythm_projector_pause_topk_anchor_step=${PAUSE_TOPK_ANCHOR_STEP}"
fi
if [[ -n "$HP_EXTRA" ]]; then
  HP_EXTRA="${DEFAULT_HP_EXTRA},${HP_EXTRA}"
else
  HP_EXTRA="$DEFAULT_HP_EXTRA"
fi

MAX_UPDATES="${MAX_UPDATES:-$((LATEST_STEP + STEP_DELTA))}"

mkdir -p "$NOTE_DIR"
if [[ -f "$CONFIG_PATH" ]]; then
  CONFIG_BACKUP="${NOTE_DIR}/config_before_pause_recall_step${LATEST_STEP}_${TIMESTAMP}.yaml"
  cp "$CONFIG_PATH" "$CONFIG_BACKUP"
else
  CONFIG_BACKUP=""
fi

NOTE_PATH="${NOTE_DIR}/pause_recall_resume_step${LATEST_STEP}_${TIMESTAMP}.md"
cat > "$NOTE_PATH" <<EOF
# Pause-recall resume note

- timestamp_utc: ${TIMESTAMP}
- exp_name: ${EXP_NAME}
- resume_from_step: ${LATEST_STEP}
- target_max_updates: ${MAX_UPDATES}
- val_check_interval: ${VAL_CHECK_INTERVAL}
- config: ${CONFIG}
- work_dir: ${WORK_DIR}
- config_backup: ${CONFIG_BACKUP:-N/A}
- train100_binary: ${TRAIN100_BIN}
- train360_binary: ${TRAIN360_BIN}
- processed_data_dir: ${TRAIN100_PROC}
- hp_extra: ${HP_EXTRA}
- pause_topk_anchor_step: ${PAUSE_TOPK_ANCHOR_STEP:-N/A}

This resume keeps the mixed lineage explicit:

\`\`\`
binary_data_dir='${TRAIN100_BIN}'
processed_data_dir='${TRAIN100_PROC}'
train_sets='${TRAIN100_BIN}|${TRAIN360_BIN}'
load_ckpt=''
\`\`\`

Legacy cache handling:

- allow_legacy_cache_resume: ${ALLOW_LEGACY_CACHE_RESUME}
- configured_legacy_cache_version: ${LEGACY_CACHE_VERSION}
- note: when enabled, this resume stays on the old v5 cache lineage explicitly and does not claim soft-boundary cache semantics.

Expected semantics:

- exact resume from work_dir checkpoint
- preserve optimizer/scheduler/global_step
- new config takes effect via RESET=1
- metric thresholds stay unchanged; only loss-side pause support changes
- optional top-k anchor can restart the pause-topk schedule relative to the copied checkpoint step
EOF

echo "[pause-recall-resume] repo       : $REPO_ROOT"
echo "[pause-recall-resume] config     : $CONFIG"
echo "[pause-recall-resume] exp        : $EXP_NAME"
echo "[pause-recall-resume] latest step: $LATEST_STEP"
echo "[pause-recall-resume] target step: $MAX_UPDATES"
echo "[pause-recall-resume] val every  : $VAL_CHECK_INTERVAL"
echo "[pause-recall-resume] train100   : $TRAIN100_BIN"
echo "[pause-recall-resume] train360   : $TRAIN360_BIN"
echo "[pause-recall-resume] topk anchor: ${PAUSE_TOPK_ANCHOR_STEP:-N/A}"
echo "[pause-recall-resume] note       : $NOTE_PATH"
if [[ -n "${CONFIG_BACKUP:-}" ]]; then
  echo "[pause-recall-resume] cfg backup : $CONFIG_BACKUP"
fi
if [[ -n "$HP_EXTRA" ]]; then
  echo "[pause-recall-resume] hp extra   : $HP_EXTRA"
fi

export RESET=1
export MAX_UPDATES
export VAL_CHECK_INTERVAL
export MAX_SENTENCES
export MAX_TOKENS
export DS_WORKERS
export CONDA_ENV
export CUDA_VISIBLE_DEVICES
export HP_EXTRA

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[pause-recall-resume] dry run only; not launching training."
  exit 0
fi

bash scripts/autodl_train_stage1.sh "$CONFIG" "$EXP_NAME"
