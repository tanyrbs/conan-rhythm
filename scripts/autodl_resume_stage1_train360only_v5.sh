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
SOURCE_EXP_NAME="${SOURCE_EXP_NAME:-teacher_offline_train100_360_stage1}"
EXP_NAME="${EXP_NAME:-teacher_offline_train360only_pause_recall_stage1}"
CONDA_ENV="${CONDA_ENV:-conan}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
STEP_DELTA="${STEP_DELTA:-2500}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-2500}"
MAX_SENTENCES="${MAX_SENTENCES:-8}"
MAX_TOKENS="${MAX_TOKENS:-12000}"
DS_WORKERS="${DS_WORKERS:-4}"
STOP_EXISTING="${STOP_EXISTING:-0}"
DRY_RUN="${DRY_RUN:-0}"
HP_EXTRA="${HP_EXTRA:-}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/${CONDA_ENV}/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[train360only-v5] missing python: $PYTHON_BIN" >&2
  exit 1
fi

SOURCE_WORK_DIR="checkpoints/${SOURCE_EXP_NAME}"
WORK_DIR="checkpoints/${EXP_NAME}"
CONFIG_PATH="${WORK_DIR}/config.yaml"
NOTE_DIR="${WORK_DIR}/resume_notes"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

TRAIN360_PROC="${TRAIN360_PROC:-data/processed/libritts_train360_formal_trainset}"
TRAIN360_BIN="${TRAIN360_BIN:-data/binary/libritts_train360_formal_trainset_rhythm_v5}"
LEGACY_CACHE_VERSION="${LEGACY_CACHE_VERSION:-5}"

require_file() {
  local path="$1"
  [[ -f "$path" ]] || {
    echo "[train360only-v5] missing required file: $path" >&2
    exit 1
  }
}

require_dir() {
  local path="$1"
  [[ -d "$path" ]] || {
    echo "[train360only-v5] missing required dir: $path" >&2
    exit 1
  }
}

latest_step() {
  python - "$SOURCE_WORK_DIR" <<'PY'
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

verify_binary_cache_version() {
  local split_path="$1"
  local label="$2"
  "$PYTHON_BIN" - "$REPO_ROOT" "$split_path" "$label" "$LEGACY_CACHE_VERSION" <<'PY'
import sys
from pathlib import Path
import numpy as np

repo_root = Path(sys.argv[1])
split_path = sys.argv[2]
label = sys.argv[3]
expected = int(sys.argv[4])
sys.path.insert(0, str(repo_root))

from utils.commons.indexed_datasets import IndexedDataset

dataset = IndexedDataset(split_path)
if len(dataset) <= 0:
    raise RuntimeError(f"{label}: empty indexed dataset: {split_path}")
item = dataset[0]
found = item.get("rhythm_cache_version", None)
if found is None:
    raise RuntimeError(f"{label}: missing rhythm_cache_version in {split_path}")
found = int(np.asarray(found).reshape(-1)[0])
if found != expected:
    raise RuntimeError(
        f"{label}: rhythm_cache_version mismatch at {split_path}: "
        f"found={found}, expected={expected}"
    )
print(f"[train360only-v5] {label}: cache version OK -> v{found}")
PY
}

copy_seed_checkpoint() {
  local src_ckpt="$1"
  local dst_ckpt="$2"
  mkdir -p "$(dirname "$dst_ckpt")"
  if [[ ! -f "$dst_ckpt" ]]; then
    cp "$src_ckpt" "$dst_ckpt"
  fi
}

ACTIVE_PIDS="$(collect_running_pids || true)"
if [[ -n "${ACTIVE_PIDS// }" ]]; then
  if [[ "$STOP_EXISTING" == "1" ]]; then
    echo "[train360only-v5] stopping existing pids: $ACTIVE_PIDS"
    kill $ACTIVE_PIDS || true
    sleep 3
  else
    echo "[train360only-v5] existing training pids detected: $ACTIVE_PIDS" >&2
    echo "[train360only-v5] rerun with STOP_EXISTING=1 to switch now." >&2
    exit 2
  fi
fi

require_dir "$SOURCE_WORK_DIR"
require_dir "$TRAIN360_PROC"
require_dir "$TRAIN360_BIN"
require_file "$TRAIN360_BIN/train_lengths.npy"
require_file "$TRAIN360_BIN/valid.idx"
require_file "$TRAIN360_BIN/test.idx"

verify_binary_cache_version "${TRAIN360_BIN}/train" "train360/train"
verify_binary_cache_version "${TRAIN360_BIN}/valid" "train360/valid"
verify_binary_cache_version "${TRAIN360_BIN}/test" "train360/test"

LATEST_STEP="$(latest_step)"
if [[ -z "$LATEST_STEP" ]]; then
  echo "[train360only-v5] no source checkpoint found under ${SOURCE_WORK_DIR}" >&2
  exit 1
fi

SOURCE_CKPT="${SOURCE_WORK_DIR}/model_ckpt_steps_${LATEST_STEP}.ckpt"
TARGET_CKPT="${WORK_DIR}/model_ckpt_steps_${LATEST_STEP}.ckpt"
require_file "$SOURCE_CKPT"
copy_seed_checkpoint "$SOURCE_CKPT" "$TARGET_CKPT"

DEFAULT_HP_EXTRA="binary_data_dir='${TRAIN360_BIN}',processed_data_dir='${TRAIN360_PROC}',train_sets='${TRAIN360_BIN}',load_ckpt='',rhythm_cache_version=${LEGACY_CACHE_VERSION},rhythm_allow_legacy_cache_resume=True"
if [[ -n "$HP_EXTRA" ]]; then
  HP_EXTRA="${DEFAULT_HP_EXTRA},${HP_EXTRA}"
else
  HP_EXTRA="$DEFAULT_HP_EXTRA"
fi

MAX_UPDATES="${MAX_UPDATES:-$((LATEST_STEP + STEP_DELTA))}"

mkdir -p "$NOTE_DIR"
CONFIG_BACKUP=""
if [[ -f "$CONFIG_PATH" ]]; then
  CONFIG_BACKUP="${NOTE_DIR}/config_before_train360only_step${LATEST_STEP}_${TIMESTAMP}.yaml"
  cp "$CONFIG_PATH" "$CONFIG_BACKUP"
fi

NOTE_PATH="${NOTE_DIR}/train360only_pause_recall_resume_step${LATEST_STEP}_${TIMESTAMP}.md"
cat > "$NOTE_PATH" <<EOF
# Train360-only legacy-v5 pause-recall resume note

- timestamp_utc: ${TIMESTAMP}
- source_exp_name: ${SOURCE_EXP_NAME}
- exp_name: ${EXP_NAME}
- resume_from_step: ${LATEST_STEP}
- target_max_updates: ${MAX_UPDATES}
- val_check_interval: ${VAL_CHECK_INTERVAL}
- config: ${CONFIG}
- source_ckpt: ${SOURCE_CKPT}
- copied_ckpt: ${TARGET_CKPT}
- train_binary: ${TRAIN360_BIN}
- processed_data_dir: ${TRAIN360_PROC}
- cache_lineage: legacy_v5_train360_only
- hp_extra: ${HP_EXTRA}
- config_backup: ${CONFIG_BACKUP:-N/A}

This run uses the complete existing train360 v5 binary for:

\`\`\`
train_sets='${TRAIN360_BIN}'
binary_data_dir='${TRAIN360_BIN}'
\`\`\`

It is intended as a fast pause-recall probe on existing data without deleting or rebuilding any raw data.
EOF

echo "[train360only-v5] repo        : $REPO_ROOT"
echo "[train360only-v5] config      : $CONFIG"
echo "[train360only-v5] source exp  : $SOURCE_EXP_NAME"
echo "[train360only-v5] target exp  : $EXP_NAME"
echo "[train360only-v5] latest step : $LATEST_STEP"
echo "[train360only-v5] target step : $MAX_UPDATES"
echo "[train360only-v5] val every   : $VAL_CHECK_INTERVAL"
echo "[train360only-v5] train binary: $TRAIN360_BIN"
echo "[train360only-v5] note        : $NOTE_PATH"
if [[ -n "$CONFIG_BACKUP" ]]; then
  echo "[train360only-v5] cfg backup  : $CONFIG_BACKUP"
fi
echo "[train360only-v5] hp extra    : $HP_EXTRA"

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
  echo "[train360only-v5] dry run only; not launching training."
  exit 0
fi

bash scripts/autodl_train_stage1.sh "$CONFIG" "$EXP_NAME"
