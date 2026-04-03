# Rhythm Train Runbook

This is the maintained pre-train checklist and stage order for the rhythm mainline.

## 0. What is and is not train-ready in this repo

- Maintained stage configs:
  - `egs/conan_emformer_rhythm_v2_minimal_v1.yaml`
  - `egs/conan_emformer_rhythm_v2_teacher_offline.yaml`
  - `egs/conan_emformer_rhythm_v2_student_kd.yaml`
  - `egs/conan_emformer_rhythm_v2_student_retimed.yaml`
- Compatibility alias only:
  - `egs/conan_emformer_rhythm_v2_cached_only.yaml`
- The bundled smoke caches under `data/binary/*smoke*` are useful for structure checks only.
- They are **not** a maintained student-stage training dataset.
- At the time of this audit, the maintained configs point to `data/binary/vc_6layer`, and that directory is absent in the repository checkout.

## 1. Preprocess / metadata

Prepare a real `processed_data_dir` first.

Typical scripts live under `scripts/`, not under a separate integration/export tree:

- `scripts/build_libritts_local_processed_metadata.py`
- `scripts/preflight_rhythm_v2.py`
- `scripts/cpu_probe_rhythm_train.py`
- `scripts/export_rhythm_teacher_targets.py`
- `scripts/integration_teacher_export_student_kd.py`

## 2. Binarize teacher-facing cache

Before any maintained training stage:

1. make sure `binary_data_dir` and `processed_data_dir` point to the same corpus generation
2. confirm split naming is current
3. on Windows, keep binarization single-process for now:
   - `N_PROC=0` or `N_PROC=1`

The current code now fails fast when a split produces no valid items and cleans half-written `.data/.idx/.npy` artifacts.

## 3. Run preflight before probe or training

Run preflight first. `cpu_probe_rhythm_train.py` does **not** replace contract/cache checks.

Example:

```bash
python scripts/preflight_rhythm_v2.py \
  --config egs/conan_emformer_rhythm_v2_teacher_offline.yaml \
  --splits train valid \
  --inspect_items 2 \
  --model_dry_run
```

Preflight should be treated as failed if any of the following happens:

- `binary_data_dir` missing
- `train/valid` indexed dataset missing
- `.data` shell exists but size is zero
- `_lengths.npy` is missing
- dataset filtering empties a split
- model dry-run fails

## 4. Maintained stage order

### Stage 1: `teacher_offline`

Purpose:

- build the learned offline planner teacher

Use:

- `egs/conan_emformer_rhythm_v2_teacher_offline.yaml`

After stage-1, export teacher assets:

```bash
python scripts/export_rhythm_teacher_targets.py --help
```

## 5. Rebuild student cache from exported teacher assets

Do not start `student_kd` or `student_retimed` from stale teacher cache.

Re-binarize so the dataset contains the expected teacher fields, including confidence/source metadata required by the maintained student configs.

## 6. Preflight again on student configs

Run preflight again after re-binarization:

```bash
python scripts/preflight_rhythm_v2.py --config egs/conan_emformer_rhythm_v2_student_kd.yaml --splits train valid --inspect_items 2 --model_dry_run
python scripts/preflight_rhythm_v2.py --config egs/conan_emformer_rhythm_v2_student_retimed.yaml --splits train valid --inspect_items 2 --model_dry_run
```

## 7. Run a small CPU probe before long training

Only after preflight passes:

```bash
python scripts/cpu_probe_rhythm_train.py \
  --config egs/conan_emformer_rhythm_v2_student_kd.yaml \
  --binary_data_dir <REAL_BINARY_DATA_DIR> \
  --processed_data_dir <REAL_PROCESSED_DATA_DIR> \
  --steps 5 \
  --warmup_steps 1 \
  --device cpu
```

Interpretation:

- passing probe means the code path is alive
- it does **not** prove data quality, convergence, or maintained-stage readiness by itself

## 8. Known blockers from the current repository checkout

- `data/binary/vc_6layer` is missing
- bundled smoke caches include empty-shell splits
- some local smoke processed/binary directories are stale and do not pair with each other
- `with_f0: true` stages still require real F0 side files in processed data

## 9. Recommended go / no-go rule

Start formal training only when all of the following are true:

- preflight is clean on `train` and `valid`
- model dry-run passes
- CPU probe passes on the same real dataset
- teacher export + re-binarize has been completed before `student_kd` / `student_retimed`
