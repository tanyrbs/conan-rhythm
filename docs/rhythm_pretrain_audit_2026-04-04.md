# Rhythm Pre-Train Audit (2026-04-04)

This audit was run on the local checkout at:

- `D:\00_project\002project\rhythm\Conan-master`

## What was checked

Six parallel code-path audits were used to re-check:

- training/loss/metrics/contracts
- inference/runtime/reference conditioning
- data extraction / binarization / multiprocess / preflight
- projector / controller / feasibility / derived state
- config / stage contract / YAML alignment
- docs / scripts / test coverage

## What is already in good shape

- raw-vs-exec budget supervision is already present in `losses.py`
- KD and runtime teacher auxiliary loss are already split
- masked KL / plan / distill numerics are stable
- projector feasibility math has invariant coverage
- main maintained configs currently pass contract evaluation:
  - `minimal_v1`
  - `teacher_offline`
  - `student_kd`
  - `student_retimed`

## What was changed in this round

### 1. Training-prep correctness / guardrails

- `modules/Conan/rhythm/stages.py`
  - add `cached_only -> minimal_v1` stage alias
  - detect `cached_only` config names as `minimal_v1`

- `tasks/Conan/rhythm/config_contract_stage_rules.py`
  - explicit `rhythm_stage: minimal_v1` now forces `minimal_v1` profile validation instead of silently falling back to `default`

- `tasks/Conan/rhythm/preflight_support.py`
  - preflight now flags:
    - zero-byte `.data` shells
    - zero-byte `.idx`
    - missing `_lengths.npy`

- `utils/commons/hparams.py`
  - programmatic `set_hparams(..., reset=True)` is now supported

- `data_gen/tts/base_binarizer.py`
- `data_gen/conan_binarizer.py`
  - Windows binarization is clamped to single-process by default
  - empty splits fail fast and clean partial files

### 2. Observability

- `modules/Conan/rhythm/bridge.py`
  - exports raw/effective/feasible budget surfaces

- `tasks/Conan/rhythm/metrics.py`
  - exports repair/gap metrics:
    - `rhythm_metric_budget_raw_exec_gap_mean`
    - `rhythm_metric_budget_raw_exec_gap_ratio_mean`
    - `rhythm_metric_budget_repair_ratio_mean`
    - `rhythm_metric_budget_repair_active_rate`

### 2.5. Runtime teacher auxiliary slicing

- `tasks/Conan/rhythm/runtime_teacher_supervision.py`
  - sliced runtime teacher views no longer collapse raw/effective budgets into truncated exec sums
  - prefix slicing now preserves raw-vs-exec / feasible-repair semantics proportionally instead of zeroing feasible deltas

- `tests/rhythm/test_runtime_teacher_supervision.py`
  - adds regression coverage for proportional prefix slicing and no-slice passthrough

### 3. Docs

- `docs/rhythm_training_stages.md`
  - clarified that `conan_emformer_rhythm_v2_cached_only.yaml` is a compatibility alias, not a stricter maintained base

- `docs/rhythm_train_runbook.md`
  - added the maintained pre-train runbook

### 4. File-boundary cleanup

- `tasks/Conan/rhythm/task_runtime_support.py`
  - pulls runtime forwarding / acoustic target bundling / loss routing / offline-confidence packaging out of `task_mixin.py`

- `tasks/Conan/rhythm/dataset_target_builder.py`
  - pulls cached-prefix adaptation and runtime-target merge logic out of `dataset_mixin.py`

## Current hard blockers for formal training

These are real blockers, not style issues:

1. `data/binary/vc_6layer` is missing in the checked-out repository
2. bundled smoke caches are not maintained student-stage data
3. `data/binary/libritts_single_smoke_rhythm_v4` contains an empty `valid.data`
4. that same smoke cache also fails maintained `minimal_v1` expectations because teacher source/surface metadata are stale:
   - found `offline_teacher_surface_v1`
   - expected `offline_teacher_surface_learned_offline_v1`
   - found teacher source id `0`
   - expected `1`
5. on Windows, binarization should currently stay at `N_PROC=0/1`
6. `with_f0: true` stages still require real F0 side files in processed data

## Important follow-ups that are real but not fixed in this round

These need another focused pass before claiming “fully train-ready mainline”:

1. `targets.py` / `config_contract_stage_rules.py`
   - `student_kd` can still reuse the same cached teacher surface for both primary supervision and KD
   - current contract only warns; it does not force a cleaner separation

2. dataset/runtime planner sidecar path
   - planner slow-rhythm sidecars are accepted by runtime conditioning
   - but dataset/collate/contract export is still incomplete, so sidecars can silently disappear

3. projector derived-state contract
   - `phase_ptr` vs `phase_progress_ratio` semantics still deserve stronger regression coverage

## Validation run in this audit

### Tests

- `python -m unittest tests.rhythm.test_preflight_readiness tests.rhythm.test_policy_contract_and_loss_routing tests.rhythm.test_loss_components tests.rhythm.test_loss_confidence_routing tests.rhythm.test_loss_dict_numerics tests.rhythm.test_metrics_masking tests.rhythm.test_budget_surfaces tests.rhythm.test_projector_invariants tests.rhythm.test_reference_sidecar tests.rhythm.test_target_builder`
  - result: `35 tests` passed

### Compile / smoke

- `python -m compileall -q modules tasks scripts tests utils data_gen`
  - passed

- `python scripts/smoke_test_rhythm_v2.py`
  - passed

### Contract / preflight

- maintained config contract evaluation:
  - `minimal_v1`: `0 error / 0 warning`
  - `teacher_offline`: `0 error / 0 warning`
  - `student_kd`: `0 error / 0 warning`
  - `student_retimed`: `0 error / 0 warning`

- `python scripts/preflight_rhythm_v2.py --config egs/conan_emformer_rhythm_v2_minimal_v1.yaml --splits train valid --inspect_items 2`
  - fails correctly because `data/binary/vc_6layer` is absent

- `python scripts/preflight_rhythm_v2.py --config egs/conan_emformer_rhythm_v2_minimal_v1.yaml --binary_data_dir data/binary/libritts_single_smoke_rhythm_v4 --splits train valid --inspect_items 2`
  - now fails explicitly on:
    - stale teacher source/surface metadata
    - empty `valid.data`

## Recommended next action order

1. prepare a real processed/binary pair for the maintained dataset
2. run stage-1 `teacher_offline`
3. export learned-offline teacher assets
4. rebuild student cache from those exports
5. rerun preflight on `student_kd` and `student_retimed`
6. run a short CPU probe
7. only then start formal training
