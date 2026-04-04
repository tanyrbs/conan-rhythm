# Rhythm Pre-Train Audit (2026-04-04)

> Historical snapshot: parts of this audit are outdated after the later
> teacher-main + shape-only KD cleanup. Prefer
> `docs/rhythm_training_stages.md`, `docs/rhythm_supervision_policy.md`, and
> `docs/rhythm_train_runbook.md` for the current maintained state.

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
- planner sidecar now survives dataset/cache/collate/runtime paths
- phase pointer drift diagnostics are now centralized through a derived `phase_ptr_gap`
- main maintained configs currently pass contract evaluation:
  - `minimal_v1`
  - `teacher_offline`
  - `student_kd`
  - `student_retimed`

## What was changed in this round

### 1. Training-prep correctness / guardrails

- `tasks/Conan/rhythm/runtime_teacher_supervision.py`
  - runtime teacher auxiliary budget targets now scale with the same proportional prefix ratios used by the sliced execution/proxy planner view
  - this removes the previous full-budget-vs-prefix-exec mismatch on teacher aux supervision

- `modules/Conan/rhythm/module.py`
  - cached/runtime reference conditioning now tags slow-summary provenance
  - malformed planner/slow sidecar tensors now fail fast instead of silently falling through to a fallback path

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
    - `_lengths.npy` / `_spk_ids.npy` length mismatches against the indexed dataset
    - inspected-item source/reference/target shape-contract violations

- `tasks/Conan/rhythm/dataset_contracts.py`
  - scalar metadata now rejects non-scalar arrays instead of silently taking the first element
  - partial reference/planner sidecar groups now fail fast instead of being skipped

- `utils/commons/hparams.py`
  - programmatic `set_hparams(..., reset=True)` is now supported

- `data_gen/tts/base_binarizer.py`
- `data_gen/conan_binarizer.py`
  - Windows binarization is clamped to single-process by default
  - speaker-embedding extraction now follows the same worker policy as the main item-processing path
  - empty splits fail fast and clean partial files
  - finalize/save failures now clean partial split artifacts instead of leaving `.data/.idx/.npy` half-products

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
  - sliced runtime teacher views are now explicitly marked as `auxiliary_proxy`
  - the slice exports full raw/effective/repair surfaces alongside prefix-proportional proxy budgets
  - prefix slicing keeps repair semantics observable without pretending it is a faithful full-teacher semantic view

- `tests/rhythm/test_runtime_teacher_supervision.py`
  - covers proportional prefix slicing, retained full-budget surfaces, and no-slice passthrough

### 2.6. Same-source KD observability

- `tasks/Conan/rhythm/targets.py`
- `tasks/Conan/rhythm/losses.py`
- `tasks/Conan/rhythm/loss_routing.py`
  - `student_kd` now exports explicit same-source KD diagnostics instead of leaving cache-primary/cache-distill overlap implicit
  - detached observability keys now include:
    - `L_kd_same_source`
    - `L_kd_same_source_exec`
    - `L_kd_same_source_budget`
    - `L_kd_same_source_prefix`

- `tasks/Conan/rhythm/config_contract_stage_rules.py`
  - the stage-2 warning now names the active overlapping components instead of only warning in the abstract

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
   - current code now makes that overlap observable, but still does not hard-separate the supervision path

## Validation run in this audit

### Tests

- `python -m unittest discover -s tests/rhythm -p "test_*.py"`
  - result: `57 tests` passed

### Compile / smoke

- `python -m compileall -q modules tasks tests`
  - passed

- `python scripts/smoke_test_rhythm_v2.py`
  - passed

- `python scripts/export_rhythm_teacher_targets.py --help`
  - passed

- `python scripts/cpu_probe_rhythm_train.py --help`
  - passed

### Contract / preflight

- maintained config contract evaluation:
  - `minimal_v1`: `0 error / 1 warning`
  - `teacher_offline`: `0 error / 0 warning`
  - `student_kd`: `0 error / 1 warning`
  - `student_retimed`: `0 error / 0 warning`

- current warnings:
  - `minimal_v1`: asks for `--model_dry_run` before treating preflight as train-ready
  - `student_kd`: explicit same-source cache-teacher warning remains expected under the current maintained stage-2 definition

- `python scripts/preflight_rhythm_v2.py --config egs/conan_emformer_rhythm_v2_minimal_v1.yaml --splits train valid --inspect_items 2`
  - fails correctly because `data/binary/vc_6layer` is absent

- rerun on:
  - `teacher_offline`
  - `student_kd`
  - `student_retimed`
  - `minimal_v1`
  - all now fail for the same expected reason in a clean checkout: `data/binary/vc_6layer` is absent
  - `student_kd` additionally emits the explicit same-source KD warning, which is expected under the current maintained stage-2 design

## Recommended next action order

1. prepare a real processed/binary pair for the maintained dataset
2. run stage-1 `teacher_offline`
3. export learned-offline teacher assets
4. rebuild student cache from those exports
5. rerun preflight on `student_kd` and `student_retimed`
6. run a short CPU probe
7. only then start formal training
