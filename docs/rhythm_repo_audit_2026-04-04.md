# Rhythm Repo Audit (2026-04-04)

This note records the latest repo-wide implementation review for the maintained Rhythm V2 path.

## Audit split

The review was divided into six tracks:

1. `modules/Conan/rhythm/**`
2. `tasks/Conan/rhythm/**`
3. `data_gen/**` + `utils/audio/**`
4. `scripts/**` + preflight/probe entrypoints
5. `tests/rhythm/**` + `.github/workflows/rhythm-ci.yml`
6. top-level docs / README / migration notes

## What was validated locally

- `python -m compileall -q modules tasks scripts tests utils data_gen`
- `python -m unittest discover -s tests/rhythm -p "test_*.py"`
- `python -u scripts/smoke_test_rhythm_v2.py`

Result: all three passed on the local checkout during this audit.

## Highest-priority finding fixed in this round

### 1. Frame-plan slot traversal assumed prefix-contiguous active slots

File:

- `modules/Conan/rhythm/frame_plan.py`

Problem:

- `build_frame_plan(...)` previously derived `visible_slots` from `slot_mask.sum()` and then iterated `range(visible_slots)`.
- That silently assumes all active slots are packed at the front.
- If a sparse/non-prefix slot mask appears, the code can include masked-out slots and skip later valid slots.

Why it matters:

- this is a correctness bug, not just a style issue
- it can corrupt retimed sampling, blank placement, and any downstream metric/loss using the shared frame-plan path

Fix applied:

- iterate explicit active slot indices from the mask
- keep the same public contract
- add a regression test covering sparse slot layouts

## Performance-focused changes applied in this round

### 1. Avoid extra allocation/cast in frame-plan padding

File:

- `modules/Conan/rhythm/frame_plan.py`

Change:

- `_pad_sequences(...)` now allocates directly with the target dtype/device instead of creating then recasting

Reason:

- this path is hit repeatedly by retimed target construction and rhythm render-plan materialization

### 2. Vectorize total-mask construction

File:

- `modules/Conan/rhythm/frame_plan.py`

Change:

- replace the per-batch Python loop for `total_mask` creation with an `arange < valid_length` broadcast

Reason:

- small but safe reduction in Python overhead in a hot utility path

### 3. Remove accidental global single-thread clamp from the main training launcher

Files:

- `utils/commons/single_thread_env.py`
- `tasks/run.py`
- `data_gen/tts/runs/*.py`

Problem:

- importing `utils.commons.single_thread_env` used to immediately set `OMP/MKL/OPENBLAS/...=1`
- `tasks/run.py` also called the clamp unconditionally
- that effectively capped every normal training launch to one CPU thread unless the user had already overridden the environment

Fix applied:

- make the clamp opt-in via `CONAN_SINGLE_THREAD_ENV`
- keep explicit clamps in smoke / preflight / CPU-probe style utilities
- preserve an escape hatch for binarization / launcher scripts that still want deterministic low-thread execution

Why it matters:

- this was a direct throughput ceiling on data loading, preprocessing, BLAS-heavy CPU work, and mixed CPU/GPU pipelines

### 4. Reuse the prepared batch across multi-optimizer training steps

File:

- `utils/commons/trainer_loop.py`

Problem:

- the training loop re-ran `_prepare_batch(...)` once per optimizer
- with generator + discriminator style training this duplicated batch copying / device transfer on the hottest path

Fix applied:

- prepare the batch once per training step
- hand each optimizer a cheap shallow copy of the already-prepared batch mapping

Why it matters:

- it removes redundant host-side work and unnecessary batch-to-device traffic without changing the optimizer contract

### 5. Stop pre-expanding endless dataloader batches

File:

- `utils/commons/dataset_utils.py`

Problem:

- the old endless-dataloader path materialized a repeated 1000x batch list up front
- that inflated Python memory, delayed startup, and added unnecessary list churn before training even began

Fix applied:

- replace the pre-expanded list path with a lightweight dynamic batch sampler
- rebuild batch groups lazily each cycle, while preserving shuffle/DDP sharding behavior

Why it matters:

- it raises the practical throughput ceiling for long-running jobs and reduces launcher overhead on large datasets

### 6. Reuse raw dataset items across stacked dataset builders

Files:

- `tasks/tts/dataset_utils.py`
- `tasks/Conan/rhythm/dataset_mixin.py`
- `tasks/Conan/rhythm/dataset_sample_builder.py`

Problem:

- upper dataset layers were re-fetching the same indexed item and reference item even when the base sample had already loaded them

Fix applied:

- stash `_raw_item` / `_raw_ref_item` during early sample assembly
- consume those cached objects in downstream dataset builders
- strip the temporary raw payloads before the final public sample leaves the assembler

Why it matters:

- it removes duplicate indexed-dataset lookups on a very common training/data-loading path

### 7. Make RMVPE refinement depend on `pyworld` only when used

File:

- `modules/pe/rmvpe/inference.py`

Problem:

- importing RMVPE inference pulled in `pyworld` immediately, even though only the optional audio-refinement branch needs it

Fix applied:

- lazy-load `pyworld` inside the refinement path and raise a targeted error only on use

Why it matters:

- it keeps optional-dependency failures out of unrelated import paths and makes setup/debug workflows more robust

## Remaining high-value opportunities

These were not changed yet, but remain the best next targets for throughput work:

1. reduce Python loops in `modules/Conan/rhythm/unit_frontend.py` / `unitizer.py` for large-batch streaming prep
2. add a lightweight benchmark threshold to `scripts/cpu_probe_rhythm_train.py` or CI so hot-path regressions fail earlier
3. keep cached item contracts minimal; planner/debug sidecars should stay opt-in for non-debug training
4. if retimed training expands further, consider caching reusable phase-feature blocks by duration across larger scopes

## Test coverage added

- `tests/rhythm/test_frame_plan.py`
  - added sparse-slot regression coverage for `build_frame_plan(...)`
- `tests/rhythm/test_single_thread_env.py`
  - verifies that thread clamping is opt-in and that numeric overrides work
- `tests/rhythm/test_trainer_loop.py`
  - verifies multi-optimizer training prepares the batch only once
- `tests/rhythm/test_dataset_utils.py`
  - verifies the endless sampler rebuilds batches lazily instead of relying on a giant pre-expanded batch list

## Suggested next action order

1. keep frame-plan invariants locked with tests
2. benchmark `step_content_tensor(...)` and retimed target preparation on a realistic batch
3. only then widen research branches or add richer auxiliary losses
