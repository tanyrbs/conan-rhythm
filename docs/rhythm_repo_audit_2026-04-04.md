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

## Remaining high-value opportunities

These were not changed yet, but remain the best next targets for throughput work:

1. reduce Python loops in `modules/Conan/rhythm/unit_frontend.py` / `unitizer.py` for large-batch streaming prep
2. add a lightweight benchmark threshold to `scripts/cpu_probe_rhythm_train.py` or CI so hot-path regressions fail earlier
3. keep cached item contracts minimal; planner/debug sidecars should stay opt-in for non-debug training
4. if retimed training expands further, consider caching reusable phase-feature blocks by duration across larger scopes

## Test coverage added

- `tests/rhythm/test_frame_plan.py`
  - added sparse-slot regression coverage for `build_frame_plan(...)`

## Suggested next action order

1. keep frame-plan invariants locked with tests
2. benchmark `step_content_tensor(...)` and retimed target preparation on a realistic batch
3. only then widen research branches or add richer auxiliary losses
