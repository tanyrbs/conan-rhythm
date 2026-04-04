# Rhythm Migration Plan (2026-04-04)

This is the only maintained document under `docs/`. Together with `README.md`, it defines the current training path, what changed recently, and what is still blocked.

## 1. Maintained path and code boundaries

The maintained rhythm path remains:

1. `teacher_offline`
2. export teacher targets / rebuild student cache
3. `student_kd`
4. prepare retimed cache + F0 side files
5. `student_retimed`

Critical code ownership is intentionally narrow:

- timing / feasibility / projector logic: `modules/Conan/rhythm/`
- stage contracts, targets, losses, metrics, preflight: `tasks/Conan/rhythm/`
- maintained validation entrypoints:
  - `scripts/smoke_test_rhythm_v2.py`
  - `scripts/preflight_rhythm_v2.py`
  - `scripts/cpu_probe_rhythm_train.py`

## 2. Critical-thinking mapping from the style review to this repo

The style-branch names from the external review do **not** map 1:1 into this repo, so they were not copied blindly.

What *does* map here:

- style-branch `mainline_train_prep.run_prep` -> this repo's `tasks/Conan/rhythm/preflight_support.py`
- style-branch `collect_control_diagnostics` -> this repo's `tasks/Conan/rhythm/metrics.py`
- local `@torch.jit.script` risk -> `modules/Conan/diff/net.py`

What does **not** exist here as a maintained rhythm feature:

- proxy-negative / style-success batch-composition logic from the style branch

So the safe rule for this repo is: improve the maintained rhythm chain where the local code has a real analogue, and do not invent cross-branch control logic that the rhythm path does not actually use.

## 3. What was absorbed into the local mainline

Recent local changes that matter for correctness and training prep:

- explicit zero-confidence now stays hard-off for component KD / retimed weighting instead of being revived by the confidence floor
- budget supervision now tracks projector repair more honestly and exposes repair / redistribution metrics
- preflight is split into context building, data-staging checks, control preview checks, and summary emission
- rhythm metrics are split into section collectors instead of growing one monolithic diagnostics function
- `scripts/preflight_rhythm_v2.py` now accepts both `--binary_data_dir` and `--processed_data_dir`
- `scripts/cpu_probe_rhythm_train.py` now allows up to 5000 probe steps, which enabled the 2000-step validation run
- `modules/Conan/diff/net.py` no longer depends on a local TorchScript helper for `silu`

## 4. Validation actually run on this checkout

The following checks were completed locally on April 4, 2026:

- `python -m unittest discover -s tests/rhythm -p "test_*.py"`
- `python -u scripts/smoke_test_rhythm_v2.py`
- `conda run -n conan python scripts/preflight_rhythm_v2.py --config egs/conan_emformer_rhythm_v2_teacher_offline.yaml --binary_data_dir data/binary/libritts_single_smoke_rhythm_v4 --processed_data_dir data/processed/libritts_local_real_smoke --splits train --inspect_items 2 --model_dry_run`
- `conda run -n conan python scripts/cpu_probe_rhythm_train.py --config egs/conan_emformer_rhythm_v2_teacher_offline.yaml --binary_data_dir data/binary/libritts_single_smoke_rhythm_v4 --processed_data_dir data/processed/libritts_local_real_smoke --steps 2000 --warmup_steps 10 --device cpu --profile_json artifacts/probe/teacher_offline_cpu_probe_2000.json`

Observed smoke/probe outcome:

- preflight on smoke assets passed
- CPU probe completed 2000 / 2000 steps
- `total_loss`: `0.3854 -> 0.1044`
- mean step time: about `240.71 ms`
- throughput: about `4.154 steps/s`
- peak CPU RSS: about `1159.91 MB`

These results mean the code path is structurally trainable in the local `conan` environment. They do **not** mean the formal maintained dataset is ready.

## 5. Remaining blockers before formal training

This shared checkout is still missing the real maintained training assets:

- `data/binary/vc_6layer/{train,valid}.{data,idx}`
- a real `data/processed/vc`
- exported `data/teacher_targets/...`
- retimed cache / side files for `student_retimed`
- F0 side files when `with_f0=true`

So the current state is:

- code readiness: **yes**
- smoke/probe readiness in `conda` env: **yes**
- formal maintained training readiness on this checkout: **no, blocked by data assets**

## 6. Training rule of thumb

Keep maintenance effort focused on changes that improve one of these four things:

- projector trust / feasible execution
- cache reproducibility / fail-fast validation
- retimed supervision consistency
- streaming stability / prefix consistency

If a proposed change only exists in another branch's research path and does not have a real local analogue, do not port it by default.
