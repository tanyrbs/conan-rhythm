# Rhythm Migration Plan (2026-04-04)

This is the only maintained document under `docs/`. Together with `README.md`, it defines the current training path, the April 4 audit outcome, and what is still blocked.

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
  - `scripts/integration_teacher_export_student_kd.py`

## 2. What the 6-agent training-prep audit actually covered

The April 4 audit was split across six concrete surfaces:

- parameter / contract defaults
- inference / runtime edge cases
- training losses / gradients / probe behavior
- data / cache / preflight assets
- docs / stage guidance
- optimizer scope / test coverage

That audit was used to update code, not just write commentary.

## 3. High-value fixes already absorbed into the local mainline

The maintained branch now includes these corrective changes:

- explicit zero-confidence stays hard-off for component KD and retimed confidence weighting
- factor intervention syncs compact edits back into `ref_rhythm_stats` / `ref_rhythm_trace` and drops stale sidecars
- budget supervision and feasibility-debt accounting are cleaner under repair-heavy cases
- preflight orchestration and rhythm diagnostics are split into smaller units instead of monoliths
- the local deprecated TorchScript helper in `modules/Conan/diff/net.py` is gone
- the acoustic runtime no longer touches pitch-embed branches when `use_pitch_embed=false`
- RMVPE / torchaudio loading is more lazy, so optional dependency failures happen only on the paths that really need them
- inference helpers tolerate `f0_denorm_pred=None` instead of indexing into a missing tensor
- integration export now covers `train/valid/test` by default for the teacher->student KD smoke chain
- stage-validation defaults now match runtime defaults for:
  - `rhythm_enable_learned_offline_teacher`
  - `rhythm_disable_pitch_loss_when_retimed`

## 4. Validation actually run on this checkout

The following were run locally in the `conda` `conan` environment:

- `conda run -n conan python -m compileall -q modules tasks scripts tests utils data_gen`
- `conda run -n conan python -m unittest discover -s tests/rhythm -p "test_*.py"`
- `conda run -n conan python -u scripts/smoke_test_rhythm_v2.py`
- teacher-offline smoke preflight / dry-run on `train`
- teacher export -> student KD integration smoke
- 2000-step CPU probe for `teacher_offline`
- 2000-step CPU probe for `student_kd`
- 2000-step CPU smoke probe for `student_retimed`

Observed result summary:

- unit coverage: **151 rhythm tests passed**
- teacher-offline probe: healthy loss descent and low gradient pressure
- student-KD probe: healthy and fast on smoke student binary
- student-retimed smoke: structurally runnable, but still high-risk because `L_base` dominates and gradients are large / clip-heavy

Important asset-level caveats from the same audit:

- `data/binary/libritts_single_smoke_rhythm_v4/valid.data` is empty, so teacher-offline smoke validation is trustworthy on `train` only
- the current teacher->student KD integration artifact under `artifacts/rhythm_teacher_export_student_kd/...` is smoke-only because its teacher ckpt mode is `bootstrap_random_init`
- formal stage-3 still needs real retimed cache plus real F0 side files; the smoke probe used `--hparams use_pitch_embed=False` to avoid claiming a false formal pass

## 5. Training-readiness conclusion by stage

### `teacher_offline`

- code path: **ready**
- smoke training path: **ready**
- formal run from this checkout: **blocked by real dataset assets**

### `student_kd`

- code path: **ready**
- teacher-export integration chain: **ready as smoke**
- formal run from this checkout: **blocked by missing real learned teacher export / rebuilt cache**

### `student_retimed`

- code path: **closer to ready**
- smoke path: **runs**
- formal run from this checkout: **not ready to bless**

Why stage-3 is still guarded:

- smoke requires `use_pitch_embed=False`
- `L_base` dominates the current smoke objective
- `grad_norm_before_clip` stays very high for long stretches
- the real F0 / retimed asset contract is not present in this shared checkout

## 6. Remaining blockers before formal training

This shared checkout is still missing the real maintained training assets:

- `data/binary/vc_6layer/{train,valid}.{data,idx}`
- a real `data/processed/vc`
- exported `data/teacher_targets/...`
- retimed cache / side files for `student_retimed`
- F0 side files when `with_f0=true`

So the current state is:

- code readiness: **yes**
- smoke/probe readiness in `conda` env: **yes**
- formal maintained training readiness on this checkout: **no**

The main blocker is still assets, not the control-loss hot path.

## 7. Remaining engineering gaps after this audit

The audit found a few remaining gaps that were intentionally not over-corrected in the same round:

- optimizer-scope coverage is better, but still benefits from more direct end-to-end param-group tests
- student-retimed needs a cleaner formal stage-3 integration smoke once real F0 assets exist
- some legacy / research-stage paths still exist outside the maintained mainline, even though the maintained docs now stop centering them

These are follow-up tasks, not blockers for merging the current cleanup.

## 8. Training rule of thumb

Keep maintenance effort focused on changes that improve one of these four things:

- projector trust / feasible execution
- cache reproducibility / fail-fast validation
- retimed supervision consistency
- streaming stability / prefix consistency

If a proposed change only exists in another branch's research path and does not have a real local analogue, do not port it by default.
