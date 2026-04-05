# Rhythm Migration Plan (2026-04-05)

This is the only maintained document under `docs/`. Together with `README.md`, it defines the current training path, the April 4-5 audit outcome, and what is still blocked.

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

The April 4-5 audit was split across six concrete surfaces:

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
- online retimed acoustic targets now inherit sample confidence and projector-repair gating instead of being implicitly trusted at weight `1.0`
- module-only stages now keep train/valid objectives aligned instead of reintroducing acoustic/pitch losses only during validation
- `teacher_offline` validation now uses the real offline-teacher branch instead of silently falling back to the student/runtime path
- `teacher_offline` preflight dry-run now distinguishes:
  - `teacher_only` / module-only runtime, where `mel_out` is optional
  - `teacher_as_main` runtime, where `mel_out` is still required for audition / export semantics
- preflight orchestration and rhythm diagnostics are split into smaller units instead of monoliths
- preflight can now escalate missing or placeholder processed paths with `--strict_processed_data_dir`
- the local deprecated TorchScript helper in `modules/Conan/diff/net.py` is gone
- the acoustic runtime no longer touches pitch-embed branches when `use_pitch_embed=false`
- flow components in `modules/Conan/Conan.py` now lazy-load `torchdyn`, so non-flow rhythm checks do not fail at import time
- RMVPE / torchaudio loading is more lazy, so optional dependency failures happen only on the paths that really need them
- rhythm-only imports no longer require the English text frontend or tensorboard at import time
- inference helpers tolerate `f0_denorm_pred=None` instead of indexing into a missing tensor
- Conan / Emformer export paths also tolerate missing source or predicted F0 instead of indexing blindly
- weighted retimed acoustic losses now normalize by the full broadcasted weight mass instead of frame count only
- retimed mel supervision now disables pitch loss when matched `retimed_f0_tgt` / `retimed_uv_tgt` are missing, unless source-axis fallback is explicitly opted in
- retimed mel / weight alignment now keeps `retimed_f0_tgt`, `retimed_uv_tgt`, and `tgt_nonpadding` on the same time axis as `mel_out`, with linear resize for `retimed_f0_tgt`, nearest / binary-preserving resize for `retimed_uv_tgt`, and trim-mode parity with the aligned mel target
- integration export now covers `train/valid/test` by default for the teacher->student KD smoke chain
- stage-validation defaults now match runtime defaults for:
  - `rhythm_enable_learned_offline_teacher`
  - `rhythm_disable_pitch_loss_when_retimed`
- frame-plan construction now uses masked sum-preserving rounding for speech slots, blank slots, and `dur_anchor_src`, so integerization no longer leaks or drops total frame mass on fractional edge cases
- projector outer hot paths are more vectorized in:
  - `_allocate_pause_budget`
  - `_project_pause_impl`
  - `_compute_commit_frontier`
  - `_advance_state`
- stage-specific optimizer scope no longer silently falls back to full-model training:
  - empty collectors now fail fast
  - post-freeze zero-trainable states now fail fast
  - foreign / stale params from the wrong model instance now fail fast
- `content_lengths` is now propagated explicitly through the rhythm dataset collater, runtime forward kwargs, and streaming evaluation instead of quietly reusing `mel_lengths`
- a conservative group-level EMA rhythm-loss balancer now exists behind an explicit opt-in flag:
  - `rhythm_loss_balance_mode: ema_group`
  - maintained default stays `none`
- context-matched KD gating now exists as an opt-in stage-2 research path:
  - `rhythm_enable_distill_context_match`
  - `rhythm_distill_context_floor`
  - `rhythm_distill_context_power`
  - `rhythm_distill_context_open_run_penalty`
- new experimental configs were added without changing the maintained default chain:
  - `egs/conan_emformer_rhythm_v2_student_kd_context_match.yaml`
  - `egs/conan_emformer_rhythm_v2_student_retimed_balanced.yaml`

## 4. Validation actually run on this checkout

The following were run locally in the `conda` `conan` environment:

- `conda run -n conan python -m compileall -q modules tasks scripts tests utils data_gen`
- `conda run -n conan python -m unittest discover -s tests/rhythm -p "test_*.py"`
- `conda run -n conan python -u scripts/smoke_test_rhythm_v2.py`
- teacher-offline smoke preflight / dry-run on `train`
- teacher-offline smoke preflight / dry-run on `train valid` to confirm the known empty-split failure
- teacher export -> student KD integration smoke
- 1-step `student_retimed` CPU probe with default `use_pitch_embed=true`
- 1-step `student_retimed` CPU probe with `--hparams use_pitch_embed=False`
- 2-step `student_retimed` CPU probe after probe-observability wiring to confirm runtime summaries are exported
- 2000-step CPU probe for `teacher_offline`
- 2000-step CPU probe for `student_kd`
- 2000-step CPU smoke probe for `student_retimed`

Observed result summary:

- unit coverage: **206 rhythm tests passed**
- latest compile/unit/smoke rerun after the teacher-offline preflight fix, online-retimed confidence/repair gating, and stage-freeze fail-fast guards: **passed**
- teacher-offline probe: healthy loss descent and low gradient pressure
- student-KD probe: healthy and fast on smoke student binary
- student-retimed smoke: structurally runnable, but still high-risk because `L_base` dominates and gradients are large / clip-heavy
- fresh stage-3 spot-check: default `student_retimed` probe fails immediately on missing `f0` in the smoke `student_binary`
- fresh stage-3 spot-check with `use_pitch_embed=False`: runs, but still shows large one-step gradient pressure (`grad_norm_before_clip ~= 274`)
- CPU probe now also exports runtime observability summaries for:
  - `rhythm_metric_disable_acoustic_train_path`
  - `rhythm_metric_module_only_objective`
  - `rhythm_metric_skip_acoustic_objective`
  - `rhythm_metric_pitch_supervision_disabled`
  - `rhythm_metric_missing_retimed_pitch_target`
  - pre-align retimed-target mismatch / resample / trim signals

The newest rerun extended regression coverage specifically around:

- preserved-sum frame-plan rounding
- projector invariants after hot-path vectorization
- target-builder context-match gating / dedupe semantics
- conservative rhythm-loss balancing
- module-only train/valid objective alignment
- teacher-offline validation routing
- teacher-offline preflight semantics for `teacher_only` vs `teacher_as_main`
- optional dependency guards for text frontend / tensorboard imports
- optional dependency guards for `torchdyn` on non-flow rhythm paths
- weighted acoustic-loss normalization over broadcasted weight mass
- retimed pitch fallback suppression, missing-target detection, and post-align length parity with `mel_out`
- online retimed target confidence / repair gating
- stage-specific parameter freeze guards and foreign-param detection
- validation / smoke observability for skipped acoustic objectives and missing matched retimed pitch targets

Important asset-level caveats from the same audit:

- `data/binary/libritts_single_smoke_rhythm_v4` is train-only in practice:
  - `valid.data` is empty
  - `test.data` is empty
- the checked-in LibriTTS smoke binary is rhythm-cache **v4 compatibility smoke**, not maintained v5 training data
- the current teacher->student KD integration artifact under `artifacts/rhythm_teacher_export_student_kd/...` is smoke-only because its teacher ckpt mode is `bootstrap_random_init`
- the generated stage-2 smoke `student_binary` already contains learned-offline teacher + retimed targets, but it still does **not** include F0
- formal stage-3 still needs real F0 side files; the smoke probe used `--hparams use_pitch_embed=False` only to avoid claiming a false formal pass

## 4.1 Training-prep audit notes that matter before you launch

- `scripts/preflight_rhythm_v2.py` is only a thin wrapper; the real validation lives in `tasks/Conan/rhythm/preflight_support.py`
- preflight is **binary-cache-first**, not a full processed-corpus validator:
  - it strongly checks indexed cache fields / contracts
  - by default it still only lightly checks `processed_data_dir`
  - for formal readiness runs, `--strict_processed_data_dir` now escalates missing or placeholder processed paths into hard errors
- `teacher_offline` preflight now intentionally treats runtime semantics, not just stage name, as the source of truth:
  - `teacher_only` / module-only dry-runs may omit `mel_out`
  - `teacher_as_main` dry-runs must still emit `mel_out`, because that path is also used for teacher audition / export
- `scripts/cpu_probe_rhythm_train.py` is a throughput / gradient probe, not a full cache-contract validator
- `scripts/integration_teacher_export_student_kd.py` exports `train/valid/test`, but the post-export smoke assertions are still centered on `train/valid`
- on the checked-in LibriTTS smoke corpus, integration split inference relies on the generated `build_summary.json`

## 5. Training-readiness conclusion by stage

### `teacher_offline`

- code path: **ready**
- smoke training path: **ready**
- formal run from this checkout: **blocked by real dataset assets**

### `student_kd`

- code path: **ready**
- teacher-export integration chain: **ready as smoke**
- formal run from this checkout: **blocked by missing real learned teacher export / rebuilt cache**
- experimental branch available: `conan_emformer_rhythm_v2_student_kd_context_match.yaml`
- maintained KD reminder: the real maintained signal is `teacher-main + shape-only KD`; exec/budget/prefix/allocation KD stay disabled in the default stage-2 config

### `student_retimed`

- code path: **closer to ready**
- smoke path: **runs**
- formal run from this checkout: **not ready to bless**
- experimental branch available: `conan_emformer_rhythm_v2_student_retimed_balanced.yaml`

Why stage-3 is still guarded:

- smoke requires `use_pitch_embed=False`
- the smoke binary already has retimed targets, but default stage-3 still fails because it lacks F0
- retimed mel no longer silently falls back to source-axis pitch; missing matched retimed pitch now disables pitch supervision first, and aligned retimed pitch tracks are kept on the same time axis as `mel_out`
- `L_base` dominates the current smoke objective
- `grad_norm_before_clip` stays very high for long stretches
- the real F0 / retimed asset contract is not present in this shared checkout

Concrete 2000-step smoke numbers from `artifacts/probe/student_retimed_cpu_probe_2000_smoke.json`:

- `L_base.mean ~= 6.5258`
- `L_rhythm_exec.mean ~= 0.00279`
- `L_stream_state.mean ~= 0.000746`
- `L_pitch.mean = 0.0` because that smoke probe had to force `use_pitch_embed=False`
- `grad_norm_before_clip.mean ~= 158.86`
- `grad_norm_before_clip.max ~= 275.92`

That is roughly:

- `L_base / L_rhythm_exec ~= 2342x`
- `L_base / L_stream_state ~= 8745x`

So the current stage-3 smoke result is not "mildly imbalanced"; it is still overwhelmingly acoustic-dominated.

The new probe-observability wiring was also validated on a fresh 2-step `student_retimed` CPU probe:

- `rhythm_metric_pitch_supervision_disabled = 1.0`
- `rhythm_metric_skip_acoustic_objective = 0.0`
- `rhythm_metric_acoustic_target_is_retimed = 1.0`

So the probe can now distinguish "pitch was intentionally disabled because matched retimed pitch is unavailable" from "the whole acoustic objective got skipped".

## 6. Remaining blockers before formal training

This shared checkout is still missing the real maintained training assets:

- `data/binary/vc_6layer/{train,valid}.{data,idx}`
- a real `data/processed/vc`
- exported `data/teacher_targets/...`
- dedicated formal retimed cache / side files for `student_retimed`
- F0 side files when `with_f0=true`

So the current state is:

- code readiness: **yes**
- smoke/probe readiness in `conda` env: **yes**
- formal maintained training readiness on this checkout: **no**

The main blocker is still assets, not the control-loss hot path.

## 6.1 Gradient / control audit notes

- maintained optimization is still centered on the composite rhythm losses:
  - execution
  - budget
  - prefix-state / carry
- many logged subterms are reporting surfaces after scaling, not independent optimizer-driving objectives
- projector / frame-plan behavior is intentionally discrete:
  - rounding
  - top-k / thresholding
  - monotonic commit frontiers
  - detached prefix reuse
- so retimed acoustic targets do **not** backprop through frame-plan/control decisions; the explicit rhythm losses remain the control-learning path

## 7. Remaining engineering gaps after this audit

The audit found a few remaining gaps that were intentionally not over-corrected in the same round:

- optimizer-scope coverage is now guarded against empty / foreign collector output, but it still benefits from more direct end-to-end param-group tests under real checkpoints
- student-retimed needs a cleaner formal stage-3 integration smoke once real F0 assets exist
- some legacy / research-stage paths still exist outside the maintained mainline, even though the maintained docs now stop centering them
- projector still keeps the row-wise bounded-simplex core as the conservative speech-path solver; only the outer hot paths were vectorized in this round
- loss balancing and context-matched KD are deliberately opt-in research knobs, not new maintained defaults

These are follow-up tasks, not blockers for merging the current cleanup.

## 8. Training rule of thumb

Keep maintenance effort focused on changes that improve one of these four things:

- projector trust / feasible execution
- cache reproducibility / fail-fast validation
- retimed supervision consistency
- streaming stability / prefix consistency

If a proposed change only exists in another branch's research path and does not have a real local analogue, do not port it by default.
