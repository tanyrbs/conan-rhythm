# Rhythm Supervision Policy (2026-04-02)

## 1. Purpose

Rhythm V2 is treating rhythm as a **time-axis control problem**, not as a side effect of style transfer.

The repository should therefore prefer:

- **offline cached rhythm supervision** for maintained experiments
- **runtime heuristic generation** only for debug or cache refresh fallback

---

## 2. Supervision tiers

### Tier A: guidance surface

Used for:

- scheduler / projector warm-start
- cheap reproducible reference-guided timing targets

Current cached fields:

- `rhythm_speech_exec_tgt`
- `rhythm_blank_exec_tgt`
- `rhythm_pause_exec_tgt` (maintained runtime/batch name)
- `rhythm_speech_budget_tgt`
- `rhythm_blank_budget_tgt`
- `rhythm_pause_budget_tgt` (maintained runtime/batch name)
- `rhythm_guidance_speech_tgt`
- `rhythm_guidance_blank_tgt`
- `rhythm_guidance_pause_tgt` (maintained runtime/batch name)

### Tier B: offline teacher surface

Used for:

- stronger phase-2 timing supervision
- sparser and smoother pause / redistribution targets

Current cached fields:

- `rhythm_teacher_speech_exec_tgt`
- `rhythm_teacher_blank_exec_tgt`
- `rhythm_teacher_pause_exec_tgt` (maintained runtime/batch name)
- `rhythm_teacher_speech_budget_tgt`
- `rhythm_teacher_blank_budget_tgt`
- `rhythm_teacher_pause_budget_tgt` (maintained runtime/batch name)
- `rhythm_teacher_prefix_clock_tgt`
- `rhythm_teacher_prefix_backlog_tgt`

Optional cached ablation fields may still exist:

- `rhythm_teacher_allocation_tgt`

Important distinction:

- current repo now has a stronger **standalone full-context offline planner teacher**
- that teacher still shares the public projector/execution contract with the student
- it is still **not** a second acoustic model, and maintained KD should stay on the shared execution surface rather than hidden-state imitation

Do not collapse those two claims.

Maintained usage preference:

- use teacher surfaces as the primary **target source** when teacher-first cached training is enabled
- maintained **student-stage** configs (`student_kd` / `student_retimed`) now mark this contract explicitly with `rhythm_strict_mainline: true`
- keep extra KD losses (`L_distill*`) optional, stage-specific, and off by default in the maintained chain
- do not keep task-side surrogate pause execution in the maintained path; if pause gradients disappear, fix the projector branch itself
- keep student-mainline runtime teacher branches off (`rhythm_enable_learned_offline_teacher: false` on `student_kd` / `student_retimed`); only `teacher_offline` and legacy dual-mode should pay runtime teacher complexity
- dual-mode KD may enable the runtime learned teacher branch, but it remains an optional research/stage-2 branch rather than the default maintained chain
- when streaming-prefix training exports a longer offline source sidecar, the runtime teacher should consume matching full-length offline cached teacher surfaces (`rhythm_offline_teacher_*`) for its own auxiliary supervision instead of reusing student-prefix-sliced teacher targets
- learned-offline teacher export should keep `rhythm_streaming_prefix_train: false` so exported `.npz` assets stay full-length and stable

---

## 3. Dataset modes

Recommended modes:

- migration / cache refresh: `rhythm_dataset_target_mode: prefer_cache`
- formal experiments: `rhythm_dataset_target_mode: cached_only`
- debug only: `rhythm_dataset_target_mode: runtime_only`

Interpretation:

- `prefer_cache` is transitional
- `cached_only` is the reproducible maintained path
- `runtime_only` should not back serious reported results

Current cache semantics note:

- current cached guidance / teacher targets are still **self-conditioned offline surfaces**
- the repo therefore defaults `rhythm_cached_reference_policy: self` when cached targets are used
- pairwise cached reference-conditioning is future work

---

## 4. Cache contract

Current maintained cache contract includes:

- source unit cache
  - `content_units`
  - `dur_anchor_src`
  - `open_run_mask`
  - `sep_hint`
- reference rhythm cache
  - `ref_rhythm_stats`
  - `ref_rhythm_trace`
- cache metadata
  - `rhythm_cache_version`
  - `rhythm_unit_hop_ms`
  - `rhythm_trace_hop_ms`
  - `rhythm_trace_bins`
  - `rhythm_trace_horizon`
  - `rhythm_reference_mode_id`
- confidence
  - `rhythm_target_confidence`
  - `rhythm_guidance_confidence`
  - `rhythm_teacher_confidence` when teacher cache exists
- surface identity
  - `rhythm_guidance_surface_name`
  - `rhythm_teacher_surface_name` when teacher cache exists
- retimed target metadata when retimed cache exists
  - `rhythm_retimed_target_source_id`
  - `rhythm_retimed_target_surface_name`
  - `rhythm_retimed_target_confidence`

Practical rule:

- if this contract changes, the dataset should be re-binarized
- `cached_only` should reject stale or mismatched cache contracts
- source-side cropped prefixes may be deterministically rebuilt from visible tokens
- cached target surfaces may be prefix-adapted and retimed targets rebuilt for cropped views
- cropped prefix adaptation should keep truncated tail units `open/unsealed` (instead of forcing all-visible units sealed)

Current maintained cache version: `5`

Compatibility note:

- loader/preflight now accept compatible `v4` cached rhythm metadata when the maintained hop/trace/reference contract still matches
- missing `rhythm_teacher_target_source_id` / `rhythm_retimed_target_source_id` are backfilled from cached surface names during loading
- this compatibility bridge is only a migration path; new caches should still be re-binarized as `v5`

---

## 4.1 Batch/schema layering

To keep the dataset sidecar from turning back into an unstructured grab-bag, read the sample/cache schema in four layers:

1. `runtime_minimal`
   - maintained timing contract used by the main rhythm path
   - `content_units`, `dur_anchor_src`, `ref_rhythm_stats`, `ref_rhythm_trace`

2. `runtime_targets`
   - executed speech/pause targets, light budget targets, stage-needed confidence surfaces
   - cached teacher / retimed targets should only be exported into the batch when the active stage actually consumes them

3. `streaming_offline_sidecar`
   - offline source-cache views and streaming prefix counters
   - required for prefix-cropped train-time streaming experiments and learned offline teacher runs, but not for the runtime-minimal path

4. `debug_cache_appendix`
   - cache version, hop/trace contract, confidence, retimed source metadata
   - source phrase cues, selector spans, slow-memory cells
   - should be validated before long runs and should fail fast in `cached_only`

The maintained batch path should default to:

- runtime-minimal
- runtime-targets needed by the active stage
- streaming/offline sidecars only when dual-mode / prefix sampling needs them

Everything else should stay opt-in.

State semantics note:

- `phase_ptr` is a committed-progress state and must be monotonic
- visible-prefix growth without new commit should not move `phase_ptr` backward
- trace-window sampling should follow committed progress, not fluctuating visible-prefix ratios
- projector should remain a thin contract layer on the strict mainline:
  - keep feasible-budget lifting
  - default to simple pause allocation, but keep boundary-aware bias active inside the simple path
  - disable boundary commit guard unless a research branch explicitly needs it
  - do not eagerly build slot/frame render artifacts on non-render stages

---

## 5. Loss policy

The rhythm path should stay minimal.

Maintained naming preference:

- use **pause** for executed non-speech target surfaces in configs / runtime batches
- keep **blank** only as a renderer / cache-compat alias where the interleaved blank-slot graph is helpful

### Core timing losses

- `L_exec_speech`
- `L_exec_pause`
- small-weight `L_budget`
- small-weight `L_prefix_state` (`L_cumplan` remains the compatibility alias)

Interpretation:

- `L_exec_*` supervises executed timing directly
- `L_budget` keeps the streaming budget honest
- `L_prefix_state` keeps cumulative prefix debt / backlog honest

Current practical weighting in config:

- `lambda_rhythm_budget = 0.10 ~ 0.25`
- `lambda_rhythm_cumplan = 0.10 ~ 0.20` (`lambda_rhythm_carry` remains a backward-compatible alias)

`L_plan` and `L_guidance` may still exist in code for ablations / migration, but they are not the maintained mainline objective.

Maintained optimizer policy:

- `teacher_offline` / `legacy_schedule_only`: optimize exactly the 4 timing losses above
- `joint retimed`: optimize compact `3+1` objectives
  - required 3: `L_base`, `L_rhythm_exec`, `L_stream_state`
  - optional +1: `L_pitch` (only when retimed pitch targets are enabled/ready)
- all other rhythm detail terms should be treated as logging/ablation surfaces unless a stage explicitly enables them
- pause-boundary emphasis must stay absorbed inside `L_exec_pause`
- projector feasibility debt must stay absorbed inside `L_budget`
- do not grow either of those into standalone optimizer losses on the maintained path
- offline-teacher confidence heads are not directly supervised on the maintained path, so they stay out of the optimizer by default; only opt in with `rhythm_train_offline_confidence_heads: true` if a real confidence objective is added

### Optional staged losses

- `L_distill`
- `L_distill_exec`
- `L_distill_budget`
- `L_distill_prefix`
- `L_base`

Policy:

- `L_distill` should focus on the student-reachable executed surface
- maintained distill breakdown = executed speech/pause + optional prefix carry + tiny budget term
- default distill target = speech exec + pause exec (`blank_exec` only as cache/internal alias)
- prefix carry distill is preferred over heavier allocation/budget distill
- allocation/budget distill should remain off by default unless the experiment explicitly needs them
- when the legacy runtime learned offline teacher KD branch is enabled, teacher confidence weights must be detached before weighting student losses, and the runtime teacher itself should receive a light direct auxiliary supervision term instead of being left purely as a detached target producer
- that legacy runtime teacher auxiliary supervision should prefer dedicated full-length offline teacher sidecars when available; otherwise it must slice the runtime teacher surface to the overlapping prefix before comparing against cached student-prefix targets
- maintained stage configs now keep module-only schedule/KD runs on the acoustic fast path (`rhythm_fastpath_disable_acoustic_when_module_only: true`) and disable F0 extraction there (`use_pitch_embed: false`, `binarization_args.with_f0: false`); formal retimed joint training re-enables pitch/F0 in stage-3
- the acoustic fast path must not stay enabled after train-time retimed rendering turns on; once stage-3 closure is active, decoder/pitch/vocoder supervision has to run on the real retimed canvas

Maintained default:

- keep `L_distill*` disabled in the default teacher-first stage; when enabled in the maintained chain, use the explicit cache-only path (`teacher_offline -> student_kd -> student_retimed`)
- if KD is enabled, keep it as an explicit branch experiment rather than a hidden always-on objective
- strict-mainline fast path now builds rhythm loss targets directly from the primary cached surface when guidance/distill are both disabled, instead of routing through runtime teacher / algorithmic teacher branches

Maintained chain note:

- `egs/conan_emformer_rhythm_v2_teacher_offline.yaml` is the offline teacher asset-build entry (runtime learned teacher enabled, no student KD, no acoustic path)
- `egs/conan_emformer_rhythm_v2_schedule_only.yaml` is a legacy warm-start / ablation entry (runtime learned teacher disabled)
- `egs/conan_emformer_rhythm_v2_student_kd.yaml` is the maintained stage-2 branch (teacher fully offline, distillation kept on cached execution/prefix surfaces; `teacher_student_kd.yaml` is an alias)
- `egs/conan_emformer_rhythm_v2_dual_mode_kd.yaml` is retained only as an optional legacy runtime-teacher branch
- `egs/conan_emformer_rhythm_v2_student_retimed.yaml` is the stage-3 formal joint entry (`retimed_train.yaml` is an alias)
- the formal stage-3 config now enables retimed train/valid closure from step 0; the older delayed-start behavior remains only on transitional configs

---

## 6. Retimed decoder policy

Retimed acoustic training should only be considered formal when:

- projector execution emits the binding frame plan
- renderer and retimed supervision read the same frame plan
- cache source (`guidance` vs `teacher`) is explicit when cached retimed targets are used
- train-time target selection (`cached` / `online` / `hybrid`) is explicit
- training no longer silently falls back to source-aligned mel when retimed render is required

This repository now enforces that distinction more strictly:

- cached retimed mel remains the first-pass / warm-start path
- online retimed mel/F0/UV can now be built from the current execution frame plan
- retimed-stage mel GAN is expected to stay off unless real/fake canvases are explicitly aligned

---

## 7. Failure modes this policy is trying to prevent

1. **label drift**
   - runtime heuristic code changes silently change the target

2. **false progress**
   - the output sounds different, but timing is not actually controlled

3. **style/rhythm entanglement**
   - the model falls back to "reference-like feel" instead of explicit timing control

4. **cache contract drift**
   - old cache assets are reused under incompatible trace or retimed settings

---

## 8. Preferred closed loop

For streaming strong-rhythm transfer, the preferred closed loop is:

```text
offline cached supervision
  -> stateful monotonic scheduler
  -> single projector
  -> retimed decoder training
  -> duration / pause / prefix-drift evaluation
```

That is the path the repository should keep hardening.
