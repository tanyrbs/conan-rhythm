# Rhythm Supervision Policy (2026-04-01)

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
- `rhythm_pause_exec_tgt` (legacy alias of the same executed blank surface)
- `rhythm_speech_budget_tgt`
- `rhythm_blank_budget_tgt`
- `rhythm_pause_budget_tgt` (legacy alias of the same blank-budget surface)
- `rhythm_guidance_speech_tgt`
- `rhythm_guidance_blank_tgt`
- `rhythm_guidance_pause_tgt` (legacy alias)

### Tier B: offline teacher surface

Used for:

- stronger phase-2 timing supervision
- sparser and smoother pause / redistribution targets

Current cached fields:

- `rhythm_teacher_speech_exec_tgt`
- `rhythm_teacher_blank_exec_tgt`
- `rhythm_teacher_pause_exec_tgt` (legacy alias)
- `rhythm_teacher_speech_budget_tgt`
- `rhythm_teacher_blank_budget_tgt`
- `rhythm_teacher_pause_budget_tgt` (legacy alias)
- `rhythm_teacher_prefix_clock_tgt`
- `rhythm_teacher_prefix_backlog_tgt`

Optional cached ablation fields may still exist:

- `rhythm_teacher_allocation_tgt`

Important distinction:

- current repo has a stronger **offline teacher surface**
- current repo does **not** yet have a true standalone full-context teacher model

Do not collapse those two claims.

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

Current cache version: `4`

---

## 4.1 Batch/schema layering

To keep the dataset sidecar from turning back into an unstructured grab-bag, read the sample/cache schema in three layers:

1. `runtime_minimal`
   - maintained timing contract used by the main rhythm path
   - `content_units`, `dur_anchor_src`, `ref_rhythm_stats`, `ref_rhythm_trace`

2. `debug_sidecar`
   - source phrase cues, offline prefix views, selector spans, streaming prefix counters
   - useful for inspection and regression, but not the maintained public control surface

3. `cache_audit_bundle`
   - cache version, hop/trace contract, confidence, retimed source metadata
   - should be validated before long runs and should fail fast in `cached_only`

This repo still materializes these fields into one training sample dict, but the maintained interpretation should stay layered.

---

## 5. Loss policy

The rhythm path should stay minimal.

Maintained naming preference:

- use **blank** for the executed non-speech slot surface in docs/configs
- keep **pause** only as a backward-compatible alias in cache/code paths

### Core timing losses

- `L_exec_speech`
- `L_exec_pause`
- small-weight `L_budget`
- small-weight `L_cumplan`

Interpretation:

- `L_exec_*` supervises executed timing directly
- `L_budget` keeps the streaming budget honest
- `L_cumplan` keeps cumulative prefix debt / backlog honest

Current practical weighting in config:

- `lambda_rhythm_budget = 0.10 ~ 0.25`
- `lambda_rhythm_cumplan = 0.10 ~ 0.20` (`lambda_rhythm_carry` remains a backward-compatible alias)

`L_plan` and `L_guidance` may still exist in code for ablations / migration, but they are not the maintained mainline objective.

### Optional staged losses

- `L_distill`
- `L_base`

Policy:

- `L_distill` should focus on the student-reachable executed surface
- default distill target = speech exec + blank exec
- prefix carry distill is optional and preferred over heavier allocation/budget distill
- allocation/budget distill should remain off by default unless the experiment explicitly needs them

---

## 6. Retimed decoder policy

Retimed acoustic training should only be considered formal when:

- retimed mel targets are cached offline
- cache source (`guidance` vs `teacher`) is explicit
- training no longer silently falls back to source-aligned mel when retimed render is required

This repository now enforces that distinction more strictly in cached-only retimed configs.

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
