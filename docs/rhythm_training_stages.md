# Rhythm Training Stages (2026-04-01)

See also:

- `docs/rhythm_module_vision.md`
- `docs/rhythm_supervision_policy.md`

## Stage 0: Structural smoke / integration

Goal:

- verify descriptor -> scheduler -> projector wiring
- verify state carry-over and committed-prefix behavior
- verify dataset fields and loss hooks are alive

Current state:

- mostly completed
- `scripts/smoke_test_rhythm_v2.py` now covers descriptor export and stateful scheduler reuse
- scheduler now also consumes a cheap internal source-boundary cue

---

## Stage 1: Reference-guided warm start

Goal:

- train the online rhythm path with cached source anchors and sampled reference rhythm conditioning
- let the student first learn stable budget / redistribution / projection behavior

Current supervision surface:

- `rhythm_speech_exec_tgt`
- `rhythm_pause_exec_tgt`
- `rhythm_speech_budget_tgt`
- `rhythm_pause_budget_tgt`
- `rhythm_guidance_speech_tgt`
- `rhythm_guidance_pause_tgt`

This stage is suitable for structural training, but it is not the final ceiling.

Current recommendation:

- prefer cached/offline targets over unconditional runtime heuristic regeneration
- use `rhythm_dataset_target_mode: cached_only` for formal experiments
- keep `prefer_cache` only as a migration / debug stage while refreshing caches

---

## Stage 2: Latency-matched teacher distillation

Goal:

- replace or supplement heuristic guidance with a stronger offline teacher
- distill onto the same public execution surface under streaming constraints

Important principle:

- do not distill unattainable full-context behavior directly
- distill a latency-matched surface that the student can actually realize

Reserved fields already exist:

- `rhythm_teacher_speech_exec_tgt`
- `rhythm_teacher_pause_exec_tgt`
- `rhythm_teacher_speech_budget_tgt`
- `rhythm_teacher_pause_budget_tgt`

Important terminology note:

- the repository currently has a stronger **offline teacher surface**
- it does **not** yet have a true full-context offline teacher model
- docs and experiments should keep that distinction explicit

---

## Stage 3: Retimed decoder training

Goal:

- reduce train/infer mismatch
- let the decoder actually learn on the retimed execution canvas

Current repository status:

- `rhythm_apply_train_override: false`
- `rhythm_apply_valid_override: false`
- `rhythm_apply_test_override: true`
- `rhythm_binarize_retimed_mel_targets: true`
- `rhythm_use_retimed_target_if_available: true`

This is intentional.
The project is still keeping train/valid on the source-aligned canvas until retimed target supervision is available.
Otherwise acoustic reconstruction would be shape-inconsistent with ground-truth mel.

Current bridge step already in repo:

- binarizer can cache a first-pass `rhythm_retimed_mel_tgt`
- cached retimed targets now also carry a per-frame confidence / weight surface
- cached retimed targets now also carry source identity / cache-contract metadata
- task code can switch mel reconstruction target to that cached retimed target when train-time rhythm rendering is enabled
- task code now resolves `rhythm_apply_mode` and retimed acoustic targets from the same flag, so train/test render and target selection no longer drift apart
- cached-only retimed training now fails fast if retimed cache is required but missing or mismatched
- retimed targets can now be aligned to decoder output either by resampling or by explicit length trimming without shape mismatch
- the minimal rhythm route can bypass the heavier local style/prosody adaptor and keep only global timbre conditioning
- the rhythm config now uses `mel_losses: "l1:1.0"` to stay aligned with the minimal `L_recon + L_plan` objective
- config now also exposes staged rollout knobs:
  - `rhythm_train_render_start_steps`
  - `rhythm_valid_render_start_steps`
  - `rhythm_retimed_target_start_steps`
- a staged experiment config is now provided at `egs/conan_emformer_rhythm_v2_retimed_train.yaml`
- a stricter cached-only warm-start config is now provided at `egs/conan_emformer_rhythm_v2_cached_only.yaml`

Recommended future config direction after retimed targets exist:

- enable train-time retimed rendering explicitly
- keep `L_recon` as the outer acoustic objective
- keep `rhythm_plan` as the main timing objective, with cumulative drift weighted above local shape

This is one of the biggest remaining blockers before claiming strong-rhythm closure.

---

## Stage 4: Streaming evaluation hardening

Need stronger evaluation around:

- pause placement quality
- local-rate transfer consistency
- prefix no-rollback stability
- long-utterance trace utilization
- chunkwise mel / wav continuity
- cold-start latency vs steady-state latency

---

## Practical status summary

As of 2026-04-01:

- the rhythm branch is **ready for warm-start / structural training**
- it is **not yet ready to claim final strong-rhythm performance**

The two biggest remaining milestones are:

1. a stronger offline teacher
2. decoder-side retimed training

## Current task focus

Right now the repository should focus on:

1. projector-centric timing supervision
2. cached-only reproducibility
3. retimed train/infer closure
4. streaming regression hardening

## Future expansion

After the current stage is stable, expand in this order:

1. stronger offline teacher / dual-mode distillation
2. richer rhythm-specific evaluation
3. progressive reference streaming
4. optional finer-grained micro-timing refinement
