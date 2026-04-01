# Rhythm Training Stages (2026-04-01)

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
- use `rhythm_dataset_target_mode: prefer_cache`

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

---

## Stage 3: Retimed decoder training

Goal:

- reduce train/infer mismatch
- let the decoder actually learn on the retimed execution canvas

Recommended config direction:

- `rhythm_apply_mode: always`
- keep `L_recon` as the outer acoustic objective
- use `rhythm_plan` to penalize local timing error and prefix cumulative drift

This is one of the biggest remaining blockers before claiming strong-rhythm closure.

---

## Stage 4: Streaming evaluation hardening

Need stronger evaluation around:

- pause placement quality
- local-rate transfer consistency
- prefix no-rollback stability
- long-utterance trace utilization
- chunkwise mel / wav continuity

---

## Practical status summary

As of 2026-04-01:

- the rhythm branch is **ready for warm-start / structural training**
- it is **not yet ready to claim final strong-rhythm performance**

The two biggest remaining milestones are:

1. a stronger offline teacher
2. decoder-side retimed training
