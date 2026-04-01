# Rhythm Training Stages (2026-04-01)

See also:

- `docs/rhythm_module_vision.md`
- `docs/rhythm_supervision_policy.md`

## Stage 0: Structural smoke / integration

Goal:

- verify descriptor -> scheduler -> projector wiring
- verify state carry-over and committed-prefix behavior
- verify dataset fields and loss hooks are alive
- verify cache/config readiness before long runs

Current state:

- mostly completed
- `scripts/smoke_test_rhythm_v2.py` now covers descriptor export and stateful scheduler reuse
- scheduler now also consumes a cheap internal source-boundary cue
- unit frontend now exports `sealed_mask + boundary_confidence`
- a stateful run-length unitizer helper is now available for incremental debugging
- explicit blank-slot graph is now public in projector / renderer / loss
- renderer also exports `rhythm_blank_mask`, `rhythm_render_slot_index`, `rhythm_render_unit_index`
- `scripts/preflight_rhythm_v2.py` can now fail fast on cache/config mismatches before training starts
- preflight now also checks cached-only contract metadata, teacher/retimed surface identity, and whether `ConanDataset` filtering empties a split
- optional `--model_dry_run` also checks dataset collation + one no-grad ConanTask forward before a long run
- the bundled smoke cache may still need `--splits train` for structural checks; formal runs should pass both `train` and `valid`

---

## Stage 1: Reference-guided warm start

Goal:

- train the online rhythm path with cached source anchors and sampled reference rhythm conditioning
- let the student first learn stable budget / redistribution / projection behavior

Current supervision surface:

- `rhythm_speech_exec_tgt`
- `rhythm_pause_exec_tgt`
- `rhythm_blank_exec_tgt` (cache/internal compatibility alias)
- `rhythm_speech_budget_tgt`
- `rhythm_pause_budget_tgt`
- `rhythm_blank_budget_tgt` (cache/internal compatibility alias)
- `rhythm_guidance_speech_tgt`
- `rhythm_guidance_pause_tgt`
- `rhythm_guidance_blank_tgt`
- cached source phrase metadata (`source_boundary_cue`, `phrase_group_*`)
- cached slow-rhythm selector outputs (`slow_rhythm_memory`, `selector_meta_*`)

This stage is suitable for structural training, but it is not the final ceiling.

Current recommendation:

- prefer cached/offline targets over unconditional runtime heuristic regeneration
- use `rhythm_dataset_target_mode: cached_only` for formal experiments
- keep `prefer_cache` only as a migration / debug stage while refreshing caches
- for the first projector warm-start, prefer `egs/conan_emformer_rhythm_v2_schedule_only.yaml`
- that config now assumes cached teacher surfaces are already present and does not treat runtime teacher construction as the mainline
- stage-1 objective should stay minimal:
  - `L_exec_speech`
  - `L_exec_pause`
  - light `L_budget`
  - light `L_cumplan`

---

## Stage 2: Latency-matched teacher distillation

Goal:

- replace or supplement heuristic guidance with a stronger offline teacher
- distill onto the same public execution surface under streaming constraints

Important principle:

- do not distill unattainable full-context behavior directly
- distill a latency-matched surface that the student can actually realize
- prefer projector-space targets over planner-surface targets

Reserved fields already exist:

- `rhythm_teacher_speech_exec_tgt`
- `rhythm_teacher_pause_exec_tgt`
- `rhythm_teacher_blank_exec_tgt` (cache/internal compatibility alias)
- `rhythm_teacher_speech_budget_tgt`
- `rhythm_teacher_pause_budget_tgt`
- `rhythm_teacher_blank_budget_tgt` (cache/internal compatibility alias)
- `rhythm_teacher_prefix_clock_tgt`
- `rhythm_teacher_prefix_backlog_tgt`

Optional ablation-only field:

- `rhythm_teacher_allocation_tgt`

Important terminology note:

- the repository currently has a stronger **offline teacher surface**
- it now also has a **dual-mode schedule teacher skeleton**
  - streaming branch = stateful projector path
  - offline branch = full-horizon / no-prefix-reuse projector pass
  - algorithmic teacher = explicit schedule bootstrap
- `egs/conan_emformer_rhythm_v2_dual_mode_kd.yaml` is the formal stage-2 config
- formal stage-2 intentionally still inherits the stage-1 schedule-only scaffold (`rhythm_schedule_only_stage: true`, `rhythm_optimize_module_only: true`)
- it still does **not** yet have a true learned non-causal offline teacher model
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
- until dedicated retimed pitch targets exist, stage-3 disables source-aligned pitch supervision automatically to avoid shape-mismatch and train/infer drift
- the minimal rhythm route can disable the heavier local style/prosody adaptor and keep only global timbre conditioning
- the rhythm config now uses `mel_losses: "l1:1.0"` to stay aligned with the minimal executed-surface objective
- rhythm cache contract is now versioned at `rhythm_cache_version: 4`
- config now also exposes staged rollout knobs:
- `rhythm_train_render_start_steps`
- `rhythm_valid_render_start_steps`
- `rhythm_retimed_target_start_steps`
- a staged experiment config is now provided at `egs/conan_emformer_rhythm_v2_retimed_train.yaml`
- that stage now explicitly inherits `egs/conan_emformer_rhythm_v2_dual_mode_kd.yaml` and is the first formal joint stage that turns `rhythm_schedule_only_stage: false`
- a stricter cached-only warm-start config is now provided at `egs/conan_emformer_rhythm_v2_cached_only.yaml`

Recommended future config direction after retimed targets exist:

- enable train-time retimed rendering explicitly
- keep `L_base` as the outer acoustic objective
- keep the main timing path on executed speech/pause + light budget / cumulative-plan guardrail
- keep KD focused on executed speech/pause plus optional prefix carry
- treat `rhythm_plan` as an optional regression/ablation term instead of the default mainline objective
- treat scheduler outputs as debug/regression tensors, and treat projector execution as the real maintained contract

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
2. schedule-only warm start before joint acoustic finetune
3. cached-only reproducibility
4. dual-mode schedule KD on the same projector contract
5. retimed train/infer closure
6. streaming regression hardening

## Future expansion

After the current stage is stable, expand in this order:

1. stronger offline teacher / dual-mode distillation
2. richer rhythm-specific evaluation
3. progressive reference streaming
4. optional finer-grained micro-timing refinement
