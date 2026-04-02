# Rhythm Training Stages (2026-04-02)

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
- Conan rhythm runtime is now split behind `modules/Conan/rhythm/runtime_adapter.py`, so `modules/Conan/Conan.py` keeps the acoustic path and delegates rhythm runtime orchestration
- scheduler now also consumes a cheap internal source-boundary cue
- unit frontend now exports `sealed_mask + boundary_confidence`
- a stateful run-length unitizer helper is now available for incremental debugging
- explicit blank-slot graph is now public in projector / renderer / loss
- renderer also exports `rhythm_blank_mask`, `rhythm_render_slot_index`, `rhythm_render_unit_index`
- `scripts/preflight_rhythm_v2.py` can now fail fast on cache/config mismatches before training starts
- preflight now also checks cached-only contract metadata, teacher/retimed surface identity, and whether `ConanDataset` filtering empties a split
- optional `--model_dry_run` also checks dataset collation + one no-grad ConanTask forward before a long run
- the bundled smoke cache may still need `--splits train` for structural checks; formal runs should pass both `train` and `valid`
- projector state semantics now require monotonic committed-progress phase (`phase_ptr` no rollback on visible-prefix growth)
- projector pause/speech projection now keeps zero-budget branches differentiable, which removes the need for the temporary task-side pause surrogate used during earlier debugging
- latest local train-ready checks passed on the smoke bundle for `py_compile`, `smoke_test`, train-only preflight dry-run, one-step `schedule_only`, one-step `dual_mode_kd`, and one-step `retimed_train`

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
- `rhythm_target_confidence`
- optional `rhythm_guidance_*` targets only when guidance ablations are active
- source phrase metadata / selector outputs should stay in debug sidecars, not the default runtime batch

This stage is suitable for structural training, but it is not the final ceiling.

Current recommendation:

- prefer cached/offline targets over unconditional runtime heuristic regeneration
- use `rhythm_dataset_target_mode: cached_only` for formal experiments
- keep `prefer_cache` only as a migration / debug stage while refreshing caches
- for the first projector warm-start, prefer `egs/conan_emformer_rhythm_v2_schedule_only.yaml`
- that config now assumes cached teacher surfaces are already present and does not treat runtime teacher construction as the mainline
- that config now also explicitly disables learned offline teacher runtime execution (`rhythm_enable_learned_offline_teacher: false`)
- stage-1 objective should stay minimal:
  - `L_exec_speech`
  - `L_exec_pause`
  - light `L_budget`
  - light `L_prefix_state`
- no extra KD/guidance/proxy losses in the maintained stage-1 default
- prefix-cropped training views should keep truncated tails open/unsealed for realistic streaming semantics

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

- the repository now has a learned **non-causal offline planner teacher**
- that teacher is now a stronger standalone planner branch with its own full-context temporal trunk, phrase pooling, global refine path, and confidence heads
- dual-mode schedule KD now means:
  - streaming branch = causal student scheduler + shared projector contract
  - offline branch = non-causal planner teacher + shared projector contract
  - algorithmic teacher = explicit bootstrap / fallback surface
- `egs/conan_emformer_rhythm_v2_dual_mode_kd.yaml` is an optional stage-2 branch config
- docs and experiments should still keep the distinction explicit: learned offline teacher is stronger than student replay, but it is still a planner teacher, not a second acoustic model
- maintained default training chain does **not** require this stage; prefer teacher-first target surfaces before adding KD losses

---

## Stage 3: Retimed decoder training

Goal:

- reduce train/infer mismatch
- let the decoder actually learn on the retimed execution canvas

Current repository status:

- the default transitional main config still keeps:
  - `rhythm_apply_train_override: false`
  - `rhythm_apply_valid_override: false`
  - `rhythm_apply_test_override: true`
- the formal stage-3 config `egs/conan_emformer_rhythm_v2_retimed_train.yaml` now explicitly sets:
  - `rhythm_apply_train_override: true`
  - `rhythm_apply_valid_override: true`
  - `rhythm_train_render_start_steps: 0`
  - `rhythm_valid_render_start_steps: 0`
  - `rhythm_retimed_target_start_steps: 0`
  - `rhythm_online_retimed_target_start_steps: 0`
- `rhythm_binarize_retimed_mel_targets: true`
- `rhythm_use_retimed_target_if_available: true`

Current bridge step already in repo:

- binarizer can cache a first-pass `rhythm_retimed_mel_tgt`
- cached retimed targets now also carry a per-frame confidence / weight surface
- cached retimed targets now also carry source identity / cache-contract metadata
- projector now emits the shared frame plan that renderer + online retimed supervision both consume
- task code now resolves `rhythm_apply_mode` and retimed acoustic targets after model forward, so current execution can drive `cached` / `online` / `hybrid` target selection
- cached-only retimed training now fails fast if retimed cache is required but missing or mismatched
- retimed targets can now be aligned to decoder output either by resampling or by explicit length trimming without shape mismatch
- online retimed bundle can now also build F0/UV targets from the same frame plan; if that path is disabled or unavailable, source-aligned pitch supervision is automatically gated off
- pause-boundary emphasis and projector feasible-debt regularization are now absorbed into maintained `L_exec_pause` / `L_budget`, rather than creating new optimizer loss names
- mel GAN should stay disabled on the retimed canvas unless real/fake targets are explicitly aligned to the same acoustic canvas
- the minimal rhythm route can disable the heavier local style/prosody adaptor and keep only global timbre conditioning
- the rhythm config now uses `mel_losses: "l1:1.0"` to stay aligned with the minimal executed-surface objective
- rhythm cache contract is now versioned at `rhythm_cache_version: 4`
- one-step schedule-only / retimed-joint checks are now passing after the projector-side pause differentiability fix; this is enough for train-ready status, but not yet evidence of long-run stability
- smoke preflight is now confirmed on `--splits train`; the bundled smoke cache still does not include a populated `valid` split, so full train+valid preflight remains a real-data check
- config now also exposes staged rollout knobs:
- `rhythm_train_render_start_steps`
- `rhythm_valid_render_start_steps`
- `rhythm_retimed_target_start_steps`
- a staged experiment config is now provided at `egs/conan_emformer_rhythm_v2_retimed_train.yaml`
- this is the maintained formal joint-training entry and does not require KD / learned offline runtime teacher to be enabled
- a stricter cached-only warm-start config is now provided at `egs/conan_emformer_rhythm_v2_cached_only.yaml`

Recommended future config direction after retimed targets exist:

- enable train-time retimed rendering explicitly
- keep `L_base` as the outer acoustic objective
- keep the main timing path on executed speech/pause + light budget / cumulative-plan guardrail
- keep optimizer view compact in joint mode: `L_base + L_rhythm_exec + L_stream_state (+ optional L_pitch)`
- keep KD focused on executed speech/pause plus optional prefix carry + tiny budget only when explicitly enabled
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

As of 2026-04-02:

- the rhythm branch is **ready for warm-start / structural training**
- the branch is also **train-ready for short validation runs** across `schedule_only`, optional `dual_mode_kd`, and `retimed_train`
- it is **not yet ready to claim final strong-rhythm performance**

The two biggest remaining milestones are:

1. stronger empirical proof that the learned offline teacher beats bootstrap/student-replay baselines in real runs
2. stronger empirical proof that retimed decoder closure remains stable beyond smoke / dry-run coverage

## Current task focus

Right now the repository should focus on:

1. projector-centric timing supervision
2. schedule-only warm start before joint acoustic finetune
3. cached-only reproducibility
4. retimed train/infer closure
5. streaming regression hardening
6. optional dual-mode schedule KD branch experiments (not default maintained path)

## Future expansion

After the current stage is stable, expand in this order:

1. stronger offline teacher / dual-mode distillation evidence
2. richer rhythm-specific evaluation
3. progressive reference streaming
4. optional finer-grained micro-timing refinement
