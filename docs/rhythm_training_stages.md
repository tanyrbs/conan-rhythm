# Rhythm Training Stages (2026-04-02)

See also:

- `docs/rhythm_module_vision.md`
- `docs/rhythm_supervision_policy.md`
- `docs/rhythm_weak_factorization_mainline.md`

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
- `scripts/cpu_probe_rhythm_train.py` now provides a stricter CPU-side mini-train probe (real batch collation + forward/backward + grad/param checks) before committing to a longer run
- base config now exposes DataLoader / DDP efficiency knobs (`dl_pin_memory`, `dl_persistent_workers`, `dl_prefetch_factor`, `ddp_find_unused_parameters`, `ddp_static_graph`) instead of hard-coding them in trainer/task code
- the bundled smoke cache may still need `--splits train` for structural checks; formal runs should pass both `train` and `valid`
- projector state semantics now require monotonic committed-progress phase (`phase_ptr` no rollback on visible-prefix growth)
- projector pause/speech projection now keeps zero-budget branches differentiable, which removes the need for the temporary task-side pause surrogate used during earlier debugging
- strict maintained student-stage configs now set `rhythm_strict_mainline: true`, so `student_kd` / `student_retimed` (and the legacy `schedule_only` warm-start) hard-reject runtime teacher, guidance loss, and algorithmic-teacher branches; only cache-backed KD remains allowed on the maintained stage-2 config
- strict mainline also keeps projector on a thinner contract by default (`pause_selection_mode=simple`, `use_boundary_commit_guard=false`, `build_render_plan=false`), while preserving boundary-aware bias inside the simple pause path so strong pause placement does not get flattened
- runtime learned-offline teacher enable resolution now comes from the shared stage helper in `modules/Conan/rhythm/stages.py`; task validation, preflight, and runtime no longer each maintain their own approximation
- slot schedule / frame plan are now lazily materialized only when render / retimed closure actually needs them; strict non-render paths keep these fields absent
- cached-only loading now accepts compatible `rhythm_cache_version: 4` metadata when the maintained `v5` hop/trace/reference contract still matches; missing teacher/retimed source ids are inferred from cached surface names during load/preflight
- latest local checks passed for `py_compile`, `smoke_test`, `export_rhythm_teacher_targets.py --help`, real-data preflight dry-run on local `libritts_single_smoke_rhythm_v4` for `teacher_offline`, and a real-data CPU mini-train probe for `teacher_offline`
- maintained `student_kd` / `student_retimed` still correctly reject that local smoke cache until learned-offline teacher assets are exported and re-binarized into cache

---

## Stage 1: teacher_offline

Goal:

- train the learned offline planner teacher as a first-class model stage
- export stable `learned_offline` teacher `.npz` assets before the maintained student chain

Current recommendation:

- use `egs/conan_emformer_rhythm_v2_teacher_offline.yaml`
- keep this stage rhythm-only (`rhythm_teacher_only_stage: true`, no acoustic path)
- bootstrap the teacher from cached guidance/self targets, not from teacher KD
- export `{split}/{item_name}.teacher.npz` with `scripts/export_rhythm_teacher_targets.py` by default (`--flat_output` keeps the old flat layout; `export_offline_teacher_assets.py` remains a wrapper)
- point later student-stage binarization at that export directory via `rhythm_teacher_target_dir`

Important contract:

- this stage is **not** the maintained student mainline, so keep `rhythm_strict_mainline: false`
- it should keep `rhythm_primary_target_surface: guidance`
- it should keep runtime learned teacher enabled
- it should keep dual-mode KD disabled

---

## Legacy warm-start: schedule_only

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
- `egs/conan_emformer_rhythm_v2_schedule_only.yaml` remains available only for ablation / projector warm-start experiments
- that config assumes cached teacher surfaces are already present, but it is no longer the maintained mainline
- that config now also explicitly disables learned offline teacher runtime execution (`rhythm_enable_learned_offline_teacher: false`)
- it stays useful when you explicitly want a cheap projector warm-start before the maintained chain, but docs/configs should treat it as legacy
- this legacy module-only config also enables the acoustic fast path (`rhythm_fastpath_disable_acoustic_when_module_only: true`) and disables pitch/F0 requirements (`use_pitch_embed: false`, `binarization_args.with_f0: false`) so warm-start runs do not pay unnecessary decoder/pitch extraction cost
- the module-only acoustic fast path is automatically gated off once train-time retimed rendering is active; formal retimed closure should still run the real decoder path instead of a fake zero-cost shortcut
- the legacy warm-start objective should stay minimal:
  - `L_exec_speech`
  - `L_exec_pause`
  - light `L_budget`
  - light `L_prefix_state`
- no extra KD/guidance/proxy losses in the legacy warm-start default
- prefix-cropped training views should keep truncated tails open/unsealed for realistic streaming semantics

---

## Stage 2: student_kd

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
- `egs/conan_emformer_rhythm_v2_student_kd.yaml` is the maintained stage-2 cache-only KD config (`teacher_student_kd.yaml` remains a compatibility alias)
- `egs/conan_emformer_rhythm_v2_dual_mode_kd.yaml` is retained only as a legacy runtime-teacher branch config
- maintained stage-2 now distills purely from cached offline teacher surfaces; it does not keep a live runtime teacher branch in the training loop
- the runtime-teacher auxiliary supervision path (`lambda_rhythm_teacher_aux`) and detached confidence weighting stay only on the legacy `legacy_dual_mode_kd` research branch
- when streaming prefix cropping is active on that legacy branch, runtime teacher auxiliary supervision prefers full-length offline cached teacher sidecars (`rhythm_offline_teacher_*`) when they exist, and otherwise slices the runtime teacher execution/state surface to the overlapping prefix instead of mixing a full-context teacher with student-prefix targets
- stage-2 module-only runs also inherit the acoustic fast path and `with_f0: false`, so KD warm-start does not pay pitch/decoder cost unless you deliberately leave the rhythm-only route
- docs and experiments should still keep the distinction explicit: learned offline teacher is stronger than student replay, but it is still a planner teacher, not a second acoustic model
- maintained default training chain does **not** require this stage; prefer teacher-first target surfaces before adding KD losses

---

## Stage 3: student_retimed

Goal:

- reduce train/infer mismatch
- let the decoder actually learn on the retimed execution canvas

Current repository status:

- the default transitional main config still keeps:
  - `rhythm_apply_train_override: false`
  - `rhythm_apply_valid_override: false`
  - `rhythm_apply_test_override: true`
- the formal stage-3 config `egs/conan_emformer_rhythm_v2_student_retimed.yaml` now explicitly sets:
  - `rhythm_apply_train_override: true`
  - `rhythm_apply_valid_override: true`
  - `rhythm_strict_mainline: true`
  - `rhythm_train_render_start_steps: 0`
  - `rhythm_valid_render_start_steps: 0`
  - `rhythm_retimed_target_start_steps: 0`
  - `rhythm_online_retimed_target_start_steps: 0`
- `rhythm_binarize_retimed_mel_targets: true`
- `rhythm_use_retimed_target_if_available: true`
- stage-3 retimed joint config explicitly turns pitch/F0 back on (`use_pitch_embed: true`, `binarization_args.with_f0: true`) because this is the formal stage where retimed pitch supervision is allowed again

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
- rhythm cache contract is now versioned at `rhythm_cache_version: 5`
- smoke / preflight structure checks are passing after the projector-side pause differentiability fix; this is enough for train-ready staging semantics, but not yet evidence of long-run stability
- smoke preflight is now confirmed on `--splits train`; the bundled smoke cache still does not include a populated `valid` split, so full train+valid preflight remains a real-data check
- config now also exposes staged rollout knobs:
- `rhythm_train_render_start_steps`
- `rhythm_valid_render_start_steps`
- `rhythm_retimed_target_start_steps`
- the maintained stage-3 config is now provided at `egs/conan_emformer_rhythm_v2_student_retimed.yaml` (`retimed_train.yaml` remains a compatibility alias)
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
- the branch is structurally ready for preflight / smoke validation across `teacher_offline`, `student_kd`, and `student_retimed`
- it is **not yet ready to claim final strong-rhythm performance**

The two biggest remaining milestones are:

1. stronger empirical proof that the learned offline teacher beats bootstrap/student-replay baselines in real runs
2. stronger empirical proof that retimed decoder closure remains stable beyond smoke / dry-run coverage

## Current task focus

Right now the repository should focus on:

1. projector-centric timing supervision
2. teacher-first offline asset build followed by student-only KD / retimed closure
3. cached-only reproducibility
4. retimed train/infer closure
5. streaming regression hardening
6. legacy dual-mode schedule KD branch experiments only when explicitly needed (not default maintained path)

## Future expansion

After the current stage is stable, expand in this order:

1. stronger offline teacher / dual-mode distillation evidence
2. richer rhythm-specific evaluation
3. progressive reference streaming
4. optional finer-grained micro-timing refinement
