# Rhythm Migration Plan (2026-04-08)

This file is the **concise architecture / migration note**.
Operational launch details live in `docs/autodl_training_handoff.md`.
Historical probe transcripts, large ablation menus, and outdated branch diaries
were intentionally removed.

## 1. Mainline decision

The maintained project path is now explicitly **teacher-first**:

1. train the offline teacher into a stable rhythm oracle;
2. verify that teacher through teacher-conditioned audio closure and export;
3. distill the student from those teacher assets;
4. close student acoustic behavior in `S3-retimed`.

The key judgment is architectural, not cosmetic:

- the teacher in this repo is a **non-causal offline teacher**
- it is naturally an oracle / asset producer / supervision source
- it is **not** naturally the final deployment model

## 2. Current implementation truth

The maintained implementation is best described as:

> **source-side boundary / commit evidence -> explicit speech/pause budget planner -> projector execution authority -> renderer**

Reference conditioning remains useful, but its role is narrower:

- global rhythm prior
- phrase-style prior
- optional compatibility / observability path

`phase_ptr`, local trace sampling, trace reliability, cold-start gating, and
active-tail sampling still exist in the codebase, but the maintained controller
is now **phase-decoupled timing** as a canonical direction: phase remains
observer/telemetry plus compatibility support, not the runtime control
authority, while compatibility-default configs may still require explicit
`rhythm_phase_decoupled_timing: true`.

Two clarifications matter for the current code, not just the paper story:

- the maintained runtime now uses **typed junction semantics**
  (`JOIN / WEAK / PHRASE`)
- phrase-bank retrieval is currently driven by the persisted runtime pointer
  `ref_phrase_ptr`, which advances with committed-frontier progress

So the practical controller is better read as:

> **committed structure -> phrase/global reference prior -> explicit timing plan -> projector**

not as "every visible lexical boundary advances phrase retrieval".

The runtime meaning should now be read more narrowly:

- this stack is a **reference-conditioned temporal pacing / timing controller**
- its maintained control surface is:
  - speech duration
  - pause placement / pause amount
  - boundary-related local lengthening hints / side signals
- full accent / full F0 contour / expressive intonation remain downstream style
  realization concerns

Boundary semantics are also typed rather than binary:

- `JOIN`
- `WEAK`
- `PHRASE`

Current execution semantics are stricter than a plain label view:

- `JOIN` = no pause insertion
- `WEAK` = pause capped by `rhythm_projector_weak_boundary_pause_cap`
- `PHRASE` = main carrier for redistributed pause mass in the editable tail

A stricter PHRASE-only phrase-pointer update is a possible future contract
change, not current checkout truth.

Reference is now best described as three-layer conditioning:

- global rhythm prior
- phrase-level prior selected by committed structure
- bounded local residual style cue

not as a continuous wall-clock script.

## 2.1 Canonical naming and interface notes

Canonical naming:

- `rhythm_phase_decoupled_timing`
- `rhythm_phase_decoupled_phrase_gate_boundary_threshold`
- `rhythm_phase_decoupled_boundary_style_residual_scale`
- `rhythm_debt_control_scale`
- `rhythm_debt_pause_priority`
- `rhythm_debt_speech_priority`
- `rhythm_projector_debt_leak`
- `rhythm_projector_debt_max_abs`
- `rhythm_projector_debt_correction_horizon`
- `rhythm_runtime_phrase_bank_enable`
- `rhythm_runtime_phrase_select_window`
- `rhythm_runtime_phrase_neighbor_mix_alpha`
- `rhythm_weak_boundary_threshold`
- `rhythm_phrase_boundary_threshold`
- `rhythm_boundary_lengthening_max`
- `rhythm_projector_weak_boundary_pause_cap`

Deprecated compatibility aliases:

- `rhythm_phase_free_timing`
- `rhythm_phase_free_phrase_boundary_threshold`
- `rhythm_phase_decoupled_phrase_boundary_threshold`

Public runtime interface:

- `ConanRhythmAdapter.forward` = maintained public runtime entry
- `run_rhythm_frontend()` = lower-level integration/test helper

Teacher runtime semantics:

- `rhythm_teacher_as_main` = offline/full-commit learned-teacher execution path for
  teacher-only audit/export style stages
- `rhythm_enable_dual_mode_teacher` = training-time paired streaming + offline teacher path
- canonical teacher projector knobs:
  - `rhythm_teacher_projector_force_full_commit`
  - `rhythm_teacher_projector_soft_pause_selection`

These teacher branches are not the deployment runtime replacement and are not
activated during `infer=true`.

Reference sidecar terminology:

- `rhythm_emit_reference_sidecar`
  - module/runtime conditioning auxiliaries
  - covers generic slow-memory and planner-side summaries
- `rhythm_export_debug_sidecars`
  - dataset/sample/cache audit export fields

`rhythm_emit_reference_sidecar` may be auto-enabled when the config requests
external sampled references, `sample_ref` cached reference policy,
descriptor-bootstrap losses, or trace-reliability fallback.

Phrase-bank conditioning is adjacent but separately materializable:

- it may be emitted with sidecars
- it may be enabled by `rhythm_runtime_phrase_bank_enable`
- it may be consumed directly when `ref_phrase_*` is already present
- if external sidecars drift from the rebuilt compact/raw contract,
  `build_reference_conditioning()` now drops those stale sidecars and rebuilds
  from the raw cached reference contract

Runtime override rule of thumb:

- supported runtime overrides are the trace window / phase-decoupled switch /
  source-boundary scaling / streaming debt shaping / projector sparsity /
  projector debt knobs
- teacher-only runtime intentionally does **not** expose the full streaming
  scheduler override surface

Planner compact contract:

- `planner_ref_stats [B,2]`
- `planner_ref_trace [B,bins,2]`
- `planner_slow_rhythm_memory [B,K,2]`
- `planner_slow_rhythm_summary [B,2]`
- `planner_ref_phrase_trace [B,P,bins,2]`

## 3. Maintained stage naming

To stop documentation drift, the active stage map is intentionally small:

| Stage | Meaning | Current implementation |
|---|---|---|
| `T1-surface` | stable offline teacher surface | `egs/conan_emformer_rhythm_v2_teacher_offline.yaml` |
| `T2-audio` | teacher-conditioned audio closure, audit, and export | maintained as a procedure centered on `scripts/export_rhythm_teacher_targets.py` |
| `S2-kd` | conservative teacher-conditioned student KD | `egs/conan_emformer_rhythm_v2_student_kd.yaml` |
| `S3-retimed` | student acoustic closure | `egs/conan_emformer_rhythm_v2_student_retimed.yaml` |

Anything outside this list should be treated as non-mainline, historical, or
experimental unless promoted explicitly later.

## 4. What changed from the old story

The active docs now commit to these decisions:

- **teacher-first, not teacher-as-deployment-model**
- **boundary / budget / execution authority first**, not phase-first storytelling
- **reference as prior**, not as a long continuous local script
- **timing controller first**, not full expressive prosody controller
- **typed boundaries**, where JOIN/WEAK remain local realization events and the
  current phrase-bank pointer is still a persisted runtime pointer advanced with
  committed-frontier progress
- **cloud1 snapshot as archive**, not as the live design document

This is the critical cleanup: the repo already behaves more like a
boundary-first, teacher-first system than the older documentation admitted.
The docs now follow the implementation instead of preserving every historical
explanation.

## 5. Migration rule of thumb

Use the smallest migration class that matches the real change.

### A. Runtime-only change

- metrics
- diagnostics
- inference override
- execution-time routing only

**Action:** no retraining required.

### B. Sampling / routing semantic change

- active-tail normalization
- open-tail semantic alignment
- train/infer sparse-selection alignment

**Action:** old checkpoints remain usable, but short fine-tuning is the honest default.

### C. Public-contract change

- teacher export truth changes
- student-visible cache contract changes
- new planner head / new loss / new embedding changes the learned target surface

**Action:** treat it as a new experiment, rebuild export/cache, and use a fresh `exp_name`.

The practical question is simple:

> if the public teacher/student truth surface changed, it is not a "small patch"

## 6. Teacher runtime semantics

The repo still contains teacher runtime branches, but they now have stricter
semantics:

- `rhythm_teacher_as_main` = **offline/full-commit audit/export execution**
- `rhythm_enable_dual_mode_teacher` = run streaming student/mainline **plus** offline teacher
  branch during training
- teacher branches do not inherit every streaming override:
  - streaming-only scheduler controls stay on the streaming branch
  - teacher branches only consume trace/phrase-gate controls, source-boundary
    scaling, teacher pause sparsity, and projector debt anti-windup
- in the current module implementation, `forward_teacher()` canonicalizes the
  teacher branch into a closed/full-commit view with internally built masks and
  fresh state rather than reusing streaming prefix state

This is intentional cleanup, not a missing feature. The offline teacher planner
is not the same object as the streaming scheduler/controller.

## 7. Cloud1 preservation rule

The cloud AutoDL state is preserved in:

- `docs/cloud1_autodl_project_status_snapshot_2026-04-08.md`

That file is a frozen archive.
This local repository is the successor surface for new work.
For actual inheritance / warm-start procedure, use:

- `docs/autodl_training_handoff.md`

## 8. Current practical boundary

The remaining blockers are mostly asset-side:

- formal processed data and binary caches are external
- teacher export assets are external
- `S3-retimed` still requires matched retimed mel / F0 / UV side assets

`T2-audio` currently exists as a maintained **procedure**, not as a separate
checked-in audio-polish training YAML. That is intentional, not a missing
artifact.

## 9. Documentation rule

If a document section does not improve one of these, it should not stay in the
mainline docs:

- teacher asset quality and auditability
- student handoff clarity
- execution authority clarity
- asset reproducibility and inheritance safety
