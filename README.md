[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

# Conan Rhythm Branch

This repository maintains a **teacher-first rhythm stack**.

## Mainline in one sentence

The maintained path is:

1. train a strong non-causal offline teacher into a stable, auditable, exportable rhythm oracle;
2. verify those teacher assets through **teacher-conditioned audio closure**;
3. distill the deployment-facing student from those teacher assets;
4. close student acoustic behavior in retimed stage 3.

This is the current engineering mainline of the repo.

## What the current implementation actually is

The maintained control story is:

> **source-side boundary / commit evidence -> explicit speech/pause budget planning -> projector execution authority -> renderer**  
> **reference acts as global / phrase prior plus bounded local boundary-style residual**  
> **phase_ptr is retained as observer / telemetry, not as the runtime control authority**

In concrete terms:

- **source side**: discrete units plus boundary evidence such as `sep_hint`, `open_run_mask`, `sealed_mask`, and `boundary_confidence`
- **planner side**: explicit speech / pause budgeting and unit-level redistribution
- **execution side**: the projector remains the binding execution authority
- **reference side**: reference conditioning is treated as global / phrase prior, not as the main continuous control script
- **phase semantics**: phase-decoupled timing is the canonical runtime direction and naming surface, but compatibility-default configs may still require explicit `rhythm_phase_decoupled_timing: true`; `phase_ptr` remains available for observability, compatibility, and render-side telemetry

## Current runtime semantics that the docs now commit to

The maintained runtime is best understood as a **timing controller**, not a
full prosody imitation controller.

- planner mainline controls:
  - speech duration budget
  - pause budget
  - unit-level duration redistribution
  - boundary-related local lengthening hints / side signals
- projector mainline enforces:
  - feasibility
  - sparse pause allocation
  - discrete commit frontier
  - timing-debt update
- downstream style / acoustic layers remain responsible for:
  - full F0 contour
  - accent realization
  - expressive pitch reset / intonation shape

### Boundary typing

The maintained code now distinguishes three boundary types:

- `JOIN`: continuous connection; projector-side execution keeps JOIN slots pause-free
- `WEAK`: weak local junction; local timing variation is allowed, but projector pause is capped by `rhythm_projector_weak_boundary_pause_cap`
- `PHRASE`: stronger phrase-like boundary class; phrase slots are the main carrier for redistributed editable-tail pause mass

This means **boundary is not the same thing as pause**, and not every word
boundary or token transition advances reference-side phrase retrieval.

### Reference usage

Reference conditioning is layered:

- **global prior**: overall rate / pause tendency
- **phrase prior**: ordered phrase-bank prototype selected through the persisted runtime pointer `ref_phrase_ptr`
- **local residual style cue**: bounded boundary-style modulation only

In the current checkout, phrase-bank retrieval is driven by the persisted
runtime pointer `ref_phrase_ptr`. When the committed frontier moves forward,
the pointer advances and is clamped against the valid phrase-bank size.
`JOIN / WEAK / PHRASE` currently constrain local realization and projector
pause behavior; a stricter PHRASE-only pointer update remains a future contract
change rather than current code truth. The reference is not treated as a
wall-clock script.

### Phase-decoupled naming

Canonical runtime naming is now:

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

Deprecated compatibility aliases still exist for older configs:

- `rhythm_phase_free_timing`
- `rhythm_phase_free_phrase_boundary_threshold`
- `rhythm_phase_decoupled_phrase_boundary_threshold`

New work should use the canonical `rhythm_phase_decoupled_*` names only.

### Sidecar terminology

Two different concepts are intentionally separated:

1. **reference sidecar**
   - enabled by `rhythm_emit_reference_sidecar`
   - means generic reference-conditioning auxiliaries such as slow-memory / planner sidecars
2. **debug export sidecars**
   - enabled by `rhythm_export_debug_sidecars`
   - means dataset/sample/cache audit fields exported for debugging

Do not treat these as the same switch.

`rhythm_emit_reference_sidecar` may also auto-enable when the repo needs:

- external reference bootstrap
- `rhythm_cached_reference_policy=sample_ref`
- reference-descriptor bootstrap losses
- trace reliability / fallback support

Phrase-bank conditioning is related but not identical:

- it may be emitted together with reference sidecars
- it may be materialized explicitly by `rhythm_runtime_phrase_bank_enable`
- it may also be consumed directly when `ref_conditioning` already carries
  `ref_phrase_*` fields

### Runtime entrypoints

- **public runtime entry**: `modules/Conan/rhythm/runtime_adapter.py::ConanRhythmAdapter`
- **lower-level helper**: `modules/Conan/rhythm/bridge.py::run_rhythm_frontend`

`run_rhythm_frontend()` is still used in tests and integration glue, but the
adapter is the maintained public runtime surface.

### Teacher runtime semantics

- `rhythm_teacher_as_main=true` means **run the learned offline teacher as the main
  execution branch for teacher-only audit/export style stages**
- it is an **offline/full-commit teacher runtime**, not a deployment streaming replacement
- `forward_teacher()` internally materializes a closed teacher view
  (`open_run_mask=0`, `sealed_mask=1`) and a fresh teacher state per call
- `rhythm_enable_dual_mode_teacher=true` means **run both streaming and offline teacher branches during training**
- teacher branches are suppressed during `infer=true`
- canonical teacher pause sparsity knob:
  `rhythm_teacher_projector_soft_pause_selection`
- scheduler-only streaming knobs such as
  `rhythm_phase_decoupled_boundary_style_residual_scale`,
  `rhythm_debt_control_scale`,
  `rhythm_debt_pause_priority`,
  `rhythm_debt_speech_priority`
  do **not** change offline teacher planning
- shared projector anti-windup knobs
  (`rhythm_projector_debt_leak`,
  `rhythm_projector_debt_max_abs`,
  `rhythm_projector_debt_correction_horizon`)
  affect both streaming and teacher projector execution

### Runtime override surface

The maintained runtime override surface is intentionally smaller than the full
training hparam surface. Supported operator-facing overrides include:

- trace controls
  - `trace_horizon`
  - `trace_active_tail_only`
  - `trace_offset_lookahead_units`
  - `trace_cold_start_min_visible_units`
  - `trace_cold_start_full_visible_units`
- phase-decoupled controls
  - `phase_decoupled_timing`
  - `phase_decoupled_phrase_gate_boundary_threshold`
- streaming-only scheduler / debt shaping
  - `phase_decoupled_boundary_style_residual_scale`
  - `debt_control_scale`
  - `debt_pause_priority`
  - `debt_speech_priority`
- source-boundary scaling
  - `source_boundary_scale_override`
  - `teacher_source_boundary_scale_override`
- projector controls
  - `projector_pause_topk_ratio_override`
  - `projector_force_full_commit`
  - `teacher_projector_force_full_commit`
  - `teacher_projector_soft_pause_selection`
- projector debt anti-windup
  - `projector_debt_leak`
  - `projector_debt_max_abs`
  - `projector_debt_correction_horizon`

Teacher-only runtime does **not** expose every streaming scheduler knob; the
offline teacher path intentionally stays narrower. If teacher-only override
keys are passed while no teacher runtime branch is active, the adapter now
treats them as unused runtime overrides rather than silently consuming them.

## Maintained stage map

| Stage | Meaning | Current implementation |
|---|---|---|
| `T1-surface` | stable offline teacher surface | `egs/conan_emformer_rhythm_v2_teacher_offline.yaml` |
| `T2-audio` | teacher-conditioned audio closure, audit, and export | procedure centered on `scripts/export_rhythm_teacher_targets.py` |
| `S2-kd` | conservative teacher-conditioned student KD | `egs/conan_emformer_rhythm_v2_student_kd.yaml` |
| `S3-retimed` | student acoustic closure | `egs/conan_emformer_rhythm_v2_student_retimed.yaml` |

## Main operator gates

- **Teacher first**: do not hand off to the student because one pause scalar improves; watch the full teacher-control set together.
- **Checkpoint selection**: choose teacher checkpoints by being **exportable, audible, and inheritable**, not by `total_loss` alone and not by pause `F1` alone.
- **Stage 2 role**: let `S2-kd` absorb the remaining streaming/prefix mismatch while keeping the teacher export truth fixed.
- **Stage 3 role**: keep `S3-retimed` cached-first first, then A/B the later `hybrid` route only after teacher and stage-2 handoff are both stable.

For the detailed operator rules, use `docs/autodl_training_handoff.md`.

## Canonical docs

The maintained documentation surface is intentionally reduced to:

1. `README.md`
2. `docs/rhythm_migration_plan.md`
3. `docs/autodl_training_handoff.md`
4. `docs/cloud1_autodl_project_status_snapshot_2026-04-08.md` (**archive only**)

The cloud snapshot is preserved for historical state, but it is not the live
design document for new work.

## Config and script map

| Surface | File |
|---|---|
| maintained teacher stage | `egs/conan_emformer_rhythm_v2_teacher_offline.yaml` |
| maintained student KD stage | `egs/conan_emformer_rhythm_v2_student_kd.yaml` |
| maintained student retimed stage | `egs/conan_emformer_rhythm_v2_student_retimed.yaml` |
| stage preflight | `scripts/preflight_rhythm_v2.py` |
| teacher asset export | `scripts/export_rhythm_teacher_targets.py` |
| teacher -> student smoke integration | `scripts/integration_teacher_export_student_kd.py` |
| maintained smoke | `scripts/smoke_test_rhythm_v2.py` |

## Current checkout status

This checkout is code-ready for the maintained path, but formal runs still
depend on external assets:

- formal processed data and binary caches are not fully checked in
- teacher export assets are not checked in
- `S3-retimed` still requires retimed cache plus matched F0/UV side data
- checked-in smoke assets are for structural verification only, not formal training claims

## Installation

```bash
git clone https://github.com/tanyrbs/conan-rhythm.git
cd conan-rhythm
conda create -n conan python=3.10
conda activate conan
pip install -r requirements.txt
```

## Minimal local verification

```bash
python -m compileall -q modules tasks scripts tests utils data_gen
python -m unittest discover -s tests/rhythm -p "test_*.py"
python -u scripts/smoke_test_rhythm_v2.py
python scripts/preflight_rhythm_v2.py --help
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

## Acknowledgements

- [FastSpeech2](https://github.com/ming024/FastSpeech2)
- [HiFi-GAN](https://github.com/jik876/hifi-gan)
- [Emformer](https://github.com/pytorch/audio)
