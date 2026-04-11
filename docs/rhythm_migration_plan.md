# Rhythm Migration Plan / Current Architecture Note (2026-04-11)

This file describes only the **current maintained rhythm architecture**.

## 1. Current maintained branch reading

The maintained line is **`rhythm_v3`**.

The maintained default is:

- explicit prompt units
- source-observed duration anchors for sealed speech units
- speech-only prompt global-rate estimation
- strict-causal source prefix-rate EMA
- static prompt summary memory
- single duration writer
- deterministic projector with residual carry

Recommended config:

- `egs/conan_emformer_rhythm_v3.yaml`
- `rhythm_v3_backbone: prompt_summary`
- `rhythm_v3_warp_mode: none`
- `rhythm_v3_anchor_mode: source_observed`

## 2. Default prompt-summary formula

For the maintained default path, the runtime reading is:

> `log d_hat_i = log a_i + (g_ref - g_src_prefix,i) + delta_i`

where:

- `a_i`: source-observed duration for sealed speech units, with frontend fallback only when needed
- `g_ref`: speech-only global prompt log-rate
- `g_src_prefix,i`: strict-causal source prefix rate EMA before unit `i`
- `delta_i`: learned residual from a causal source query against a static prompt summary

In other words:

- the prompt is distilled once into static conditioning
- the writer is source-anchored
- only sealed speech units are committed
- the projector only handles integerization and carry

## 3. Scope boundary

The current v3 mainline is a maintained duration-transfer path for streaming VC.
It is not a claim of full prosody transfer or fully stateful end-to-end
low-latency deployment.

## 4. What stays in code but is not the default reading

`rhythm_v3` still contains comparative runtime modes, including:

- `global_only`
- `operator`
- `progress`
- `detector`
- source-residual operator variants

These stay for controlled comparison, but the maintained default branch reading
is the prompt-summary path above.

## 5. Prompt-side contract

The maintained v3 path requires explicit prompt-unit conditioning:

- `prompt_content_units`
- `prompt_duration_obs`
- `prompt_unit_mask`

The prompt encoder produces:

- `global_rate`
- `summary_state`
- diagnostic slot statistics:
  - `role_value`
  - `role_var`
  - `role_coverage`

Those statistics remain useful for losses and observability. Runtime only needs
static prompt conditioning rather than a prompt-time evolution story.

## 6. Source-side/runtime contract

The maintained prompt-summary runtime uses:

- `content_units`
- `dur_anchor_src`
- `unit_anchor_base`
- sealed/open masks from the frontend

For the default path:

- source-observed durations anchor committed speech units
- strict-causal local-rate state is the key streaming rate state
- separator units are excluded from speech-duration supervision and reporting

## 7. Projector contract

The maintained projector story is narrow:

- deterministic projection to integer frames
- residual carry

## 8. Task/data split now in code

The task layer is now split into real packages:

- `tasks/Conan/rhythm/common/`
- `tasks/Conan/rhythm/duration_v3/`
- `tasks/Conan/rhythm/rhythm_v2/`

Meaning:

- `common/`: shared orchestration, runtime helpers, collate / batching, validation helpers
- `duration_v3/`: canonical v3 task and dataset logic
- `rhythm_v2/`: legacy v2 teacher/export compatibility logic

Top-level files remain only as compatibility facades:

- `tasks/Conan/rhythm/task_mixin.py`
- `tasks/Conan/rhythm/dataset_mixin.py`
- `tasks/Conan/rhythm/losses.py`
- `tasks/Conan/rhythm/targets.py`
- `tasks/Conan/rhythm/metrics.py`
- `tasks/Conan/rhythm/task_config.py`
- `tasks/Conan/rhythm/runtime_modes.py`
- `tasks/Conan/rhythm/task_runtime_support.py`

## 9. Current file map

### Runtime

- `modules/Conan/rhythm_v3/module.py`
- `modules/Conan/rhythm_v3/summary_memory.py`
- `modules/Conan/rhythm_v3/reference_memory.py`
- `modules/Conan/rhythm_v3/projector.py`
- `modules/Conan/rhythm_v3/unit_frontend.py`

### Training surfaces

- `tasks/Conan/rhythm/common/`
- `tasks/Conan/rhythm/duration_v3/`
- `tasks/Conan/rhythm/rhythm_v2/`

### Inference/runtime helpers

- `inference/Conan.py`
- `inference/run_voice_conversion.py`
- `inference/run_streaming_latency_report.py`

## 10. Documentation rule

When documentation disagrees, prefer the smallest truthful current reading:

- `rhythm_v3` is the maintained line
- `prompt_summary + source_observed + carry-only projector` is the maintained default
- top-level task files are compatibility shells, not the main implementation location
- legacy v2 docs are archive notes, not the mainline spec

## 11. Legacy status of v2

`rhythm_v2` remains only for:

- old checkpoints
- teacher/export compatibility
- legacy experiments
- archival operational notes

## 12. Current validation snapshot

Validated locally after the real task-layer split and mixin cleanup:

- `python -m compileall -q tasks/Conan/rhythm tasks/Conan/Conan.py tasks/Conan/dataset.py inference/Conan.py`
- import smoke for:
  - `tasks.Conan.rhythm`
  - `tasks.Conan.Conan`
  - `tasks.Conan.dataset`
  - `inference.Conan`
- pytest bundle:
  - `tests/rhythm/test_rhythm_v3_losses.py`
  - `tests/rhythm/test_rhythm_v3_metrics.py`
  - `tests/rhythm/test_loss_components.py`
  - `tests/rhythm/test_target_builder.py`
  - `tests/rhythm/test_task_config_v3.py`
  - `tests/rhythm/test_runtime_modes_v3.py`
  - `tests/rhythm/test_task_runtime_support.py`
  - `tests/rhythm/test_inference_entrypoints.py`
  - `tests/rhythm/test_reference_sidecar.py`
  - `tests/rhythm/test_optimizer_param_collection.py`
  - `tests/rhythm/test_pitch_supervision_runtime.py`
  - `tests/rhythm/test_loss_confidence_routing.py`
  - `tests/rhythm/test_metrics_masking.py`
  - `tests/rhythm/test_streaming_chunk_metrics.py`
  - `tests/rhythm/test_budget_surfaces.py`
  - `tests/rhythm/test_cache_contracts.py`
  - `tests/rhythm/test_reference_bootstrap_runtime.py`
  - `tests/rhythm/test_runtime_validation_alignment.py`

Result: **212 passed**.
