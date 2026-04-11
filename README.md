# Conan Rhythm Branch

This checkout has one maintained rhythm code mainline: **`rhythm_v3`**.

The maintained default is **streaming VC unit-duration transfer**.

## Current maintained default

Use:

- `egs/conan_emformer_rhythm_v3.yaml`
- `rhythm_v3_backbone: prompt_summary`
- `rhythm_v3_warp_mode: none`
- `rhythm_v3_anchor_mode: source_observed`

In that default path, the writer is:

> `log d_hat_i = log a_i + (g_ref - g_src_prefix,i) + delta_i`

where:

- `a_i`: source-unit anchor from observed duration when sealed, with frontend fallback only when needed
- `g_ref`: speech-only prompt global log-rate
- `g_src_prefix,i`: strict-causal source prefix rate EMA before unit `i`
- `delta_i`: small learned residual from a causal source query against a static prompt summary

The maintained explanation is:

- **source-anchored duration writing**
- **static prompt summary memory**
- **strict-causal prefix-rate correction**
- **carry-only integer projection**

## What the default v3 path does

### Prompt side

`PromptDurationMemoryEncoder` consumes explicit prompt units:

- `prompt_content_units`
- `prompt_duration_obs`
- `prompt_unit_mask`

It produces:

- `global_rate`
- `summary_state`
- diagnostic slot statistics (`role_value`, `role_var`, `role_coverage`)

The diagnostic slot statistics remain useful for losses and observability, while
runtime uses the prompt summary as a static conditioning object.

### Source side

The streaming writer uses:

- `content_units`
- `dur_anchor_src`
- `unit_anchor_base`
- sealed/open masks from the frontend

For the maintained `prompt_summary + source_observed` path, committed speech
units use source-observed duration as the anchor and keep only a strict-causal
local-rate EMA in runtime state.

### Projector

The projector stays deterministic:

- integer frame projection
- residual carry

## Supported but non-default v3 modes

`rhythm_v3` still contains additional comparative modes for ablation:

- `global_only`
- `operator`
- `progress`
- `detector`
- source-residual operator variants

These remain in code for controlled comparison, but they are **not** the
recommended maintained default reading of the branch.

## Current training contract

Mainline v3 training expects explicit prompt-unit conditioning and unit-level
source supervision.

Required public inputs:

- `content_units`
- `dur_anchor_src`
- `unit_anchor_base`
- `prompt_content_units`
- `prompt_duration_obs`
- `prompt_unit_mask`

Required public outputs:

- `speech_duration_exec`
- `rhythm_frame_plan`
- `commit_frontier`
- `rhythm_state_next`

Compact public losses:

- `rhythm_total`
- `rhythm_v3_dur`
- `rhythm_v3_summary`
- `rhythm_v3_pref`
- `rhythm_v3_cons`

Recommended default weights in `egs/conan_emformer_rhythm_v3.yaml` currently keep:

- duration loss on
- prompt-summary reconstruction on
- prefix consistency off by default
- cross-prefix consistency on

## Training/task code layout

The rhythm task layer is now split into three real packages:

- `tasks/Conan/rhythm/common/`
- `tasks/Conan/rhythm/duration_v3/`
- `tasks/Conan/rhythm/rhythm_v2/`

Top-level files remain only as compatibility facades for existing imports:

- `tasks/Conan/rhythm/task_mixin.py`
- `tasks/Conan/rhythm/dataset_mixin.py`
- `tasks/Conan/rhythm/losses.py`
- `tasks/Conan/rhythm/targets.py`
- `tasks/Conan/rhythm/metrics.py`
- `tasks/Conan/rhythm/task_config.py`
- `tasks/Conan/rhythm/runtime_modes.py`
- `tasks/Conan/rhythm/task_runtime_support.py`

Current split:

- `common/`: shared orchestration and runtime helpers
- `duration_v3/`: canonical v3 task/data logic
- `rhythm_v2/`: legacy v2 teacher/export compatibility logic

## Main files for the current story

### Runtime

- `modules/Conan/rhythm_v3/module.py`
- `modules/Conan/rhythm_v3/summary_memory.py`
- `modules/Conan/rhythm_v3/reference_memory.py`
- `modules/Conan/rhythm_v3/projector.py`
- `modules/Conan/rhythm_v3/unit_frontend.py`

### Inference

- `inference/Conan.py`
- `inference/run_voice_conversion.py`
- `inference/run_streaming_latency_report.py`

## Legacy status

These are still kept only for compatibility, archive experiments, and old
checkpoints:

- `modules/Conan/rhythm/`
- `tasks/Conan/rhythm/rhythm_v2/`
- v2 teacher/export configs in `egs/`
- `docs/autodl_training_handoff.md`
- `docs/cloud1_autodl_project_status_snapshot_2026-04-08.md`

## Current readiness reading

Ready now:

- one maintained `rhythm_v3` line
- prompt-summary default path for streaming VC duration transfer
- real `common / duration_v3 / rhythm_v2` task split
- deterministic projector contract
- explicit prompt-unit training semantics

Not claimed ready:

- full prosody transfer
- fully stateful end-to-end low-latency deployment
- a claim that non-default v3 ablation modes are already irrelevant

## Authoritative current docs

- `README.md`
- `docs/rhythm_migration_plan.md`
- `inference/README.md`

## Quick validation

```bash
py -3 -B -m pytest -q ^
  tests/rhythm/test_rhythm_v3_losses.py ^
  tests/rhythm/test_rhythm_v3_metrics.py ^
  tests/rhythm/test_loss_components.py ^
  tests/rhythm/test_target_builder.py ^
  tests/rhythm/test_task_config_v3.py ^
  tests/rhythm/test_runtime_modes_v3.py ^
  tests/rhythm/test_task_runtime_support.py ^
  tests/rhythm/test_inference_entrypoints.py ^
  tests/rhythm/test_reference_sidecar.py ^
  tests/rhythm/test_optimizer_param_collection.py ^
  tests/rhythm/test_pitch_supervision_runtime.py ^
  tests/rhythm/test_loss_confidence_routing.py ^
  tests/rhythm/test_metrics_masking.py ^
  tests/rhythm/test_streaming_chunk_metrics.py ^
  tests/rhythm/test_budget_surfaces.py ^
  tests/rhythm/test_cache_contracts.py ^
  tests/rhythm/test_reference_bootstrap_runtime.py ^
  tests/rhythm/test_runtime_validation_alignment.py
```

Current local result for the list above: **212 passed**.

## License

MIT. See [LICENSE](LICENSE).
