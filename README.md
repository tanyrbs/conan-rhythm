# Conan Rhythm Branch

This checkout has one maintained rhythm code mainline: **`rhythm_v3`**.

The maintained default is **streaming VC unit-duration transfer**.

## Current maintained default

Use:

- `egs/conan_emformer_rhythm_v3.yaml`
- `rhythm_v3_backbone: unit_run`  (`prompt_summary` / `role_memory` remain accepted aliases)
- `rhythm_v3_warp_mode: none`
- `rhythm_v3_anchor_mode: source_observed`

In that default path, the writer is:

> `z_hat_i = (g_ref - g_src_prefix,i) + c_hat_i + r_hat_i`

where:

- `g_ref`: speech-only prompt global rate on content-normalized log-duration residuals
- `g_src_prefix,i`: strict-causal source prefix rate EMA on content-normalized source log-duration residuals before unit `i`
- `c_hat_i`: small prefix-coarse correction from pooled causal source state + static prompt summary + speaker vector
- `r_hat_i`: bounded local speech-run residual

and committed speech duration is written as:

> `log d_hat_i = log a_i + z_hat_i`

Silence-like runs remain in the retimed prefix but only follow the coarse/global bias (clipped for stability) without a local residual, so they stretch or compress slightly with the overall pace while speech-only statistics stay focused on speaking runs.
The canonical writer also keeps local residuals conservative at cold start: open runs are not committed, and the first few committed speech runs are biased toward analytic/coarse control before local residuals fully open up.

The maintained explanation is:

- **source-anchored duration writing**
- **static prompt summary memory**
- **prefix-level coarse correction over the analytic source/ref rate gap**
- **strict-causal prefix-rate correction**
- **carry-only integer projection**
- **explicit silence-run frontend that materializes speech vs. pause runs**
- **silence runs follow the coarse/global bias (clipped) without a local residual, keeping them tied to the overall rate**
- **paired-target supervision drawn from dedicated target data instead of prompt sidecars**
- **runtime enforces prefix unit-budget clamping while retaining raw open tail tokens**

## What the default v3 path does

### Prompt side

`PromptDurationMemoryEncoder` consumes explicit prompt units:

- `prompt_content_units`
- `prompt_duration_obs`
- `prompt_unit_mask`
- optional `prompt_valid_mask`
- optional `prompt_speech_mask`
- optional `prompt_spk_embed`

It produces:

- `global_rate`
- `summary_state`
- `spk_embed`
- diagnostic slot statistics (`role_value`, `role_var`, `role_coverage`)

The diagnostic slot statistics remain useful for losses and observability, while
runtime now directly uses `summary_state + spk_embed` as static conditioning;
slot statistics are diagnostics only.

A `rhythm_v3_summary_pool_speech_only` flag keeps that summary pooling speech-only, masking out silence runs before computing the pooled mean/std so `global_rate` and `summary_state` stay focused on speaking rhythm rather than pause counts.

A `rhythm_v3_summary_pool_speech_only` flag keeps that summary pooling speech-only, masking out silence runs before computing the pooled mean/std so `global_rate` and `summary_state` stay focused on speaking rhythm rather than pause counts.

### Source side

The streaming writer uses:

- `content_units`
- `dur_anchor_src`
- `unit_anchor_base`
- sealed/open masks from the frontend

For the maintained `unit_run/prompt_summary + source_observed` path, committed speech
units use source-observed duration as the anchor and keep only a strict-causal
local-rate EMA in runtime state. `sep_mask` is not treated as a speech mask in
the canonical path; the maintained v3 frontend now materializes explicit
silence runs and exports `source_silence_mask` directly.
Silence runs still exist in the stream but commit through the coarse/global bias path only (no local residual), and `source_silence_mask` keeps speech-only statistics from being contaminated by those paused units.

### Projector

The projector stays deterministic:

- integer frame projection
- residual carry
- explicit prefix unit-budget clamp that keeps `O_p = Σ(q_i - n_i)` within configured bounds
- runtime keeps the raw uncommitted open tail appended to the retimed prefix so downstream code still sees unreleased units

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

Mainline v3 training expects explicit prompt-unit conditioning plus a
separate paired-target supervision chain. Canonical paired training projects
the target run lattice onto the source run lattice, fills `unit_duration_tgt`
and optional `unit_confidence_tgt`, and keeps that signal strictly apart from
prompt-side diagnostics. Source-self fallback is disabled by default and
becomes available only with the explicit `rhythm_v3_allow_source_self_target_fallback`
escape hatch.

Required public inputs:

- `content_units`
- `dur_anchor_src`
- `unit_anchor_base`
- `prompt_content_units`
- `prompt_duration_obs`
- `prompt_unit_mask`

Optional prompt sidecars used by the maintained default:

- `prompt_valid_mask`
- `prompt_speech_mask`
- `prompt_spk_embed`

Required public outputs:

- `speech_duration_exec`
- `rhythm_frame_plan`
- `commit_frontier`
- `rhythm_state_next`

Compact public losses:

- `rhythm_total`
- `rhythm_v3_dur`
- `rhythm_v3_bias`
- `rhythm_v3_summary`
- `rhythm_v3_pref`
- `rhythm_v3_cons`

Recommended default weights in `egs/conan_emformer_rhythm_v3.yaml` currently keep:

- duration loss on
- prefix-coarse loss on (`lambda_rhythm_bias`, now supervising the coarse branch even though the compatibility metric still exports `global_bias_scalar`)
- prompt-summary residual self-fit disabled by default (`lambda_rhythm_summary=0`)
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
- unit-run default path for streaming VC duration transfer
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

Current local result for the list above: **216 passed**.

## License

MIT. See [LICENSE](LICENSE).
