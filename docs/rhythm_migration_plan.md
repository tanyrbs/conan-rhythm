# Rhythm Migration Plan / Current Architecture Note (2026-04-11)

This file describes only the **current maintained rhythm architecture**.

## 1. Current maintained branch reading

The maintained line is **`rhythm_v3`**.

The maintained default is:

- explicit prompt units
- source-observed duration anchors for sealed speech units
- speech-only prompt global-rate estimation
- prefix-level coarse correction around the analytic source/ref rate gap
- strict-causal source prefix-rate EMA
- static prompt summary memory
- single duration writer
- deterministic projector with residual carry
- explicit silence-run frontend that exports `source_silence_mask`
- paired-target supervision is built from a dedicated target item/sequence, keeping training signals off the prompt-conditioning chain

Recommended config:

- `egs/conan_emformer_rhythm_v3.yaml`
- `rhythm_v3_backbone: unit_run` (`prompt_summary` / `role_memory` remain accepted aliases)
- `rhythm_v3_warp_mode: none`
- `rhythm_v3_anchor_mode: source_observed`

Canonical training intentionally keeps the prompt-summary auxiliary loss disabled (`lambda_rhythm_summary=0`) so the maintained path focuses on duration/bias supervision plus the optional prefix consistency term.

## 1.1 External rationale checkpoints

The maintained narrowing of this branch is aligned with a few external anchors:

- Conan paper: discrete content labels are the controllable interface, so the retimer is attached on the content-code stream instead of the chunk scheduler — https://arxiv.org/abs/2507.14534
- R-VC: explicit token-level duration modeling is useful; their ablations report that sentence-level duration is less stable and that removing duration-side speaker conditioning hurts WER / UTMOS / style transfer — https://aclanthology.org/2025.acl-long.790/
- L2-ARCTIC: 24 non-native speakers with manually annotated pronunciation deviations make it suitable for run-level confidence weighting — https://psi.engr.tamu.edu/l2-arctic-corpus/
- CMU ARCTIC / ARCTIC prompts: shared prompt inventory remains the simplest native-native / L2-native sanity surface — https://www.festvox.org/cmu_arctic/

## 2. Default unit-run writer formula

For the maintained default path, the runtime reading is:

> `z_hat_i = (g_ref - g_src_prefix,i) + c_hat_i + r_hat_i`
>
> `log d_hat_i = log a_i + z_hat_i`

where:

- `a_i`: source-observed duration for sealed speech units, with frontend fallback only when needed
- `g_ref`: speech-only global prompt log-rate
- `g_src_prefix,i`: strict-causal source prefix rate EMA before unit `i`
- `c_hat_i`: small prefix-coarse correction from pooled causal source state + static prompt summary + speaker vector
- `r_hat_i`: learned speech-run residual from a causal source query against the same static prompt summary

In other words:

- the prompt is distilled once into static conditioning
- the writer is source-anchored
- the coarse branch stays close to the analytic `g_ref - g_src_prefix,i` term
- only sealed speech units are committed
- silence-like runs stay source-observed in the canonical v1 path
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
is the unit-run / prompt-summary path above.

## 5. Prompt-side contract

The maintained v3 path requires explicit prompt-unit conditioning:

- `prompt_content_units`
- `prompt_duration_obs`
- `prompt_unit_mask`
- optional `prompt_valid_mask`
- optional `prompt_speech_mask`
- optional `prompt_spk_embed`

The prompt encoder produces:

- `global_rate`
- `summary_state`
- `spk_embed`
- diagnostic slot statistics:
  - `role_value`
  - `role_var`
  - `role_coverage`

Those statistics remain useful for losses and observability. Runtime only needs
static prompt conditioning rather than a prompt-time evolution story, and the
writer now consumes `summary_state` directly instead of treating it as a pure
diagnostic sidecar.

Paired-target supervision is kept separate: the canonical path pulls `unit_duration_tgt`
from an explicit paired target item or `paired_target_*` inputs instead of recycling
the prompt diagnostics, and it only accepts a source-self fallback when
`rhythm_v3_allow_source_self_target_fallback` is intentionally enabled.

## 6. Source-side/runtime contract

The maintained unit-run / prompt-summary runtime uses:

- `content_units`
- `dur_anchor_src`
- `unit_anchor_base`
- sealed/open masks from the frontend

For the default path:

- source-observed durations anchor committed speech units
- strict-causal local-rate state is the key streaming rate state
- `sep_mask` is not the canonical speech mask
- `source_silence_mask` gates speech-only supervision and rate tracking
- the maintained v3 frontend now materializes explicit silence runs instead of collapsing silence into separator-only markers

## 7. Projector contract

The maintained projector story is narrow:

- deterministic projection to integer frames
- residual carry
- explicit prefix unit-budget clamp so the cumulative `O_p = Σ(q_i - n_i)` stays within configured bounds
- raw uncommitted open-tail units are retained and concatenated after the retimed prefix for downstream processing

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
- `unit_run + source_observed + carry-only projector` is the maintained default
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

Result: **216 passed**.
