# Conan Rhythm Branch

This checkout has one maintained rhythm code mainline: **`rhythm_v3`**.

The maintained default is **streaming VC unit-duration transfer**.

The current recommended research workflow is not "train everything first". It
is a **three-gate falsification loop** over the maintained `rhythm_v3` line:

1. Gate 0: static `g` audit
2. Gate 1: analytic monotonicity
3. Gate 2: coarse-only vs learned stability trade-off

## Current maintained default

Use:

- `egs/conan_emformer_rhythm_v3.yaml`
- `rhythm_v3_backbone: prompt_summary`  (`unit_run` / `role_memory` remain accepted legacy aliases)
- `rhythm_v3_warp_mode: none`
- `rhythm_v3_anchor_mode: source_observed`
- `rhythm_v3_minimal_v1_profile: true`
- `rhythm_v3_rate_mode: simple_global`
- `rhythm_v3_simple_global_stats: true`
- `rhythm_v3_g_variant: raw_median`
- `rhythm_v3_unit_prior_path: null`  (`unit_norm` only; build with `scripts/build_unit_log_prior.py`)
- `rhythm_v3_use_continuous_alignment: true`
- `rhythm_v3_alignment_mode: continuous_viterbi_v1`
- `rhythm_v3_gate_quality_strict: true`  (intent marker; strict gate enforcement is driven by `scripts/rhythm_v3_debug_records.py`)
- `rhythm_v3_eval_mode: learned`

In that default path, the writer is:

> `z_hat_i = (g_ref - g_src_prefix,i) + c_hat_i + r_hat_i`

where:

- `g_ref`: speech-only prompt global tempo statistic on raw log-duration
- `g_src_prefix,i`: strict-causal source prefix tempo EMA on raw source log-duration before unit `i`
- `c_hat_i`: small utterance-level coarse scalar, conditioned on speaker and, only in ablation/diagnostic modes, optional static prompt summary
- `r_hat_i`: bounded local speech-run residual

and committed speech duration is written as:

> `log d_hat_i = log a_i + z_hat_i`

Silence-like runs remain in the retimed prefix but only follow the coarse/global bias (clipped for stability) without a local residual, so they stretch or compress slightly with the overall pace while speech-only statistics stay focused on speaking runs.
The canonical writer also keeps local residuals conservative at cold start: open runs are not committed, and the first few committed speech runs are biased toward analytic/coarse control before local residuals fully open up.

The maintained explanation is:

- **source-anchored duration writing**
- **global-stat prompt conditioning by default**
- **utterance-level scalar coarse correction over the analytic source/ref rate gap, broadcast over the visible prefix**
- **strict-causal prefix-rate correction**
- **carry+budget integer projection core with exported boundary-side smoothing telemetry**
- **explicit silence-run frontend that materializes speech vs. pause runs**
- **stable-lattice suppression of short flicker runs and micro-silence islands before retiming**
- **silence runs follow the coarse/global bias (clipped) without a local residual, keeping them tied to the overall rate**
- **minimal V1 keeps silence on a constant clip surface; short-gap / leading-silence scaling remains a non-minimal head feature**
- **runtime and target construction now share the same silence-`tau` helper; minimal stays constant-clip, non-minimal stays boundary-aware**
- **reference summary diagnostics remain available but are disabled in the default V1-G writer**
- **paired-target supervision drawn from dedicated target data instead of prompt sidecars**
- **same-text rule split by role: reference stays different-text, paired-target supervision stays same-text unless `unit_duration_tgt` is already cached**
- **runtime enforces prefix unit-budget clamping while retaining raw open tail tokens**

The structural mainline contract is therefore:

- `prompt_summary + strict + simple_global`
- explicit `prompt_speech_mask` on the prompt side
- default runtime surface `raw_median + learned`

For falsification work, `analytic`, `coarse_only`, and `learned` are three
evaluation modes on that same runtime surface, not three separate systems.

For falsification work, stay on that same structural path and only override the
controlled analysis knobs (`analytic` / `coarse_only`, alternative `g_variant`)
instead of opening a second runtime branch.

The maintained gate presets are checked in as config overlays rather than left
as comment-only edits:

- `egs/overrides/rhythm_v3_gate0_g_audit.yaml`
- `egs/overrides/rhythm_v3_gate1_analytic.yaml`
- `egs/overrides/rhythm_v3_gate2_coarse_only.yaml`

## What the default v3 path does

### Prompt side

`PromptDurationMemoryEncoder` consumes explicit prompt units:

- `prompt_content_units`
- `prompt_duration_obs`
- `prompt_unit_mask`
- `prompt_speech_mask`
- optional `prompt_valid_mask`
- optional `prompt_silence_mask`
- optional `prompt_spk_embed`

It produces:

- `global_rate`
- `summary_state`
- `spk_embed`
- diagnostic slot statistics (`role_value`, `role_var`, `role_coverage`)

The diagnostic slot statistics remain useful for observability, but the
maintained V1-G writer defaults to:

- `rhythm_v3_minimal_v1_profile: true`
- `rhythm_v3_rate_mode: simple_global`
- `rhythm_v3_simple_global_stats: true`
- `rhythm_v3_use_log_base_rate: false`
- `rhythm_v3_use_reference_summary: false`
- `rhythm_v3_use_learned_residual_gate: false`
- `rhythm_v3_disable_learned_gate: true`

So the main retiming path consumes the prompt's speech-only global statistic and
speaker embedding by default; `summary_state` / slot statistics stay available
for diagnostics and ablations instead of defining the default writer.

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

The current local executor still layers boundary-side smoothing heuristics
(`boundary_carry_decay`, `since_last_boundary`) on top of that carry+budget
core. Those signals are exported for audit, but they should be read as
execution-side heuristics rather than as a second writer definition.
`projector_boundary_hit` marks boundary/phrase-final events, while
`projector_boundary_decay_applied` only marks the subset where decay was
actually applied.

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

For review/export, `unit_duration_proj_raw_tgt` is the explicit raw-projection
alias and should be preferred when present. When `unit_duration_tgt` is cached
for the maintained continuous path, keep `unit_duration_proj_raw_tgt`,
`unit_alignment_mode_id_tgt`, `unit_alignment_kind_tgt`,
`unit_alignment_source_tgt`, `unit_alignment_version_tgt`,
`unit_alignment_source_cache_signature_tgt`,
`unit_alignment_target_cache_signature_tgt`, and
`unit_alignment_sidecar_signature_tgt` alongside it so the target surface
remains provenance-clean. Training still consumes
decomposed surfaces such as `global_shift_tgt`, `coarse_logstretch_tgt`,
`local_residual_tgt`, and the clipped coarse-derived silence target rather than
treating the raw projection surface as the final supervision object.

For the maintained `rhythm_v3_minimal_v1_profile`, the pairing rule is:

- **reference prompt**: same-speaker / different-text
- **paired target supervision**: same-text target-to-source projection (`rhythm_v3_require_same_text_paired_target: true`), or an explicitly cached `unit_duration_tgt`

`rhythm_v3_use_continuous_alignment: true` is intentionally split into three
public layers:

- `continuous_precomputed`: consume external continuous metadata / sidecars
- `continuous_viterbi_v1`: repo-native offline hard monotonic source-run /
  target-frame alignment
- `continuous_fwdbwd_v1`: planned posterior-bearing forward-backward upgrade
  surface, not the maintained default

The maintained public branch currently ships two continuous-alignment surfaces:
`continuous_precomputed` for external frame/content-space sidecars, and
`continuous_viterbi_v1` for repo-native hard monotonic source-run alignment.
Posterior-bearing forward-backward alignment remains a planned upgrade surface
rather than the maintained default. If neither explicit continuous provenance
nor the required frame-state sidecars are available, the maintained path fails
fast instead of silently falling back to discrete projection.

In the maintained canonical YAML, continuous paired-target supervision is
explicitly enabled and pinned to `continuous_viterbi_v1`; switching back to a
discrete paired-target path is now a deliberate ablation rather than an
implicit default.

`continuous_viterbi_v1` is a hard-path aligner: it exports Viterbi/source-run
occupancy plus heuristic confidence/provenance, not posterior expected
occupancy. Treat posterior expected-duration surfaces as unavailable until
`continuous_fwdbwd_v1` exists. When `target_frame_weight` sidecars are present,
the maintained path can also exclude frames below
`rhythm_v3_alignment_min_dp_weight` before DP instead of letting low-weight
observations silently steer the path search.

The maintained config also carries explicit alignment-quality thresholds
(`rhythm_v3_alignment_unmatched_speech_ratio_max`,
`rhythm_v3_alignment_mean_local_confidence_speech_min`,
`rhythm_v3_alignment_mean_coarse_confidence_speech_min`) and a
`rhythm_v3_gate_quality_strict: true` intent marker. Actual fail-hard behavior
is enforced by `scripts/rhythm_v3_debug_records.py` when `--strict-gates` is
used, or automatically when `--review-dir` / `--gate-status-json` is requested
without `--allow-partial-gates`.

The legacy filename `egs/conan_emformer_rhythm_v2_minimal_v1.yaml` is now only a
compatibility alias that inherits the maintained `rhythm_v3` contract; it is no
longer a separate V2 experiment surface.

If you move from `raw_median` / `weighted_median` / `trimmed_mean` to
`rhythm_v3_g_variant: unit_norm`, treat that as a data-contract change rather
than a pure knob flip: the maintained repo now expects a reproducible unit prior
bundle under `rhythm_v3_unit_prior_path`, and the recommended way to build it
is `scripts/build_unit_log_prior.py`. That bundle now carries prior-policy and
frontend provenance metadata, not just raw prior values.

Required public inputs:

- `content_units`
- `dur_anchor_src`
- `unit_anchor_base`
- `prompt_content_units`
- `prompt_duration_obs`
- `prompt_unit_mask`
- `prompt_speech_mask`

Optional prompt sidecars used by the maintained default:

- `prompt_valid_mask`
- `prompt_silence_mask` for compatibility-only validation
- `prompt_spk_embed`

For the maintained strict minimal path, `prompt_speech_mask` is no longer an
optional sidecar. It is part of the explicit public conditioning contract.

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
- cross-prefix consistency off by default until strict-causal fine-tuning

## Falsification-first scripts

The maintained evaluation surface is now intentionally reduced to one export
script plus one explicit health check:

- `scripts/preflight_rhythm_v3.py`
- `scripts/rhythm_v3_debug_records.py`

`scripts/rhythm_v3_debug_records.py` is the single maintained review/export
entrypoint. It writes the row summary CSV, the retained five-figure bundle, and
the gate-oriented tables/figures from the same `utils/plot/rhythm_v3_viz/`
util layer instead of keeping separate per-gate wrapper CLIs.
The remaining maintained helper stays on purpose:

- `preflight_rhythm_v3.py`: config/data/cache contract checks

The old zero-data standalone smoke script has been retired from the maintained
surface. Structural smoke coverage now lives in focused entrypoint tests and
the short training smoke recipe documented below.

Some earlier cleanup ideas are already obsolete in the local codebase:

- `modules/Conan/rhythm_v3/g_stats.py` is already the single `g` definition used
  by training/runtime/review
- `analytic | coarse_only | learned` already switch inside the same maintained runtime
- projector drift/budget signals are already exported; the remaining job is to
  audit them, not redesign projector semantics

For Gate-0 and contamination-slice style review, prefer debug bundles exported
from the maintained train/eval path with pair metadata still attached
(`pair_id`, prompt ids, same-text flags, `lexical_mismatch`, `ref_len_sec`,
`speech_ratio`). Inference-only bundles can still be summarized, but some
stability/cross-text panels may collapse to partial evidence rather than a full
falsification readout.

The row summary and gate bundle now also carry analysis-friendly aliases such as
`same_speaker_reference`, `same_speaker_target`, `tempo_delta`, and
`mono_triplet_ok`, so one export is enough for both row inspection and gate
aggregation.

The shared `g` implementation now stays fail-fast and speech-only: if prompt
speech support disappears after mask/domain filtering, the maintained path
raises instead of silently falling back to non-speech support.

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

### CI / smoke surface

Maintained automation should exercise the V3 path:

- `scripts/preflight_rhythm_v3.py`
- `scripts/rhythm_v3_debug_records.py`

Structural smoke is expected to come from targeted `tests/rhythm/` coverage
and short task-level smoke runs rather than a separate standalone v3 smoke CLI.

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
- `docs/rhythm_v3_training_guide.md`
- `docs/rhythm_v3_validation_stack.md`
- `docs/rhythm_migration_plan.md`
- `inference/README.md`

## Practical setup and training

If you want the repo-specific "what files matter / how do I prepare data / how do I actually train" walkthrough, use:

- `docs/rhythm_v3_training_guide.md`
- `docs/rhythm_v3_validation_stack.md` for pre-train label audit / runtime debug-record export

Short version:

1. create a dataset-specific yaml that inherits from `egs/conan_emformer_rhythm_v3.yaml`
2. prepare `metadata.json` (or `metadata_vctk_librittsr_gt.json`) plus `spker_set.json`
3. generate the expected F0 side files if `with_f0: true`
4. provide canonical paired-target supervision via `rhythm_pair_manifest_path` or explicit `unit_duration_tgt`
5. binarize with `python -m data_gen.tts.runs.binarize --config <yaml>`
6. train with `python tasks/run.py --config <yaml> --exp_name <exp> --reset`

Most important operational caveat:

- if `checkpoints/<exp_name>/config.yaml` already exists, it overrides newly provided yaml values unless you pass `--reset`

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
