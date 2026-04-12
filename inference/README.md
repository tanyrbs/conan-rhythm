# Streaming inference notes

This directory is a runtime/eval helper surface for the maintained
**`rhythm_v3`** branch.

Authoritative current docs:

- `README.md`
- `docs/rhythm_v3_training_guide.md`
- `docs/rhythm_migration_plan.md`

## Current maintained inference path

The maintained default inference path is the same as the maintained training
mainline:

- `rhythm_v3_backbone: prompt_summary` (`unit_run` / `role_memory` remain accepted legacy aliases)
- `rhythm_v3_warp_mode: none`
- `rhythm_v3_anchor_mode: source_observed`
- `rhythm_v3_minimal_v1_profile: true`
- `rhythm_v3_rate_mode: simple_global`
- `rhythm_v3_simple_global_stats: true`

`inference/Conan.py` now prepares explicit prompt-unit conditioning for v3:

- `prompt_content_units`
- `prompt_duration_obs`
- `prompt_unit_mask`
- `prompt_valid_mask`
- `prompt_speech_mask`
- `prompt_spk_embed` when available

The maintained v3 inference story is:

- prompt reduced to a speech-only global tempo statistic by default
- speaker embedding passed explicitly into the duration writer
- source-observed sealed-unit anchors
- explicit simple-global rate mode (`rhythm_v3_rate_mode: simple_global`) with no prompt/base normalization in the maintained path
- analytic source/ref rate gap, computed on speech-only raw log-duration statistics, corrected by a small coarse scalar
- bounded local speech-run residual on top of that coarse term
- optional prompt summary memory kept for diagnostics/ablation, not the default writer path
- cold-start gating keeps the first committed speech runs close to analytic/coarse control before local residuals fully open up
- strict-causal prefix-rate state
- incremental unit-run source-cache updates via `rhythm_unit_frontend.step_content_tensor()` instead of recompressing the full prefix every chunk
- conservative content-history windowing trims already-committed tokens before the acoustic forward pass
- deterministic carry-only projection
- runtime enforces the prefix unit-budget clamp so retimed counts stay within configured drift bounds
- raw uncommitted open-tail units are kept intact and appended after the retimed prefix before the main model consumes them
- `rhythm_v3_emit_silence_runs` exposes `source_silence_mask` so the runtime can distinguish speech runs from intentional pauses
- `rhythm_v3_summary_pool_speech_only` keeps any optional summary pooling speech-only so global-rate conditioning remains clean

Silence-like runs remain in the emitted sequence, but they no longer rely on a separate pause planner or on a full local residual. Instead their log-stretch prediction is clipped around the coarse bias term, and the silence contribution to loss is down-weighted via `rhythm_v3_silence_coarse_weight` while the clip range is governed by `rhythm_v3_silence_max_logstretch`. Global rate statistics (`g_ref` / `g_src`) remain speech-only, so the coarse-only policy keeps speech and silence aligned without overfitting noisy pause labels.

The maintained inference contract also assumes the shared lattice stabilizer is
already cleaning up obvious short-lived noise before retiming. In practice, the
V1-G reading is: short flicker bridges and micro-silence islands belong to the
stable-lattice frontend contract, not to the duration writer. Training cache
construction and inference prompt extraction now use the same `rhythm_v3`
source-cache builder so that the lattice contract does not drift across phases.

The rhythm frontend is incremental, but the final tail is still flushed from the last full-prefix pass rather than from a separate strict committed-only seal step. In other words, incremental run-lattice state is real, while the final end-of-utterance tail remains a pragmatic last-pass flush.

## What this directory currently serves

Maintained helpers:

- `inference/Conan.py`
- `inference/run_voice_conversion.py`
- `inference/run_streaming_latency_report.py`
- `inference/streaming_runtime.py`

For runtime-side debug export, `StreamingVoiceConversion.infer_once(...)` also
accepts:

- `return_debug_bundle=True`

which emits a serializable rhythm-v3 debug record compatible with
`utils.plot.rhythm_v3_viz`.

These are suitable for:

- streaming-oriented evaluation
- latency accounting
- prefix/commit behavior inspection

## Current streaming status

The honest current status is:

- Emformer frontend runs in real stateful streaming mode
- rhythm execution is prefix/commit oriented and uses the current `rhythm_v3` runtime
- acoustic decoding still recomputes over the emitted prefix
- shipped vocoder wrappers are still stateless by default

So this directory is useful for streaming evaluation, but it is **not** yet a
claim of fully stateful end-to-end low-latency deployment.

## Recommended config

For maintained branch checks, use:

- `egs/conan_emformer_rhythm_v3.yaml`

Legacy v2 configs may still exist for archive/compatibility checks, but they
are not the recommended runtime surface for the current branch.

## Legacy note

Legacy helpers remain in this directory, including:

- `inference/Conan_previous.py`

Treat them as archive/compatibility surfaces only; maintained CI/smoke coverage
targets the V3 preflight/smoke scripts instead.

## Practical note

If you first need to prepare data, binarize, or train the maintained `rhythm_v3`
path before running inference helpers, start with:

- `docs/rhythm_v3_training_guide.md`

## Vocoder capability surface

`tasks/tts/vocoder_infer/base_vocoder.py` exposes:

- `supports_native_streaming()`
- `reset_stream()`
- `spec2wav_stream()`

The shipped HiFiGAN wrappers still report that native streaming is **not**
enabled in the mainline. Runtime therefore uses bounded left-context recompute
instead of pretending the vocoder is truly stateful.
