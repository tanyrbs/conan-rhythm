# Streaming inference notes

This directory is a runtime/eval helper surface for the maintained
**`rhythm_v3`** branch.

Authoritative current docs:

- `README.md`
- `docs/rhythm_migration_plan.md`

## Current maintained inference path

The maintained default inference path is the same as the maintained training
mainline:

- `rhythm_v3_backbone: prompt_summary`
- `rhythm_v3_warp_mode: none`
- `rhythm_v3_anchor_mode: source_observed`

`inference/Conan.py` now prepares explicit prompt-unit conditioning for v3:

- `prompt_content_units`
- `prompt_duration_obs`
- `prompt_unit_mask`

The maintained v3 inference story is:

- prompt distilled into static summary conditioning
- source-observed sealed-unit anchors
- strict-causal prefix-rate state
- deterministic carry-only projection

## What this directory currently serves

Maintained helpers:

- `inference/Conan.py`
- `inference/run_voice_conversion.py`
- `inference/run_streaming_latency_report.py`
- `inference/streaming_runtime.py`

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

Treat them as archive/compatibility surfaces only.

## Vocoder capability surface

`tasks/tts/vocoder_infer/base_vocoder.py` exposes:

- `supports_native_streaming()`
- `reset_stream()`
- `spec2wav_stream()`

The shipped HiFiGAN wrappers still report that native streaming is **not**
enabled in the mainline. Runtime therefore uses bounded left-context recompute
instead of pretending the vocoder is truly stateful.
