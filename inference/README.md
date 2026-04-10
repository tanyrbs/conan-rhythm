# Streaming inference notes

This directory is a **runtime/eval helper surface**, not the authoritative
training or mechanism document. For the current maintained branch semantics,
use:

- `README.md`
- `docs/rhythm_migration_plan.md`

## What this directory currently serves

The maintained inference/runtime helpers are:

- `inference/Conan.py`
- `inference/run_voice_conversion.py`
- `inference/run_streaming_latency_report.py`
- `inference/streaming_runtime.py`

They now point at the current `rhythm_v3` mainline, not the old v2 teacher path.

## Current status of streaming

The honest runtime status is:

- Emformer frontend runs in real stateful streaming mode
- rhythm execution is prefix/commit oriented and uses the current `rhythm_v3` runtime
- acoustic decoding still recomputes over the emitted prefix
- shipped vocoder wrappers are still stateless by default

So this directory is suitable for:

- **streaming-oriented evaluation**
- **latency accounting**
- **prefix/commit behavior inspection**

It is **not** yet evidence of fully stateful end-to-end low-latency deployment.

## Maintained entrypoints

### `inference/Conan.py`

Current streaming VC helper built on:

- `modules/Conan/Conan.py`
- `rhythm_v3`
- explicit prompt-unit extraction for v3 inference
- prompt-side boundary / phrase sidecars for detector-bank experiments
- `prepare_rhythm_reference(...)` is now v2-only

### `inference/run_voice_conversion.py`

Batch helper that now imports:

- `from inference.Conan import StreamingVoiceConversion`

This is a convenience runner, not the training-contract authority.

### `inference/run_streaming_latency_report.py`

Latency/recompute report helper.

Current default config:

- `egs/conan_emformer_rhythm_v3.yaml`

## Recommended config for maintained branch checks

For current branch-level runtime checks, prefer:

- `egs/conan_emformer_rhythm_v3.yaml`

Legacy v2 configs may still exist for archive/compatibility checks, but they
are not the recommended maintained branch runtime surface anymore.

## Legacy note

Legacy helpers remain in this directory, including:

- `inference/Conan_previous.py`

Treat them as compatibility/archive surfaces only.

## Vocoder capability surface

`tasks/tts/vocoder_infer/base_vocoder.py` exposes:

- `supports_native_streaming()`
- `reset_stream()`
- `spec2wav_stream()`

The shipped HiFiGAN wrappers still report that native streaming is **not**
enabled in mainline. The runtime therefore uses bounded left-context recompute
instead of pretending the vocoder is truly stateful.
