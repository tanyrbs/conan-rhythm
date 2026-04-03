# Streaming inference notes

This directory now exposes a small runtime utility layer for the Conan streaming path.

## What is safe in mainline

- Emformer frontend runs in real stateful streaming mode.
- Acoustic model still recomputes over the emitted prefix.
- Vocoder wrappers are still stateless by default.

So the current mainline is suitable for **streaming-oriented evaluation**, not for claiming **true end-to-end low-latency stateful deployment**.

## New pieces

- `inference/streaming_runtime.py`
  - resolves `vocoder_left_context_frames` with fallback to legacy `vocoder_stream_context`
  - reports theoretical latency / recompute multipliers
  - provides lightweight `PrefixCodeBuffer` and `RollingMelContextBuffer`
- `inference/run_streaming_latency_report.py`
  - prints the theoretical latency report without loading checkpoints

## Usage

```bash
python inference/run_streaming_latency_report.py
python inference/run_streaming_latency_report.py --config egs/conan_emformer.yaml --duration_seconds 5
```

## Vocoder capability surface

`tasks/tts/vocoder_infer/base_vocoder.py` now exposes:

- `supports_native_streaming()`
- `reset_stream()`
- `spec2wav_stream()`

The shipped HiFiGAN wrappers explicitly report that native streaming is **not** enabled in mainline. The runtime therefore falls back to bounded left-context recompute instead of pretending the vocoder path is truly stateful.
