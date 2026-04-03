# Streaming low-latency mainline note — 2026-04-03

## Scope

This refactor does **not** switch mainline Conan inference to a true end-to-end low-latency stateful streaming stack.

That would require at least:

- cache-based incremental acoustic decoding
- native streaming vocoder inference
- quality regression for the changed acoustic/vocoder path

Those pieces are not established in the current repository, so mainline stays on the safe path.

## What changed

1. Streaming runtime utilities were centralized in `inference/streaming_runtime.py`.
2. `inference/Conan.py` no longer does repeated Python-side full-prefix concatenation for content codes.
3. Non-streaming vocoder fallback is now explicit:
   - keep only bounded left mel context
   - recompute over that bounded window
   - crop the newly generated samples
4. Vocoder wrappers now expose a capability surface:
   - `supports_native_streaming()`
   - `reset_stream()`
   - `spec2wav_stream()`
5. A latency CLI was added:
   - `python inference/run_streaming_latency_report.py`

## Interpretation

Current mainline should be described as:

- **stateful streaming frontend**
- **prefix-recompute acoustic backend**
- **stateless vocoder with bounded left-context recompute**

That is appropriate for streaming-oriented evaluation and engineering instrumentation. It is not honest to market it as a full-chain native low-latency streaming deployment path.

## Latency accounting

The runtime utility reports:

- mel frame duration from `audio_sample_rate` and `hop_size`
- chunk frames from `chunk_size`
- algorithmic first-packet latency = `chunk_size + right_context`
- steady-state vocoder window / recompute multiplier
- cumulative acoustic prefix recompute multiplier for a given utterance duration

If a future vocoder wrapper enables `supports_native_streaming() == True`, the report automatically collapses the vocoder recompute multiplier to `1.0`.
