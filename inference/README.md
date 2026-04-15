# Streaming Inference Notes

This directory is the maintained inference helper surface for `rhythm_v3`.

Authoritative current docs:

- `README.md`
- `docs/rhythm_v3_status.md`
- `docs/rhythm_v3_training_guide.md`
- `docs/rhythm_v3_validation_stack.md`

## Maintained Inference Contract

Inference follows the same maintained online contract as training:

- `rhythm_v3_backbone: minimal_v1_global`
- `rhythm_v3_anchor_mode: source_observed`
- prompt/reference cue: `weighted_median`
- source prefix state: `ema`
- source init: `first_speech`
- deterministic greedy integer projection with prefix budget

The maintained reading is narrow:

- prompt side supplies a speech-focused global cue
- source side remains strict-causal and state-sufficient
- silence remains a coarse-only follower rather than a separate pause planner

## Maintained Helpers

- `inference/Conan.py`
- `inference/run_voice_conversion.py`
- `inference/run_streaming_latency_report.py`
- `inference/streaming_runtime.py`

`StreamingVoiceConversion.infer_once(...)` can export a serializable debug bundle for the same review utilities used by the validation stack.

## Current Status

The inference stack is useful for streaming-oriented evaluation and runtime inspection, but it is not yet a claim of fully stateful end-to-end low-latency deployment. The maintained official online contract is still blocked by the checked-in gate bundle.
