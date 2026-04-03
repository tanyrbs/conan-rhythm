# Checkpoints

This directory should only keep core reusable runtime assets.

Kept on purpose:
- `Emformer/`
- `hifigan_vc/`
- `rmvpe/`

Cleaned on purpose:
- temporary probe outputs
- debug experiment folders
- preflight scratch folders
- stale local model dumps not referenced by the current maintained configs

Current mainline config expectation:
- `egs/conan_emformer.yaml` -> `checkpoints/Emformer`
- vocoder -> `checkpoints/hifigan_vc`
- RMVPE -> `checkpoints/rmvpe`
