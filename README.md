# Conan Rhythm

This repository now keeps one maintained rhythm mainline only: `rhythm_v3`.

The maintained V1 contract is:

- prompt/reference global cue: `weighted_median`
- source prefix state: `ema`
- source rate init: `first_speech`
- prompt policy: `meaningful_reference`
- source anchor: `source_observed`
- maintained projector surface: `greedy | prefix_optimal`

The checked-in official strict fingerprint currently uses:

- `rhythm_v3_projection_mode=greedy`

The official maintained online path is configured by:

- `egs/conan_emformer_rhythm_v3.yaml`
- `egs/overrides/rhythm_v3_official_strict_gate.yaml`
- `egs/overrides/rhythm_v3_gate_status.json`

Current status on 2026-04-15:

- official strict training is still blocked
- official gate bundle remains `gate0=false`, `gate1=false`, `gate2=false`, `gate3=false`
- local upper-bound evidence exists, but it is diagnostic only and does not unblock the official online contract
- local experiment configs and gate bundles now live under `egs/experiments/local/`

Authoritative repo docs:

- `docs/rhythm_v3_status.md`
- `docs/rhythm_v3_training_guide.md`
- `docs/rhythm_v3_validation_stack.md`

## Maintained Reading

`rhythm_v3` is a source-anchored, strict-causal retimer over a stabilized run lattice.

The maintained V1 reading is intentionally narrow:

- prompt side provides a speech-focused global cue
- source side stays state-sufficient across chunk continuation
- silence is not an independent pause planner
- official claims only count when the checked-in strict gate overlay and official gate-status JSON are used together

The base config keeps strict gating off so repair runs and diagnostics are not blocked by the checked-in pass bits. Official claims must use the strict overlay.

## Official Vs Local

Use the repository surfaces like this:

- official maintained base: `egs/conan_emformer_rhythm_v3.yaml`
- official strict gate overlay: `egs/overrides/rhythm_v3_official_strict_gate.yaml`
- official gate bundle: `egs/overrides/rhythm_v3_gate_status.json`
- local experiments only: `egs/experiments/local/`
- local example quick config: `egs/local_examples/local_arctic_rhythm_v3_quick.example.yaml`
- local ARCTIC/L2-ARCTIC metadata builder: `scripts/build_local_arctic_l2_processed_metadata.py`

## Local Quick Data Build

For a machine-local ARCTIC/L2-ARCTIC quick path:

1. Build processed metadata and pair manifests with `scripts/build_local_arctic_l2_processed_metadata.py`.
2. Point an untracked local config at the generated `processed_data_dir` / `binary_data_dir`.
3. Run `python -m data_gen.tts.runs.binarize --config <local-yaml>`.

The maintained repo surface does not ship machine-specific paths. Keep local quick configs in ignored `egs/*.local.yaml` files.

Local candidate configs, local gate JSON snapshots, and checkpoint-specific findings are diagnostics only. They are not part of the maintained online contract and must not be reused as official unblock evidence.

## Repo Layout

- `modules/Conan/rhythm_v3/`: maintained runtime implementation
- `tasks/Conan/rhythm/duration_v3/`: maintained task and dataset contract
- `tasks/Conan/rhythm/duration_v3/gate_status.py`: maintained gate-status and review-contract library
- `egs/`: maintained configs and local experiment overlays
- `inference/`: runtime and evaluation helpers
- `scripts/rhythm_v3_debug_records.py`: maintained review/export CLI over the gate-status library
- `tests/rhythm/`: regression coverage for runtime, task config, losses, metrics, and projector invariants

## Current Blocker

The current blocker is no longer prompt-domain collapse. The main blocker is still the maintained online execution path:

- local reviews show cue survival on stronger diagnostic surfaces
- the maintained online writer can show some pre-projection movement
- execution/projector behavior still appears to flatten too much of that signal
- therefore the official gate bundle remains blocked until the maintained online contract itself is rerun and passes under the official fingerprint
