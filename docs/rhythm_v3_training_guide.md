# rhythm_v3 Training Guide

This guide describes the maintained training surface only.

## Maintained Entry Points

Use these files for the maintained V1 line:

- `egs/conan_emformer_rhythm_v3.yaml`
- `egs/overrides/rhythm_v3_gate1_analytic.yaml`
- `egs/overrides/rhythm_v3_gate2_coarse_only.yaml`
- `egs/overrides/rhythm_v3_gate3_learned.yaml`
- `egs/overrides/rhythm_v3_official_strict_gate.yaml`

The maintained online contract is:

- prompt/reference cue: `weighted_median`
- source prefix state: `ema`
- source init: `first_speech`
- prompt policy: `meaningful_reference`
- source anchor: `source_observed`
- integer projector: `greedy`

The base recipe keeps `rhythm_v3_gate_quality_strict=false`. That is intentional. Repair and diagnostic runs should use the maintained configs without mutating the checked-in official gate bits.

Official claim runs must add:

- `egs/overrides/rhythm_v3_official_strict_gate.yaml`

That overlay binds the run to:

- `egs/overrides/rhythm_v3_gate_status.json`

## Current Official Status

As of 2026-04-15:

- official strict training is blocked
- official status file scope is `official_maintained_online_contract`
- the checked-in official contract id is `8052e96e745d5aad`
- `gate0_pass=false`
- `gate1_pass=false`
- `gate2_pass=false`
- `gate3_pass=false`

Do not replace the official gate bundle with local-candidate JSON files. The strict validator now rejects local candidate gate bundles on the official path.

## Training Stages

Treat the maintained line as one online contract with staged validation and training modes:

1. Gate1 analytic
2. Gate2 coarse-only
3. Gate3 learned
4. prefix fine-tune only after the upstream gates are actually cleared

The stages are not separate systems. They are staged views over the same maintained online contract.

## Recommended Usage

For repository-default work:

- start from `egs/conan_emformer_rhythm_v3.yaml`
- use checked-in gate overlays instead of hand-editing the base yaml
- keep official strict gate off during repair runs
- enable strict gate only when you are intentionally testing the official claim surface

For machine-local quick experiments:

- copy `egs/local_examples/local_arctic_rhythm_v3_quick.example.yaml`
- write dataset-specific paths into an untracked local yaml
- keep local candidate configs under `egs/experiments/local/`

## Official Vs Diagnostic Surfaces

The repo keeps a hard distinction:

- official maintained online contract:
  `weighted_median + ema + first_speech`
- stronger local upper-bound diagnostic contract:
  `weighted_median + exact_global_family`

The second one is for proving the cue exists. It is not the maintained online truth and does not unblock official training.

## Runtime Reading

The maintained writer is a source-anchored retimer:

- prompt side contributes a speech-focused global cue
- source side keeps a strict-causal prefix-rate state
- silence remains a clipped coarse follower
- projection is deterministic and budgeted

That is the only reading this guide endorses. Legacy alias language, dated candidate overlays, and checkpoint-specific local diagnostics are intentionally omitted from the maintained training documentation.

## Repo Pointers

- runtime code: `modules/Conan/rhythm_v3/`
- task/data contract: `tasks/Conan/rhythm/duration_v3/`
- tests: `tests/rhythm/`
- current project status: `docs/rhythm_v3_status.md`
- validation and gate workflow: `docs/rhythm_v3_validation_stack.md`
