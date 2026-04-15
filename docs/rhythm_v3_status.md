# rhythm_v3 Status

Date: 2026-04-15

## Maintained Mainline

The repository currently maintains one rhythm mainline only: `rhythm_v3`.

The maintained official online contract is:

- prompt/reference cue: `weighted_median`
- source prefix state: `ema`
- source init: `first_speech`
- prompt policy: `meaningful_reference`
- source anchor: `source_observed`
- maintained projector surface: `greedy | prefix_optimal`

The checked-in official strict fingerprint currently binds:

- `rhythm_v3_projection_mode=greedy`
- `rhythm_v3_integer_projection_mode=greedy`

Official entrypoints:

- `egs/conan_emformer_rhythm_v3.yaml`
- `egs/overrides/rhythm_v3_official_strict_gate.yaml`
- `egs/overrides/rhythm_v3_gate_status.json`

## Current Project Status

Official truth:

- strict official training is blocked
- official gate bundle status is `blocked_pending_gate0_gate1`
- `gate0_pass=false`
- `gate1_pass=false`
- `gate2_pass=false`
- `gate3_pass=false`
- contract id: `795e286bac5bee76`

Current technical read:

- the repo is not in a fully broken state
- local diagnostics show the global cue can survive on stronger non-official surfaces
- the maintained online writer can show some pre-projection movement
- execution/projector behavior still appears to flatten too much of that signal
- therefore the official online contract is still not cleared

## Official Vs Local

Keep these separate:

- official maintained online contract:
  `weighted_median + ema + first_speech`
- local upper-bound diagnostic contract:
  `weighted_median + exact_global_family`

The upper-bound contract is useful evidence that cue signal exists. It is not the maintained online contract and must not be used as official gate evidence.

All local experiments now belong under:

- `egs/experiments/local/`

Those files are diagnostics only.

## Gate Library

Maintained gate semantics now live in:

- `tasks/Conan/rhythm/duration_v3/gate_status.py`

The export script:

- `scripts/rhythm_v3_debug_records.py`

is a CLI and bundle-export surface over that library, not the source of truth for gate criteria.

## Current Priority

The next step is not to expand candidate surfaces. The priority is to clear the maintained online contract itself under the official fingerprint:

1. rerun Gate1 on the maintained online contract
2. clear Gate2 on that same contract
3. clear Gate3 on that same contract
4. only then reopen stricter downstream training claims
