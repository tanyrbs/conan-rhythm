# rhythm_v3 Validation Stack

This document defines the maintained validation surface for `rhythm_v3`.

## Official Validation Surface

The maintained official online contract is:

- prompt/reference cue: `weighted_median`
- source prefix state: `ema`
- source init: `first_speech`
- prompt policy: `meaningful_reference`
- source anchor: `source_observed`
- maintained projection surface: `greedy | prefix_optimal`

The checked-in official strict fingerprint currently uses:

- `rhythm_v3_projection_mode=greedy`
- `rhythm_v3_integer_projection_mode=greedy`

Official validation is anchored by:

- `egs/overrides/rhythm_v3_official_strict_gate.yaml`
- `egs/overrides/rhythm_v3_gate_status.json`

The checked-in official gate bundle is intentionally blocked on 2026-04-15. That is the current truth of record for official training.

## What Counts As Maintained

Maintained validation surfaces:

- `Gate1-online`: `egs/overrides/rhythm_v3_gate1_analytic.yaml`
- `Gate2`: `egs/overrides/rhythm_v3_gate2_coarse_only.yaml`
- `Gate3`: `egs/overrides/rhythm_v3_gate3_learned.yaml`
- strict official claim overlay: `egs/overrides/rhythm_v3_official_strict_gate.yaml`

Diagnostic-only surfaces:

- local upper-bound exact-family checks
- local candidate Gate2 and Gate3 configs
- dated local gate JSON snapshots under `egs/experiments/local/`

Those diagnostic surfaces help explain failure modes, but they are not official claim surfaces.

## Current Read On The Project

As of 2026-04-15:

- official strict training remains blocked
- local upper-bound evidence indicates the cue is not dead
- the maintained online path still needs the same contract to pass Gate1, Gate2, and Gate3 under the official fingerprint
- execution/projector behavior remains the leading bottleneck rather than prompt-cue existence

The correct interpretation is therefore:

- the repository is not globally broken
- the official maintained online contract is not yet cleared
- local diagnostic wins must not be collapsed into official gate truth

## Validation Workflow

The maintained workflow is:

1. run the maintained online contract
2. export debug records and gate summaries
3. compare against the official gate-status fingerprint
4. use local diagnostics only to explain failures, not to replace the official contract

Keep the distinction between these two questions:

- does a stronger diagnostic surface show usable cue signal?
- did the maintained online contract itself clear the official gate?

Only the second question controls official training readiness.

## Primary Surfaces

Maintained code and utility surfaces:

- runtime: `modules/Conan/rhythm_v3/`
- task/runtime glue: `tasks/Conan/rhythm/duration_v3/`
- gate-status library: `tasks/Conan/rhythm/duration_v3/gate_status.py`
- export CLI: `scripts/rhythm_v3_debug_records.py`
- review utilities: `utils/plot/rhythm_v3_viz/`
- regression tests: `tests/rhythm/`

Local experiments belong under:

- `egs/experiments/local/`

They are intentionally separated from `egs/overrides/` so the maintained config surface stays singular.

## Gate Discipline

The gate bundle must describe one object only: the maintained online contract.

The maintained source of truth for gate calculations is the library layer:

- `tasks/Conan/rhythm/duration_v3/gate_status.py`

The CLI:

- `scripts/rhythm_v3_debug_records.py`

must stay a thin export surface over that library.

That means:

- contract fingerprints must match the actual runtime surface
- strict official gating must not read local-candidate JSON
- local upper-bound exact-family evidence must stay separate
- documentation must not present local candidate outcomes as official readiness

## Status Reference

For the current status summary, read:

- `docs/rhythm_v3_status.md`
