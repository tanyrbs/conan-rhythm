# rhythm_v3 falsification log

## 2026-04-14 deep follow-up rerun

### Scope

- maintained `rhythm_v3` minimal-V1 only
- local quick-ARCTIC only
- zero-train Gate0 / Gate1 rerun after deeper contract cleanup

### What was fixed in this pass

- Gate1 prompt tempo display now reuses prompt-side weighting:
  `prompt_global_weight`
- Gate1 source tempo readouts now reuses source-side weighting:
  `source_run_stability`
- Gate0 audit defaults to the clean locally-constructed slice:
  `reference_mode=target_as_ref`
- Gate0 gained two missing diagnostics:
  - speech-total duration log-ratio
  - affine runtime residual
- debug/review records now carry prompt-side weight sidecars needed by weighted
  prompt tempo reconstruction
- training target construction now reuses
  `build_causal_source_prefix_rate_seq(...)` instead of hardcoding EMA-only
  prefix targets

### What changed in the reading

Two earlier explanations are now substantially narrowed:

1. "Gate1 is all fail"

This is no longer true locally.

For `weighted_median + exact_global_family`, the analytic Gate1 probe now passes:

- preclip: `4/4`
- continuous: `4/4`
- projected: `4/4`

2. "Gate0 only fails because the protocol slice is dirty"

This is also no longer enough.

The audit now runs the clean local slice by default and still fails:

- `clean_total_claim_items = 64`
- `valid_clean_total_claim_items = 63`
- `valid total median slope = 0.0000`
- `valid total mean slope = -0.0216`
- `valid total duration-logratio slope = 0.0124`

So the stronger current reading is:

- protocol dirtiness was a real contamination factor
- after cleaning it up locally, the maintained total-control claim still does
  not recover

### Clean-slice Gate0 reading

Strongest current local clean-slice result:

- variant: `weighted_median`
- prefix mode: `exact_global_family`
- candidate token: `57`
- `drop_edge_runs_for_g = 1`

Key diagnostics:

| Metric | Value |
| --- | ---: |
| valid analytic median slope | `0.1672` |
| valid analytic runtime mean slope | `0.0000` |
| valid residual mean slope | `-0.1636` |
| valid residual runtime mean slope | `-0.0216` |
| valid residual runtime affine slope | `0.0100` |
| mean analytic saturation | `1.0000` |

Interpretation:

- there is still analytic signal before runtime saturation
- the runtime-aligned analytic surface remains heavily flattened
- the affine residual view weakens the old "residual is strongly anti-global"
  story, so part of the residual negativity was decomposition-baked
- even after this cleanup, total evidence is still too weak to pass Gate0

### Gate1 reading

Current strongest local Gate1 result:

- variant: `weighted_median`
- prefix mode: `exact_global_family`
- projected pass: `4/4`

Additional runtime probe:

- variant: `raw_median`
- prefix mode: `exact_global_family`
- `analytic_gap_clip=0.6`
- projected pass: `3/4`

Interpretation:

- runtime clip still matters
- `weighted_median` is stronger than `raw_median` on this local surface
- Gate1 is no longer the main blocker on the strongest fixed surface

### Current conclusion

The maintained conclusion after the deeper follow-up rerun is:

- keep Gate 2 blocked
- keep Gate 3 blocked
- keep formal training blocked on this local surface
- stop using "support collapse" as the primary explanation
- stop using "Gate1 all fail" as the primary explanation
- use the sharper local reading instead:
  the strongest remaining blocker is Gate0 total control, not Gate1 ordering

The project should now describe the local falsification state as:

- the analytic runtime path can be made locally operational on the strongest
  fixed variant
- the maintained single-scalar raw-duration line still does not establish a
  strong enough total-target claim on the cleaner Gate0 slice

## Retained script surface

The maintained zero-train diagnostics remain:

- `scripts/preflight_rhythm_v3.py`
- `scripts/audit_rhythm_v3_boundary_support.py`
- `scripts/audit_rhythm_v3_counterfactual_static_gate0.py`
- `scripts/probe_rhythm_v3_gate1_analytic.py`
- `scripts/probe_rhythm_v3_gate1_silent_counterfactual.py`
- `scripts/rhythm_v3_probe_cases.py`
- `scripts/rhythm_v3_debug_records.py`
