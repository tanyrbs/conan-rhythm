# rhythm_v3 falsification log

## 2026-04-14 runtime-clean rerun

### Scope

- maintained `rhythm_v3` minimal-V1 only
- local quick-ARCTIC only
- zero-train Gate0 / Gate1 rerun after runtime-contract cleanup
- frozen falsification family:
  - `raw_median`
  - `weighted_median`
  - `trimmed_mean`

### Protocol cleanup in this pass

- Gate1 case selection and monotonicity ordering now use runtime-equivalent
  `prompt_g_ref`, not display-only `prompt_tempo_ref`
- Gate0 source-prefix audit now uses the shared source-prefix contract, and this
  rerun fixes the prefix mode at `exact_global_family`
- Gate0 now reports:
  - hostile vs clean protocol slices
  - mean-based and median-based totals
  - runtime-clipped analytic/residual views
- Gate1 now reports three runtime layers:
  - `tempo_out_preclip`
  - `tempo_out_continuous`
  - `tempo_out_projected`
- silent counterfactual probe now uses the same `prompt_g_ref` ordering

### What is no longer a valid explanation

The repaired local surface still contradicts the old explanation:

- "Gate fails because support collapses before `g` exists"

Current evidence:

- Gate0 valid prompt-domain rows remain `63/64`
- `mean_support_count = 4.8281`
- Gate0 / Gate1 still fail after contract cleanup

### Current Gate0 result

This local surface has no clean total-claim rows:

- `clean_total_claim_items = 0/64`
- `protocol_misaligned_items = 64/64`

So the current Gate0 reading comes entirely from the hostile slice
`cross_text_prompt_vs_cross_speaker_target`.

Summary from `tmp/gate_reaudit_20260414_runtime_clean/gate0_*/report.json`:

| Variant | total median slope | total mean slope | analytic median slope | analytic runtime mean slope | residual runtime mean slope |
| --- | ---: | ---: | ---: | ---: | ---: |
| `raw_median` | `0.0000` | `-0.0253` | `0.8418` | `0.0415` | `-0.2082` |
| `weighted_median` | `0.0000` | `-0.0253` | `0.7019` | `0.0612` | `-0.2070` |
| `trimmed_mean` | `-0.0000` | `-0.0351` | `0.9757` | `0.0369` | `-0.2545` |

Additional context:

- `valid_zero_total_median_items = 42/63`
- runtime analytic saturation remains high:
  - `raw_median`: `0.7891`
  - `weighted_median`: `0.7741`
  - `trimmed_mean`: `0.8340`

Reading:

- unclipped analytic signal is real
- runtime-aligned analytic signal becomes weak
- runtime residual stays negative
- total signal remains flat or slightly inverse on the only local slice we have

This is enough to reject the maintained engineering claim on this surface.
It is not, by itself, a clean theory-terminal falsifier, because the local
surface provides no `clean_total_claim` rows.

### Current Gate1 result

Summary from `tmp/gate_reaudit_20260414_runtime_clean/gate1_*/summary.json`:

| Variant | preclip pass | continuous pass | projected pass | mean preclip slope | mean projected slope | mean projected range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `raw_median` | `0/4` | `0/4` | `0/4` | `-0.1322` | `-0.0290` | `0.0333` |
| `weighted_median` | `0/4` | `0/4` | `0/4` | `-0.1721` | `-0.0743` | `0.1233` |
| `trimmed_mean` | `0/4` | `0/4` | `0/4` | `-0.2438` | `-0.0726` | `0.0646` |

Supporting telemetry:

- mean analytic saturation:
  - `raw_median`: `0.8067`
  - `weighted_median`: `0.6597`
  - `trimmed_mean`: `0.7990`
- mean projector boundary-hit rate:
  - `raw_median`: `0.3245`
  - `weighted_median`: `0.3810`
  - `trimmed_mean`: `0.3144`
- projected exec-ratio / exec-logstretch slopes stay positive on average

Reading:

- Gate1 is not failing only because of projector quantization
- `preclip` already fails, so the analytic operational cue is wrong-signed or
  collapsed before projection
- projector and discrete readout still compress the signal further, but they
  are downstream amplifiers of an earlier failure

### Silent counterfactual regression

Summary from `tmp/gate_reaudit_20260414_runtime_clean/gate1_silent_raw/summary.json`:

- the script now uses the same `prompt_g_ref` contract
- two sources remain anti-monotone
- two sources collapse to zero-range outputs
- `monotone_by_neg_delta_g` passes on all four sources

Reading:

- the old Gate1 helper mismatch is no longer the best explanation
- the remaining failure is consistent with sign inversion or collapse

### Current conclusion

The maintained conclusion after this runtime-clean rerun is:

- keep Gate 2 blocked
- keep Gate 3 blocked
- keep formal training blocked on this local surface
- stop describing the failure as support-domain collapse
- state the narrower conclusion:
  the maintained single-scalar raw-duration mainline fails under a sharper
  runtime-aligned protocol

The important caveat is also fixed in writing:

- this local rerun does not provide a clean total-claim slice
- so it is strong evidence against the maintained line, not a universal proof
  that every single-scalar mixed global cue is dead

### Retained script surface

The maintained zero-train diagnostics are now:

- `scripts/preflight_rhythm_v3.py`
- `scripts/audit_rhythm_v3_boundary_support.py`
- `scripts/audit_rhythm_v3_counterfactual_static_gate0.py`
- `scripts/probe_rhythm_v3_gate1_analytic.py`
- `scripts/probe_rhythm_v3_gate1_silent_counterfactual.py`
- `scripts/rhythm_v3_probe_cases.py`
- `scripts/rhythm_v3_debug_records.py`
