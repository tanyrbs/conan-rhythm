# rhythm_v3 local status snapshot (2026-04-14)

This document records the current maintained local rerun after the latest
runtime/audit contract cleanup, source-prefix baseline alignment, and zero-train
gate rerun.

## 1. Scope

- maintained `rhythm_v3` minimal-V1 path only
- local quick-ARCTIC configs only
- zero-train falsification only
- frozen falsification family for this pass:
  - `raw_median`
  - `weighted_median`
  - `trimmed_mean`
- Gate 2 / Gate 3 training still blocked

## 2. What changed in this round

- runtime, review, and Gate1 case selection now use the same prompt-side `g`
  contract
- source-prefix baseline gained explicit modes:
  - `ema`
  - `family_hybrid`
  - `exact_global_family`
- this falsification rerun uses `rhythm_v3_src_prefix_stat_mode=exact_global_family`
  for Gate0 and Gate1
- Gate0 static audit now reports:
  - protocol slices: `clean_total_claim` vs
    `cross_text_prompt_vs_cross_speaker_target`
  - median-based and mean-based total/analytic/residual views
  - runtime-clipped analytic/residual views aligned with writer clip contract
- Gate1 analytic probe now:
  - selects slow/mid/fast by runtime-equivalent `prompt_g_ref`
  - sorts monotonicity by `prompt_g_ref`, not display-only `prompt_tempo_ref`
  - reports `tempo_out_preclip`, `tempo_out_continuous`,
    `tempo_out_projected`
  - reports continuous count/logstretch readouts in parallel with discrete
    `tempo_out`
- silent counterfactual probe now uses the same `prompt_g_ref` contract as the
  analytic probe

## 3. Current artifacts

Current rerun artifacts live in:

- `tmp/gate_reaudit_20260414_runtime_clean/`

Primary files:

- `tmp/gate_reaudit_20260414_runtime_clean/gate0_raw/report.json`
- `tmp/gate_reaudit_20260414_runtime_clean/gate0_weighted/report.json`
- `tmp/gate_reaudit_20260414_runtime_clean/gate0_trimmed/report.json`
- `tmp/gate_reaudit_20260414_runtime_clean/gate1_raw/summary.json`
- `tmp/gate_reaudit_20260414_runtime_clean/gate1_weighted/summary.json`
- `tmp/gate_reaudit_20260414_runtime_clean/gate1_trimmed/summary.json`
- `tmp/gate_reaudit_20260414_runtime_clean/gate1_silent_raw/summary.json`

## 4. Frontend/data-contract check

Latest preflight and boundary audit still support the same conclusion as the
previous repaired surface:

- prompt/source silence sidecars are present even when raw `silent_token`
  evidence is absent
- boundary support is restored on the local quick surface
- Gate0 valid prompt-domain rows remain `63/64`

Interpretation:

- the old "support collapses before `g` exists" story is no longer the main
  blocker on this local surface
- the current failure is downstream of support recovery

## 5. Gate verdict

| Gate | Verdict | Why |
| --- | --- | --- |
| Gate 0 | fail | prompt-domain validity is restored, but this local surface exposes only the hostile protocol slice, and total explainability stays flat or slightly negative there |
| Gate 1 | fail | with `prompt_g_ref` ordering and `exact_global_family` prefix baseline, `preclip`, `continuous`, and `projected` all fail monotonicity for `raw`, `weighted_median`, and `trimmed_mean` |
| Gate 2 | blocked | Gate 0 / Gate 1 still fail |
| Gate 3 | blocked | Gate 2 not admissible |

## 6. Gate0 static audit

The cleaned Gate0 result is more precise than the earlier rebuilt2 report:

- `g_domain_valid_items = 63/64`
- `mean_support_count = 4.8281`
- `valid_zero_total_median_items = 42/63`
- `clean_total_claim_items = 0/64`
- `protocol_misaligned_items = 64/64`

So this local surface currently provides no clean total-claim slice. Gate0 can
only judge the hostile slice
`cross_text_prompt_vs_cross_speaker_target`.

| Variant | total median slope | total mean slope | analytic median slope | analytic runtime mean slope | residual runtime mean slope | mean saturation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `raw_median` | `0.0000` | `-0.0253` | `0.8418` | `0.0415` | `-0.2082` | `0.7891` |
| `weighted_median` | `0.0000` | `-0.0253` | `0.7019` | `0.0612` | `-0.2070` | `0.7741` |
| `trimmed_mean` | `-0.0000` | `-0.0351` | `0.9757` | `0.0369` | `-0.2545` | `0.8340` |

Interpretation:

- unclipped analytic signal is still positively aligned with `delta_g`
- after runtime clip alignment, analytic mean slope shrinks toward zero
- runtime residual mean slope stays negative
- the exact-family prefix baseline does not rescue total signal on this surface
- because `clean_total_claim` is absent locally, this is enough to fail the
  maintained engineering claim, but it is not a terminal theory-proof against
  every single-scalar mixed-cue hypothesis

## 7. Gate1 analytic runtime probe

All Gate1 runs below use:

- `prompt_g_ref`-ordered slow/mid/fast triplets
- `rhythm_v3_src_prefix_stat_mode=exact_global_family`
- analytic-only runtime mode

| Variant | preclip pass | continuous pass | projected pass | mean preclip slope | mean projected slope | mean projected range | mean saturation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `raw_median` | `0/4` | `0/4` | `0/4` | `-0.1322` | `-0.0290` | `0.0333` | `0.8067` |
| `weighted_median` | `0/4` | `0/4` | `0/4` | `-0.1721` | `-0.0743` | `0.1233` | `0.6597` |
| `trimmed_mean` | `0/4` | `0/4` | `0/4` | `-0.2438` | `-0.0726` | `0.0646` | `0.7990` |

Additional continuous readouts confirm the failure is not a pure metric artifact:

- projected exec-ratio slope stays positive on average:
  - `raw_median`: `0.1680`
  - `weighted_median`: `0.3192`
  - `trimmed_mean`: `0.1251`
- projected exec-logstretch slope also stays positive on average:
  - `raw_median`: `0.1502`
  - `weighted_median`: `0.2925`
  - `trimmed_mean`: `0.1391`

Interpretation:

- failure begins before projector:
  `tempo_out_preclip` is already non-monotone and usually anti-monotone
- projector and discrete readout flatten the remaining dynamic range further,
  but they are not the sole failure source
- the maintained analytic chain does still move some continuous duration mass,
  yet the operational cue exposed to Gate1 remains wrong-signed or collapsed

## 8. Silent counterfactual regression check

The silent counterfactual probe was rerun only for `raw_median` to verify the
new `prompt_g_ref` contract on the sibling script surface.

Result:

- `src_prefix_stat_mode = exact_global_family`
- `aba_train_arctic_a0022` and `bdl_train_arctic_a0022` stay anti-monotone
  with negative `prompt_g_ref -> tempo_out` slopes
- `asi_train_arctic_a0020` and `slt_train_arctic_a0019` collapse to
  zero-range outputs
- `monotone_by_neg_delta_g` passes on all four sources

Interpretation:

- the old selector/sorter drift is no longer the main explanation
- the residual Gate1 failure is now consistent with sign inversion or collapse,
  not a mismatched prompt-ordering helper

## 9. Current reading

The cleaned local reading is:

- support collapse is no longer the main failure mode
- the maintained single-scalar raw-duration chain still fails under a sharper,
  more runtime-aligned protocol
- the exact-family source prefix baseline improves contract cleanliness but does
  not recover Gate0 or Gate1
- Gate1 now fails even before projection, and projection/readout then compress
  the remaining variation further

So the maintained conclusion is:

- keep Gate 2 blocked
- keep Gate 3 blocked
- do not start formal training on this local surface
- do not promote `raw_median`, `weighted_median`, or `trimmed_mean` as a valid
  maintained V1 main control on this surface
- keep the claim narrow:
  the maintained line is engineering-failed here, while clean total-signal
  theory falsification remains unavailable locally because `clean_total_claim`
  rows are absent

The project remains in zero-train falsification mode.
