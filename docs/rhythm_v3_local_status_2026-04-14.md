# rhythm_v3 local status snapshot (2026-04-14)

This document records the latest maintained local rerun after the Gate1
direction fix, source-boundary contract cleanup, prefix-final audit cleanup,
and zero-train gate rerun.

## 1. Scope

- maintained `rhythm_v3` minimal-V1 path only
- local quick-ARCTIC configs only
- zero-train falsification only
- local execution is still CPU-only in this workspace:
  the host machine has `GTX 1050 Ti`, but the repo `.venv` currently ships
  `torch 2.5.1+cpu`, so no CUDA path was available for this rerun
- frozen falsification family for this pass:
  - `raw_median`
  - `weighted_median`
  - `trimmed_mean`
- Gate 2 / Gate 3 training still blocked

## 2. What changed in this round

- Gate1 case selection labels are corrected:
  smallest `g_ref` is now treated as `fast`, largest as `slow`
- Gate1 monotonicity direction is corrected:
  higher `prompt_g_ref` means slower speech, so `tempo_out` must decrease, not
  increase
- runtime, review, and Gate1 tempo readout now use the same prompt/source-side
  `g` contract, including:
  - `closed_mask`
  - `boundary_confidence`
  - `min_boundary_confidence_for_g`
- source-prefix baseline gained explicit modes:
  - `ema`
  - `family_hybrid`
  - `exact_global_family`
- this falsification rerun uses `rhythm_v3_src_prefix_stat_mode=exact_global_family`
  for Gate0 and Gate1
- source-side runtime now prefers `boundary_confidence` when available and only
  falls back to `source_boundary_cue`
- Gate0 static audit now reports:
  - protocol slices: `clean_total_claim` vs
    `cross_text_prompt_vs_cross_speaker_target`
  - median-based and mean-based total/analytic/residual views
  - runtime-clipped analytic/residual views aligned with writer clip contract
  - final-prefix audit fields:
    `g_src_prefix_final` and `delta_g_ref_minus_src_prefix_final`
- Gate1 analytic probe now:
  - selects slow/mid/fast by runtime-equivalent `prompt_g_ref`
  - sorts monotonicity by `prompt_g_ref`, not display-only `prompt_tempo_ref`
  - reports `tempo_out_preclip`, `tempo_out_continuous`,
    `tempo_out_projected`
  - reports continuous count/logstretch readouts in parallel with discrete
    `tempo_out`
- silent counterfactual probe now uses the same `prompt_g_ref` contract as the
  analytic probe and no longer counts zero-range collapse as a pass

## 3. Current artifacts

Current rerun artifacts live in:

- `tmp/gate_reaudit_20260414_runtime_fixed/`

Primary files:

- `tmp/gate_reaudit_20260414_runtime_fixed/gate0_raw/report.json`
- `tmp/gate_reaudit_20260414_runtime_fixed/gate0_weighted/report.json`
- `tmp/gate_reaudit_20260414_runtime_fixed/gate0_trimmed/report.json`
- `tmp/gate_reaudit_20260414_runtime_fixed/gate1_raw/summary.json`
- `tmp/gate_reaudit_20260414_runtime_fixed/gate1_weighted/summary.json`
- `tmp/gate_reaudit_20260414_runtime_fixed/gate1_trimmed/summary.json`
- `tmp/gate_reaudit_20260414_runtime_fixed/gate1_silent_raw/summary.json`

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
| Gate 1 | mixed / still not promotable | the old all-fail reading was contaminated by a sign bug; after correction, `weighted_median` passes all 4 local triplets end-to-end, while `raw_median` and `trimmed_mean` still collapse on part of the slice |
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
| `raw_median` | `4/4` | `2/4` | `2/4` | `-0.1322` | `-0.0290` | `0.0333` | `0.8067` |
| `weighted_median` | `4/4` | `4/4` | `4/4` | `-0.1721` | `-0.0743` | `0.1233` | `0.6597` |
| `trimmed_mean` | `4/4` | `3/4` | `2/4` | `-0.2438` | `-0.0726` | `0.0646` | `0.7990` |

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

- the previous "Gate1 all fail" conclusion was wrong:
  the sign expectation in the probe was reversed
- negative `prompt_g_ref -> tempo_out` slope is the correct direction because
  `tempo_out = exp(-g)`
- after correcting the direction, all three variants are monotone before clip;
  the main remaining failure is downstream flattening
- `weighted_median` is the strongest local candidate on this slice:
  it survives `preclip`, `continuous`, and `projected` readouts on all 4 cases
- `raw_median` and `trimmed_mean` still lose dynamic range between
  `continuous` and `projected`, so the execution surface is still eating a
  meaningful part of the cue
- Gate1 therefore no longer justifies "analytic chain is dead"; it now supports
  the narrower reading that runtime flattening and variant choice matter

## 8. Silent counterfactual regression check

The silent counterfactual probe was rerun only for `raw_median` to verify the
new `prompt_g_ref` contract on the sibling script surface.

Result:

- `src_prefix_stat_mode = exact_global_family`
- with corrected direction and range gating, `2/4` sources pass
  `prompt_g_ref -> tempo_out`
- `aba_train_arctic_a0022` and `bdl_train_arctic_a0022` pass with the expected
  negative slope
- `asi_train_arctic_a0020` and `slt_train_arctic_a0019` still collapse to
  zero-range outputs and now correctly fail instead of being counted as trivial
  monotone passes

Interpretation:

- the old selector/sorter drift was a real bug
- after fixing it, the remaining silent-counterfactual failure mode is mostly
  zero-range collapse, not sign inversion

## 9. Current reading

The cleaned local reading is:

- support collapse is no longer the main failure mode
- the maintained single-scalar raw-duration chain still fails Gate0 on this
  local surface
- the exact-family source prefix baseline improves contract cleanliness but does
  not recover Gate0 total signal
- Gate1 is no longer a blanket failure:
  `weighted_median` now passes the local 4-case analytic probe end-to-end after
  the direction fix
- the remaining Gate1 weakness is concentrated in runtime flattening and
  variant fragility, not in a universally wrong-signed analytic chain

So the maintained conclusion is:

- keep Gate 2 blocked
- keep Gate 3 blocked
- do not start formal training on this local surface
- do not promote the whole frozen family as a valid maintained V1 main control
  on this surface just because `weighted_median` passes one small local slice
- keep the claim narrow:
  the maintained line is still engineering-failed here because Gate0 stays
  flat and Gate1 is not robust across the frozen family, while clean
  total-signal theory falsification remains unavailable locally because
  `clean_total_claim` rows are absent

The project remains in zero-train falsification mode.
