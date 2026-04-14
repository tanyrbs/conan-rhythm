# rhythm_v3 local status snapshot (2026-04-14)

This is the latest local zero-train falsification snapshot after the deeper
contract cleanup pass.

## 1. Scope

- maintained `rhythm_v3` minimal-V1 path only
- local quick-ARCTIC configs only
- zero-train Gate0 / Gate1 reruns only
- local execution is still CPU-only in this workspace:
  the host machine has `GTX 1050 Ti`, but the repo `.venv` currently ships
  `torch 2.5.1+cpu`, so no CUDA path was available
- frozen family used in the latest reruns:
  - `raw_median`
  - `weighted_median`
  - `trimmed_mean`

## 2. What changed in the latest follow-up

- Gate1 prompt ordering and monotonicity remain fixed on runtime-equivalent
  `prompt_g_ref`
- Gate1 tempo readout now also reuses weighted prompt/source contracts:
  - prompt-side `prompt_global_weight`
  - source-side `source_run_stability`
- debug/review records now carry prompt-side weight sidecars needed for
  weighted prompt tempo reconstruction
- Gate0 default protocol now audits the clean local slice first:
  `reference_mode=target_as_ref`
- Gate0 now reports two extra diagnostics that were missing before:
  - speech-total duration log-ratio
  - affine runtime residual view
- training target construction no longer hardcodes an EMA-only prefix path;
  it now reuses `build_causal_source_prefix_rate_seq(...)`, so the target
  builder can follow the same prefix-stat family as runtime
- source-side weighted `g_src` analysis remains aligned across runtime,
  review, and Gate0 audit

## 3. Current artifacts

Latest follow-up artifacts live in:

- `tmp/gate_reaudit_20260414_followup/`

Primary files:

- `tmp/gate_reaudit_20260414_followup/gate0_weighted_report.json`
- `tmp/gate_reaudit_20260414_followup/gate0_weighted_rows.csv`
- `tmp/gate_reaudit_20260414_followup/gate1_weighted_summary.json`
- `tmp/gate_reaudit_20260414_followup/gate1_weighted_rows.csv`
- `tmp/gate_reaudit_20260414_followup/gate1_raw_clip06_summary.json`
- `tmp/gate_reaudit_20260414_followup/gate1_raw_clip06_rows.csv`

Earlier supporting artifacts still matter for comparison:

- `tmp/gate_reaudit_20260414_runtime_fixed/`
- `tmp/gate_reaudit_20260414_deep/`
- `tmp/gate1_runtime_probe/`

## 4. Gate verdict

| Gate | Verdict | Why |
| --- | --- | --- |
| Gate 0 | fail | the clean local slice is now available, but total signal is still flat on median totals and only weakly positive on the new duration-logratio view |
| Gate 1 | pass on the strongest fixed variant, still not universal | `weighted_median + exact_global_family` now passes all 4 local analytic triplets end-to-end; `raw_median` still needs clip help and still does not fully generalize |
| Gate 2 | blocked | Gate 0 still fails |
| Gate 3 | blocked | Gate 2 not admissible |

## 5. Gate0 clean-slice audit

The most important change is that Gate0 is no longer forced to judge only the
hostile manifest slice. The current default rerun uses the locally constructed
clean slice:

- `reference_mode=target_as_ref`
- `clean_total_claim_items = 64`
- `valid_clean_total_claim_items = 63`

For the strongest current local variant
`weighted_median + exact_global_family`, candidate token `57`,
`drop_edge_runs_for_g = 1`, the clean-slice summary is:

| Metric | Value |
| --- | ---: |
| valid total median slope | `0.0000` |
| valid total mean slope | `-0.0216` |
| valid total duration-logratio slope | `0.0124` |
| valid analytic median slope | `0.1672` |
| valid analytic runtime mean slope | `0.0000` |
| valid residual mean slope | `-0.1636` |
| valid residual runtime mean slope | `-0.0216` |
| valid residual runtime affine slope | `0.0100` |
| mean analytic saturation | `1.0000` |

Interpretation:

- the earlier protocol objection was real and is now addressed locally
- even on the clean slice, the old median-total reading still does not move
- the new duration-logratio view is slightly positive, so the correct reading
  is no longer "strictly zero everywhere"
- that positive total view is still far too weak to promote the maintained
  single-scalar total-control claim
- unclipped analytic signal is real
- runtime-aligned analytic signal is still flattened almost completely because
  the runtime surface is saturated
- the affine residual view reduces the apparent negative residual story, which
  means part of the old residual negativity was indeed decomposition-baked
  rather than purely empirical

Bottom line:

- Gate0 is now cleaner than before
- Gate0 still fails
- the remaining failure no longer looks like a simple audit-bug artifact

## 6. Gate1 analytic runtime probe

For `weighted_median + exact_global_family`, the latest rerun is now fully
passing on the local 4-case analytic probe:

| Variant | preclip pass | continuous pass | projected pass |
| --- | ---: | ---: | ---: |
| `weighted_median` | `4/4` | `4/4` | `4/4` |

Per-source projected slopes/ranges:

| Source | projected slope | projected range |
| --- | ---: | ---: |
| `aba_train_arctic_a0022` | `-0.0398` | `0.0500` |
| `asi_train_arctic_a0027` | `-0.1223` | `0.2500` |
| `bdl_train_arctic_a0022` | `-0.0486` | `0.0833` |
| `slt_train_arctic_a0020` | `-0.0866` | `0.1097` |

This confirms:

- Gate1 is no longer the main blocker on the strongest fixed variant
- the remaining generalization problem is variant/runtime fragility, not a
  blanket "analytic chain is dead" reading

An additional raw probe with more permissive clip confirms the runtime-surface
story:

| Variant | config | projected pass |
| --- | --- | ---: |
| `raw_median` | `exact_global_family + analytic_gap_clip=0.6` | `3/4` |

That probe helps interpret the surface:

- widening clip helps `raw_median`
- it still does not fully solve the runtime failure set
- `weighted_median` remains the strongest current local candidate

## 7. Current reading

The latest local reading is now sharper:

- support collapse is not the main blocker
- Gate1 evaluation drift was real and is now largely fixed
- weighted prompt/source tempo readouts are now aligned with the actual `g`
  contract
- the clean local Gate0 slice now exists and still fails
- the remaining Gate0 blocker is not just protocol dirtiness
- the strongest current failure mode is:
  single-scalar `g` can still drive an analytic direction on this surface, but
  it is not strong enough to establish a robust total-target claim

So the maintained conclusion for this local surface is:

- keep Gate 2 blocked
- keep Gate 3 blocked
- do not promote the maintained V1 total-control claim
- keep the narrower claim:
  `weighted_median + exact_global_family` can pass the local analytic runtime
  probe, but the maintained single-scalar raw-duration line still fails the
  cleaner Gate0 total audit
