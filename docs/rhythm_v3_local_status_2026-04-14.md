# rhythm_v3 local status snapshot (2026-04-14)

This document records the current maintained local rerun after the latest
frontend/data-contract repair, cache rebuild, script cleanup, and zero-train
gate rerun.

## 1. Scope

- maintained `rhythm_v3` minimal-V1 path only
- local quick-ARCTIC configs only
- zero-train falsification only
- Gate 2 / Gate 3 training still blocked

## 2. What changed in this round

- source/prompt silence no longer depends on raw `silent_token=57` appearing on
  the tokenizer surface; cache build, binarization, runtime, and inference now
  accept acoustic silence inferred from mel
- prompt metadata construction now filters shared prompt ids by maintained
  `3.0s-8.0s` duration before split selection
- `g` audit code now reports three separate static views instead of collapsing
  everything into one residual-style readout:
  - total target signal
  - analytic component
  - residual coarse component
- Gate1 probe cases are now auto-selected from the current split instead of
  relying on stale hard-coded ids
- old token-guessing helper scripts that no longer belong to the maintained
  mainline were removed

## 3. Current artifacts

Current rerun artifacts live in:

- `tmp/gate_reaudit_20260414_rebuilt2/`

Primary files:

- `tmp/gate_reaudit_20260414_rebuilt2/boundary_summary.json`
- `tmp/gate_reaudit_20260414_rebuilt2/gate0_raw/report.json`
- `tmp/gate_reaudit_20260414_rebuilt2/gate0_weighted/report.json`
- `tmp/gate_reaudit_20260414_rebuilt2/gate0_trimmed/report.json`
- `tmp/gate_reaudit_20260414_rebuilt2/gate0_softclean/report.json`
- `tmp/gate_reaudit_20260414_rebuilt2/gate0_softclean_wtmean/report.json`
- `tmp/gate_reaudit_20260414_rebuilt2/gate1_raw/summary.json`
- `tmp/gate_reaudit_20260414_rebuilt2/gate1_weighted/summary.json`
- `tmp/gate_reaudit_20260414_rebuilt2/gate1_trimmed/summary.json`
- `tmp/gate_reaudit_20260414_rebuilt2/gate1_softclean/summary.json`
- `tmp/gate_reaudit_20260414_rebuilt2/gate1_softclean_wtmean/summary.json`

## 4. Frontend/data-contract check

Latest preflight and boundary audit show:

- `source_silence_items=16/16` on inspected train / valid / test slices
- `sep_items=16/16` on inspected train / valid / test slices
- `raw_silent_items=0/N` is no longer a blocker because acoustic silence sidecars
  are present
- `max_boundary_confidence` now reaches about `0.96-0.98`
- boundary support at `min_boundary_confidence_for_g=0.5`:
  - train `32/32`
  - valid `16/16`
  - test `16/16`

Non-blocking remaining preflight warning:

- config stage still reports `transitional`

Interpretation:

- the old "prompt domain collapses before `g` exists" failure mode has been
  repaired on this local surface
- current Gate0 / Gate1 failure is no longer attributable to stale cache,
  missing silence runs, or broken boundary scaling

## 5. Gate verdict

| Gate | Verdict | Why |
| --- | --- | --- |
| Gate 0 | fail | prompt-domain coverage is repaired, but total static signal slope stays flat |
| Gate 1 | fail | runtime analytic control is flat for `raw`, `weighted_median`, and `trimmed_mean`; softclean variants only recover 1 source |
| Gate 2 | blocked | Gate 0 / Gate 1 still fail |
| Gate 3 | blocked | Gate 2 not admissible |

## 6. Gate0 static audit

All variants keep prompt-domain validity at `63/64`, so support-domain failure
is no longer the main story.

| Variant | total slope (`Δg_utt -> zbar*`) | analytic slope (`Δg_utt -> abar*`) | residual slope (`Δg_utt -> c*`) | prefix-total slope (`Δg_prefix -> zbar*`) |
| --- | ---: | ---: | ---: | ---: |
| `raw_median` | `0.0000` | `0.5316` | `-0.6329` | `0.0000` |
| `weighted_median` | `0.0000` | `0.5316` | `-0.6329` | `0.0000` |
| `trimmed_mean` | `-0.0000` | `0.5857` | `-0.6601` | `-0.0000` |
| `softclean_wmed` | `0.0000` | `0.3833` | `-0.6666` | `-0.0000` |
| `softclean_wtmean` | `0.0000` | `0.5615` | `-0.6852` | `0.0000` |

Interpretation:

- the analytic component is not dead
- the total target signal is still flat for every tested single-scalar variant
- the residual branch still points negative
- the current runtime prefix baseline does not rescue the total-signal failure

So the remaining blocker is not "no valid `g`". It is "the present scalar cue
does not explain the full target stretch signal on this surface".

## 7. Gate1 analytic runtime probe

| Variant | monotone sources | mean transfer slope | mean real tempo range | Notes |
| --- | ---: | ---: | ---: | --- |
| `raw_median` | `0/4` | `~0.0000` | `0.0000` | all outputs collapse to `source_only` / negative-control behavior |
| `weighted_median` | `0/4` | `~0.0000` | `0.0000` | same as `raw_median` |
| `trimmed_mean` | `0/4` | `~0.0000` | `0.0000` | same as `raw_median` |
| `softclean_wmed` | `1/4` | `0.1068` | `0.0393` | only `asi_train_arctic_a0019` shows real push |
| `softclean_wtmean` | `1/4` | `0.1000` | `0.0333` | same single positive source, slightly weaker than `softclean_wmed` |

Per-source positive case:

- `asi_train_arctic_a0019`
  - `softclean_wmed`: `transfer_slope=0.4505`, `real_tempo_range=0.1071`
  - `softclean_wtmean`: `transfer_slope=0.4232`, `real_tempo_range=0.0833`

Interpretation:

- `weighted_median` and `trimmed_mean` do not improve over `raw_median`
- softclean weighting helps, but not enough to qualify the maintained claim
- the current issue is now "push exists only on a narrow subset", not "prompt
  domain cannot be formed"

## 8. Current reading

Current failure should be stated narrowly:

- the earlier frontend/data-surface failure was real, and this round repaired it
- after that repair, the maintained path still fails
- the remaining failure is not support-domain collapse
- the remaining failure is weak or unstable single-scalar control force

So the maintained conclusion is:

- do not start Gate 2 training
- do not start Gate 3 training
- do not claim `raw_median` is a qualified maintained default on this local
  surface
- do not claim `weighted_median`, `trimmed_mean`, or current softclean variants
  have fully repaired the problem

The project remains in zero-train falsification mode.
