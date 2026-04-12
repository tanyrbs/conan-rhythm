# rhythm_v3 minimal-V1 falsification log

This document records the **falsification-first** workflow for the maintained
`rhythm_v3` minimal-V1 path.

The goal is still narrow:

> In short, speech-dominant, same-speaker / different-text references,
> `g_ref` should behave like a stable and transferable speech-only global cue,
> while speech moves more than silence and committed prefixes remain usable.

If that claim fails, the right answer is not "add more branches". The right
answer is to question the statistic, the interface, or the runtime contract.

## 1. What stays fixed in the local implementation

The maintained writer is already close to:

- source-anchor run lattice
- prompt-summary / source-observed
- speech-only global cue `g_ref`
- strict-causal source prefix tempo `g_src_prefix`
- scalar coarse correction
- speech-only local residual
- silence as clipped coarse follow
- carry rounding + prefix budget + committed-prefix execution

This log therefore treats the current line as something to **stress and audit**,
not something to repackage into a second big framework.

## 2. What changed in this cleanup round

### 2.1 Review logic moved into one util module

File:

- `utils/plot/rhythm_v3_viz/review.py`

This is now the single home for:

- `g`, `p_i`, `a_i`, `b*` reconstruction helpers
- unified table builders
- per-figure summaries
- lightweight plotting helpers for the five retained figures

### 2.2 Extra falsification script/test surface was removed

To keep the project lean:

- the small `scripts/rhythm_v3/` falsification wrappers were removed
- the dedicated `tests/rhythm/test_rhythm_v3_falsification.py` file was removed

The intent is deliberate:

- keep the **core implementation** and **review util**
- avoid keeping parallel demo/test shells that grow stale

### 2.3 Runtime/debug export remains the same

The runtime still exports the key audit signals:

- `rhythm_debug_g_ref`
- `rhythm_debug_g_src_prefix`
- `rhythm_debug_analytic_gap`
- `rhythm_debug_coarse_bias`
- `rhythm_debug_local_residual`
- `rhythm_debug_is_speech`
- `rhythm_debug_budget_hit_pos`
- `rhythm_debug_budget_hit_neg`
- `unit_duration_exec`

So the review surface got smaller, but the underlying observability was not
reduced.

## 3. The five retained main figures

The main review surface is now explicitly restricted to five figures.

### Figure A: run-lattice stability

Claim:

- the source run lattice behaves like a stable interface

Primary questions:

- do closed runs stop changing across prefixes?
- does run-count drift stay small after stabilization?
- do alternative chunk schemes still converge to the same closed prefix?

### Figure B: global cue survival

Claim:

- `g` is stable enough and informative enough to remain the default global cue

Primary questions:

- does `g` stay stable under crop in the 3-8s domain?
- does `delta_g` explain oracle coarse bias `c*`?
- where does lexical contamination become visible?

### Figure C: oracle decomposition

Claim:

- scalar coarse is a real variable, not a cosmetic residual bucket

Primary questions:

- how far does `a_i` alone go?
- how much does scalar `b*` buy over pure analytic shift?
- does the residual after `a_i + b*` still contain low-frequency drift?

### Figure D: silence theory audit

Claim:

- silence should stay coarse-only in V1 unless the boundary audit disproves it

Primary questions:

- does raw silence target diverge from clipped coarse pseudo-target?
- is that divergence boundary-dependent?
- does the evidence point to boundary-aware clipping rather than silence local residual?

### Figure E: online commit semantics

Claim:

- closed-prefix + carry rounding + budget form a usable online contract

Primary questions:

- do committed runs stop rewriting as prefix grows?
- how much gap remains between short and long prefixes?
- does budget drift stay bounded, and how often does budget actually hit?

## 4. What is intentionally not a main figure anymore

These may still be useful, but they are no longer allowed to carry the main
theoretical burden:

- alignment heatmaps
- label conservation pages
- single-case multi-track dashboards

They belong in appendix, dataset QA, or internal debugging.

## 5. Falsification gates remain the same

### Gate 0: static `g` audit

Stop early if:

- crop stability is weak
- `delta_g` barely explains `c*`
- different-text collapses while same-text looks fine

### Gate 1: pure analytic control

Run `analytic` mode and stop early if:

- slow / mid / fast references do not induce monotone speech tempo movement

### Gate 2: coarse-only before local

Only keep strong local residual if it improves control without obviously
damaging silence leakage or prefix stability.

## 6. Recommended util-first workflow

1. export debug bundles from the maintained runtime
2. build:
   - `run_table`
   - `ref_crop_table`
   - `prefix_replay_table`
3. render the five retained figures from the review util
4. update this log with actual measurements
5. only then decide whether to keep, weaken, or replace the current default `g`

Typical entry points:

```python
from utils.plot.rhythm_v3_viz import (
    build_prefix_replay_table,
    build_ref_crop_table,
    build_run_table,
    save_review_figure_bundle,
)

run_df = build_run_table(records)
crop_df = build_ref_crop_table(records)
prefix_df = build_prefix_replay_table(records)

paths = save_review_figure_bundle(records, output_dir="artifacts/rhythm_v3_review")
```

## 7. Current empirical status

This workspace now has a slimmer and more coherent review surface, but it still
has **not** run a real CMU-ARCTIC or L2-ARCTIC experiment inside this turn.

So the current status is:

- code status: updated
- review surface: simplified
- empirical verdict: still open

## 8. Result table template

### Gate 0 summary

| date | split | g_variant | rho(delta_g,c*) | robust_slope | r2_like | notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| pending | pending | raw_median | pending | pending | pending | no experiment run yet |

### Gate 1 summary

| date | split | eval_mode | monotonicity_rate | notes |
| --- | --- | --- | ---: | --- |
| pending | pending | analytic | pending | no experiment run yet |

### Gate 2 summary

| date | split | eval_mode | monotonicity_rate | silence_leakage | prefix_discrepancy | budget_hit_rate | notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| pending | pending | coarse_only | pending | pending | pending | pending | no experiment run yet |
| pending | pending | learned | pending | pending | pending | pending | no experiment run yet |
