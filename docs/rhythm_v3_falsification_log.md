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

### 2.2 Falsification script surface stays thin

To keep the project lean, the maintained script surface is intentionally small:

- `scripts/preflight_rhythm_v3.py`
- `scripts/rhythm_v3_debug_records.py`

The intent is deliberate:

- keep the **core implementation** and **review util** authoritative
- keep the CLI surface to one maintained review/export command
- avoid growing a second plotting or experiment framework

The old zero-data standalone smoke script has been retired. Structural smoke
coverage now belongs to focused entrypoint tests plus the short task-level
smoke run in the training guide, not to a second maintained v3 wrapper CLI.

The maintained export command is most informative when the debug bundle still carries pair
metadata (`pair_id`, prompt ids, same-text flags, `lexical_mismatch`,
`ref_len_sec`, `speech_ratio`). If that metadata is absent, the scripts still
export tables, but Gate-0 contamination slices should be read as incomplete.

The export surface also now keeps small but useful analysis aliases together:

- `same_speaker_reference` / `same_speaker_target`
- `tempo_delta = tempo_out - tempo_src`
- `mono_triplet_ok`
- ladder-level `tempo_transfer_slope`
- `alignment_kind`
- `target_duration_surface`
- `ref_condition`
- `g_trim_ratio`
- `prompt_global_weight_present`
- `prompt_unit_log_prior_present`

### 2.3 Runtime/debug export keeps the same core signals plus clearer boundary provenance

The runtime still exports the key audit signals:

- `rhythm_debug_g_ref`
- `rhythm_debug_g_src_prefix`
- `rhythm_debug_analytic_gap`
- `rhythm_debug_coarse_bias`
- `rhythm_debug_local_residual`
- `rhythm_debug_is_speech`
- `rhythm_debug_budget_hit_pos`
- `rhythm_debug_budget_hit_neg`
- `rhythm_debug_projector_boundary_hit`
- `rhythm_debug_projector_boundary_decay`
- `rhythm_debug_projector_since_last_boundary`
- `unit_duration_exec`

So the review surface got smaller, but the underlying observability was not
reduced.
`rhythm_debug_projector_boundary_hit` marks boundary events;
`rhythm_debug_projector_boundary_decay` marks only the subset where decay was
actually applied.

### 2.4 `g` is now locked to one support path

The local implementation should keep training, runtime diagnostics, and review
on the same `g` semantics:

- speech-only support first
- optional `rhythm_v3_drop_edge_runs_for_g` cleanup
- edge-drop fallback only back to raw speech support, never to non-speech valid/full support
- `prompt_speech_mask` carried through as an explicit contract field
- runtime/debug export now also carries `g_support_count`,
  `g_support_ratio_vs_speech`, `g_support_ratio_vs_valid`, `g_valid`,
  `g_drop_edge_runs`, and `g_strict_speech_only` so Gate-0 analysis can tell
  whether `g` survived on real speech support instead of assuming it did

### 2.5 Continuous alignment provenance is now by contract

The maintained `continuous` path now has two provenance-clean variants,
`continuous_precomputed` and `continuous_viterbi_v1`, and it is no longer
accepted by convention alone.

- paired-target alignment arrays now only feed the maintained continuous path
  when paired metadata explicitly marks them with a continuous
  `alignment_kind`
- keep `unit_alignment_kind_tgt` readable in tables, while `alignment_source`
  and `alignment_version` preserve the exact producer (`continuous_precomputed`
  vs `run_state_viterbi`) for later review
- otherwise the existing fail-fast path remains in charge
- projection export also carries
  `unit_alignment_unmatched_speech_ratio_tgt`,
  `unit_alignment_mean_local_confidence_speech_tgt`, and
  `unit_alignment_mean_coarse_confidence_speech_tgt`
  so weak supervision does not hide inside a single opaque confidence surface

### 2.6 `unit_norm` now has a repo-native producer

The maintained repo now ships `scripts/build_unit_log_prior.py` plus a matching
`rhythm_v3_unit_prior_path` loading path in dataset and inference code. So
`unit_norm` is no longer just "supported if you hand-wire an array"; it can now
be treated as a reproducible experiment path.

### 2.7 Static `g_src_utt` is kept separate from runtime `g_src_prefix`

The local workspace now makes this split explicit:

- `g_src_utt`: analysis-side full-utterance source statistic used for static
  `delta_g` plots
- `g_src_prefix`: runtime causal prefix summary used by the online retimer

This avoids the earlier drift where static explainability could accidentally
mix a full-reference statistic with a prefix-state statistic.

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

- does the raw projected target surface (`unit_duration_proj_raw_tgt`, or
  legacy `unit_duration_tgt` when no alias is present) diverge from the clipped
  coarse pseudo-target?
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
- no negative-control reference (`source_only`, `random_ref`, `shuffled_ref`)
  is exported alongside the real-reference triplets

### Gate 2: coarse-only before local

Only keep strong local residual if it improves control without obviously
damaging silence leakage or prefix stability.

### Outdated suggestions

Several earlier "next steps" are now already implemented locally and should not
be re-proposed as new refactors:

- shared `g` computation already lives in `modules/Conan/rhythm_v3/g_stats.py`
- `analytic / coarse_only / learned` already run inside the same maintained runtime
- projector budget/drift observability is already exported; the remaining work
  is interpretation, especially `boundary_hit` vs `boundary_decay_applied`, not
  projector replacement

## 6. Recommended util-first workflow

1. export debug bundles from the maintained runtime
2. build:
   - `run_table`
   - `ref_crop_table`
   - `prefix_replay_table`
3. render the five retained figures from the review util
4. update this log with actual measurements
5. only then decide whether to keep, weaken, or replace the current default `g`

The log should also state explicitly whether the current bundle is a full gate
bundle or only a partial audit. In this workspace, the maintained
`scripts/rhythm_v3_debug_records.py` path now fails by default for maintained
review exports (`--review-dir` / `--gate-status-json`) and warns-only only when
`--allow-partial-gates` is set. The gate checks cover:

- `g_valid` coverage is weak
- unmatched speech alignment is high
- continuous coverage is incomplete for the current `alignment_kind` split
- `analytic / coarse_only / learned` modes are missing
- real-reference triplets are incomplete
- negative controls are absent

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

Or directly from the maintained script:

```bash
py -3 scripts\rhythm_v3_debug_records.py ^
  --input path\to\debug_bundle.pt ^
  --output artifacts\rhythm_v3_summary.csv ^
  --review-dir artifacts\rhythm_v3_review ^
  --g-variant raw_median ^
  --drop-edge-runs 1
```

That single command now exports:

- the row-level summary CSV
- the retained five-figure review bundle
- the gate-oriented monotonicity / stability / ladder bundle

## 7. Current empirical status

This workspace now has a slimmer and more coherent review surface, but it still
has **not** run a real CMU-ARCTIC or L2-ARCTIC experiment inside this turn.

So the current status is:

- code status: updated
- review surface: simplified
- empirical verdict: still open

## 8. Result table template

### Gate 0 summary

| date | split | g_variant | alignment_kind | target_duration_surface | rho(delta_g,c*) | robust_slope | r2_like | notes |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| pending | pending | raw_median | pending | pending | pending | pending | pending | no experiment run yet |

### Gate 1 summary

| date | split | eval_mode | monotonicity_rate | notes |
| --- | --- | --- | ---: | --- |
| pending | pending | analytic | pending | no experiment run yet |

### Gate 2 summary

| date | split | eval_mode | monotonicity_rate | silence_leakage | prefix_discrepancy | budget_hit_rate | projector_boundary_hit_rate | projector_boundary_decay_rate | notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| pending | pending | coarse_only | pending | pending | pending | pending | pending | pending | no experiment run yet |
| pending | pending | learned | pending | pending | pending | pending | pending | pending | no experiment run yet |
