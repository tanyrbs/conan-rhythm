# rhythm_v3 / V1-G validation stack

This document defines the **util-first** validation workflow for the maintained
`rhythm_v3` / minimal-V1 path.

The current goal is not to grow a larger dashboard. The goal is to make the
core theory easy to **audit and falsify** with the smallest stable surface.

## 1. What we validate first

The maintained V1-G line makes five hard claims:

1. the source run lattice is a stable interface
2. `g` is a usable speech-only global cue
3. scalar coarse bias is a meaningful variable, not a placeholder
4. silence should stay coarse-only rather than become a pause planner
5. closed-prefix + carry rounding + budget form a real online commit contract

That is why the main review surface is now restricted to **five figures**:

- Figure A: run-lattice stability
- Figure B: global cue survival
- Figure C: oracle decomposition
- Figure D: silence theory audit
- Figure E: online commit semantics

Everything else is secondary:

- alignment heatmaps belong in appendix or local diagnosis
- label conservation pages belong in dataset QA
- single-case multi-track pages belong in internal debugging

## 2. Current local code support

### 2.1 Shared alignment/projection module

File:

- `tasks/Conan/rhythm/duration_v3/alignment_projection.py`

This remains the shared projection module for:

- dataset target construction
- offline audit/debug reconstruction

`use_continuous_alignment=True` now accepts either
`continuous_precomputed` provenance or built-in `continuous_viterbi_v1`
outputs. If neither explicit continuous provenance nor the required frame-state
sidecars are available, the maintained path fails fast instead of silently
falling back to discrete projection.
Paired target alignment arrays are only accepted for this path when the cache
also explicitly marks them with a continuous `alignment_kind`, currently
`continuous_precomputed` or `continuous_viterbi_v1`; arrays without provenance still do not count as
continuous metadata.

### 2.2 Debug-record schema

Files:

- `utils/plot/rhythm_v3_viz/core.py`
- `utils/plot/rhythm_v3_viz/alignment.py`

These are still the stable bridge between runtime/data code and later review.
The maintained debug/summary surface now includes both:

- `g` support certificates such as `g_support_count`, `g_support_ratio_vs_speech`,
  `g_support_ratio_vs_valid`, and `g_valid`
- alignment-quality certificates such as
  `alignment_unmatched_speech_ratio`,
  `alignment_mean_local_confidence_speech`, and
  `alignment_mean_coarse_confidence_speech`
- prompt-side audit sidecars such as `g_trim_ratio`,
  `prompt_global_weight_present`, `prompt_unit_log_prior_present`, and
  `prompt_unit_prior_vocab_size`

### 2.3 Five-figure review util

File:

- `utils/plot/rhythm_v3_viz/review.py`

This is now the single place for:

- shared scalar reconstruction such as `g`, `p_i`, `a_i`, `b*`
- analysis-side `g_src_utt` and speech-tempo reconstruction
- unified table builders
- per-figure summaries
- lightweight plotting helpers
- gate-oriented figures built from the same tables

The project was intentionally simplified here:

- no extra dedicated falsification test file is kept
- review logic should live in the util layer, not be copied across scripts
- local `g` reconstruction should call the shared speech-only support path,
  including `rhythm_v3_drop_edge_runs_for_g` when the experiment enables it
- the maintained CLI should stay singular: `scripts/rhythm_v3_debug_records.py`
  exports the shared review/gate bundle, while narrower gate inspection should
  come from the emitted CSVs or direct util calls instead of extra wrapper CLIs
- the old standalone `scripts/smoke_test_rhythm_v3.py` wrapper is retired;
  structural smoke coverage now comes from targeted tests plus short task-level
  smoke runs

Some earlier implementation suggestions are therefore already outdated in this
workspace:

- no new `g_stats.py` is needed because the shared helper already exists
- no separate per-gate wrapper CLIs are kept anymore
- no second runtime surface is needed because `analytic / coarse_only / learned`
  already switch inside the maintained `rhythm_v3` path

## 3. Unified analysis tables

The review util is built around three shared tables instead of five unrelated
figure-specific code paths.

### 3.1 `run_table`

One source run per row. Used by Figure C and Figure D.

Key columns:

- `utt_id`, `pair_id`, `run_idx`, `unit_id`
- `run_type`, `boundary_type`
- `n_src`, `n_star`, `omega`
- `p_i`, `g`, `a_i`
- `z_star`, `b_star`, `b_pred`
- `r_star`, `r_pred`
- `n_pred_cont`, `k_pred_disc`

### 3.2 `ref_crop_table`

One prompt/reference crop per row. Used by Figure B.

Key columns:

- `pair_id`, `crop_id`
- `ref_len_sec`, `speech_ratio`
- `same_text_reference`, `same_text_target`
- `same_speaker_reference`, `same_speaker_target`
- `lexical_mismatch`
- `g_crop`, `g_full`, `g_src_utt`, `g_src_prefix_mean`, `delta_g`
- `c_star`, `zbar_sp_star`

### 3.3 `prefix_replay_table`

One prefix-step run row. Used by Figure A and Figure E.

Key columns:

- `utt_id`, `chunk_scheme`, `prefix_ratio`, `commit_lag`
- `run_idx`, `is_closed`, `is_committed`
- `n_prefix`, `n_full`
- `n_pred_cont`, `k_pred_disc`, `k_full_disc`
- `budget_drift`, `budget_hit`

### 3.4 Gate tables

On top of the retained five theory-review figures, the same util layer now
keeps three gate-oriented tables for rapid falsification:

- `monotonicity_table`
- `prefix_silence_review_table`
- `mode_ladder_table`

Important convenience fields now emitted by the single export path:

- row summary: `same_speaker_reference`, `same_speaker_target`, `tempo_delta`,
  `alignment_kind`, `alignment_source`, `alignment_version`,
  `target_duration_surface`, `projector_boundary_hit_rate`,
  `projector_boundary_decay_rate`
- monotonicity table: `sample_id`, `pair_id`, `tempo_delta`, `mono_triplet_ok`
- ladder table: `tempo_transfer_slope`

## 4. Review util entry points

Typical usage:

```python
from utils.plot.rhythm_v3_viz import (
    build_prefix_replay_table,
    build_ref_crop_table,
    build_run_table,
    save_review_figure_bundle,
    save_validation_gate_bundle,
)

run_df = build_run_table(records)
crop_df = build_ref_crop_table(records)
prefix_df = build_prefix_replay_table(records)

paths = save_review_figure_bundle(records, output_dir="artifacts/rhythm_v3_review")
gate_paths = save_validation_gate_bundle(records, output_dir="artifacts/rhythm_v3_review")
```

Useful public helpers:

- `build_run_table(...)`
- `build_ref_crop_table(...)`
- `build_prefix_replay_table(...)`
- `build_monotonicity_table(...)`
- `build_prefix_silence_review_table(...)`
- `compute_source_global_rate_for_analysis(...)`
- `compute_speech_tempo_for_analysis(...)`
- `compute_run_stability(...)`
- `summarize_falsification_ladder(...)`
- `summarize_global_cue_review(...)`
- `summarize_oracle_decomposition(...)`
- `build_silence_audit_table(...)`
- `compute_commit_metrics(...)`
- `save_review_figure_bundle(...)`
- `save_validation_gate_bundle(...)`

Maintained CLI entrypoint built on top of the same util layer:

- `scripts/rhythm_v3_debug_records.py`

### 4.1 `g_src_utt` vs `g_src_prefix`

The local code now keeps these two meanings separate:

- `g_src_utt`: full-utterance source statistic used in static explainability
  analysis such as `delta_g = g_ref - g_src_utt`
- `g_src_prefix`: causal runtime prefix state exported by the online planner

They should not be mixed when making falsification plots.

### 4.2 `unit_norm` is now reproducible inside the repo

The maintained repo now ships `scripts/build_unit_log_prior.py` and a matching
`rhythm_v3_unit_prior_path` loading path. That makes `unit_norm` a real
reproducible experiment path instead of a consumer-only interface, but the
bundle should still be treated as provenance-bearing data rather than an
implicit constant.

### 4.3 Debug-bundle metadata expectations

The maintained export script can summarize any valid debug bundle, but the
strongest Gate-0 / contamination-slice conclusions assume the bundle still
carries:

- `pair_id`
- prompt ids or source/reference signatures
- `same_text_reference`
- `same_text_target`
- `lexical_mismatch`
- `ref_len_sec`
- `speech_ratio`
- `unit_duration_proj_raw_tgt` or another explicit raw target-surface alias
- `unit_alignment_mode_id_tgt` and `unit_alignment_kind_tgt` when paired-target
  provenance is available

If those fields are missing, the scripts still export tables and figures, but
you should read the result as a partial audit rather than a full falsification
verdict.
The same caution applies when the gate contract itself is incomplete:

- missing `analytic`, `coarse_only`, or `learned` means the mode ladder is incomplete
- missing full `slow / mid / fast` triplets means Gate 1 is incomplete even if a figure exists
- missing `ref_condition` / negative-control exports means you only have correlation evidence, not a full control audit

## 5. Important local implementation boundaries

These figure utilities are aligned to the current local codebase, not to an
idealized future dataset.

### 5.1 Boundary typing is a proxy

`boundary_type` is currently reconstructed from local signals:

- `sep_mask`
- `source_boundary_cue`
- utterance-final position

So Figure D should be read as a **boundary-aware audit proxy**, not as a
linguistic gold boundary annotation.

### 5.2 Crop stability requires actual crop groups

Figure B Panel A only becomes meaningful when the debug records contain
multiple crops for the same `pair_id`. If there is only one crop, the review
util keeps the field but does not pretend the stability question was answered.

### 5.3 Prompt speech masks are part of the contract

For the maintained prompt-summary / minimal-V1 line, `prompt_speech_mask` is no
longer treated as an optional convenience field. It is part of the explicit
speech-only `g` contract, and the config/runtime surfaces should keep that
visible. In the current strict minimal runtime, silence-token fallback should
not be treated as equivalent evidence; `prompt_silence_mask` remains useful for
consistency checks, but it does not replace explicit `prompt_speech_mask`.

### 5.4 Projector debug now distinguishes carry from boundary-side smoothing

The maintained projector surface should be read in two layers:

- core carry + budget execution
- boundary-side smoothing heuristics exported for audit

So when reviewing prefix drift, prefer keeping these signals together:

- `projector_rounding_residual`
- `projector_budget_hit_pos` / `projector_budget_hit_neg`
- `projector_boundary_hit`
- `projector_boundary_decay_applied`
- `projector_since_last_boundary`

`projector_boundary_hit` marks boundary/phrase-final events, while
`projector_boundary_decay_applied` only marks the subset where decay was
actually applied.

### 5.5 Closed and committed are different

Figure A uses `is_closed`.
Figure E uses `is_committed`.

That separation is deliberate. Open tails should not be counted as interface
instability, and closed-but-not-yet-committed runs should not be counted as a
commit violation.

## 6. What the five figures should prove

- Figure A should tell us whether the source lattice behaves like an interface.
- Figure B should tell us whether `g` is stable and informative enough to keep.
- Figure C should tell us whether scalar coarse is sufficient before escalating
  to phrasewise coarse.
- Figure D should tell us whether silence really belongs on the coarse-only
  side of the theory boundary.
- Figure E should tell us whether the runtime is actually commit-safe online.

If one of these five figures fails badly, the right next step is usually to
repair the corresponding interface or statistic first, not to add more model
capacity.

## 7. One-script export

The maintained export surface is deliberately small:

```bash
py -3 scripts\rhythm_v3_debug_records.py ^
  --input path\to\debug_bundle.pt ^
  --output artifacts\rhythm_v3_summary.csv ^
  --review-dir artifacts\rhythm_v3_review ^
  --g-variant raw_median ^
  --drop-edge-runs 1
```

This now writes:

- the row-level falsification summary CSV
- the retained five-figure review bundle
- the gate-oriented monotonicity / stability / ladder bundle

When `--review-dir` or `--gate-status-json` is present, the maintained script
now fails non-zero on gate issues by default. Use `--allow-partial-gates` only
for an explicitly partial audit export.
