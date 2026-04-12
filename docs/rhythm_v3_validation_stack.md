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

### 2.2 Debug-record schema

Files:

- `utils/plot/rhythm_v3_viz/core.py`
- `utils/plot/rhythm_v3_viz/alignment.py`

These are still the stable bridge between runtime/data code and later review.

### 2.3 Five-figure review util

File:

- `utils/plot/rhythm_v3_viz/review.py`

This is now the single place for:

- shared scalar reconstruction such as `g`, `p_i`, `a_i`, `b*`
- unified table builders
- per-figure summaries
- lightweight plotting helpers

The project was intentionally simplified here:

- no extra falsification CLI layer is required
- no extra dedicated falsification test file is kept
- review logic should live in the util layer, not be copied across scripts
- local `g` reconstruction should call the shared speech-only support path,
  including `rhythm_v3_drop_edge_runs_for_g` when the experiment enables it

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
- `same_text`, `lexical_mismatch`
- `g_crop`, `g_full`, `delta_g`
- `c_star`, `zbar_sp_star`

### 3.3 `prefix_replay_table`

One prefix-step run row. Used by Figure A and Figure E.

Key columns:

- `utt_id`, `chunk_scheme`, `prefix_ratio`, `commit_lag`
- `run_idx`, `is_closed`, `is_committed`
- `n_prefix`, `n_full`
- `n_pred_cont`, `k_pred_disc`, `k_full_disc`
- `budget_drift`, `budget_hit`

## 4. Review util entry points

Typical usage:

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

Useful public helpers:

- `build_run_table(...)`
- `build_ref_crop_table(...)`
- `build_prefix_replay_table(...)`
- `compute_run_stability(...)`
- `summarize_global_cue_review(...)`
- `summarize_oracle_decomposition(...)`
- `build_silence_audit_table(...)`
- `compute_commit_metrics(...)`
- `save_review_figure_bundle(...)`

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
visible.

### 5.4 Closed and committed are different

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
