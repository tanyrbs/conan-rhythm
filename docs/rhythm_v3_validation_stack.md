# rhythm_v3 / V1-G validation stack

This document defines the **util-first** validation workflow for the maintained
`rhythm_v3` / minimal-V1 path.

The current goal is not to grow a larger dashboard. The goal is to make the
core theory easy to **audit and falsify** with the smallest stable surface.

## 0. Current local snapshot

Latest maintained local rerun:

- `2026-04-15`

Canonical status snapshot:

- `docs/rhythm_v3_local_status_2026-04-15.md`

Latest exported local artifacts:

- `egs/overrides/rhythm_v3_gate_status.json`
- `egs/overrides/rhythm_v3_gate_status_local_candidate_20260414.json`
- supporting local-candidate artifacts:
  - `tmp/gate2_candidate_20260415_s75_srcgap/`
  - `checkpoints/rhythm_v3_gate2_candidate_20260415_s76_srcgap/`
  - `checkpoints/rhythm_v3_gate3_candidate_20260415_s126_srcgapfix1/`

Current verdict on the local quick-ARCTIC surface:

- prompt-domain support is repaired on the rebuilt cache surface
- Gate 0 now passes on the strongest fixed local contract after the final
  source-support / init-parity repair
- Gate 1-upper also passes on that same strongest fixed contract:
  `weighted_median + exact_global_family`
- Gate 2-online local `src_gap` candidate was rerun on the maintained online
  contract
- aggregate Gate2 results were essentially unchanged relative to the earlier
  candidate
- recurring failure mode remains execution-side:
  `preproj` often shows signal while `exec` still flattens or ties
- Gate 3 local candidate training is now unblocked by a runtime/config wiring
  fix and advanced locally to a checkpointed run, but this is not an official
  Gate3 pass

So this validation stack should currently be read as a repaired
**gate + local-candidate diagnostic surface**, not as evidence that the
repository is ready for new official training.
The checked-in local candidate JSON remains a machine-readable summary of the
latest strongest-contract local result, while the newer Gate2/Gate3 reruns are
still local online candidates rather than official training-unblock artifacts.

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
  `prompt_global_weight_present`, actual `prompt_global_weight`, and
  `prompt_unit_log_prior_present`, and
  `prompt_unit_prior_vocab_size`
- source-prefix reconstruction sidecars such as
  `rhythm_v3_source_rate_init` and `rhythm_v3_src_rate_init_mode`, so offline
  audit can replay the same init contract as runtime
- writer / projector observability such as
  `rhythm_debug_detach_global_term_in_local_head`,
  `projector_preclamp_exec`,
  `projector_clamp_mass`, and
  `projector_rounding_regret`

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
- the maintained export CLI should stay singular:
  `scripts/rhythm_v3_debug_records.py` exports the shared review/gate bundle
- narrower gate inspection is still allowed, but only through a small set of
  targeted zero-train helpers:
  `audit_rhythm_v3_boundary_support.py`,
  `audit_rhythm_v3_counterfactual_static_gate0.py`,
  `probe_rhythm_v3_gate1_analytic.py`,
  `probe_rhythm_v3_gate1_silent_counterfactual.py`, and
  `rhythm_v3_probe_cases.py`
- the old standalone `scripts/smoke_test_rhythm_v3.py` wrapper is retired;
  structural smoke coverage now comes from targeted tests plus short task-level
  smoke runs

Some earlier implementation suggestions are therefore already outdated in this
workspace:

- no new `g_stats.py` is needed because the shared helper already exists
- no second layer of posthoc reducer CLIs is kept anymore
- no second runtime surface is needed because `analytic / coarse_only / learned`
  already switch inside the maintained `rhythm_v3` path

Recent local review utilities also now interpret real-reference triplets using
the current `slow / mid / fast` semantics rather than the older `real`-only
bucket assumption. Current Gate1/Gate2 readings should be interpreted on that
updated semantics.

The recent Gate3 change in this workspace should be read as a runtime wiring
fix only:

- local minimal-V1 candidate work under
  `strict_minimal_claim_profile=false` may relax
  `rhythm_v3_disable_learned_gate=false`
- but minimal-V1 runtime must still avoid reinterpreting that as
  `use_learned_residual_gate=true`
- this fixed a local training unblock; it did not add a new model capability

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
- ladder table: `analytic_signal_slope`, `prefix_signal_explainability_slope`
- gate status: `analytic_negative_control_gap`, `analytic_same_text_gap`,
  `alignment_mean_local_confidence_speech`,
  `alignment_mean_coarse_confidence_speech`

Gate-1 interpretation is now stricter than an earlier "record-only" phase:

- `analytic_negative_control_gap <= 0` is a hard gate failure
- large positive `analytic_same_text_gap` is a hard gate failure
- the same threshold is used in both the issue list and `gate1_pass`, so the
  audit summary and the pass/fail verdict no longer diverge

Gate-0 is also now tied to the maintained runtime support surface:

- static `g` review uses the same clean-support builder as runtime when
  `closed_mask` / `boundary_confidence` sidecars are present
- Gate-0 validity now also requires source-side support/domain validity, not
  just prompt-side validity
- Gate-0 y-side speech summaries are now computed on the same source-support
  surface used to define `delta_g`
- prompt speech-ratio auditing is duration-weighted, matching dataset/runtime
- strict gate fails on low alignment confidence means and low
  `alignment_local_margin_p10`, not just on missing metadata
- Gate-0 static audit now separates:
  `delta_g -> total_signal`,
  `delta_g -> analytic_signal`,
  `delta_g -> coarse_residual`,
  plus a prefix-baseline comparison
- Gate-0 static audit now also reports:
  - `clean_total_claim` vs
    `cross_text_prompt_vs_cross_speaker_target`
  - median-based and mean-based totals
  - speech-total duration log-ratio
  - runtime-clipped analytic/residual views aligned with writer clip
  - affine runtime residual diagnostics
- Gate1-upper uses `rhythm_v3_src_prefix_stat_mode=exact_global_family` to
  measure the stronger offline/local upper bound
- Gate1-online uses the maintained `weighted_median + ema` contract
- current local falsification also defaults Gate0 to
  `reference_mode=target_as_ref`, so the clean local total slice is now
  available by default
- current local falsification now also relies on runtime init parity for
  `source_rate_seq` reconstruction instead of allowing zero-init drift
- training target construction now carries the same `g` / prefix contract
  fields used by Gate0 and Gate1, avoiding silent fallback to default
  `g_variant` / `ema` target settings
- `exact_global_family` should still be read here as a local/offline gate
  contract: current strict online continuation still carries only a scalar
  prefix state, so full-history exact robust-prefix semantics are not yet a
  maintained runtime guarantee

Gate-1 is now tied to the runtime `g` contract, but it exposes both the upper
and online readings explicitly:

- probe-case selection uses runtime-equivalent `prompt_g_ref`
- monotonicity ordering uses `prompt_tempo_ref_runtime`, not display-only
  prompt tempo proxies
- higher `prompt_g_ref` means slower speech, so the positive Gate1-online
  control axis is `prompt_tempo_ref_runtime = exp(-g_ref)`
- prompt/source tempo readouts now reuse weighted prompt/source contracts when
  `g_variant` is weighted or softclean
- Gate-1 reports layered readouts:
  - `tempo_out_raw`
  - `tempo_out_preproj`
  - `tempo_out_exec`
- Gate-1 also reports continuous count/logstretch readouts in parallel with the
  discrete tempo readout so projector/readout collapse can be separated from
  earlier writer failure
- silent counterfactual Gate-1 now also rejects zero-range collapse instead of
  treating it as a trivial monotone pass

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
- `g_src_prefix_final`: final committed prefix state, which is the main
  runtime-aligned scalar for Gate0/Gate1 prefix comparisons

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
- negative controls only count when they actually lose to real references on the
  exported control metrics; existence alone is not a verdict

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

That proxy is only as good as the cached sidecars. In the maintained dataset
path, adapted-source debug export now preserves cached `source_boundary_cue`,
`phrase_group_index`, `phrase_group_pos`, and `phrase_final_mask` when their
shapes still match the visible prefix, instead of zero-filling them
unconditionally.

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
When prompt-side clean-support sidecars are present (`prompt_closed_mask`,
`prompt_boundary_confidence`, `prompt_ref_len_sec`), the maintained mainline
`g` path uses them for domain gating even though the generic review summary
still reports the core speech-support counters as the stable cross-run audit
surface. Missing strict sidecars now fail closed in dataset/runtime contracts,
and runtime debug reads the exported `prompt_g_*` evidence instead of
recomputing a looser speech-only support mask.
For minimal-V1 training data, this check is now pushed forward into the dataset
layer as well: missing prompt speech / closed / length sidecars are treated as
contract violations before the batch reaches model forward.

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
- `projector_preclamp_exec`
- `projector_clamp_mass`
- `projector_rounding_regret`

`projector_boundary_hit` marks boundary/phrase-final events, while
`projector_boundary_decay_applied` only marks the subset where decay was
actually applied. `projector_preclamp_exec`, `projector_clamp_mass`, and
`projector_rounding_regret` make the pre-vs-post projector audit explicit
instead of inferring it indirectly from budget hits alone.

### 5.5 Closed and committed are different

Figure A uses `is_closed`.
Figure E uses `is_committed`.

That separation is deliberate. Open tails should not be counted as interface
instability, and closed-but-not-yet-committed runs should not be counted as a
commit violation.

### 5.6 Canonical minimal-V1 files are narrower than the legacy wrapper names

For review and maintenance, the maintained minimal-V1 implementation should be
read through:

- `modules/Conan/rhythm_v3/global_condition.py`
- `modules/Conan/rhythm_v3/minimal_writer.py`
- `modules/Conan/rhythm_v3/run_encoder.py`

The following files still exist, but mostly as generic containers or
compatibility surfaces:

- `modules/Conan/rhythm_v3/summary_memory.py`
- `modules/Conan/rhythm_v3/minimal_head.py`
- `modules/Conan/rhythm_v3/role_memory.py`

That distinction is intentional. It keeps the maintained V1 theory surface
smaller than the total legacy import surface exposed by the package.

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
  --g-variant weighted_median ^
  --drop-edge-runs 1
```

This now writes:

- the row-level falsification summary CSV
- the retained five-figure review bundle
- the gate-oriented monotonicity / stability / ladder bundle

When `--review-dir` or `--gate-status-json` is present, the maintained script
now fails non-zero on gate issues by default. Use `--allow-partial-gates` only
for an explicitly partial audit export.
