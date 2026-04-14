# rhythm_v3 minimal-V1 falsification log

This document records the **falsification-first** workflow for the maintained
`rhythm_v3` minimal-V1 path.

The goal is still narrow:

> In short, speech-dominant, same-speaker / different-text references,
> `g_ref` should behave like a stable and transferable speech-only global cue,
> while speech moves more than silence and committed prefixes remain usable.

If that claim fails, the right answer is not "add more branches". The right
answer is to question the statistic, the interface, or the runtime contract.

## 0. Latest maintained local rerun

Latest local rerun date:

- `2026-04-14`

Canonical status snapshot:

- `docs/rhythm_v3_local_status_2026-04-14.md`

Latest local artifacts:

- `tmp/gate_reaudit_20260414/full_split_boundary_audit_report.json`
- `tmp/gate_reaudit_20260414/counterfactual_static_gate0_report.json`
- `tmp/gate_reaudit_20260414/counterfactual_static_gate0_direction_report.json`
- `tmp/gate_reaudit_20260414/gate1_analytic_summary.json`

Current verdict:

- Gate 0: fail
- Gate 1: fail
- Gate 2: blocked
- Gate 3: blocked

Why the project is currently stopped before training:

- maintained `min_boundary_confidence_for_g=0.5` still yields
  `g_domain_valid_items = 0/32` on train, `0/16` on valid, and `0/16` on test
- the only recovered positive Gate-0 total-signal slice remains a
  counterfactual non-maintained setting (`token=71`, `drop_edge=1`,
  `valid_signal_slope=0.3106`, `valid_prefix_signal_slope=0.2601`,
  `valid=13/64`)
- strict Gate-1 analytic rerun now completes without crashing, but all four
  probe sources still end with `valid_real=0/3`, `transfer_slope=nan`, and
  `monotone_by_prompt_tempo=false`

So the current maintained local conclusion is:

- do not start Gate-2 / Gate-3 training
- do not start official training
- keep work on zero-train audit, prompt-domain cleanup, and `g` falsification
  only

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
- `rhythm_debug_detach_global_term_in_local_head`
- `rhythm_debug_is_speech`
- `rhythm_debug_budget_hit_pos`
- `rhythm_debug_budget_hit_neg`
- `rhythm_debug_projector_boundary_hit`
- `rhythm_debug_projector_boundary_decay`
- `rhythm_debug_projector_since_last_boundary`
- `projector_preclamp_exec`
- `projector_clamp_mass`
- `projector_rounding_regret`
- `unit_duration_exec`

So the review surface got smaller, but the underlying observability was not
reduced.
`rhythm_debug_projector_boundary_hit` marks boundary events;
`rhythm_debug_projector_boundary_decay` marks only the subset where decay was
actually applied. `projector_preclamp_exec`, `projector_clamp_mass`, and
`projector_rounding_regret` expose how much of the final execution came from
the writer versus projector correction.

### 2.4 `g` is now locked to one support path

The local implementation should keep training, runtime diagnostics, and review
on the same `g` semantics:

- speech-only support first
- optional `rhythm_v3_drop_edge_runs_for_g` cleanup
- prompt-side closed / boundary-confidence filtering when the clean-support
  sidecars are available
- the maintained minimal-V1 domain also expects speech-dominant 3-8 second
  references when `prompt_ref_len_sec` is provided
- edge-drop fallback only back to raw speech support for generic summary
  diagnostics, never to non-speech valid/full support in the maintained
  minimal-V1 prompt path
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

## 5. Falsification gates remain the same in shape, but stricter in evidence

### Gate 0: static `g` audit

Stop early if:

- crop stability is weak
- `delta_g` barely explains `c*`
- different-text collapses while same-text looks fine
- the analytic explainability slope is missing or non-positive on the
  maintained gate export

### Gate 1: pure analytic control

Run `analytic` mode and stop early if:

- slow / mid / fast references do not induce monotone speech tempo movement
- no negative-control reference (`source_only`, `random_ref`, `shuffled_ref`)
  is exported alongside the real-reference triplets
- the exported negative-control gap fails to stay positive

### Gate 2: coarse-only before local

Gate 2 is the stand-alone coarse-only stop/go check. At this stage the question
is whether `a_i + b` already behaves like a usable source-anchored global
retimer without local residual help.

### Gate 3: local residual after coarse-only

Only keep strong local residual if it improves control without obviously
damaging silence leakage or prefix stability. In the maintained minimal-V1
contract, `detach_global_term_in_local_head=true` is the default here; compare
`learned + no_detach` only as an explicit ablation, not as the maintained
baseline. Gate 3 should reject a learned writer that regresses monotonicity or
transfer slope relative to `coarse_only`, steals coarse bias, or leaks local
delta into silence.

### Outdated suggestions

Several earlier "next steps" are now already implemented locally and should not
be re-proposed as new refactors:

- shared `g` computation already lives in `modules/Conan/rhythm_v3/g_stats.py`
- `analytic / coarse_only / learned` already run inside the same maintained runtime
- projector budget/drift observability is already exported; the remaining work
  is interpretation, especially `boundary_hit` vs `boundary_decay_applied` and
  the new pre-vs-post correction telemetry, not projector replacement

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
- analytic explainability slope is missing or non-positive
- unmatched speech alignment is high
- continuous coverage is incomplete for the current `alignment_kind` split
- `analytic / coarse_only / learned` modes are missing
- real-reference triplets are incomplete
- negative controls are absent
- negative controls fail to lose on the maintained analytic gap metric

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
- implementation checkpoints:
  - detach default: closed
  - boundary-clean `g` support path: closed
  - projector pre/post telemetry: closed
  - CI pytest coverage for rhythm tests: closed
  - negative-control failure semantics in strict gate export: closed
- empirical verdict: still open

### 2026-04-14 quick ARCTIC Gate-1 diagnostic

This turn did run a focused **zero-training analytic probe** on the local quick
ARCTIC subset, but it should still be read as a diagnostic rather than a full
maintained gate pass.

Files / artifacts:

- `scripts/probe_rhythm_v3_gate1_analytic.py`
- `scripts/audit_rhythm_v3_boundary_support.py`
- `tmp/gate1_boundary_audit/train_boundary_stats.json`
- `tmp/gate1_boundary_audit/valid_boundary_stats.json`
- `tmp/gate1_boundary_audit/full_split_boundary_audit_rows.csv`
- `tmp/gate1_boundary_audit/full_split_boundary_audit_report.json`
- `tmp/gate1_analytic_probe_runtimefixed/summary.json`
- `tmp/gate1_analytic_probe_bc045/summary.json`
- `tmp/gate1_analytic_probe_bc04/summary.json`
- `tmp/gate1_analytic_probe_bc0/summary.json`

Two implementation bugs were fixed before reading the result:

- prompt clean-support sidecars were being dropped in
  `normalize_duration_v3_conditioning`
- prompt clean-support sidecars were also missing from the
  `_RHYTHM_RUNTIME_MINIMAL_KEYS` batch contract, so collate/runtime could not
  carry them through

After those fixes, the maintained analytic runtime still failed on the quick
subset for a stronger reason:

- quick `train`: `0 / 32` prompts have `clean_count > 0` at
  `rhythm_v3_min_boundary_confidence_for_g = 0.5`
- quick `valid`: `0 / 16` prompts have `clean_count > 0` at the same threshold
- `bc_max` stays below `0.49` for every inspected quick prompt
- source caches also show `sep_hint == 0` for every inspected quick prompt, so
  boundary confidence is driven only by local peak/jump cues and never gets the
  `+0.55 * sep_hint` boost

Threshold sweep on quick prompt clean-support viability:

- `0.50`: train `0 / 32`, valid `0 / 16`
- `0.45`: train `24 / 32`, valid `11 / 16`
- `0.40`: train `30 / 32`, valid `15 / 16`
- `0.35`: train `32 / 32`, valid `16 / 16`

That means the maintained `0.5` threshold is not "barely missed"; on the quick
subset it is structurally unreachable with the current cache generator and data
surface.

The reusable full-split boundary audit now makes the upstream mismatch more
explicit:

- all `64 / 64` local prompts have `raw_silent_token_count == 0` for the
  configured `silent_token = 57`
- all `64 / 64` prompts also have `sep_hint == 0` and
  `source_silence_run_count == 0`
- at the maintained threshold `0.50`, `clean_positive_items = 0 / 64` and
  `g_domain_valid_items = 0 / 64`
- even after relaxing to `0.45`, only `48 / 64` prompts become
  boundary-clean-positive and just `38 / 64` remain runtime-domain-valid
- at `0.35`, `clean_positive_items` finally reaches `64 / 64`, but
  `g_domain_valid_items` still caps at `46 / 64` because the maintained
  `3-8s` prompt-length gate continues to reject short references

So the current local data surface is missing the separator sentinel that the
maintained clean-support contract implicitly expects; this is stronger than
"boundary confidence is weak" and should be treated as a cache/runtime contract
mismatch.

Gate-1 analytic probe result under the maintained threshold (`0.5`):

- all four sampled same-speaker / different-text slow-mid-fast triplets
  collapsed to constant `tempo_out`
- `g_ref = 0`, `g_domain_valid = 0`, and real references were indistinguishable
  from `source_only` / `random_ref`
- this is a valid maintained **Gate-1 fail**, but the failure is upstream of
  analytic monotonicity itself: the prompt clean-support contract is never met

Diagnostic threshold relaxation did **not** show a simple smooth recovery:

- at `0.45`, several speakers became **anti-monotone**
- at `0.40`, some cases remained flat and one stayed anti-monotone
- only at `0.0` did `asi / bdl / slt` recover positive transfer slopes, while
  `aba` still remained effectively flat

So the strongest current reading is:

- maintained Gate 1 on quick ARCTIC fails because prompt clean-support is not
  available at the maintained threshold
- this is not enough evidence to green-light Gate 2 training
- relaxing the threshold a little does not stably recover the claim; it first
  exposes unstable / anti-monotone behavior
- fully removing the boundary-confidence gate partially revives analytic
  control, which suggests the analytic path is not completely broken, but the
  maintained prompt-domain contract is not satisfied by the current quick cache

Additional boundary-cue audit:

- quick `train`: `sep_hint` is `0` for all `32 / 32` inspected prompt/source
  caches
- quick `valid`: `sep_hint` is `0` for all `16 / 16`
- full local split: raw HuBERT units contain `0 / 64` occurrences of the
  configured `silent_token = 57`
- full local split: `sep_hint` is `0` for all `64 / 64` prompts, and
  `source_silence_run_count` is also `0 / 64`
- `prompt_phrase_final_mask` still exists because phrase-cache construction
  marks phrase-final runs from the floating boundary cue plus the final visible
  run; it is not evidence that separator-aware boundary support exists

That matters because the current boundary-confidence generator is:

- local duration peak cue
- local duration jump cue
- `+ 0.55 * sep_hint`
- then open-run discount

With `silent_token = 57` absent in the raw HuBERT stream and `sep_hint == 0`
everywhere, quick ARCTIC boundary confidence never gets the separator boost, so
the maintained `0.5` clean-support threshold is structurally difficult to
reach.

Phrase-final diagnostic:

- as a pure diagnostic, replacing `prompt_boundary_confidence` with
  `max(prompt_boundary_confidence, prompt_phrase_final_mask)` and recomputing
  prompt global weights does force `clean_count > 0`
- but it does **not** rescue a stable maintained Gate-1 claim
- the resulting analytic behavior stays unstable / flat / partially
  anti-monotone depending on speaker

So `phrase_final_mask` can help explain where boundary-like events might be,
but it is not a valid substitute for the maintained floating
`prompt_boundary_confidence` contract and does not resolve the current
falsification outcome.

Counterfactual silent-token sweep:

- `scripts/sweep_rhythm_v3_silent_token_candidates.py`
- `scripts/probe_rhythm_v3_gate1_silent_counterfactual.py`
- `tmp/gate1_boundary_audit/silent_token_sweep_report.json`
- `tmp/gate1_boundary_audit/silent_token_sweep_rows.csv`
- `tmp/gate1_counterfactual_probe/token71_summary.json`
- `tmp/gate1_counterfactual_probe/token72_summary.json`
- `tmp/gate1_counterfactual_probe/token63_summary.json`
- `tmp/gate1_counterfactual_probe/token57_summary.json`

This was a pure diagnostic and **not** a maintained config change:

- prompt-side caches were rebuilt with `emit_silence_runs=false`
- only the prompt conditioning was replaced
- the maintained analytic runtime stayed unchanged

The sweep answers a narrower question:

- if the current failure were just "wrong silent token id", is there a better
  candidate in the raw HuBERT stream that rescues prompt clean-support and then
  revives analytic control?

Raw-token audit says `57` is not a plausible answer:

- `57` occurs `0` times in the local `metadata.json` HuBERT streams
- the most boundary-like tokens are split by edge type rather than collapsing
  into one clean universal silence token:
  - `71`: strongest sentence-initial candidate
  - `72`: strongest sentence-final candidate
  - `63`: weaker sentence-final candidate
- high-frequency tokens like `12` and `4` can trivially create separator events
  if forcibly dropped, but they behave like content units rather than a clean
  pause symbol

Counterfactual support sweep result:

- if we forcibly treat `12`, `4`, or `63` as dropped-silence tokens, we can
  manufacture `sep_hint` and boundary-clean support on almost every item
- this does **not** validate those tokens as silence; it shows the support
  statistic is easy to game if a common content token is reinterpreted as a
  separator
- the more plausible edge candidates (`71`, `72`, `63`) do improve
  `clean_count` relative to `57`, but they still do not define a stable,
  speaker-independent pause token

Runtime counterfactual Gate-1 probe:

- `57` baseline: `0 / 4` sources showed any real slow/mid/fast movement; all
  triplets stayed flat relative to controls
- `71` (best candidate): only `2 / 4` sources showed any real movement relative
  to controls; `2 / 4` still stayed flat
- `72`: only `1 / 4` source moved, and one source became non-monotone
- `63`: `2 / 4` sources moved, but one source became non-monotone and the mean
  negative-control gap was slightly negative

One important nuance:

- plain monotonicity rate is not enough here, because a completely flat
  slow/mid/fast triplet also counts as monotone
- the stronger read is "did real references separate from `source_only` /
  `random_ref` by a useful margin?"
- under that criterion, even the best plausible counterfactual token (`71`)
  only produces partial movement and does not rescue a robust Gate-1 claim

So the current evidence is:

- the maintained failure is not just "token `57` is wrong"
- even after allowing a non-maintained dropped-silence counterfactual, the best
  plausible boundary-like token does not restore stable analytic control
- therefore Gate 1 should still be treated as failed, and Gate 2 training
  remains blocked on the local data surface

Structural audit of the best candidate tokens:

- `scripts/audit_rhythm_v3_candidate_token_structure.py`
- `tmp/gate1_boundary_audit/candidate_token_structure_report.json`

This audit asks a stricter question:

- even if `71 / 72 / 63` look more boundary-like than `57`, do they behave like
  clean pause symbols, or are they just boundary-biased content tokens?

The answer is closer to "boundary-biased content tokens":

- `71` appears in `64 / 64` items and always at sentence start, but it is also
  present internally in `64 / 64` items; its run-level edge fraction is only
  `0.184`
- `72` looks more sentence-final than `71`, but it still appears internally in
  `51 / 64` items; its run-level edge fraction is `0.143`
- `63` is even less boundary-pure: internal in `64 / 64` items, with
  run-level edge fraction only `0.034`

Cross-speaker / cross-prompt slices make the artifact story stronger:

- all `16 / 16` prompts start with `71` for all four speakers, so `71` is
  strongly sentence-initial but not sentence-initial-only
- none of `71 / 72 / 63` gives a uniform sentence-final token across prompts;
  prompt endings are mixed across speakers
- `72` is especially speaker-biased at sentence end:
  `aba: 3`, `asi: 5`, `bdl: 5`, `slt: 11` end-of-utterance hits
- `63` is also mixed by speaker at sentence end:
  `aba: 5`, `asi: 4`, `bdl: 6`, `slt: 1`

So even the best counterfactual candidates are not a clean replacement for a
single maintained `silent_token`:

- they have some edge bias
- they are heavily reused inside utterances
- their final-token behavior varies across speakers on the same prompt

That is more consistent with recording/extractor boundary coloration mixed into
ordinary content units than with a stable, speaker-independent pause symbol.

Boundary-trim counterfactual probe:

- `scripts/probe_rhythm_v3_gate1_silent_counterfactual.py`
- `tmp/gate1_counterfactual_probe/boundary_trim/token71_trim_h1_summary.json`
- `tmp/gate1_counterfactual_probe/boundary_trim/token71_trim_h1_t1_summary.json`
- `tmp/gate1_counterfactual_probe/boundary_trim/token72_trim_t1_summary.json`
- `tmp/gate1_counterfactual_probe/boundary_trim/token63_trim_t1_summary.json`

This probe strengthens the previous counterfactual in one specific way:

- the prompt cache is trimmed **before**
  `_build_reference_prompt_unit_conditioning()`
- so `prompt_global_weight`, `prompt_speech_ratio_scalar`, and
  `prompt_ref_len_sec` are recomputed on the trimmed run lattice rather than
  hacked after the fact
- the runtime summary now also logs:
  - `real_tempo_range`
  - `max_gap_vs_source_only`
  - `max_gap_vs_random_ref`

That matters because plain monotonicity is still too weak: a fully flat
slow/mid/fast triplet remains monotone, but it is not a valid control claim.

`71` head-trim result:

- baseline `71` had aggregate mean `max_gap_vs_random_ref = 0.0788`
- trimming the prompt head by one run reduced that mean to `0.0438`
- trimming both head and tail stayed effectively the same at `0.0438`
- the stricter "moved by > 0.05 on either real-range or control-gap" count
  dropped from `3 / 4` sources to `2 / 4`

Per-source read for `71`:

- `aba` stayed completely flat in all settings
- `bdl` stayed flat, and its only previous separation from `random_ref`
  collapsed from `0.0651` to `0.0000` after head trim
- `slt` kept the same real triplet range (`0.1667`), but its
  `random_ref` gap shrank from `0.1667` to `0.0918`
- `asi` was essentially unchanged by trim and kept only a weak partial
  separation (`real_range = 0.0833`, both controls at `0.0833`)

So the best reading is:

- some of `71`'s apparent control signal **was** boundary-adjacent contamination
- but trimming literal first/last runs does **not** collapse the whole effect
- the residual movement is still too weak and too speaker-specific to count as
  a robust prompt tempo cue

Tail-trim checks on the more sentence-final candidates did not help:

- `72` with `trim_tail_runs = 1` remained `3 / 4` monotone only because three
  sources stayed flat; the only moving source (`slt`) remained non-monotone
- `63` with `trim_tail_runs = 1` remained weak and unstable:
  `bdl` stayed non-monotone and only `slt` showed a modest positive range

So the boundary-trim experiment does **not** rescue the analytic claim:

- `71` is not explained by a clean first-run pause cue
- `72 / 63` are not rescued by dropping the final run
- the surviving behavior is more consistent with inward-spreading boundary
  coloration or speaker/content confounds than with a stable pause token

Current Gate-1 conclusion after the trim probe:

- Gate 1 still fails on the local data surface
- this is no longer just "configured silent token `57` is absent"
- even the best non-maintained counterfactual token keeps only partial,
  non-general movement after edge trimming
- Gate 2 training remains blocked until a better speech-only global control
  surface is found

Edge-depth counterfactual probe:

- `tmp/gate1_counterfactual_probe/edge_depth/token71_drop2_summary.json`
- `tmp/gate1_counterfactual_probe/edge_depth/token71_drop3_summary.json`
- `tmp/gate1_counterfactual_probe/edge_depth/token71_trim_h1_drop2_summary.json`
- `tmp/gate1_counterfactual_probe/edge_depth/token71_trim_h1_drop3_summary.json`
- `tmp/gate1_counterfactual_probe/edge_depth/token71_drop3_support_audit.json`
- `tmp/gate1_counterfactual_probe/edge_depth/token71_trim_h1_drop3_support_audit.json`

This probe tightens the boundary story one step further:

- instead of only trimming the prompt cache, it also increases
  `rhythm_v3_drop_edge_runs_for_g`
- the runtime default already drops `1` support edge run
- this diagnostic forces the global support computation to drop `2` and then
  `3` edge runs before computing `g`

Aggregate read for the best counterfactual token (`71`):

- baseline (`drop_edge = 1`):
  - `4 / 4` monotone
  - `2 / 4` sources with `real_range > 0.05`
  - `2 / 4` sources with both control gaps `> 0.05`
- stronger drop (`drop_edge = 2`):
  - monotonicity falls to `3 / 4`
  - `slt` becomes non-monotone
  - dual-control separation stays only `2 / 4`
- strongest tested drop (`drop_edge = 3`):
  - only `1 / 4` source still has `real_range > 0.05`
  - only `1 / 4` source still separates from both `source_only` and
    `random_ref`
  - mean `real_tempo_range` falls from `0.0625` to `0.0208`
  - mean `max_gap_vs_source_only` falls from `0.0625` to `0.0208`

The same pattern survives after head trim:

- `token71 + trim_head_runs = 1 + drop_edge = 3` also falls to just `1 / 4`
  source with both non-flat real tempo range and dual-control separation
- the only surviving source is still `asi`, and even there the behavior is
  only two-level (`slow` differs a bit; `mid` and `fast` collapse together)

Why `asi` survives even at `drop_edge = 3`:

- the support audit shows `support_seed_count = 1.0` for all three `asi`
  real references, both with and without head trim
- in other words, the counterfactual prompt conditioning supplies only a
  **singleton** support run for `g`
- `_drop_edge_support_1d()` only drops edge runs when there are more than
  `2 * drop_edge_runs` active support positions; otherwise it falls back to the
  seed support instead of deleting everything
- so the residual `asi` response at `drop_edge = 3` is not evidence of a
  healthy interior prompt cue; it is the degenerate case where edge dropping
  can no longer cut deeper because the support has already collapsed to one run

Support audit details:

- `asi`
  - all `slow / mid / fast` refs stay at `support_seed_count = support_count = 1`
  - this remains true even after `trim_head_runs = 1`
- `slt`
  - `slow / mid` keep interior multi-run support (`support_count = 5`)
  - `fast` shrinks from `support_seed_count = 9` to `support_count = 3`
    without trim, and to `support_count = 2` after head trim
  - correspondingly, its earlier apparent movement collapses to a flat
    `tempo_out`, leaving at most a one-sided gap against `random_ref`

So the strongest current reading is:

- `token71` does not survive stricter edge removal as a robust global tempo cue
- once edge removal becomes more aggressive, the counterfactual either:
  - turns flat,
  - turns non-monotone, or
  - survives only as singleton-support fallback
- this is stronger evidence for boundary-adjacent contamination and weak
  support degeneracy, not for a usable speech-only prompt tempo variable

Gate-1 status after edge-depth falsification:

- still failed
- the remaining `token71` signal is not enough to justify Gate 2
- the local data surface still lacks a convincing analytic control variable
  under the maintained V1 interpretation

Full-split counterfactual support-degeneracy audit:

- `scripts/audit_rhythm_v3_counterfactual_support_degeneracy.py`
- `tmp/gate1_boundary_audit/counterfactual_support_degeneracy_report.json`
- `tmp/gate1_boundary_audit/counterfactual_support_degeneracy_rows.csv`

This audit asks a broader question than the four-source runtime probe:

- across the full local split (`train + valid + test`), how often do the best
  counterfactual tokens produce a healthy multi-run support surface for `g`?
- and how much of that support collapses to singleton or near-singleton cases
  once edge dropping gets stronger?

The audit was run for `71 / 72 / 63` at `drop_edge_runs_for_g = 1` and `3`.

Full-split summary:

- `71`
  - `drop_edge = 1`: `singleton_seed = 6 / 64`, `singleton_support = 13 / 64`
  - `drop_edge = 3`: `singleton_seed = 6 / 64`, `singleton_support = 12 / 64`
  - `g_domain_valid_items = 23 / 64`
- `72`
  - `drop_edge = 1`: `singleton_seed = 15 / 64`, `singleton_support = 24 / 64`
  - `drop_edge = 3`: `singleton_seed = 15 / 64`, `singleton_support = 15 / 64`
  - `g_domain_valid_items = 29 / 64`
- `63`
  - `drop_edge = 1`: `singleton_seed = 4 / 64`, `singleton_support = 9 / 64`
  - `drop_edge = 3`: `singleton_seed = 4 / 64`, `singleton_support = 8 / 64`
  - `g_domain_valid_items = 32 / 64`

Two important reads follow from that:

- the best runtime candidate (`71`) still shows meaningful full-split support
  degeneracy: roughly one fifth of prompts collapse to singleton support under
  `drop_edge = 3`
- but the opposite failure mode also exists: `63` and `72` often keep broader
  support surfaces, yet they still do **not** rescue stable runtime Gate 1
  control

That means the problem is not just "too few support runs":

- for `71`, singleton / weak-support degeneracy is a real issue
- for `63` and `72`, even thicker support is not enough to establish a valid
  speech-only global tempo cue
- so "support survives" and "tempo control survives" are separate claims, and
  the latter still fails

Speaker slice for `71` at `drop_edge = 3`:

- `asi`: `singleton_seed = 4 / 16`, `singleton_support = 5 / 16`
- `aba`: `singleton_support = 4 / 16`
- `slt`: `singleton_support = 2 / 16`
- `bdl`: no singleton-support rows in the full audit, but its runtime probe
  cases still stayed flat

That matches the earlier runtime probe:

- `asi` is the speaker most prone to singleton-support fallback, which explains
  why it can preserve a weak residual response even after strong edge dropping
- `bdl` shows the opposite pattern: it can keep non-singleton support without
  producing useful runtime control

Concrete singleton-support examples for `71` at `drop_edge = 3`:

- `asi_train_arctic_a0013`: `support_seed_count = support_count = 1`
- `asi_train_arctic_a0014`: `support_seed_count = support_count = 1`
- `asi_train_arctic_a0015`: `support_seed_count = support_count = 1`
- `bdl_valid_arctic_a0004`: `support_seed_count = support_count = 1`
- `aba_test_arctic_a0007`: `support_seed_count = support_count = 1`

So the current global reading is stronger than before:

- `71` is partly a weak-support / singleton-support artifact
- `72` and `63` can preserve more support mass, but that support still does
  not behave like a usable global tempo variable in runtime analytic control
- therefore the local Gate-1 failure is not just a support-construction bug;
  it is a deeper failure of the candidate counterfactual tokens to instantiate
  the claimed `g`

Counterfactual static Gate-0 audit:

- `scripts/audit_rhythm_v3_counterfactual_static_gate0.py`
- `tmp/gate1_boundary_audit/counterfactual_static_gate0_report.json`
- `tmp/gate1_boundary_audit/counterfactual_static_gate0_rows.csv`

This is a stricter static falsification than the runtime triplet probe:

- it walks the full local pair surface (`64` pairs)
- rebuilds prompt conditioning with counterfactual tokens
  `57 / 71 / 72 / 63`
- recomputes prompt-side `g_ref` under the maintained clean-support semantics
- then compares `delta_g = g_ref - g_src_utt` against the pair's
  coarse oracle `c_star`

Important property of this pair surface:

- all `64 / 64` rows are `cross-text`
- there is no same-text slice to hide behind here
- so the sign of `delta_g -> c_star` is already the cross-text sign

Result:

- every tested token gives a **non-positive** static Gate-0 slope
- the better-supported candidates (`63` and `72`) are not merely weak; they are
  strongly **negative**

Detailed summary:

- `57`
  - `g_domain_valid = 0 / 64`
  - overall robust slope:
    - `drop_edge = 1`: `-0.3504`
    - `drop_edge = 3`: `-0.2339`
- `71`
  - `g_domain_valid = 13 / 64`
  - overall robust slope:
    - `drop_edge = 1`: `-0.7666`
    - `drop_edge = 3`: `-0.6897`
  - valid cross-text slope:
    - `drop_edge = 1`: `-0.4659`
    - `drop_edge = 3`: `-0.0781`
- `72`
  - `g_domain_valid = 28 / 64`
  - overall robust slope:
    - `drop_edge = 1`: `-0.7678`
    - `drop_edge = 3`: `-0.6907`
  - valid cross-text slope:
    - `drop_edge = 1`: `-0.7525`
    - `drop_edge = 3`: `-0.6505`
- `63`
  - `g_domain_valid = 29 / 64`
  - overall robust slope:
    - `drop_edge = 1`: `-0.7893`
    - `drop_edge = 3`: `-0.5917`
  - valid cross-text slope:
    - `drop_edge = 1`: `-0.8167`
    - `drop_edge = 3`: `-0.4612`

This matters more than the earlier support-only audits:

- `63` and `72` were the tokens with relatively thicker support surfaces
- but even when restricted to their valid-domain rows, the static
  `delta_g -> c_star` relationship still points the wrong way
- so thicker support does not rescue the core Gate-0 claim

For `71`, the picture is also now clearer:

- the token is weakly valid on only `13 / 64` pairs
- once restricted to those valid rows and `drop_edge = 3`, the slope moves
  toward zero (`-0.0781`) rather than becoming positive
- so the earlier residual runtime behavior was not a hidden positive cue; it
  was the edge of degeneracy, not evidence for the theory

Concrete row-level examples make the sign error explicit:

- `aba_train_arctic_a0009 <- aba_train_arctic_a0010` under `71, drop_edge = 1`:
  `delta_g = +1.5041`, `c_star = -2.0794`
- `aba_train_arctic_a0011 <- aba_train_arctic_a0009` under `71, drop_edge = 1`:
  `delta_g = +0.6020`, `c_star = -1.2906`
- `aba_train_arctic_a0016 <- aba_train_arctic_a0009` under `71, drop_edge = 1`:
  `delta_g = +0.7458`, `c_star = -1.4145`

So the strongest current reading is now:

- the candidate counterfactual tokens do not merely fail to give a clean prompt
  support surface
- on the actual local pair surface, they also fail the more basic Gate-0 test:
  `delta_g` does not explain `c_star` with the correct sign
- this is a deeper falsification than Gate 1 alone and it further blocks any
  move toward Gate 2 training on this local surface

Counterfactual static Gate-0 directionality audit:

- `scripts/analyze_counterfactual_static_gate0_direction.py`
- `tmp/gate1_boundary_audit/counterfactual_static_gate0_direction_report.json`

This audit tightens the static Gate-0 result by asking whether the negative
slopes above are just weak / noisy, or whether the candidate `delta_g` is
systematically pointing in the **wrong direction**.

Method:

- reuse the full pair-surface rows from
  `tmp/gate1_boundary_audit/counterfactual_static_gate0_rows.csv`
- count same-sign vs opposite-sign agreement between `delta_g` and `c_star`
  after excluding near-zero rows
- compare the original explainability slope to the sign-flipped slope
  obtained from `(-delta_g) -> c_star`

Result:

- `63 / 71 / 72` are opposite-sign more often than same-sign
- sign-flipping `delta_g` makes the robust slope positive with nearly the same
  magnitude
- so these candidates are not merely uninformative; they are anti-aligned with
  the maintained control direction

Overall directionality summary at `drop_edge = 1`:

- `63`
  - same-sign rate: `0.328`
  - opposite-sign rate: `0.672`
  - original slope: `-0.7893`
  - flipped slope: `+0.7884`
- `71`
  - same-sign rate: `0.317`
  - opposite-sign rate: `0.683`
  - original slope: `-0.7666`
  - flipped slope: `+0.7657`
- `72`
  - same-sign rate: `0.309`
  - opposite-sign rate: `0.691`
  - original slope: `-0.7678`
  - flipped slope: `+0.7676`

Valid-domain slices preserve the same story:

- `63`
  - valid rows: `29`
  - same-sign rate: `0.346`
  - opposite-sign rate: `0.654`
  - original slope: `-0.8167`
  - flipped slope: `+0.8167`
- `71`
  - valid rows: `13`
  - same-sign rate: `0.250`
  - opposite-sign rate: `0.750`
  - original slope: `-0.4659`
  - flipped slope: `+0.4659`
- `72`
  - valid rows: `28`
  - same-sign rate: `0.250`
  - opposite-sign rate: `0.750`
  - original slope: `-0.7525`
  - flipped slope: `+0.7525`

Speaker-valid slices are noisier because some counts are tiny, but they still
show the same pattern on the best-supported speaker slices:

- `63`, `aba`: same-sign `0.308`, opposite-sign `0.692`
- `72`, `aba`: same-sign `0.231`, opposite-sign `0.769`
- `71`, `aba`: same-sign `0.300`, opposite-sign `0.700`

This sharpens the interpretation of the earlier static Gate-0 failure:

- the candidate counterfactual tokens are not failing because the slope is too
  small to trust
- they are failing because the extracted `delta_g` tends to push in the
  opposite direction from the oracle coarse target
- therefore “maybe the cue is present but too weak” is no longer a credible
  explanation for `63 / 71 / 72`

Practical consequence:

- the next falsification step should stay zero-training
- but it should move from “can we make support survive?” to
  “does any runtime behavior remain if we read these candidates as an
  anti-control signal rather than a maintained control signal?”

Counterfactual runtime anti-control audit:

- `scripts/probe_rhythm_v3_gate1_silent_counterfactual.py`
  - extended to emit grouped summaries under three orderings:
    `prompt_tempo_ref`, `delta_g`, and `-delta_g`
- `scripts/analyze_counterfactual_runtime_anticontrol.py`
- `tmp/gate1_counterfactual_probe/anti_control/runtime_anti_control_report.json`

This runtime audit asks a narrower question than Gate 1:

- not “do these candidates support the maintained analytic control claim?”
- but “when any residual runtime movement survives, does it line up with
  maintained `delta_g`, or does it line up better with `-delta_g`?”

The audit uses the existing zero-training runtime probe outputs and counts a
source as an active runtime response only when both conditions hold:

- `real_tempo_range >= 0.01`
- `max(max_gap_vs_source_only, max_gap_vs_random_ref) >= 0.01`

Then it asks whether each active source is explained by:

- prompt control:
  monotone under `prompt_tempo_ref` with positive prompt slope
- maintained `delta_g` control:
  monotone under `delta_g` with positive `delta_g` slope
- anti-control:
  monotone under `-delta_g` with positive `-delta_g` slope

Result:

- maintained `delta_g` explains **zero** active runtime sources in every run
  variant tested
- `-delta_g` explains the surviving non-flat runtime cases more often than
  maintained prompt control for `63`, and strictly more often than prompt
  control for `72`
- but this still does **not** rescue Gate 1, because most sources remain flat
  or control-indistinct and the static pair-surface sign is still wrong

Runtime rollup:

- `63`, base, `drop_edge = 1`
  - active sources: `2 / 4`
  - prompt-control responses: `1`
  - maintained-`delta_g` responses: `0`
  - anti-control responses: `2`
  - anti-only source: `bdl_train_arctic_a0014`
- `71`, base, `drop_edge = 1`
  - active sources: `2 / 4`
  - prompt-control responses: `2`
  - maintained-`delta_g` responses: `0`
  - anti-control responses: `2`
  - anti-control and prompt both cover the same residual runtime cases
- `72`, base, `drop_edge = 1`
  - active sources: `1 / 4`
  - prompt-control responses: `0`
  - maintained-`delta_g` responses: `0`
  - anti-control responses: `1`
  - anti-only source: `slt_train_arctic_a0014`
- `71`, `drop_edge = 3`
  - active sources: `1 / 4`
  - prompt-control responses: `1`
  - maintained-`delta_g` responses: `0`
  - anti-control responses: `1`
- `71`, `trim_head_runs = 1`, `drop_edge = 3`
  - active sources: `1 / 4`
  - prompt-control responses: `1`
  - maintained-`delta_g` responses: `0`
  - anti-control responses: `1`

The strongest runtime examples also point the same way as the static
directionality audit:

- `63`, `bdl_train_arctic_a0014`
  - prompt slope: `-0.0168`
  - maintained `delta_g` slope: `-0.0257`
  - anti-control slope: `+0.0257`
- `63`, `slt_train_arctic_a0014`
  - prompt slope: `+0.5963`
  - maintained `delta_g` slope: `-0.0494`
  - anti-control slope: `+0.0494`
- `71`, `asi_train_arctic_a0011`
  - prompt slope: `+0.3585`
  - maintained `delta_g` slope: `-0.0429`
  - anti-control slope: `+0.0429`
- `71`, `slt_train_arctic_a0014`
  - prompt slope: `+0.5000`
  - maintained `delta_g` slope: `-0.0930`
  - anti-control slope: `+0.0930`
- `72`, `slt_train_arctic_a0014`
  - prompt slope: `+0.1429`, but prompt monotonicity still fails
  - maintained `delta_g` slope: `-0.0990`
  - anti-control slope: `+0.0990`

Interpretation:

- the residual runtime movement that survives in a few cases is not evidence
  that maintained `delta_g` is weak-but-correct
- it is more consistent with an anti-control reading, especially because the
  static pair-surface audit already showed `delta_g -> c_star` is globally
  anti-aligned
- therefore the right conclusion is still falsification, not “train harder”

Current blocking conclusion for Gate 2:

- on this local data surface, candidate counterfactual tokens still fail as a
  maintained global tempo cue
- the few surviving runtime responses are either flat, tied with prompt-only
  orderings, or better explained by `-delta_g`
- so the repository should continue to treat Gate 2 training on this local
  surface as blocked

Counterfactual full-source runtime sweep:

- `scripts/generate_counterfactual_probe_cases.py`
- `tmp/gate1_counterfactual_probe/full_sweep/token63_cases.json`
- `tmp/gate1_counterfactual_probe/full_sweep/token71_cases.json`
- `tmp/gate1_counterfactual_probe/full_sweep/token72_cases.json`
- `tmp/gate1_counterfactual_probe/full_sweep/token63_summary.json`
- `tmp/gate1_counterfactual_probe/full_sweep/token71_summary.json`
- `tmp/gate1_counterfactual_probe/full_sweep/token72_summary.json`
- `tmp/gate1_counterfactual_probe/full_sweep/runtime_anti_control_report.json`

The earlier runtime anti-control audit was still based on four hand-picked
sources. This sweep removes that objection by generating a candidate-specific
slow/mid/fast/random bundle for **every** train source on the local surface.

Sweep setup:

- split: `train`
- source coverage: `32` sources total
- speaker coverage: `4` speakers × `8` sources each
- candidate tokens: `63 / 71 / 72`
- for each candidate and each source:
  - score all `7` same-speaker different-text refs under the same
    counterfactual prompt build used by the runtime probe
  - choose `slow / mid / fast` from the prompt-tempo ordering
  - choose `random_ref` from another speaker
- all `32 / 32` sources had `7` valid same-speaker refs for all three tokens,
  so this sweep is not relying on relaxed fallback selection

The runtime anti-control aggregator keeps the same active-response criterion:

- `real_tempo_range >= 0.01`
- `max(max_gap_vs_source_only, max_gap_vs_random_ref) >= 0.01`

Then it asks, over those active sources, whether the surviving runtime movement
is explained by:

- prompt order
- maintained `delta_g`
- anti-control `-delta_g`

Full-sweep result:

- `63`
  - active runtime sources: `13 / 32`
  - prompt-control responses: `9`
  - maintained-`delta_g` responses: `1`
  - anti-control responses: `11`
- `71`
  - active runtime sources: `20 / 32`
  - prompt-control responses: `19`
  - maintained-`delta_g` responses: `1`
  - anti-control responses: `19`
- `72`
  - active runtime sources: `14 / 32`
  - prompt-control responses: `5`
  - maintained-`delta_g` responses: `0`
  - anti-control responses: `13`

This is the main new fact:

- across the full `32`-source runtime surface, maintained `delta_g` almost
  never explains the surviving runtime movement
- the counts are `1 / 13`, `1 / 20`, and `0 / 14` on the active-source slices
  for `63 / 71 / 72`
- so the earlier “wrong-direction” diagnosis is not a small-case artifact

Speaker slices show the same pattern:

- `63`
  - active sources by speaker: `aba 6`, `bdl 5`, `slt 2`
  - anti-control responses by speaker: `aba 6`, `bdl 4`, `slt 1`
  - maintained `delta_g` response count: only `1`, on `slt`
- `71`
  - active sources by speaker: `aba 4`, `asi 6`, `bdl 5`, `slt 5`
  - anti-control responses by speaker: `aba 3`, `asi 6`, `bdl 5`, `slt 5`
  - maintained `delta_g` response count: only `1`, on `aba`
- `72`
  - active sources by speaker: `aba 7`, `bdl 1`, `slt 6`
  - anti-control responses by speaker: `aba 6`, `bdl 1`, `slt 6`
  - maintained `delta_g` response count: `0`

The full sweep also clarifies the one case that could otherwise be
misinterpreted, namely `71`:

- on `71`, prompt control and anti-control both cover `19 / 20` active sources
- but in `17 / 19` of those sources, the prompt ordering is identical to the
  anti-control ordering
- so this is not evidence that maintained `delta_g` is working; it means the
  selected prompt ordering often already agrees with the sign-flipped view

Representative full-sweep runtime examples:

- `63`, `bdl_train_arctic_a0014`
  - prompt slope: `-0.0267`
  - maintained `delta_g` slope: negative
  - anti-control slope: positive
- `72`, `slt_train_arctic_a0012`
  - prompt slope: `-2.3499`
  - maintained `delta_g` slope: negative
  - anti-control slope: `+0.3586`
- `72`, `slt_train_arctic_a0016`
  - prompt slope: `-1.0247`
  - maintained `delta_g` slope: negative
  - anti-control slope: `+0.2307`

Strengthened interpretation:

- the earlier four-case runtime audit was not a cherry-picked artifact
- on the full local train surface, maintained `delta_g` is still almost never
  the right runtime control variable
- the surviving runtime movement is much better described as anti-control,
  especially for `63` and `72`
- for `71`, the tie between prompt and anti-control is still consistent with a
  sign-inverted cue, not a valid maintained Gate-1 control signal

Current stop/go status:

- Gate 1 remains failed on the local data surface
- the failure now holds at three levels simultaneously:
  - support/pathology audits
  - static pair-surface directionality
  - full-source runtime anti-control sweep
- Gate 2 training should remain blocked here

Delta-positive case audit:

- `scripts/audit_counterfactual_runtime_delta_positive_cases.py`
- `tmp/gate1_counterfactual_probe/full_sweep/delta_positive_audit_report.json`

After the full-source runtime sweep, only two source-level groups still had
`delta_g_control_response = true`:

- `63`, `slt_train_arctic_a0013`
- `71`, `aba_train_arctic_a0012`

Those are the only remaining candidates that could be misread as
maintained-`delta_g` positives. This audit checks them against their own
row-level `delta_g -> c_star` relationship rather than stopping at the
monotonicity flag.

Result for `63`, `slt_train_arctic_a0013`:

- runtime `delta_g` monotonicity passes, but prompt monotonicity fails
- local `delta_g -> c_star` slope is strongly **negative**: `-0.5566`
- sign-flipped `(-delta_g) -> c_star` slope is equally **positive**: `+0.5566`
- the runtime response has only two distinct output levels
- so the apparent `delta_g` pass is coming from the prompt order breaking and
  then being re-sorted by a sign-inverted cue, not from a correct maintained
  control variable

Row-level view makes that explicit:

- slow:
  - `prompt_tempo_ref = 0.1690`
  - `delta_g = +0.0527`
  - `tempo_out = 0.4082`
  - `c_star = +0.1696`
- mid:
  - `prompt_tempo_ref = 0.3333`
  - `delta_g = +0.2209`
  - `tempo_out = 0.5000`
  - `c_star = +0.0014`
- fast:
  - `prompt_tempo_ref = 0.5000`
  - `delta_g = -0.4055`
  - `tempo_out = 0.4082`
  - `c_star = +0.3500`

So the only reason `delta_g` looks monotone here is that the fast reference
lands on the negative side of `delta_g`, while the output itself only toggles
between two levels.

Result for `71`, `aba_train_arctic_a0012`:

- runtime `delta_g` monotonicity passes, but prompt monotonicity fails
- local `delta_g -> c_star` slope is again **negative**: `-0.1300`
- sign-flipped `(-delta_g) -> c_star` slope is **positive**: `+0.1300`
- the runtime range is tiny: `0.0159`
- one of the real refs is singleton-support
- the same-ref static row is already sign-mismatched

Row-level view:

- slow:
  - `prompt_tempo_ref = 0.2000`
  - `delta_g = -0.0912`
  - `tempo_out = 0.1826`
  - `c_star = -0.0873`
- mid:
  - `prompt_tempo_ref = 0.2500`
  - `delta_g = -1.2425`
  - `tempo_out = 0.1667`
  - `c_star = +0.0623`
- fast:
  - `prompt_tempo_ref = 0.3333`
  - `delta_g = -1.1309`
  - `tempo_out = 0.1667`
  - `c_star = +0.0623`

Again, this is not a stable maintained-control success:

- the output has only two distinct levels
- the slope is being driven by one small slow-point offset against two nearly
  identical mid/fast outputs
- and the local oracle sign still prefers `-delta_g`

Targeted stricter reruns on the two delta-positive cases:

- `tmp/gate1_counterfactual_probe/delta_positive_cases/token63_drop3_summary.json`
- `tmp/gate1_counterfactual_probe/delta_positive_cases/token63_trim_h1_drop3_summary.json`
- `tmp/gate1_counterfactual_probe/delta_positive_cases/token71_drop3_summary.json`
- `tmp/gate1_counterfactual_probe/delta_positive_cases/token71_trim_h1_drop3_summary.json`
- `tmp/gate1_counterfactual_probe/delta_positive_cases/runtime_anti_control_report.json`
- `tmp/gate1_counterfactual_probe/delta_positive_cases/delta_positive_audit_report.json`

These reruns ask whether the two residual delta-positive cases survive stronger
edge-depth / boundary-trim perturbations.

`63`, `slt_train_arctic_a0013`:

- at `drop_edge = 3`, `delta_g` no longer passes
- the residual response flips to anti-control:
  - prompt responses: `0`
  - maintained `delta_g` responses: `0`
  - anti-control responses: `1`
- at `trim_head_runs = 1, drop_edge = 3`, `delta_g` passes again
  but the local `delta_g -> c_star` slope becomes even more negative:
  `-0.7835`, with flipped slope `+0.7835`

So for `63`, the only remaining delta-positive configuration is not robust:

- one stricter rerun removes it entirely
- the other keeps the runtime monotonicity flag but strengthens the
  sign-inverted local oracle reading

`71`, `aba_train_arctic_a0012`:

- at `drop_edge = 3`, `delta_g` still passes
  but local `delta_g -> c_star` remains negative: `-0.1815`
- at `trim_head_runs = 1, drop_edge = 3`, `delta_g` no longer passes
  and the residual response is covered by prompt + anti-control instead

So for `71`, the apparent positive is also unstable:

- one stricter rerun keeps the runtime flag but not the oracle sign
- the other stricter rerun removes the runtime delta-positive flag entirely

Strengthened conclusion after the positive-case audit:

- the full-source sweep's two remaining `delta_g` positives do not survive
  source-level oracle reconciliation as genuine maintained-control evidence
- both are better explained as ordering artifacts over tiny / low-cardinality
  runtime responses
- both are unstable under stricter edge-depth / trim perturbations
- therefore there is no credible maintained-`delta_g` survivor left on the
  local runtime surface

Updated stop/go status:

- Gate 1 remains failed
- the local surface now has:
  - broad static anti-alignment
  - broad runtime anti-control dominance
  - no surviving robust maintained-`delta_g` positive after source-level audit
- Gate 2 should remain blocked on this local surface

Active-source local oracle audit:

- `scripts/audit_counterfactual_runtime_active_local_oracle.py`
- `tmp/gate1_counterfactual_probe/full_sweep/active_local_oracle_report.json`

The previous steps established three things:

- the static pair surface is globally anti-aligned
- the full runtime sweep is mostly explained by anti-control rather than
  maintained `delta_g`
- the two apparent runtime `delta_g` positives do not survive source-level
  oracle scrutiny

This audit removes the last remaining loophole by asking a stronger question:

- if we restrict to the **runtime-active** sources only, does any active source
  still have a locally positive `delta_g -> c_star` relationship?

Here “runtime-active” uses the same criterion as the full-sweep runtime audit:

- `real_tempo_range >= 0.01`
- `max(max_gap_vs_source_only, max_gap_vs_random_ref) >= 0.01`

Then for each active source, using its own `slow / mid / fast` rows from the
runtime sweep, the audit computes:

- local `delta_g -> c_star` robust slope
- local `(-delta_g) -> c_star` robust slope
- local sign-agreement rate between `delta_g` and `c_star`

Result:

- `63`
  - active sources: `13`
  - local negative-slope sources: `13 / 13`
  - local positive-slope sources: `0 / 13`
  - median local slope: `-0.4481`
  - median flipped slope: `+0.4481`
- `71`
  - active sources: `20`
  - local negative-slope sources: `20 / 20`
  - local positive-slope sources: `0 / 20`
  - median local slope: `-0.2956`
  - median flipped slope: `+0.2956`
- `72`
  - active sources: `14`
  - local negative-slope sources: `14 / 14`
  - local positive-slope sources: `0 / 14`
  - median local slope: `-0.3166`
  - median flipped slope: `+0.3166`

This is the strongest runtime-side statement so far:

- among the sources where the model actually moves at all, **none** support a
  locally positive maintained `delta_g -> c_star` relationship
- every active source still prefers the sign-flipped interpretation

Even the response buckets that might look superficially favorable remain
negative when checked against the local oracle:

- `63`
  - `prompt_anti`: `8 / 8` negative local slopes
  - `anti_only`: `3 / 3` negative local slopes
  - `prompt_only`: `1 / 1` negative local slope
  - `delta_only`: `1 / 1` negative local slope
- `71`
  - `prompt_anti`: `19 / 19` negative local slopes
  - `delta_only`: `1 / 1` negative local slope
- `72`
  - `prompt_anti`: `5 / 5` negative local slopes
  - `anti_only`: `8 / 8` negative local slopes
  - `none`: `1 / 1` negative local slope

So even the runtime-active cases that still satisfy prompt control do **not**
validate maintained `delta_g`; they are still locally anti-aligned with the
oracle coarse target.

Strengthened blocking conclusion:

- the local surface no longer has any remaining runtime-active source that
  supports maintained `delta_g`
- the failure is not just “most cases fail”; it is now
  “all runtime-active cases still point the wrong way locally”
- Gate 2 remains blocked on this local surface

Prompt-vs-delta local oracle split:

- `scripts/audit_counterfactual_runtime_active_local_oracle.py`
- `tmp/gate1_counterfactual_probe/full_sweep/active_local_oracle_report.json`

The active-source oracle audit above already showed that all runtime-active
sources have locally negative `delta_g -> c_star` slopes. The next question is
whether that failure comes from bad reference selection, or from the later
`g / delta_g` extraction step.

Using the same active-source report, we can compare three local mappings for
each runtime-active source:

- `prompt_tempo_ref -> c_star`
- `delta_g -> c_star`
- `(-delta_g) -> c_star`

Result:

- `63`
  - active sources: `13`
  - prompt-positive local slopes: `13 / 13`
  - delta-positive local slopes: `0 / 13`
  - prompt-positive + delta-negative sources: `13 / 13`
  - prompt exact-order matches to `c_star`: `7 / 13`
  - maintained `delta_g` exact-order matches to `c_star`: `0 / 13`
  - sign-flipped exact-order matches to `c_star`: `11 / 13`
- `71`
  - active sources: `20`
  - prompt-positive local slopes: `20 / 20`
  - delta-positive local slopes: `0 / 20`
  - prompt-positive + delta-negative sources: `20 / 20`
  - prompt exact-order matches to `c_star`: `20 / 20`
  - maintained `delta_g` exact-order matches to `c_star`: `0 / 20`
  - sign-flipped exact-order matches to `c_star`: `17 / 20`
- `72`
  - active sources: `14`
  - prompt-positive local slopes: `6 / 14`
  - prompt-negative local slopes: `6 / 14`
  - delta-positive local slopes: `0 / 14`
  - prompt-positive + delta-negative sources: `6 / 14`
  - prompt exact-order matches to `c_star`: `6 / 14`
  - maintained `delta_g` exact-order matches to `c_star`: `0 / 14`
  - sign-flipped exact-order matches to `c_star`: `10 / 14`

This cleanly separates the failure modes:

- `63` and `71`
  - the prompt-side slow/mid/fast ordering is locally consistent with the
    oracle coarse target
  - but the extracted maintained `delta_g` still points the wrong way on
    every active source
  - so the problem is not “we picked the wrong reference ordering”; it is that
    the candidate cue becomes sign-inverted when converted into `g / delta_g`
- `72`
  - prompt ordering is itself unstable on the active slice
  - and maintained `delta_g` is still worse, never matching the oracle order
  - so `72` fails even earlier and more obviously

This is a stronger falsification than the earlier runtime anti-control counts:

- for `63` and `71`, prompt-side tempo ordering can look locally plausible
- yet the maintained `delta_g` mapping still fails on **every** active source
- therefore the surviving runtime motion cannot be defended as “reference
  selection is good, so the cue must be basically right”

Current interpretation after this split:

- `63 / 71`: ref ordering is often locally sensible, but the maintained global
  cue flips direction when summarized into `delta_g`
- `72`: both ref ordering and maintained `delta_g` are bad, with `delta_g`
  still strictly worse
- none of the three candidates support the maintained Gate-1 claim

Prompt-to-g inversion audit:

- `scripts/audit_counterfactual_runtime_active_local_oracle.py`
- `tmp/gate1_counterfactual_probe/full_sweep/active_local_oracle_report.json`

The prompt-vs-delta split above already showed that prompt-side ordering can be
locally sensible while `delta_g -> c_star` still points the wrong way. The next
question is whether that inversion happens only at the final `delta_g` step, or
whether it already appears when prompt tempo is mapped into `g_ref`.

Using the same runtime-active source slice, the audit now also measures:

- `prompt_tempo_ref -> delta_g`
- `prompt_tempo_ref -> g_ref`
- exact-order agreement between prompt ordering and `delta_g / g_ref` ordering

Result:

- `63`
  - active sources: `13`
  - prompt-positive local `prompt_tempo_ref -> c_star`: `13 / 13`
  - negative `prompt_tempo_ref -> delta_g`: `13 / 13`
  - negative `prompt_tempo_ref -> g_ref`: `13 / 13`
  - prompt-order equals `delta_g` order: `0 / 13`
  - prompt-order equals `g_ref` order: `0 / 13`
  - median `prompt_tempo_ref -> delta_g` slope: `-4.1576`
- `71`
  - active sources: `20`
  - prompt-positive local `prompt_tempo_ref -> c_star`: `20 / 20`
  - negative `prompt_tempo_ref -> delta_g`: `20 / 20`
  - negative `prompt_tempo_ref -> g_ref`: `20 / 20`
  - prompt-order equals `delta_g` order: `0 / 20`
  - prompt-order equals `g_ref` order: `0 / 20`
  - median `prompt_tempo_ref -> delta_g` slope: `-7.7979`
- `72`
  - active sources: `14`
  - negative `prompt_tempo_ref -> delta_g`: `9 / 14`
  - negative `prompt_tempo_ref -> g_ref`: `9 / 14`
  - prompt-order equals `delta_g` order: `1 / 14`
  - prompt-order equals `g_ref` order: `1 / 14`
  - median `prompt_tempo_ref -> delta_g` slope: `-2.7726`

This pins the inversion point down much more sharply:

- for `63` and `71`, prompt-side tempo is locally aligned with the oracle
  coarse target on every active source
- but the same prompt ordering is locally anti-aligned with both `g_ref` and
  `delta_g` on every active source
- so the failure is not “prompts were chosen badly”
- it is that the candidate cue flips sign when converted from prompt tempo into
  the maintained global-rate statistic

For `72`, the prompt side is already unstable, but the maintained
`g_ref / delta_g` mapping is still worse:

- only `1 / 14` active sources keeps the same prompt-vs-`delta_g` order
- the median prompt-to-`delta_g` slope is still strongly negative

Strongest current reading:

- `63 / 71` fail specifically at the `prompt_tempo_ref -> g_ref -> delta_g`
  summarization stage
- `72` fails both at prompt ordering and at the maintained global-rate stage
- there is no remaining path on this local surface by which the maintained
  `g` can be defended as the right control quantity

Anti-control chain coherence audit:

- `scripts/audit_counterfactual_anti_control_chain.py`
- `tmp/gate1_counterfactual_probe/full_sweep/anti_control_chain_report.json`

The prompt-to-g inversion audit above says where the direction flips:

- prompt-side tempo can be locally sensible
- but the maintained `g_ref / delta_g` mapping reverses that order

The next question is whether this reversal is just a collection of local
failures, or whether it forms a coherent alternative chain:

- `prompt_tempo_ref -> c_star` positive
- `prompt_tempo_ref -> delta_g` negative
- `(-delta_g) -> c_star` positive

That is, do the candidates instantiate a **stable anti-control chain** rather
than a usable maintained control chain?

We summarize two versions:

- weak anti-chain:
  `prompt_tempo_ref -> delta_g` negative and `(-delta_g) -> c_star` positive
- coherent anti-chain:
  `prompt_tempo_ref -> c_star` positive,
  `prompt_tempo_ref -> delta_g` negative,
  and `(-delta_g) -> c_star` positive
- strict coherent anti-chain:
  the above plus exact-order agreement for both
  `prompt_tempo_ref -> c_star` and `(-delta_g) -> c_star`

For comparison, the maintained chain would require:

- `prompt_tempo_ref -> c_star` positive
- `prompt_tempo_ref -> delta_g` positive
- `delta_g -> c_star` positive

Result on the runtime-active source slice:

- `63`
  - active sources: `13`
  - weak anti-chain: `13 / 13`
  - coherent anti-chain: `13 / 13`
  - strict coherent anti-chain: `5 / 13`
  - maintained chain: `0 / 13`
  - strict maintained chain: `0 / 13`
- `71`
  - active sources: `20`
  - weak anti-chain: `20 / 20`
  - coherent anti-chain: `20 / 20`
  - strict coherent anti-chain: `17 / 20`
  - maintained chain: `0 / 20`
  - strict maintained chain: `0 / 20`
- `72`
  - active sources: `14`
  - weak anti-chain: `9 / 14`
  - coherent anti-chain: `6 / 14`
  - strict coherent anti-chain: `3 / 14`
  - maintained chain: `0 / 14`
  - strict maintained chain: `0 / 14`

This is a stronger statement than “the sign is wrong”:

- `63` and `71` do not merely fail the maintained Gate-1 chain
- on the entire runtime-active slice, they instantiate a coherent
  sign-inverted chain instead
- `72` is weaker and noisier, but it still has zero maintained-control support

Practical interpretation:

- `63 / 71` are not weak versions of the maintained `g`
- they are better described as stable anti-control statistics on this local
  surface
- so the local falsification is now complete enough to say the issue is
  structural, not just low-signal or underpowered

Cross-split runtime reproduction:

- `tmp/gate1_counterfactual_probe/valid_sweep/token63_summary.json`
- `tmp/gate1_counterfactual_probe/valid_sweep/token71_summary.json`
- `tmp/gate1_counterfactual_probe/valid_sweep/token72_summary.json`
- `tmp/gate1_counterfactual_probe/valid_sweep/runtime_anti_control_report.json`
- `tmp/gate1_counterfactual_probe/valid_sweep/active_local_oracle_report.json`
- `tmp/gate1_counterfactual_probe/valid_sweep/anti_control_chain_report.json`
- `tmp/gate1_counterfactual_probe/valid_sweep/delta_positive_audit_report.json`
- `tmp/gate1_counterfactual_probe/test_sweep/token63_summary.json`
- `tmp/gate1_counterfactual_probe/test_sweep/token71_summary.json`
- `tmp/gate1_counterfactual_probe/test_sweep/token72_summary.json`
- `tmp/gate1_counterfactual_probe/test_sweep/runtime_anti_control_report.json`
- `tmp/gate1_counterfactual_probe/test_sweep/active_local_oracle_report.json`
- `tmp/gate1_counterfactual_probe/test_sweep/anti_control_chain_report.json`
- `tmp/gate1_counterfactual_probe/test_sweep/delta_positive_audit_report.json`

To check whether the train-split result was a small-surface artifact, the same
zero-training runtime sweep and downstream audits were repeated on the local
`valid` and `test` surfaces.

Coverage:

- `valid`: `16` sources total, each with exactly `3` valid same-speaker refs
- `test`: `16` sources total, each with exactly `3` valid same-speaker refs
- candidates: `63 / 71 / 72`

`valid` runtime summary:

- `63`
  - active runtime sources: `10 / 16`
  - prompt-control responses: `7`
  - maintained-`delta_g` responses: `1`
  - anti-control responses: `8`
- `71`
  - active runtime sources: `3 / 16`
  - prompt-control responses: `2`
  - maintained-`delta_g` responses: `0`
  - anti-control responses: `3`
- `72`
  - active runtime sources: `1 / 16`
  - prompt-control responses: `0`
  - maintained-`delta_g` responses: `1`
  - anti-control responses: `0`

`test` runtime summary:

- `63`
  - active runtime sources: `8 / 16`
  - prompt-control responses: `7`
  - maintained-`delta_g` responses: `0`
  - anti-control responses: `7`
- `71`
  - active runtime sources: `3 / 16`
  - prompt-control responses: `1`
  - maintained-`delta_g` responses: `0`
  - anti-control responses: `3`
- `72`
  - active runtime sources: `3 / 16`
  - prompt-control responses: `1`
  - maintained-`delta_g` responses: `0`
  - anti-control responses: `3`

So the train result reproduces:

- `63` and `71` continue to favor anti-control over maintained `delta_g`
  on both held-out splits
- `72` remains weak / noisy, but it still does not produce stable
  maintained-control evidence

`valid` / `test` active-local-oracle audit:

- `valid`
  - `63`: active `10`, local negative `delta_g -> c_star` slopes `10 / 10`
  - `71`: active `3`, local negative `delta_g -> c_star` slopes `3 / 3`
  - `72`: active `1`, local negative slopes `0 / 1`, positive slopes `0 / 1`
    because the only active row is effectively zero-slope / degenerate
- `test`
  - `63`: active `8`, local negative `delta_g -> c_star` slopes `7 / 8`
  - `71`: active `3`, local negative `delta_g -> c_star` slopes `3 / 3`
  - `72`: active `3`, local negative `delta_g -> c_star` slopes `3 / 3`

The held-out `delta_g` positives also do not survive source-level audit:

- `valid`, `63`, `bdl_valid_arctic_a0003`
  - runtime `delta_g` pass exists
  - but local `delta_g -> c_star` slope is `-0.4479`
  - flipped slope is `+0.4479`
  - static same-ref row is sign-mismatched
- `valid`, `72`, `aba_valid_arctic_a0004`
  - runtime `delta_g` pass exists
  - local slope is effectively `0`
  - the case contains singleton support and only two runtime output levels
- `test`
  - `delta_g` positives: `0`

So the cross-split evidence strengthens the earlier conclusion:

- the train-surface inversion is not a one-off artifact
- on held-out local surfaces, maintained `delta_g` still fails to provide
  stable positive local oracle alignment
- the few apparent positives either vanish or reduce to degenerate / noisy
  cases under source-level audit

Cross-split anti-control chain summary:

- `valid`
  - `63`: coherent anti-chain `9 / 10`, strict `6 / 10`
  - `71`: coherent anti-chain `2 / 3`, strict `2 / 3`
  - maintained chain: `0` for all three tokens
- `test`
  - `63`: coherent anti-chain `5 / 8`, strict `4 / 8`
  - `71`: coherent anti-chain `1 / 3`, strict `1 / 3`
  - `72`: coherent anti-chain `1 / 3`, strict `1 / 3`
  - maintained chain: `0` for all three tokens

Updated global reading across `train / valid / test`:

- `63` and `71` consistently behave like sign-inverted cues rather than weak
  versions of the maintained `g`
- `72` stays unstable, but still never produces a convincing maintained chain
- there is no split on the local CMU/L2 surface where the maintained Gate-1
  story becomes the dominant explanation

## 8. Result table template

### Gate 0 summary

| date | split | g_variant | alignment_kind | target_duration_surface | rho(delta_g,c*) | robust_slope | r2_like | notes |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| pending | pending | raw_median | pending | pending | pending | pending | pending | no experiment run yet |

### Gate 1 summary

| date | split | eval_mode | monotonicity_rate | negative_control_gap | same_text_gap | notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| pending | pending | analytic | pending | pending | pending | no experiment run yet |

### Gate 2 summary

| date | split | eval_mode | monotonicity_rate | tempo_transfer_slope | silence_leakage | prefix_discrepancy | budget_hit_rate | projector_boundary_hit_rate | projector_boundary_decay_rate | clamp_mass | rounding_regret | notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| pending | pending | coarse_only | pending | pending | pending | pending | pending | pending | pending | pending | pending | no experiment run yet |
| pending | pending | learned_detach | pending | pending | pending | pending | pending | pending | pending | pending | pending | no experiment run yet |
| pending | pending | learned_no_detach | pending | pending | pending | pending | pending | pending | pending | pending | pending | ablation only |
