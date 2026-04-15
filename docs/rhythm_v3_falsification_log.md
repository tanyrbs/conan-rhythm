# rhythm_v3 falsification log

Historical note:

- this log records the exact-family upper-bound falsification pass that proved
  the weighted global cue was alive
- it should now be read together with the newer maintained online mainline
  `weighted_median + ema + first_speech`, not as the online default itself

## 2026-04-14 final Gate0 contract-repair rerun

### Scope

- maintained `rhythm_v3` minimal-V1 only
- local quick-ARCTIC only
- zero-train Gate0 / Gate1 rerun after the deeper Gate0 contract review
- no new gradient training run in this pass

### Hidden contract bugs fixed in this pass

This pass found three additional bugs that materially affected Gate0:

1. Gate0 static audit was not fully isomorphic on `x / y / valid`.

- `delta_g` was source-side and support-filtered
- the y-side speech summaries were not consistently restricted to the same
  source-support surface
- validity also did not require source-side support/domain validity

2. Training-side target construction was not carrying the intended `g` and
   prefix contract.

- target builder config could silently fall back to defaults such as
  `raw_median`, `ema`, `drop_edge_runs=0`, and no boundary threshold

3. Offline source prefix reconstruction had init drift.

- review/audit paths could reconstruct `source_rate_seq` with implicit zero
  init instead of the runtime init contract such as
  `src_rate_init_mode=first_speech`

These are now fixed.

### What changed in the reading

Two older readings are no longer current after this repair pass:

1. "Gate0 still fails cleanly on the local clean slice"

This is no longer true on the repaired audit surface.

For the clean exact slice
`reference_mode=target_as_ref` and `src_prefix_stat_mode=exact_global_family`,
the valid total slope is now positive on:

- `weighted_median`: `8/8` cells
- `raw_median`: `7/8` cells
- `trimmed_mean`: `7/8` cells

Strongest weighted cell:

- candidate token: `63`
- `drop_edge_runs_for_g=1`
- `g_domain_valid_items=28`
- `source_g_domain_valid_items=28`
- `valid_total_signal_robust_slope=0.4796`
- `valid_total_mean_signal_robust_slope=0.0282`
- `valid_total_logratio_signal_robust_slope=0.1723`
- `valid_analytic_signal_robust_slope=0.1452`
- `valid_analytic_runtime_mean_signal_robust_slope=0.0000`
- `valid_residual_signal_robust_slope=0.2022`
- `valid_residual_runtime_mean_signal_robust_slope=0.0129`
- `valid_residual_runtime_affine_signal_robust_slope=0.0350`
- `mean_analytic_saturation_rate=0.7757`

So the old "Gate0 is inherently flat" reading was substantially a contract
bug, not a clean falsification result.

2. "Gate1 might have regressed once Gate0 was repaired"

This is also not true locally.

For `weighted_median + exact_global_family`, Gate1 still passes:

- preclip monotonicity: `4/4`
- continuous monotonicity: `4/4`
- projected monotonicity: `4/4`

Per-source projected slopes remain strictly ordered:

- `aba_train_arctic_a0022`: `-0.0398`
- `asi_train_arctic_a0027`: `-0.1223`
- `bdl_train_arctic_a0022`: `-0.0486`
- `slt_train_arctic_a0020`: `-0.0866`

### Current conclusion after the final rerun

The strongest current local reading is now:

- Gate0 and Gate1 both survive on the repaired weighted local surface
- the earlier Gate0-fail conclusion was contaminated by real source-side
  audit/runtime drift
- the training target path also had real contract drift and is now fixed
- runtime saturation still exists, so this is not a claim that every variant
  or every broader V1 surface is solved
- `exact_global_family` remains a local/offline gate contract at this stage;
  strict online continuation still only carries a scalar prefix state and does
  not preserve full-history robust-prefix support

The maintained local claim should therefore now be phrased as:

- on the repaired local surface, the strongest fixed contract is
  `weighted_median + exact_global_family + target_as_ref`
- that contract now survives Gate0 and Gate1 zero-train falsification
- Gate2 and Gate3 remain pending because this pass did not rerun training
- the checked-in machine-readable summary for this result is now
  `egs/overrides/rhythm_v3_gate_status_local_candidate_20260414.json`

## 2026-04-14 deep follow-up rerun

This section is retained as history only. It is superseded by the final
contract-repair rerun above.

### Historical reading

- this pass correctly found that Gate1 ordering drift was real
- this pass also correctly introduced cleaner runtime-aligned diagnostics
- but its remaining "Gate0 still fails" conclusion was later weakened by the
  source-side support/validity mismatch and init-drift bugs fixed above

## Retained script surface

The maintained zero-train diagnostics remain:

- `scripts/preflight_rhythm_v3.py`
- `scripts/audit_rhythm_v3_boundary_support.py`
- `scripts/audit_rhythm_v3_counterfactual_static_gate0.py`
- `scripts/probe_rhythm_v3_gate1_analytic.py`
- `scripts/probe_rhythm_v3_gate1_silent_counterfactual.py`
- `scripts/rhythm_v3_probe_cases.py`
- `scripts/rhythm_v3_debug_records.py`

## 2026-04-15 Gate2-online and Gate3 local-candidate follow-up

### Scope

- maintained `rhythm_v3` minimal-V1 path only
- local quick-ARCTIC only
- local online candidate reruns, not official gate promotion
- local quick dataset I/O migrated to `D:/conan_data/...` for faster local
  reads

### What was rerun

- Gate2-online local coarse-only candidate:
  `checkpoints/rhythm_v3_gate2_candidate_20260415_s76_srcgap/`
- reviewed Gate2 status:
  `tmp/gate2_candidate_20260415_s75_srcgap/review/gate_status.json`
- Gate3 local candidate:
  `checkpoints/rhythm_v3_gate3_candidate_20260415_s126_srcgapfix1/`

### New blocker that was found and fixed

The first Gate3 local candidate did not fail because of model quality. It
failed before training because config/runtime semantics were inconsistent:

- `strict_minimal_claim_profile=false` was supposed to allow local candidate
  relaxation
- but task/runtime validation still forced
  `rhythm_v3_disable_learned_gate=true`
- and runtime wiring could incorrectly reinterpret
  `rhythm_v3_disable_learned_gate=false` as
  `use_learned_residual_gate=true` under minimal-V1

That wiring bug is now fixed. The correct local-candidate reading is:

- local Gate3 candidate work may use
  `rhythm_v3_disable_learned_gate=false` under
  `strict_minimal_claim_profile=false`
- minimal-V1 still should not promote that into
  `use_learned_residual_gate=true`

This was a local training unblock only, not a Gate3 pass.

### Local Gate2-online result

The local `src_gap` coarse-only candidate did not materially improve the gate
picture.

Reviewed status:

- `gate1_pass=false`
- `gate2_pass=false`
- `gate3_pass=false`
- `analytic_tempo_monotonicity_rate=0.3529`
- `analytic_tempo_transfer_slope=-0.0149`
- `analytic_tempo_tie_rate=0.6471`
- `coarse_only_runtime_metrics.cumulative_drift=8.9612`

Interpretation:

- simply exposing `src_gap` to the coarse head did not unlock the stalled
  online surface
- the main failure signature still looks like execution flattening / drift
  rather than prompt-`g` absence

### Item-level reading

The bottleneck is mixed, but projector/discrete execution remains a primary
collapse point:

- some items are already flat before projection
  - `bdl_train_arctic_a0017`: raw/preproj/exec all flat
  - `slt_train_arctic_a0020`: raw/preproj/exec all flat
- some items clearly lose signal at execution
  - `asi_train_arctic_a0023`: preproj monotone, exec flat
- some items still survive end-to-end with reduced range
  - `asi_test_arctic_a0011`: ordered through exec, but compressed

So the stronger current reading is:

- writer-side failures still exist on some items
- but there is also direct evidence that projector/discrete execution is
  eating already-limited preproj signal on other items

### Current conclusion

- Gate2-online local candidate remains failing
- `src_gap` is not the dominant fix by itself
- the next debugging priority is projector/clipping/budget/headroom and exec
  bucketization
- Gate3 local training is unblocked and advanced to a checkpointed run, but no
  Gate3 pass should be claimed from this log
