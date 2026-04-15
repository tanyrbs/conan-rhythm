# rhythm_v3 local status snapshot (2026-04-15)

This is the current local status snapshot after the local Gate2-online rerun,
the Gate3 wiring fix, and the local quick-config dataset migration to
`D:/conan_data/...`.

## Snapshot

- scope: maintained `rhythm_v3` minimal-V1, local quick-ARCTIC only
- runtime default under test: `weighted_median + ema + first_speech`
- strongest upper-bound local validation still retained separately as
  `weighted_median + exact_global_family`
- official training status: blocked
- local online candidate status: advanced, but still failing review gates

## What changed on 2026-04-15

1. Local quick-config dataset I/O was moved from `C:/project/...` to
   `D:/conan_data/...`.

- changed file:
  `egs/local_arctic_rhythm_v3_quick.yaml`
- copied and hash-checked:
  - `C:/project/00-2 project/data/processed/vc`
    -> `D:/conan_data/processed/vc`
  - `C:/project/00-2 project/data/binary/vc_6layer`
    -> `D:/conan_data/binary/vc_6layer`
- this is a local quick-config operational change only, not a repo-wide
  contract change

2. A local Gate2-online coarse-only candidate was rerun with
   `rhythm_v3_use_src_gap_in_coarse_head=true` and
   `rhythm_v3_strict_minimal_claim_profile=false`.

- checkpoint family:
  `checkpoints/rhythm_v3_gate2_candidate_20260415_s76_srcgap/`
- reviewed status:
  `tmp/gate2_candidate_20260415_s75_srcgap/review/gate_status.json`

3. Gate3 local training was unblocked by fixing a config/runtime mismatch.

- the local candidate now allows
  `rhythm_v3_disable_learned_gate=false` under
  `strict_minimal_claim_profile=false`
- minimal-V1 runtime still does **not** reinterpret that as
  `use_learned_residual_gate=true`
- this is a local candidate wiring fix, not a Gate3 pass

4. A projector/headroom falsification pass was run after the `src_gap`
   candidate review.

- boundary no-decay was falsified:
  `rhythm_v3_boundary_carry_decay=1.0` and
  `rhythm_v3_boundary_offset_decay=1.0` kept raw/preproj unchanged but worsened
  execution collapse and drift
- optional projector diagnostics were added for local experiments:
  `rhythm_v3_integer_projection_mode=prefix_optimal` and
  `rhythm_v3_integer_projection_anchor_mode=continuous`
- the review CLI was fixed so `ref_bin=slow|mid|fast` is promoted to
  `ref_condition`/`triplet_id` when loading debug-record bundles, and tempo
  transfer uses `tempo_ref_runtime` or sign-correct source-gap instead of raw
  `delta_g`
- the strongest no-retrain execution-headroom probe so far is:
  `rhythm_v3_analytic_gap_clip=0.80`,
  `rhythm_v3_prefix_budget_pos=72`,
  `rhythm_v3_prefix_budget_neg=72`,
  `rhythm_v3_min_prefix_budget=24`,
  `rhythm_v3_max_prefix_budget=96`,
  `rhythm_v3_dynamic_budget_ratio=0.35`

## Current gate reading

### Gate0 / Gate1 upper-bound

Still alive on the strongest local validation surface:

- `weighted_median + exact_global_family`
- read this as upper-bound/local evidence only
- do **not** collapse it into the maintained online runtime contract

### Gate2-online local candidate

Current reviewed result:

- `gate1_pass=false`
- `gate2_pass=false`
- `gate3_pass=false`
- `analytic_tempo_monotonicity_rate=0.3529`
- `analytic_tempo_transfer_slope=-0.0149`
- `analytic_tempo_tie_rate=0.6471`
- `coarse_only_runtime_metrics.cumulative_drift=8.9612`
- `coarse_only_control_regressions=["monotonicity_rate","speech_weighted_mae"]`

Projector/headroom follow-up:

- `boundary_carry_decay=1.0` / `boundary_offset_decay=1.0` is not the fix:
  train exec range dropped and `budget_hit_any_rate` rose sharply
- `prefix_optimal` projection and continuous-anchor debt tracking are now
  available for controlled local trials, but on the current quick data they do
  not materially change the Gate1 analytic result by themselves
- wider budget plus `analytic_gap_clip=0.80` improves train transfer
  (`monotone_source_count=7/12`, mean exec slope about `1.02`) but still fails
  the gate because ties and non-monotone triplets remain
- valid/test remain insufficient:
  valid `1/3` exec pass, test `1/2` exec pass on the same no-retrain probe
- the fixed review status for that best no-retrain probe still reads
  `gate1_pass=false`, `gate2_pass=false`, `gate3_pass=false`,
  `analytic_tempo_monotonicity_rate=0.50`,
  `analytic_tempo_tie_rate=0.417`

Interpretation:

- `src_gap` alone did not materially improve the maintained online surface
- the dominant bottleneck still looks execution-side rather than prompt-side
  `g` existence

### Gate3 local candidate

Current local artifact state:

- work dir:
  `checkpoints/rhythm_v3_gate3_candidate_20260415_s126_srcgapfix1/`
- latest checkpoint:
  `model_ckpt_steps_125.ckpt`
- config target:
  `max_updates: 126`
- TensorBoard scalar evidence reaches `step 125`
- no `step_126` checkpoint or terminal log artifact is present

Factual reading:

- Gate3 definitely advanced through `step_125`
- from local files only, we cannot prove that an observable `step_126+`
  artifact completed
- no Gate3 pass should be claimed

## Current blocker

The current blocker is not "prompt domain collapses before `g` exists".

It is closer to:

- some sources are already weak or flat at raw/preproj, so writer-side failure
  still exists on part of the surface
- some sources show real preproj ordering that gets flattened or bucketized at
  exec, so projector/discrete execution remains a primary bottleneck
- aggregate online metrics still show too many ties and too much cumulative
  drift

Concrete examples from the current local candidate review:

- `asi_train_arctic_a0023`: preproj monotone, exec flat
- `bdl_train_arctic_a0017`: raw/preproj/exec all flat
- `slt_train_arctic_a0020`: raw/preproj/exec all flat
- `asi_test_arctic_a0011`: signal survives through exec but with compressed
  range

So the next debugging priority is:

- projector/clipping/budget/headroom
- exec bucketization / tie formation
- source-side state upgrades such as dual EMA before any non-streaming exact
  family promotion
- not simply increasing model freedom again

## Official vs local

Keep these truths separate:

- official gate status:
  `egs/overrides/rhythm_v3_gate_status.json`
  still blocked
- local upper-bound validation:
  Gate0/Gate1 alive on
  `weighted_median + exact_global_family`
- local online candidates:
  Gate2 rerun and Gate3-progress evidence exist, but they are diagnostic only

Therefore:

- do not claim Gate2 passed officially
- do not claim Gate3 passed
- do not describe the D-drive migration as a repo-wide default
- do not treat upper-bound exact-family evidence as the maintained online
  contract
