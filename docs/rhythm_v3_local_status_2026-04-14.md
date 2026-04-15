# rhythm_v3 local status snapshot (2026-04-14)

Superseded by:

- `docs/rhythm_v3_local_status_2026-04-15.md`

This is the latest local zero-train falsification snapshot after the final
Gate0 contract-repair pass.

Historical note:

- this snapshot records the strongest local `Gate1-upper` exact-family result
  available on `2026-04-14`
- the maintained online mainline has since moved to
  `weighted_median + ema + first_speech`
- do not read this historical snapshot as the current online-unblock contract

## 1. Scope

- maintained `rhythm_v3` minimal-V1 path only
- local quick-ARCTIC configs only
- zero-train Gate0 / Gate1 reruns only in this pass
- this snapshot was produced on a CPU-only `.venv` in this workspace:
  the host machine has `GTX 1050 Ti`, but no CUDA-enabled PyTorch runtime was
  available during the rerun that produced these numbers
- frozen family used in the latest reruns:
  - `raw_median`
  - `weighted_median`
  - `trimmed_mean`

## 2. What changed in the final pass

The main finding of this round is that Gate0 still had hidden contract drift.
The earlier local reading that "Gate0 is inherently flat" is no longer
defensible after the following fixes:

- Gate0 static audit now enforces source-side support validity instead of only
  prompt-side validity
- Gate0 y-side statistics are now computed on the same source-support surface
  used to define `delta_g`
- offline `source_rate_seq` reconstruction now uses the exported runtime
  init contract instead of silently drifting to zero-init reconstruction
- training target construction now carries the intended `g` and prefix fields:
  `g_variant`, `g_trim_ratio`, `src_prefix_stat_mode`,
  `src_prefix_min_support`, `g_drop_edge_runs`,
  `min_boundary_confidence_for_g`
- analysis-side crop tables now reuse the full prompt/source `g` contract,
  including prompt weight and source boundary/support sidecars

This means the latest Gate0 rerun is materially cleaner than the earlier
`followup` snapshot.

## 3. Current artifacts

Checked-in machine-readable status snapshots:

- official training gate:
  `egs/overrides/rhythm_v3_gate_status.json`
- latest local strongest-contract candidate:
  `egs/overrides/rhythm_v3_gate_status_local_candidate_20260414.json`

Tracked historical comparison bundles that still live in the repo:

- `tmp/gate_reaudit_20260414_rebuilt2/`
- `tmp/gate_reaudit_20260414_runtime_clean/`
- `tmp/gate_reaudit_20260414_runtime_fixed/`

The ephemeral `gate0_final` rerun bundle that produced the latest local
candidate result is summarized into the checked-in local-candidate JSON and the
numbers below instead of being kept as another checked-in `tmp/` tree.

## 4. Gate verdict

| Gate | Verdict | Why |
| --- | --- | --- |
| Gate 0 | pass on the strongest fixed local contracts | after fixing source-side validity/support mismatch and init drift, the clean exact slice is no longer flat; `weighted_median` is positive on `8/8` clean exact cells and `raw` / `trimmed` are positive on `7/8` |
| Gate 1 | pass on the strongest fixed local contract | `weighted_median + exact_global_family` remains `4/4` pass on preclip, continuous, and projected analytic triplets |
| Gate 2 | not rerun in this pass | this round repaired audit/runtime contracts and reran zero-train gates only; no new full training conclusion should be claimed from this document |
| Gate 3 | not rerun in this pass | depends on a fresh Gate2-capable training rerun, which was out of scope here |

## 5. Gate0 clean-slice final audit

The current Gate0 reading should be taken from the clean exact slice:

- `reference_mode=target_as_ref`
- `src_prefix_stat_mode=exact_global_family`

Coverage by fixed `g` family:

| Variant | clean exact cells | positive valid-total slopes | zero slopes | best valid-total slope |
| --- | ---: | ---: | ---: | ---: |
| `weighted_median` | `8` | `8` | `0` | `0.4796` |
| `raw_median` | `8` | `7` | `1` | `0.3455` |
| `trimmed_mean` | `8` | `7` | `1` | `0.3995` |

Strongest current weighted cell
(`gate0_weighted_report.json`, candidate `63`, `drop_edge_runs_for_g=1`):

| Metric | Value |
| --- | ---: |
| `g_domain_valid_items` | `28` |
| `source_g_domain_valid_items` | `28` |
| valid total median slope | `0.4796` |
| valid total mean slope | `0.0282` |
| valid total duration-logratio slope | `0.1723` |
| valid analytic median slope | `0.1452` |
| valid analytic runtime mean slope | `0.0000` |
| valid residual median slope | `0.2022` |
| valid residual runtime mean slope | `0.0129` |
| valid residual runtime affine slope | `0.0350` |
| mean analytic saturation | `0.7757` |

Interpretation:

- the earlier Gate0 flat reading was substantially contaminated by a real
  measurement bug: source-support-aware `x`, `y`, and validity were not fully
  isomorphic
- after repair, the clean exact slice is not flat anymore
- Gate0 is therefore no longer evidence that the strongest local
  single-scalar line is dead on arrival
- the runtime-aligned analytic surface is still heavily saturated, so the
  improved Gate0 reading does not imply that runtime execution is easy
- residual statistics are no longer cleanly "anti-global" once the audit is
  support-aligned and affine-aware; the old residual-negativity story was
  overstated

## 6. Gate1 analytic runtime probe

For `weighted_median + exact_global_family`, the latest rerun still passes the
local 4-case analytic probe end-to-end:

| Layer | Pass count |
| --- | ---: |
| preclip monotonicity | `4/4` |
| continuous monotonicity | `4/4` |
| projected monotonicity | `4/4` |

Per-source projected slopes and ranges:

| Source | projected slope | projected range | mean saturation | boundary hit rate |
| --- | ---: | ---: | ---: | ---: |
| `aba_train_arctic_a0022` | `-0.0398` | `0.0500` | `0.6296` | `0.5250` |
| `asi_train_arctic_a0027` | `-0.1223` | `0.2500` | `0.6667` | `0.4643` |
| `bdl_train_arctic_a0022` | `-0.0486` | `0.0833` | `0.6757` | `0.1500` |
| `slt_train_arctic_a0020` | `-0.0866` | `0.1097` | `0.6667` | `0.3846` |

This confirms:

- Gate1 weighted did not regress during the Gate0 contract repair
- the layered preclip / continuous / projected split is stable on the strongest
  current local contract
- runtime saturation and projector compression still exist, but they no longer
  erase monotonic ordering on this fixed surface
- `exact_global_family` should still be read here as a local/offline gate
  contract; the maintained online runtime state still stores only a scalar
  prefix state and is not yet a strict full-history exact-prefix contract

## 7. Current reading

The latest local reading is now materially sharper than the earlier
`followup` snapshot:

- support collapse is not the main blocker
- Gate1 ordering drift was real and is now fixed on the maintained weighted
  local surface
- Gate0's earlier flat reading was substantially caused by source-side
  validity/support mismatch plus offline init drift
- after fixing those bugs, Gate0 clean-slice evidence becomes positive instead
  of flat on the strongest local contracts
- training-side target configuration also had real contract drift and is now
  wired to the intended `g` / prefix family

So the maintained conclusion for this local zero-train surface is now:

- the old "Gate0 still fails cleanly" reading is no longer current
- the strongest local fixed contract is now
  `weighted_median + exact_global_family + target_as_ref`
- Gate0 and Gate1 both survive on that local surface
- this does not yet promote `exact_global_family` to the maintained strict
  online runtime default
- this is still not a claim that the broader V1 line is universally solved,
  because this pass did not rerun full training or reopen Gate2 / Gate3
