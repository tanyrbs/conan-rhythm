# rhythm_v3 local status snapshot (2026-04-14)

This file records the latest local maintained rerun after the current round of
Gate-path fixes.

## 1. Scope

Surface:

- maintained `rhythm_v3` minimal-V1 path
- local quick-ARCTIC configs and probe scripts
- strict prompt-domain contract enabled

Key implementation fixes completed before rerun:

- source-cache `rhythm_v3_cache_meta` now survives sample assembly, collate,
  and runtime source-cache collection
- cached-source `source_boundary_cue` now survives the runtime-minimal sample
  contract instead of being dropped from strict minimal-V1 batches
- Gate-1 analytic probe now records strict-invalid rows rather than crashing on
  prompt-domain failures
- review/debug code already includes the current Gate-0 signal-vs-residual
  split and stricter Gate-1 semantics

Regression coverage run after these fixes:

- `389 passed` across the rhythm_v3/runtime/review/config test slices

## 2. Artifacts

Current rerun artifacts live in:

- `tmp/gate_reaudit_20260414/`

Primary files:

- `tmp/gate_reaudit_20260414/full_split_boundary_audit_report.json`
- `tmp/gate_reaudit_20260414/full_split_boundary_audit_rows.csv`
- `tmp/gate_reaudit_20260414/counterfactual_static_gate0_report.json`
- `tmp/gate_reaudit_20260414/counterfactual_static_gate0_direction_report.json`
- `tmp/gate_reaudit_20260414/counterfactual_static_gate0_rows.csv`
- `tmp/gate_reaudit_20260414/gate1_analytic_results.csv`
- `tmp/gate_reaudit_20260414/gate1_analytic_summary.json`

## 3. Gate verdict

| Gate | Verdict | Why |
| --- | --- | --- |
| Gate 0 | fail | maintained clean-support domain collapses at `boundary-clean@0.5` |
| Gate 1 | fail | no strict-valid analytic `slow / mid / fast` triplet survives |
| Gate 2 | blocked | Gate 0 / Gate 1 not passed |
| Gate 3 | blocked | Gate 2 not admissible |

Current training decision:

- do not start Gate-2 training
- do not start Gate-3 training
- do not start official training
- do not start prefix fine-tune

## 4. Gate 0 metrics

### 4.1 Gate 0-A: maintained domain/support

Configured maintained threshold:

- `rhythm_v3_min_boundary_confidence_for_g = 0.5`

Maintained domain result:

- train: `g_domain_valid_items = 0/32`
- valid: `g_domain_valid_items = 0/16`
- test: `g_domain_valid_items = 0/16`

Counterfactual threshold recovery:

- at `0.45`: train `21/32`, valid `9/16`, test `8/16`
- at `0.40`: train `24/32`, valid `12/16`, test `8/16`

Interpretation:

- current failure is not only a signal-direction problem
- the maintained prompt clean-support domain itself is not standing on this
  local surface

### 4.2 Gate 0-B: counterfactual direction

Selected static counterfactual summary:

| candidate token | drop_edge | valid items | valid total-signal slope | valid prefix-signal slope | note |
| --- | ---: | ---: | ---: | ---: | --- |
| `57` | `1` | `0/64` | `nan` | `nan` | maintained token is dead |
| `63` | `1` | `29/64` | `-0.0000` | `0.0000` | support recovers but control stays flat |
| `71` | `1` | `13/64` | `0.3106` | `0.2601` | only non-maintained slice with non-trivial positive total-signal slope |
| `72` | `1` | `28/64` | `0.0000` | `0.0000` | support recovers but control stays flat |

Important caveat:

- even the `token=71, drop_edge=1` slice does **not** rescue the maintained
  path
- coarse-residual slopes in the same counterfactual audit remain negative
- this is therefore a diagnostic recovery slice, not a training green light

## 5. Gate 1 metrics

Strict analytic rerun summary:

- probe cases: `4`
- total rows: `20`
- strict-valid rows: `0`

Per-source summary:

| source | valid real refs | transfer slope | monotone by prompt tempo |
| --- | ---: | ---: | --- |
| `aba_train_arctic_a0010` | `0/3` | `nan` | `false` |
| `asi_train_arctic_a0011` | `0/3` | `nan` | `false` |
| `bdl_train_arctic_a0014` | `0/3` | `nan` | `false` |
| `slt_train_arctic_a0014` | `0/3` | `nan` | `false` |

Invalid-row reasons across the `20` probe rows:

- `14` rows: `V1-G prompt conditioning requires non-empty closed/boundary-clean support for g.`
- `6` rows: prompt `ref_len_sec < 3.0`

Interpretation:

- Gate 1 no longer fails because of a probe/runtime plumbing bug
- Gate 1 now fails because the strict maintained domain contract rejects every
  analytic real-reference triplet on this local probe surface

## 6. Practical conclusion

The current maintained local claim should be stated narrowly:

- on this local quick-ARCTIC surface, maintained `raw_median + boundary-clean@0.5`
  has not earned entry into Gate-2
- some counterfactual Gate-0 signal recovery exists, but only on a
  non-maintained slice and only on a subset
- the correct next step is more zero-train audit and prompt-domain / boundary
  support cleanup, not parameter training

Until a new rerun produces:

- non-collapsed maintained Gate-0 domain coverage, and
- real strict-valid Gate-1 analytic triplets with positive transfer

the repository should continue to treat Gate-2 / Gate-3 training as blocked.
