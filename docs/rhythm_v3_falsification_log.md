# rhythm_v3 falsification log

## 2026-04-14 local rerun

### Scope

- maintained `rhythm_v3` minimal-V1 only
- local quick-ARCTIC only
- zero-train Gate0 / Gate1 rerun after cache rebuild and frontend repair

### What was repaired first

- prompt/source silence sidecars now come from acoustic silence when raw
  `silent_token` evidence is absent
- processed metadata now filters prompt references to the maintained
  `3.0s-8.0s` contract before split construction
- source cache version was bumped so stale binaries cannot be silently reused
- probe-case selection is now generated from the current split instead of stale
  hand-maintained ids

### What is no longer true

The old statement below is no longer valid on the rebuilt local surface:

- "Gate fails because boundary-clean support collapses first"

Current evidence contradicting that old claim:

- boundary support at `min_boundary_confidence_for_g=0.5` is now
  `32/32` train, `16/16` valid, `16/16` test
- Gate0 prompt-domain validity is `63/64` for every tested `g` variant

### Current Gate0 result

Support exists, but total-signal explainability remains flat.

Summary from `tmp/gate_reaudit_20260414_rebuilt2/gate0_*/report.json`:

| Variant | total slope | analytic slope | residual slope |
| --- | ---: | ---: | ---: |
| `raw_median` | `0.0000` | `0.5316` | `-0.6329` |
| `weighted_median` | `0.0000` | `0.5316` | `-0.6329` |
| `trimmed_mean` | `-0.0000` | `0.5857` | `-0.6601` |
| `softclean_wmed` | `0.0000` | `0.3833` | `-0.6666` |
| `softclean_wtmean` | `0.0000` | `0.5615` | `-0.6852` |

Reading:

- the analytic component is present
- the total target signal still does not move with `Δg`
- residual decomposition remains anti-aligned

### Current Gate1 result

Summary from `tmp/gate_reaudit_20260414_rebuilt2/gate1_*/summary.json`:

| Variant | monotone sources | mean transfer slope | mean real tempo range |
| --- | ---: | ---: | ---: |
| `raw_median` | `0/4` | `~0.0000` | `0.0000` |
| `weighted_median` | `0/4` | `~0.0000` | `0.0000` |
| `trimmed_mean` | `0/4` | `~0.0000` | `0.0000` |
| `softclean_wmed` | `1/4` | `0.1068` | `0.0393` |
| `softclean_wtmean` | `1/4` | `0.1000` | `0.0333` |

Single positive source:

- `asi_train_arctic_a0019`

Reading:

- `weighted_median` and `trimmed_mean` do not rescue runtime control
- softclean weighting helps, but only on one source and is not stable enough for
  a maintained claim

### Current conclusion

The maintained conclusion after this rerun is:

- keep Gate 2 blocked
- keep Gate 3 blocked
- keep formal training blocked on this local surface
- stop describing the failure as support-domain collapse
- describe the failure as weak or unstable single-scalar control after frontend
  repair

### Retained script surface

The maintained zero-train diagnostics are now:

- `scripts/preflight_rhythm_v3.py`
- `scripts/audit_rhythm_v3_boundary_support.py`
- `scripts/audit_rhythm_v3_counterfactual_static_gate0.py`
- `scripts/probe_rhythm_v3_gate1_analytic.py`
- `scripts/probe_rhythm_v3_gate1_silent_counterfactual.py`
- `scripts/rhythm_v3_probe_cases.py`
- `scripts/rhythm_v3_debug_records.py`

Removed in this cleanup pass:

- token-guessing silent-token sweep helpers
- support-degeneracy reducer tied to the old token-candidate debugging path
- earlier one-off counterfactual reducer scripts already deleted in the same
  cleanup line
