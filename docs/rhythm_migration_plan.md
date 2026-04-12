# Rhythm Migration Plan / Maintained V1 Spec (2026-04-12)

This file is the canonical architecture note for the maintained `rhythm_v3` line.

For the practical repo setup / data preparation / training workflow, see
`docs/rhythm_v3_training_guide.md`.

## 1. Current maintained reading

The maintained line is **`rhythm_v3`**.

The final maintained V1 is a **two-layer system**:

- **Layer 0: Stable Lattice Interface**
  - turns the raw unit stream into a stable, causal, commit-safe run interface
  - keeps training cache generation and runtime frontends on the same stabilizer reading
- **Layer 1: Minimal Retimer**
  - learns only a source-anchored relative retiming on top of the stable run lattice
  - keeps explanation power limited to analytic global rate gap, one coarse scalar, and speech-only local residual

The maintained default remains:

- explicit prompt units
- source-observed anchors
- explicit silence runs
- speech-only prompt global-rate estimation
- shared train/infer lattice stabilizer
- prompt summary memory kept as an optional diagnostic/ablation surface
- deterministic carry-rounding projector with prefix budget
- paired-target supervision projected back onto the source lattice with split local/coarse confidence

Recommended config surface:

- `egs/conan_emformer_rhythm_v3.yaml`
- `rhythm_v3_backbone: prompt_summary` (`unit_run` / `role_memory` remain accepted legacy aliases)
- `rhythm_v3_warp_mode: none`
- `rhythm_v3_anchor_mode: source_observed`
- `rhythm_v3_minimal_v1_profile: true`
- `rhythm_v3_rate_mode: simple_global`
- `rhythm_v3_simple_global_stats: true`
- `rhythm_v3_use_reference_summary: false`
- `rhythm_v3_disable_learned_gate: true`
- `rhythm_v3_emit_silence_runs: true`
- `rhythm_v3_require_same_text_paired_target: true`
- `rhythm_tail_open_units: 2`
- `rhythm_v3_debounce_min_run_frames: 2`

One-sentence summary:

> V1 first converts the raw unit stream into a stable, reliability-aware, commit-safe run lattice, then learns a minimal source-anchored retimer whose only learned freedoms are one utterance-level coarse correction and speech-only local residuals; silence remains a clipped coarse follower rather than an independent pause planner.

## 2. External rationale checkpoints

The maintained narrowing of this branch stays aligned with a few external anchors:

- Conan paper: discrete content labels are the controllable interface, so the retimer should attach to the content-code stream instead of the chunk scheduler.
- R-VC: token/run-level duration control is more stable than sentence-level duration control, and duration-side conditioning matters for quality.
- L2-ARCTIC: non-native pronunciation deviations make confidence-aware paired supervision necessary instead of blindly trusting monotonic projection.
- CMU ARCTIC prompt sharing remains the simplest native-native / L2-native sanity surface.

## 3. Stable-Lattice Source-Anchored Coarse-to-Local Run Retiming

### 3.1 Motivation

In Conan, `chunk` is the streaming input and scheduling unit; it is not the right semantic object for duration control. The natural modeling object is the **run lattice** obtained after causal debounce and adjacent same-code merging.

That observation is necessary but not sufficient. If the run lattice itself is noisy, then all downstream quantities are corrupted together:

- source multiplicity `n_i`
- speech/silence routing `s_i`
- boundary and phrase sidecars
- target projection back onto the source lattice
- relative log-stretch labels `z_i*`

So V1 can no longer treat "light debounce + same-code merge" as a disposable pre-processing detail. It has to elevate that step into a **stable input contract**. The duration head is not supposed to be a universal denoiser for dirty lattices; the correct order is:

1. stabilize the lattice
2. propagate uncertainty
3. supervise duration on top of the stabilized interface

This leads to a two-layer definition:

- **Layer 0** builds a stable lattice interface from raw unit streams.
- **Layer 1** performs minimal source-anchored retiming only on that interface.

That keeps V1 identifiable, causal, and robust enough for deployment.

### 3.2 Layer 0: Stable Lattice Interface

Given the cumulative raw unit stream `a_{1:T}`, we define a shared stabilizer `S` and construct the source run lattice

`R_src = S(a_{1:T}) = {(u_i, n_i, s_i, omega_i, kappa_i)}_{i=1}^N`.

Each run contains:

- `u_i`: unit id
- `n_i`: source multiplicity / source run duration
- `s_i in {0,1}`: speech/silence flag
- `omega_i in [0,1]`: run reliability
- `kappa_i in {0,1}`: whether the run is closed and may be committed

This is a **spec-level contract**. In the current repo, the same contract is realized by a combination of:

- run ids and durations from the unitizer/frontend
- `source_silence_mask`
- `sealed_mask` / open-tail handling
- paired-target confidence fields such as `unit_confidence_local_tgt` and `unit_confidence_coarse_tgt`

The key rule is that the stabilizer `S` must be shared by training cache generation and runtime streaming. Otherwise train/infer lattice drift breaks V1 at the interface level.

The stabilizer may use fixed rules or lightweight parametric rules, but the maintained reading is conservative:

- suppress short flicker bridges such as `A-x-A`
- suppress short fake silence islands such as `speech-silence-speech` when boundary evidence is weak
- preserve causal commit discipline: already committed prefix runs are never rewritten
- keep explicit silence runs in the lattice instead of collapsing them into separator-only markers

Current code surfaces that implement this contract include:

- `modules/Conan/rhythm_v3/unitizer.py`
- `modules/Conan/rhythm_v3/unit_frontend.py`
- `modules/Conan/rhythm_v3/source_cache.py`
- `tasks/Conan/rhythm/duration_v3/dataset_mixin.py`
- `inference/Conan.py`

Training projects target durations back onto the source lattice and obtains target occupancy `n_i*`. The supervision target is the relative log-stretch

`z_i* = log((n_i* + eps) / (n_i + eps))`.

This preserves the source-anchor reading: the model learns how to stretch the fixed source skeleton, not how to generate a new absolute duration sequence from scratch.

### 3.3 Layer 1: Minimal Retimer

#### 3.3.1 Simple speech-only global tempo statistic

V1-G deliberately drops content-normalized rate modeling from the default path.
For speech runs only, it uses the raw log-duration statistic

`d_i = log(n_i + eps)`, with `s_i = 0`.

The source prefix speech tempo is the strict-causal EMA

`p_i = EMA_{k < i, s_k = 0}(d_k)`.

The reference-side global tempo statistic is

`g = median_{j: s_j^ref = 0}(d_j^ref)`.

So the maintained claim is intentionally narrower: `g` is a stable speech-only
global tempo proxy, not a fully content-normalized speaking-rate estimate. The
default reference condition is therefore just

`c = [e, g]`

where `e` is the speaker embedding. Low-leakage prompt summary memory remains
available for diagnostics and ablation, but it is not part of the default V1-G
writer. The maintained CI/preflight/smoke surface should therefore follow the
same V3-only reading rather than the archived V2 smoke path.

#### 3.3.2 Three explanatory terms: `a_i`, `b`, `r_i`

V1 keeps exactly three explanatory terms.

1. **Analytic global rate gap**

   `a_i = g - p_i`

2. **Single coarse correction**

   `b = delta * tanh(F_b(e))`

   `b` is an utterance-level scalar, not a time-varying vector. This is deliberate: V1 does not allow the coarse branch to compete with the local branch in explaining the same local structure.

3. **Speech-only local residual**

   Let the source encoder produce a causal source state `h_i`. Then

   `r_i = alpha * tanh(F_r(h_i, e, a_i + b))`

   `r_i` only applies to stable, closed, speech runs. Low-confidence or still-open runs are implicitly gated down.

The final predicted log-stretch is

- speech runs: `z_hat_i = a_i + b + r_i`
- silence runs: `z_hat_i = clip(a_i + b, -tau, tau)`

So speech gets **coarse + local**, while silence gets **coarse-only clipped follow**. Silence stays in the sequence but does not get its own writer.

#### 3.3.3 Execution and causal commit

The continuous multiplicity prediction is

`n_tilde_i = n_i * exp(z_hat_i)`.

Discrete execution uses carry rounding:

`k_i = max(1, round_carry(n_tilde_i))`.

The prefix budget is

`o_i = sum_{t <= i}(k_t - n_t)`, constrained to `[-B_-, B_+]`.

Only runs with `kappa_i = 1` are committed. Committed runs are never rewritten.

So V1 is causal in the stronger sense:

- the network does not look into the future
- only closed prefix runs are committed
- committed prefix predictions are not allowed to change later

This is the maintained strict-causal commit discipline.

### 3.4 Training

#### 3.4.1 Labels and reliability

Monotonic paired projection should not collapse every mismatch into duration truth. In addition to `n_i*`, training carries run-level reliability `omega_i`, driven by:

- alignment path cost / margin
- run-lattice stability
- speech vs silence routing confidence
- optional mispronunciation annotations or exclusion surfaces

A key rule is:

> unstable silence may remain on the lattice, but it must not be elevated into high-confidence hard truth.

This is why V1 splits the existence of silence runs from the right to supervise them strongly.

For the maintained `minimal_v1_profile`, data pairing is also intentionally asymmetric:

- the **reference prompt** should stay same-speaker / different-text to reduce prompt leakage
- the **paired supervision target** should stay same-text so source-anchor projection is not polluted by lexical mismatch
- if precomputed `unit_duration_tgt` is already cached, that explicit cached target can replace the online same-text projection step

In the current code, this is expressed through split confidence routing:

- `unit_confidence_local_tgt`
- `unit_confidence_coarse_tgt`

rather than one shared confidence scalar.

#### 3.4.2 Minimal loss set: 2 + 1

V1 keeps only a minimal objective family.

First define the speech-only coarse target

`b* = wmed_{i: s_i = 0}(z_i* - a_i ; omega_i)`.

Then define the speech local target

`r_i* = z_i* - a_i - b*`.

The main losses are:

1. **Speech residual loss**

   `L_loc = sum_{i: s_i = 0} omega_i * Huber(r_i, r_i*) / sum_{i: s_i = 0} omega_i`

2. **Coarse loss**

   `L_crs = Huber(b, b*)`

3. **Committed-prefix consistency loss** (opened only for strict-causal fine-tuning)

   `L_con = (1 / |C|) * sum_{i in C} Huber(z_hat_i^short, sg(z_hat_i^long))`

The total objective is

`L = L_loc + lambda_c * L_crs + lambda_p * L_con`.

V1 intentionally does **not** introduce a standalone silence primary loss. Silence is a deterministic clipped coarse follower, not an independent modeling object. If training needs a silence target for prefix or consistency bookkeeping, it must be a **coarse-derived pseudo-target**, never a raw full pause target.

That rule keeps the implementation aligned with the theory:

- speech local residuals learn the local rhythm change
- coarse learns the utterance-level pace shift
- silence cannot re-enter as a hidden pause planner through auxiliary losses

### 3.5 Boundary / non-goal statement

This V1 is **not**:

- a pause planner
- a reference timeline matcher
- a prompt-specific lexical micro-rhythm copier
- a joint duration-F0-energy full prosody transfer model

It is a **stable-lattice, source-anchored, speech-dominant, causally committed relative retimer**.

That contraction is intentional. It trades away some expressive power in exchange for:

- clearer boundaries
- better identifiability
- simpler ablations
- lower deployment risk

### 3.6 Upgrade interfaces reserved for later versions

V1 leaves room for later upgrades, but they should not be mixed into this version prematurely.

- **Boundary-aware silence branch**
  - upgrade constant `tau` into boundary-aware `tau_i`
  - optionally add silence-specific residuals
  - belongs to V2, not V1

- **Phrasewise coarse**
  - upgrade scalar `b` into phrasewise `b_i`
  - only after scalar coarse is shown insufficient

- **Richer reference memory**
  - only after proving the current summary `m` remains stable under same-speaker different-text conditioning
  - otherwise it easily collapses into lexical template matching

- **Joint prosody transfer**
  - only after the duration frontend is stable
  - do not let duration and decoder-side style/pitch pathways compensate for each other during V1

## 4. Current repo contract mapping

### 4.1 Prompt-side contract

The maintained prompt-summary path requires explicit prompt-unit evidence:

- `prompt_content_units`
- `prompt_duration_obs`
- `prompt_unit_mask`

For the maintained prompt-summary reading, speech-only rate semantics must be preserved. So runtime accepts, in order:

1. `prompt_speech_mask`
2. `prompt_silence_mask`
3. derivation from explicit silence tokens when `rhythm_v3_emit_silence_runs=true`

It no longer silently falls back to `prompt_valid_mask` for prompt-summary mode.

The prompt encoder exports:

- `global_rate`
- `summary_state`
- `spk_embed`
- diagnostic summary slots / role statistics

Pooling and prompt global-rate estimation remain speech-only.

### 4.2 Source/runtime contract

The maintained source-side contract uses:

- `content_units`
- `source_duration_obs` / `dur_anchor_src`
- `unit_anchor_base`
- `unit_rate_log_base`
- `source_silence_mask`
- `source_boundary_cue`
- `phrase_group_pos`
- `phrase_final_mask`
- sealed/open masks from the frontend

Important current rules:

- precomputed source durations stay float-valued; they are not forcibly truncated to integers
- prompt-summary + explicit silence runs requires `source_silence_mask` in precomputed source caches
- runtime performs EOS tail closing instead of leaving the final open tail permanently outside committed retiming
- the rhythm frontend can run incrementally rather than rebuilding the whole source lattice every chunk

### 4.3 Training-side contract

Paired-target supervision is kept separate from prompt conditioning. The maintained paired projection path returns:

- `unit_duration_tgt`
- `unit_confidence_local_tgt`
- `unit_confidence_coarse_tgt`
- alignment diagnostics such as coverage / match / cost

That split is central to the maintained V1 reading:

- high-confidence speech runs may supervise local residuals
- lower-confidence speech runs may still supervise coarse structure
- silence never becomes a high-confidence local target

## 5. Current file map

### Runtime

- `modules/Conan/rhythm_v3/module.py`
- `modules/Conan/rhythm_v3/summary_memory.py`
- `modules/Conan/rhythm_v3/projector.py`
- `modules/Conan/rhythm_v3/runtime_adapter.py`
- `modules/Conan/rhythm_v3/unit_frontend.py`
- `modules/Conan/rhythm_v3/unitizer.py`

### Training/data

- `tasks/Conan/rhythm/common/`
- `tasks/Conan/rhythm/duration_v3/`
- `modules/Conan/rhythm/supervision.py`
- `modules/Conan/rhythm/unit_frontend.py`
- `modules/Conan/rhythm/unitizer.py`

### Inference/runtime helpers

- `inference/Conan.py`
- `inference/run_voice_conversion.py`
- `inference/run_streaming_latency_report.py`

## 6. Documentation rule

When documentation disagrees, prefer the smallest truthful current reading:

- `rhythm_v3` is the maintained line
- final V1 is the two-layer reading above: stable lattice interface + minimal retimer
- explicit silence runs, speech-only global rate, speech coarse+local, silence coarse-only, carry rounding, and prefix budget are the defining constraints
- top-level task files are compatibility shells, not the main implementation location
- legacy v2 notes are archival, not the mainline specification

## 7. Legacy status of v2

`rhythm_v2` remains only for:

- old checkpoints
- teacher/export compatibility
- legacy experiments
- archival operational notes

It is not the maintained architecture target.

## 8. Validation snapshot (2026-04-12)

Validated locally after the stable-lattice / prompt-summary contract tightening and paired-projection confidence split:

- `py -3 -m py_compile` on the touched v3 runtime, frontend, dataset, task-config, and test files
- pytest bundle:
  - `tests/rhythm/test_task_config_v3.py`
  - `tests/rhythm/test_runtime_modes_v3.py`
  - `tests/rhythm/test_task_runtime_support.py`
  - `tests/rhythm/test_cache_contracts.py`
  - `tests/rhythm/test_rhythm_v3_losses.py`
  - `tests/rhythm/test_rhythm_v3_metrics.py`
  - `tests/rhythm/test_rhythm_v3_runtime.py`
  - `tests/rhythm/test_rhythm_v3_unit_frontend.py`
  - with `-k "not test_protocol_duration_baseline_table_prior_file_offsets_nominal_anchor" -p no:cacheprovider -p no:tmpdir`

Result: **184 passed, 1 deselected**.
