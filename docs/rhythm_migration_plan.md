# Rhythm Migration Plan (2026-04-10)

This file is now the **short current-mechanism note** for rhythm.
Historical slot-memory, pointer/planner, and oversized v2 stage narratives were
removed on purpose.

## 1. What the current branch actually is

The maintained code mainline is **`rhythm_v3`**.

The honest interpretation of current code is:

> a **duration-only adjudication platform** built around a hard nominal baseline,
> a speech-only global stretch, and three candidate residual layers:
> progress warp, detector bank, and local operator.

So the repository should currently be read as:

- **single code mainline**: `rhythm_v3`
- **single task scope**: speech-unit duration transfer
- **single legacy bucket**: `rhythm_v2` compatibility/history
- **not yet a single proved theorem** about shared local bases

## 2. Current code-level decomposition

The maintained execution form is:

> `log d_hat_i = log b_i + g_ref + w_prog(i) + w_det(i) + 1_local * phi_i^T s_ref + 1_srcres * r_src(i)`

with:

- `b_i`: nominal duration baseline
- `g_ref`: speech-only prompt-global stretch
- `w_prog(i)`: optional progress-indexed warp
- `w_det(i)`: optional fixed detector-bank response
- `phi_i^T s_ref`: optional prompt-conditioned local operator
- `r_src(i)`: optional centered source-residual ablation

This matters because the code has already moved beyond the older
operator-only documentation. The current default example config is
`global_only`; progress/detector/operator remain candidate residual layers.

## 3. What is hard, and what is still candidate

### Hard maintained invariants

These are now the real core:

1. **baseline protocol**
2. **speech-only global stretch**
3. **deterministic projector**
4. **speech-only supervision/reporting**
5. **explicit prompt-unit training semantics**

### Candidate residual layers

These remain under adjudication:

- **progress warp**
- **detector bank**
- **shared-basis local operator**
- **centered source residual**

So the repository should no longer be described as
"the shared-basis operator mechanism" without qualification.

The preferred runtime control surface is now:

- `rhythm_v3_backbone: global_only | operator`
- `rhythm_v3_warp_mode: none | progress | detector`
- `rhythm_v3_allow_hybrid: false | true`

## 4. Current baseline protocol

Baseline is no longer just a small anchor network. In current code it is a
protocol object in:

- `modules/Conan/rhythm_v3/unit_frontend.py`

It contains:

1. **optional frozen table prior**
2. **strict causal local trunk**

Any slow / segment-scale structure is adjudicated outside the baseline,
through explicit progress/detector candidate layers rather than inside the nominal baseline.

Do not mix old precomputed anchor/log-base artifacts with the current baseline
contract. Progress-warp sampling is baseline-referenced, so stale baseline caches
will cause calibration drift even if the runtime interfaces still match.

Preferred progress-warp config names:

- `rhythm_progress_bins`
- `rhythm_progress_support_tau`

Legacy `rhythm_coarse_bins` / `rhythm_coarse_support_tau` remain as
compatibility aliases only.

Current lifecycle/config surface:

- `rhythm_v3_baseline_train_mode: joint | frozen | pretrain`
- `rhythm_v3_freeze_baseline`
- `rhythm_v3_baseline_ckpt`
- `rhythm_baseline_table_prior_path`
- `rhythm_v3_baseline_target_mode: raw | deglobalized`
- `lambda_rhythm_base`

This is still a **scaffold**, not a completed alternating protocol. The branch
can now run a baseline-only pretrain stage with deglobalized targets, but it
does not yet claim the full baseline-0 -> deglobalized baseline-1 -> residual
audit schedule is finished. Prompt conditioning is not required in this
baseline-only pretrain mode.

### Baseline separation rule

The branch now enforces a stricter reading:

- operator-side prompt/source baseline features are detached
- stream-side prefix/consistency supervision rebuilds durations from
  `sg(unit_anchor_base) * exp(unit_logstretch)`
- baseline is not allowed to absorb prompt-style transfer through the operator path

## 5. Current public claim boundary

The maintained claim is intentionally narrow:

- **speech-unit duration transfer**
- **causal / sealed-unit prediction**
- **prompt-conditioned duration adaptation**

The branch does **not** currently claim:

- full rhythm/prosody transfer
- pause/boundary as the main mechanism
- runtime pointer traversal as the explanation
- shared basis as already proven necessary

## 6. Current training semantics

### Prompt evidence

Mainline training requires:

- `prompt_content_units`
- `prompt_duration_obs`
- `prompt_unit_mask`

Trace/proxy-only conditioning is removed from the v3 path.
Inference must pass explicit prompt units (or prebuilt duration memory).

### Supervision

- v3 main supervision is speech-only
- separator units are excluded from duration/prefix supervision and reporting
- holdout prompt self-fit is a diagnostic/probe surface

### Consistency

- `lambda_rhythm_cons` defaults to `0.0`
- current consistency is diagnostic-only
- it should not be elevated again until raw short/long prefix views are compared
  before freeze/projector

## 7. Current compact v3 public surface

The compact committed surface is the one in:

- `egs/conan_emformer_rhythm_v3.yaml`

### Inputs

- `content_units`
- `dur_anchor_src`
- `unit_anchor_base`
- `prompt_content_units`
- `prompt_duration_obs`
- `prompt_unit_mask`

### Outputs

- `speech_duration_exec`
- `rhythm_frame_plan`
- `commit_frontier`
- `rhythm_state_next`

### Compact public losses

- `rhythm_total`
- `rhythm_v3_dur`
- `rhythm_v3_op`
- `rhythm_v3_pref`
- `rhythm_v3_zero`

Diagnostics like `rhythm_v3_cons` and `rhythm_v3_ortho` may still exist in
code/tests, but they are not the compact example contract.

## 8. File map for the current story

### Runtime

- `modules/Conan/rhythm_v3/runtime_adapter.py`
- `modules/Conan/rhythm_v3/module.py`
- `modules/Conan/rhythm_v3/reference_memory.py`
- `modules/Conan/rhythm_v3/projector.py`
- `modules/Conan/rhythm_v3/unit_frontend.py`

### Training surfaces

- `tasks/Conan/rhythm/targets.py`
- `tasks/Conan/rhythm/losses.py`
- `tasks/Conan/rhythm/metrics.py`
- `tasks/Conan/rhythm/task_config.py`
- `tasks/Conan/rhythm/task_mixin.py`

### Inference/runtime utilities

- `inference/Conan.py`
- `inference/run_voice_conversion.py`
- `inference/run_streaming_latency_report.py`

## 9. Readiness reading

### Ready enough for

- one-branch `rhythm_v3` code maintenance
- local mechanism ablations
- baseline/global/progress/detector/operator comparative experiments
- focused streaming-oriented evaluation

### Not ready to over-claim

The project is **not yet ready** to claim that:

- local operator/shared basis is already the final mechanism
- progress warp or detector bank is unnecessary
- baseline pretraining/de-style protocol is finished as a standalone stage
- inference is fully stateful end-to-end streaming deployment

## 10. Legacy status of v2

`modules/Conan/rhythm/` and older v2 teacher/planner/export docs remain only
for:

- old checkpoints
- compatibility
- teacher/export history
- archival experiments

They are not the authoritative mechanism description for the current branch.

## 11. Documentation rule

When docs disagree, prefer the smallest truthful current reading:

- `rhythm_v3` is the only maintained code mainline
- baseline/global are hard requirements
- progress/detector/local are candidate residual layers under adjudication
- speech-unit duration transfer is the scope
- deterministic projector is part of the execution contract

Authoritative current docs:

- `README.md`
- `docs/rhythm_migration_plan.md`
- `inference/README.md`
