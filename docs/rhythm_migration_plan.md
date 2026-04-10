# Rhythm Migration Plan (2026-04-10)

This file is now the **concise current-mechanism note** for rhythm.
Historical branch diaries, slot-memory narratives, phrase-pointer controller
stories, and oversized v2 stage menus were intentionally removed.

## 1. Current mechanism mainline

The current rhythm mechanism mainline is **`rhythm_v3`**:

> `log d_hat_i = b_i + g_ref + phi_i^T s_ref`

with:

- `b_i`: content-conditioned nominal duration baseline
- `g_ref`: speech-only prompt-global stretch
- `phi_i`: shared causal basis activation
- `s_ref`: prompt-conditioned operator coefficients

This is a **neural mixed-effects / low-rank operator** formulation:

- **fixed effect**: content baseline
- **random intercept**: prompt-global stretch
- **random slopes**: prompt-conditioned operator over shared bases

## 2. Five maintained parts only

The maintained main path keeps only five necessary parts:

1. **content baseline `B`**
2. **speech-only global stretch `g_ref`**
3. **shared causal basis `F`**
4. **prompt-conditioned operator `s_ref`**
5. **deterministic projector**

Everything else is secondary, diagnostic, ablation-only, or legacy.

## 3. What was explicitly removed from the main story

The current docs no longer treat these as the core mechanism:

- slot memory
- role codebooks
- static prompt memory queried at runtime
- posterior precision as an inference-time control path
- runtime retrieval weights as the central explanation
- source residual as a required core variable
- boundary controller / pause controller as main duration writers
- reference timeline / pointer traversal as the duration-conditioning story

If any of these still exist in legacy code or archives, read them as
compatibility-era surfaces, not as the current mainline claim.

## 4. Scope of the current claim

The maintained v3 claim is intentionally narrow:

- **speech-unit duration transfer**
- **causal / sealed-unit prediction**
- **prompt-conditioned local duration response**

The mainline does **not** claim full rhythm/prosody transfer.
Separator / pause duration is outside the core mechanism and should not be used
to redefine the mainline story.

## 5. Current implementation invariants

The current repo implementation commits to the following:

### 5.1 Prompt evidence and training semantics

- mainline training requires explicit prompt-unit conditioning:
  - `prompt_content_units`
  - `prompt_duration_obs`
  - `prompt_unit_mask`
- trace/proxy-only conditioning is fallback/inference-side only
- `ReferenceDurationMemory` should be read as a **prompt summary / operator
  evidence pack**, not as slot memory

### 5.2 Baseline separation

- baseline is content-only
- operator-side prompt/source baseline features are detached
- baseline is not allowed to absorb prompt-style transfer through the operator path

### 5.3 Global/local factorization

- `g_ref` is estimated from **speech-only** prompt units
- local response is represented through shared causal bases plus prompt operator coefficients
- separator units do not define the main duration mechanism

### 5.4 Supervision/reporting

- v3 main supervision is **speech-only**
- separator units are excluded from duration/prefix supervision and v3 speech-side metrics
- holdout operator self-fit is the maintained prompt diagnostic

### 5.5 Consistency semantics

- `lambda_rhythm_cons` defaults to `0.0`
- current consistency is diagnostic-only, not a mainline mechanism claim
- a stronger consistency loss should only be revived after raw short/long prefix
  views are compared before freeze/projector

## 6. Current public v3 surface

The compact public surface is the one in
`egs/conan_emformer_rhythm_v3.yaml`.

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

Optional/internal diagnostics such as consistency or orthogonality should not
be mistaken for the compact public contract.

## 7. File map for the current story

### Runtime

- `modules/Conan/rhythm_v3/runtime_adapter.py`
- `modules/Conan/rhythm_v3/module.py`
- `modules/Conan/rhythm_v3/reference_memory.py`
- `modules/Conan/rhythm_v3/projector.py`

### Training surfaces

- `tasks/Conan/rhythm/targets.py`
- `tasks/Conan/rhythm/losses.py`
- `tasks/Conan/rhythm/metrics.py`
- `tasks/Conan/rhythm/task_config.py`
- `tasks/Conan/rhythm/task_runtime_support.py`

## 8. Legacy status of v2

`modules/Conan/rhythm/` and the associated teacher/planner/pointer-era docs
remain in the repository for:

- old checkpoints
- compatibility
- teacher/export history
- archival experiments

They should not be read as the current mechanism mainline.

## 9. Documentation policy from now on

When docs disagree, prefer the smallest current v3 surface:

- operator, not slot memory
- speech-unit duration transfer, not full prosody control
- explicit prompt-unit training semantics
- speech-only supervision
- deterministic projector

Older v2 boundary/planner/phrase-bank descriptions may remain in archival docs,
but they are not authoritative for the current mainline.
