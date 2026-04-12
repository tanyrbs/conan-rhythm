# rhythm_v3 / V1-G validation stack

This document defines the **validation-first** workflow for the maintained
`rhythm_v3` / minimal-V1 path.

The goal is not to add more loss terms first. The goal is to make the theory
**falsifiable** before large-scale training:

- source-anchor run retiming must be visible in the labels
- the speech-only global cue must carry usable signal
- coarse/local factorization must be auditable
- inference must expose the same contract as training

## 1. What to validate first

Think in three layers:

1. **Before training — label and alignment audit**
   - Is the source run lattice stable enough to be treated as the anchor?
   - Does target-to-source projection preserve mass and monotonicity?
   - Is `g` stable enough to act as a speech-only global tempo proxy?
   - Is scalar coarse still sufficient, or is low-frequency drift leaking into local residual?
   - Does silence look like bounded coarse follow rather than a pause script?

2. **During training — semantic audit**
   - Is the model actually learning `a_i + b + r_i` rather than hiding everything in one branch?
   - Is `b` learning sentence-level correction?
   - Is `r_i` remaining speech-local?
   - Are low-confidence alignment regions really weaker supervision?

3. **During inference — contract audit**
   - Does the model expose the same stable-lattice / source-anchor surface as training?
   - Are commit frontier, prefix offset, and carried rounding visible?
   - Can a single inference run be exported as a debug bundle for case review?

## 2. Current code support

This repo now provides a **non-plotting debug layer first**.

### 2.1 Shared alignment/projection module

File:

- `tasks/Conan/rhythm/duration_v3/alignment_projection.py`

This is the extracted target-to-source projection logic shared by:

- training dataset construction
- offline validation/debug tooling

It avoids putting debug-only logic into dataset mixins.

### 2.2 Debug-record schema

Files:

- `utils/plot/rhythm_v3_viz/core.py`
- `utils/plot/rhythm_v3_viz/alignment.py`

Main exported helpers:

- `build_debug_record(...)`
- `build_debug_records_from_batch(...)`
- `derive_record(...)`
- `record_summary(...)`
- `save_debug_records(...)`
- `load_debug_records(...)`
- `build_projection_debug_payload(...)`
- `attach_projection_debug(...)`

The debug record is the stable intermediate artifact for later visualization.

### 2.3 Inference-side debug export

File:

- `inference/Conan.py`

`StreamingVoiceConversion.infer_once(...)` now supports:

- `return_debug_bundle=True`

and stores the last bundle in:

- `self.last_rhythm_debug_bundle`

This means inference no longer needs a separate ad-hoc tracing path.

### 2.4 CSV summary script

File:

- `scripts/rhythm_v3_debug_records.py`

This script converts exported debug bundles into a flat summary CSV for
inspection, filtering, and later plotting.

Example:

```bash
python scripts/rhythm_v3_debug_records.py ^
  --input artifacts/rhythm_v3_debug ^
  --output artifacts/rhythm_v3_debug/summary.csv
```

## 3. Recommended validation variables

Before any charting layer, every sample should be reducible to the following
quantities:

- source run multiplicity `n_i`
- projected target multiplicity `n_i*`
- source-anchor target `z_i* = log((n_i* + eps)/(n_i + eps))`
- run confidence `ω_i`
- prompt global cue `g`
- source prefix tempo `p_i`
- analytic shift `a_i`
- oracle coarse `b*`
- oracle local residual `r_i*`

The current debug-record code reconstructs these from:

- dataset sample targets
- runtime source batch
- reference memory
- execution surface

## 4. How to use it in practice

### 4.1 Before training: label / alignment audit

Build a source-side debug record from a sample:

```python
from utils.plot.rhythm_v3_viz import build_debug_records_from_batch

records = build_debug_records_from_batch(sample=batch)
```

If you also have paired-target run info outside the dataset path, attach
projection debug **without touching the mixin**:

```python
from utils.plot.rhythm_v3_viz import attach_projection_debug

record = records[0]
record = attach_projection_debug(
    record,
    target_units=paired_target_units,
    target_durations=paired_target_durations,
    target_valid_mask=paired_target_valid_mask,
    target_speech_mask=paired_target_speech_mask,
)
```

Then persist:

```python
from utils.plot.rhythm_v3_viz import save_debug_records

save_debug_records([record], "artifacts/rhythm_v3_debug/pretrain_sample.pt")
```

### 4.2 During training: semantic audit

Inside a validation hook or notebook, export records from batch + model output:

```python
records = build_debug_records_from_batch(
    sample=batch,
    model_output=outputs,
    metadata={"phase": "valid"},
)
save_debug_records(records, "artifacts/rhythm_v3_debug/valid_step_1000.pt")
```

This captures:

- source lattice
- prompt conditioning summary
- target supervision
- predicted global/coarse/local surfaces
- commit mask / prefix offset

### 4.3 During inference: contract audit

```python
wav, mel, meta, debug_bundle = engine.infer_once(
    inp,
    return_metadata=True,
    return_debug_bundle=True,
)
```

Then save:

```python
import torch
torch.save([debug_bundle], "artifacts/rhythm_v3_debug/infer_case.pt")
```

## 5. Training alignment vs inference usage

This distinction should stay explicit.

### Training-time alignment

Alignment is a **label builder**:

- it projects paired target occupancy back to source runs
- it creates `n_i*`, `z_i*`, and confidence weights
- it enables oracle decomposition into `a_i`, `b*`, and `r_i*`

### Inference-time execution

Inference does **not** run target alignment online.

It only uses:

- prompt-side `g`
- source-side prefix tempo `p_i`
- model-predicted coarse/local terms
- carry rounding
- prefix offset / commit discipline

### Offline evaluation

Alignment reappears again only as an **evaluation tool**, not as an online module.

## 6. What to prove before re-adding plotting

The plotting layer was intentionally postponed. Before adding figures back,
ensure the exported debug records are sufficient to answer:

1. Is `g` stable on 3–8s speech-dominant prompts?
2. Does `g - p_i` correlate with oracle utterance-level stretch?
3. Is scalar `b*` enough, or is phrasewise coarse already necessary?
4. Does silence look bounded and low-freedom?
5. Are commit frontier and prefix offset stable enough for strict-causal deployment?

If the answer to those questions cannot be reconstructed from debug bundles and
summary CSV alone, the record schema is still missing information.
