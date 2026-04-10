[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

# Conan Rhythm Branch

This checkout now documents and maintains **one current rhythm mechanism story**:
`rhythm_v3`, a **duration-only neural mixed-effects / operator model**.

The current prediction form is:

> `log d_hat_i = b_i + g_ref + phi_i^T s_ref`

where:

- `b_i`: content-conditioned nominal duration baseline
- `g_ref`: speech-only prompt-global stretch
- `phi_i`: shared causal local response basis activation
- `s_ref`: prompt-conditioned operator coefficients

Legacy `rhythm_v2` code still exists for compatibility, teacher/export history,
and old checkpoints, but it is **not** the current mechanism mainline described
by this README.

## Current mainline in one sentence

Encode the prompt once, estimate a speech-only global stretch and a
prompt-conditioned low-rank duration operator, then apply that operator to each
sealed source speech unit and commit frames through a deterministic projector.

## The five necessary pieces

The current main path is intentionally small:

1. **content baseline `B`**
   - nominal duration from local content only
2. **speech-only global stretch `g_ref`**
   - robust prompt-global intercept estimated from prompt speech units
3. **shared causal basis `F`**
   - low-rank shared basis space for prompt/source local duration response
4. **prompt-conditioned operator `s_ref`**
   - prompt-specific coefficients solved once from prompt evidence
5. **deterministic projector**
   - rounding-residual carry and commit bookkeeping, not a learned controller

## What is explicitly not in the main path

The current `rhythm_v3` story is **not**:

- slot memory
- role codebooks
- posterior precision in inference
- runtime retrieval / reference pointer traversal
- source residual as a required core variable
- boundary controller / pause controller as main duration writers
- reference timeline consumption

Separator / pause behavior is **not** the core mechanism claim here.
The maintained v3 scope is **speech-unit duration transfer**.

## Current training contract

The current implementation enforces these rules:

- **explicit prompt units are required for mainline training**
  - `prompt_content_units`
  - `prompt_duration_obs`
  - `prompt_unit_mask`
- **baseline must stay out of the operator path**
  - prompt/source operator-side baseline features are detached
- **prompt global stretch is speech-only**
  - separator/pause units do not define `g_ref`
- **operator self-fit is holdout-style**
  - prompt diagnostics use fit/eval masks rather than pure in-sample replay
- **main supervision is speech-only**
  - duration / prefix supervision and reporting exclude separator units
- **consistency is default-off**
  - `lambda_rhythm_cons: 0.0`
  - current consistency is diagnostic-only until raw short/long prefix views are introduced
- **trace proxy is not the main training semantics**
  - proxy / trace-only conditioning is fallback/inference-side only, not the mainline training contract

## Current inference contract

At inference time, the mainline is:

1. build prompt evidence once
2. estimate `g_ref`
3. solve prompt operator coefficients `s_ref`
4. as each source unit becomes sealed:
   - compute `b_i`
   - compute `phi_i`
   - predict `log d_hat_i = b_i + g_ref + phi_i^T s_ref`
5. project to integer frames with deterministic residual carry
6. commit immediately through the runtime state / frame plan surface

The mainline does **not** require:

- a reference cursor
- a reference timeline pointer
- runtime prompt retrieval
- pause-side secondary writers

## Public v3 surface

The compact v3 surface is defined by `egs/conan_emformer_rhythm_v3.yaml`.

### Public inputs

- `content_units`
- `dur_anchor_src`
- `unit_anchor_base`
- `prompt_content_units`
- `prompt_duration_obs`
- `prompt_unit_mask`

### Public outputs

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

Optional/internal diagnostics such as `rhythm_v3_cons` and `rhythm_v3_ortho`
may still exist in codepaths and tests, but they are not the compact public
surface committed by the v3 example config.

## Main files

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

### Example config

- `egs/conan_emformer_rhythm_v3.yaml`

## Legacy note

The following remain in the repository as **legacy / compatibility / historical**
surfaces and should not be read as the current rhythm mechanism story:

- `modules/Conan/rhythm/`
- v2 teacher / planner / phrase-bank / pointer-era docs
- old teacher-first stage maps and export handoff procedures

If you need the current mechanism description, use:

- `README.md`
- `docs/rhythm_migration_plan.md`

If you need legacy v2 operational notes only, use:

- `docs/autodl_training_handoff.md`

## Quick validation

```bash
py -3 -B -m pytest -q tests/rhythm/test_rhythm_v3_runtime.py tests/rhythm/test_rhythm_v3_losses.py tests/rhythm/test_rhythm_v3_metrics.py
```

## License

MIT. See [LICENSE](LICENSE).
