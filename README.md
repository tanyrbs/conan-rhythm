[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

# Conan Rhythm Branch

This checkout now has **one maintained rhythm code mainline**:
`rhythm_v3`.

The honest reading of the current repository is:

- **task scope**: speech-unit duration transfer
- **research shape**: layered duration decomposition / decision platform
- **current default config**: `baseline + global + deterministic projector`
- **legacy v2**: compatibility, teacher/export history, old checkpoints only

## Current code-level prediction form

The exact maintained v3 execution surface is:

> `log d_hat_i = log b_i + g_ref + w_prog(i) + w_det(i) + 1_local * phi_i^T s_ref + 1_srcres * r_src(i)`

where:

- `b_i`: nominal duration baseline in frames
- `g_ref`: speech-only prompt-global stretch
- `w_prog(i)`: optional progress-indexed warp candidate
- `w_det(i)`: optional fixed detector-bank candidate
- `phi_i^T s_ref`: optional prompt-conditioned local operator response
- `r_src(i)`: optional centered source-residual term

In code:

- `b_i` is `unit_anchor_base`
- the predicted stretch is `unit_logstretch`
- executed duration is `unit_anchor_base * exp(unit_logstretch)`

So the repository is **already narrower than the old v2 multi-controller stack**,
but it is **not yet reduced to a pure five-piece operator-only theorem**. The
current mainline still keeps progress/detector/local adjudication as a first-class decision.

## What the project currently is

The best description of the current branch is:

> a **rhythm_v3 duration-mechanism platform** for testing whether target
> duration style mostly lives in:
>
> 1. baseline only  
> 2. baseline + speech-only global stretch  
> 3. baseline + global + progress warp  
> 4. baseline + global + detector bank  
> 5. baseline + global + local operator  
> 6. baseline + global + progress + local operator

This is important: the code should be read as a **mechanism-candidacy and
ablation platform**, not as proof that shared local bases are already the final
answer.

## The maintained baseline protocol

The baseline is no longer just "a small network". In current code it is a
protocol object:

- `modules/Conan/rhythm_v3/unit_frontend.py`

It now contains:

1. **optional frozen table prior**
2. **strict causal local trunk**

Slow structure is **not** part of the maintained baseline contract.
If modeled, it belongs to an explicit progress/detector candidate layer, not the baseline.

Migration note: do **not** mix old precomputed anchor/log-base artifacts with
the new baseline contract. Progress warp is baseline-referenced, so
stale baseline caches can create calibration drift.

The intended semantics are:

- speaker-free
- style-neutral
- content-conditioned
- no prompt / no runtime progress / no boundary controller / no pause writer

Current lifecycle knobs:

- `rhythm_v3_baseline_train_mode: joint | frozen | pretrain`
- `rhythm_v3_freeze_baseline`
- `rhythm_v3_baseline_ckpt`
- `rhythm_baseline_table_prior_path`
- `rhythm_v3_baseline_target_mode: raw | deglobalized`
- `lambda_rhythm_base`

Current code now has a **baseline-pretrain scaffold**, but not a full finished
EM-like recipe. The implemented pretrain reading is:

- freeze optimization to baseline parameters only
- supervise baseline against either raw or deglobalized speech-unit targets
- allow baseline-only training without prompt conditioning
- keep the stronger alternating/de-style schedule as future work

### Important invariant

Downstream prompt/operator paths use **stop-gradient baseline features**.
Also, v3 prefix/consistency training now rebuilds speech duration from:

- `sg(unit_anchor_base) * exp(unit_logstretch)`

instead of supervising through the live execution-side baseline path.

## Current maintained modeling pieces

### Always maintained

1. **baseline protocol**
2. **speech-only global stretch**
3. **deterministic projector**

### Maintained candidate response layers

4. **progress warp candidate**
5. **detector-bank candidate**
6. **shared causal local basis + prompt-conditioned operator**
7. **optional centered source residual ablation**

Only the first three are unconditional. The rest are controlled by
`rhythm_v3_backbone` / `rhythm_v3_warp_mode` / `rhythm_v3_allow_hybrid`.

For the progress-warp candidate, the preferred config keys are now:

- `rhythm_progress_bins`
- `rhythm_progress_support_tau`

Legacy `rhythm_coarse_bins` / `rhythm_coarse_support_tau` remain accepted as
compatibility aliases.

## What is explicitly not the mainline anymore

The current v3 story is **not**:

- slot memory
- role codebooks
- runtime reference pointer traversal
- planner/pause/boundary controllers as the main duration writers
- teacher-first v2 stage menus as the current architecture
- trace proxy as the training semantics

Separator / pause duration is **not** the core mechanism claim. The branch
scope is speech-unit duration transfer.

## Current training contract

The current implementation enforces:

- **explicit prompt units for mainline training**
  - `prompt_content_units`
  - `prompt_duration_obs`
  - `prompt_unit_mask`
- **speech-only global estimation**
  - separator units do not define `g_ref`
- **speech-only source supervision**
  - separator units are excluded from duration/prefix supervision and metrics
- **holdout prompt self-fit diagnostics**
  - fit/eval masks are present for operator diagnostics
- **consistency default-off**
  - `lambda_rhythm_cons: 0.0`
- **no v3 mel-proxy reference path**
  - v3 now requires explicit prompt units or prebuilt duration memory

## Current public v3 surface

The compact public contract is defined by:

- `egs/conan_emformer_rhythm_v3.yaml`

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

Optional/internal diagnostics like `rhythm_v3_cons` and `rhythm_v3_ortho`
exist in code/tests but are not the compact public example surface.

## Main files

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

### Inference/runtime helpers

- `inference/Conan.py`
- `inference/run_voice_conversion.py`
- `inference/run_streaming_latency_report.py`

## Readiness snapshot

### What is ready now

- a single `rhythm_v3` code mainline
- baseline protocol with lifecycle control
- stop-gradient baseline separation through operator and stream-side losses
- compact v3 config/validator surface
- focused tests around v3 runtime, losses, config, metrics, optimizer collection, and baseline frontend

### What is not yet "paper-claimed ready"

- the project has **not yet proven** that shared local bases are necessary
- baseline pretrain / de-style training protocol is not yet a fully separate stage
- progress/detector/local mechanism adjudication is still an active research question
- inference remains streaming-oriented evaluation, not final end-to-end fully stateful low-latency deployment

## Legacy note

The following remain only as **legacy / compatibility / archive** surfaces:

- `modules/Conan/rhythm/`
- `inference/Conan_previous.py`
- old v2 teacher / planner / pointer / pause-controller docs and configs

Authoritative current docs are:

- `README.md`
- `docs/rhythm_migration_plan.md`
- `inference/README.md`

Legacy operational notes only:

- `docs/autodl_training_handoff.md`

## Quick validation

```bash
py -3 -B -m pytest -q ^
  tests/rhythm/test_task_config_v3.py ^
  tests/rhythm/test_conan_task_build_tts_model_v3.py ^
  tests/rhythm/test_optimizer_param_collection.py ^
  tests/rhythm/test_rhythm_v3_losses.py ^
  tests/rhythm/test_rhythm_v3_runtime.py ^
  tests/rhythm/test_rhythm_v3_metrics.py ^
  tests/rhythm/test_inference_entrypoints.py
```

## License

MIT. See [LICENSE](LICENSE).
