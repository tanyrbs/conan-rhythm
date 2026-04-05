[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

# Conan Rhythm Branch

This repository is a Conan-based engineering fork focused on the maintained Rhythm V2 / minimal strong-rhythm path.

> This README describes the current maintained branch behavior, not the original upstream Conan packaging.

## Maintained focus

Current maintained priorities:

- explicit `descriptor -> scheduler -> projector -> renderer` timing chain
- projector execution as the binding timing authority
- teacher-first rhythm training chain:
  - `teacher_offline`
  - `student_kd`
  - `student_retimed`
- cache-backed reproducibility, preflight checks, and retimed closure

Canonical documentation is intentionally reduced to:

1. `README.md`
2. `docs/rhythm_migration_plan.md`

Historical rhythm notes were retired from `docs/` so training guidance now has a single maintained source of truth plus one migration note.

## Quickstart

Recommended maintained stage order:

1. prepare and verify a real binary cache
2. run `teacher_offline`
3. export learned-offline teacher targets
4. rebuild student-facing cache from exported teacher assets
5. run `student_kd`
6. prepare retimed cache + F0 side files
7. run `student_retimed`

Formal student runs are not meaningful until the teacher export and cache rebuild are complete.

## Installation

```bash
git clone https://github.com/tanyrbs/conan-rhythm.git
cd conan-rhythm
conda create -n conan python=3.10
conda activate conan
pip install -r requirements.txt
```

Optional / path-specific dependencies:

- `resemblyzer`: speaker embedding extraction
- `pyworld`: `pe=pw` or RMVPE audio-refinement path
- `pretty_midi`, `mir_eval`: MIDI export / evaluation helpers

## Environment

Preferred workflow assumes the `conan` Conda environment:

```bash
conda activate conan
```

- Python 3.10 is the maintained baseline.
- `pip install -r requirements.txt` covers the maintained rhythm stack.
- Optional extras are only needed for the specific paths that mention them.
- Normal `python tasks/run.py ...` launches no longer force `OMP/MKL/...=1`; if you need a low-noise debug run, set `CONAN_SINGLE_THREAD_ENV=1` explicitly.

## Current checkout status

### Formal training blockers

This shared checkout still does **not** contain the maintained formal dataset assets:

- `data/binary/vc_6layer/{train,valid}.{data,idx}` are missing
- `data/processed/vc` is only a placeholder/example path, not a real processed corpus
- `data/teacher_targets/...` is absent
- dedicated formal retimed cache / F0 side files for `student_retimed` are absent

So formal `teacher_offline -> student_kd -> student_retimed` runs are still blocked by data, not by code.

### Local smoke / probe assets that do exist

This repo does contain lightweight smoke assets that are good enough for structural verification:

- `data/binary/libritts_single_smoke_rhythm_v4`
- `data/processed/libritts_local_real_smoke`
- `artifacts/rhythm_teacher_export_student_kd/<run-id>/...`:
  - bootstrap teacher checkpoint
  - teacher export for `train/valid/test`
  - rebuilt student binary for stage-2 smoke

They are for probe/smoke validation only, not for formal maintained training claims.

Important caveats:

- the `artifacts/rhythm_teacher_export_student_kd/...` teacher checkpoint is `bootstrap_random_init`, so that chain is an integration smoke, not a real learned teacher asset
- `data/binary/libritts_single_smoke_rhythm_v4` is train-only in practice:
  - `valid.data` is empty
  - `test.data` is empty
- the checked-in `libritts_single_smoke_rhythm_v4` items are rhythm-cache **v4 compatibility smoke**, not maintained v5 training assets
- the generated stage-2 `student_binary` smoke assets already contain learned-offline teacher + retimed targets, but they still **do not include F0**, so default `student_retimed` smoke remains blocked when `use_pitch_embed=true`
- retimed mel supervision now refuses to silently reuse source-axis pitch by default; without matched `retimed_f0_tgt` / `retimed_uv_tgt`, pitch supervision is disabled first and fail-fast checks can still escalate during formal stage-3 training

### Runnable now vs blocked now

| Surface | Current smoke checkout status |
|---|---|
| `teacher_offline` preflight / probe | runnable on `data/binary/libritts_single_smoke_rhythm_v4` **train only** |
| `student_kd` preflight / smoke | runnable on `artifacts/rhythm_teacher_export_student_kd/<run-id>/student_binary` |
| `student_retimed` default smoke | blocked on this checkout by missing F0 in the smoke `student_binary` |
| `student_retimed --hparams use_pitch_embed=False` | structurally runnable only; not a formal stage-3 pass |

## Preflight and validation

Run preflight before probe or training. The script accepts both binary and processed path overrides:

```bash
python scripts/preflight_rhythm_v2.py   --config egs/conan_emformer_rhythm_v2_teacher_offline.yaml   --binary_data_dir data/binary/your_dataset   --processed_data_dir data/processed/your_dataset   --splits train valid   --inspect_items 2   --model_dry_run
```

Preflight is intentionally **binary-cache-first**. It validates indexed rhythm/cache fields and stage contracts first. By default `processed_data_dir` is still only lightly checked, but formal readiness runs can now add `--strict_processed_data_dir` to escalate missing or placeholder processed paths into hard errors instead of warnings.

Useful local verification commands:

```bash
python -m compileall -q modules tasks scripts tests utils data_gen
python -m unittest discover -s tests/rhythm -p "test_*.py"
python -u scripts/smoke_test_rhythm_v2.py
python scripts/integration_teacher_export_student_kd.py --help
python scripts/preflight_rhythm_v2.py --help
```

## Conda `conan` environment validation actually run

The following checks were run in the `conan` environment on April 4-5, 2026.

### 1) Compile / unit / maintained smoke checks

```bash
conda run -n conan python -m compileall -q modules tasks scripts tests utils data_gen
conda run -n conan python -m unittest discover -s tests/rhythm -p "test_*.py"
conda run -n conan python -u scripts/smoke_test_rhythm_v2.py
```

Result:

- compileall: **passed**
- rhythm unittests: **190 passed**
- maintained smoke test: **passed**

This latest rerun includes the new coverage added for:

- frame-plan preserved-sum integer rounding
- projector hot-path invariants after vectorization
- context-matched KD gating / dedupe interaction
- conservative EMA group loss balancing
- module-only train/valid objective alignment
- teacher-offline validation using the real offline-teacher branch
- optional dependency guards for lazy text frontend import and no-op tensorboard fallback
- weighted retimed acoustic loss normalization over full broadcasted weight mass
- conservative retimed-pitch guard that disables wrong-axis source-pitch fallback unless explicitly opted in
- validation / smoke metrics now surface runtime flags such as skipped acoustic objectives and missing matched retimed pitch targets

The longer 2000-step probe results below remain the latest recorded long-run measurements for the maintained default configs; the new `context_match` / `balanced` configs are still experimental and were not silently promoted to the maintained baseline.

### 2) Teacher-offline preflight + model dry-run on smoke assets

```bash
conda run -n conan python scripts/preflight_rhythm_v2.py   --config egs/conan_emformer_rhythm_v2_teacher_offline.yaml   --binary_data_dir data/binary/libritts_single_smoke_rhythm_v4   --processed_data_dir data/processed/libritts_local_real_smoke   --splits train   --inspect_items 2   --model_dry_run
```

Result: **passed on `train`**.

Counter-check:

```bash
conda run -n conan python scripts/preflight_rhythm_v2.py   --config egs/conan_emformer_rhythm_v2_teacher_offline.yaml   --binary_data_dir data/binary/libritts_single_smoke_rhythm_v4   --processed_data_dir data/processed/libritts_local_real_smoke   --splits train valid   --inspect_items 2   --model_dry_run
```

Result: **fails on `valid`** because `data/binary/libritts_single_smoke_rhythm_v4/valid.data` is empty. That is a smoke-asset defect, not a stage-contract or model-logic failure.

### 3) Teacher export -> student KD integration smoke

```bash
conda run -n conan python scripts/integration_teacher_export_student_kd.py   --teacher_config egs/conan_emformer_rhythm_v2_teacher_offline.yaml   --student_config egs/conan_emformer_rhythm_v2_student_kd.yaml   --processed_data_dir data/processed/libritts_local_real_smoke
```

Result: **passed**.

Integration notes:

- export coverage now includes `train/valid/test`
- summary artifact is written under `artifacts/rhythm_teacher_export_student_kd/<run-id>/summary.json`
- this is still smoke-only because the teacher ckpt mode is `bootstrap_random_init`
- on the checked-in LibriTTS smoke corpus, split inference relies on the generated `build_summary.json`

### 4) 2000-step CPU probe: `teacher_offline`

```bash
conda run -n conan python scripts/cpu_probe_rhythm_train.py   --config egs/conan_emformer_rhythm_v2_teacher_offline.yaml   --binary_data_dir data/binary/libritts_single_smoke_rhythm_v4   --processed_data_dir data/processed/libritts_local_real_smoke   --steps 2000   --warmup_steps 10   --device cpu   --profile_json artifacts/probe/teacher_offline_cpu_probe_2000_clean.json
```

Result summary:

- completed **2000/2000** steps
- `total_loss`: `0.3854 -> 0.1044`
- `L_rhythm_exec`: `0.2483 -> 0.0962`
- `L_stream_state`: `0.1371 -> 0.0082`
- mean step time: `240.71 ms`
- throughput: `4.154 steps/s`
- peak CPU RSS: `1159.91 MB`

### 5) 2000-step CPU probe: `student_kd`

```bash
conda run -n conan python scripts/cpu_probe_rhythm_train.py   --config egs/conan_emformer_rhythm_v2_student_kd.yaml   --binary_data_dir artifacts/rhythm_teacher_export_student_kd/5e1bc8ca5f/student_binary   --processed_data_dir data/processed/libritts_local_real_smoke   --steps 2000   --warmup_steps 10   --device cpu   --profile_json artifacts/probe/student_kd_cpu_probe_2000_default_chain.json
```

Result summary:

- completed **2000/2000** steps
- `total_loss`: `0.0154 -> 0.0111`
- `L_rhythm_exec`: `0.0070 -> 0.0069`
- `L_stream_state`: `0.0075 -> 0.00018`
- mean step time: `46.53 ms`
- throughput: about `21.49 steps/s`
- peak CPU RSS: about `977.65 MB`

### 6) 2000-step CPU probe: `student_retimed` smoke

```bash
conda run -n conan python scripts/cpu_probe_rhythm_train.py   --config egs/conan_emformer_rhythm_v2_student_retimed.yaml   --binary_data_dir artifacts/rhythm_teacher_export_student_kd/5e1bc8ca5f/student_binary   --processed_data_dir data/processed/libritts_local_real_smoke   --hparams use_pitch_embed=False   --steps 2000   --warmup_steps 10   --device cpu   --profile_json artifacts/probe/student_retimed_cpu_probe_2000_smoke.json
```

Result summary:

- completed **2000/2000** steps
- `total_loss`: `48.41 -> 2.67`
- `L_base` dominates the objective
- `grad_norm_before_clip`: mean about `158.86`, max about `275.92`
- this is a structural smoke pass only, **not** a clean stage-3 readiness signal

## Training commands

Teacher stage template:

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py   --config egs/conan_emformer_rhythm_v2_teacher_offline.yaml   --exp_name conan_rhythm_v2_teacher_offline   --reset
```

For real runs, also override the actual dataset roots, for example:

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py   --config egs/conan_emformer_rhythm_v2_teacher_offline.yaml   --exp_name conan_rhythm_v2_teacher_offline   --reset   -hp "binary_data_dir='data/binary/your_dataset',processed_data_dir='data/processed/your_dataset'"
```

Export teacher targets template:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/export_rhythm_teacher_targets.py   --config egs/conan_emformer_rhythm_v2_teacher_offline.yaml   --ckpt checkpoints/conan_rhythm_v2_teacher_offline   --output_dir data/teacher_targets/conan_rhythm_v2   --splits train valid test
```

Student KD template:

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py   --config egs/conan_emformer_rhythm_v2_student_kd.yaml   --exp_name conan_rhythm_v2_teacher_kd   --reset
```

Example with required overrides:

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py   --config egs/conan_emformer_rhythm_v2_student_kd.yaml   --exp_name conan_rhythm_v2_teacher_kd   --reset   -hp "binary_data_dir='data/binary/your_stage2_binary',processed_data_dir='data/processed/your_dataset',rhythm_teacher_target_dir='data/teacher_targets/conan_rhythm_v2'"
```

Student retimed template:

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py   --config egs/conan_emformer_rhythm_v2_student_retimed.yaml   --exp_name conan_rhythm_v2_retimed   --reset
```

Real stage-3 runs additionally require a binary cache that already contains retimed targets plus valid F0 side data. The checked-in smoke `student_binary` does not satisfy that default `use_pitch_embed=true` contract.

## Config matrix

| Config | Status | Purpose | Needs teacher cache | Needs retimed cache | Needs F0 |
|---|---|---|---:|---:|---:|
| `conan_emformer_rhythm_v2_teacher_offline.yaml` | maintained | learned offline teacher stage | no | no | usually no |
| `conan_emformer_rhythm_v2_student_kd.yaml` | maintained | student runtime + cached teacher supervision; maintained path usually keeps KD shape-only | yes | no | no |
| `conan_emformer_rhythm_v2_student_kd_context_match.yaml` | experimental | prefix-truncated stage-2 branch with context-matched KD gate + conservative EMA loss balance | yes | no | no |
| `conan_emformer_rhythm_v2_student_retimed.yaml` | maintained | retimed acoustic closure | yes | yes | yes |
| `conan_emformer_rhythm_v2_student_retimed_balanced.yaml` | experimental | stage-3 retimed branch with conservative EMA group loss balancing | yes | yes | yes |
| `conan_emformer_rhythm_v2_minimal_v1.yaml` | maintained | minimal maintained profile / contract baseline | yes | no | stage-dependent |
| `conan_emformer_rhythm_v2.yaml` | transitional | migration / debug path | optional | optional | stage-dependent |
| `conan_emformer_rhythm_v2_schedule_only.yaml` | legacy | schedule-only ablation; not part of maintained training prep | optional | no | no |
| `conan_emformer_rhythm_v2_dual_mode_kd.yaml` | legacy | runtime dual-mode teacher research / ablation | optional | no | no |

## Training-prep conclusion

Current branch conclusion after code review, probes, and smoke integration:

- `teacher_offline`: **code-ready**, but the checked-in smoke `valid` split is defective
- `student_kd`: **structurally ready** once a real trained teacher export exists for all required splits
- `student_retimed`: **not formally ready to bless** from this checkout; the current smoke student binary is missing F0, so default stage-3 smoke fails unless pitch embed is disabled, and even then gradient pressure stays high

So the next formal sequence should be:

1. train a real `teacher_offline` checkpoint
2. export teacher targets for `train/valid/test`
3. rebuild stage-2 binary from that export
4. validate `student_kd`
5. prepare dedicated retimed cache + F0 side files
6. re-check `student_retimed` with real assets before long training

Experimental notes:

- `conan_emformer_rhythm_v2_student_kd_context_match.yaml` is an opt-in stage-2 research branch, not the maintained default
- `conan_emformer_rhythm_v2_student_retimed_balanced.yaml` is an opt-in stage-3 A/B config, not a blessed replacement for `student_retimed`
- maintained readiness claims in this README still refer to the default `teacher_offline -> student_kd -> student_retimed` chain

## Inference boundary

Current checked-in inference is best treated as a streaming-oriented evaluation path, not a polished native low-latency production deployment.

See `inference/README.md` for that separate boundary.

If historical notes mention fixed latency numbers or SOTA-style claims, read them as upstream / historical context, not as a fresh claim for this branch state.

Also note that the legacy `inference/run_voice_conversion*.py` runners are not maintained branch-quality entrypoints for this rhythm fork; use the latency-report helper and training/preflight scripts as the maintained surface.

## Performance notes

Recent branch work improved the maintained path around:

- sum-preserving frame-plan integerization for speech / blank groups and `dur_anchor_src`
- cache-contract hardening and fail-fast preflight
- train/valid objective alignment for module-only stages such as `teacher_offline` and `student_kd`
- correct teacher-offline validation routing instead of silently falling back to the student/runtime branch
- reduced duplicate Python work in the maintained student path
- vectorized projector outer hot paths such as pause projection, commit-frontier resolution, and state advance
- smaller orchestration entrypoints for preflight/metrics
- removal of a deprecated local TorchScript helper in `modules/Conan/diff/net.py`
- weighted retimed acoustic losses now normalize by full broadcasted weight mass instead of frame count only
- retimed mel supervision now blocks accidental source-axis pitch fallback unless explicitly allowed for debugging
- rhythm-only imports no longer require the English text frontend or tensorboard at import time

Recent branch work also added opt-in experimental surfaces:

- bounded `ema_group` rhythm-loss balancing (`rhythm_loss_balance_mode: none` remains the default)
- context-matched KD gating for prefix-truncated stage-2 experiments (`rhythm_enable_distill_context_match: false` remains the default)

Current performance headroom is still mostly in:

- data loading / item reuse
- batch transfer efficiency
- binarization throughput
- Windows-vs-Linux preprocessing throughput

## Repository hygiene

- generated `artifacts/` should stay untracked except curated patch/probe summaries that you explicitly want to keep
- `.gitattributes` pins text files to LF to reduce Windows CRLF noise
- the rhythm CI lane should cover both Ubuntu and Windows

## Citation

If you use the upstream Conan work, cite the original paper/reference:

```bibtex
@article{zhang2025conan,
  title={Conan: A Chunkwise Online Network for Zero-Shot Adaptive Voice Conversion},
  author={Zhang, Yu and Tian, Baotong and Duan, Zhiyao},
  journal={arXiv preprint arXiv:2507.14534},
  year={2025}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

## Acknowledgements

- [FastSpeech2](https://github.com/ming024/FastSpeech2)
- [HiFi-GAN](https://github.com/jik876/hifi-gan)
- [Emformer](https://github.com/pytorch/audio)
