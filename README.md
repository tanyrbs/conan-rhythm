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
- retimed cache / F0 side files for `student_retimed` are absent

So formal `teacher_offline -> student_kd -> student_retimed` runs are still blocked by data, not by code.

### Local smoke / probe assets that do exist

This repo does contain lightweight smoke assets that are good enough for structural verification:

- `data/binary/libritts_single_smoke_rhythm_v4`
- `data/processed/libritts_local_real_smoke`

They are for probe/smoke validation only, not for formal maintained training claims.

## Preflight and validation

Run preflight before probe or training. The script accepts both binary and processed path overrides:

```bash
python scripts/preflight_rhythm_v2.py   --config egs/conan_emformer_rhythm_v2_teacher_offline.yaml   --binary_data_dir data/binary/your_dataset   --processed_data_dir data/processed/your_dataset   --splits train valid   --inspect_items 2   --model_dry_run
```

Useful local verification commands:

```bash
python -m compileall -q modules tasks scripts tests utils data_gen
python -m unittest discover -s tests/rhythm -p "test_*.py"
python -u scripts/smoke_test_rhythm_v2.py
python scripts/preflight_rhythm_v2.py --help
```

## Conda `conan` environment validation actually run

The following checks were run in the `conan` environment on April 4, 2026.

### 1) Teacher-offline preflight + model dry-run on smoke assets

```bash
conda run -n conan python scripts/preflight_rhythm_v2.py   --config egs/conan_emformer_rhythm_v2_teacher_offline.yaml   --binary_data_dir data/binary/libritts_single_smoke_rhythm_v4   --processed_data_dir data/processed/libritts_local_real_smoke   --splits train   --inspect_items 2   --model_dry_run
```

Result: **passed**.

### 2) 2000-step CPU mini-train probe

```bash
conda run -n conan python scripts/cpu_probe_rhythm_train.py   --config egs/conan_emformer_rhythm_v2_teacher_offline.yaml   --binary_data_dir data/binary/libritts_single_smoke_rhythm_v4   --processed_data_dir data/processed/libritts_local_real_smoke   --steps 2000   --warmup_steps 10   --device cpu   --profile_json artifacts/probe/teacher_offline_cpu_probe_2000.json
```

Result summary:

- completed **2000/2000** steps
- `total_loss`: `0.3854 -> 0.1044`
- `L_rhythm_exec`: `0.2483 -> 0.0962`
- `L_stream_state`: `0.1371 -> 0.0082`
- mean step time: `240.71 ms`
- throughput: `4.154 steps/s`
- peak CPU RSS: `1159.91 MB`

One real issue showed up during this validation: `scripts/cpu_probe_rhythm_train.py` had an unnecessary `--steps <= 500` hard cap. That cap was lifted to `5000`, so 2000-step verification is now supported directly.

## Training commands

Teacher stage:

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py   --config egs/conan_emformer_rhythm_v2_teacher_offline.yaml   --exp_name conan_rhythm_v2_teacher_offline   --reset
```

Export teacher targets:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/export_rhythm_teacher_targets.py   --config egs/conan_emformer_rhythm_v2_teacher_offline.yaml   --ckpt checkpoints/conan_rhythm_v2_teacher_offline   --output_dir data/teacher_targets/conan_rhythm_v2   --splits train valid
```

Student KD:

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py   --config egs/conan_emformer_rhythm_v2_student_kd.yaml   --exp_name conan_rhythm_v2_teacher_kd   --reset
```

Student retimed:

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py   --config egs/conan_emformer_rhythm_v2_student_retimed.yaml   --exp_name conan_rhythm_v2_retimed   --reset
```

## Config matrix

| Config | Status | Purpose | Needs teacher cache | Needs retimed cache | Needs F0 |
|---|---|---|---:|---:|---:|
| `conan_emformer_rhythm_v2_teacher_offline.yaml` | maintained | learned offline teacher stage | no | no | usually no |
| `conan_emformer_rhythm_v2_student_kd.yaml` | maintained | cache-only teacher-main + shape-only KD | yes | no | no |
| `conan_emformer_rhythm_v2_student_retimed.yaml` | maintained | retimed acoustic closure | yes | yes | yes |
| `conan_emformer_rhythm_v2_minimal_v1.yaml` | maintained | formal minimal base | yes | no | stage-dependent |
| `conan_emformer_rhythm_v2.yaml` | transitional | migration / debug path | optional | optional | stage-dependent |
| `conan_emformer_rhythm_v2_schedule_only.yaml` | legacy | ablation only | optional | no | no |
| `conan_emformer_rhythm_v2_dual_mode_kd.yaml` | legacy | runtime dual-mode teacher research | optional | no | no |

## Inference boundary

Current checked-in inference is best treated as a streaming-oriented evaluation path, not a polished native low-latency production deployment.

See `inference/README.md` for that separate boundary.

If historical notes mention fixed latency numbers or SOTA-style claims, read them as upstream / historical context, not as a fresh claim for this branch state.

## Performance notes

Recent branch work improved the maintained path around:

- shared frame-plan sampling for retimed targets
- cache-contract hardening and fail-fast preflight
- reduced duplicate Python work in the maintained student path
- smaller orchestration entrypoints for preflight/metrics
- removal of a deprecated local TorchScript helper in `modules/Conan/diff/net.py`

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
