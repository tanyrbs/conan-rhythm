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

Primary implementation paths:

- `modules/Conan/rhythm/`
- `tasks/Conan/rhythm/`
- `docs/rhythm_training_stages.md`
- `docs/rhythm_supervision_policy.md`
- `docs/rhythm_train_runbook.md`
- `docs/rhythm_migration_plan.md`

## Quickstart: maintained Rhythm V2 path

Recommended stage order:

1. prepare and verify a real binary cache
2. run `teacher_offline`
3. export learned-offline teacher targets
4. rebuild student-facing cache from exported teacher assets
5. run `student_kd`
6. run `student_retimed`

See:

- `docs/rhythm_train_runbook.md`
- `docs/rhythm_training_stages.md`
- `docs/rhythm_migration_plan.md`

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

## Data preparation

For the maintained Rhythm V2 path, `metadata.json` alone is not enough.

Typical required inputs:

- processed metadata for the Conan VC binarizer path
- `spker_set.json`
- raw wav paths referenced by metadata
- HuBERT token sequences in metadata entries
- F0 side files for stages with `with_f0: true`
- exported teacher target bundles before formal student-stage cache rebuild

### Important cache rule

`egs/conan_emformer_rhythm_v2_cached_only.yaml` is now only a compatibility alias to `egs/conan_emformer_rhythm_v2_minimal_v1.yaml`.

For new maintained work, prefer:

- `egs/conan_emformer_rhythm_v2_minimal_v1.yaml`
- `egs/conan_emformer_rhythm_v2_teacher_offline.yaml`
- `egs/conan_emformer_rhythm_v2_student_kd.yaml`
- `egs/conan_emformer_rhythm_v2_student_retimed.yaml`

### First-pass vs student-pass binarization

There are two different cache moments:

1. **teacher build preparation**
   - prepare the dataset
   - run preflight
   - train `teacher_offline`
2. **student cache rebuild**
   - export learned-offline teacher targets
   - point binarization at `rhythm_teacher_target_dir`
   - rebuild cache so student configs see maintained teacher fields

That second rebuild is required before treating `student_kd` / `student_retimed` as formal runs.

### F0 extraction

Use RMVPE only when the target stage actually needs F0:

```bash
python utils/extract_f0_rmvpe.py \
  --config egs/conan_emformer_rhythm_v2_student_retimed.yaml \
  --batch-size 80
```

### Binarization

```bash
python data_gen/tts/runs/binarize.py \
  --config egs/conan_emformer_rhythm_v2_teacher_offline.yaml
```

After teacher export, rebuild student cache with the student-facing config / overrides that point to exported teacher assets.

## Config matrix

| Config | Status | Purpose | Needs teacher cache | Needs retimed cache | Needs F0 |
|---|---|---|---:|---:|---:|
| `conan_emformer_rhythm_v2_minimal_v1.yaml` | maintained | formal minimal base | yes | no | stage-dependent |
| `conan_emformer_rhythm_v2_teacher_offline.yaml` | maintained | learned offline teacher stage | no | no | usually no |
| `conan_emformer_rhythm_v2_student_kd.yaml` | maintained | cache-only teacher-main + shape-only KD | yes | no | no |
| `conan_emformer_rhythm_v2_student_retimed.yaml` | maintained | retimed acoustic closure | yes | yes | yes |
| `conan_emformer_rhythm_v2_cached_only.yaml` | alias | compatibility alias to `minimal_v1` | same as base | same as base | same as base |
| `conan_emformer_rhythm_v2_schedule_only.yaml` | legacy | warm-start / ablation only | optional | no | no |
| `conan_emformer_rhythm_v2_dual_mode_kd.yaml` | legacy | runtime dual-mode teacher research branch | optional | no | no |
| `conan_emformer_rhythm_v2.yaml` | transitional | migration / debug path | optional | optional | stage-dependent |

## Preflight and validation

Run preflight before probe or training:

```bash
python scripts/preflight_rhythm_v2.py \
  --config egs/conan_emformer_rhythm_v2_teacher_offline.yaml \
  --binary_data_dir data/binary/your_dataset \
  --splits train valid \
  --inspect_items 2 \
  --model_dry_run
```

Useful local verification commands:

```bash
python -m compileall -q modules tasks scripts tests utils data_gen
python -m unittest discover -s tests/rhythm -p "test_*.py"
python -u scripts/smoke_test_rhythm_v2.py
python scripts/preflight_rhythm_v2.py --help
```

CPU mini-train probe:

```bash
python scripts/cpu_probe_rhythm_train.py \
  --config egs/conan_emformer_rhythm_v2_student_kd.yaml \
  --binary_data_dir <REAL_BINARY_DATA_DIR> \
  --processed_data_dir <REAL_PROCESSED_DATA_DIR> \
  --steps 5 \
  --warmup_steps 1 \
  --device cpu
```

Normal `python tasks/run.py ...` launches no longer force `OMP/MKL/...=1`. If you need a low-noise debug run, set `CONAN_SINGLE_THREAD_ENV=1` (or a numeric thread count) explicitly.

## Training

### Maintained path

Teacher stage:

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py \
  --config egs/conan_emformer_rhythm_v2_teacher_offline.yaml \
  --exp_name conan_rhythm_v2_teacher_offline \
  --reset
```

Export teacher targets:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/export_rhythm_teacher_targets.py \
  --config egs/conan_emformer_rhythm_v2_teacher_offline.yaml \
  --ckpt checkpoints/conan_rhythm_v2_teacher_offline \
  --output_dir data/teacher_targets/conan_rhythm_v2 \
  --splits train valid
```

Student KD:

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py \
  --config egs/conan_emformer_rhythm_v2_student_kd.yaml \
  --exp_name conan_rhythm_v2_teacher_kd \
  --reset
```

Student retimed:

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py \
  --config egs/conan_emformer_rhythm_v2_student_retimed.yaml \
  --exp_name conan_rhythm_v2_retimed \
  --reset
```

### Legacy / upstream-style baselines

These configs are kept for compatibility or comparison, not as the maintained branch default:

- `egs/emformer.yaml`
- `egs/conan_emformer.yaml`
- `egs/hifi_16k320_shuffle.yaml`
- `egs/conan_emformer_rhythm_v2_schedule_only.yaml`
- `egs/conan_emformer_rhythm_v2_dual_mode_kd.yaml`

## Inference boundary

Current checked-in inference is best treated as a streaming-oriented evaluation path, not a polished native low-latency production deployment.

See:

- `inference/README.md`
- `docs/streaming_low_latency_mainline_note_20260403.md`

If historical notes mention fixed latency numbers or SOTA-style claims, read them as upstream / historical context, not as a fresh claim for this branch state.

## Performance notes

Recent branch work improved the maintained path around:

- shared frame-plan sampling for retimed targets
- cache-contract hardening and fail-fast preflight
- reduced duplicate Python work in the maintained student path

Current performance headroom is still mostly in:

- data loading / item reuse
- batch transfer efficiency
- binarization throughput
- Windows-vs-Linux preprocessing throughput

## Repository hygiene

- generated `artifacts/` should stay untracked except curated patch files
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
