
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.51+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository is **our Conan-based engineering fork / modified branch**, focused on the current Rhythm V2 / Minimal Strong-Rhythm path.

> **Repository scope**
>
> - this is **not** presented here as the official Conan release
> - this README documents **our local redesign and maintained training path**, not the original paper packaging
> - if historical Conan descriptions disagree with the current code, **the current code and docs in this repo take precedence**
>
> Current maintained focus:
>
> - explicit reference rhythm conditioning via `RefRhythmDescriptor`
> - explicit `descriptor -> scheduler -> projector -> renderer` timing chain
> - projector execution as the only binding timing authority
> - teacher-first / student-only rhythm training chain:
>   - `teacher_offline`
>   - `student_kd`
>   - `student_retimed`
> - cache-backed reproducibility and preflight checks
>
> Key paths:
>
> - `modules/Conan/rhythm/`
> - `tasks/Conan/rhythm/`
> - `docs/rhythm_module_vision.md`
> - `docs/rhythm_migration_plan.md`
> - `docs/rhythm_repo_audit_2026-04-04.md`
> - `docs/rhythm_training_stages.md`
> - `docs/rhythm_supervision_policy.md`
>
> Current note (2026-04-04):
>
> - the repo has been heavily reworked around explicit rhythm planning, projector feasibility, cache-backed KD, and retimed training closure
> - several legacy Conan / schedule-only / dual-mode branches are kept only for compatibility or research migration
> - maintained `student_kd` is now explicitly **teacher-main + shape-only KD**, and its cache contract only requires teacher core + shape-confidence sidecars by default
> - stage-3 retimed sampling has been tightened around the shared frame-plan path, including batched frame-plan gathering to reduce Python-loop overhead in retimed target construction
> - README content below should be read as **project-specific implementation notes for this modified branch**, not as an official Conan introduction

## Requirements

### System Requirements
- Python 3.10+

## 🚀 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/tanyrbs/conan-rhythm.git
cd conan-rhythm
```

2. **Create a virtual environment**:
```bash
conda create -n conan python=3.10
conda activate conan
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```
## 📊 Data Preparation

### Dataset Structure
For the current Conan / Rhythm V2 path, `metadata.json` alone is **not** enough.

At minimum, prepare:

- `metadata_vctk_librittsr_gt.json` for the current VC binarizer path
- `spker_set.json`
- raw wav paths referenced by metadata
- HuBERT token sequences in metadata entries
- RMVPE F0 files for main-model training

If rhythm cache generation is enabled, the binarizer can additionally cache:

- source unit cache: `content_units`, `dur_anchor_src`, `open_run_mask`, `sealed_mask`, `boundary_confidence`
- source phrase cache: `source_boundary_cue`, `phrase_group_index`, `phrase_group_pos`, `phrase_final_mask`
- reference rhythm cache: `ref_rhythm_stats`, `ref_rhythm_trace`, `slow_rhythm_memory`, `slow_rhythm_summary`
- selector metadata: `selector_meta_indices`, `selector_meta_scores`, `selector_meta_starts`, `selector_meta_ends`
- cached guidance / teacher targets; note that maintained stage-2 shape-only KD only needs teacher core targets + teacher shape confidence by default, while allocation / prefix sidecars remain research/optional
- cached retimed mel targets

Keep the cache / batch schema layered:

- runtime-minimal contract: `content_units`, `dur_anchor_src`, `ref_rhythm_stats`, `ref_rhythm_trace`
- runtime-target contract: executed speech/pause targets, light budget targets, stage-needed confidence / teacher / retimed targets
- streaming/offline sidecars: offline source-cache views plus streaming prefix counters, only when dual-mode / prefix sampling actually needs them
- debug/cache appendix: source phrase cues, selector spans, cache version, hop/trace contract, retimed source metadata

Example:
```text
data/
`-- processed/
    |-- metadata_vctk_librittsr_gt.json
    `-- spker_set.json
```
### Metadata Format
There is an example "example_metadata.json" file in the `data/processed/vc/` directory.
The metadata file should contain entries like:
```json
[
  {
    "item_name": "speaker1_audio1",
    "wav_fn": "data/raw/speaker1/audio1.wav", // Path to the raw audio file
    "spk_embed": "0.1 0.2 0.3 ...", // Speaker embedding vector
    "duration": 3.5, // Duration in seconds
    "hubert": "12 34 56 ..." // HuBERT features as space-separated string
  }
]
```

### Data Preprocessing Steps

1. **Extract F0 features using RMVPE (needed only for main model training)**:
```bash
export PYTHONPATH=/storage/baotong/workspace/Conan:$PYTHONPATH # (optional) you may need to set the PYTHONPATH for import dependencies
python utils/extract_f0_rmvpe.py \
    --config egs/conan_emformer_rhythm_v2_cached_only.yaml \
    --batch-size 80 \
    --save-dir /path/to/audio  
```
F0 will be saved to the same level folder as the audio folder.
File structure: (an example below)
```data/
└── audio/
    ├── p225_001.wav
    ├── ...
└── audio_f0/
    ├── p225_001.npy
    ├── ...
```
2. **Binarize the dataset**:
```bash
python data_gen/tts/runs/binarize.py --config egs/conan_emformer_rhythm_v2_cached_only.yaml
```

For formal Rhythm V2 experiments:

- use the cached-only config for binarization
- prefer the maintained `minimal_v1` base when starting a new formal training chain
- train/export the offline planner teacher first when you want `learned_offline` teacher assets
- keep `rhythm_binarize_teacher_targets: true`
- set `rhythm_teacher_target_source: learned_offline` and provide precomputed offline teacher bundles when building formal teacher-backed caches
- use `scripts/export_rhythm_teacher_targets.py` to write `{split}/{item_name}.teacher.npz` assets into `rhythm_teacher_target_dir` by default (`--flat_output` keeps the old flat layout; `scripts/export_offline_teacher_assets.py` remains a compatibility wrapper)
- treat teacher surfaces as the preferred **target source**; maintained `student_kd` now uses teacher-main supervision plus a small shape-only KD regularizer
- re-binarize whenever `rhythm_cache_version` changes
- treat `prefer_cache` only as a migration/debug mode

For formal Rhythm V2 experiments, binarization should be re-run with rhythm cache enabled so training can use offline cached targets instead of runtime heuristics.
### Configuration
Update the configuration files in `egs/` directory to match your dataset:
- `egs/conan_emformer_rhythm_v2.yaml`: transitional rhythm config (`prefer_cache`)
- `egs/conan_emformer_rhythm_v2_minimal_v1.yaml`: maintained formal base config
- `egs/conan_emformer_rhythm_v2_cached_only.yaml`: legacy alias to the maintained formal base
- `egs/conan_emformer_rhythm_v2_teacher_offline.yaml`: maintained offline teacher asset-build stage
- `egs/conan_emformer_rhythm_v2_schedule_only.yaml`: legacy schedule warm-start / ablation config
- `egs/conan_emformer_rhythm_v2_student_kd.yaml`: maintained stage-2 cache-only teacher-main supervision + small shape-only KD regularization
- `egs/conan_emformer_rhythm_v2_dual_mode_kd.yaml`: legacy stage-2 runtime dual-mode KD branch
- `egs/conan_emformer_rhythm_v2_student_retimed.yaml`: maintained retimed acoustic training stage
- `egs/conan_emformer.yaml`: legacy / baseline main training configuration
- `egs/emformer.yaml`: Emformer training configuration
- `egs/hifi_16k320_shuffle.yaml`: Vocoder training configuration

Key parameters to adjust:
```yaml
# Dataset paths
binary_data_dir: '<YOUR_REAL_BINARY_DATA_DIR>'
processed_data_dir: '<YOUR_REAL_PROCESSED_DATA_DIR>'
```
## 🎯 Training

### Stage 1: Train Emformer
We first prepare the data and HuBERT tokens from the s3prl package using ```s3prl.nn.S3PRLUpstream("hubert")```.
```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py \
    --config egs/emformer.yaml \
    --exp_name emformer_training \
    --reset
```

### Stage 2: Train Main Conan Model
We fix the Emformer and Vocoder components, and prepare hubert entries of the data by applying the trained Emformer through the datasets (extracted chunk-wise).
```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py \
    --config egs/conan_emformer.yaml \
    --exp_name conan_training \
    --reset
```

### Stage 3: Train HiFi-GAN Vocoder
```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py \
    --config egs/hifi_16k320_shuffle.yaml \
    --exp_name hifigan_training \
    --reset
```

### Rhythm V2 warm-start / retimed training

Before starting any formal Rhythm V2 run, do a cache/config preflight:
```bash
python scripts/preflight_rhythm_v2.py \
    --config egs/conan_emformer_rhythm_v2_teacher_offline.yaml \
    --binary_data_dir data/binary/your_dataset \
    --model_dry_run
```

Formal expectation:

- the checked-in default path `data/binary/vc_6layer` is **not** bundled in a clean checkout; override `binary_data_dir` to a real local dataset before treating preflight as actionable
- `train` and `valid` must both pass raw cache inspection **and** survive `ConanDataset` filtering
- preflight now also checks split-array consistency (`_lengths.npy` / `_spk_ids.npy` when present) and inspected-item shape contracts for cached rhythm/reference/retimed fields
- repeat preflight with the exact config for each stage
- the bundled smoke cache is only for structural sanity checks; if its `valid` split is intentionally filtered empty, use `--splits train` for smoke-only checks, but do not treat that as formal training readiness

After stage-1 teacher export and student-cache rebuild, rerun preflight on:

- `egs/conan_emformer_rhythm_v2_student_kd.yaml`
- `egs/conan_emformer_rhythm_v2_student_retimed.yaml`

Before a real long run, you can also do a CPU mini-train probe on the same config:
```bash
python scripts/cpu_probe_rhythm_train.py \
    --config egs/conan_emformer_rhythm_v2_student_retimed.yaml \
    --binary_data_dir data/binary/libritts_single_smoke_rhythm_v4 \
    --processed_data_dir data/processed/libritts_single \
    --steps 20
```

What this probe checks:

- real dataset collation + model forward/backward on CPU
- loss decomposition (`L_base`, `L_rhythm_exec`, `L_stream_state`, `L_pitch`)
- gradient finiteness and clipped gradient norm
- whether representative parameters actually move

Recommended formal path:

0. `teacher_offline` + export `learned_offline` teacher assets
1. rebuild student-facing cached teacher surfaces
2. `student_kd` (teacher-main supervision + small shape-only KD)
3. `student_retimed`

Optional legacy ablations:

- `legacy_schedule_only`
- `legacy_dual_mode_kd`

Optional branch:

- use `legacy_dual_mode_kd` only when explicitly running legacy runtime-teacher experiments

Transitional warm-start:
```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py \
    --config egs/conan_emformer_rhythm_v2.yaml \
    --exp_name conan_rhythm_v2 \
    --reset
```

Strict cached-only warm-start:
```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py \
    --config egs/conan_emformer_rhythm_v2_cached_only.yaml \
    --exp_name conan_rhythm_v2_cached \
    --reset
```

Offline teacher asset build:
```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py \
    --config egs/conan_emformer_rhythm_v2_teacher_offline.yaml \
    --exp_name conan_rhythm_v2_teacher_offline \
    --reset
```

Export learned offline teacher assets:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/export_rhythm_teacher_targets.py \
    --config egs/conan_emformer_rhythm_v2_teacher_offline.yaml \
    --ckpt checkpoints/conan_rhythm_v2_teacher_offline \
    --output_dir data/teacher_targets/conan_rhythm_v2 \
    --splits train valid
```

Legacy schedule-only warm-start (optional ablation):
```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py \
    --config egs/conan_emformer_rhythm_v2_schedule_only.yaml \
    --exp_name conan_rhythm_v2_sched \
    --reset
```

Formal stage-2 cache-only teacher->student KD:
```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py \
    --config egs/conan_emformer_rhythm_v2_student_kd.yaml \
    --exp_name conan_rhythm_v2_teacher_kd \
    --reset
```

Legacy stage-2 dual-mode schedule KD:
```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py \
    --config egs/conan_emformer_rhythm_v2_dual_mode_kd.yaml \
    --exp_name conan_rhythm_v2_dual_kd \
    --reset
```

Strict cached-only student retimed-train experiment:
```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py \
    --config egs/conan_emformer_rhythm_v2_student_retimed.yaml \
    --exp_name conan_rhythm_v2_retimed \
    --reset
```

## 🔮 Inference

### Streaming Voice Conversion
```bash
CUDA_VISIBLE_DEVICES=0 python inference/Conan.py \
    --config egs/conan_emformer_rhythm_v2.yaml \
    --exp_name conan
```
Use the exp_name that contains the trained main model checkpoints, and update your config with the trained Emformer checkpoint and HifiGAN checkpoint.

## Checkpoints
You can download pre-trained model checkpoints from [Google Drive](https://drive.google.com/drive/folders/1QhnECo2L4xfXDgdrnM6L1xpsH7u3iRvj?usp=sharing).

Main system checkpoint folders: Emformer, Conan, hifigan_vc

Fast system checkpoint folders: Emformer_fast, Conan_fast, hifigan_vc (you may need to change the "right_context" in the config file to 0 instead of 2)

Note: As we previous developed the Emformer training branch on another codebase, we provided another inference script for it `inference/Conan_previous.py`.
## 📁 Project Structure

```
Conan/
├── modules/                    # Core model implementations
│   ├── Conan/                 # Main Conan model
│   ├── Emformer/              # Emformer feature extractor
│   ├── vocoder/               # HiFi-GAN vocoder
│   └── ...
├── tasks/                     # Training and evaluation tasks
│   ├── Conan/                 # Conan training task
│   └── ...
├── inference/                 # Inference scripts
│   ├── Conan.py              # Main inference script
│   ├── run_voice_conversion.py
│   └── ...
├── data_gen/                  # Data preprocessing
│   ├── conan_binarizer.py    # Data binarization
│   └── ...
├── egs/                       # Configuration files
│   ├── conan_emformer_rhythm_v2.yaml
│   ├── conan_emformer_rhythm_v2_cached_only.yaml
│   ├── conan_emformer_rhythm_v2_teacher_offline.yaml
│   ├── conan_emformer_rhythm_v2_schedule_only.yaml
│   ├── conan_emformer_rhythm_v2_student_kd.yaml
│   ├── conan_emformer_rhythm_v2_student_retimed.yaml
│   ├── conan_emformer_rhythm_v2_dual_mode_kd.yaml
│   ├── emformer.yaml         # Emformer config
│   └── ...
├── utils/                     # Utility functions
└── checkpoints/              # Model checkpoints
```
## 📈 Performance

The Conan system achieves state-of-the-art performance on voice conversion tasks:

- **Latency**: ~80ms streaming latency (37ms latency for fast system)
- **Quality**: High-quality voice conversion with natural prosody
- **Robustness**: Robust to different speaking styles and content

## 📄 Citation

If you use Conan in your research, please cite our work:

```bibtex
@article{zhang2025conan,
  title={Conan: A Chunkwise Online Network for Zero-Shot Adaptive Voice Conversion},
  author={Zhang, Yu and Tian, Baotong and Duan, Zhiyao},
  journal={arXiv preprint arXiv:2507.14534},
  year={2025}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [FastSpeech2](https://github.com/ming024/FastSpeech2) for the codebase and base TTS architectures
- [HiFi-GAN](https://github.com/jik876/hifi-gan) for the neural vocoder
- [Emformer](https://github.com/pytorch/audio) for efficient transformer implementation
