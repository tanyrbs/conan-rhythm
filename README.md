[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

# Conan Rhythm Branch

This repository maintains a **teacher-first rhythm stack**.

## Mainline in one sentence

The maintained path is:

1. train a strong non-causal offline teacher into a stable, auditable, exportable rhythm oracle;
2. verify those teacher assets through **teacher-conditioned audio closure**;
3. distill the deployment-facing student from those teacher assets;
4. close student acoustic behavior in retimed stage 3.

This is the current engineering mainline of the repo.

## What the current implementation actually is

The maintained control story is:

> **source-side boundary / commit evidence -> explicit speech/pause budget planning -> projector execution authority -> renderer**  
> **reference acts as global / phrase prior**  
> **phase / trace remain diagnostics and compatibility paths**

In concrete terms:

- **source side**: discrete units plus boundary evidence such as `sep_hint`, `open_run_mask`, `sealed_mask`, and `boundary_confidence`
- **planner side**: explicit speech / pause budgeting and unit-level redistribution
- **execution side**: the projector remains the binding execution authority
- **reference side**: reference conditioning is treated as global / phrase prior, not as the main continuous control script
- **diagnostics only**: `phase_ptr`, local trace sampling, trace reliability, cold-start gating, and active-tail sampling remain available for observability and compatibility

## Maintained stage map

| Stage | Meaning | Current implementation |
|---|---|---|
| `T1-surface` | stable offline teacher surface | `egs/conan_emformer_rhythm_v2_teacher_offline.yaml` |
| `T2-audio` | teacher-conditioned audio closure, audit, and export | procedure centered on `scripts/export_rhythm_teacher_targets.py` |
| `S2-kd` | conservative teacher-conditioned student KD | `egs/conan_emformer_rhythm_v2_student_kd.yaml` |
| `S3-retimed` | student acoustic closure | `egs/conan_emformer_rhythm_v2_student_retimed.yaml` |

## Main operator gates

- **Teacher first**: do not hand off to the student because one pause scalar improves; watch the full teacher-control set together.
- **Checkpoint selection**: choose teacher checkpoints by being **exportable, audible, and inheritable**, not by `total_loss` alone and not by pause `F1` alone.
- **Stage 2 role**: let `S2-kd` absorb the remaining streaming/prefix mismatch while keeping the teacher export truth fixed.
- **Stage 3 role**: keep `S3-retimed` cached-first first, then A/B the later `hybrid` route only after teacher and stage-2 handoff are both stable.

For the detailed operator rules, use `docs/autodl_training_handoff.md`.

## Canonical docs

The maintained documentation surface is intentionally reduced to:

1. `README.md`
2. `docs/rhythm_migration_plan.md`
3. `docs/autodl_training_handoff.md`
4. `docs/cloud1_autodl_project_status_snapshot_2026-04-08.md` (**archive only**)

The cloud snapshot is preserved for historical state, but it is not the live
design document for new work.

## Config and script map

| Surface | File |
|---|---|
| maintained teacher stage | `egs/conan_emformer_rhythm_v2_teacher_offline.yaml` |
| maintained student KD stage | `egs/conan_emformer_rhythm_v2_student_kd.yaml` |
| maintained student retimed stage | `egs/conan_emformer_rhythm_v2_student_retimed.yaml` |
| stage preflight | `scripts/preflight_rhythm_v2.py` |
| teacher asset export | `scripts/export_rhythm_teacher_targets.py` |
| teacher -> student smoke integration | `scripts/integration_teacher_export_student_kd.py` |
| maintained smoke | `scripts/smoke_test_rhythm_v2.py` |

## Current checkout status

This checkout is code-ready for the maintained path, but formal runs still
depend on external assets:

- formal processed data and binary caches are not fully checked in
- teacher export assets are not checked in
- `S3-retimed` still requires retimed cache plus matched F0/UV side data
- checked-in smoke assets are for structural verification only, not formal training claims

## Installation

```bash
git clone https://github.com/tanyrbs/conan-rhythm.git
cd conan-rhythm
conda create -n conan python=3.10
conda activate conan
pip install -r requirements.txt
```

## Minimal local verification

```bash
python -m compileall -q modules tasks scripts tests utils data_gen
python -m unittest discover -s tests/rhythm -p "test_*.py"
python -u scripts/smoke_test_rhythm_v2.py
python scripts/preflight_rhythm_v2.py --help
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

## Acknowledgements

- [FastSpeech2](https://github.com/ming024/FastSpeech2)
- [HiFi-GAN](https://github.com/jik876/hifi-gan)
- [Emformer](https://github.com/pytorch/audio)
