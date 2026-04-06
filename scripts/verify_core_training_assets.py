from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = Path('/root/autodl-tmp/data/LibriTTS')


def _status(path: Path) -> dict:
    return {"path": str(path), "exists": path.exists(), "is_dir": path.is_dir()}


def build_report(raw_root: Path) -> dict:
    configs = {
        "stage1_train100": REPO_ROOT / 'egs/conan_emformer_rhythm_v2_teacher_offline_train100.yaml',
        "stage1_train100_360": REPO_ROOT / 'egs/conan_emformer_rhythm_v2_teacher_offline_train100_360.yaml',
        "stage2_student_kd": REPO_ROOT / 'egs/conan_emformer_rhythm_v2_student_kd.yaml',
        "stage3_student_retimed": REPO_ROOT / 'egs/conan_emformer_rhythm_v2_student_retimed.yaml',
    }
    scripts = {
        "prepare_train100": REPO_ROOT / 'scripts/autodl_prepare_train100_formal.sh',
        "prepare_train100_360": REPO_ROOT / 'scripts/autodl_prepare_train100_360_formal.sh',
        "train_stage1": REPO_ROOT / 'scripts/autodl_train_stage1.sh',
    }
    checkpoints = {
        "conan": REPO_ROOT / 'checkpoints/Conan/model_ckpt_steps_200000.ckpt',
        "emformer": REPO_ROOT / 'checkpoints/Emformer/model_ckpt_steps_700000.ckpt',
        "rmvpe": REPO_ROOT / 'checkpoints/rmvpe/rmvpe.pt',
        "hifigan": REPO_ROOT / 'checkpoints/hifigan_vc/model_ckpt_steps_1000000.ckpt',
    }
    raw_splits = {
        name: raw_root / name
        for name in ('train-clean-100', 'train-clean-360', 'dev-clean', 'test-clean')
    }
    outputs = {
        'train100_processed': REPO_ROOT / 'data/processed/libritts_train100_formal',
        'train100_binary': REPO_ROOT / 'data/binary/libritts_train100_formal_rhythm_v5',
        'train100_360_processed': REPO_ROOT / 'data/processed/libritts_train100_360_formal',
        'train100_360_binary': REPO_ROOT / 'data/binary/libritts_train100_360_formal_rhythm_v5',
    }

    report = {
        'repo_root': str(REPO_ROOT),
        'raw_root': str(raw_root),
        'configs': {k: _status(v) for k, v in configs.items()},
        'scripts': {k: _status(v) for k, v in scripts.items()},
        'checkpoints': {k: _status(v) for k, v in checkpoints.items()},
        'raw_splits': {k: _status(v) for k, v in raw_splits.items()},
        'outputs': {k: _status(v) for k, v in outputs.items()},
    }

    train100_ready = all(report['configs'][k]['exists'] for k in ['stage1_train100']) \
        and all(report['scripts'][k]['exists'] for k in ['prepare_train100', 'train_stage1']) \
        and all(report['checkpoints'][k]['exists'] for k in ['conan', 'emformer', 'rmvpe', 'hifigan']) \
        and all(report['raw_splits'][k]['exists'] for k in ['train-clean-100', 'dev-clean', 'test-clean'])

    train100_360_ready = all(report['configs'][k]['exists'] for k in ['stage1_train100_360']) \
        and report['scripts']['prepare_train100_360']['exists'] \
        and all(report['checkpoints'][k]['exists'] for k in ['conan', 'emformer', 'rmvpe', 'hifigan']) \
        and all(report['raw_splits'][k]['exists'] for k in ['train-clean-100', 'train-clean-360', 'dev-clean', 'test-clean'])

    report['summary'] = {
        'train100_recipe_ready': train100_ready,
        'train100_360_recipe_ready': train100_360_ready,
        'train100_outputs_built': report['outputs']['train100_processed']['exists'] and report['outputs']['train100_binary']['exists'],
        'train100_360_outputs_built': report['outputs']['train100_360_processed']['exists'] and report['outputs']['train100_360_binary']['exists'],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description='Verify core AutoDL training assets.')
    parser.add_argument('--raw_root', default=str(DEFAULT_RAW_ROOT))
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    report = build_report(Path(args.raw_root))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    print(f"[verify-core] repo_root={report['repo_root']}")
    print(f"[verify-core] raw_root={report['raw_root']}")
    print('[verify-core] summary:')
    for key, value in report['summary'].items():
        print(f"  - {key}: {value}")
    print('[verify-core] missing raw splits:')
    missing = [name for name, meta in report['raw_splits'].items() if not meta['exists']]
    if missing:
        for name in missing:
            print(f"  - {name}")
    else:
        print('  - none')


if __name__ == '__main__':
    main()
