from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


REQUIRED_TEACHER_FIELDS = (
    "rhythm_teacher_speech_exec_tgt",
    "rhythm_teacher_pause_exec_tgt",
    "rhythm_teacher_speech_budget_tgt",
    "rhythm_teacher_pause_budget_tgt",
    "rhythm_teacher_allocation_tgt",
    "rhythm_teacher_prefix_clock_tgt",
    "rhythm_teacher_prefix_backlog_tgt",
    "rhythm_teacher_confidence",
    "rhythm_teacher_surface_name",
    "rhythm_teacher_target_source_id",
)


def _as_posix(value) -> str:
    if isinstance(value, Path):
        return value.resolve().as_posix()
    return str(value).replace("\\", "/")


def _encode_hparam_value(value) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, Path):
        return repr(_as_posix(value))
    if isinstance(value, (list, tuple)):
        return "[" + " ".join(_encode_hparam_value(v) for v in value) + "]"
    return repr(_as_posix(value))


def _encode_hparams(overrides: dict) -> str:
    return ",".join(f"{key}={_encode_hparam_value(value)}" for key, value in overrides.items())


def _run_command(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print(f"[integration] RUN {subprocess.list2cmdline(cmd)}")
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


def _load_split_prefixes(processed_data_dir: Path) -> tuple[list[str], list[str]]:
    summary_path = processed_data_dir / "build_summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        split_tags = payload.get("split_tags", {}) or {}
        valid_tag = str(split_tags.get("valid", "")).strip()
        test_tag = str(split_tags.get("test", "")).strip()
        if valid_tag and test_tag:
            return [f"_{valid_tag}_"], [f"_{test_tag}_"]

    metadata_path = processed_data_dir / "metadata.json"
    if metadata_path.exists():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        split_to_prefix: dict[str, str] = {}
        for item in payload:
            split = str(item.get("split", "")).strip().lower()
            item_name = str(item.get("item_name", "")).strip()
            if not split or not item_name:
                continue
            parts = item_name.split("_")
            if len(parts) >= 3:
                split_to_prefix.setdefault(split, f"_{parts[1]}_")
        valid_prefix = split_to_prefix.get("valid") or split_to_prefix.get("dev") or split_to_prefix.get("dev-clean")
        test_prefix = split_to_prefix.get("test") or split_to_prefix.get("test-clean")
        if valid_prefix and test_prefix:
            return [valid_prefix], [test_prefix]

    raise RuntimeError(
        f"Could not infer valid/test prefixes from processed_data_dir={processed_data_dir}. "
        "Provide build_summary.json with split_tags or metadata.json with split labels."
    )


def _normalize_export_splits(
    export_splits: list[str] | tuple[str, ...],
    *,
    include_valid: bool,
    include_test: bool,
) -> list[str]:
    alias = {
        "dev": "valid",
        "val": "valid",
        "validation": "valid",
    }
    normalized: list[str] = []
    for split in export_splits:
        value = alias.get(str(split).strip().lower(), str(split).strip().lower())
        if value and value not in normalized:
            normalized.append(value)
    if "train" not in normalized:
        normalized.insert(0, "train")
    if include_valid and "valid" not in normalized:
        normalized.append("valid")
    if include_test and "test" not in normalized:
        normalized.append("test")
    return normalized


def _create_bootstrap_teacher_ckpt(
    *,
    teacher_config: str,
    teacher_hparams: str,
    output_ckpt: Path,
) -> Path:
    import torch
    from tasks.Conan.Conan import ConanTask
    from utils.commons.hparams import set_hparams

    output_ckpt.parent.mkdir(parents=True, exist_ok=True)
    set_hparams(
        config=teacher_config,
        exp_name=f"bootstrap_teacher_{uuid.uuid4().hex[:8]}",
        hparams_str=teacher_hparams,
        global_hparams=True,
        print_hparams=False,
    )
    task = ConanTask()
    task.build_tts_model()
    state_dict = {
        f"model.{name}": tensor.detach().cpu()
        for name, tensor in task.model.state_dict().items()
    }
    torch.save(
        {
            "state_dict": state_dict,
            "global_step": 0,
        },
        str(output_ckpt),
    )
    print(f"[integration] bootstrap_teacher_ckpt={output_ckpt}")
    return output_ckpt


def _assert_cached_teacher_fields(binary_data_dir: Path) -> dict[str, list[str]]:
    from modules.Conan.rhythm.supervision import materialize_rhythm_cache_compat_fields
    from utils.commons.indexed_datasets import IndexedDataset

    results: dict[str, list[str]] = {}
    for split in ("train", "valid"):
        ds = IndexedDataset(str(binary_data_dir / split))
        if len(ds) <= 0:
            raise RuntimeError(f"Split '{split}' is empty in {binary_data_dir}.")
        item = materialize_rhythm_cache_compat_fields(ds[0])
        missing = [key for key in REQUIRED_TEACHER_FIELDS if key not in item]
        if missing:
            raise RuntimeError(f"Split '{split}' missing cached teacher fields: {missing}")
        results[split] = sorted(k for k in REQUIRED_TEACHER_FIELDS if k in item)
    return results


def _run_student_kd_smoke(
    *,
    student_config: str,
    student_hparams: str,
) -> dict[str, object]:
    import torch
    from tasks.Conan.Conan import ConanTask
    from tasks.Conan.dataset import ConanDataset
    from utils.commons.hparams import set_hparams

    set_hparams(
        config=student_config,
        exp_name=f"student_kd_smoke_{uuid.uuid4().hex[:8]}",
        hparams_str=student_hparams,
        global_hparams=True,
        print_hparams=False,
    )
    dataset = ConanDataset("train", shuffle=False)
    if len(dataset) <= 0:
        raise RuntimeError("Student KD smoke dataset is empty after filtering.")
    batch = dataset.collater([dataset[0]])
    task = ConanTask()
    task.build_tts_model()
    task.global_step = 0
    with torch.no_grad():
        losses, output = task.run_model(batch, infer=False, test=False)
    if "rhythm_execution" not in output or output["rhythm_execution"] is None:
        raise RuntimeError("Student KD smoke run did not produce rhythm_execution.")
    if output.get("rhythm_offline_execution") is not None:
        raise RuntimeError("Student KD smoke run unexpectedly produced rhythm_offline_execution.")
    if bool(output.get("rhythm_teacher_as_main", 0.0)):
        raise RuntimeError("Student KD smoke run unexpectedly reported rhythm_teacher_as_main=true.")
    return {
        "dataset_len": len(dataset),
        "loss_keys": sorted(losses.keys()),
        "rhythm_stage": output.get("rhythm_stage"),
        "mel_out_shape": tuple(output["mel_out"].shape) if "mel_out" in output else None,
        "speech_exec_shape": tuple(output["speech_duration_exec"].shape) if "speech_duration_exec" in output else None,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the real teacher_offline -> export -> re-binarize -> student_kd cached_only integration chain."
    )
    parser.add_argument("--teacher_config", default="egs/conan_emformer_rhythm_v2_teacher_offline.yaml")
    parser.add_argument("--student_config", default="egs/conan_emformer_rhythm_v2_student_kd.yaml")
    parser.add_argument("--processed_data_dir", default="data/processed/libritts_local_real_smoke")
    parser.add_argument("--teacher_ckpt", default="", help="Optional trained teacher_offline checkpoint. If empty, a bootstrap random checkpoint is created for structural integration.")
    parser.add_argument("--work_root", default="artifacts/rhythm_teacher_export_student_kd")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument(
        "--export_splits",
        nargs="+",
        default=["train", "valid", "test"],
        help="Teacher target export splits. Missing valid/test will be auto-added when the processed corpus exposes those splits.",
    )
    parser.add_argument("--max_export_items", type=int, default=-1)
    parser.add_argument("--inspect_items", type=int, default=4)
    parser.add_argument("--skip_model_dry_run", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    processed_data_dir = (REPO_ROOT / args.processed_data_dir).resolve() if not Path(args.processed_data_dir).is_absolute() else Path(args.processed_data_dir).resolve()
    if not processed_data_dir.exists():
        raise FileNotFoundError(f"processed_data_dir does not exist: {processed_data_dir}")

    valid_prefixes, test_prefixes = _load_split_prefixes(processed_data_dir)
    export_splits = _normalize_export_splits(
        args.export_splits,
        include_valid=bool(valid_prefixes),
        include_test=bool(test_prefixes),
    )
    run_dir = ((REPO_ROOT / args.work_root) if not Path(args.work_root).is_absolute() else Path(args.work_root)).resolve() / uuid.uuid4().hex[:10]
    teacher_binary_dir = run_dir / "teacher_binary"
    export_dir = run_dir / "teacher_export"
    student_binary_dir = run_dir / "student_binary"
    bootstrap_ckpt = run_dir / "bootstrap_teacher" / "model_ckpt_steps_0.ckpt"
    run_dir.mkdir(parents=True, exist_ok=True)

    common_overrides = {
        "processed_data_dir": processed_data_dir,
        "valid_prefixes": valid_prefixes,
        "test_prefixes": test_prefixes,
        "style": True,
        "rhythm_minimal_style_only": True,
        "ds_workers": 0,
        "max_sentences": 1,
        "max_tokens": 3000,
    }
    teacher_overrides = {
        **common_overrides,
        "binary_data_dir": teacher_binary_dir,
    }
    student_overrides = {
        **common_overrides,
        "binary_data_dir": student_binary_dir,
        "rhythm_teacher_target_dir": export_dir,
    }
    teacher_hparams = _encode_hparams(teacher_overrides)
    student_hparams = _encode_hparams(student_overrides)

    env = os.environ.copy()
    env.setdefault("N_PROC", "1")

    teacher_ckpt = Path(args.teacher_ckpt).resolve() if args.teacher_ckpt else _create_bootstrap_teacher_ckpt(
        teacher_config=args.teacher_config,
        teacher_hparams=teacher_hparams,
        output_ckpt=bootstrap_ckpt,
    )

    _run_command(
        [
            sys.executable,
            "-m",
            "data_gen.tts.runs.binarize",
            "--reset",
            "--config",
            args.teacher_config,
            "--exp_name",
            f"integration_teacher_bin_{uuid.uuid4().hex[:8]}",
            "-hp",
            teacher_hparams,
        ],
        env=env,
    )
    _run_command(
        [
            sys.executable,
            "scripts/preflight_rhythm_v2.py",
            "--config",
            args.teacher_config,
            "--binary_data_dir",
            _as_posix(teacher_binary_dir),
            "--hparams",
            _encode_hparams({k: v for k, v in teacher_overrides.items() if k != "binary_data_dir"}),
            "--inspect_items",
            str(args.inspect_items),
            "--splits",
            "train",
            "valid",
            *(["--model_dry_run"] if not args.skip_model_dry_run else []),
        ],
        env=env,
    )
    _run_command(
        [
            sys.executable,
            "scripts/export_rhythm_teacher_targets.py",
            "--config",
            args.teacher_config,
            "--ckpt",
            _as_posix(teacher_ckpt),
            "--output_dir",
            _as_posix(export_dir),
            "--binary_data_dir",
            _as_posix(teacher_binary_dir),
            "--processed_data_dir",
            _as_posix(processed_data_dir),
            "--device",
            args.device,
            "--num_workers",
            "0",
            "--max_sentences",
            "1",
            "--max_tokens",
            "3000",
            "--max_items",
            str(args.max_export_items),
            "--overwrite",
            "--splits",
            *export_splits,
        ],
        env=env,
    )
    _run_command(
        [
            sys.executable,
            "-m",
            "data_gen.tts.runs.binarize",
            "--reset",
            "--config",
            args.student_config,
            "--exp_name",
            f"integration_student_bin_{uuid.uuid4().hex[:8]}",
            "-hp",
            student_hparams,
        ],
        env=env,
    )
    _run_command(
        [
            sys.executable,
            "scripts/preflight_rhythm_v2.py",
            "--config",
            args.student_config,
            "--binary_data_dir",
            _as_posix(student_binary_dir),
            "--hparams",
            _encode_hparams({k: v for k, v in student_overrides.items() if k != "binary_data_dir"}),
            "--inspect_items",
            str(args.inspect_items),
            "--splits",
            "train",
            "valid",
            *(["--model_dry_run"] if not args.skip_model_dry_run else []),
        ],
        env=env,
    )

    field_report = _assert_cached_teacher_fields(student_binary_dir)
    smoke_report = _run_student_kd_smoke(
        student_config=args.student_config,
        student_hparams=student_hparams,
    )
    summary = {
        "processed_data_dir": _as_posix(processed_data_dir),
        "teacher_config": args.teacher_config,
        "student_config": args.student_config,
        "teacher_ckpt": _as_posix(teacher_ckpt),
        "teacher_ckpt_mode": "provided" if args.teacher_ckpt else "bootstrap_random_init",
        "teacher_binary_dir": _as_posix(teacher_binary_dir),
        "export_dir": _as_posix(export_dir),
        "student_binary_dir": _as_posix(student_binary_dir),
        "valid_prefixes": valid_prefixes,
        "test_prefixes": test_prefixes,
        "export_splits": export_splits,
        "teacher_field_report": field_report,
        "student_kd_smoke": smoke_report,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[integration] OK summary={summary_path}")


if __name__ == "__main__":
    main()
