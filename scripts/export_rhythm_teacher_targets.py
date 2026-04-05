from __future__ import annotations

import argparse
import json
import re
import sys
from contextlib import nullcontext
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _move_to_device(obj, device, torch):
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=(getattr(device, "type", "cpu") == "cuda"))
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device, torch) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_to_device(v, device, torch) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_to_device(v, device, torch) for v in obj)
    return obj


def _tensor_scalar_to_float(value, np, torch, default=1.0):
    if value is None:
        return float(default)
    if isinstance(value, torch.Tensor):
        if value.numel() <= 0:
            return float(default)
        return float(value.detach().float().reshape(-1)[0].cpu())
    if isinstance(value, (float, int)):
        return float(value)
    arr = np.asarray(value)
    if arr.size <= 0:
        return float(default)
    return float(arr.reshape(-1)[0])


def _sample_scalar_to_float(value, sample_idx: int, np, torch, default=1.0):
    if isinstance(value, torch.Tensor) and value.dim() > 0:
        value = value[sample_idx]
    elif isinstance(value, np.ndarray) and value.ndim > 0:
        value = value[sample_idx]
    elif isinstance(value, (list, tuple)) and len(value) > sample_idx:
        value = value[sample_idx]
    return _tensor_scalar_to_float(value, np, torch, default=default)


def _bool_like_metric(value, *, np, torch, default: float = 0.0) -> bool:
    return bool(_tensor_scalar_to_float(value, np, torch, default=default) > 0.5)


def _require_teacher_export_runtime(output, *, np, torch):
    execution = output.get("rhythm_execution")
    unit_batch = output.get("rhythm_unit_batch")
    if execution is None or unit_batch is None:
        raise RuntimeError("Teacher target export expected rhythm_execution and rhythm_unit_batch.")
    teacher_as_main = _bool_like_metric(output.get("rhythm_teacher_as_main", 0.0), np=np, torch=torch)
    teacher_only_stage = _bool_like_metric(output.get("rhythm_teacher_only_stage", 0.0), np=np, torch=torch)
    if not (teacher_as_main or teacher_only_stage):
        raise RuntimeError(
            "Teacher target export expected teacher_offline runtime semantics "
            "(rhythm_teacher_as_main or rhythm_teacher_only_stage)."
        )
    if output.get("rhythm_offline_execution") is not None:
        raise RuntimeError(
            "Teacher target export should not emit separate rhythm_offline_execution; "
            "export consumes the primary teacher execution only."
        )
    return execution, unit_batch


def _resolve_asset_path(output_dir: Path, *, split: str, item_name: str, flat_output: bool) -> Path:
    if flat_output:
        return output_dir / f"{item_name}.teacher.npz"
    return output_dir / split / f"{item_name}.teacher.npz"


def _resolve_ckpt_path_and_step(ckpt_arg: str):
    import torch
    from utils.commons.ckpt_utils import get_last_checkpoint

    ckpt_path = Path(ckpt_arg)
    if ckpt_path.is_file():
        checkpoint = torch.load(str(ckpt_path), map_location="cpu")
        path = str(ckpt_path)
    else:
        checkpoint, path = get_last_checkpoint(str(ckpt_arg))
        if checkpoint is None or path is None:
            raise FileNotFoundError(f"No checkpoint found under: {ckpt_arg}")
    global_step = checkpoint.get("global_step", checkpoint.get("step", None))
    if global_step is None:
        match = re.search(r"steps_(\d+)\.ckpt$", str(path))
        global_step = int(match.group(1)) if match else 0
    return str(path), int(global_step)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export learned-offline teacher targets to *.teacher.npz bundles.")
    parser.add_argument("--config", required=True, help="Teacher-offline config.")
    parser.add_argument("--ckpt", required=True, help="Checkpoint file or checkpoint directory.")
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write teacher target bundles. By default assets are written to output_dir/{split}/.",
    )
    parser.add_argument("--binary_data_dir", default=None)
    parser.add_argument("--processed_data_dir", default=None)
    parser.add_argument("--exp_name", default="export_rhythm_teacher_targets")
    parser.add_argument("--splits", nargs="+", default=["train", "valid"])
    parser.add_argument("--max_items", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--flat_output",
        action="store_true",
        help="Write assets directly under output_dir/ instead of per-split subdirectories."
             " Flat output is less safe when item_name collisions exist across splits.",
    )
    parser.add_argument("--amp", action="store_true", help="Enable CUDA autocast during export inference.")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_sentences", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=6000)
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    import numpy as np
    import torch
    from tqdm import tqdm

    from modules.Conan.rhythm.supervision import build_learned_offline_teacher_export_bundle
    from tasks.Conan.Conan import ConanTask
    from tasks.Conan.dataset import ConanDataset
    from utils.commons.ckpt_utils import load_ckpt
    from utils.commons.dataset_utils import build_dataloader
    from utils.commons.hparams import set_hparams

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    hp_overrides = [
        f"max_sentences={args.max_sentences}",
        f"max_tokens={args.max_tokens}",
        f"ds_workers={args.num_workers}",
        "num_sanity_val_steps=0",
        "style=True",
        "rhythm_minimal_style_only=True",
        "rhythm_stage=teacher_offline",
        "rhythm_teacher_only_stage=True",
        "rhythm_teacher_as_main=True",
        "rhythm_schedule_only_stage=False",
        "rhythm_optimize_module_only=True",
        "rhythm_enable_learned_offline_teacher=True",
        "rhythm_runtime_enable_learned_offline_teacher=True",
        "rhythm_enable_dual_mode_teacher=False",
        "rhythm_primary_target_surface=guidance",
        "rhythm_distill_surface=none",
        "lambda_rhythm_distill=0.0",
        "rhythm_streaming_prefix_train=False",
    ]
    if args.binary_data_dir:
        hp_overrides.append(f"binary_data_dir='{args.binary_data_dir}'")
    if args.processed_data_dir:
        hp_overrides.append(f"processed_data_dir='{args.processed_data_dir}'")
    set_hparams(
        config=args.config,
        exp_name=args.exp_name,
        hparams_str=",".join(hp_overrides),
    )

    ckpt_path, ckpt_step = _resolve_ckpt_path_and_step(args.ckpt)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task = ConanTask()
    task.build_tts_model()
    load_ckpt(task.model, ckpt_path)
    task.model.to(device)
    task.model.eval()
    task.global_step = ckpt_step

    manifest_items: list[dict[str, str | int | float]] = []
    seen_asset_paths: dict[str, dict[str, str]] = {}
    total_written = 0
    total_skipped = 0
    amp_enabled = bool(args.amp and device.type == "cuda")

    print(f"[export-rhythm-teacher] config={args.config}")
    print(f"[export-rhythm-teacher] ckpt={ckpt_path}")
    print(f"[export-rhythm-teacher] global_step={ckpt_step}")
    print(f"[export-rhythm-teacher] device={device}")
    print(f"[export-rhythm-teacher] output_dir={output_dir}")
    print(f"[export-rhythm-teacher] flat_output={args.flat_output}")
    print(f"[export-rhythm-teacher] amp={amp_enabled}")

    for split in args.splits:
        dataset = ConanDataset(split, shuffle=False)
        loader = build_dataloader(
            dataset,
            shuffle=False,
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences,
            apply_batch_by_size=True,
            pin_memory=device.type == "cuda",
            use_ddp=False,
        )
        split_written = 0
        split_skipped = 0
        split_processed = 0
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"export:{split}")):
            if args.max_items > 0 and split_processed >= args.max_items:
                break
            item_names = batch["item_name"]
            batch = _move_to_device(batch, device, torch)
            inference_ctx = torch.autocast(device_type="cuda", enabled=True) if amp_enabled else nullcontext()
            with torch.no_grad():
                with inference_ctx:
                    _, output = task.run_model(batch, infer=False, test=False)
            execution, unit_batch = _require_teacher_export_runtime(
                output,
                np=np,
                torch=torch,
            )
            remaining = args.max_items - split_processed if args.max_items > 0 else len(item_names)
            item_count = min(len(item_names), remaining)
            for sample_idx, item_name in enumerate(item_names[:item_count]):
                asset_path = _resolve_asset_path(
                    output_dir,
                    split=split,
                    item_name=item_name,
                    flat_output=args.flat_output,
                )
                asset_path.parent.mkdir(parents=True, exist_ok=True)
                path_key = str(asset_path)
                prev_seen = seen_asset_paths.get(path_key)
                if prev_seen is not None:
                    raise RuntimeError(
                        "Duplicate teacher export target detected for "
                        f"item='{item_name}' split='{split}' at {asset_path}. "
                        f"Previous writer: item='{prev_seen['item_name']}' split='{prev_seen['split']}'. "
                        "Use the default per-split layout or make item_name globally unique before export."
                    )
                seen_asset_paths[path_key] = {"item_name": item_name, "split": split}
                if asset_path.exists() and not args.overwrite:
                    split_skipped += 1
                    manifest_items.append({
                        "item_name": item_name,
                        "path": str(asset_path),
                        "split": split,
                        "status": "skipped_exists",
                    })
                    split_processed += 1
                    continue
                confidence = _sample_scalar_to_float(
                    output.get("rhythm_offline_confidence"),
                    sample_idx,
                    np,
                    torch,
                    default=1.0,
                )
                bundle = build_learned_offline_teacher_export_bundle(
                    speech_exec_tgt=execution.speech_duration_exec[sample_idx].detach().cpu().numpy(),
                    pause_exec_tgt=execution.pause_after_exec[sample_idx].detach().cpu().numpy(),
                    dur_anchor_src=unit_batch.dur_anchor_src[sample_idx].detach().cpu().numpy(),
                    unit_mask=unit_batch.unit_mask[sample_idx].detach().cpu().numpy(),
                    confidence=confidence,
                )
                expected_units = int(unit_batch.dur_anchor_src[sample_idx].detach().cpu().numel())
                written_units = int(np.asarray(bundle["rhythm_teacher_speech_exec_tgt"]).reshape(-1).shape[0])
                if written_units != expected_units:
                    raise RuntimeError(
                        f"Teacher export length mismatch for item='{item_name}': "
                        f"written_units={written_units}, expected_units={expected_units}."
                    )
                component_conf = {}
                for key, out_key in (
                    ("rhythm_teacher_confidence_exec", "rhythm_offline_confidence_exec"),
                    ("rhythm_teacher_confidence_budget", "rhythm_offline_confidence_budget"),
                    ("rhythm_teacher_confidence_prefix", "rhythm_offline_confidence_prefix"),
                    ("rhythm_teacher_confidence_allocation", "rhythm_offline_confidence_allocation"),
                    ("rhythm_teacher_confidence_shape", "rhythm_offline_confidence_shape"),
                ):
                    value = output.get(out_key)
                    if value is None and key == "rhythm_teacher_confidence_shape":
                        value = output.get("rhythm_offline_confidence_exec")
                    if value is not None:
                        component_conf[key] = np.asarray(
                            [
                                _tensor_scalar_to_float(
                                    value[sample_idx] if isinstance(value, torch.Tensor) and value.dim() > 0 else value,
                                    np,
                                    torch,
                                )
                            ],
                            dtype=np.float32,
                        )
                np.savez(
                    asset_path,
                    item_name=np.asarray([item_name], dtype=np.str_),
                    export_split=np.asarray([split], dtype=np.str_),
                    export_global_step=np.asarray([ckpt_step], dtype=np.int64),
                    **bundle,
                    **component_conf,
                )
                split_written += 1
                manifest_items.append({
                    "item_name": item_name,
                    "path": str(asset_path),
                    "split": split,
                    "status": "written",
                    "confidence": confidence,
                })
                split_processed += 1
        total_written += split_written
        total_skipped += split_skipped
        print(
            f"[export-rhythm-teacher] split={split} dataset_len={len(dataset)} "
            f"written={split_written} skipped={split_skipped}"
        )

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": args.config,
                "ckpt": ckpt_path,
                "global_step": ckpt_step,
                "splits": args.splits,
                "written": total_written,
                "skipped": total_skipped,
                "items": manifest_items,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[export-rhythm-teacher] written={total_written} skipped={total_skipped}")
    print(f"[export-rhythm-teacher] manifest={manifest_path}")


if __name__ == "__main__":
    main()
