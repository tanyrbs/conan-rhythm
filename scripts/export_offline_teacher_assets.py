import argparse
import json
import re
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules.Conan.rhythm.supervision import build_learned_offline_teacher_bundle
from tasks.Conan.Conan import ConanTask
from tasks.Conan.dataset import ConanDataset
from utils.commons.ckpt_utils import get_last_checkpoint, load_ckpt
from utils.commons.hparams import set_hparams


def _move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_to_device(v, device) for v in obj)
    return obj


def _tensor_scalar_to_float(value, default=1.0):
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


def _resolve_ckpt_path_and_step(ckpt_arg: str) -> tuple[str, int]:
    ckpt_path = Path(ckpt_arg)
    if ckpt_path.is_file():
        checkpoint = torch.load(str(ckpt_path), map_location="cpu")
        path = str(ckpt_path)
    else:
        checkpoint, path = get_last_checkpoint(str(ckpt_path))
        if checkpoint is None or path is None:
            raise FileNotFoundError(f"No checkpoint found under: {ckpt_arg}")
    global_step = checkpoint.get("global_step", checkpoint.get("step", None))
    if global_step is None:
        match = re.search(r"steps_(\d+)\.ckpt$", str(path))
        global_step = int(match.group(1)) if match else 0
    return str(path), int(global_step)


def main():
    parser = argparse.ArgumentParser(description="Export learned offline teacher surfaces to .npz assets")
    parser.add_argument("--config", required=True, help="Offline-teacher training config.")
    parser.add_argument("--ckpt", required=True, help="Checkpoint file or directory.")
    parser.add_argument("--output_dir", required=True, help="Directory to write {item_name}.teacher.npz files.")
    parser.add_argument("--binary_data_dir", default=None)
    parser.add_argument("--processed_data_dir", default=None)
    parser.add_argument("--exp_name", default="export_offline_teacher")
    parser.add_argument("--splits", nargs="+", default=["train", "valid"])
    parser.add_argument("--max_items", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_sentences", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=6000)
    args = parser.parse_args()

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
        "rhythm_teacher_only_stage=true",
        "rhythm_schedule_only_stage=false",
        "rhythm_optimize_module_only=true",
        "rhythm_enable_learned_offline_teacher=true",
        "rhythm_runtime_enable_learned_offline_teacher=true",
        "rhythm_enable_dual_mode_teacher=false",
        "rhythm_primary_target_surface=guidance",
        "rhythm_distill_surface=none",
        "lambda_rhythm_distill=0.0",
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

    manifest: dict[str, dict[str, str | int | float]] = {}
    total_written = 0
    total_skipped = 0

    print(f"[export-offline-teacher] config={args.config}")
    print(f"[export-offline-teacher] ckpt={ckpt_path}")
    print(f"[export-offline-teacher] global_step={ckpt_step}")
    print(f"[export-offline-teacher] device={device}")
    print(f"[export-offline-teacher] output_dir={output_dir}")

    for split in args.splits:
        dataset = ConanDataset(split, shuffle=False)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=dataset.collater,
        )
        split_written = 0
        split_skipped = 0
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"export:{split}")):
            if args.max_items > 0 and batch_idx >= args.max_items:
                break
            item_names = batch["item_name"]
            batch = _move_to_device(batch, device)
            with torch.no_grad():
                _, output = task.run_model(batch, infer=False, test=False)
            execution = output.get("rhythm_execution")
            unit_batch = output.get("rhythm_unit_batch")
            if execution is None or unit_batch is None:
                raise RuntimeError("Offline teacher export expected rhythm_execution and rhythm_unit_batch.")
            for sample_idx, item_name in enumerate(item_names):
                asset_path = output_dir / f"{item_name}.teacher.npz"
                if asset_path.exists() and not args.overwrite:
                    split_skipped += 1
                    manifest[item_name] = {
                        "path": str(asset_path),
                        "split": split,
                        "status": "skipped_exists",
                    }
                    continue
                confidence = _tensor_scalar_to_float(
                    output.get("rhythm_offline_confidence"),
                    default=1.0,
                )
                bundle = build_learned_offline_teacher_bundle(
                    speech_exec_tgt=execution.speech_duration_exec[sample_idx].detach().cpu().numpy(),
                    pause_exec_tgt=execution.pause_after_exec[sample_idx].detach().cpu().numpy(),
                    dur_anchor_src=unit_batch.dur_anchor_src[sample_idx].detach().cpu().numpy(),
                    unit_mask=unit_batch.unit_mask[sample_idx].detach().cpu().numpy(),
                    confidence=confidence,
                )
                component_conf = {}
                for key, out_key in (
                    ("rhythm_teacher_confidence_exec", "rhythm_offline_confidence_exec"),
                    ("rhythm_teacher_confidence_budget", "rhythm_offline_confidence_budget"),
                    ("rhythm_teacher_confidence_prefix", "rhythm_offline_confidence_prefix"),
                    ("rhythm_teacher_confidence_allocation", "rhythm_offline_confidence_allocation"),
                ):
                    value = output.get(out_key)
                    if value is not None:
                        component_conf[key] = np.asarray(
                            [_tensor_scalar_to_float(value[sample_idx] if isinstance(value, torch.Tensor) and value.dim() > 0 else value)],
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
                manifest[item_name] = {
                    "path": str(asset_path),
                    "split": split,
                    "status": "written",
                    "confidence": confidence,
                }
        total_written += split_written
        total_skipped += split_skipped
        print(
            f"[export-offline-teacher] split={split} dataset_len={len(dataset)} "
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
                "items": manifest,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[export-offline-teacher] written={total_written} skipped={total_skipped}")
    print(f"[export-offline-teacher] manifest={manifest_path}")


if __name__ == "__main__":
    main()
