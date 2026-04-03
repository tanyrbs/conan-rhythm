from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.Conan.rhythm.factorization import (  # noqa: E402
    collect_planner_surface_bundle,
    compute_surface_distance_report,
)
from tasks.Conan.Conan import ConanTask  # noqa: E402
from tasks.Conan.dataset import ConanDataset  # noqa: E402
from utils.commons.ckpt_utils import load_ckpt  # noqa: E402
from utils.commons.hparams import set_hparams  # noqa: E402


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


def _aggregate_reports(reports: list[dict[str, float]]) -> dict[str, float]:
    if not reports:
        return {}
    keys = sorted(reports[0].keys())
    return {key: float(sum(report[key] for report in reports) / len(reports)) for key in keys}


def _surface_leakage_view(report: dict[str, float]) -> dict[str, float]:
    return {
        "speech_budget_win": report["speech_budget_rel_l1"],
        "pause_budget_win": report["pause_budget_rel_l1"],
        "dur_shape_unit": report["dur_shape_l1"],
        "pause_shape_unit": report["pause_shape_kl"],
        "boundary_score_unit": report["boundary_score_l1"],
        "speech_exec": report["speech_exec_rel_l1"],
        "pause_exec": report["pause_exec_rel_l1"],
        "commit_frontier": report["commit_frontier_l1"],
    }


def _build_task(
    *,
    config: str,
    binary_data_dir: str,
    processed_data_dir: str,
    ckpt: str,
    seed: int | None,
    device: torch.device,
    exp_name: str,
):
    if seed is not None:
        torch.manual_seed(int(seed))
    set_hparams(
        config=config,
        exp_name=exp_name,
        hparams_str=(
            f"binary_data_dir='{Path(binary_data_dir).resolve().as_posix()}',"
            f"processed_data_dir='{Path(processed_data_dir).resolve().as_posix()}',"
            "ds_workers=0,max_sentences=1,max_tokens=3000,"
            "num_sanity_val_steps=0,style=True,rhythm_minimal_style_only=True"
        ),
        global_hparams=True,
        print_hparams=False,
    )
    task = ConanTask()
    task.build_tts_model()
    if ckpt:
        load_ckpt(task.model, ckpt)
    task.model.to(device)
    task.model.eval()
    return task


def _symmetrize(report_ab: dict[str, float], report_ba: dict[str, float]) -> dict[str, float]:
    return {
        key: float((report_ab[key] + report_ba[key]) * 0.5)
        for key in sorted(report_ab.keys())
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate seed/checkpoint stability of compact planner surfaces.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--binary_data_dir", required=True)
    parser.add_argument("--processed_data_dir", required=True)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--max_items", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--exp_name", default="eval_rhythm_seed_stability")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--ckpts", nargs="*", default=[])
    parser.add_argument("--seeds", nargs="*", type=int, default=[])
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if not args.ckpts and not args.seeds:
        raise ValueError("Provide --ckpts and/or --seeds for stability comparison.")

    set_hparams(
        config=args.config,
        exp_name=args.exp_name,
        hparams_str=(
            f"binary_data_dir='{Path(args.binary_data_dir).resolve().as_posix()}',"
            f"processed_data_dir='{Path(args.processed_data_dir).resolve().as_posix()}',"
            "ds_workers=0,max_sentences=1,max_tokens=3000,"
            "num_sanity_val_steps=0,style=True,rhythm_minimal_style_only=True"
        ),
        global_hparams=True,
        print_hparams=False,
    )
    dataset = ConanDataset(args.split, shuffle=False)
    if len(dataset) <= 0:
        raise RuntimeError(f"Dataset split '{args.split}' is empty.")
    max_items = min(len(dataset), max(1, int(args.max_items)))
    batches = [_move_to_device(dataset.collater([dataset[item_idx]]), device) for item_idx in range(max_items)]

    model_specs = []
    for idx, ckpt in enumerate(args.ckpts):
        label = f"ckpt_{idx}"
        model_specs.append((label, str(Path(ckpt).resolve()), None))
    for seed in args.seeds:
        label = f"seed_{int(seed)}"
        model_specs.append((label, "", int(seed)))
    if len(model_specs) < 2:
        raise ValueError("Need at least two checkpoints/seeds for stability comparison.")

    outputs_by_model: dict[str, list[dict[str, torch.Tensor]]] = {}
    for label, ckpt, seed in model_specs:
        task = _build_task(
            config=args.config,
            binary_data_dir=args.binary_data_dir,
            processed_data_dir=args.processed_data_dir,
            ckpt=ckpt,
            seed=seed,
            device=device,
            exp_name=f"{args.exp_name}_{label}",
        )
        collected = []
        with torch.no_grad():
            for batch in batches:
                _, output = task.run_model(batch, infer=False, test=False)
                collected.append({
                    "item_name": str(batch["item_name"][0]),
                    "unit_mask": output["rhythm_unit_batch"].unit_mask.detach(),
                    "surfaces": collect_planner_surface_bundle(output["rhythm_execution"]),
                })
        outputs_by_model[label] = collected

    pair_reports = {}
    for (label_a, _, _), (label_b, _, _) in itertools.combinations(model_specs, 2):
        reports = []
        items = outputs_by_model[label_a]
        other_items = outputs_by_model[label_b]
        for item_a, item_b in zip(items, other_items):
            report_ab = compute_surface_distance_report(
                item_a["surfaces"],
                item_b["surfaces"],
                unit_mask=item_a["unit_mask"],
            )
            report_ba = compute_surface_distance_report(
                item_b["surfaces"],
                item_a["surfaces"],
                unit_mask=item_b["unit_mask"],
            )
            reports.append(_symmetrize(report_ab, report_ba))
        aggregated = _aggregate_reports(reports)
        pair_reports[f"{label_a}__vs__{label_b}"] = {
            "surface_delta": aggregated,
            "leakage_view": _surface_leakage_view(aggregated),
        }

    overall_surface_delta = _aggregate_reports([entry["surface_delta"] for entry in pair_reports.values()])
    summary = {
        "config": args.config,
        "binary_data_dir": str(Path(args.binary_data_dir).resolve()),
        "processed_data_dir": str(Path(args.processed_data_dir).resolve()),
        "split": args.split,
        "max_items": max_items,
        "models": [
            {
                "label": label,
                "ckpt": ckpt or None,
                "seed": seed,
            }
            for label, ckpt, seed in model_specs
        ],
        "pairwise": pair_reports,
        "overall": {
            "surface_delta": overall_surface_delta,
            "leakage_view": _surface_leakage_view(overall_surface_delta),
        },
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        print(f"[seed-stability] wrote {output_path}")
    print(text)


if __name__ == "__main__":
    main()
