from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.Conan.rhythm.factorization import (  # noqa: E402
    CompactPlannerIntervention,
    apply_compact_reference_intervention,
    collect_planner_surface_bundle,
    compute_surface_distance_report,
)
from tasks.Conan.Conan import ConanTask  # noqa: E402
from tasks.Conan.dataset import ConanDataset  # noqa: E402
from utils.commons.ckpt_utils import load_ckpt  # noqa: E402
from utils.commons.hparams import set_hparams  # noqa: E402


DEFAULT_INTERVENTIONS = (
    CompactPlannerIntervention(name="global_rate_up", global_rate_scale=1.15),
    CompactPlannerIntervention(name="pause_ratio_up", pause_ratio_delta=0.10),
    CompactPlannerIntervention(name="local_rate_shape_up", local_rate_scale=1.20),
    CompactPlannerIntervention(name="boundary_trace_up", boundary_trace_bias=0.20),
)


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
    return {
        key: float(sum(report[key] for report in reports) / len(reports))
        for key in keys
    }


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
    device: torch.device,
    exp_name: str,
):
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


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate factor-wise intervention leakage on the compact rhythm planner.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--binary_data_dir", required=True)
    parser.add_argument("--processed_data_dir", required=True)
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--split", default="valid")
    parser.add_argument("--max_items", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--exp_name", default="eval_rhythm_factor_intervention")
    parser.add_argument("--output_json", default="")
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

    task = _build_task(
        config=args.config,
        binary_data_dir=args.binary_data_dir,
        processed_data_dir=args.processed_data_dir,
        ckpt=args.ckpt,
        device=device,
        exp_name=args.exp_name,
    )
    dataset = ConanDataset(args.split, shuffle=False)
    if len(dataset) <= 0:
        raise RuntimeError(f"Dataset split '{args.split}' is empty.")
    rhythm_module = task.model.rhythm_module

    intervention_reports: dict[str, list[dict[str, float]]] = {item.name: [] for item in DEFAULT_INTERVENTIONS}
    item_reports = []
    max_items = min(len(dataset), max(1, int(args.max_items)))
    for item_idx in range(max_items):
        batch = dataset.collater([dataset[item_idx]])
        batch = _move_to_device(batch, device)
        with torch.no_grad():
            _, output = task.run_model(batch, infer=False, test=False)
        execution = output["rhythm_execution"]
        ref_conditioning = output["rhythm_ref_conditioning"]
        unit_batch = output["rhythm_unit_batch"]
        baseline_surfaces = collect_planner_surface_bundle(execution)

        item_result = {
            "item_name": str(batch["item_name"][0]),
            "interventions": {},
        }
        for intervention in DEFAULT_INTERVENTIONS:
            intervened_ref = apply_compact_reference_intervention(ref_conditioning, intervention)
            with torch.no_grad():
                perturbed_execution = rhythm_module(
                    content_units=unit_batch.content_units,
                    dur_anchor_src=unit_batch.dur_anchor_src,
                    ref_conditioning=intervened_ref,
                    unit_mask=unit_batch.unit_mask,
                    open_run_mask=unit_batch.open_run_mask,
                    sealed_mask=unit_batch.sealed_mask,
                    sep_hint=unit_batch.sep_hint,
                    boundary_confidence=unit_batch.boundary_confidence,
                    state=rhythm_module.init_state(
                        batch_size=unit_batch.content_units.size(0),
                        device=unit_batch.content_units.device,
                    ),
                )
            perturbed_surfaces = collect_planner_surface_bundle(perturbed_execution)
            report = compute_surface_distance_report(
                baseline_surfaces,
                perturbed_surfaces,
                unit_mask=unit_batch.unit_mask,
            )
            item_result["interventions"][intervention.name] = {
                "surface_delta": report,
                "leakage_view": _surface_leakage_view(report),
            }
            intervention_reports[intervention.name].append(report)
        item_reports.append(item_result)

    aggregated = {
        name: {
            "surface_delta": _aggregate_reports(reports),
            "leakage_view": _surface_leakage_view(_aggregate_reports(reports)),
        }
        for name, reports in intervention_reports.items()
    }
    summary = {
        "config": args.config,
        "ckpt": args.ckpt or None,
        "binary_data_dir": str(Path(args.binary_data_dir).resolve()),
        "processed_data_dir": str(Path(args.processed_data_dir).resolve()),
        "split": args.split,
        "max_items": max_items,
        "aggregated": aggregated,
        "items": item_reports,
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        print(f"[factor-intervention] wrote {output_path}")
    print(text)


if __name__ == "__main__":
    main()
