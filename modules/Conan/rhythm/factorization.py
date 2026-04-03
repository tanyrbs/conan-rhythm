from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


COMPACT_REFERENCE_KEYS = (
    "global_rate",
    "pause_ratio",
    "local_rate_trace",
    "boundary_trace",
)


COMPACT_REFERENCE_ALIAS_KEYS = (
    "planner_ref_stats",
    "planner_ref_trace",
)


PLANNER_SURFACE_KEYS = (
    "speech_budget_win",
    "pause_budget_win",
    "dur_shape_unit",
    "pause_shape_unit",
    "boundary_score_unit",
    "speech_exec",
    "pause_exec",
    "commit_frontier",
)


@dataclass(frozen=True)
class CompactPlannerIntervention:
    name: str
    global_rate_scale: float = 1.0
    pause_ratio_delta: float = 0.0
    local_rate_scale: float = 1.0
    local_rate_bias: float = 0.0
    boundary_trace_scale: float = 1.0
    boundary_trace_bias: float = 0.0


def _clone_value(value: Any):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: _clone_value(v) for k, v in value.items()}
    return value


def clone_reference_conditioning(ref_conditioning: dict[str, Any]) -> dict[str, Any]:
    return {key: _clone_value(value) for key, value in ref_conditioning.items()}


def extract_compact_reference_contract(ref_conditioning: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    compact = {}
    for key in COMPACT_REFERENCE_KEYS:
        if key in ref_conditioning and isinstance(ref_conditioning[key], torch.Tensor):
            compact[key] = ref_conditioning[key]
    if {"global_rate", "pause_ratio"} <= compact.keys():
        compact["planner_ref_stats"] = torch.cat([compact["global_rate"], compact["pause_ratio"]], dim=-1)
    elif "planner_ref_stats" in ref_conditioning and isinstance(ref_conditioning["planner_ref_stats"], torch.Tensor):
        compact["planner_ref_stats"] = ref_conditioning["planner_ref_stats"]
    if {"local_rate_trace", "boundary_trace"} <= compact.keys():
        compact["planner_ref_trace"] = torch.cat([compact["local_rate_trace"], compact["boundary_trace"]], dim=-1)
    elif "planner_ref_trace" in ref_conditioning and isinstance(ref_conditioning["planner_ref_trace"], torch.Tensor):
        compact["planner_ref_trace"] = ref_conditioning["planner_ref_trace"]
    return compact


def _scale_centered_trace(trace: torch.Tensor, scale: float, bias: float = 0.0) -> torch.Tensor:
    trace = trace.float()
    center = trace.mean(dim=1, keepdim=True)
    adjusted = center + (trace - center) * float(scale)
    if abs(float(bias)) > 1e-8:
        adjusted = adjusted + float(bias)
    return adjusted


def apply_compact_reference_intervention(
    ref_conditioning: dict[str, Any],
    intervention: CompactPlannerIntervention,
) -> dict[str, Any]:
    compact = clone_reference_conditioning(ref_conditioning)
    if "global_rate" in compact:
        compact["global_rate"] = compact["global_rate"].float() * float(intervention.global_rate_scale)
    if "pause_ratio" in compact:
        compact["pause_ratio"] = (
            compact["pause_ratio"].float() + float(intervention.pause_ratio_delta)
        ).clamp(0.0, 1.0)
    if "local_rate_trace" in compact:
        compact["local_rate_trace"] = _scale_centered_trace(
            compact["local_rate_trace"],
            scale=float(intervention.local_rate_scale),
            bias=float(intervention.local_rate_bias),
        )
    if "boundary_trace" in compact:
        compact["boundary_trace"] = (
            compact["boundary_trace"].float() * float(intervention.boundary_trace_scale)
            + float(intervention.boundary_trace_bias)
        ).clamp(0.0, 1.0)
    if {"global_rate", "pause_ratio"} <= compact.keys():
        compact["planner_ref_stats"] = torch.cat([compact["global_rate"], compact["pause_ratio"]], dim=-1)
    if {"local_rate_trace", "boundary_trace"} <= compact.keys():
        compact["planner_ref_trace"] = torch.cat([compact["local_rate_trace"], compact["boundary_trace"]], dim=-1)
    return compact


def collect_planner_surface_bundle(execution) -> dict[str, torch.Tensor]:
    planner = execution.planner
    return {
        "speech_budget_win": planner.speech_budget_win.detach(),
        "pause_budget_win": planner.pause_budget_win.detach(),
        "dur_shape_unit": planner.dur_shape_unit.detach(),
        "pause_shape_unit": planner.pause_shape_unit.detach(),
        "boundary_score_unit": planner.boundary_score_unit.detach(),
        "speech_exec": execution.speech_duration_exec.detach(),
        "pause_exec": getattr(execution, "blank_duration_exec", execution.pause_after_exec).detach(),
        "commit_frontier": execution.commit_frontier.detach().float().unsqueeze(-1),
    }


def _masked_relative_l1(pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    pred = pred.float()
    ref = ref.float()
    if mask is None:
        mask = torch.ones_like(pred)
    else:
        mask = mask.float()
        while mask.dim() < pred.dim():
            mask = mask.unsqueeze(-1)
    diff = (pred - ref).abs() * mask
    denom = (ref.abs() * mask).sum().clamp_min(1e-6)
    return diff.sum() / denom


def _masked_mean_abs(pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    pred = pred.float()
    ref = ref.float()
    if mask is None:
        mask = torch.ones_like(pred)
    else:
        mask = mask.float()
        while mask.dim() < pred.dim():
            mask = mask.unsqueeze(-1)
    denom = mask.sum().clamp_min(1.0)
    return ((pred - ref).abs() * mask).sum() / denom


def _masked_distribution_kl(pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred = (pred.float().clamp_min(0.0) * mask.float()).clamp_min(1e-6)
    ref = (ref.float().clamp_min(0.0) * mask.float()).clamp_min(1e-6)
    pred = pred / pred.sum(dim=1, keepdim=True).clamp_min(1e-6)
    ref = ref / ref.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return (ref * (torch.log(ref) - torch.log(pred))).sum(dim=1).mean()


def compute_surface_distance_report(
    baseline: dict[str, torch.Tensor],
    perturbed: dict[str, torch.Tensor],
    *,
    unit_mask: torch.Tensor,
) -> dict[str, float]:
    mask = unit_mask.float()
    report = {
        "speech_budget_rel_l1": float(_masked_relative_l1(perturbed["speech_budget_win"], baseline["speech_budget_win"])),
        "pause_budget_rel_l1": float(_masked_relative_l1(perturbed["pause_budget_win"], baseline["pause_budget_win"])),
        "dur_shape_l1": float(_masked_mean_abs(perturbed["dur_shape_unit"], baseline["dur_shape_unit"], mask)),
        "pause_shape_kl": float(_masked_distribution_kl(perturbed["pause_shape_unit"], baseline["pause_shape_unit"], mask)),
        "boundary_score_l1": float(_masked_mean_abs(perturbed["boundary_score_unit"], baseline["boundary_score_unit"], mask)),
        "speech_exec_rel_l1": float(_masked_relative_l1(perturbed["speech_exec"], baseline["speech_exec"], mask)),
        "pause_exec_rel_l1": float(_masked_relative_l1(perturbed["pause_exec"], baseline["pause_exec"], mask)),
        "speech_exec_shape_kl": float(_masked_distribution_kl(perturbed["speech_exec"], baseline["speech_exec"], mask)),
        "pause_exec_shape_kl": float(_masked_distribution_kl(perturbed["pause_exec"], baseline["pause_exec"], mask)),
        "commit_frontier_l1": float(_masked_mean_abs(perturbed["commit_frontier"], baseline["commit_frontier"])),
    }
    return report


__all__ = [
    "COMPACT_REFERENCE_KEYS",
    "COMPACT_REFERENCE_ALIAS_KEYS",
    "PLANNER_SURFACE_KEYS",
    "CompactPlannerIntervention",
    "apply_compact_reference_intervention",
    "clone_reference_conditioning",
    "collect_planner_surface_bundle",
    "compute_surface_distance_report",
    "extract_compact_reference_contract",
]
