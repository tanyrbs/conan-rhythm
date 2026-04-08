from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .reference_descriptor import global_rate_to_mean_speech_frames
from .source_boundary import resolve_boundary_score_unit


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


INTERVENTION_REBUILD_KEYS = (
    "slow_rhythm_memory",
    "slow_rhythm_summary",
    "slow_rhythm_summary_source",
    "planner_slow_rhythm_memory",
    "planner_slow_rhythm_summary",
    "planner_slow_rhythm_summary_source",
    "selector_meta_indices",
    "selector_meta_scores",
    "selector_meta_starts",
    "selector_meta_ends",
    "ref_phrase_trace",
    "planner_ref_phrase_trace",
    "ref_phrase_valid",
    "ref_phrase_lengths",
    "ref_phrase_starts",
    "ref_phrase_ends",
    "ref_phrase_boundary_strength",
    "ref_phrase_stats",
)

_REF_STATS_PAUSE_IDX = 0
_REF_STATS_MEAN_SPEECH_IDX = 2
_REF_TRACE_LOCAL_RATE_IDX = 1
_REF_TRACE_BOUNDARY_IDX = 2


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


def _refresh_compact_reference_aliases(ref_conditioning: dict[str, Any]) -> dict[str, Any]:
    if {"global_rate", "pause_ratio"} <= ref_conditioning.keys():
        ref_conditioning["planner_ref_stats"] = torch.cat(
            [ref_conditioning["global_rate"], ref_conditioning["pause_ratio"]],
            dim=-1,
        )
    elif "planner_ref_stats" in ref_conditioning and not isinstance(ref_conditioning["planner_ref_stats"], torch.Tensor):
        ref_conditioning.pop("planner_ref_stats", None)
    if {"local_rate_trace", "boundary_trace"} <= ref_conditioning.keys():
        ref_conditioning["planner_ref_trace"] = torch.cat(
            [ref_conditioning["local_rate_trace"], ref_conditioning["boundary_trace"]],
            dim=-1,
        )
    elif "planner_ref_trace" in ref_conditioning and not isinstance(ref_conditioning["planner_ref_trace"], torch.Tensor):
        ref_conditioning.pop("planner_ref_trace", None)
    return ref_conditioning


def extract_compact_reference_contract(ref_conditioning: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    compact = {
        key: ref_conditioning[key]
        for key in COMPACT_REFERENCE_KEYS
        if key in ref_conditioning and isinstance(ref_conditioning[key], torch.Tensor)
    }
    for key in COMPACT_REFERENCE_ALIAS_KEYS:
        if key not in compact and isinstance(ref_conditioning.get(key), torch.Tensor):
            compact[key] = ref_conditioning[key]
    return _refresh_compact_reference_aliases(compact)


def _scale_centered_trace(trace: torch.Tensor, scale: float, bias: float = 0.0) -> torch.Tensor:
    trace = trace.float()
    center = trace.mean(dim=1, keepdim=True)
    adjusted = center + (trace - center) * float(scale)
    if abs(float(bias)) > 1e-8:
        adjusted = adjusted + float(bias)
    return adjusted


def _ensure_channel_last(value: torch.Tensor) -> torch.Tensor:
    if value.dim() == 2:
        return value.unsqueeze(-1)
    return value


def _sync_raw_reference_contract(ref_conditioning: dict[str, Any]) -> None:
    stats = ref_conditioning.get("ref_rhythm_stats")
    if isinstance(stats, torch.Tensor) and stats.dim() == 2 and stats.size(-1) > _REF_STATS_MEAN_SPEECH_IDX:
        synced_stats = stats.clone()
        batch_size = synced_stats.size(0)
        pause_ratio = ref_conditioning.get("pause_ratio")
        if isinstance(pause_ratio, torch.Tensor):
            synced_stats[:, _REF_STATS_PAUSE_IDX:_REF_STATS_PAUSE_IDX + 1] = pause_ratio.float().reshape(
                batch_size, -1
            )[:, :1].to(device=synced_stats.device, dtype=synced_stats.dtype)
        global_rate = ref_conditioning.get("global_rate")
        if isinstance(global_rate, torch.Tensor):
            mean_speech = global_rate_to_mean_speech_frames(global_rate).reshape(batch_size, -1)[:, :1]
            synced_stats[:, _REF_STATS_MEAN_SPEECH_IDX:_REF_STATS_MEAN_SPEECH_IDX + 1] = mean_speech.to(
                device=synced_stats.device,
                dtype=synced_stats.dtype,
            )
        ref_conditioning["ref_rhythm_stats"] = synced_stats

    trace = ref_conditioning.get("ref_rhythm_trace")
    if isinstance(trace, torch.Tensor) and trace.dim() == 3 and trace.size(-1) > _REF_TRACE_BOUNDARY_IDX:
        synced_trace = trace.clone()
        batch_size, trace_bins = synced_trace.size(0), synced_trace.size(1)
        local_rate_trace = ref_conditioning.get("local_rate_trace")
        if isinstance(local_rate_trace, torch.Tensor):
            local_rate_trace = _ensure_channel_last(local_rate_trace).float().reshape(batch_size, trace_bins, -1)[:, :, :1]
            synced_trace[:, :, _REF_TRACE_LOCAL_RATE_IDX:_REF_TRACE_LOCAL_RATE_IDX + 1] = local_rate_trace.to(
                device=synced_trace.device,
                dtype=synced_trace.dtype,
            )
        boundary_trace = ref_conditioning.get("boundary_trace")
        if isinstance(boundary_trace, torch.Tensor):
            boundary_trace = _ensure_channel_last(boundary_trace).float().reshape(batch_size, trace_bins, -1)[:, :, :1]
            synced_trace[:, :, _REF_TRACE_BOUNDARY_IDX:_REF_TRACE_BOUNDARY_IDX + 1] = boundary_trace.to(
                device=synced_trace.device,
                dtype=synced_trace.dtype,
            )
        ref_conditioning["ref_rhythm_trace"] = synced_trace


def _drop_stale_intervention_sidecars(ref_conditioning: dict[str, Any]) -> None:
    for key in INTERVENTION_REBUILD_KEYS:
        ref_conditioning.pop(key, None)


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
    _refresh_compact_reference_aliases(compact)
    _sync_raw_reference_contract(compact)
    _drop_stale_intervention_sidecars(compact)
    return compact


def collect_planner_surface_bundle(execution) -> dict[str, torch.Tensor]:
    planner = execution.planner
    boundary_score_unit = resolve_boundary_score_unit(planner)
    if boundary_score_unit is None:
        boundary_score_unit = planner.pause_shape_unit.detach().new_zeros(planner.pause_shape_unit.shape)
    else:
        boundary_score_unit = boundary_score_unit.detach()
    return {
        "speech_budget_win": planner.speech_budget_win.detach(),
        "pause_budget_win": planner.pause_budget_win.detach(),
        "dur_shape_unit": planner.dur_shape_unit.detach(),
        "pause_shape_unit": planner.pause_shape_unit.detach(),
        "boundary_score_unit": boundary_score_unit,
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
