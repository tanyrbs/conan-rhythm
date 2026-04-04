from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class BudgetProjectionRepairStats:
    raw_speech_budget: torch.Tensor
    raw_pause_budget: torch.Tensor
    effective_speech_budget: torch.Tensor
    effective_pause_budget: torch.Tensor
    total_shift: torch.Tensor
    total_shift_abs: torch.Tensor
    redistribution_mass: torch.Tensor
    repair_mass: torch.Tensor


def _flatten_batch_scalar(value: torch.Tensor) -> torch.Tensor:
    value = value.float()
    if value.dim() == 0:
        return value.view(1)
    if value.dim() == 1:
        return value
    return value.reshape(value.size(0), -1).mean(dim=1)


def _planner_budget_view(planner: Any, effective_name: str) -> torch.Tensor:
    fallback_name = effective_name.replace("effective_", "")
    effective_value = getattr(planner, effective_name, None)
    if effective_value is None:
        effective_value = getattr(planner, fallback_name)
    return _flatten_batch_scalar(effective_value)


def compute_budget_projection_repair_stats(planner: Any) -> BudgetProjectionRepairStats:
    effective_speech_budget = _planner_budget_view(planner, "effective_speech_budget_win")
    effective_pause_budget = _planner_budget_view(planner, "effective_pause_budget_win")
    raw_speech_budget = _flatten_batch_scalar(
        getattr(planner, "raw_speech_budget_win", getattr(planner, "speech_budget_win"))
    )
    raw_pause_budget = _flatten_batch_scalar(
        getattr(planner, "raw_pause_budget_win", getattr(planner, "pause_budget_win"))
    )

    speech_shift = effective_speech_budget - raw_speech_budget
    pause_shift = effective_pause_budget - raw_pause_budget
    total_shift = speech_shift + pause_shift
    total_shift_abs = total_shift.abs()
    view_gap = speech_shift.abs() + pause_shift.abs()
    redistribution_mass = 0.5 * (view_gap - total_shift_abs).clamp_min(0.0)
    repair_mass = 0.5 * (view_gap + total_shift_abs)

    feasible_total_budget_delta = getattr(planner, "feasible_total_budget_delta", None)
    if feasible_total_budget_delta is not None:
        feasible_total_budget_delta = _flatten_batch_scalar(feasible_total_budget_delta).abs()
        repair_mass = torch.maximum(repair_mass, feasible_total_budget_delta)
        total_shift_abs = torch.maximum(total_shift_abs, feasible_total_budget_delta)

    return BudgetProjectionRepairStats(
        raw_speech_budget=raw_speech_budget,
        raw_pause_budget=raw_pause_budget,
        effective_speech_budget=effective_speech_budget,
        effective_pause_budget=effective_pause_budget,
        total_shift=total_shift,
        total_shift_abs=total_shift_abs,
        redistribution_mass=redistribution_mass,
        repair_mass=repair_mass,
    )


__all__ = ["BudgetProjectionRepairStats", "compute_budget_projection_repair_stats"]
