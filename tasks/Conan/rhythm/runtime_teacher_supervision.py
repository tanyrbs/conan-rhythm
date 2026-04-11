from __future__ import annotations

from types import SimpleNamespace

import torch

from modules.Conan.rhythm.source_boundary import resolve_boundary_score_unit
from tasks.Conan.rhythm.rhythm_v2.losses import RhythmLossTargets


def _normalize_optional_confidence(confidence, *, batch_size: int, device: torch.device) -> torch.Tensor:
    if confidence is None:
        return torch.ones((batch_size, 1), device=device)
    if isinstance(confidence, torch.Tensor):
        tensor = confidence.detach().float().reshape(batch_size, -1)[:, :1].to(device=device)
    else:
        tensor = torch.as_tensor(confidence, dtype=torch.float32, device=device).reshape(batch_size, -1)[:, :1]
    return tensor.clamp(min=0.0, max=1.0)


def _coerce_batch_scalar(value, *, batch_size: int, device: torch.device, fallback: torch.Tensor) -> torch.Tensor:
    if value is None:
        return fallback
    if isinstance(value, torch.Tensor):
        tensor = value.float().reshape(batch_size, -1)[:, :1].to(device=device)
    else:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device).reshape(batch_size, -1)[:, :1]
    return tensor


def _resolve_runtime_budget_ratio(
    execution,
    *,
    attr_name: str,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    planner = getattr(execution, "planner", None)
    fallback = torch.ones((batch_size, 1), dtype=torch.float32, device=device)
    if planner is None:
        return fallback
    return _coerce_batch_scalar(
        getattr(planner, attr_name, None),
        batch_size=batch_size,
        device=device,
        fallback=fallback,
    ).clamp_(0.0, 1.0)


def _prefix_budget_ratio(
    prefix_mass: torch.Tensor,
    full_mass: torch.Tensor,
    *,
    teacher_units: int,
    current_units: int,
) -> torch.Tensor:
    prefix_ratio = prefix_mass.new_full(
        prefix_mass.shape,
        float(teacher_units) / float(max(current_units, 1)),
    )
    valid_full = full_mass > 1e-6
    scaled = prefix_mass / full_mass.clamp_min(1e-6)
    return torch.where(valid_full, scaled, prefix_ratio).clamp_(0.0, 1.0)


def slice_runtime_teacher_execution(execution, *, teacher_units: int):
    if execution is None:
        return None
    current_units = int(execution.speech_duration_exec.size(1))
    if teacher_units >= current_units:
        return execution
    planner = getattr(execution, "planner", None)
    speech_exec = execution.speech_duration_exec[:, :teacher_units]
    blank_exec = getattr(execution, "blank_duration_exec", execution.pause_after_exec)[:, :teacher_units]
    pause_exec = execution.pause_after_exec[:, :teacher_units]
    speech_exec_prefix = speech_exec.float().sum(dim=1, keepdim=True)
    pause_exec_prefix = blank_exec.float().sum(dim=1, keepdim=True)
    if planner is not None:
        batch_size = speech_exec.size(0)
        device = speech_exec.device
        speech_exec_full = execution.speech_duration_exec.float().sum(dim=1, keepdim=True)
        pause_exec_full = getattr(execution, "blank_duration_exec", execution.pause_after_exec).float().sum(
            dim=1, keepdim=True
        )
        speech_ratio = _prefix_budget_ratio(
            speech_exec_prefix,
            speech_exec_full,
            teacher_units=teacher_units,
            current_units=current_units,
        )
        pause_ratio = _prefix_budget_ratio(
            pause_exec_prefix,
            pause_exec_full,
            teacher_units=teacher_units,
            current_units=current_units,
        )
        full_effective_speech_budget = _coerce_batch_scalar(
            getattr(planner, "effective_speech_budget_win", getattr(planner, "speech_budget_win", None)),
            batch_size=batch_size,
            device=device,
            fallback=speech_exec_prefix,
        )
        full_effective_pause_budget = _coerce_batch_scalar(
            getattr(planner, "effective_pause_budget_win", getattr(planner, "pause_budget_win", None)),
            batch_size=batch_size,
            device=device,
            fallback=pause_exec_prefix,
        )
        full_raw_speech_budget = _coerce_batch_scalar(
            getattr(planner, "raw_speech_budget_win", getattr(planner, "speech_budget_win", None)),
            batch_size=batch_size,
            device=device,
            fallback=_coerce_batch_scalar(
                getattr(planner, "speech_budget_win", None),
                batch_size=batch_size,
                device=device,
                fallback=speech_exec_prefix,
            ),
        )
        full_raw_pause_budget = _coerce_batch_scalar(
            getattr(planner, "raw_pause_budget_win", getattr(planner, "pause_budget_win", None)),
            batch_size=batch_size,
            device=device,
            fallback=_coerce_batch_scalar(
                getattr(planner, "pause_budget_win", None),
                batch_size=batch_size,
                device=device,
                fallback=pause_exec_prefix,
            ),
        )
        full_feasible_speech_delta = _coerce_batch_scalar(
            getattr(planner, "feasible_speech_budget_delta", None),
            batch_size=batch_size,
            device=device,
            fallback=full_effective_speech_budget.new_zeros(full_effective_speech_budget.shape),
        )
        full_feasible_pause_delta = _coerce_batch_scalar(
            getattr(planner, "feasible_pause_budget_delta", None),
            batch_size=batch_size,
            device=device,
            fallback=full_effective_pause_budget.new_zeros(full_effective_pause_budget.shape),
        )
        speech_budget = full_effective_speech_budget * speech_ratio
        pause_budget = full_effective_pause_budget * pause_ratio
        raw_speech_budget = full_raw_speech_budget * speech_ratio
        raw_pause_budget = full_raw_pause_budget * pause_ratio
        feasible_speech_delta = full_feasible_speech_delta * speech_ratio
        feasible_pause_delta = full_feasible_pause_delta * pause_ratio
        feasible_total_delta = feasible_speech_delta + feasible_pause_delta
        full_feasible_total_delta = full_feasible_speech_delta + full_feasible_pause_delta
    else:
        speech_budget = speech_exec_prefix
        pause_budget = pause_exec_prefix
        raw_speech_budget = speech_budget
        raw_pause_budget = pause_budget
        feasible_speech_delta = speech_budget.new_zeros(speech_budget.shape)
        feasible_pause_delta = pause_budget.new_zeros(pause_budget.shape)
        feasible_total_delta = speech_budget.new_zeros(speech_budget.shape)
        full_effective_speech_budget = speech_budget
        full_effective_pause_budget = pause_budget
        full_raw_speech_budget = raw_speech_budget
        full_raw_pause_budget = raw_pause_budget
        full_feasible_speech_delta = feasible_speech_delta
        full_feasible_pause_delta = feasible_pause_delta
        full_feasible_total_delta = feasible_total_delta
    boundary_score = resolve_boundary_score_unit(planner, fallback=speech_exec.new_zeros(speech_exec.shape))
    if torch.is_tensor(boundary_score):
        boundary_score = boundary_score[:, :teacher_units]
    planner_view = SimpleNamespace(
        speech_budget_win=speech_budget,
        pause_budget_win=pause_budget,
        raw_speech_budget_win=raw_speech_budget,
        raw_pause_budget_win=raw_pause_budget,
        effective_speech_budget_win=speech_budget,
        effective_pause_budget_win=pause_budget,
        blank_budget_win=pause_budget,
        boundary_score_unit=boundary_score,
        boundary_latent=boundary_score,
        source_boundary_cue=(
            planner.source_boundary_cue[:, :teacher_units]
            if planner is not None and torch.is_tensor(getattr(planner, "source_boundary_cue", None))
            else getattr(planner, "source_boundary_cue", None)
            if planner is not None
            else None
        ),
        feasible_speech_budget_delta=feasible_speech_delta,
        feasible_pause_budget_delta=feasible_pause_delta,
        feasible_total_budget_delta=feasible_total_delta,
        runtime_budget_slice_mode="proportional_prefix",
        runtime_budget_slice_semantics="auxiliary_proxy",
        runtime_budget_slice_preserves_raw_exec_semantics=False,
        runtime_budget_slice_ratio_speech=speech_ratio if planner is not None else speech_budget.new_ones(speech_budget.shape),
        runtime_budget_slice_ratio_pause=pause_ratio if planner is not None else pause_budget.new_ones(pause_budget.shape),
        full_effective_speech_budget_win=full_effective_speech_budget,
        full_effective_pause_budget_win=full_effective_pause_budget,
        full_raw_speech_budget_win=full_raw_speech_budget,
        full_raw_pause_budget_win=full_raw_pause_budget,
        full_feasible_speech_budget_delta=full_feasible_speech_delta,
        full_feasible_pause_budget_delta=full_feasible_pause_delta,
        full_feasible_total_budget_delta=full_feasible_total_delta,
    )
    return SimpleNamespace(
        speech_duration_exec=speech_exec,
        blank_duration_exec=blank_exec,
        pause_after_exec=pause_exec,
        planner=planner_view,
    )


def build_runtime_teacher_supervision_targets(
    *,
    output,
    sample,
    plan_local_weight: float,
    plan_cum_weight: float,
    pause_boundary_weight: float,
    budget_raw_weight: float,
    budget_exec_weight: float,
    feasible_debt_weight: float,
):
    runtime_teacher = output.get("rhythm_offline_execution")
    offline_unit_batch = output.get("rhythm_offline_unit_batch")
    unit_batch = output.get("rhythm_unit_batch")
    if runtime_teacher is None:
        return None
    if offline_unit_batch is not None and all(
        key in sample
        for key in (
            "rhythm_offline_teacher_speech_exec_tgt",
            "rhythm_offline_teacher_pause_exec_tgt",
        )
    ):
        batch_for_targets = offline_unit_batch
        speech_exec_key = "rhythm_offline_teacher_speech_exec_tgt"
        pause_exec_key = "rhythm_offline_teacher_pause_exec_tgt"
        speech_budget_key = "rhythm_offline_teacher_speech_budget_tgt"
        pause_budget_key = "rhythm_offline_teacher_pause_budget_tgt"
    else:
        batch_for_targets = unit_batch if unit_batch is not None else offline_unit_batch
        speech_exec_key = "rhythm_teacher_speech_exec_tgt"
        pause_exec_key = (
            "rhythm_teacher_pause_exec_tgt"
            if "rhythm_teacher_pause_exec_tgt" in sample
            else "rhythm_teacher_blank_exec_tgt"
        )
        speech_budget_key = "rhythm_teacher_speech_budget_tgt"
        pause_budget_key = (
            "rhythm_teacher_pause_budget_tgt"
            if "rhythm_teacher_pause_budget_tgt" in sample
            else "rhythm_teacher_blank_budget_tgt"
        )
    if batch_for_targets is None or not all(key in sample for key in (speech_exec_key, pause_exec_key)):
        return None
    teacher_units = min(
        int(runtime_teacher.speech_duration_exec.size(1)),
        int(batch_for_targets.dur_anchor_src.size(1)),
        int(sample[speech_exec_key].size(1)),
        int(sample[pause_exec_key].size(1)),
    )
    if teacher_units <= 0:
        return None
    unit_mask = batch_for_targets.unit_mask
    if unit_mask is None:
        unit_mask = batch_for_targets.dur_anchor_src.gt(0).float()
    speech_exec_tgt = sample[speech_exec_key][:, :teacher_units].float()
    pause_exec_tgt = sample[pause_exec_key][:, :teacher_units].float()
    teacher_execution = slice_runtime_teacher_execution(runtime_teacher, teacher_units=teacher_units)
    speech_budget_ratio = _resolve_runtime_budget_ratio(
        teacher_execution,
        attr_name="runtime_budget_slice_ratio_speech",
        batch_size=speech_exec_tgt.size(0),
        device=speech_exec_tgt.device,
    )
    pause_budget_ratio = _resolve_runtime_budget_ratio(
        teacher_execution,
        attr_name="runtime_budget_slice_ratio_pause",
        batch_size=speech_exec_tgt.size(0),
        device=speech_exec_tgt.device,
    )
    speech_budget_tgt = sample.get(speech_budget_key)
    pause_budget_tgt = sample.get(pause_budget_key)
    if speech_budget_tgt is None:
        speech_budget_tgt = speech_exec_tgt.sum(dim=1, keepdim=True)
    else:
        speech_budget_tgt = _coerce_batch_scalar(
            speech_budget_tgt,
            batch_size=speech_exec_tgt.size(0),
            device=speech_exec_tgt.device,
            fallback=speech_exec_tgt.sum(dim=1, keepdim=True),
        ) * speech_budget_ratio
    if pause_budget_tgt is None:
        pause_budget_tgt = pause_exec_tgt.sum(dim=1, keepdim=True)
    else:
        pause_budget_tgt = _coerce_batch_scalar(
            pause_budget_tgt,
            batch_size=pause_exec_tgt.size(0),
            device=pause_exec_tgt.device,
            fallback=pause_exec_tgt.sum(dim=1, keepdim=True),
        ) * pause_budget_ratio
    sample_confidence = sample.get(
        "rhythm_offline_teacher_confidence",
        sample.get("rhythm_teacher_confidence", sample.get("rhythm_target_confidence")),
    )
    sample_confidence = _normalize_optional_confidence(
        sample_confidence,
        batch_size=speech_exec_tgt.size(0),
        device=speech_exec_tgt.device,
    )
    targets = RhythmLossTargets(
        speech_exec_tgt=speech_exec_tgt,
        pause_exec_tgt=pause_exec_tgt,
        speech_budget_tgt=speech_budget_tgt,
        pause_budget_tgt=pause_budget_tgt,
        unit_mask=unit_mask[:, :teacher_units],
        dur_anchor_src=batch_for_targets.dur_anchor_src[:, :teacher_units],
        unit_logratio_weight=0.0,
        plan_local_weight=float(plan_local_weight),
        plan_cum_weight=float(plan_cum_weight),
        plan_segment_shape_weight=0.0,
        plan_pause_release_weight=0.0,
        sample_confidence=sample_confidence,
        pause_boundary_weight=float(pause_boundary_weight),
        budget_raw_weight=float(budget_raw_weight),
        budget_exec_weight=float(budget_exec_weight),
        feasible_debt_weight=float(feasible_debt_weight),
    )
    return teacher_execution, targets


__all__ = [
    "build_runtime_teacher_supervision_targets",
    "slice_runtime_teacher_execution",
]
