from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from modules.Conan.rhythm.prefix_state import (
    build_prefix_state_from_exec_torch,
    normalize_prefix_state_torch,
)
from modules.Conan.rhythm.source_boundary import resolve_boundary_score_unit


@dataclass
class RhythmLossTargets:
    speech_exec_tgt: torch.Tensor
    pause_exec_tgt: torch.Tensor
    speech_budget_tgt: torch.Tensor
    pause_budget_tgt: torch.Tensor
    unit_mask: torch.Tensor
    dur_anchor_src: torch.Tensor
    plan_local_weight: float = 0.5
    plan_cum_weight: float = 1.0
    sample_confidence: Optional[torch.Tensor] = None
    guidance_speech_tgt: Optional[torch.Tensor] = None
    guidance_pause_tgt: Optional[torch.Tensor] = None
    guidance_confidence: Optional[torch.Tensor] = None
    distill_speech_tgt: Optional[torch.Tensor] = None
    distill_pause_tgt: Optional[torch.Tensor] = None
    distill_speech_budget_tgt: Optional[torch.Tensor] = None
    distill_pause_budget_tgt: Optional[torch.Tensor] = None
    distill_allocation_tgt: Optional[torch.Tensor] = None
    distill_prefix_clock_tgt: Optional[torch.Tensor] = None
    distill_prefix_backlog_tgt: Optional[torch.Tensor] = None
    distill_confidence: Optional[torch.Tensor] = None
    distill_exec_confidence: Optional[torch.Tensor] = None
    distill_budget_confidence: Optional[torch.Tensor] = None
    distill_prefix_confidence: Optional[torch.Tensor] = None
    distill_allocation_confidence: Optional[torch.Tensor] = None
    distill_shape_confidence: Optional[torch.Tensor] = None
    distill_budget_weight: float = 1.0
    distill_allocation_weight: float = 1.0
    distill_prefix_weight: float = 1.0
    distill_speech_shape_weight: float = 0.0
    distill_pause_shape_weight: float = 0.0
    budget_raw_weight: float = 1.0
    budget_exec_weight: float = 0.25
    pause_boundary_weight: float = 0.35
    feasible_debt_weight: float = 0.05

    @property
    def blank_exec_tgt(self) -> torch.Tensor:
        return self.pause_exec_tgt

    @property
    def blank_budget_tgt(self) -> torch.Tensor:
        return self.pause_budget_tgt


def _prepare_batch_weight(weight: Optional[torch.Tensor], ref: torch.Tensor) -> Optional[torch.Tensor]:
    if weight is None:
        return None
    weight = weight.float()
    if weight.dim() == 0:
        return weight.view(1).expand(ref.size(0)).clamp(min=0.0, max=1.0)
    return weight.reshape(weight.size(0), -1)[:, 0].clamp(min=0.0, max=1.0)


def _reduce_batch_loss(loss: torch.Tensor, batch_weight: Optional[torch.Tensor]) -> torch.Tensor:
    if batch_weight is None:
        return loss.mean()
    batch_weight = _prepare_batch_weight(batch_weight, loss)
    return (loss * batch_weight).sum() / batch_weight.sum().clamp_min(1e-6)


def _resolve_component_batch_weight(
    component_weight: Optional[torch.Tensor],
    fallback_weight: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    return component_weight if component_weight is not None else fallback_weight


def _merge_batch_weight(
    base_weight: Optional[torch.Tensor],
    gate_weight: Optional[torch.Tensor],
    ref: torch.Tensor,
) -> Optional[torch.Tensor]:
    merged = _prepare_batch_weight(base_weight, ref) if base_weight is not None else None
    if gate_weight is None:
        return merged
    gate = _prepare_batch_weight(gate_weight, ref)
    return gate if merged is None else merged * gate


def _masked_huber(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 1.0,
    batch_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    mask = mask.float()
    loss = F.smooth_l1_loss(pred, tgt, beta=beta, reduction='none')
    while mask.dim() < loss.dim():
        mask = mask.unsqueeze(-1)
    reduce_dims = tuple(range(1, loss.dim()))
    masked_loss = (loss * mask).sum(dim=reduce_dims)
    masked_denom = mask.sum(dim=reduce_dims).clamp_min(1.0)
    return _reduce_batch_loss(masked_loss / masked_denom, batch_weight)


def _masked_log_huber(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 0.5,
    batch_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return _masked_huber(
        torch.log1p(pred.float().clamp_min(0.0)),
        torch.log1p(tgt.float().clamp_min(0.0)),
        mask,
        beta=beta,
        batch_weight=batch_weight,
    )


def _batch_l1(pred: torch.Tensor, tgt: torch.Tensor, batch_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    loss = F.l1_loss(pred, tgt, reduction='none')
    if loss.dim() > 1:
        loss = loss.reshape(loss.size(0), -1).mean(dim=1)
    return _reduce_batch_loss(loss, batch_weight)


def _masked_cumsum(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(values * mask.float(), dim=1)


def _masked_normalize(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    mask_mass = mask.sum(dim=1, keepdim=True)
    values = values.float().clamp_min(0.0) * mask
    total = values.sum(dim=1, keepdim=True)
    normalized = values / total.clamp_min(1e-6)
    masked_uniform = mask / mask_mass.clamp_min(1.0)
    dense_uniform = torch.full_like(values, 1.0 / max(values.size(1), 1))
    fallback = torch.where(mask_mass > 0.0, masked_uniform, dense_uniform)
    return torch.where(total > 1e-6, normalized, fallback)


def _masked_probability_distribution(
    values: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    mask = mask.float()
    mask_mass = mask.sum(dim=1, keepdim=True)
    probs = _masked_normalize(values, mask)
    masked_probs = probs * mask + eps * mask
    masked_probs = masked_probs / masked_probs.sum(dim=1, keepdim=True).clamp_min(eps)
    dense_uniform = torch.full_like(masked_probs, 1.0 / max(masked_probs.size(1), 1))
    return torch.where(mask_mass > 0.0, masked_probs, dense_uniform)


def _positive_mass_gate(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((values.float() * mask.float()).sum(dim=1) > 1e-6).float()


def _resolve_budget_views(execution) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    planner = execution.planner
    raw_speech = getattr(planner, "raw_speech_budget_win", planner.speech_budget_win).float()
    raw_pause = getattr(planner, "raw_pause_budget_win", planner.pause_budget_win).float()
    exec_speech = planner.speech_budget_win.float()
    exec_pause = planner.pause_budget_win.float()
    return raw_speech, raw_pause, exec_speech, exec_pause


def _budget_surfaces_from_speech_pause(
    speech_budget: torch.Tensor,
    pause_budget: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    speech_budget = speech_budget.float().clamp_min(0.0)
    pause_budget = pause_budget.float().clamp_min(0.0)
    total_budget = speech_budget + pause_budget
    pause_share = pause_budget / total_budget.clamp_min(1e-6)
    pause_share = torch.where(total_budget > 1e-6, pause_share, torch.zeros_like(pause_share))
    return total_budget, pause_share


def _budget_surface_loss(
    pred_speech_budget: torch.Tensor,
    pred_pause_budget: torch.Tensor,
    tgt_speech_budget: torch.Tensor,
    tgt_pause_budget: torch.Tensor,
    batch_weight: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_total_budget, pred_pause_share = _budget_surfaces_from_speech_pause(
        pred_speech_budget,
        pred_pause_budget,
    )
    tgt_total_budget, tgt_pause_share = _budget_surfaces_from_speech_pause(
        tgt_speech_budget,
        tgt_pause_budget,
    )
    total_loss = _batch_l1(
        torch.log1p(pred_total_budget),
        torch.log1p(tgt_total_budget),
        batch_weight=batch_weight,
    )
    pause_share_loss = _batch_l1(
        pred_pause_share,
        tgt_pause_share,
        batch_weight=batch_weight,
    )
    return total_loss + pause_share_loss, total_loss, pause_share_loss


def _compute_budget_supervision(
    execution,
    *,
    speech_budget_tgt: torch.Tensor,
    pause_budget_tgt: torch.Tensor,
    raw_weight: float,
    exec_weight: float,
    batch_weight: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return weighted budget-view contributions plus compact surface splits.

    `rhythm_budget_raw_surface` / `rhythm_budget_exec_surface` should match the
    actual optimizer contribution of each budget view, not the unweighted
    diagnostic loss. This keeps the logged surfaces additive with `L_budget`.
    """
    raw_speech, raw_pause, exec_speech, exec_pause = _resolve_budget_views(execution)
    raw_loss, raw_total_loss, raw_pause_share_loss = _budget_surface_loss(
        raw_speech,
        raw_pause,
        speech_budget_tgt,
        pause_budget_tgt,
        batch_weight=batch_weight,
    )
    exec_loss, exec_total_loss, exec_pause_share_loss = _budget_surface_loss(
        exec_speech,
        exec_pause,
        speech_budget_tgt,
        pause_budget_tgt,
        batch_weight=batch_weight,
    )
    raw_contrib = float(raw_weight) * raw_loss
    exec_contrib = float(exec_weight) * exec_loss
    total_surface = float(raw_weight) * raw_total_loss + float(exec_weight) * exec_total_loss
    pause_share_surface = (
        float(raw_weight) * raw_pause_share_loss + float(exec_weight) * exec_pause_share_loss
    )
    total_loss = raw_contrib + exec_contrib
    return total_loss, raw_contrib, exec_contrib, total_surface, pause_share_surface


def _batch_kl_div(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    mask: torch.Tensor,
    batch_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    pred = _masked_probability_distribution(pred, mask)
    tgt = _masked_probability_distribution(tgt, mask)
    support = mask.float()
    while support.dim() < pred.dim():
        support = support.unsqueeze(-1)
    pred = pred.clamp_min(1e-6)
    tgt = tgt.clamp_min(1e-6)
    reduce_dims = tuple(range(1, pred.dim()))
    loss = (tgt * (torch.log(tgt) - torch.log(pred)) * support).sum(dim=reduce_dims)
    return _reduce_batch_loss(loss, batch_weight)


def _masked_cumulative_fraction(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    cumulative = _masked_cumsum(values.float(), mask)
    total = (values.float() * mask).sum(dim=1, keepdim=True).clamp_min(1e-6)
    return (cumulative / total) * mask


def _resolve_pause_exec_mask(
    execution,
    unit_mask: torch.Tensor,
    *,
    boundary_weight: float,
) -> torch.Tensor:
    """Keep pause supervision on executed surface, but upweight boundary-like units."""
    unit_mask = unit_mask.float()
    if boundary_weight <= 0.0 or execution is None or getattr(execution, "planner", None) is None:
        return unit_mask
    planner = execution.planner
    boundary_hint = getattr(planner, "source_boundary_cue", None)
    if boundary_hint is None:
        boundary_hint = resolve_boundary_score_unit(planner)
        if boundary_hint is not None:
            boundary_hint = boundary_hint.detach()
    if boundary_hint is None:
        return unit_mask
    boundary_hint = boundary_hint.float().clamp_min(0.0) * unit_mask
    visible = unit_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    boundary_mean = boundary_hint.sum(dim=1, keepdim=True) / visible
    boundary_excess = (boundary_hint - boundary_mean).clamp_min(0.0)
    boundary_scale = boundary_excess.amax(dim=1, keepdim=True).clamp_min(1e-6)
    boundary_norm = boundary_excess / boundary_scale
    return unit_mask * (1.0 + float(boundary_weight) * boundary_norm)


def _compute_feasible_debt_penalty(
    execution,
    *,
    dur_anchor_src: Optional[torch.Tensor] = None,
    unit_mask: Optional[torch.Tensor] = None,
    batch_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if execution is None or getattr(execution, "planner", None) is None:
        ref = execution.speech_duration_exec if execution is not None else torch.zeros(1)
        return ref.new_tensor(0.0)
    planner = execution.planner
    feasible_total = getattr(planner, "feasible_total_budget_delta", None)
    if feasible_total is None:
        feasible_speech = getattr(planner, "feasible_speech_budget_delta", None)
        feasible_pause = getattr(planner, "feasible_pause_budget_delta", None)
        if feasible_speech is not None and feasible_pause is not None:
            feasible_total = feasible_speech + feasible_pause
        else:
            return execution.speech_duration_exec.new_tensor(0.0)
    if feasible_total.dim() > 1:
        feasible_total = feasible_total.reshape(feasible_total.size(0), -1).mean(dim=1)
    feasible_total = feasible_total.float().clamp_min(0.0)
    if dur_anchor_src is not None and unit_mask is not None:
        source_total = (dur_anchor_src.float() * unit_mask.float()).sum(dim=1).clamp_min(1.0)
        feasible_total = feasible_total / source_total
    debt = torch.log1p(feasible_total)
    return _reduce_batch_loss(debt, batch_weight)


def build_rhythm_loss_dict(execution, targets: RhythmLossTargets) -> dict[str, torch.Tensor]:
    unit_mask = targets.unit_mask.float()
    blank_exec = getattr(execution, "blank_duration_exec", execution.pause_after_exec)
    pred_prefix_clock, pred_prefix_backlog = build_prefix_state_from_exec_torch(
        speech_exec=execution.speech_duration_exec.float(),
        pause_exec=blank_exec.float(),
        dur_anchor_src=targets.dur_anchor_src.float(),
        unit_mask=unit_mask,
    )
    target_prefix_clock, target_prefix_backlog = build_prefix_state_from_exec_torch(
        speech_exec=targets.speech_exec_tgt.float(),
        pause_exec=targets.pause_exec_tgt.float(),
        dur_anchor_src=targets.dur_anchor_src.float(),
        unit_mask=unit_mask,
    )
    l_exec_speech = _masked_log_huber(
        execution.speech_duration_exec,
        targets.speech_exec_tgt.float(),
        unit_mask,
        batch_weight=targets.sample_confidence,
    )
    pause_mask = _resolve_pause_exec_mask(
        execution,
        unit_mask,
        boundary_weight=float(targets.pause_boundary_weight),
    )
    l_exec_pause = _masked_log_huber(
        blank_exec,
        targets.pause_exec_tgt.float(),
        pause_mask,
        batch_weight=targets.sample_confidence,
    )
    (
        l_budget,
        l_budget_raw,
        l_budget_exec,
        l_budget_total_surface,
        l_budget_pause_share_surface,
    ) = _compute_budget_supervision(
        execution,
        speech_budget_tgt=targets.speech_budget_tgt.float(),
        pause_budget_tgt=targets.pause_budget_tgt.float(),
        raw_weight=float(targets.budget_raw_weight),
        exec_weight=float(targets.budget_exec_weight),
        batch_weight=targets.sample_confidence,
    )
    l_feasible_debt = _compute_feasible_debt_penalty(
        execution,
        dur_anchor_src=targets.dur_anchor_src,
        unit_mask=unit_mask,
        batch_weight=targets.sample_confidence,
    )
    l_budget = l_budget + float(targets.feasible_debt_weight) * l_feasible_debt
    pred_prefix_clock_norm = normalize_prefix_state_torch(
        pred_prefix_clock,
        dur_anchor_src=targets.dur_anchor_src.float(),
        unit_mask=unit_mask,
    )
    target_prefix_clock_norm = normalize_prefix_state_torch(
        target_prefix_clock,
        dur_anchor_src=targets.dur_anchor_src.float(),
        unit_mask=unit_mask,
    )
    pred_prefix_backlog_norm = normalize_prefix_state_torch(
        pred_prefix_backlog,
        dur_anchor_src=targets.dur_anchor_src.float(),
        unit_mask=unit_mask,
    )
    target_prefix_backlog_norm = normalize_prefix_state_torch(
        target_prefix_backlog,
        dur_anchor_src=targets.dur_anchor_src.float(),
        unit_mask=unit_mask,
    )
    l_prefix_clock = _masked_huber(
        pred_prefix_clock_norm,
        target_prefix_clock_norm,
        unit_mask,
        beta=0.25,
        batch_weight=targets.sample_confidence,
    )
    l_prefix_backlog = _masked_huber(
        pred_prefix_backlog_norm,
        target_prefix_backlog_norm,
        unit_mask,
        beta=0.25,
        batch_weight=targets.sample_confidence,
    )
    l_carry = l_prefix_clock + l_prefix_backlog
    l_plan_local = execution.speech_duration_exec.new_tensor(0.0)
    l_plan_cum = execution.speech_duration_exec.new_tensor(0.0)
    l_plan = execution.speech_duration_exec.new_tensor(0.0)
    if float(targets.plan_local_weight) > 0.0 or float(targets.plan_cum_weight) > 0.0:
        exec_total = (execution.speech_duration_exec + blank_exec).float()
        target_total = (targets.speech_exec_tgt + targets.pause_exec_tgt).float()
        plan_shape_weight = _merge_batch_weight(
            targets.sample_confidence,
            _positive_mass_gate(target_total, unit_mask),
            exec_total,
        )
        l_plan_local = _batch_kl_div(
            exec_total,
            target_total,
            unit_mask,
            batch_weight=plan_shape_weight,
        )
        l_plan_cum = _masked_huber(
            _masked_cumulative_fraction(exec_total, unit_mask),
            _masked_cumulative_fraction(target_total, unit_mask),
            unit_mask,
            beta=0.10,
            batch_weight=plan_shape_weight,
        )
        l_plan = float(targets.plan_local_weight) * l_plan_local + float(targets.plan_cum_weight) * l_plan_cum
    if targets.guidance_speech_tgt is not None and targets.guidance_pause_tgt is not None:
        l_guidance = _masked_log_huber(
            execution.speech_duration_exec,
            targets.guidance_speech_tgt.float(),
            unit_mask,
            batch_weight=targets.guidance_confidence,
        ) + _masked_log_huber(
            blank_exec,
            targets.guidance_pause_tgt.float(),
            unit_mask,
            batch_weight=targets.guidance_confidence,
        )
    else:
        l_guidance = execution.speech_duration_exec.new_tensor(0.0)
    l_distill_exec = execution.speech_duration_exec.new_tensor(0.0)
    l_distill_budget = execution.speech_duration_exec.new_tensor(0.0)
    l_distill_budget_raw = execution.speech_duration_exec.new_tensor(0.0)
    l_distill_budget_exec = execution.speech_duration_exec.new_tensor(0.0)
    l_distill_budget_total_surface = execution.speech_duration_exec.new_tensor(0.0)
    l_distill_budget_pause_share_surface = execution.speech_duration_exec.new_tensor(0.0)
    l_distill_prefix = execution.speech_duration_exec.new_tensor(0.0)
    l_distill_allocation = execution.speech_duration_exec.new_tensor(0.0)
    l_distill_speech_shape = execution.speech_duration_exec.new_tensor(0.0)
    l_distill_pause_shape = execution.speech_duration_exec.new_tensor(0.0)
    if targets.distill_speech_tgt is not None and targets.distill_pause_tgt is not None:
        distill_exec_weight = _resolve_component_batch_weight(
            targets.distill_exec_confidence,
            targets.distill_confidence,
        )
        distill_budget_weight = _resolve_component_batch_weight(
            targets.distill_budget_confidence,
            targets.distill_confidence,
        )
        distill_prefix_weight = _resolve_component_batch_weight(
            targets.distill_prefix_confidence,
            targets.distill_confidence,
        )
        distill_allocation_weight = _resolve_component_batch_weight(
            targets.distill_allocation_confidence,
            targets.distill_confidence,
        )
        distill_shape_weight = _resolve_component_batch_weight(
            targets.distill_shape_confidence,
            distill_exec_weight,
        )
        l_distill_exec = _masked_log_huber(
            execution.speech_duration_exec,
            targets.distill_speech_tgt.float(),
            unit_mask,
            batch_weight=distill_exec_weight,
        ) + _masked_log_huber(
            blank_exec,
            targets.distill_pause_tgt.float(),
            unit_mask,
            batch_weight=distill_exec_weight,
        )
        l_distill = l_distill_exec
        if (
            float(targets.distill_budget_weight) > 0.0
            and targets.distill_speech_budget_tgt is not None
            and targets.distill_pause_budget_tgt is not None
        ):
            (
                l_distill_budget,
                l_distill_budget_raw,
                l_distill_budget_exec,
                l_distill_budget_total_surface,
                l_distill_budget_pause_share_surface,
            ) = _compute_budget_supervision(
                execution,
                speech_budget_tgt=targets.distill_speech_budget_tgt.float(),
                pause_budget_tgt=targets.distill_pause_budget_tgt.float(),
                raw_weight=float(targets.budget_raw_weight),
                exec_weight=float(targets.budget_exec_weight),
                batch_weight=distill_budget_weight,
            )
            l_distill = l_distill + float(targets.distill_budget_weight) * l_distill_budget
        if targets.distill_prefix_clock_tgt is not None or targets.distill_prefix_backlog_tgt is not None:
            l_distill_prefix = execution.speech_duration_exec.new_tensor(0.0)
            if targets.distill_prefix_clock_tgt is not None:
                l_distill_prefix = l_distill_prefix + _masked_huber(
                    pred_prefix_clock_norm,
                    normalize_prefix_state_torch(
                        targets.distill_prefix_clock_tgt.float(),
                        dur_anchor_src=targets.dur_anchor_src.float(),
                        unit_mask=unit_mask,
                    ),
                    unit_mask,
                    beta=0.25,
                    batch_weight=distill_prefix_weight,
                )
            if targets.distill_prefix_backlog_tgt is not None:
                l_distill_prefix = l_distill_prefix + _masked_huber(
                    pred_prefix_backlog_norm,
                    normalize_prefix_state_torch(
                        targets.distill_prefix_backlog_tgt.float(),
                        dur_anchor_src=targets.dur_anchor_src.float(),
                        unit_mask=unit_mask,
                    ),
                    unit_mask,
                    beta=0.25,
                    batch_weight=distill_prefix_weight,
                )
            l_distill = l_distill + float(targets.distill_prefix_weight) * l_distill_prefix
        if float(targets.distill_allocation_weight) > 0.0:
            allocation_target = targets.distill_allocation_tgt
            if allocation_target is None:
                allocation_target = targets.distill_speech_tgt.float() + targets.distill_pause_tgt.float()
            l_distill_allocation = _batch_kl_div(
                execution.speech_duration_exec + blank_exec,
                allocation_target.float(),
                unit_mask,
                batch_weight=distill_allocation_weight,
            )
            l_distill = l_distill + float(targets.distill_allocation_weight) * l_distill_allocation
        if float(targets.distill_speech_shape_weight) > 0.0:
            speech_shape_weight = _merge_batch_weight(
                distill_shape_weight,
                _positive_mass_gate(targets.distill_speech_tgt.float(), unit_mask),
                execution.speech_duration_exec,
            )
            l_distill_speech_shape = _batch_kl_div(
                execution.speech_duration_exec,
                targets.distill_speech_tgt.float(),
                unit_mask,
                batch_weight=speech_shape_weight,
            )
            l_distill = l_distill + float(targets.distill_speech_shape_weight) * l_distill_speech_shape
        if float(targets.distill_pause_shape_weight) > 0.0:
            pause_shape_weight = _merge_batch_weight(
                distill_shape_weight,
                _positive_mass_gate(targets.distill_pause_tgt.float(), unit_mask),
                blank_exec,
            )
            l_distill_pause_shape = _batch_kl_div(
                blank_exec,
                targets.distill_pause_tgt.float(),
                unit_mask,
                batch_weight=pause_shape_weight,
            )
            l_distill = l_distill + float(targets.distill_pause_shape_weight) * l_distill_pause_shape
    else:
        l_distill = execution.speech_duration_exec.new_tensor(0.0)
    # Maintained compatibility alias: public cumplan/prefix_state supervision maps
    # to prefix carry/backlog alignment, while `rhythm_plan_cum` is the separate
    # cumulative execution proxy loss.
    return {
        'rhythm_exec_speech': l_exec_speech,
        'rhythm_exec_pause': l_exec_pause,
        'rhythm_budget': l_budget,
        'rhythm_budget_raw_surface': l_budget_raw,
        'rhythm_budget_exec_surface': l_budget_exec,
        'rhythm_budget_total_surface': l_budget_total_surface,
        'rhythm_budget_pause_share_surface': l_budget_pause_share_surface,
        'rhythm_feasible_debt': l_feasible_debt,
        'rhythm_prefix_clock': l_prefix_clock,
        'rhythm_prefix_backlog': l_prefix_backlog,
        'rhythm_prefix_state': l_carry,
        'rhythm_carry': l_carry.detach(),
        'rhythm_cumplan': l_carry.detach(),
        'rhythm_plan_local': l_plan_local,
        'rhythm_plan_cum': l_plan_cum,
        'rhythm_plan': l_plan,
        'rhythm_guidance': l_guidance,
        'rhythm_distill_exec': l_distill_exec,
        'rhythm_distill_budget': l_distill_budget,
        'rhythm_distill_budget_raw_surface': l_distill_budget_raw,
        'rhythm_distill_budget_exec_surface': l_distill_budget_exec,
        'rhythm_distill_budget_total_surface': l_distill_budget_total_surface,
        'rhythm_distill_budget_pause_share_surface': l_distill_budget_pause_share_surface,
        'rhythm_distill_prefix': l_distill_prefix,
        'rhythm_distill_speech_shape': l_distill_speech_shape,
        'rhythm_distill_pause_shape': l_distill_pause_shape,
        'rhythm_distill_allocation': l_distill_allocation,
        'rhythm_distill': l_distill,
        'rhythm_total': l_exec_speech + l_exec_pause + l_budget + l_carry + l_plan + l_guidance + l_distill,
    }
