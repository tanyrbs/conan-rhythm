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
from tasks.Conan.rhythm.budget_repair import compute_budget_projection_repair_stats


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
    distill_exec_weight: float = 1.0
    distill_budget_weight: float = 1.0
    distill_allocation_weight: float = 1.0
    distill_prefix_weight: float = 1.0
    distill_speech_shape_weight: float = 0.0
    distill_pause_shape_weight: float = 0.0
    distill_same_source_exec: bool = False
    distill_same_source_budget: bool = False
    distill_same_source_prefix: bool = False
    distill_same_source_allocation: bool = False
    distill_same_source_shape: bool = False
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


@dataclass(frozen=True)
class RhythmLossState:
    unit_mask: torch.Tensor
    blank_exec: torch.Tensor
    pred_prefix_clock_norm: torch.Tensor
    pred_prefix_backlog_norm: torch.Tensor


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


def _reduce_batch_loss_with_scale(
    loss: torch.Tensor,
    batch_weight: Optional[torch.Tensor],
    loss_scale: Optional[torch.Tensor],
) -> torch.Tensor:
    if loss_scale is None:
        return _reduce_batch_loss(loss, batch_weight)
    loss_scale = _prepare_batch_weight(loss_scale, loss)
    if batch_weight is None:
        return (loss * loss_scale).mean()
    batch_weight = _prepare_batch_weight(batch_weight, loss)
    return (loss * batch_weight * loss_scale).sum() / batch_weight.sum().clamp_min(1e-6)


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


def _resolve_exec_total_budget_gate(
    execution,
    *,
    tgt_speech_budget: torch.Tensor,
    tgt_pause_budget: torch.Tensor,
) -> Optional[torch.Tensor]:
    planner = getattr(execution, "planner", None)
    if planner is None:
        return None
    feasible_total = getattr(planner, "feasible_total_budget_delta", None)
    if feasible_total is None:
        return None
    feasible_total = feasible_total.detach().float()
    if feasible_total.dim() == 0:
        feasible_total = feasible_total.view(1, 1)
    else:
        feasible_total = feasible_total.reshape(feasible_total.size(0), -1)[:, :1]
    feasible_total = feasible_total.clamp_min(0.0)
    tgt_total_budget, _ = _budget_surfaces_from_speech_pause(
        tgt_speech_budget,
        tgt_pause_budget,
    )
    gate = tgt_total_budget.float().clamp_min(0.0) / (
        tgt_total_budget.float().clamp_min(0.0) + feasible_total
    ).clamp_min(1e-6)
    return gate.detach().clamp(min=0.0, max=1.0)


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
    _, exec_pause_share = _budget_surfaces_from_speech_pause(exec_speech, exec_pause)
    tgt_total_budget, tgt_pause_share = _budget_surfaces_from_speech_pause(
        speech_budget_tgt,
        pause_budget_tgt,
    )
    exec_total_gate = _resolve_exec_total_budget_gate(
        execution,
        tgt_speech_budget=speech_budget_tgt,
        tgt_pause_budget=pause_budget_tgt,
    )
    exec_total_loss_vec = F.l1_loss(
        torch.log1p(exec_speech + exec_pause),
        torch.log1p(tgt_total_budget),
        reduction='none',
    )
    if exec_total_loss_vec.dim() > 1:
        exec_total_loss_vec = exec_total_loss_vec.reshape(exec_total_loss_vec.size(0), -1).mean(dim=1)
    exec_total_loss = _reduce_batch_loss_with_scale(
        exec_total_loss_vec,
        batch_weight=batch_weight,
        loss_scale=exec_total_gate,
    )
    exec_pause_share_loss = _batch_l1(
        exec_pause_share,
        tgt_pause_share,
        batch_weight=batch_weight,
    )
    exec_loss = exec_total_loss + exec_pause_share_loss
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
    repair_mass = compute_budget_projection_repair_stats(execution.planner).repair_mass
    if dur_anchor_src is not None and unit_mask is not None:
        source_total = (dur_anchor_src.float() * unit_mask.float()).sum(dim=1).clamp_min(1.0)
        repair_mass = repair_mass / source_total
    debt = torch.log1p(repair_mass)
    return _reduce_batch_loss(debt, batch_weight)


def _scalar_flag(ref: torch.Tensor, enabled: bool) -> torch.Tensor:
    return ref.new_tensor(1.0 if enabled else 0.0)


def _compute_plan_losses(
    execution,
    targets: RhythmLossTargets,
    state: RhythmLossState,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    zero = execution.speech_duration_exec.new_tensor(0.0)
    if float(targets.plan_local_weight) <= 0.0 and float(targets.plan_cum_weight) <= 0.0:
        return zero, zero, zero
    exec_total = (execution.speech_duration_exec + state.blank_exec).float()
    target_total = (targets.speech_exec_tgt + targets.pause_exec_tgt).float()
    plan_shape_weight = _merge_batch_weight(
        targets.sample_confidence,
        _positive_mass_gate(target_total, state.unit_mask),
        exec_total,
    )
    l_plan_local = _batch_kl_div(
        exec_total,
        target_total,
        state.unit_mask,
        batch_weight=plan_shape_weight,
    )
    l_plan_cum = _masked_huber(
        _masked_cumulative_fraction(exec_total, state.unit_mask),
        _masked_cumulative_fraction(target_total, state.unit_mask),
        state.unit_mask,
        beta=0.10,
        batch_weight=plan_shape_weight,
    )
    l_plan = float(targets.plan_local_weight) * l_plan_local + float(targets.plan_cum_weight) * l_plan_cum
    return l_plan_local, l_plan_cum, l_plan


def _compute_guidance_loss(
    execution,
    targets: RhythmLossTargets,
    state: RhythmLossState,
) -> torch.Tensor:
    if targets.guidance_speech_tgt is None or targets.guidance_pause_tgt is None:
        return execution.speech_duration_exec.new_tensor(0.0)
    return _masked_log_huber(
        execution.speech_duration_exec,
        targets.guidance_speech_tgt.float(),
        state.unit_mask,
        batch_weight=targets.guidance_confidence,
    ) + _masked_log_huber(
        state.blank_exec,
        targets.guidance_pause_tgt.float(),
        state.unit_mask,
        batch_weight=targets.guidance_confidence,
    )


def _compute_distill_prefix_loss(
    execution,
    targets: RhythmLossTargets,
    state: RhythmLossState,
    *,
    batch_weight: Optional[torch.Tensor],
) -> torch.Tensor:
    if targets.distill_prefix_clock_tgt is None and targets.distill_prefix_backlog_tgt is None:
        return execution.speech_duration_exec.new_tensor(0.0)
    loss = execution.speech_duration_exec.new_tensor(0.0)
    if targets.distill_prefix_clock_tgt is not None:
        loss = loss + _masked_huber(
            state.pred_prefix_clock_norm,
            normalize_prefix_state_torch(
                targets.distill_prefix_clock_tgt.float(),
                dur_anchor_src=targets.dur_anchor_src.float(),
                unit_mask=state.unit_mask,
            ),
            state.unit_mask,
            beta=0.25,
            batch_weight=batch_weight,
        )
    if targets.distill_prefix_backlog_tgt is not None:
        loss = loss + _masked_huber(
            state.pred_prefix_backlog_norm,
            normalize_prefix_state_torch(
                targets.distill_prefix_backlog_tgt.float(),
                dur_anchor_src=targets.dur_anchor_src.float(),
                unit_mask=state.unit_mask,
            ),
            state.unit_mask,
            beta=0.25,
            batch_weight=batch_weight,
        )
    return loss


def _compute_distill_allocation_loss(
    execution,
    targets: RhythmLossTargets,
    state: RhythmLossState,
    *,
    batch_weight: Optional[torch.Tensor],
) -> torch.Tensor:
    if float(targets.distill_allocation_weight) <= 0.0:
        return execution.speech_duration_exec.new_tensor(0.0)
    allocation_target = targets.distill_allocation_tgt
    if allocation_target is None:
        allocation_target = targets.distill_speech_tgt.float() + targets.distill_pause_tgt.float()
    allocation_shape_weight = _merge_batch_weight(
        batch_weight,
        _positive_mass_gate(allocation_target.float(), state.unit_mask),
        execution.speech_duration_exec + state.blank_exec,
    )
    return _batch_kl_div(
        execution.speech_duration_exec + state.blank_exec,
        allocation_target.float(),
        state.unit_mask,
        batch_weight=allocation_shape_weight,
    )


def _compute_distill_shape_losses(
    execution,
    targets: RhythmLossTargets,
    state: RhythmLossState,
    *,
    batch_weight: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    zero = execution.speech_duration_exec.new_tensor(0.0)
    speech_shape_loss = zero
    pause_shape_loss = zero
    if float(targets.distill_speech_shape_weight) > 0.0:
        speech_shape_weight = _merge_batch_weight(
            batch_weight,
            _positive_mass_gate(targets.distill_speech_tgt.float(), state.unit_mask),
            execution.speech_duration_exec,
        )
        speech_shape_loss = _batch_kl_div(
            execution.speech_duration_exec,
            targets.distill_speech_tgt.float(),
            state.unit_mask,
            batch_weight=speech_shape_weight,
        )
    if float(targets.distill_pause_shape_weight) > 0.0:
        pause_shape_weight = _merge_batch_weight(
            batch_weight,
            _positive_mass_gate(targets.distill_pause_tgt.float(), state.unit_mask),
            state.blank_exec,
        )
        pause_shape_loss = _batch_kl_div(
            state.blank_exec,
            targets.distill_pause_tgt.float(),
            state.unit_mask,
            batch_weight=pause_shape_weight,
        )
    return speech_shape_loss, pause_shape_loss


def _compute_distill_losses(
    execution,
    targets: RhythmLossTargets,
    state: RhythmLossState,
) -> dict[str, torch.Tensor]:
    zero = execution.speech_duration_exec.new_tensor(0.0)
    losses = {
        "rhythm_distill_exec": zero,
        "rhythm_distill_budget": zero,
        "rhythm_distill_budget_raw_surface": zero,
        "rhythm_distill_budget_exec_surface": zero,
        "rhythm_distill_budget_total_surface": zero,
        "rhythm_distill_budget_pause_share_surface": zero,
        "rhythm_distill_prefix": zero,
        "rhythm_distill_speech_shape": zero,
        "rhythm_distill_pause_shape": zero,
        "rhythm_distill_allocation": zero,
        "rhythm_distill": zero,
    }
    if targets.distill_speech_tgt is None or targets.distill_pause_tgt is None:
        return losses

    distill_exec_batch_weight = _resolve_component_batch_weight(
        targets.distill_exec_confidence,
        targets.distill_confidence,
    )
    distill_budget_batch_weight = _resolve_component_batch_weight(
        targets.distill_budget_confidence,
        targets.distill_confidence,
    )
    distill_prefix_batch_weight = _resolve_component_batch_weight(
        targets.distill_prefix_confidence,
        targets.distill_confidence,
    )
    distill_allocation_batch_weight = _resolve_component_batch_weight(
        targets.distill_allocation_confidence,
        targets.distill_confidence,
    )
    distill_shape_batch_weight = _resolve_component_batch_weight(
        targets.distill_shape_confidence,
        distill_exec_batch_weight,
    )
    total = zero
    if float(targets.distill_exec_weight) > 0.0:
        losses["rhythm_distill_exec"] = (
            _masked_log_huber(
                execution.speech_duration_exec,
                targets.distill_speech_tgt.float(),
                state.unit_mask,
                batch_weight=distill_exec_batch_weight,
            )
            + _masked_log_huber(
                state.blank_exec,
                targets.distill_pause_tgt.float(),
                state.unit_mask,
                batch_weight=distill_exec_batch_weight,
            )
        ) * float(targets.distill_exec_weight)
        total = total + losses["rhythm_distill_exec"]
    if (
        float(targets.distill_budget_weight) > 0.0
        and targets.distill_speech_budget_tgt is not None
        and targets.distill_pause_budget_tgt is not None
    ):
        (
            losses["rhythm_distill_budget"],
            losses["rhythm_distill_budget_raw_surface"],
            losses["rhythm_distill_budget_exec_surface"],
            losses["rhythm_distill_budget_total_surface"],
            losses["rhythm_distill_budget_pause_share_surface"],
        ) = _compute_budget_supervision(
            execution,
            speech_budget_tgt=targets.distill_speech_budget_tgt.float(),
            pause_budget_tgt=targets.distill_pause_budget_tgt.float(),
            raw_weight=float(targets.budget_raw_weight),
            exec_weight=float(targets.budget_exec_weight),
            batch_weight=distill_budget_batch_weight,
        )
        total = total + float(targets.distill_budget_weight) * losses["rhythm_distill_budget"]
    losses["rhythm_distill_prefix"] = _compute_distill_prefix_loss(
        execution,
        targets,
        state,
        batch_weight=distill_prefix_batch_weight,
    )
    total = total + float(targets.distill_prefix_weight) * losses["rhythm_distill_prefix"]
    losses["rhythm_distill_allocation"] = _compute_distill_allocation_loss(
        execution,
        targets,
        state,
        batch_weight=distill_allocation_batch_weight,
    )
    total = total + float(targets.distill_allocation_weight) * losses["rhythm_distill_allocation"]
    (
        losses["rhythm_distill_speech_shape"],
        losses["rhythm_distill_pause_shape"],
    ) = _compute_distill_shape_losses(
        execution,
        targets,
        state,
        batch_weight=distill_shape_batch_weight,
    )
    total = total + float(targets.distill_speech_shape_weight) * losses["rhythm_distill_speech_shape"]
    total = total + float(targets.distill_pause_shape_weight) * losses["rhythm_distill_pause_shape"]
    losses["rhythm_distill"] = total
    return losses


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
    state = RhythmLossState(
        unit_mask=unit_mask,
        blank_exec=blank_exec,
        pred_prefix_clock_norm=pred_prefix_clock_norm,
        pred_prefix_backlog_norm=pred_prefix_backlog_norm,
    )
    l_plan_local, l_plan_cum, l_plan = _compute_plan_losses(execution, targets, state)
    l_guidance = _compute_guidance_loss(execution, targets, state)
    distill_losses = _compute_distill_losses(execution, targets, state)
    # Maintained compatibility alias: public cumplan/prefix_state supervision maps
    # to prefix carry/backlog alignment, while `rhythm_plan_cum` is the separate
    # cumulative execution proxy loss.
    kd_same_source_exec = _scalar_flag(execution.speech_duration_exec, targets.distill_same_source_exec)
    kd_same_source_budget = _scalar_flag(execution.speech_duration_exec, targets.distill_same_source_budget)
    kd_same_source_prefix = _scalar_flag(execution.speech_duration_exec, targets.distill_same_source_prefix)
    kd_same_source_allocation = _scalar_flag(execution.speech_duration_exec, targets.distill_same_source_allocation)
    kd_same_source_shape = _scalar_flag(execution.speech_duration_exec, targets.distill_same_source_shape)
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
        **distill_losses,
        'rhythm_distill_same_source_exec': kd_same_source_exec,
        'rhythm_distill_same_source_budget': kd_same_source_budget,
        'rhythm_distill_same_source_prefix': kd_same_source_prefix,
        'rhythm_distill_same_source_allocation': kd_same_source_allocation,
        'rhythm_distill_same_source_shape': kd_same_source_shape,
        'rhythm_distill_same_source_any': torch.maximum(
            torch.maximum(kd_same_source_exec, kd_same_source_budget),
            torch.maximum(
                kd_same_source_prefix,
                torch.maximum(kd_same_source_allocation, kd_same_source_shape),
            ),
        ),
        'rhythm_total': l_exec_speech + l_exec_pause + l_budget + l_carry + l_plan + l_guidance + distill_losses['rhythm_distill'],
    }
