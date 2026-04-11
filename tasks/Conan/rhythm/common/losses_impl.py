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
from ..budget_repair import compute_budget_projection_repair_stats


@dataclass
class RhythmLossTargets:
    speech_exec_tgt: torch.Tensor
    pause_exec_tgt: torch.Tensor
    speech_budget_tgt: torch.Tensor
    pause_budget_tgt: torch.Tensor
    unit_mask: torch.Tensor
    dur_anchor_src: torch.Tensor
    unit_logratio_weight: float = 0.0
    srmdp_role_consistency_weight: float = 0.0
    srmdp_notimeline_weight: float = 0.0
    srmdp_memory_role_weight: float = 0.0
    srmdp_role_id_src_tgt: Optional[torch.Tensor] = None
    srmdp_ref_memory_role_id_tgt: Optional[torch.Tensor] = None
    srmdp_ref_memory_mask_tgt: Optional[torch.Tensor] = None
    plan_local_weight: float = 0.5
    plan_cum_weight: float = 1.0
    plan_segment_shape_weight: float = 0.0
    plan_pause_release_weight: float = 0.0
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
    distill_context_match: Optional[torch.Tensor] = None
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
    pause_event_weight: float = 0.0
    pause_support_weight: float = 0.0
    pause_allocation_weight: float = 0.0
    pause_event_threshold: float = 0.5
    pause_event_temperature: float = 0.25
    pause_event_pos_weight: float = 2.0

    @property
    def blank_exec_tgt(self) -> torch.Tensor:
        return self.pause_exec_tgt

    @property
    def blank_budget_tgt(self) -> torch.Tensor:
        return self.pause_budget_tgt


@dataclass
class DurationV3LossTargets:
    unit_duration_tgt: torch.Tensor
    unit_anchor_base: torch.Tensor
    unit_mask: torch.Tensor
    committed_mask: torch.Tensor
    unit_confidence_tgt: Optional[torch.Tensor] = None
    prediction_anchor: Optional[torch.Tensor] = None
    baseline_duration_tgt: Optional[torch.Tensor] = None
    baseline_mask: Optional[torch.Tensor] = None
    baseline_global_tgt: Optional[torch.Tensor] = None
    global_rate: Optional[torch.Tensor] = None
    global_shift_tgt: Optional[torch.Tensor] = None
    residual_logstretch_tgt: Optional[torch.Tensor] = None
    global_bias_tgt: Optional[torch.Tensor] = None
    local_residual_tgt: Optional[torch.Tensor] = None
    prompt_basis_activation: Optional[torch.Tensor] = None
    prompt_random_target_tgt: Optional[torch.Tensor] = None
    prompt_mask: Optional[torch.Tensor] = None
    prompt_fit_mask: Optional[torch.Tensor] = None
    prompt_eval_mask: Optional[torch.Tensor] = None
    prompt_operator_fit_pred: Optional[torch.Tensor] = None
    prompt_operator_cv_fit_pred: Optional[torch.Tensor] = None
    prompt_role_attn: Optional[torch.Tensor] = None
    prompt_role_fit_pred: Optional[torch.Tensor] = None
    prompt_role_value: Optional[torch.Tensor] = None
    prompt_role_var: Optional[torch.Tensor] = None
    prompt_log_duration: Optional[torch.Tensor] = None
    prompt_log_residual: Optional[torch.Tensor] = None
    consistency_duration_tgt: Optional[torch.Tensor] = None
    consistency_mask: Optional[torch.Tensor] = None
    consistency_local_residual_tgt: Optional[torch.Tensor] = None
    lambda_dur: float = 1.0
    lambda_op: float = 0.25
    lambda_pref: float = 0.20
    lambda_bias: float = 0.0
    lambda_base: float = 0.0
    lambda_cons: float = 0.0
    lambda_zero: float = 0.0
    lambda_ortho: float = 0.0
    baseline_pretrain_only: bool = False

    @property
    def lambda_mem(self) -> float:
        return float(self.lambda_op)


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


def _weighted_masked_huber(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    mask: torch.Tensor,
    *,
    weight: Optional[torch.Tensor] = None,
    beta: float = 1.0,
    batch_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    base_mask = mask.float()
    if weight is None:
        eff_weight = base_mask
    else:
        eff_weight = weight.float() * base_mask
    loss = F.smooth_l1_loss(pred.float(), tgt.float(), beta=beta, reduction="none")
    while eff_weight.dim() < loss.dim():
        eff_weight = eff_weight.unsqueeze(-1)
    reduce_dims = tuple(range(1, loss.dim()))
    masked_loss = (loss * eff_weight).sum(dim=reduce_dims)
    masked_denom = eff_weight.sum(dim=reduce_dims).clamp_min(1.0)
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


def _masked_bce_with_logits(
    logits: torch.Tensor,
    tgt: torch.Tensor,
    mask: torch.Tensor,
    *,
    batch_weight: Optional[torch.Tensor] = None,
    pos_weight: float | None = None,
) -> torch.Tensor:
    mask = mask.float()
    loss = F.binary_cross_entropy_with_logits(logits.float(), tgt.float(), reduction="none")
    if pos_weight is not None and float(pos_weight) != 1.0:
        scale = torch.where(
            tgt.float() > 0.5,
            loss.new_full(loss.shape, float(pos_weight)),
            loss.new_ones(loss.shape),
        )
        loss = loss * scale
    reduce_dims = tuple(range(1, loss.dim()))
    masked_loss = (loss * mask).sum(dim=reduce_dims)
    masked_denom = mask.sum(dim=reduce_dims).clamp_min(1.0)
    return _reduce_batch_loss(masked_loss / masked_denom, batch_weight)


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


def _resolve_exec_pause_share_gate(
    execution,
    *,
    tgt_speech_budget: torch.Tensor,
    tgt_pause_budget: torch.Tensor,
) -> Optional[torch.Tensor]:
    planner = getattr(execution, "planner", None)
    if planner is None:
        return None
    repair_mass = compute_budget_projection_repair_stats(planner).repair_mass.detach().float()
    if repair_mass.dim() == 0:
        repair_mass = repair_mass.view(1, 1)
    else:
        repair_mass = repair_mass.reshape(repair_mass.size(0), -1)[:, :1]
    repair_mass = repair_mass.clamp_min(0.0)
    tgt_total_budget, _ = _budget_surfaces_from_speech_pause(
        tgt_speech_budget,
        tgt_pause_budget,
    )
    gate = tgt_total_budget.float().clamp_min(0.0) / (
        tgt_total_budget.float().clamp_min(0.0) + repair_mass
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

    The executed pause-share branch is also soft-gated by projection repair
    mass. When feasibility repair materially changes the speech/pause split,
    the executed pause share becomes projection-constrained rather than a pure
    teacher/control target; the raw-view budget surface and feasible-debt term
    stay as the authoritative correction signal in that regime.
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
    exec_pause_share_gate = _resolve_exec_pause_share_gate(
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
    exec_pause_share_loss_vec = F.l1_loss(
        exec_pause_share,
        tgt_pause_share,
        reduction='none',
    )
    if exec_pause_share_loss_vec.dim() > 1:
        exec_pause_share_loss_vec = exec_pause_share_loss_vec.reshape(exec_pause_share_loss_vec.size(0), -1).mean(dim=1)
    exec_pause_share_loss = _reduce_batch_loss_with_scale(
        exec_pause_share_loss_vec,
        batch_weight=batch_weight,
        loss_scale=exec_pause_share_gate,
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


def _compute_pause_event_loss(
    blank_exec: torch.Tensor,
    pause_exec_tgt: torch.Tensor,
    pause_mask: torch.Tensor,
    *,
    threshold: float,
    temperature: float,
    pos_weight: float,
    batch_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Support-first auxiliary loss for missed pause events.

    The primary pause loss still regresses pause magnitude. This auxiliary term
    answers a different question: whether a pause event should fire at all.
    That makes it a better fit for the common "precision > recall" failure mode,
    where support is too sparse even when aggregate pause budget is acceptable.
    """
    pause_mask = pause_mask.float().clamp_min(0.0)
    target_event = (pause_exec_tgt.float() > float(threshold)).float()
    logits = (blank_exec.float() - float(threshold)) / max(float(temperature), 1e-4)
    loss = F.binary_cross_entropy_with_logits(logits, target_event, reduction='none')
    if float(pos_weight) > 1.0:
        positive_scale = torch.where(
            target_event > 0.5,
            loss.new_full(target_event.shape, float(pos_weight)),
            loss.new_ones(target_event.shape),
        )
        pause_mask = pause_mask * positive_scale
    reduce_dims = tuple(range(1, loss.dim()))
    masked_loss = (loss * pause_mask).sum(dim=reduce_dims)
    masked_denom = pause_mask.sum(dim=reduce_dims).clamp_min(1.0)
    return _reduce_batch_loss(masked_loss / masked_denom, batch_weight)


def _compute_pause_support_loss(
    execution,
    pause_exec_tgt: torch.Tensor,
    pause_mask: torch.Tensor,
    *,
    threshold: float,
    pos_weight: float,
    batch_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Supervise planner-side pause support before projector sparsification."""
    planner = getattr(execution, "planner", None)
    if planner is None:
        return pause_exec_tgt.new_zeros(())
    pause_support_prob = getattr(planner, "pause_support_prob_unit", None)
    pause_support_logit = getattr(planner, "pause_support_logit_unit", None)
    if pause_support_prob is None and pause_support_logit is None:
        pause_shape_unit = getattr(planner, "pause_shape_unit", None)
        if pause_shape_unit is None:
            return pause_exec_tgt.new_zeros(())
        pause_mask = pause_mask.float().clamp_min(0.0)
        target_pause = pause_exec_tgt.float().clamp_min(0.0)
        support_gate = _positive_mass_gate(target_pause, pause_mask)
        support_batch_weight = _merge_batch_weight(batch_weight, support_gate, target_pause)
        return _batch_kl_div(
            pause_shape_unit.float().clamp_min(0.0),
            target_pause,
            pause_mask,
            batch_weight=support_batch_weight,
        )
    pause_mask = pause_mask.float().clamp_min(0.0)
    target_event = (pause_exec_tgt.float() > float(threshold)).float()
    if pause_support_logit is None:
        pause_support_prob = pause_support_prob.float().clamp(1.0e-4, 1.0 - 1.0e-4)
        loss = F.binary_cross_entropy(
            pause_support_prob,
            target_event,
            reduction='none',
        )
    else:
        loss = F.binary_cross_entropy_with_logits(
            pause_support_logit.float(),
            target_event,
            reduction='none',
        )
    if float(pos_weight) > 1.0:
        positive_scale = torch.where(
            target_event > 0.5,
            loss.new_full(target_event.shape, float(pos_weight)),
            loss.new_ones(target_event.shape),
        )
        pause_mask = pause_mask * positive_scale
    reduce_dims = tuple(range(1, loss.dim()))
    masked_loss = (loss * pause_mask).sum(dim=reduce_dims)
    masked_denom = pause_mask.sum(dim=reduce_dims).clamp_min(1.0)
    return _reduce_batch_loss(masked_loss / masked_denom, batch_weight)


def _compute_pause_allocation_loss(
    execution,
    pause_exec_tgt: torch.Tensor,
    pause_mask: torch.Tensor,
    *,
    batch_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    planner = getattr(execution, "planner", None)
    pause_allocation = getattr(planner, "pause_allocation_weight_unit", None) if planner is not None else None
    if pause_allocation is None:
        pause_allocation = getattr(planner, "pause_shape_unit", None) if planner is not None else None
    if pause_allocation is None:
        return pause_exec_tgt.new_zeros(())
    target_pause = pause_exec_tgt.float().clamp_min(0.0)
    pause_mask = pause_mask.float().clamp_min(0.0)
    allocation_gate = _positive_mass_gate(target_pause, pause_mask)
    allocation_batch_weight = _merge_batch_weight(batch_weight, allocation_gate, target_pause)
    return _batch_kl_div(
        pause_allocation.float().clamp_min(0.0),
        target_pause,
        pause_mask,
        batch_weight=allocation_batch_weight,
    )


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


def _target_unit_logratio(
    *,
    speech_exec_tgt: torch.Tensor,
    dur_anchor_src: torch.Tensor,
    unit_mask: torch.Tensor,
) -> torch.Tensor:
    unit_mask = unit_mask.float()
    eps = 1.0e-4
    target = torch.log(
        (speech_exec_tgt.float().clamp_min(0.0) + eps)
        / (dur_anchor_src.float().clamp_min(0.0) + eps)
    )
    target = target * unit_mask
    mean = (target * unit_mask).sum(dim=1, keepdim=True) / unit_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (target - mean) * unit_mask


def _compute_unit_logratio_loss(execution, targets: RhythmLossTargets) -> torch.Tensor:
    weight = float(getattr(targets, "unit_logratio_weight", 0.0) or 0.0)
    planner = getattr(execution, "planner", None)
    pred = getattr(planner, "dur_logratio_unit", None) if planner is not None else None
    if weight <= 0.0 or pred is None:
        ref = (
            execution.speech_duration_exec
            if isinstance(execution.speech_duration_exec, torch.Tensor)
            else targets.speech_exec_tgt
        )
        return ref.new_tensor(0.0)
    target = _target_unit_logratio(
        speech_exec_tgt=targets.speech_exec_tgt,
        dur_anchor_src=targets.dur_anchor_src,
        unit_mask=targets.unit_mask,
    )
    loss = _masked_huber(
        pred.float(),
        target,
        targets.unit_mask.float(),
        beta=0.15,
        batch_weight=targets.sample_confidence,
    )
    return weight * loss


def _resolve_srmdp_feature(execution) -> Optional[torch.Tensor]:
    planner = getattr(execution, "planner", None)
    if planner is None:
        return None
    retrieved = getattr(planner, "role_memory_retrieved_unit", None)
    if isinstance(retrieved, torch.Tensor):
        return retrieved.float()
    local_query = getattr(planner, "local_role_query_unit", None)
    if isinstance(local_query, torch.Tensor):
        return local_query.float()
    logratio = getattr(planner, "dur_logratio_unit", None)
    if isinstance(logratio, torch.Tensor):
        if logratio.dim() == 2:
            return logratio.float().unsqueeze(-1)
        return logratio.float()
    return None


def _compute_srmdp_role_consistency_loss(execution, targets: RhythmLossTargets) -> torch.Tensor:
    weight = float(getattr(targets, "srmdp_role_consistency_weight", 0.0) or 0.0)
    role_ids = getattr(targets, "srmdp_role_id_src_tgt", None)
    feature = _resolve_srmdp_feature(execution)
    if weight <= 0.0 or role_ids is None or feature is None:
        return execution.speech_duration_exec.new_tensor(0.0)
    if feature.dim() == 2:
        feature = feature.unsqueeze(-1)
    if feature.dim() != 3:
        return execution.speech_duration_exec.new_tensor(0.0)
    roles = role_ids.long()
    if roles.dim() > 2:
        roles = roles.squeeze(-1)
    mask = targets.unit_mask.float()
    if roles.shape[:2] != mask.shape[:2] or feature.shape[:2] != mask.shape[:2]:
        return execution.speech_duration_exec.new_tensor(0.0)
    per_batch: list[torch.Tensor] = []
    for batch_idx in range(mask.size(0)):
        valid = mask[batch_idx] > 0.5
        if int(valid.sum().item()) < 2:
            per_batch.append(mask.new_zeros(()))
            continue
        batch_roles = roles[batch_idx][valid]
        batch_feature = feature[batch_idx][valid]
        role_losses: list[torch.Tensor] = []
        for role_id in torch.unique(batch_roles):
            if int(role_id.item()) < 0:
                continue
            role_mask = batch_roles == role_id
            if int(role_mask.sum().item()) < 2:
                continue
            role_feature = batch_feature[role_mask]
            role_mean = role_feature.mean(dim=0, keepdim=True)
            role_losses.append(((role_feature - role_mean) ** 2).mean())
        if role_losses:
            per_batch.append(torch.stack(role_losses).mean())
        else:
            per_batch.append(mask.new_zeros(()))
    per_batch_loss = torch.stack(per_batch)
    return weight * _reduce_batch_loss(per_batch_loss, targets.sample_confidence)


def _compute_srmdp_notimeline_loss(execution, targets: RhythmLossTargets) -> torch.Tensor:
    weight = float(getattr(targets, "srmdp_notimeline_weight", 0.0) or 0.0)
    planner = getattr(execution, "planner", None)
    top_index = getattr(planner, "role_memory_top_index_unit", None) if planner is not None else None
    if weight <= 0.0 or not isinstance(top_index, torch.Tensor):
        return execution.speech_duration_exec.new_tensor(0.0)
    if top_index.dim() == 3:
        slot_idx = torch.arange(top_index.size(-1), device=top_index.device, dtype=top_index.dtype).view(1, 1, -1)
        top_index = (top_index.float() * slot_idx).sum(dim=-1)
    elif top_index.dim() == 2:
        top_index = top_index.float()
    else:
        return execution.speech_duration_exec.new_tensor(0.0)
    mask = targets.unit_mask.float()
    if top_index.shape != mask.shape:
        return execution.speech_duration_exec.new_tensor(0.0)
    position = torch.arange(mask.size(1), device=mask.device, dtype=top_index.dtype)
    if mask.size(1) > 1:
        position = position / float(mask.size(1) - 1)
    else:
        position = torch.zeros_like(position)
    per_batch: list[torch.Tensor] = []
    for batch_idx in range(mask.size(0)):
        valid = mask[batch_idx] > 0.5
        if int(valid.sum().item()) < 2:
            per_batch.append(mask.new_zeros(()))
            continue
        x = top_index[batch_idx][valid]
        y = position[valid]
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        denom = torch.sqrt((x_centered.pow(2).mean() * y_centered.pow(2).mean()).clamp_min(1e-6))
        corr = (x_centered * y_centered).mean() / denom
        per_batch.append(corr.abs())
    per_batch_loss = torch.stack(per_batch)
    return weight * _reduce_batch_loss(per_batch_loss, targets.sample_confidence)


def _compute_srmdp_memory_role_loss(execution, targets: RhythmLossTargets) -> torch.Tensor:
    weight = float(getattr(targets, "srmdp_memory_role_weight", 0.0) or 0.0)
    planner = getattr(execution, "planner", None)
    top_index = getattr(planner, "role_memory_top_index_unit", None) if planner is not None else None
    src_role = getattr(targets, "srmdp_role_id_src_tgt", None)
    ref_role = getattr(targets, "srmdp_ref_memory_role_id_tgt", None)
    if (
        weight <= 0.0
        or not isinstance(top_index, torch.Tensor)
        or src_role is None
        or ref_role is None
    ):
        return execution.speech_duration_exec.new_tensor(0.0)
    if top_index.dim() == 3:
        top_index = torch.argmax(top_index.float(), dim=-1)
    elif top_index.dim() == 2:
        top_index = torch.round(top_index.float()).long()
    else:
        return execution.speech_duration_exec.new_tensor(0.0)
    src_role = src_role.long()
    if src_role.dim() > 2:
        src_role = src_role.squeeze(-1)
    ref_role = ref_role.long()
    if ref_role.dim() > 2:
        ref_role = ref_role.squeeze(-1)
    if top_index.shape[:2] != src_role.shape[:2]:
        return execution.speech_duration_exec.new_tensor(0.0)
    if ref_role.dim() != 2 or ref_role.size(0) != top_index.size(0):
        return execution.speech_duration_exec.new_tensor(0.0)
    ref_mask = getattr(targets, "srmdp_ref_memory_mask_tgt", None)
    if isinstance(ref_mask, torch.Tensor):
        ref_mask = ref_mask.float()
        if ref_mask.dim() > 2:
            ref_mask = ref_mask.squeeze(-1)
        if ref_mask.shape != ref_role.shape:
            ref_mask = None
    else:
        ref_mask = None
    unit_mask = targets.unit_mask.float()
    per_batch: list[torch.Tensor] = []
    for batch_idx in range(top_index.size(0)):
        max_slot = ref_role.size(1) - 1
        idx = top_index[batch_idx].clamp(min=0, max=max_slot)
        pred_role = ref_role[batch_idx].gather(0, idx)
        valid = (unit_mask[batch_idx] > 0.5) & (src_role[batch_idx] >= 0)
        if ref_mask is not None:
            slot_valid = ref_mask[batch_idx].gather(0, idx) > 0.5
            valid = valid & slot_valid
        if int(valid.sum().item()) <= 0:
            per_batch.append(unit_mask.new_zeros(()))
            continue
        mismatch = (pred_role != src_role[batch_idx]).float()
        per_batch.append(mismatch[valid].mean())
    per_batch_loss = torch.stack(per_batch)
    return weight * _reduce_batch_loss(per_batch_loss, targets.sample_confidence)


def _compute_plan_segment_shape_loss(
    execution,
    targets: RhythmLossTargets,
    state: RhythmLossState,
    *,
    batch_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    planner = getattr(execution, "planner", None)
    zero = execution.speech_duration_exec.new_tensor(0.0)
    if planner is None:
        return zero
    segment_shape = getattr(planner, "segment_shape_context_unit", None)
    open_tail_mask = getattr(planner, "open_tail_mask_unit", None)
    plan_logratio = getattr(planner, "dur_logratio_unit", None)
    if segment_shape is None or open_tail_mask is None or plan_logratio is None:
        return zero
    if segment_shape.size(-1) < 3:
        return zero
    mask = (open_tail_mask.float() * state.unit_mask.float()).clamp_min(0.0)
    if float(mask.sum().item()) <= 0.0:
        return zero
    pred_mass = targets.dur_anchor_src.float().clamp_min(0.0) * torch.exp(plan_logratio.float())
    local_rate = segment_shape[:, :, 0].float()
    duration_bias = segment_shape[:, :, 2].float()
    target_mass = targets.dur_anchor_src.float().clamp_min(0.0) * F.softplus(local_rate + 0.5 * duration_bias)
    gate = _positive_mass_gate(target_mass, mask)
    shape_batch_weight = _merge_batch_weight(batch_weight, gate, pred_mass)
    return _batch_kl_div(
        pred_mass.clamp_min(0.0),
        target_mass.clamp_min(0.0),
        mask,
        batch_weight=shape_batch_weight,
    )


def _compute_plan_pause_release_loss(
    execution,
    targets: RhythmLossTargets,
    state: RhythmLossState,
    *,
    batch_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    planner = getattr(execution, "planner", None)
    zero = execution.speech_duration_exec.new_tensor(0.0)
    if planner is None:
        return zero
    segment_shape = getattr(planner, "segment_shape_context_unit", None)
    open_tail_mask = getattr(planner, "open_tail_mask_unit", None)
    if segment_shape is None or open_tail_mask is None or segment_shape.size(-1) < 2:
        return zero
    pause_support = getattr(planner, "pause_support_prob_unit", None)
    if pause_support is None:
        pause_support = getattr(planner, "pause_weight_unit", None)
    if pause_support is None:
        return zero
    segment_roll = getattr(planner, "segment_roll_alpha_unit", None)
    if segment_roll is None:
        segment_roll = open_tail_mask.new_zeros(open_tail_mask.shape)
    source_boundary = getattr(planner, "source_boundary_cue", None)
    if source_boundary is None:
        source_boundary = open_tail_mask.new_zeros(open_tail_mask.shape)
    boundary_strength = segment_shape[:, :, 1].float().clamp_min(0.0)
    release_target = (
        F.softplus(boundary_strength)
        * (0.25 + 0.75 * segment_roll.float())
        * (0.35 + 0.65 * source_boundary.float().clamp(0.0, 1.0))
    )
    mask = (open_tail_mask.float() * state.unit_mask.float()).clamp_min(0.0)
    if float(mask.sum().item()) <= 0.0:
        return zero
    gate = _positive_mass_gate(release_target, mask)
    release_batch_weight = _merge_batch_weight(batch_weight, gate, pause_support)
    return _batch_kl_div(
        pause_support.float().clamp_min(0.0),
        release_target.clamp_min(0.0),
        mask,
        batch_weight=release_batch_weight,
    )


def _compute_plan_losses(
    execution,
    targets: RhythmLossTargets,
    state: RhythmLossState,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    zero = execution.speech_duration_exec.new_tensor(0.0)
    if (
        float(targets.plan_local_weight) <= 0.0
        and float(targets.plan_cum_weight) <= 0.0
        and float(targets.plan_segment_shape_weight) <= 0.0
        and float(targets.plan_pause_release_weight) <= 0.0
    ):
        return zero, zero, zero, zero, zero
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
    l_plan_segment_shape = zero
    if float(targets.plan_segment_shape_weight) > 0.0:
        l_plan_segment_shape = _compute_plan_segment_shape_loss(
            execution,
            targets,
            state,
            batch_weight=targets.sample_confidence,
        )
    l_plan_pause_release = zero
    if float(targets.plan_pause_release_weight) > 0.0:
        l_plan_pause_release = _compute_plan_pause_release_loss(
            execution,
            targets,
            state,
            batch_weight=targets.sample_confidence,
        )
    l_plan = (
        float(targets.plan_local_weight) * l_plan_local
        + float(targets.plan_cum_weight) * l_plan_cum
        + float(targets.plan_segment_shape_weight) * l_plan_segment_shape
        + float(targets.plan_pause_release_weight) * l_plan_pause_release
    )
    return l_plan_local, l_plan_cum, l_plan_segment_shape, l_plan_pause_release, l_plan


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


def _build_duration_v3_duration_loss(
    *,
    execution,
    pred_speech: torch.Tensor,
    targets: DurationV3LossTargets,
    committed_mask: torch.Tensor,
) -> torch.Tensor:
    pred_residual = getattr(execution, "local_residual", getattr(execution, "local_response", None))
    if (
        isinstance(pred_residual, torch.Tensor)
        and isinstance(targets.local_residual_tgt, torch.Tensor)
    ):
        return _weighted_masked_huber(
            pred_residual.float(),
            targets.local_residual_tgt.float(),
            committed_mask,
            weight=targets.unit_confidence_tgt,
            beta=0.25,
            batch_weight=None,
        )
    prediction_anchor = (
        targets.prediction_anchor
        if isinstance(targets.prediction_anchor, torch.Tensor)
        else targets.unit_anchor_base
    )
    prediction_anchor = prediction_anchor.float().clamp_min(1.0e-6)
    pred_logstretch = execution.unit_logstretch.float()
    tgt_logstretch = torch.log(targets.unit_duration_tgt.float().clamp_min(1.0e-6)) - torch.log(prediction_anchor)
    return _masked_huber(
        pred_logstretch,
        tgt_logstretch,
        committed_mask,
        beta=0.25,
        batch_weight=None,
    )


def _build_duration_v3_baseline_loss(
    *,
    pred_speech: torch.Tensor,
    targets: DurationV3LossTargets,
) -> torch.Tensor:
    if targets.baseline_duration_tgt is None or targets.baseline_mask is None:
        return pred_speech.new_tensor(0.0)
    pred_log_anchor = torch.log(targets.unit_anchor_base.float().clamp_min(1.0e-6))
    tgt_log_anchor = torch.log(targets.baseline_duration_tgt.float().clamp_min(1.0e-6))
    return _masked_huber(
        pred_log_anchor,
        tgt_log_anchor,
        targets.baseline_mask.float(),
        beta=0.25,
        batch_weight=None,
    )


def _build_duration_v3_operator_loss(
    *,
    pred_speech: torch.Tensor,
    targets: DurationV3LossTargets,
) -> torch.Tensor:
    zero = pred_speech.new_tensor(0.0)
    operator_loss = zero
    if (
        isinstance(targets.prompt_role_fit_pred, torch.Tensor)
        and isinstance(targets.prompt_log_residual, torch.Tensor)
        and isinstance(targets.prompt_mask, torch.Tensor)
    ):
        prompt_mask = targets.prompt_mask.float()
        fit_loss = _masked_huber(
            targets.prompt_role_fit_pred.float(),
            targets.prompt_log_residual.float(),
            prompt_mask,
            beta=0.25,
            batch_weight=None,
        )
        reg = zero
        if isinstance(targets.prompt_role_attn, torch.Tensor):
            usage = targets.prompt_role_attn.float().mean(dim=(0, 1))
            uniform = torch.full_like(usage, 1.0 / max(1, usage.numel()))
            reg = F.kl_div(usage.clamp_min(1.0e-6).log(), uniform, reduction="batchmean")
        return fit_loss + (0.01 * reg)
    if (
        isinstance(targets.prompt_role_attn, torch.Tensor)
        and isinstance(targets.prompt_role_value, torch.Tensor)
        and isinstance(targets.prompt_role_var, torch.Tensor)
        and isinstance(targets.prompt_log_duration, torch.Tensor)
        and isinstance(targets.prompt_mask, torch.Tensor)
        and isinstance(targets.global_rate, torch.Tensor)
    ):
        prompt_mask = targets.prompt_mask.float()
        centered = (targets.prompt_log_duration.float() - targets.global_rate.float()) * prompt_mask
        mean_fit = torch.einsum("btm,bm->bt", targets.prompt_role_attn.float(), targets.prompt_role_value.float())
        var_fit = torch.einsum("btm,bm->bt", targets.prompt_role_attn.float(), targets.prompt_role_var.float()).clamp_min(1.0e-4)
        nll = (((centered - mean_fit) ** 2) / var_fit) + torch.log(var_fit)
        operator_loss = (nll * prompt_mask).sum() / prompt_mask.sum().clamp_min(1.0)
        usage = targets.prompt_role_attn.float().mean(dim=(0, 1))
        uniform = torch.full_like(usage, 1.0 / max(1, usage.numel()))
        reg = F.kl_div(usage.clamp_min(1.0e-6).log(), uniform, reduction="batchmean")
        return operator_loss + (0.01 * reg)
    if targets.prompt_random_target_tgt is not None:
        prompt_pred = targets.prompt_operator_cv_fit_pred
        prompt_mask = targets.prompt_eval_mask
        if prompt_pred is None:
            prompt_pred = targets.prompt_operator_fit_pred
        if prompt_mask is None:
            prompt_mask = targets.prompt_mask
        if prompt_pred is None and float(targets.lambda_op) > 0.0:
            raise ValueError("Duration V3 operator loss requires prompt_operator_fit on rhythm_ref_conditioning.")
        if prompt_pred is not None:
            prompt_mask = prompt_mask if prompt_mask is not None else torch.ones_like(targets.prompt_random_target_tgt)
            prompt_delta = F.smooth_l1_loss(
                prompt_pred.float(),
                targets.prompt_random_target_tgt.float(),
                beta=0.25,
                reduction="none",
            )
            operator_loss = (prompt_delta * prompt_mask.float()).sum() / prompt_mask.float().sum().clamp_min(1.0)
    return operator_loss


def _build_duration_v3_zero_loss(
    *,
    pred_speech: torch.Tensor,
    targets: DurationV3LossTargets,
) -> torch.Tensor:
    if targets.prompt_operator_fit_pred is None or targets.prompt_mask is None:
        return pred_speech.new_tensor(0.0)
    prompt_mask = targets.prompt_mask.float()
    zero_center = (targets.prompt_operator_fit_pred.float() * prompt_mask).sum(dim=1) / prompt_mask.sum(
        dim=1
    ).clamp_min(1.0)
    return F.smooth_l1_loss(
        zero_center,
        torch.zeros_like(zero_center),
        beta=0.1,
        reduction="mean",
    )


def _build_duration_v3_ortho_loss(
    *,
    execution,
    pred_speech: torch.Tensor,
    targets: DurationV3LossTargets,
    committed_mask: torch.Tensor,
) -> torch.Tensor:
    def _masked_cov_identity_loss(phi: torch.Tensor | None, mask: torch.Tensor | None) -> torch.Tensor:
        if phi is None or mask is None:
            return pred_speech.new_tensor(0.0)
        if phi.numel() <= 0 or phi.size(-1) <= 0:
            return pred_speech.new_tensor(0.0)
        mask_f = mask.float().unsqueeze(-1)
        support = mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        cov = torch.matmul((phi.float() * mask_f).transpose(1, 2), phi.float() * mask_f) / support.unsqueeze(-1)
        eye = torch.eye(phi.size(-1), device=phi.device, dtype=phi.dtype).unsqueeze(0)
        return ((cov - eye) ** 2).mean()

    prompt_loss = _masked_cov_identity_loss(targets.prompt_basis_activation, targets.prompt_mask)
    source_loss = _masked_cov_identity_loss(getattr(execution, "basis_activation", None), committed_mask)
    return prompt_loss + source_loss


def _build_duration_v3_stream_losses(
    *,
    pred_speech: torch.Tensor,
    execution,
    targets: DurationV3LossTargets,
    committed_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_prefix = torch.cumsum(pred_speech * committed_mask, dim=1)
    tgt_prefix = torch.cumsum(targets.unit_duration_tgt.float() * committed_mask, dim=1)
    l_pref = _masked_huber(
        pred_prefix,
        tgt_prefix,
        committed_mask,
        beta=1.0,
        batch_weight=None,
    )
    l_cons = pred_speech.new_tensor(0.0)
    if (
        float(targets.lambda_cons) > 0.0
        and targets.consistency_duration_tgt is not None
        and targets.consistency_mask is not None
    ):
        if (
            isinstance(getattr(execution, "local_residual", getattr(execution, "local_response", None)), torch.Tensor)
            and isinstance(targets.consistency_local_residual_tgt, torch.Tensor)
        ):
            l_cons = _weighted_masked_huber(
                getattr(execution, "local_residual", getattr(execution, "local_response", None)).float(),
                targets.consistency_local_residual_tgt.float(),
                targets.consistency_mask.float(),
                weight=targets.unit_confidence_tgt,
                beta=0.25,
                batch_weight=None,
            )
        else:
            pred_cons = getattr(execution, "unit_duration_raw", None)
            if not isinstance(pred_cons, torch.Tensor):
                pred_cons = pred_speech
            l_cons = _masked_huber(
                pred_cons.float(),
                targets.consistency_duration_tgt.float(),
                targets.consistency_mask.float(),
                beta=0.25,
                batch_weight=None,
            )
    return l_pref, l_cons, l_pref + l_cons


def _build_duration_v3_bias_loss(
    *,
    execution,
    pred_speech: torch.Tensor,
    targets: DurationV3LossTargets,
) -> torch.Tensor:
    if not isinstance(getattr(execution, "global_bias_scalar", None), torch.Tensor):
        return pred_speech.new_tensor(0.0)
    if not isinstance(targets.global_bias_tgt, torch.Tensor):
        return pred_speech.new_tensor(0.0)
    return F.smooth_l1_loss(
        execution.global_bias_scalar.float().reshape(targets.global_bias_tgt.shape),
        targets.global_bias_tgt.float(),
        beta=0.25,
        reduction="mean",
    )


def _rebuild_duration_v3_sgbase_prediction(
    *,
    execution,
    targets: DurationV3LossTargets,
) -> torch.Tensor:
    anchor = (
        targets.prediction_anchor
        if isinstance(targets.prediction_anchor, torch.Tensor)
        else targets.unit_anchor_base
    )
    anchor = anchor.float().detach().clamp_min(1.0e-6)
    return anchor * torch.exp(execution.unit_logstretch.float())


def _build_duration_v3_loss_dict(execution, targets: DurationV3LossTargets) -> dict[str, torch.Tensor]:
    unit_mask = targets.unit_mask.float()
    committed_mask = targets.committed_mask.float() * unit_mask
    pred_speech = execution.speech_duration_exec.float()
    pred_speech_sgbase = _rebuild_duration_v3_sgbase_prediction(
        execution=execution,
        targets=targets,
    )
    l_dur = _build_duration_v3_duration_loss(
        execution=execution,
        pred_speech=pred_speech,
        targets=targets,
        committed_mask=committed_mask,
    )
    l_base = _build_duration_v3_baseline_loss(
        pred_speech=pred_speech,
        targets=targets,
    )
    l_bias = _build_duration_v3_bias_loss(
        execution=execution,
        pred_speech=pred_speech,
        targets=targets,
    )
    l_op = _build_duration_v3_operator_loss(
        pred_speech=pred_speech,
        targets=targets,
    )
    l_zero = _build_duration_v3_zero_loss(
        pred_speech=pred_speech,
        targets=targets,
    )
    l_ortho = _build_duration_v3_ortho_loss(
        execution=execution,
        pred_speech=pred_speech,
        targets=targets,
        committed_mask=committed_mask,
    )
    l_pref, l_cons, l_stream = _build_duration_v3_stream_losses(
        pred_speech=pred_speech_sgbase,
        execution=execution,
        targets=targets,
        committed_mask=committed_mask,
    )
    pretrain_only = bool(targets.baseline_pretrain_only)
    scaled_dur = pred_speech.new_tensor(0.0) if pretrain_only else l_dur * float(targets.lambda_dur)
    scaled_op = pred_speech.new_tensor(0.0) if pretrain_only else l_op * float(targets.lambda_op)
    scaled_bias = pred_speech.new_tensor(0.0) if pretrain_only else l_bias * float(targets.lambda_bias)
    scaled_zero = pred_speech.new_tensor(0.0) if pretrain_only else l_zero * float(targets.lambda_zero)
    scaled_ortho = pred_speech.new_tensor(0.0) if pretrain_only else l_ortho * float(targets.lambda_ortho)
    scaled_stream = (
        pred_speech.new_tensor(0.0)
        if pretrain_only
        else (l_pref * float(targets.lambda_pref)) + (l_cons * float(targets.lambda_cons))
    )
    scaled_base = l_base * float(targets.lambda_base)
    total = scaled_dur + scaled_op + scaled_bias + scaled_zero + scaled_ortho + scaled_stream + scaled_base
    loss_dict = {
        "rhythm_exec_speech": scaled_dur,
        "rhythm_exec_stretch": scaled_op + scaled_bias + scaled_zero + scaled_ortho,
        "rhythm_prefix_state": scaled_stream,
        "rhythm_v3_base": l_base.detach(),
        "rhythm_v3_dur": l_dur.detach(),
        "rhythm_v3_bias": l_bias.detach(),
        "rhythm_v3_op": l_op.detach(),
        "rhythm_v3_summary": l_op.detach(),
        "rhythm_v3_zero": l_zero.detach(),
        "rhythm_v3_ortho": l_ortho.detach(),
        "rhythm_v3_pref": l_pref.detach(),
        "rhythm_v3_cons": l_cons.detach(),
        "rhythm_v3_stream": l_stream.detach(),
        "rhythm_is_v3_bundle": pred_speech.new_tensor(1.0),
        "rhythm_total": total,
    }
    if (
        (
            isinstance(targets.prompt_role_fit_pred, torch.Tensor)
            and isinstance(targets.prompt_log_residual, torch.Tensor)
            and isinstance(targets.prompt_mask, torch.Tensor)
        )
        or (
            isinstance(targets.prompt_role_attn, torch.Tensor)
            and isinstance(targets.prompt_role_value, torch.Tensor)
            and isinstance(targets.prompt_role_var, torch.Tensor)
        )
    ):
        loss_dict["rhythm_v3_mem"] = loss_dict["rhythm_v3_summary"]
    return loss_dict


def build_rhythm_loss_dict(execution, targets: RhythmLossTargets) -> dict[str, torch.Tensor]:
    if isinstance(targets, DurationV3LossTargets):
        return _build_duration_v3_loss_dict(execution, targets)
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
    l_exec_stretch_base = _compute_unit_logratio_loss(execution, targets)
    l_srmdp_role_consistency = _compute_srmdp_role_consistency_loss(execution, targets)
    l_srmdp_notimeline = _compute_srmdp_notimeline_loss(execution, targets)
    l_srmdp_memory_role = _compute_srmdp_memory_role_loss(execution, targets)
    l_exec_stretch = l_exec_stretch_base + l_srmdp_role_consistency + l_srmdp_notimeline + l_srmdp_memory_role
    pause_mask = _resolve_pause_exec_mask(
        execution,
        unit_mask,
        boundary_weight=float(targets.pause_boundary_weight),
    )
    event_mask = unit_mask.float()
    pause_exec_tgt = targets.pause_exec_tgt.float()
    l_exec_pause_value = _masked_log_huber(
        blank_exec,
        pause_exec_tgt,
        pause_mask,
        batch_weight=targets.sample_confidence,
    )
    l_pause_event = blank_exec.new_tensor(0.0)
    if float(targets.pause_event_weight) > 0.0:
        l_pause_event = _compute_pause_event_loss(
            blank_exec,
            pause_exec_tgt,
            event_mask,
            threshold=float(targets.pause_event_threshold),
            temperature=float(targets.pause_event_temperature),
            pos_weight=float(targets.pause_event_pos_weight),
            batch_weight=targets.sample_confidence,
        )
    l_pause_event = float(targets.pause_event_weight) * l_pause_event
    l_pause_support = blank_exec.new_tensor(0.0)
    if float(targets.pause_support_weight) > 0.0:
        l_pause_support = _compute_pause_support_loss(
            execution,
            pause_exec_tgt,
            event_mask,
            threshold=float(targets.pause_event_threshold),
            pos_weight=float(targets.pause_event_pos_weight),
            batch_weight=targets.sample_confidence,
        )
    l_pause_support = float(targets.pause_support_weight) * l_pause_support
    l_pause_allocation = blank_exec.new_tensor(0.0)
    if float(targets.pause_allocation_weight) > 0.0:
        l_pause_allocation = _compute_pause_allocation_loss(
            execution,
            pause_exec_tgt,
            event_mask,
            batch_weight=targets.sample_confidence,
        )
    l_pause_allocation = float(targets.pause_allocation_weight) * l_pause_allocation
    l_exec_pause = l_exec_pause_value + l_pause_event + l_pause_support + l_pause_allocation
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
    l_plan_local, l_plan_cum, l_plan_segment_shape, l_plan_pause_release, l_plan = _compute_plan_losses(
        execution,
        targets,
        state,
    )
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
        'rhythm_exec_stretch': l_exec_stretch,
        'rhythm_exec_stretch_base': l_exec_stretch_base.detach(),
        'rhythm_srmdp_role_consistency': l_srmdp_role_consistency.detach(),
        'rhythm_srmdp_notimeline': l_srmdp_notimeline.detach(),
        'rhythm_srmdp_memory_role': l_srmdp_memory_role.detach(),
        'rhythm_exec_pause': l_exec_pause,
        'rhythm_exec_pause_value': l_exec_pause_value.detach(),
        'rhythm_pause_event': l_pause_event.detach(),
        'rhythm_pause_support': l_pause_support.detach(),
        'rhythm_pause_allocation': l_pause_allocation.detach(),
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
        'rhythm_plan_segment_shape': l_plan_segment_shape,
        'rhythm_plan_pause_release': l_plan_pause_release,
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
        **(
            {
                'rhythm_distill_context_match': targets.distill_context_match.detach().mean()
            }
            if isinstance(targets.distill_context_match, torch.Tensor)
            else {}
        ),
        'rhythm_total': l_exec_speech + l_exec_stretch + l_exec_pause + l_budget + l_carry + l_plan + l_guidance + distill_losses['rhythm_distill'],
    }
