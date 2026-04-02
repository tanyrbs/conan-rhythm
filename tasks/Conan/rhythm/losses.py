from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


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
    distill_budget_weight: float = 1.0
    distill_allocation_weight: float = 1.0
    distill_prefix_weight: float = 1.0

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
        return weight.view(1).expand(ref.size(0))
    return weight.reshape(weight.size(0), -1)[:, 0]


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
    values = values.float() * mask.float()
    return values / values.sum(dim=1, keepdim=True).clamp_min(1e-6)


def _batch_kl_div(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    mask: torch.Tensor,
    batch_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    pred = _masked_normalize(pred, mask).clamp_min(1e-6)
    tgt = _masked_normalize(tgt, mask).clamp_min(1e-6)
    loss = (tgt * (torch.log(tgt) - torch.log(pred))).sum(dim=1)
    return _reduce_batch_loss(loss, batch_weight)


def _build_prefix_carry(
    *,
    speech_exec: torch.Tensor,
    blank_exec: torch.Tensor,
    dur_anchor_src: torch.Tensor,
    unit_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    unit_mask = unit_mask.float()
    prefix_clock = torch.cumsum(
        ((speech_exec + blank_exec) - dur_anchor_src.float()) * unit_mask,
        dim=1,
    ) * unit_mask
    prefix_backlog = prefix_clock.clamp_min(0.0) * unit_mask
    return prefix_clock, prefix_backlog


def _normalize_prefix_carry(prefix: torch.Tensor, dur_anchor_src: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    prefix_budget = torch.cumsum(dur_anchor_src.float() * mask.float(), dim=1).clamp_min(1.0)
    return prefix.float() / prefix_budget


def build_rhythm_loss_dict(execution, targets: RhythmLossTargets) -> dict[str, torch.Tensor]:
    unit_mask = targets.unit_mask.float()
    blank_exec = getattr(execution, "blank_duration_exec", execution.pause_after_exec)
    pred_prefix_clock, pred_prefix_backlog = _build_prefix_carry(
        speech_exec=execution.speech_duration_exec.float(),
        blank_exec=blank_exec.float(),
        dur_anchor_src=targets.dur_anchor_src.float(),
        unit_mask=unit_mask,
    )
    target_prefix_clock, target_prefix_backlog = _build_prefix_carry(
        speech_exec=targets.speech_exec_tgt.float(),
        blank_exec=targets.pause_exec_tgt.float(),
        dur_anchor_src=targets.dur_anchor_src.float(),
        unit_mask=unit_mask,
    )
    l_exec_speech = _masked_log_huber(
        execution.speech_duration_exec,
        targets.speech_exec_tgt.float(),
        unit_mask,
        batch_weight=targets.sample_confidence,
    )
    l_exec_pause = _masked_log_huber(
        blank_exec,
        targets.pause_exec_tgt.float(),
        unit_mask,
        batch_weight=targets.sample_confidence,
    )
    l_budget = (
        _batch_l1(
            torch.log1p(execution.planner.speech_budget_win.float().clamp_min(0.0)),
            torch.log1p(targets.speech_budget_tgt.float().clamp_min(0.0)),
            batch_weight=targets.sample_confidence,
        )
        + _batch_l1(
            torch.log1p(execution.planner.pause_budget_win.float().clamp_min(0.0)),
            torch.log1p(targets.pause_budget_tgt.float().clamp_min(0.0)),
            batch_weight=targets.sample_confidence,
        )
    )
    l_carry = _masked_huber(
        _normalize_prefix_carry(pred_prefix_clock, targets.dur_anchor_src.float(), unit_mask),
        _normalize_prefix_carry(target_prefix_clock, targets.dur_anchor_src.float(), unit_mask),
        unit_mask,
        beta=0.25,
        batch_weight=targets.sample_confidence,
    ) + _masked_huber(
        _normalize_prefix_carry(pred_prefix_backlog, targets.dur_anchor_src.float(), unit_mask),
        _normalize_prefix_carry(target_prefix_backlog, targets.dur_anchor_src.float(), unit_mask),
        unit_mask,
        beta=0.25,
        batch_weight=targets.sample_confidence,
    )
    exec_total = (execution.speech_duration_exec + blank_exec).float()
    target_total = (targets.speech_exec_tgt + targets.pause_exec_tgt).float()
    l_plan_local = _masked_log_huber(
        exec_total,
        target_total,
        unit_mask,
        beta=0.5,
        batch_weight=targets.sample_confidence,
    )
    l_plan_cum = _masked_huber(
        torch.log1p(_masked_cumsum(exec_total, unit_mask)),
        torch.log1p(_masked_cumsum(target_total, unit_mask)),
        unit_mask,
        beta=1.0,
        batch_weight=targets.sample_confidence,
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
        l_distill = _masked_log_huber(
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
        if (
            float(targets.distill_budget_weight) > 0.0
            and targets.distill_speech_budget_tgt is not None
            and targets.distill_pause_budget_tgt is not None
        ):
            budget_distill = _batch_l1(
                torch.log1p(execution.planner.speech_budget_win.float().clamp_min(0.0)),
                torch.log1p(targets.distill_speech_budget_tgt.float().clamp_min(0.0)),
                batch_weight=distill_budget_weight,
            ) + _batch_l1(
                torch.log1p(execution.planner.pause_budget_win.float().clamp_min(0.0)),
                torch.log1p(targets.distill_pause_budget_tgt.float().clamp_min(0.0)),
                batch_weight=distill_budget_weight,
            )
            l_distill = l_distill + float(targets.distill_budget_weight) * budget_distill
        if targets.distill_prefix_clock_tgt is not None or targets.distill_prefix_backlog_tgt is not None:
            prefix_loss = execution.speech_duration_exec.new_tensor(0.0)
            if targets.distill_prefix_clock_tgt is not None:
                prefix_loss = prefix_loss + _masked_huber(
                    _normalize_prefix_carry(pred_prefix_clock, targets.dur_anchor_src.float(), unit_mask),
                    _normalize_prefix_carry(targets.distill_prefix_clock_tgt.float(), targets.dur_anchor_src.float(), unit_mask),
                    unit_mask,
                    beta=0.25,
                    batch_weight=distill_prefix_weight,
                )
            if targets.distill_prefix_backlog_tgt is not None:
                prefix_loss = prefix_loss + _masked_huber(
                    _normalize_prefix_carry(pred_prefix_backlog, targets.dur_anchor_src.float(), unit_mask),
                    _normalize_prefix_carry(targets.distill_prefix_backlog_tgt.float(), targets.dur_anchor_src.float(), unit_mask),
                    unit_mask,
                    beta=0.25,
                    batch_weight=distill_prefix_weight,
                )
            l_distill = l_distill + float(targets.distill_prefix_weight) * prefix_loss
        if float(targets.distill_allocation_weight) > 0.0 and targets.distill_allocation_tgt is not None:
            l_distill = l_distill + float(targets.distill_allocation_weight) * _batch_kl_div(
                execution.speech_duration_exec + blank_exec,
                targets.distill_allocation_tgt.float(),
                unit_mask,
                batch_weight=distill_allocation_weight,
            )
    else:
        l_distill = execution.speech_duration_exec.new_tensor(0.0)
    return {
        'rhythm_exec_speech': l_exec_speech,
        'rhythm_exec_pause': l_exec_pause,
        'rhythm_budget': l_budget,
        'rhythm_carry': l_carry,
        'rhythm_cumplan': l_carry,
        'rhythm_plan_local': l_plan_local,
        'rhythm_plan_cum': l_plan_cum,
        'rhythm_plan': l_plan,
        'rhythm_guidance': l_guidance,
        'rhythm_distill': l_distill,
        'rhythm_total': l_exec_speech + l_exec_pause + l_budget + l_carry + l_plan + l_guidance + l_distill,
    }
