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
    distill_confidence: Optional[torch.Tensor] = None


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


def _batch_l1(pred: torch.Tensor, tgt: torch.Tensor, batch_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    loss = F.l1_loss(pred, tgt, reduction='none')
    if loss.dim() > 1:
        loss = loss.reshape(loss.size(0), -1).mean(dim=1)
    return _reduce_batch_loss(loss, batch_weight)


def _masked_cumsum(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(values * mask.float(), dim=1)


def build_rhythm_loss_dict(execution, targets: RhythmLossTargets) -> dict[str, torch.Tensor]:
    unit_mask = targets.unit_mask.float()
    l_exec_speech = _masked_huber(
        execution.speech_duration_exec,
        targets.speech_exec_tgt.float(),
        unit_mask,
        batch_weight=targets.sample_confidence,
    )
    l_exec_pause = _masked_huber(
        execution.pause_after_exec,
        targets.pause_exec_tgt.float(),
        unit_mask,
        batch_weight=targets.sample_confidence,
    )
    l_budget = (
        _batch_l1(
            execution.planner.speech_budget_win,
            targets.speech_budget_tgt.float(),
            batch_weight=targets.sample_confidence,
        )
        + _batch_l1(
            execution.planner.pause_budget_win,
            targets.pause_budget_tgt.float(),
            batch_weight=targets.sample_confidence,
        )
    )
    exec_total = (execution.speech_duration_exec + execution.pause_after_exec).float()
    target_total = (targets.speech_exec_tgt + targets.pause_exec_tgt).float()
    l_plan_local = _masked_huber(
        torch.log1p(exec_total),
        torch.log1p(target_total),
        unit_mask,
        beta=0.5,
        batch_weight=targets.sample_confidence,
    )
    l_plan_cum = _masked_huber(
        _masked_cumsum(exec_total, unit_mask),
        _masked_cumsum(target_total, unit_mask),
        unit_mask,
        beta=1.0,
        batch_weight=targets.sample_confidence,
    )
    l_plan = float(targets.plan_local_weight) * l_plan_local + float(targets.plan_cum_weight) * l_plan_cum
    if targets.guidance_speech_tgt is not None and targets.guidance_pause_tgt is not None:
        l_guidance = _masked_huber(
            execution.speech_duration_exec,
            targets.guidance_speech_tgt.float(),
            unit_mask,
            batch_weight=targets.guidance_confidence,
        ) + _masked_huber(
            execution.pause_after_exec,
            targets.guidance_pause_tgt.float(),
            unit_mask,
            batch_weight=targets.guidance_confidence,
        )
    else:
        l_guidance = execution.speech_duration_exec.new_tensor(0.0)
    if (
        targets.distill_speech_tgt is not None
        and targets.distill_pause_tgt is not None
        and targets.distill_speech_budget_tgt is not None
        and targets.distill_pause_budget_tgt is not None
    ):
        l_distill = _masked_huber(
            execution.speech_duration_exec,
            targets.distill_speech_tgt.float(),
            unit_mask,
            batch_weight=targets.distill_confidence,
        ) + _masked_huber(
            execution.pause_after_exec,
            targets.distill_pause_tgt.float(),
            unit_mask,
            batch_weight=targets.distill_confidence,
        ) + _batch_l1(
            execution.planner.speech_budget_win,
            targets.distill_speech_budget_tgt.float(),
            batch_weight=targets.distill_confidence,
        ) + _batch_l1(
            execution.planner.pause_budget_win,
            targets.distill_pause_budget_tgt.float(),
            batch_weight=targets.distill_confidence,
        )
    else:
        l_distill = execution.speech_duration_exec.new_tensor(0.0)
    return {
        'rhythm_exec_speech': l_exec_speech,
        'rhythm_exec_pause': l_exec_pause,
        'rhythm_budget': l_budget,
        'rhythm_plan_local': l_plan_local,
        'rhythm_plan_cum': l_plan_cum,
        'rhythm_plan': l_plan,
        'rhythm_guidance': l_guidance,
        'rhythm_distill': l_distill,
        'rhythm_total': l_exec_speech + l_exec_pause + l_budget + l_plan + l_guidance + l_distill,
    }
