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
    guidance_speech_tgt: Optional[torch.Tensor] = None
    guidance_pause_tgt: Optional[torch.Tensor] = None
    distill_speech_tgt: Optional[torch.Tensor] = None
    distill_pause_tgt: Optional[torch.Tensor] = None
    distill_speech_budget_tgt: Optional[torch.Tensor] = None
    distill_pause_budget_tgt: Optional[torch.Tensor] = None


def _masked_huber(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    mask = mask.float()
    loss = F.smooth_l1_loss(pred, tgt, beta=beta, reduction='none')
    while mask.dim() < loss.dim():
        mask = mask.unsqueeze(-1)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def _masked_cumsum(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(values * mask.float(), dim=1)


def build_rhythm_loss_dict(execution, targets: RhythmLossTargets) -> dict[str, torch.Tensor]:
    unit_mask = targets.unit_mask.float()
    l_exec_speech = _masked_huber(execution.speech_duration_exec, targets.speech_exec_tgt.float(), unit_mask)
    l_exec_pause = _masked_huber(execution.pause_after_exec, targets.pause_exec_tgt.float(), unit_mask)
    l_budget = (
        F.l1_loss(execution.planner.speech_budget_win, targets.speech_budget_tgt.float())
        + F.l1_loss(execution.planner.pause_budget_win, targets.pause_budget_tgt.float())
    )
    exec_total = (execution.speech_duration_exec + execution.pause_after_exec).float()
    target_total = (targets.speech_exec_tgt + targets.pause_exec_tgt).float()
    l_plan_local = _masked_huber(
        torch.log1p(exec_total),
        torch.log1p(target_total),
        unit_mask,
        beta=0.5,
    )
    l_plan_cum = _masked_huber(
        _masked_cumsum(exec_total, unit_mask),
        _masked_cumsum(target_total, unit_mask),
        unit_mask,
        beta=1.0,
    )
    l_plan = l_plan_local + 0.5 * l_plan_cum
    if targets.guidance_speech_tgt is not None and targets.guidance_pause_tgt is not None:
        l_guidance = _masked_huber(
            execution.speech_duration_exec,
            targets.guidance_speech_tgt.float(),
            unit_mask,
        ) + _masked_huber(
            execution.pause_after_exec,
            targets.guidance_pause_tgt.float(),
            unit_mask,
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
        ) + _masked_huber(
            execution.pause_after_exec,
            targets.distill_pause_tgt.float(),
            unit_mask,
        ) + F.l1_loss(
            execution.planner.speech_budget_win,
            targets.distill_speech_budget_tgt.float(),
        ) + F.l1_loss(
            execution.planner.pause_budget_win,
            targets.distill_pause_budget_tgt.float(),
        )
    else:
        l_distill = execution.speech_duration_exec.new_tensor(0.0)
    return {
        'rhythm_exec_speech': l_exec_speech,
        'rhythm_exec_pause': l_exec_pause,
        'rhythm_budget': l_budget,
        'rhythm_plan': l_plan,
        'rhythm_guidance': l_guidance,
        'rhythm_distill': l_distill,
        'rhythm_total': l_exec_speech + l_exec_pause + l_budget + l_plan + l_guidance + l_distill,
    }
