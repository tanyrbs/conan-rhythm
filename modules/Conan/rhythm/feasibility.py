from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FeasibleBudgetProjection:
    speech_budget_win: torch.Tensor
    pause_budget_win: torch.Tensor
    speech_budget_delta: torch.Tensor
    pause_budget_delta: torch.Tensor
    total_budget_delta: torch.Tensor


def lift_projector_budgets_to_feasible_region(
    *,
    dur_anchor_src: torch.Tensor,
    unit_mask: torch.Tensor,
    speech_budget_win: torch.Tensor,
    pause_budget_win: torch.Tensor,
    previous_speech_exec: torch.Tensor | None,
    previous_pause_exec: torch.Tensor | None,
    commit_frontier: torch.Tensor,
    reuse_prefix: bool,
    min_speech_frames: float,
    max_speech_expand: float,
) -> FeasibleBudgetProjection:
    """Project planner budgets into the minimal feasible region.

    Semantics:
    - keep raw total budget unchanged whenever feasibility can be recovered by
      reallocation between speech and pause;
    - only lift total budget when the raw total itself is below the minimum
      feasible speech lower bound plus committed-prefix pause;
    - never construct a per-unit speech upper bound below the speech lower
      bound induced by ``min_speech_frames``.
    """
    speech_rows = []
    pause_rows = []
    speech_delta_rows = []
    pause_delta_rows = []
    total_delta_rows = []

    for batch_idx in range(unit_mask.size(0)):
        raw_speech = speech_budget_win[batch_idx].float().reshape(1, 1)
        raw_pause = pause_budget_win[batch_idx].float().reshape(1, 1)
        mask_row = unit_mask[batch_idx].float()
        frontier = int(commit_frontier[batch_idx].item()) if reuse_prefix else 0

        prefix_speech = raw_speech.new_zeros((1, 1))
        prefix_pause = raw_pause.new_zeros((1, 1))
        tail_mask = mask_row.clone()
        if frontier > 0:
            if previous_speech_exec is not None and batch_idx < previous_speech_exec.size(0):
                valid_frontier = min(frontier, int(previous_speech_exec.size(1)), int(mask_row.size(0)))
                prefix_speech = (
                    previous_speech_exec[batch_idx, :valid_frontier].float() * mask_row[:valid_frontier]
                ).sum().reshape(1, 1)
                tail_mask[:valid_frontier] = 0.0
            if previous_pause_exec is not None and batch_idx < previous_pause_exec.size(0):
                valid_frontier = min(frontier, int(previous_pause_exec.size(1)), int(mask_row.size(0)))
                prefix_pause = (
                    previous_pause_exec[batch_idx, :valid_frontier].float() * mask_row[:valid_frontier]
                ).sum().reshape(1, 1)

        active_tail = tail_mask > 0
        active_tail_count = int(active_tail.sum().item())
        speech_lower = prefix_speech + raw_speech.new_tensor(float(active_tail_count * min_speech_frames)).reshape(1, 1)

        if float(max_speech_expand) > 0.0 and active_tail_count > 0:
            tail_anchor = dur_anchor_src[batch_idx].float().clamp_min(0.0)[active_tail]
            tail_upper = torch.maximum(
                tail_anchor * float(max_speech_expand),
                tail_anchor.new_full(tail_anchor.shape, float(min_speech_frames)),
            )
            speech_upper = prefix_speech + tail_upper.sum().reshape(1, 1)
        else:
            speech_upper = raw_speech.new_full((1, 1), float("inf"))

        raw_total = raw_speech + raw_pause
        minimal_feasible_total = speech_lower + prefix_pause
        # If the raw total already covers the feasibility lower bound, stay on
        # the same total budget and only reallocate between speech/pause.
        feasible_total = torch.maximum(raw_total, minimal_feasible_total)
        speech_upper = torch.minimum(speech_upper, feasible_total - prefix_pause)
        speech_feasible = torch.minimum(torch.maximum(raw_speech, speech_lower), speech_upper)
        pause_feasible = feasible_total - speech_feasible

        speech_rows.append(speech_feasible)
        pause_rows.append(pause_feasible)
        speech_delta_rows.append((speech_feasible - raw_speech).clamp_min(0.0))
        pause_delta_rows.append((pause_feasible - raw_pause).clamp_min(0.0))
        total_delta_rows.append(feasible_total - raw_total)

    return FeasibleBudgetProjection(
        speech_budget_win=torch.cat(speech_rows, dim=0),
        pause_budget_win=torch.cat(pause_rows, dim=0),
        speech_budget_delta=torch.cat(speech_delta_rows, dim=0),
        pause_budget_delta=torch.cat(pause_delta_rows, dim=0),
        total_budget_delta=torch.cat(total_delta_rows, dim=0),
    )
