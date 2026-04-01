from __future__ import annotations

import torch
import torch.nn as nn

from .controller import UnitRedistributionHead, WindowBudgetController, masked_mean
from .contracts import RhythmPlannerOutputs


class OfflineRhythmTeacherPlanner(nn.Module):
    """Non-causal full-horizon planner used as the learned offline teacher."""

    def __init__(
        self,
        *,
        hidden_size: int,
        stats_dim: int,
        trace_dim: int,
        max_total_logratio: float = 0.8,
        max_unit_logratio: float = 0.6,
        pause_share_max: float = 0.45,
        boundary_feature_scale: float = 0.35,
        boundary_source_cue_weight: float = 0.35,
        pause_boundary_latent_weight: float = 0.35,
        pause_source_boundary_weight: float = 0.20,
    ) -> None:
        super().__init__()
        self.window_budget = WindowBudgetController(
            hidden_size=hidden_size,
            stats_dim=stats_dim,
            trace_dim=trace_dim,
            max_total_logratio=max_total_logratio,
            pause_share_max=pause_share_max,
            boundary_feature_scale=boundary_feature_scale,
            causal=False,
        )
        self.unit_redistribution = UnitRedistributionHead(
            hidden_size=hidden_size,
            trace_dim=trace_dim,
            max_unit_logratio=max_unit_logratio,
            boundary_feature_scale=boundary_feature_scale,
            boundary_source_cue_weight=boundary_source_cue_weight,
            pause_boundary_latent_weight=pause_boundary_latent_weight,
            pause_source_boundary_weight=pause_source_boundary_weight,
            causal=False,
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size + stats_dim + trace_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        *,
        unit_states: torch.Tensor,
        dur_anchor_src: torch.Tensor,
        unit_mask: torch.Tensor,
        ref_conditioning: dict[str, torch.Tensor],
        trace_context: torch.Tensor,
        source_boundary_cue: torch.Tensor | None = None,
    ) -> tuple[RhythmPlannerOutputs, torch.Tensor]:
        if source_boundary_cue is None:
            source_boundary_cue = unit_mask.new_zeros(unit_mask.shape)
        batch_size = unit_states.size(0)
        zeros = unit_states.new_zeros((batch_size,))
        budget_outputs = self.window_budget(
            unit_states=unit_states,
            dur_anchor_src=dur_anchor_src,
            unit_mask=unit_mask,
            ref_rhythm_stats=ref_conditioning["ref_rhythm_stats"],
            trace_context=trace_context,
            slow_rhythm_summary=ref_conditioning.get("slow_rhythm_summary"),
            source_boundary_cue=source_boundary_cue,
            phase_ptr=zeros,
            backlog=zeros,
            clock_delta=zeros,
        )
        redistribution_outputs = self.unit_redistribution(
            hidden=budget_outputs["hidden"],
            trace_context=trace_context,
            unit_mask=unit_mask,
            slow_rhythm_summary=ref_conditioning.get("slow_rhythm_summary"),
            source_boundary_cue=source_boundary_cue,
        )
        planner = RhythmPlannerOutputs(
            speech_budget_win=budget_outputs["speech_budget_win"],
            pause_budget_win=budget_outputs["pause_budget_win"],
            dur_logratio_unit=redistribution_outputs["dur_logratio_unit"],
            pause_weight_unit=redistribution_outputs["pause_weight_unit"],
            total_budget_win=budget_outputs["total_budget_win"],
            pause_share_win=budget_outputs["pause_share_win"],
            anchor_gate=budget_outputs["anchor_gate"],
            boundary_latent=redistribution_outputs["boundary_latent"],
            trace_context=trace_context,
            source_boundary_cue=source_boundary_cue,
        )
        pooled_hidden = masked_mean(budget_outputs["hidden"], unit_mask, dim=1)
        pooled_trace = masked_mean(trace_context, unit_mask, dim=1)
        confidence = self.confidence_head(
            torch.cat([pooled_hidden, ref_conditioning["ref_rhythm_stats"], pooled_trace], dim=-1)
        ).clamp(0.05, 1.0)
        return planner, confidence
