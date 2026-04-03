from __future__ import annotations

import torch
import torch.nn as nn

from .controller import UnitRedistributionHead, WindowBudgetController, masked_mean
from .contracts import RhythmPlannerOutputs, StreamingRhythmState
from .source_boundary import compose_boundary_score_unit


class MonotonicRhythmScheduler(nn.Module):
    """Hierarchical monotonic rhythm planner."""

    def __init__(
        self,
        *,
        hidden_size: int,
        stats_dim: int,
        trace_dim: int,
        max_total_logratio: float = 0.8,
        max_unit_logratio: float = 0.6,
        pause_share_max: float = 0.45,
        pause_share_residual_max: float = 0.12,
        min_speech_frames: float = 1.0,
        boundary_feature_scale: float = 0.35,
        boundary_source_cue_weight: float = 0.65,
        pause_source_boundary_weight: float = 0.20,
    ) -> None:
        super().__init__()
        self.boundary_source_cue_weight = float(boundary_source_cue_weight)
        self.window_budget = WindowBudgetController(
            hidden_size=hidden_size,
            stats_dim=stats_dim,
            trace_dim=trace_dim,
            max_total_logratio=max_total_logratio,
            pause_share_max=pause_share_max,
            pause_share_residual_max=pause_share_residual_max,
            min_speech_frames=min_speech_frames,
            boundary_feature_scale=boundary_feature_scale,
        )
        self.unit_redistribution = UnitRedistributionHead(
            hidden_size=hidden_size,
            trace_dim=trace_dim,
            max_unit_logratio=max_unit_logratio,
            boundary_feature_scale=boundary_feature_scale,
            pause_source_boundary_weight=pause_source_boundary_weight,
        )

    @staticmethod
    def _resolve_planner_stats(ref_conditioning: dict[str, torch.Tensor]) -> torch.Tensor:
        planner_ref_stats = ref_conditioning.get("planner_ref_stats")
        if planner_ref_stats is not None:
            return planner_ref_stats
        return torch.cat([ref_conditioning["global_rate"], ref_conditioning["pause_ratio"]], dim=-1)

    @staticmethod
    def _resolve_planner_slow_summary(
        ref_conditioning: dict[str, torch.Tensor],
        planner_trace_context: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> torch.Tensor:
        planner_slow = ref_conditioning.get("planner_slow_rhythm_summary")
        if planner_slow is not None:
            return planner_slow
        slow_full = ref_conditioning.get("slow_rhythm_summary")
        if slow_full is not None:
            if slow_full.dim() == 2 and slow_full.size(-1) >= 3:
                return torch.cat([slow_full[:, 1:2], slow_full[:, 2:3]], dim=-1)
            if slow_full.dim() == 2 and slow_full.size(-1) == planner_trace_context.size(-1):
                return slow_full
        return masked_mean(planner_trace_context, unit_mask.float(), dim=1)

    def forward(
        self,
        *,
        unit_states: torch.Tensor,
        dur_anchor_src: torch.Tensor,
        unit_mask: torch.Tensor,
        ref_conditioning: dict[str, torch.Tensor],
        trace_context: torch.Tensor,
        planner_trace_context: torch.Tensor,
        state: StreamingRhythmState,
        source_boundary_cue: torch.Tensor | None = None,
    ) -> RhythmPlannerOutputs:
        planner_ref_stats = self._resolve_planner_stats(ref_conditioning)
        slow_rhythm_summary = self._resolve_planner_slow_summary(
            ref_conditioning,
            planner_trace_context,
            unit_mask,
        )
        boundary_trace = (
            planner_trace_context[:, :, 1]
            if planner_trace_context.size(-1) > 1
            else planner_trace_context.squeeze(-1)
        )
        boundary_score_unit = compose_boundary_score_unit(
            unit_mask=unit_mask,
            source_boundary_cue=source_boundary_cue,
            boundary_trace=boundary_trace,
            source_weight=self.boundary_source_cue_weight,
        )
        budget_outputs = self.window_budget(
            unit_states=unit_states,
            dur_anchor_src=dur_anchor_src,
            unit_mask=unit_mask,
            planner_ref_stats=planner_ref_stats,
            planner_trace_context=planner_trace_context,
            slow_rhythm_summary=slow_rhythm_summary,
            boundary_score_unit=boundary_score_unit,
            phase_ptr=state.phase_ptr,
            clock_delta=state.clock_delta,
        )
        redistribution_outputs = self.unit_redistribution(
            unit_states=unit_states,
            dur_anchor_src=dur_anchor_src,
            planner_trace_context=planner_trace_context,
            unit_mask=unit_mask,
            slow_rhythm_summary=slow_rhythm_summary,
            boundary_score_unit=boundary_score_unit,
        )
        return RhythmPlannerOutputs(
            speech_budget_win=budget_outputs["speech_budget_win"],
            pause_budget_win=budget_outputs["pause_budget_win"],
            dur_logratio_unit=redistribution_outputs["dur_logratio_unit"],
            pause_weight_unit=redistribution_outputs["pause_weight_unit"],
            boundary_score_unit=boundary_score_unit,
            trace_context=trace_context,
            source_boundary_cue=source_boundary_cue,
        )
