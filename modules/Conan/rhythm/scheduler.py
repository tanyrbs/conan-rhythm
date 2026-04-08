from __future__ import annotations

import torch
import torch.nn as nn

from .controller import UnitRedistributionHead, WindowBudgetController, masked_mean
from .contracts import RhythmPlannerOutputs, StreamingRhythmState, TraceReliabilityBundle
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
        pause_support_split_enable: bool = False,
        pause_breath_features_enable: bool = False,
        pause_breath_reset_threshold: float = 0.55,
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
            pause_support_split_enable=pause_support_split_enable,
            pause_breath_features_enable=pause_breath_features_enable,
            pause_breath_reset_threshold=pause_breath_reset_threshold,
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
        trace_reliability: TraceReliabilityBundle | None = None,
        trace_exhaustion_final_cell_suppress: float = 0.0,
    ) -> torch.Tensor:
        planner_memory = ref_conditioning.get("planner_slow_rhythm_memory")
        selector_scores = ref_conditioning.get("selector_meta_scores")
        selector_indices = ref_conditioning.get("selector_meta_indices")
        use_dynamic_memory = (
            planner_memory is not None
            and planner_memory.dim() == 3
            and trace_reliability is not None
            and bool((trace_reliability.trace_reliability.float() < 0.999).any().item())
        )
        if use_dynamic_memory:
            if selector_scores is None:
                weights = torch.ones(
                    planner_memory.size(0),
                    planner_memory.size(1),
                    device=planner_memory.device,
                    dtype=planner_memory.dtype,
                )
            else:
                weights = 1.0 + selector_scores.float().to(device=planner_memory.device).clamp_min(0.0)
                if weights.dim() == 1:
                    weights = weights.unsqueeze(0)
            if (
                trace_reliability is not None
                and float(trace_exhaustion_final_cell_suppress) > 0.0
                and selector_indices is not None
            ):
                selector_indices = selector_indices.long().to(device=planner_memory.device)
                final_index = selector_indices.max(dim=1, keepdim=True).values
                final_mask = selector_indices.eq(final_index)
                suppress = 1.0 - (
                    (1.0 - trace_reliability.trace_reliability.float()).to(device=planner_memory.device).unsqueeze(1)
                    * float(trace_exhaustion_final_cell_suppress)
                )
                weights = torch.where(final_mask, weights * suppress, weights)
            denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
            return (planner_memory.float() * weights.unsqueeze(-1)).sum(dim=1) / denom
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
        trace_reliability: TraceReliabilityBundle | None = None,
        trace_exhaustion_final_cell_suppress: float = 0.0,
    ) -> RhythmPlannerOutputs:
        planner_ref_stats = self._resolve_planner_stats(ref_conditioning)
        slow_rhythm_summary = self._resolve_planner_slow_summary(
            ref_conditioning,
            planner_trace_context,
            unit_mask,
            trace_reliability=trace_reliability,
            trace_exhaustion_final_cell_suppress=trace_exhaustion_final_cell_suppress,
        )
        local_trace_path_weight = (
            trace_reliability.local_trace_path_weight.float()
            if trace_reliability is not None
            else torch.ones(
                planner_trace_context.size(0),
                device=planner_trace_context.device,
                dtype=planner_trace_context.dtype,
            )
        )
        boundary_trace_path_weight = (
            trace_reliability.boundary_trace_path_weight.float()
            if trace_reliability is not None
            else torch.ones(
                planner_trace_context.size(0),
                device=planner_trace_context.device,
                dtype=planner_trace_context.dtype,
            )
        )
        effective_planner_trace_context = planner_trace_context * local_trace_path_weight[:, None, None]
        boundary_trace = (
            effective_planner_trace_context[:, :, 1]
            if effective_planner_trace_context.size(-1) > 1
            else effective_planner_trace_context.squeeze(-1)
        )
        boundary_trace = boundary_trace * boundary_trace_path_weight[:, None]
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
            planner_trace_context=effective_planner_trace_context,
            slow_rhythm_summary=slow_rhythm_summary,
            boundary_score_unit=boundary_score_unit,
            phase_ptr=state.phase_ptr,
            clock_delta=state.clock_delta,
        )
        redistribution_outputs = self.unit_redistribution(
            unit_states=unit_states,
            dur_anchor_src=dur_anchor_src,
            planner_trace_context=effective_planner_trace_context,
            unit_mask=unit_mask,
            slow_rhythm_summary=slow_rhythm_summary,
            boundary_score_unit=boundary_score_unit,
        )
        planner = RhythmPlannerOutputs(
            speech_budget_win=budget_outputs["speech_budget_win"],
            pause_budget_win=budget_outputs["pause_budget_win"],
            dur_logratio_unit=redistribution_outputs["dur_logratio_unit"],
            pause_weight_unit=redistribution_outputs["pause_weight_unit"],
            boundary_score_unit=boundary_score_unit,
            trace_context=trace_context,
            pause_support_prob_unit=redistribution_outputs.get("pause_support_prob_unit"),
            pause_allocation_weight_unit=redistribution_outputs.get("pause_allocation_weight_unit"),
            pause_support_logit_unit=redistribution_outputs.get("pause_support_logit_unit"),
            pause_run_length_unit=redistribution_outputs.get("pause_run_length_unit"),
            pause_breath_debt_unit=redistribution_outputs.get("pause_breath_debt_unit"),
            source_boundary_cue=source_boundary_cue,
            trace_reliability=(
                trace_reliability.trace_reliability if trace_reliability is not None else None
            ),
            local_trace_path_weight=local_trace_path_weight,
            boundary_trace_path_weight=boundary_trace_path_weight,
            trace_phase_gap=(trace_reliability.phase_gap if trace_reliability is not None else None),
            trace_phase_gap_runtime=(
                trace_reliability.phase_gap_runtime if trace_reliability is not None else None
            ),
            trace_phase_gap_anchor=(
                trace_reliability.phase_gap_anchor if trace_reliability is not None else None
            ),
            trace_coverage_alpha=(
                trace_reliability.coverage_alpha if trace_reliability is not None else None
            ),
            trace_blend=(trace_reliability.blend if trace_reliability is not None else None),
            trace_tail_reuse_count=(
                trace_reliability.tail_reuse_count if trace_reliability is not None else None
            ),
            trace_tail_alpha=(trace_reliability.tail_alpha if trace_reliability is not None else None),
            trace_gap_alpha=(trace_reliability.gap_alpha if trace_reliability is not None else None),
            trace_reuse_alpha=(trace_reliability.reuse_alpha if trace_reliability is not None else None),
        )
        planner.raw_speech_budget_win = budget_outputs.get("raw_speech_budget_win", planner.speech_budget_win)
        planner.raw_pause_budget_win = budget_outputs.get("raw_pause_budget_win", planner.pause_budget_win)
        return planner
