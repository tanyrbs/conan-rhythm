from __future__ import annotations

import torch
import torch.nn as nn

from .compat import resolve_phase_decoupled_flag
from .controller import ChunkStateHead, UnitRedistributionHead, WindowBudgetController, masked_mean
from .contracts import RhythmPlannerOutputs, StreamingRhythmState, TraceReliabilityBundle
from .source_boundary import compose_boundary_score_unit


class MonotonicRhythmScheduler(nn.Module):
    """Hierarchical monotonic rhythm planner.

    This targets streaming timing / temporal pacing control. In the
    phase-decoupled path, source-side structure decides boundary existence,
    while reference priors only modulate realization style via bounded residuals.
    """

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
        chunk_state_enable: bool = True,
        budget_phase_feature_scale: float = 0.0,
        phase_decoupled_timing: bool | None = None,
        phase_free_timing: bool | None = None,
        phase_decoupled_boundary_style_residual_scale: float = 0.18,
        debt_control_scale: float = 4.0,
        debt_pause_priority: float = 0.15,
        debt_speech_priority: float = 0.25,
    ) -> None:
        super().__init__()
        self.boundary_source_cue_weight = float(boundary_source_cue_weight)
        self.chunk_state_enable = bool(chunk_state_enable)
        self.phase_decoupled_timing = resolve_phase_decoupled_flag(
            default=False,
            phase_decoupled_timing=phase_decoupled_timing,
            phase_free_timing=phase_free_timing,
            where="MonotonicRhythmScheduler.__init__",
        )
        self.phase_free_timing = self.phase_decoupled_timing
        self.phase_decoupled_boundary_style_residual_scale = float(
            max(0.0, phase_decoupled_boundary_style_residual_scale)
        )
        self.debt_control_scale = float(max(debt_control_scale, 1.0e-3))
        self.debt_pause_priority = float(max(debt_pause_priority, 0.0))
        self.debt_speech_priority = float(max(debt_speech_priority, 0.0))
        self.chunk_state_head = ChunkStateHead() if self.chunk_state_enable else None
        self.window_budget = WindowBudgetController(
            hidden_size=hidden_size,
            stats_dim=stats_dim,
            trace_dim=trace_dim,
            max_total_logratio=max_total_logratio,
            pause_share_max=pause_share_max,
            pause_share_residual_max=pause_share_residual_max,
            min_speech_frames=min_speech_frames,
            boundary_feature_scale=boundary_feature_scale,
            phase_feature_scale=budget_phase_feature_scale,
            phase_decoupled_timing=self.phase_decoupled_timing,
            debt_control_scale=self.debt_control_scale,
            debt_pause_priority=self.debt_pause_priority,
            debt_speech_priority=self.debt_speech_priority,
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

    def _apply_phase_decoupled_boundary_style_residual(
        self,
        *,
        source_boundary_cue: torch.Tensor | None,
        unit_mask: torch.Tensor,
        phrase_selection: dict[str, torch.Tensor] | None,
        prompt_reliability: torch.Tensor,
        residual_scale: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        effective_residual_scale = float(
            self.phase_decoupled_boundary_style_residual_scale
            if residual_scale is None
            else max(0.0, float(residual_scale))
        )
        base = (
            source_boundary_cue.float().clamp(0.0, 1.0) * unit_mask.float()
            if source_boundary_cue is not None
            else unit_mask.new_zeros(unit_mask.shape)
        )
        if (
            phrase_selection is None
            or effective_residual_scale <= 0.0
        ):
            return base, unit_mask.new_zeros(unit_mask.shape)
        boundary_strength = phrase_selection.get("selected_phrase_prototype_boundary_strength")
        if boundary_strength is None:
            boundary_strength = phrase_selection.get("selected_ref_phrase_boundary_strength")
        if boundary_strength is None:
            return base, unit_mask.new_zeros(unit_mask.shape)
        boundary_strength = boundary_strength.float().reshape(unit_mask.size(0), 1).clamp(0.0, 1.0)
        style_gate = prompt_reliability.float().reshape(unit_mask.size(0), 1).clamp(0.0, 1.0)
        residual_scalar = ((boundary_strength - 0.5) * 2.0).clamp(-1.0, 1.0) * style_gate
        residual_unit = base * residual_scalar
        gain = 1.0 + residual_scalar * effective_residual_scale
        boundary_score = (base * gain).clamp(0.0, 1.0) * unit_mask.float()
        return boundary_score, residual_unit

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
        sep_hint: torch.Tensor | None = None,
        open_run_mask: torch.Tensor | None = None,
        sealed_mask: torch.Tensor | None = None,
        boundary_confidence: torch.Tensor | None = None,
        phrase_selection: dict[str, torch.Tensor] | None = None,
        trace_reliability: TraceReliabilityBundle | None = None,
        trace_exhaustion_final_cell_suppress: float = 0.0,
        chunk_state: ChunkStateBundle | None = None,
        phase_decoupled_timing: bool | None = None,
        phase_free_timing: bool | None = None,
        phase_decoupled_boundary_style_residual_scale: float | None = None,
        debt_control_scale: float | None = None,
        debt_pause_priority: float | None = None,
        debt_speech_priority: float | None = None,
    ) -> RhythmPlannerOutputs:
        effective_phase_decoupled_timing = resolve_phase_decoupled_flag(
            default=self.phase_decoupled_timing,
            phase_decoupled_timing=phase_decoupled_timing,
            phase_free_timing=phase_free_timing,
            where="MonotonicRhythmScheduler.forward",
        )
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
        if effective_phase_decoupled_timing:
            effective_planner_trace_context = planner_trace_context
        else:
            effective_planner_trace_context = planner_trace_context * local_trace_path_weight[:, None, None]
        if trace_reliability is not None:
            prompt_metric = (
                trace_reliability.phrase_blend
                if effective_phase_decoupled_timing
                else trace_reliability.trace_reliability
            )
            prompt_reliability = prompt_metric.float().reshape(unit_mask.size(0), 1)
        else:
            prompt_reliability = torch.ones((unit_mask.size(0), 1), device=unit_mask.device, dtype=unit_mask.dtype)
        selected_valid = None
        phrase_prototype_summary = None
        phrase_prototype_stats = None
        if phrase_selection is not None:
            phrase_prototype_summary = phrase_selection.get("selected_phrase_prototype_summary")
            phrase_prototype_stats = phrase_selection.get("selected_phrase_prototype_stats")
            if (
                phrase_prototype_stats is not None
                and phrase_prototype_stats.size(-1) != planner_ref_stats.size(-1)
            ):
                phrase_prototype_stats = phrase_prototype_summary
            selected_valid = phrase_selection.get("selected_phrase_prototype_valid")
            if selected_valid is not None:
                selected_valid = selected_valid.float().reshape(unit_mask.size(0), 1).clamp(0.0, 1.0)
                prompt_reliability = prompt_reliability * selected_valid
        if effective_phase_decoupled_timing:
            boundary_score_unit, boundary_style_residual_unit = self._apply_phase_decoupled_boundary_style_residual(
                source_boundary_cue=source_boundary_cue,
                unit_mask=unit_mask,
                phrase_selection=phrase_selection,
                prompt_reliability=prompt_reliability,
                residual_scale=phase_decoupled_boundary_style_residual_scale,
            )
        else:
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
            boundary_style_residual_unit = unit_mask.new_zeros(unit_mask.shape)
        if chunk_state is None and self.chunk_state_head is not None:
            chunk_state = self.chunk_state_head(
                unit_mask=unit_mask,
                state=state,
                source_boundary_cue=source_boundary_cue,
                boundary_score_unit=boundary_score_unit,
                sep_hint=sep_hint,
                open_run_mask=open_run_mask,
                sealed_mask=sealed_mask,
                boundary_confidence=boundary_confidence,
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
            commit_frontier=state.commit_frontier,
            chunk_state=chunk_state,
            phrase_prototype_summary=phrase_prototype_summary,
            phrase_prototype_stats=phrase_prototype_stats,
            prompt_reliability=prompt_reliability,
            phase_decoupled_timing=effective_phase_decoupled_timing,
            debt_control_scale=debt_control_scale,
            debt_pause_priority=debt_pause_priority,
            debt_speech_priority=debt_speech_priority,
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
            trace_phrase_blend=(
                trace_reliability.phrase_blend if trace_reliability is not None else None
            ),
            trace_global_blend=(
                trace_reliability.global_blend if trace_reliability is not None else None
            ),
            trace_tail_reuse_count=(
                trace_reliability.tail_reuse_count if trace_reliability is not None else None
            ),
            trace_tail_alpha=(trace_reliability.tail_alpha if trace_reliability is not None else None),
            trace_gap_alpha=(trace_reliability.gap_alpha if trace_reliability is not None else None),
            trace_reuse_alpha=(trace_reliability.reuse_alpha if trace_reliability is not None else None),
            chunk_summary=(chunk_state.chunk_summary if chunk_state is not None else None),
            chunk_structure_progress=(chunk_state.structure_progress if chunk_state is not None else None),
            chunk_commit_prob=(chunk_state.commit_now_prob if chunk_state is not None else None),
            phrase_open_prob=(chunk_state.phrase_open_prob if chunk_state is not None else None),
            phrase_close_prob=(chunk_state.phrase_close_prob if chunk_state is not None else None),
            phrase_role_prob=(chunk_state.phrase_role_prob if chunk_state is not None else None),
            phrase_prototype_summary=phrase_prototype_summary,
            phrase_prototype_stats=phrase_prototype_stats,
            prompt_reliability=prompt_reliability,
            boundary_style_residual_unit=boundary_style_residual_unit,
            intra_phrase_alpha=(chunk_state.structure_progress if chunk_state is not None else None),
        )
        planner.raw_speech_budget_win = budget_outputs.get("raw_speech_budget_win", planner.speech_budget_win)
        planner.raw_pause_budget_win = budget_outputs.get("raw_pause_budget_win", planner.pause_budget_win)
        return planner
