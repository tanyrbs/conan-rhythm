from __future__ import annotations

import torch
import torch.nn as nn
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
        phase_decoupled_timing: bool = False,
        phase_decoupled_boundary_style_residual_scale: float = 0.18,
        phase_decoupled_segment_shape_scale: float = 0.0,
        phase_decoupled_local_rho_scale: float = 0.0,
        phase_decoupled_soft_rollover_scale: float = 0.0,
        phase_decoupled_rollover_start: float = 0.68,
        phase_decoupled_rollover_end: float = 0.92,
        debt_control_scale: float = 4.0,
        debt_pause_priority: float = 0.15,
        debt_speech_priority: float = 0.25,
    ) -> None:
        super().__init__()
        self.boundary_source_cue_weight = float(boundary_source_cue_weight)
        self.chunk_state_enable = bool(chunk_state_enable)
        self.phase_decoupled_timing = bool(phase_decoupled_timing)
        self.phase_decoupled_boundary_style_residual_scale = float(
            max(0.0, phase_decoupled_boundary_style_residual_scale)
        )
        self.phase_decoupled_segment_shape_scale = float(max(phase_decoupled_segment_shape_scale, 0.0))
        self.phase_decoupled_local_rho_scale = float(max(phase_decoupled_local_rho_scale, 0.0))
        self.phase_decoupled_soft_rollover_scale = float(max(phase_decoupled_soft_rollover_scale, 0.0))
        self.phase_decoupled_rollover_start = float(
            min(max(phase_decoupled_rollover_start, 0.0), 1.0)
        )
        self.phase_decoupled_rollover_end = float(
            min(max(phase_decoupled_rollover_end, self.phase_decoupled_rollover_start + 1.0e-6), 1.0)
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
            segment_shape_scale=self.phase_decoupled_segment_shape_scale,
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
            segment_shape_scale=self.phase_decoupled_segment_shape_scale,
            local_rho_scale=self.phase_decoupled_local_rho_scale,
            soft_rollover_scale=self.phase_decoupled_soft_rollover_scale,
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

    @staticmethod
    def _compute_source_local_rho(
        *,
        dur_anchor_src: torch.Tensor,
        unit_mask: torch.Tensor,
        commit_frontier: torch.Tensor,
    ) -> torch.Tensor:
        open_tail_mask = MonotonicRhythmScheduler._build_open_tail_mask(
            unit_mask=unit_mask,
            commit_frontier=commit_frontier,
        )
        return MonotonicRhythmScheduler._compute_source_local_rho_from_mask(
            dur_anchor_src=dur_anchor_src,
            unit_mask=unit_mask,
            open_tail_mask=open_tail_mask,
        )

    @staticmethod
    def _build_open_tail_mask(
        *,
        unit_mask: torch.Tensor,
        commit_frontier: torch.Tensor,
    ) -> torch.Tensor:
        unit_mask_f = unit_mask.float()
        steps = torch.arange(unit_mask.size(1), device=unit_mask.device)[None, :]
        visible_len = unit_mask_f.sum(dim=1).long().clamp(min=0, max=unit_mask.size(1))
        frontier = commit_frontier.long().to(device=unit_mask.device).reshape(unit_mask.size(0), -1)[:, 0]
        frontier = frontier.clamp(min=0, max=unit_mask.size(1))
        return (
            (steps >= frontier[:, None])
            & (steps < visible_len[:, None])
            & (unit_mask_f > 0.5)
        ).float()

    @staticmethod
    def _compute_source_local_rho_from_mask(
        *,
        dur_anchor_src: torch.Tensor,
        unit_mask: torch.Tensor,
        open_tail_mask: torch.Tensor,
    ) -> torch.Tensor:
        open_tail_mask = open_tail_mask.float() * unit_mask.float()
        if float(open_tail_mask.sum().item()) <= 0.0:
            return unit_mask.new_zeros(unit_mask.shape)
        anchor_mass = dur_anchor_src.float().clamp_min(0.0) * open_tail_mask
        mass_total = anchor_mass.sum(dim=1, keepdim=True)
        mass_cum = torch.cumsum(anchor_mass, dim=1)
        unit_cum = torch.cumsum(open_tail_mask, dim=1)
        unit_total = open_tail_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        rho = torch.where(
            mass_total > 1.0e-6,
            mass_cum / mass_total.clamp_min(1.0e-6),
            unit_cum / unit_total,
        )
        return rho.clamp(0.0, 1.0) * open_tail_mask

    @staticmethod
    def _sample_phrase_segment_shape(
        ref_phrase_trace: torch.Tensor | None,
        local_rho_unit: torch.Tensor | None,
        unit_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        if ref_phrase_trace is None or local_rho_unit is None:
            return None
        trace = ref_phrase_trace
        if trace.dim() == 4:
            if trace.size(1) != 1:
                return None
            trace = trace[:, 0]
        if trace.dim() != 3 or trace.size(-1) < 4:
            return None
        batch_size, bins, _ = trace.shape
        if local_rho_unit.size(0) != batch_size:
            return None
        active_mask = ((local_rho_unit.float() > 0.0).float() * unit_mask.float()).clamp(0.0, 1.0)
        if float(active_mask.sum().item()) <= 0.0:
            return torch.zeros(
                (unit_mask.size(0), unit_mask.size(1), 3),
                device=unit_mask.device,
                dtype=trace.dtype,
            )
        trace = trace.to(device=unit_mask.device, dtype=torch.float32)
        rho = local_rho_unit.float().to(device=unit_mask.device).clamp(0.0, 1.0)
        if bins <= 1:
            context = trace[:, :1, 1:4].expand(-1, unit_mask.size(1), -1)
            return context * active_mask.unsqueeze(-1)
        pos = rho * float(bins - 1)
        low = pos.floor().long().clamp(min=0, max=bins - 1)
        high = pos.ceil().long().clamp(min=0, max=bins - 1)
        batch_index = torch.arange(batch_size, device=unit_mask.device)[:, None]
        low_ctx = trace[batch_index, low, 1:4]
        high_ctx = trace[batch_index, high, 1:4]
        frac = (pos - low.float()).unsqueeze(-1)
        context = low_ctx * (1.0 - frac) + high_ctx * frac
        return context * active_mask.unsqueeze(-1)

    @staticmethod
    def _gather_phrase_bank_row(value: torch.Tensor | None, index: torch.Tensor) -> torch.Tensor | None:
        if value is None:
            return None
        if value.dim() < 2:
            raise ValueError(f"phrase-bank value must have rank >= 2, got {tuple(value.shape)}")
        gather_index = index.long().clamp_min(0).view(value.size(0), 1, *([1] * (value.dim() - 2)))
        expand_shape = [value.size(0), 1, *list(value.shape[2:])]
        gather_index = gather_index.expand(*expand_shape)
        gathered = value.gather(1, gather_index)
        return gathered.squeeze(1)

    @staticmethod
    def _smoothstep(value: torch.Tensor, *, start: float, end: float) -> torch.Tensor:
        denom = max(float(end) - float(start), 1.0e-6)
        alpha = ((value.float() - float(start)) / denom).clamp(0.0, 1.0)
        return alpha * alpha * (3.0 - 2.0 * alpha)

    def _build_phase_decoupled_segment_shape_bundle(
        self,
        *,
        ref_conditioning: dict[str, torch.Tensor],
        phrase_selection: dict[str, torch.Tensor] | None,
        dur_anchor_src: torch.Tensor,
        unit_mask: torch.Tensor,
        commit_frontier: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
        if phrase_selection is None:
            return {}, None
        current_phrase_trace = phrase_selection.get("selected_ref_phrase_trace")
        if current_phrase_trace is None:
            return {}, None
        open_tail_mask = self._build_open_tail_mask(
            unit_mask=unit_mask,
            commit_frontier=commit_frontier,
        )
        local_rho = self._compute_source_local_rho_from_mask(
            dur_anchor_src=dur_anchor_src,
            unit_mask=unit_mask,
            open_tail_mask=open_tail_mask,
        )
        current_shape = self._sample_phrase_segment_shape(
            current_phrase_trace,
            local_rho,
            unit_mask,
        )
        bundle = {
            "open_tail_mask_unit": open_tail_mask,
            "local_rho_prior_unit": local_rho,
        }
        next_phrase_trace = phrase_selection.get("next_ref_phrase_trace")
        if next_phrase_trace is None:
            phrase_bank = ref_conditioning.get("ref_phrase_trace")
            phrase_valid = ref_conditioning.get("ref_phrase_valid")
            selected_index = phrase_selection.get("selected_ref_phrase_index")
            if phrase_bank is not None and phrase_valid is not None and selected_index is not None:
                selected_index = selected_index.long().reshape(phrase_bank.size(0), -1)[:, 0]
                valid_count = phrase_valid.long().sum(dim=1).clamp_min(1)
                next_index = torch.minimum(selected_index + 1, valid_count - 1)
                next_phrase_trace = self._gather_phrase_bank_row(phrase_bank, next_index)
        roll_alpha = self._smoothstep(
            local_rho,
            start=self.phase_decoupled_rollover_start,
            end=self.phase_decoupled_rollover_end,
        ) * open_tail_mask
        bundle["segment_roll_alpha_unit"] = roll_alpha
        alpha_scalar = (
            roll_alpha.sum(dim=1, keepdim=True)
            / open_tail_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        ).clamp(0.0, 1.0)
        blended_ref_phrase_trace = self._blend_phrase_trace(
            current_phrase_trace,
            next_phrase_trace,
            alpha_scalar,
        )
        if current_shape is None:
            return bundle, blended_ref_phrase_trace
        if next_phrase_trace is not None and next_phrase_trace.dim() == 4 and next_phrase_trace.size(1) == 1:
            next_phrase_trace = next_phrase_trace[:, 0]
        if next_phrase_trace is not None and next_phrase_trace.dim() == 3 and next_phrase_trace.size(-1) >= 4:
            next_entry = next_phrase_trace[:, :1, 1:4].to(device=unit_mask.device, dtype=current_shape.dtype)
            next_entry = next_entry.expand(-1, unit_mask.size(1), -1)
            segment_shape_context = (
                current_shape * (1.0 - roll_alpha.unsqueeze(-1))
                + next_entry * roll_alpha.unsqueeze(-1)
            )
        else:
            segment_shape_context = current_shape
        bundle["segment_shape_context_unit"] = segment_shape_context * open_tail_mask.unsqueeze(-1)
        return bundle, blended_ref_phrase_trace

    @staticmethod
    def _blend_segment_shape_context(
        *,
        current_ctx: torch.Tensor | None,
        next_ctx: torch.Tensor | None,
        local_rho_unit: torch.Tensor | None,
        unit_mask: torch.Tensor,
        rollover_start: float,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if current_ctx is None:
            return next_ctx, None, None
        if local_rho_unit is None:
            return current_ctx, None, None
        active_mask = ((local_rho_unit.float() > 0.0).float() * unit_mask.float()).clamp(0.0, 1.0)
        if float(active_mask.sum().item()) <= 0.0:
            return current_ctx, unit_mask.new_zeros(unit_mask.shape), unit_mask.new_zeros((unit_mask.size(0), 1))
        denom = max(1.0 - float(rollover_start), 1.0e-6)
        alpha_unit = ((local_rho_unit.float() - float(rollover_start)) / denom).clamp(0.0, 1.0) * active_mask
        if next_ctx is not None:
            blended_ctx = current_ctx.float() * (1.0 - alpha_unit.unsqueeze(-1)) + next_ctx.float() * alpha_unit.unsqueeze(-1)
        else:
            blended_ctx = current_ctx.float()
        alpha_scalar = (
            alpha_unit.sum(dim=1, keepdim=True)
            / active_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        ).clamp(0.0, 1.0)
        return blended_ctx * active_mask.unsqueeze(-1), alpha_unit, alpha_scalar

    @staticmethod
    def _blend_phrase_trace(
        current_trace: torch.Tensor | None,
        next_trace: torch.Tensor | None,
        soft_rollover_alpha: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if current_trace is None:
            return next_trace
        if next_trace is None or soft_rollover_alpha is None:
            return current_trace
        alpha = soft_rollover_alpha.float().reshape(current_trace.size(0), *([1] * (current_trace.dim() - 1)))
        return current_trace.float() * (1.0 - alpha) + next_trace.float() * alpha

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
        phase_decoupled_boundary_style_residual_scale: float | None = None,
        segment_shape_context_unit: torch.Tensor | None = None,
        local_rho_prior_unit: torch.Tensor | None = None,
        segment_roll_alpha_unit: torch.Tensor | None = None,
        open_tail_mask_unit: torch.Tensor | None = None,
        debt_control_scale: float | None = None,
        debt_pause_priority: float | None = None,
        debt_speech_priority: float | None = None,
    ) -> RhythmPlannerOutputs:
        effective_phase_decoupled_timing = (
            self.phase_decoupled_timing
            if phase_decoupled_timing is None
            else bool(phase_decoupled_timing)
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
        source_local_rho_unit = local_rho_prior_unit
        effective_segment_shape_context_unit = segment_shape_context_unit
        effective_segment_roll_alpha_unit = segment_roll_alpha_unit
        effective_open_tail_mask_unit = open_tail_mask_unit
        blended_ref_phrase_trace = None
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
            computed_bundle, blended_ref_phrase_trace = self._build_phase_decoupled_segment_shape_bundle(
                ref_conditioning=ref_conditioning,
                phrase_selection=phrase_selection,
                dur_anchor_src=dur_anchor_src,
                unit_mask=unit_mask,
                commit_frontier=state.commit_frontier,
            )
            if source_local_rho_unit is None:
                source_local_rho_unit = computed_bundle.get("local_rho_prior_unit")
            if effective_segment_shape_context_unit is None:
                effective_segment_shape_context_unit = computed_bundle.get("segment_shape_context_unit")
            if effective_segment_roll_alpha_unit is None:
                effective_segment_roll_alpha_unit = computed_bundle.get("segment_roll_alpha_unit")
            if effective_open_tail_mask_unit is None:
                effective_open_tail_mask_unit = computed_bundle.get("open_tail_mask_unit")
            if effective_segment_shape_context_unit is not None:
                style_gate = prompt_reliability.float().reshape(unit_mask.size(0), 1, 1).clamp(0.0, 1.0)
                effective_segment_shape_context_unit = effective_segment_shape_context_unit * style_gate
            if effective_segment_roll_alpha_unit is not None:
                effective_segment_roll_alpha_unit = (
                    effective_segment_roll_alpha_unit
                    * prompt_reliability.float().reshape(unit_mask.size(0), 1).clamp(0.0, 1.0)
                )
            if blended_ref_phrase_trace is None and phrase_selection is not None:
                blended_ref_phrase_trace = phrase_selection.get("selected_ref_phrase_trace")
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
            segment_shape_context_unit=effective_segment_shape_context_unit,
            local_rho_unit=source_local_rho_unit,
            segment_roll_alpha_unit=effective_segment_roll_alpha_unit,
            open_tail_mask_unit=effective_open_tail_mask_unit,
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
            segment_shape_context_unit=effective_segment_shape_context_unit,
            local_rho_unit=source_local_rho_unit,
            segment_roll_alpha_unit=effective_segment_roll_alpha_unit,
            open_tail_mask_unit=effective_open_tail_mask_unit,
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
            ref_phrase_trace=blended_ref_phrase_trace,
            local_rho_prior_unit=source_local_rho_unit,
            segment_shape_context_unit=effective_segment_shape_context_unit,
            segment_roll_alpha_unit=effective_segment_roll_alpha_unit,
            open_tail_mask_unit=effective_open_tail_mask_unit,
            boundary_style_residual_unit=boundary_style_residual_unit,
        )
        planner.raw_speech_budget_win = budget_outputs.get("raw_speech_budget_win", planner.speech_budget_win)
        planner.raw_pause_budget_win = budget_outputs.get("raw_pause_budget_win", planner.pause_budget_win)
        return planner
