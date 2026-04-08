from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn

from .contracts import BoundaryCommitDecision, RhythmTeacherTargets, StreamingRhythmState, TraceReliabilityBundle
from .controller import BoundaryCommitController, ChunkStateBundle, CommitConfig
from .offline_teacher import OfflineRhythmTeacherPlanner, OfflineTeacherConfig
from .projector import ProjectorConfig, StreamingRhythmProjector
from .reference_descriptor import RefRhythmDescriptor
from .reference_encoder import REF_RHYTHM_STATS_KEYS, REF_RHYTHM_TRACE_KEYS
from .scheduler import MonotonicRhythmScheduler
from .source_boundary import build_source_boundary_cue
from .teacher import AlgorithmicRhythmTeacher, AlgorithmicTeacherConfig


class StreamingRhythmModule(nn.Module):
    """Minimal strong-rhythm mainline for streaming timing / temporal pacing.

    Public contract stays small:
    - content_units
    - dur_anchor_src
    - ref_rhythm_stats / ref_rhythm_trace (or ref_mel)

    Internal execution stays layered but explicit:
    - reference descriptor
    - monotonic scheduler
    - single projector authority

    This module targets streaming rhythm / temporal pacing control rather than
    full expressive prosody reconstruction. In the phase-decoupled path, phase
    remains observer/telemetry state while timing control uses committed
    boundaries, timing debt, and reference priors.
    """

    def __init__(
        self,
        *,
        num_units: int = 128,
        hidden_size: int = 256,
        trace_bins: int = 24,
        stats_dim: int = 6,
        trace_dim: int = 5,
        trace_horizon: float = 0.35,
        slow_topk: int = 6,
        selector_cell_size: int = 3,
        trace_smooth_kernel: int = 5,
        maintained_stats_trace_only: bool = True,
        emit_reference_sidecar: bool | None = None,
        runtime_phrase_bank_enable: bool = False,
        runtime_phrase_bank_max_phrases: int = 6,
        runtime_phrase_bank_bins: int = 8,
        runtime_phrase_select_window: int = 3,
        phrase_selection_boundary_weight: float = 0.28,
        phrase_selection_local_rate_weight: float = 0.28,
        phrase_selection_pause_weight: float = 0.18,
        phrase_selection_voice_weight: float = 0.16,
        phrase_selection_final_bias_weight: float = 0.10,
        phrase_selection_monotonic_bias: float = 0.0,
        phrase_selection_length_bias: float = 0.0,
        max_total_logratio: float = 0.8,
        max_unit_logratio: float = 0.6,
        pause_share_max: float = 0.45,
        pause_share_residual_max: float = 0.12,
        boundary_feature_scale: float = 0.35,
        boundary_source_cue_weight: float = 0.65,
        pause_source_boundary_weight: float = 0.20,
        pause_support_split_enable: bool = False,
        pause_breath_features_enable: bool = False,
        pause_breath_reset_threshold: float = 0.55,
        min_speech_frames: float = 1.0,
        trace_reliability_enable: bool = False,
        trace_exhaustion_gap_start: float = 0.08,
        trace_exhaustion_gap_end: float = 0.22,
        trace_exhaustion_local_floor: float = 0.20,
        trace_exhaustion_boundary_floor: float = 0.05,
        trace_exhaustion_reuse_full_count: int = 3,
        trace_exhaustion_final_cell_suppress: float = 0.65,
        trace_anchor_aware_sampling: bool = False,
        trace_cold_start_min_visible_units: int = 0,
        trace_cold_start_full_visible_units: int = 0,
        trace_active_tail_only: bool = False,
        trace_offset_lookahead_units: int = 0,
        chunk_state_enable: bool = True,
        budget_phase_feature_scale: float = 0.0,
        phase_decoupled_timing: bool | None = None,
        phase_decoupled_phrase_gate_boundary_threshold: float | None = None,
        phase_decoupled_boundary_style_residual_scale: float = 0.18,
        debt_control_scale: float = 4.0,
        debt_pause_priority: float = 0.15,
        debt_speech_priority: float = 0.25,
        phase_free_timing: bool | None = None,
        phase_free_phrase_boundary_threshold: float | None = None,
        commit_config: CommitConfig | None = None,
        projector_config: ProjectorConfig | None = None,
        teacher_config: AlgorithmicTeacherConfig | None = None,
        offline_teacher_config: OfflineTeacherConfig | None = None,
        enable_learned_offline_teacher: bool = False,
    ) -> None:
        super().__init__()
        if phase_decoupled_timing is None:
            phase_decoupled_timing = False if phase_free_timing is None else bool(phase_free_timing)
        if phase_decoupled_phrase_gate_boundary_threshold is None:
            phase_decoupled_phrase_gate_boundary_threshold = (
                0.55 if phase_free_phrase_boundary_threshold is None else float(phase_free_phrase_boundary_threshold)
            )
        expected_public_stats_dim = len(REF_RHYTHM_STATS_KEYS)
        expected_public_trace_dim = len(REF_RHYTHM_TRACE_KEYS)
        self.public_stats_dim = int(stats_dim)
        self.public_trace_dim = int(trace_dim)
        if self.public_stats_dim != expected_public_stats_dim:
            raise ValueError(
                f"StreamingRhythmModule expects cached ref_rhythm_stats dim={expected_public_stats_dim}, "
                f"got {self.public_stats_dim}."
            )
        if self.public_trace_dim != expected_public_trace_dim:
            raise ValueError(
                f"StreamingRhythmModule expects cached ref_rhythm_trace dim={expected_public_trace_dim}, "
                f"got {self.public_trace_dim}."
            )
        self.planner_stats_dim = 2
        self.planner_trace_dim = 2
        self.unit_embedding = nn.Embedding(num_units, hidden_size)
        self.reference_descriptor = RefRhythmDescriptor(
            trace_bins=trace_bins,
            trace_horizon=trace_horizon,
            slow_topk=slow_topk,
            selector_cell_size=selector_cell_size,
            smooth_kernel=trace_smooth_kernel,
            maintained_stats_trace_only=maintained_stats_trace_only,
            emit_reference_sidecar=emit_reference_sidecar,
            runtime_phrase_bank_enable=runtime_phrase_bank_enable,
            runtime_phrase_bank_max_phrases=runtime_phrase_bank_max_phrases,
            runtime_phrase_bank_bins=runtime_phrase_bank_bins,
            runtime_phrase_select_window=runtime_phrase_select_window,
            phrase_selection_boundary_weight=phrase_selection_boundary_weight,
            phrase_selection_local_rate_weight=phrase_selection_local_rate_weight,
            phrase_selection_pause_weight=phrase_selection_pause_weight,
            phrase_selection_voice_weight=phrase_selection_voice_weight,
            phrase_selection_final_bias_weight=phrase_selection_final_bias_weight,
            phrase_selection_monotonic_bias=phrase_selection_monotonic_bias,
            phrase_selection_length_bias=phrase_selection_length_bias,
        )
        self.commit_config = commit_config or CommitConfig()
        self.commit_controller = (
            BoundaryCommitController(self.commit_config)
            if self.commit_config.mode == "boundary_phrase"
            else None
        )
        self.scheduler = MonotonicRhythmScheduler(
            hidden_size=hidden_size,
            stats_dim=self.planner_stats_dim,
            trace_dim=self.planner_trace_dim,
            max_total_logratio=max_total_logratio,
            max_unit_logratio=max_unit_logratio,
            pause_share_max=pause_share_max,
            pause_share_residual_max=pause_share_residual_max,
            boundary_feature_scale=boundary_feature_scale,
            boundary_source_cue_weight=boundary_source_cue_weight,
            pause_source_boundary_weight=pause_source_boundary_weight,
            pause_support_split_enable=pause_support_split_enable,
            pause_breath_features_enable=pause_breath_features_enable,
            pause_breath_reset_threshold=pause_breath_reset_threshold,
            min_speech_frames=min_speech_frames,
            chunk_state_enable=chunk_state_enable,
            budget_phase_feature_scale=budget_phase_feature_scale,
            phase_free_timing=bool(phase_decoupled_timing),
            phase_decoupled_boundary_style_residual_scale=phase_decoupled_boundary_style_residual_scale,
            debt_control_scale=debt_control_scale,
            debt_pause_priority=debt_pause_priority,
            debt_speech_priority=debt_speech_priority,
        )
        self.enable_learned_offline_teacher = bool(enable_learned_offline_teacher)
        if offline_teacher_config is None:
            offline_teacher_config = OfflineTeacherConfig(
                max_total_logratio=max_total_logratio,
                max_unit_logratio=max_unit_logratio,
                pause_share_max=pause_share_max,
                pause_share_residual_max=pause_share_residual_max,
                boundary_feature_scale=boundary_feature_scale,
                boundary_source_cue_weight=boundary_source_cue_weight,
                pause_source_boundary_weight=pause_source_boundary_weight,
                pause_support_split_enable=pause_support_split_enable,
                pause_breath_features_enable=pause_breath_features_enable,
                pause_breath_reset_threshold=pause_breath_reset_threshold,
                min_speech_frames=min_speech_frames,
            )
        self.offline_teacher = (
            OfflineRhythmTeacherPlanner(
                hidden_size=hidden_size,
                stats_dim=self.planner_stats_dim,
                trace_dim=self.planner_trace_dim,
                config=offline_teacher_config,
            )
            if self.enable_learned_offline_teacher
            else None
        )
        self.projector = StreamingRhythmProjector(projector_config)
        self.teacher = AlgorithmicRhythmTeacher(teacher_config)
        self.trace_reliability_enable = bool(trace_reliability_enable)
        self.trace_exhaustion_gap_start = float(max(0.0, trace_exhaustion_gap_start))
        self.trace_exhaustion_gap_end = float(max(self.trace_exhaustion_gap_start + 1.0e-6, trace_exhaustion_gap_end))
        self.trace_exhaustion_local_floor = float(min(max(trace_exhaustion_local_floor, 0.0), 1.0))
        self.trace_exhaustion_boundary_floor = float(min(max(trace_exhaustion_boundary_floor, 0.0), 1.0))
        self.trace_exhaustion_reuse_full_count = max(1, int(trace_exhaustion_reuse_full_count))
        self.trace_exhaustion_final_cell_suppress = float(
            min(max(trace_exhaustion_final_cell_suppress, 0.0), 1.0)
        )
        self.trace_anchor_aware_sampling = bool(trace_anchor_aware_sampling)
        self.trace_cold_start_min_visible_units = max(0, int(trace_cold_start_min_visible_units))
        self.trace_cold_start_full_visible_units = max(
            self.trace_cold_start_min_visible_units,
            int(trace_cold_start_full_visible_units),
        )
        self.trace_active_tail_only = bool(trace_active_tail_only)
        self.trace_offset_lookahead_units = max(0, int(trace_offset_lookahead_units))
        self.chunk_state_enable = bool(chunk_state_enable)
        self.budget_phase_feature_scale = float(min(max(budget_phase_feature_scale, 0.0), 1.0))
        self.phase_decoupled_timing = bool(phase_decoupled_timing)
        self.phase_free_timing = self.phase_decoupled_timing
        self.phase_decoupled_phrase_gate_boundary_threshold = float(
            min(max(phase_decoupled_phrase_gate_boundary_threshold, 0.0), 1.0)
        )
        self.phase_free_phrase_boundary_threshold = self.phase_decoupled_phrase_gate_boundary_threshold

        # Compatibility aliases for the older V2 surface.
        self.reference_encoder = self.reference_descriptor.encoder
        self.budget_controller = self.scheduler.window_budget
        self.redistribution_head = self.scheduler.unit_redistribution

    def init_state(self, batch_size: int, device: torch.device) -> StreamingRhythmState:
        return self.projector.init_state(batch_size=batch_size, device=device)

    @staticmethod
    def _scale_source_boundary_cue(source_boundary_cue: torch.Tensor, scale: float | None) -> torch.Tensor:
        if scale is None:
            return source_boundary_cue
        return source_boundary_cue * float(scale)

    @staticmethod
    def _resolve_unit_mask(dur_anchor_src: torch.Tensor, unit_mask: torch.Tensor | None) -> torch.Tensor:
        if unit_mask is None:
            return dur_anchor_src.gt(0).float()
        return unit_mask.float()

    def _prepare_runtime_inputs(
        self,
        *,
        dur_anchor_src: torch.Tensor,
        ref_conditioning: dict[str, torch.Tensor] | None = None,
        ref_rhythm_stats: torch.Tensor | None = None,
        ref_rhythm_trace: torch.Tensor | None = None,
        ref_mel: torch.Tensor | None = None,
        unit_mask: torch.Tensor | None = None,
        sep_hint: torch.Tensor | None = None,
        open_run_mask: torch.Tensor | None = None,
        sealed_mask: torch.Tensor | None = None,
        boundary_confidence: torch.Tensor | None = None,
        source_boundary_scale_override: float | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        ref_conditioning = self.build_reference_conditioning(
            ref_conditioning=ref_conditioning,
            ref_rhythm_stats=ref_rhythm_stats,
            ref_rhythm_trace=ref_rhythm_trace,
            ref_mel=ref_mel,
        )
        unit_mask = self._resolve_unit_mask(dur_anchor_src, unit_mask)
        source_boundary_cue = build_source_boundary_cue(
            dur_anchor_src=dur_anchor_src,
            unit_mask=unit_mask,
            sep_hint=sep_hint,
            open_run_mask=open_run_mask,
            sealed_mask=sealed_mask,
            boundary_confidence=boundary_confidence,
        )
        source_boundary_cue = self._scale_source_boundary_cue(
            source_boundary_cue,
            source_boundary_scale_override,
        )
        return ref_conditioning, unit_mask, source_boundary_cue

    @staticmethod
    def _visible_sizes(unit_mask: torch.Tensor) -> torch.Tensor:
        return unit_mask.float().sum(dim=1).long().clamp_min(1)

    @staticmethod
    def _resolve_reference_summary(
        ref_conditioning: dict[str, torch.Tensor],
        *,
        summary_key: str,
        fallback_trace_key: str,
    ) -> torch.Tensor:
        summary = ref_conditioning.get(summary_key)
        if summary is not None:
            if summary.dim() != 2:
                raise ValueError(f"{summary_key} must be [B,D], got {tuple(summary.shape)}")
            return summary
        trace = ref_conditioning[fallback_trace_key]
        if trace.dim() != 3:
            raise ValueError(f"{fallback_trace_key} must be [B,T,D], got {tuple(trace.shape)}")
        return trace.mean(dim=1)

    @staticmethod
    def _expand_summary_as_trace(summary: torch.Tensor, *, steps: int) -> torch.Tensor:
        if summary.dim() != 2:
            raise ValueError(f"summary must be [B,D], got {tuple(summary.shape)}")
        return summary.unsqueeze(1).expand(-1, steps, -1)

    @staticmethod
    def _resolve_selected_phrase_trace_summary(
        phrase_selection: dict[str, torch.Tensor] | None,
        *,
        trace_dim: int,
    ) -> torch.Tensor | None:
        if not phrase_selection:
            return None
        summary = phrase_selection.get("selected_ref_phrase_stats")
        if isinstance(summary, torch.Tensor) and summary.dim() == 2 and summary.size(-1) == trace_dim:
            return summary.float()
        trace = phrase_selection.get("selected_ref_phrase_trace")
        if isinstance(trace, torch.Tensor):
            if trace.dim() == 3 and trace.size(-1) == trace_dim:
                return trace.float().mean(dim=1)
            if trace.dim() == 2 and trace.size(-1) == trace_dim:
                return trace.float()
        return None

    @staticmethod
    def _resolve_selected_phrase_planner_summary(
        phrase_selection: dict[str, torch.Tensor] | None,
        *,
        trace_dim: int,
    ) -> torch.Tensor | None:
        if not phrase_selection:
            return None
        summary = phrase_selection.get("selected_phrase_prototype_summary")
        if isinstance(summary, torch.Tensor) and summary.dim() == 2 and summary.size(-1) == trace_dim:
            return summary.float()
        trace = phrase_selection.get("selected_planner_ref_phrase_trace")
        if isinstance(trace, torch.Tensor):
            if trace.dim() == 3 and trace.size(-1) == trace_dim:
                return trace.float().mean(dim=1)
            if trace.dim() == 2 and trace.size(-1) == trace_dim:
                return trace.float()
        return None

    @staticmethod
    def _resolve_phase_decoupled_phrase_gate(
        *,
        state: StreamingRhythmState | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        min_committed_units: int,
        full_committed_units: int,
        phrase_valid: torch.Tensor | None = None,
        phrase_boundary_strength: torch.Tensor | None = None,
        boundary_threshold: float = 0.55,
    ) -> torch.Tensor:
        if state is None:
            committed_units = torch.zeros((batch_size,), device=device, dtype=dtype)
        else:
            committed_units = state.commit_frontier.float().to(device=device, dtype=dtype)
            if committed_units.dim() == 0:
                committed_units = committed_units.unsqueeze(0)
        if full_committed_units > min_committed_units:
            gate = (
                (committed_units - float(min_committed_units))
                / float(max(full_committed_units - min_committed_units, 1))
            ).clamp(0.0, 1.0)
        elif full_committed_units > 0:
            gate = (committed_units >= float(full_committed_units)).float()
        else:
            gate = torch.ones((batch_size,), device=device, dtype=dtype)
        if phrase_valid is not None:
            valid = phrase_valid.float().to(device=device, dtype=dtype).reshape(batch_size, -1)[:, 0].clamp(0.0, 1.0)
            gate = gate * valid
        if phrase_boundary_strength is not None:
            boundary_strength = (
                phrase_boundary_strength.float().to(device=device, dtype=dtype).reshape(batch_size, -1)[:, 0]
            ).clamp(0.0, 1.0)
            threshold = float(min(max(boundary_threshold, 0.0), 1.0))
            if threshold >= 1.0:
                boundary_alpha = (boundary_strength >= threshold).float()
            else:
                boundary_alpha = ((boundary_strength - threshold) / max(1.0 - threshold, 1.0e-6)).clamp(0.0, 1.0)
            gate = gate * boundary_alpha
        return gate

    def _blend_trace_with_phrase_and_global(
        self,
        sampled_trace: torch.Tensor,
        *,
        phrase_summary: torch.Tensor | None,
        global_summary: torch.Tensor,
        trace_reliability: TraceReliabilityBundle,
    ) -> torch.Tensor:
        blended = sampled_trace.float()
        if phrase_summary is not None:
            phrase_context = self._expand_summary_as_trace(
                phrase_summary.to(device=sampled_trace.device, dtype=sampled_trace.dtype),
                steps=sampled_trace.size(1),
            )
            phrase_blend = trace_reliability.phrase_blend.to(
                device=sampled_trace.device,
                dtype=sampled_trace.dtype,
            )
            blended = (
                blended * (1.0 - phrase_blend)[:, None, None]
                + phrase_context * phrase_blend[:, None, None]
            )
        global_context = self._expand_summary_as_trace(
            global_summary.to(device=sampled_trace.device, dtype=sampled_trace.dtype),
            steps=sampled_trace.size(1),
        )
        global_blend = trace_reliability.global_blend.to(
            device=sampled_trace.device,
            dtype=sampled_trace.dtype,
        )
        return (
            blended * (1.0 - global_blend)[:, None, None]
            + global_context * global_blend[:, None, None]
        )

    def _resolve_trace_runtime_controls(
        self,
        *,
        trace_active_tail_only: bool | None = None,
        trace_offset_lookahead_units: int | None = None,
        trace_cold_start_min_visible_units: int | None = None,
        trace_cold_start_full_visible_units: int | None = None,
    ) -> tuple[bool, int | None, int, int]:
        active_tail_only = (
            self.trace_active_tail_only if trace_active_tail_only is None else bool(trace_active_tail_only)
        )
        lookahead_units = (
            self.trace_offset_lookahead_units
            if trace_offset_lookahead_units is None
            else max(0, int(trace_offset_lookahead_units))
        )
        if lookahead_units <= 0:
            lookahead_units = None
        min_visible = (
            self.trace_cold_start_min_visible_units
            if trace_cold_start_min_visible_units is None
            else max(0, int(trace_cold_start_min_visible_units))
        )
        full_visible = (
            self.trace_cold_start_full_visible_units
            if trace_cold_start_full_visible_units is None
            else max(0, int(trace_cold_start_full_visible_units))
        )
        full_visible = max(min_visible, full_visible)
        return active_tail_only, lookahead_units, min_visible, full_visible

    @staticmethod
    def _resolve_tail_reuse_count(
        state: StreamingRhythmState | None,
        *,
        phase_ptr: torch.Tensor,
    ) -> torch.Tensor:
        if state is None or state.trace_tail_reuse_count is None:
            return torch.zeros_like(phase_ptr, dtype=torch.long)
        reuse = state.trace_tail_reuse_count.long().to(device=phase_ptr.device)
        if reuse.dim() == 0:
            reuse = reuse.unsqueeze(0)
        return reuse

    @staticmethod
    def _compute_trace_coverage_alpha(
        *,
        phase_ptr: torch.Tensor,
        visible_units: torch.Tensor | None,
        min_visible: int,
        full_visible: int,
    ) -> torch.Tensor:
        ones = torch.ones_like(phase_ptr)
        if visible_units is None:
            return ones
        visible_units = visible_units.float().to(device=phase_ptr.device, dtype=phase_ptr.dtype)
        if full_visible > min_visible:
            return (
                (visible_units - float(min_visible)) / float(max(full_visible - min_visible, 1))
            ).clamp(0.0, 1.0)
        if full_visible > 0:
            return (visible_units >= float(full_visible)).float()
        return ones

    def _build_trace_reliability(
        self,
        *,
        phase_ptr: torch.Tensor,
        phase_gap: torch.Tensor | None = None,
        phase_gap_runtime: torch.Tensor | None = None,
        phase_gap_anchor: torch.Tensor | None = None,
        horizon: float,
        visible_units: torch.Tensor | None = None,
        cold_start_min_visible_units: int | None = None,
        cold_start_full_visible_units: int | None = None,
        tail_reuse_count: torch.Tensor | None = None,
    ) -> TraceReliabilityBundle:
        phase_ptr = phase_ptr.float()
        ones = torch.ones_like(phase_ptr)
        zeros = torch.zeros_like(phase_ptr)
        min_visible = (
            self.trace_cold_start_min_visible_units
            if cold_start_min_visible_units is None
            else max(0, int(cold_start_min_visible_units))
        )
        full_visible = (
            self.trace_cold_start_full_visible_units
            if cold_start_full_visible_units is None
            else max(0, int(cold_start_full_visible_units))
        )
        full_visible = max(min_visible, full_visible)
        if phase_gap_runtime is None:
            phase_gap_runtime = phase_gap
        coverage_alpha = self._compute_trace_coverage_alpha(
            phase_ptr=phase_ptr,
            visible_units=visible_units,
            min_visible=min_visible,
            full_visible=full_visible,
        )
        tail_start = max(0.0, 1.0 - float(horizon))
        tail_alpha = ((phase_ptr - tail_start) / max(float(horizon), 1.0e-6)).clamp(0.0, 1.0)
        if tail_reuse_count is None:
            tail_reuse_count = torch.zeros_like(phase_ptr, dtype=torch.long)
        if phase_gap_runtime is None:
            phase_gap_runtime = zeros
        phase_gap_runtime = phase_gap_runtime.float().to(device=phase_ptr.device, dtype=phase_ptr.dtype)
        phase_gap_runtime_for_gate = phase_gap_runtime.clamp_min(0.0)
        if phase_gap_anchor is None:
            phase_gap_anchor = zeros
        phase_gap_anchor = phase_gap_anchor.float().to(device=phase_ptr.device, dtype=phase_ptr.dtype)
        base_gate = ones
        if self.trace_reliability_enable:
            denom = max(self.trace_exhaustion_gap_end - self.trace_exhaustion_gap_start, 1.0e-6)
            gap_alpha = ((phase_gap_runtime_for_gate - self.trace_exhaustion_gap_start) / denom).clamp(0.0, 1.0)
            reuse_alpha = (
                tail_reuse_count.float().to(device=phase_ptr.device, dtype=phase_ptr.dtype)
                / float(max(self.trace_exhaustion_reuse_full_count, 1))
            ).clamp(0.0, 1.0)
            blend = (gap_alpha * tail_alpha * reuse_alpha).clamp(0.0, 1.0)
            trace_reliability = 1.0 - blend
            gating_floor_mask = phase_gap_runtime_for_gate >= float(self.trace_exhaustion_gap_end)
            trace_reliability = torch.where(gating_floor_mask, phase_ptr.new_full(trace_reliability.shape, self.trace_exhaustion_local_floor), trace_reliability)
            base_gate = (1.0 - blend * (1.0 - self.trace_exhaustion_local_floor)).clamp(0.0, 1.0)
        else:
            trace_reliability = ones
            gap_alpha = zeros
            reuse_alpha = zeros
            blend = zeros
            gating_floor_mask = torch.zeros_like(phase_ptr, dtype=torch.bool)
        coverage_blend = (1.0 - coverage_alpha).clamp(0.0, 1.0)
        phrase_blend = torch.maximum(coverage_blend, gap_alpha).clamp(0.0, 1.0)
        global_blend = blend
        local_floor = phase_ptr.new_full(trace_reliability.shape, self.trace_exhaustion_local_floor)
        boundary_floor = phase_ptr.new_full(trace_reliability.shape, self.trace_exhaustion_boundary_floor)
        gate_before_cov = torch.where(
            gating_floor_mask,
            local_floor,
            torch.maximum(base_gate, local_floor),
        )
        local_gate_candidate = gate_before_cov * coverage_alpha
        local_gate = torch.minimum(local_gate_candidate, coverage_alpha)
        local_trace_path_weight = local_gate
        boundary_trace_path_weight = torch.maximum(trace_reliability.square(), boundary_floor)
        return TraceReliabilityBundle(
            trace_reliability=trace_reliability,
            local_trace_path_weight=local_trace_path_weight,
            boundary_trace_path_weight=boundary_trace_path_weight,
            phase_gap=phase_gap_runtime,
            phase_gap_runtime=phase_gap_runtime,
            phase_gap_anchor=phase_gap_anchor,
            coverage_alpha=coverage_alpha,
            local_gate=local_gate,
            blend=blend,
            tail_alpha=tail_alpha,
            gap_alpha=gap_alpha,
            reuse_alpha=reuse_alpha,
            tail_reuse_count=tail_reuse_count.float(),
            tail_active=tail_alpha.gt(0.0).float(),
            phrase_blend=phrase_blend,
            global_blend=global_blend,
        )

    def _next_trace_tail_reuse_count(
        self,
        *,
        state: StreamingRhythmState,
        horizon: float,
    ) -> torch.Tensor:
        previous_count = (
            state.trace_tail_reuse_count.long()
            if state.trace_tail_reuse_count is not None
            else torch.zeros_like(state.commit_frontier.long())
        )
        tail_start = max(0.0, 1.0 - float(horizon))
        tail_active = state.phase_ptr.float() >= (tail_start - 1.0e-6)
        return torch.where(tail_active, previous_count + 1, torch.zeros_like(previous_count))

    def _sample_trace_pair(
        self,
        *,
        ref_conditioning: dict[str, torch.Tensor],
        phase_ptr: torch.Tensor,
        window_size: int,
        unit_mask: torch.Tensor,
        dur_anchor_src: torch.Tensor | None = None,
        horizon: float | None,
        state: StreamingRhythmState | None = None,
        phrase_selection: dict[str, torch.Tensor] | None = None,
        trace_active_tail_only: bool | None = None,
        trace_offset_lookahead_units: int | None = None,
        trace_cold_start_min_visible_units: int | None = None,
        trace_cold_start_full_visible_units: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, TraceReliabilityBundle]:
        visible_sizes = self._visible_sizes(unit_mask)
        effective_horizon = self.reference_descriptor.encoder.trace_horizon if horizon is None else float(horizon)
        (
            effective_trace_active_tail_only,
            effective_trace_offset_lookahead_units,
            effective_trace_cold_start_min_visible_units,
            effective_trace_cold_start_full_visible_units,
        ) = self._resolve_trace_runtime_controls(
            trace_active_tail_only=trace_active_tail_only,
            trace_offset_lookahead_units=trace_offset_lookahead_units,
            trace_cold_start_min_visible_units=trace_cold_start_min_visible_units,
            trace_cold_start_full_visible_units=trace_cold_start_full_visible_units,
        )
        anchor_durations = None
        if self.trace_anchor_aware_sampling and dur_anchor_src is not None:
            anchor_durations = dur_anchor_src.float() * unit_mask.float()
        commit_frontier = state.commit_frontier if state is not None else None
        trace_context = self.sample_trace_window(
            ref_conditioning=ref_conditioning,
            phase_ptr=phase_ptr,
            window_size=window_size,
            horizon=effective_horizon,
            visible_sizes=visible_sizes,
            anchor_durations=anchor_durations,
            commit_frontier=commit_frontier,
            lookahead_units=effective_trace_offset_lookahead_units,
            active_tail_only=effective_trace_active_tail_only,
        )
        planner_trace_context = self.sample_planner_trace_window(
            ref_conditioning=ref_conditioning,
            phase_ptr=phase_ptr,
            window_size=window_size,
            horizon=effective_horizon,
            visible_sizes=visible_sizes,
            anchor_durations=anchor_durations,
            commit_frontier=commit_frontier,
            lookahead_units=effective_trace_offset_lookahead_units,
            active_tail_only=effective_trace_active_tail_only,
        )
        phase_gap_anchor = state.phase_ptr_gap if state is not None else None
        phase_gap_runtime = None
        if (
            state is not None
            and state.phase_anchor_progress is not None
            and dur_anchor_src is not None
        ):
            visible_total = (dur_anchor_src.float().clamp_min(0.0) * unit_mask.float()).sum(dim=1).clamp_min(1.0)
            current_progress_ratio = state.phase_anchor_progress.float() / visible_total
            phase_gap_runtime = phase_ptr.float() - current_progress_ratio
        trace_reliability = self._build_trace_reliability(
            phase_ptr=phase_ptr,
            phase_gap_runtime=phase_gap_runtime,
            phase_gap_anchor=phase_gap_anchor,
            horizon=effective_horizon,
            visible_units=visible_sizes.float(),
            cold_start_min_visible_units=effective_trace_cold_start_min_visible_units,
            cold_start_full_visible_units=effective_trace_cold_start_full_visible_units,
            tail_reuse_count=self._resolve_tail_reuse_count(state, phase_ptr=phase_ptr),
        )
        phrase_summary = self._resolve_selected_phrase_trace_summary(
            phrase_selection,
            trace_dim=trace_context.size(-1),
        )
        global_summary = self._resolve_reference_summary(
            ref_conditioning,
            summary_key="slow_rhythm_summary",
            fallback_trace_key="ref_rhythm_trace",
        )
        trace_context = self._blend_trace_with_phrase_and_global(
            trace_context,
            phrase_summary=phrase_summary,
            global_summary=global_summary,
            trace_reliability=trace_reliability,
        )
        return trace_context, planner_trace_context, trace_reliability

    def _sample_phase_decoupled_trace_pair(
        self,
        *,
        ref_conditioning: dict[str, torch.Tensor],
        window_size: int,
        unit_mask: torch.Tensor,
        state: StreamingRhythmState | None = None,
        phrase_selection: dict[str, torch.Tensor] | None = None,
        dur_anchor_src: torch.Tensor | None = None,
        trace_cold_start_min_visible_units: int | None = None,
        trace_cold_start_full_visible_units: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, TraceReliabilityBundle]:
        batch_size = int(unit_mask.size(0))
        device = unit_mask.device
        dtype = unit_mask.dtype
        (
            _,
            _,
            effective_trace_cold_start_min_visible_units,
            effective_trace_cold_start_full_visible_units,
        ) = self._resolve_trace_runtime_controls(
            trace_cold_start_min_visible_units=trace_cold_start_min_visible_units,
            trace_cold_start_full_visible_units=trace_cold_start_full_visible_units,
        )
        phrase_trace_summary = self._resolve_selected_phrase_trace_summary(
            phrase_selection,
            trace_dim=ref_conditioning["ref_rhythm_trace"].size(-1),
        )
        planner_phrase_summary = self._resolve_selected_phrase_planner_summary(
            phrase_selection,
            trace_dim=ref_conditioning["planner_ref_trace"].size(-1),
        )
        global_trace_summary = self._resolve_reference_summary(
            ref_conditioning,
            summary_key="slow_rhythm_summary",
            fallback_trace_key="ref_rhythm_trace",
        ).to(device=device, dtype=ref_conditioning["ref_rhythm_trace"].dtype)
        planner_global_summary = self._resolve_reference_summary(
            ref_conditioning,
            summary_key="planner_slow_rhythm_summary",
            fallback_trace_key="planner_ref_trace",
        ).to(device=device, dtype=ref_conditioning["planner_ref_trace"].dtype)
        selected_valid = None if not phrase_selection else (
            phrase_selection.get("selected_phrase_prototype_valid")
            if phrase_selection.get("selected_phrase_prototype_valid") is not None
            else phrase_selection.get("selected_ref_phrase_valid")
        )
        selected_boundary_strength = None if not phrase_selection else (
            phrase_selection.get("selected_phrase_prototype_boundary_strength")
            if phrase_selection.get("selected_phrase_prototype_boundary_strength") is not None
            else phrase_selection.get("selected_ref_phrase_boundary_strength")
        )
        phrase_gate = self._resolve_phase_decoupled_phrase_gate(
            state=state,
            batch_size=batch_size,
            device=device,
            dtype=global_trace_summary.dtype,
            min_committed_units=effective_trace_cold_start_min_visible_units,
            full_committed_units=effective_trace_cold_start_full_visible_units,
            phrase_valid=selected_valid,
            phrase_boundary_strength=selected_boundary_strength,
            boundary_threshold=self.phase_decoupled_phrase_gate_boundary_threshold,
        )
        global_gate = (1.0 - phrase_gate).clamp(0.0, 1.0)
        if phrase_trace_summary is None:
            trace_summary = global_trace_summary
            phrase_gate = torch.zeros_like(phrase_gate)
            global_gate = torch.ones_like(global_gate)
        else:
            phrase_trace_summary = phrase_trace_summary.to(device=device, dtype=global_trace_summary.dtype)
            trace_summary = (
                phrase_trace_summary * phrase_gate[:, None]
                + global_trace_summary * global_gate[:, None]
            )
        if planner_phrase_summary is None:
            planner_summary = planner_global_summary
        else:
            planner_phrase_summary = planner_phrase_summary.to(device=device, dtype=planner_global_summary.dtype)
            planner_summary = (
                planner_phrase_summary * phrase_gate[:, None]
                + planner_global_summary * global_gate[:, None]
            )
        trace_context = self._expand_summary_as_trace(trace_summary, steps=window_size)
        planner_trace_context = self._expand_summary_as_trace(planner_summary, steps=window_size)
        if state is not None:
            phase_ptr = state.phase_ptr.float().to(device=device)
            phase_gap_anchor = state.phase_ptr_gap
            if phase_gap_anchor is None:
                phase_gap_anchor = torch.zeros_like(phase_ptr)
            else:
                phase_gap_anchor = phase_gap_anchor.float().to(device=device)
        else:
            phase_ptr = torch.zeros((batch_size,), device=device, dtype=trace_context.dtype)
            phase_gap_anchor = torch.zeros_like(phase_ptr)
        phase_gap_runtime = torch.zeros_like(phase_ptr)
        if (
            state is not None
            and state.phase_anchor_progress is not None
            and dur_anchor_src is not None
        ):
            visible_total = (dur_anchor_src.float().clamp_min(0.0) * unit_mask.float()).sum(dim=1).clamp_min(1.0)
            current_progress_ratio = state.phase_anchor_progress.float() / visible_total
            phase_gap_runtime = phase_ptr.float() - current_progress_ratio.to(device=device)
        zeros = torch.zeros_like(phrase_gate)
        trace_reliability = TraceReliabilityBundle(
            trace_reliability=torch.ones_like(phrase_gate),
            local_trace_path_weight=zeros,
            boundary_trace_path_weight=zeros,
            phase_gap=phase_gap_runtime,
            phase_gap_runtime=phase_gap_runtime,
            phase_gap_anchor=phase_gap_anchor,
            coverage_alpha=phrase_gate,
            blend=zeros,
            tail_alpha=zeros,
            gap_alpha=zeros,
            reuse_alpha=zeros,
            tail_reuse_count=self._resolve_tail_reuse_count(state, phase_ptr=phase_ptr).float(),
            tail_active=zeros,
            local_gate=zeros,
            phrase_blend=phrase_gate,
            global_blend=global_gate,
        )
        return trace_context, planner_trace_context, trace_reliability

    def encode_reference(self, ref_mel: torch.Tensor) -> dict[str, torch.Tensor]:
        return self._finalize_reference_conditioning(self.reference_descriptor(ref_mel))

    @staticmethod
    def _finalize_reference_summary(
        enriched: dict[str, torch.Tensor],
        *,
        summary_key: str,
        memory_key: str,
        fallback_trace_key: str | None = None,
    ) -> None:
        source_key = f"{summary_key}_source"
        memory = enriched.get(memory_key)
        if memory is not None and memory.dim() != 3:
            raise ValueError(
                f"{memory_key} must be rank-3 [B, K, D], got {tuple(memory.shape)}."
            )
        summary = enriched.get(summary_key)
        if summary is not None:
            if source_key not in enriched:
                enriched[source_key] = "sidecar" if memory is not None else "provided"
            return
        if memory is not None:
            enriched[summary_key] = memory.mean(dim=1)
            enriched[source_key] = f"{memory_key}_mean"
            return
        if fallback_trace_key is not None:
            trace = enriched.get(fallback_trace_key)
            if trace is not None:
                if trace.dim() != 3:
                    raise ValueError(
                        f"{fallback_trace_key} must be rank-3 [B, T, D], got {tuple(trace.shape)}."
                    )
                enriched[summary_key] = trace.mean(dim=1)
                enriched[source_key] = f"{fallback_trace_key}_mean"
                return
        enriched[source_key] = "absent"

    def _finalize_reference_conditioning(
        self,
        enriched: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        self._finalize_reference_summary(
            enriched,
            summary_key="slow_rhythm_summary",
            memory_key="slow_rhythm_memory",
        )
        self._finalize_reference_summary(
            enriched,
            summary_key="planner_slow_rhythm_summary",
            memory_key="planner_slow_rhythm_memory",
            fallback_trace_key="planner_ref_trace",
        )
        return enriched

    @staticmethod
    def _merge_reference_sidecars(
        enriched: dict[str, torch.Tensor],
        ref_conditioning: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if not ref_conditioning:
            return enriched
        sidecar_keys = {
            "slow_rhythm_memory",
            "slow_rhythm_summary",
            "planner_slow_rhythm_memory",
            "planner_slow_rhythm_summary",
            "selector_meta_indices",
            "selector_meta_scores",
            "selector_meta_starts",
            "selector_meta_ends",
            "ref_phrase_trace",
            "planner_ref_phrase_trace",
            "ref_phrase_valid",
            "ref_phrase_lengths",
            "ref_phrase_starts",
            "ref_phrase_ends",
            "ref_phrase_boundary_strength",
            "ref_phrase_stats",
        }
        for key in sidecar_keys:
            value = ref_conditioning.get(key)
            if value is not None:
                enriched[key] = value
        return enriched

    @staticmethod
    def _resolve_ref_phrase_ptr(
        *,
        state: StreamingRhythmState,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        ref_phrase_ptr = state.ref_phrase_ptr
        if ref_phrase_ptr is None:
            return torch.zeros((batch_size,), dtype=torch.long, device=device)
        ref_phrase_ptr = ref_phrase_ptr.long().to(device=device)
        if ref_phrase_ptr.dim() == 0:
            ref_phrase_ptr = ref_phrase_ptr.unsqueeze(0)
        if ref_phrase_ptr.size(0) != batch_size:
            raise ValueError(
                f"ref_phrase_ptr batch mismatch: got {tuple(ref_phrase_ptr.shape)} for batch_size={batch_size}"
            )
        return ref_phrase_ptr

    def _select_scheduler_phrase_bank(
        self,
        *,
        ref_conditioning: dict[str, torch.Tensor],
        state: StreamingRhythmState,
        batch_size: int,
        device: torch.device,
        chunk_state: ChunkStateBundle | None = None,
        strict_pointer_only: bool = False,
    ) -> dict[str, torch.Tensor] | None:
        has_phrase_sidecar = "ref_phrase_trace" in ref_conditioning
        if not (self.reference_descriptor.runtime_phrase_bank_enable or has_phrase_sidecar):
            return None
        ref_phrase_ptr = self._resolve_ref_phrase_ptr(
            state=state,
            batch_size=batch_size,
            device=device,
        )
        return self.reference_descriptor.select_phrase_bank(
            ref_conditioning,
            ref_phrase_ptr=ref_phrase_ptr,
            query_chunk_summary=(chunk_state.chunk_summary if chunk_state is not None else None),
            query_commit_confidence=(
                chunk_state.commit_now_prob if chunk_state is not None else state.commit_confidence
            ),
            query_phrase_close_prob=(chunk_state.phrase_close_prob if chunk_state is not None else None),
            strict_pointer_only=strict_pointer_only,
        )

    def _plan_discrete_commit(
        self,
        *,
        planner,
        source_boundary_cue: torch.Tensor,
        unit_mask: torch.Tensor,
        state: StreamingRhythmState,
        boundary_confidence: torch.Tensor | None,
        sep_hint: torch.Tensor | None,
        open_run_mask: torch.Tensor | None,
        sealed_mask: torch.Tensor | None,
        force_full_commit: bool,
    ):
        if self.commit_controller is None:
            return None
        if force_full_commit:
            visible_len = unit_mask.float().sum(dim=1).long()
            return BoundaryCommitDecision(
                commit_end=visible_len,
                committed=visible_len > state.commit_frontier.long(),
                commit_score_unit=planner.boundary_score_unit.float() * unit_mask.float(),
                eligible_mask_unit=unit_mask.float(),
                commit_confidence=torch.ones_like(visible_len, dtype=planner.boundary_score_unit.dtype).unsqueeze(-1),
            )
        return self.commit_controller(
            boundary_score_unit=planner.boundary_score_unit,
            source_boundary_cue=source_boundary_cue,
            unit_mask=unit_mask,
            state=state,
            boundary_confidence=boundary_confidence,
            sep_hint=sep_hint,
            open_run_mask=open_run_mask,
            sealed_mask=sealed_mask,
        )

    @staticmethod
    def _build_active_phrase_bounds(
        *,
        state: StreamingRhythmState,
        planned_commit_frontier: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visible_len = unit_mask.float().sum(dim=1).long()
        active_start = state.commit_frontier.long().clamp(min=0, max=unit_mask.size(1))
        active_start = torch.minimum(active_start, visible_len)
        active_end = planned_commit_frontier.long().clamp(min=0, max=unit_mask.size(1))
        active_end = torch.minimum(torch.maximum(active_end, active_start), visible_len)
        return active_start, active_end

    def _build_segment_masks(
        self,
        *,
        boundary_score_unit: torch.Tensor,
        unit_mask: torch.Tensor,
        active_phrase_start: torch.Tensor,
        active_phrase_end: torch.Tensor,
        sep_hint: torch.Tensor | None,
        open_run_mask: torch.Tensor | None,
        sealed_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        steps = torch.arange(unit_mask.size(1), device=unit_mask.device)[None, :]
        segment_mask = (
            (steps >= active_phrase_start[:, None])
            & (steps < active_phrase_end[:, None])
            & (unit_mask > 0.5)
        ).float()
        pause_mask = segment_mask.clone()
        if sealed_mask is not None:
            pause_mask = pause_mask * (sealed_mask > 0.5).float()
        if open_run_mask is not None:
            pause_mask = pause_mask * (open_run_mask <= 0).float()
        boundary_gate = (boundary_score_unit.float() >= float(self.projector.config.boundary_commit_threshold)).float()
        if sep_hint is not None:
            boundary_gate = torch.maximum(boundary_gate, sep_hint.float().clamp(0.0, 1.0))
        pause_mask = pause_mask * boundary_gate
        if bool((pause_mask.sum(dim=1) <= 0.0).any().item()):
            fallback = torch.zeros_like(pause_mask)
            valid_rows = active_phrase_end > active_phrase_start
            if bool(valid_rows.any().item()):
                fallback_index = (active_phrase_end[valid_rows] - 1).long().clamp(min=0, max=unit_mask.size(1) - 1)
                fallback[valid_rows] = fallback[valid_rows].scatter(
                    1,
                    fallback_index.unsqueeze(1),
                    torch.ones((int(valid_rows.sum().item()), 1), device=unit_mask.device, dtype=unit_mask.dtype),
                )
                pause_mask = torch.where((pause_mask.sum(dim=1, keepdim=True) > 0.0), pause_mask, fallback * segment_mask)
        return segment_mask, pause_mask

    @staticmethod
    def _sum_prefix_exec(
        previous_exec: torch.Tensor | None,
        *,
        frontier: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> torch.Tensor:
        if previous_exec is None:
            return unit_mask.new_zeros((unit_mask.size(0), 1))
        previous_exec = previous_exec.float().to(device=unit_mask.device)
        if previous_exec.size(1) < unit_mask.size(1):
            pad = previous_exec.new_zeros((previous_exec.size(0), unit_mask.size(1) - previous_exec.size(1)))
            previous_exec = torch.cat([previous_exec, pad], dim=1)
        elif previous_exec.size(1) > unit_mask.size(1):
            previous_exec = previous_exec[:, : unit_mask.size(1)]
        steps = torch.arange(unit_mask.size(1), device=unit_mask.device)[None, :]
        prefix_mask = (steps < frontier[:, None]).float() * unit_mask.float()
        return (previous_exec * prefix_mask).sum(dim=1, keepdim=True)

    def _build_phrase_budget_views(
        self,
        *,
        planner,
        state: StreamingRhythmState,
        dur_anchor_src: torch.Tensor,
        unit_mask: torch.Tensor,
        segment_mask: torch.Tensor,
        pause_segment_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total_speech_budget = planner.speech_budget_win.float()
        total_pause_budget = planner.pause_budget_win.float()
        prefix_speech = self._sum_prefix_exec(
            state.previous_speech_exec,
            frontier=state.commit_frontier.long(),
            unit_mask=unit_mask,
        )
        prefix_pause = self._sum_prefix_exec(
            state.previous_pause_exec,
            frontier=state.commit_frontier.long(),
            unit_mask=unit_mask,
        )
        segment_mask_f = segment_mask.float()
        pause_segment_mask_f = pause_segment_mask.float()
        segment_any = segment_mask_f.sum(dim=1, keepdim=True) > 0.0
        desired_speech_unit = dur_anchor_src.float().clamp_min(0.0) * torch.exp(planner.dur_logratio_unit.float())
        desired_segment_speech = (desired_speech_unit * segment_mask_f).sum(dim=1, keepdim=True)
        remaining_speech_budget = (total_speech_budget - prefix_speech).clamp_min(0.0)
        desired_segment_speech = torch.minimum(desired_segment_speech, remaining_speech_budget)
        phrase_speech_budget = torch.where(
            segment_any,
            prefix_speech + desired_segment_speech,
            prefix_speech,
        )

        pause_support = getattr(planner, "pause_support_prob_unit", None)
        if pause_support is None:
            pause_support = getattr(planner, "pause_allocation_weight_unit", None)
        if pause_support is None:
            pause_support = planner.pause_weight_unit
        pause_support = pause_support.float().clamp_min(0.0)
        pause_segment_any = pause_segment_mask_f.sum(dim=1, keepdim=True) > 0.0
        pause_strength = torch.where(
            pause_segment_any,
            (pause_support * pause_segment_mask_f).sum(dim=1, keepdim=True)
            / pause_segment_mask_f.sum(dim=1, keepdim=True).clamp_min(1.0),
            torch.zeros_like(total_pause_budget),
        ).clamp(0.0, 1.0)
        boundary_strength = torch.where(
            pause_segment_any,
            (planner.boundary_score_unit.float() * pause_segment_mask_f).amax(dim=1, keepdim=True).clamp(0.0, 1.0),
            torch.zeros_like(total_pause_budget),
        )
        phrase_close_prob = getattr(planner, "phrase_close_prob", None)
        if phrase_close_prob is None:
            phrase_close_prob = boundary_strength
        else:
            phrase_close_prob = phrase_close_prob.float().reshape(total_pause_budget.size(0), 1).clamp(0.0, 1.0)
        pause_ratio_hint = (total_pause_budget / total_speech_budget.clamp_min(1.0e-6)).clamp(0.0, 1.0)
        desired_segment_pause = desired_segment_speech * pause_ratio_hint * torch.maximum(
            pause_strength,
            torch.maximum(boundary_strength, phrase_close_prob),
        )
        remaining_pause_budget = (total_pause_budget - prefix_pause).clamp_min(0.0)
        desired_segment_pause = torch.minimum(desired_segment_pause, remaining_pause_budget)
        phrase_pause_budget = torch.where(
            pause_segment_any,
            prefix_pause + desired_segment_pause,
            prefix_pause,
        )
        return phrase_speech_budget, phrase_pause_budget

    def _attach_commit_phrase_plan_to_planner(
        self,
        *,
        planner,
        commit_decision,
        active_phrase_start: torch.Tensor,
        active_phrase_end: torch.Tensor,
        segment_mask: torch.Tensor,
        pause_segment_mask: torch.Tensor,
        phrase_speech_budget: torch.Tensor,
        phrase_pause_budget: torch.Tensor,
        phrase_selection: dict[str, torch.Tensor] | None,
    ):
        planner.commit_boundary_logit_unit = commit_decision.commit_score_unit
        planner.commit_mask_unit = commit_decision.eligible_mask_unit
        planner.planned_commit_frontier = commit_decision.commit_end
        planner.commit_confidence = commit_decision.commit_confidence
        planner.active_phrase_start = active_phrase_start
        planner.active_phrase_end = active_phrase_end
        planner.segment_mask_unit = segment_mask
        planner.pause_segment_mask_unit = pause_segment_mask
        planner.phrase_speech_budget_win = phrase_speech_budget
        planner.phrase_pause_budget_win = phrase_pause_budget
        if phrase_selection is not None:
            planner.ref_phrase_index = phrase_selection.get("selected_ref_phrase_index")
            planner.ref_phrase_trace = phrase_selection.get("selected_ref_phrase_trace")
            planner.ref_phrase_stats = phrase_selection.get("selected_ref_phrase_stats")
        return planner

    def _enrich_planner_with_runtime_commit(
        self,
        *,
        planner,
        ref_conditioning: dict[str, torch.Tensor],
        dur_anchor_src: torch.Tensor,
        unit_mask: torch.Tensor,
        state: StreamingRhythmState,
        source_boundary_cue: torch.Tensor,
        boundary_confidence: torch.Tensor | None,
        sep_hint: torch.Tensor | None,
        open_run_mask: torch.Tensor | None,
        sealed_mask: torch.Tensor | None,
        force_full_commit: bool,
        phrase_selection: dict[str, torch.Tensor] | None = None,
    ):
        if self.commit_controller is None:
            return planner, None
        commit_decision = self._plan_discrete_commit(
            planner=planner,
            source_boundary_cue=source_boundary_cue,
            unit_mask=unit_mask,
            state=state,
            boundary_confidence=boundary_confidence,
            sep_hint=sep_hint,
            open_run_mask=open_run_mask,
            sealed_mask=sealed_mask,
            force_full_commit=force_full_commit,
        )
        active_phrase_start, active_phrase_end = self._build_active_phrase_bounds(
            state=state,
            planned_commit_frontier=commit_decision.commit_end,
            unit_mask=unit_mask,
        )
        segment_mask, pause_segment_mask = self._build_segment_masks(
            boundary_score_unit=planner.boundary_score_unit,
            unit_mask=unit_mask,
            active_phrase_start=active_phrase_start,
            active_phrase_end=active_phrase_end,
            sep_hint=sep_hint,
            open_run_mask=open_run_mask,
            sealed_mask=sealed_mask,
        )
        phrase_speech_budget, phrase_pause_budget = self._build_phrase_budget_views(
            planner=planner,
            state=state,
            dur_anchor_src=dur_anchor_src,
            unit_mask=unit_mask,
            segment_mask=segment_mask,
            pause_segment_mask=pause_segment_mask,
        )
        selection = phrase_selection
        if selection is None and (
            self.reference_descriptor.runtime_phrase_bank_enable or "ref_phrase_trace" in ref_conditioning
        ):
            ref_phrase_ptr = self._resolve_ref_phrase_ptr(
                state=state,
                batch_size=unit_mask.size(0),
                device=unit_mask.device,
            )
            selection = self.reference_descriptor.select_phrase_bank(
                ref_conditioning,
                ref_phrase_ptr=ref_phrase_ptr,
            )
        planner = self._attach_commit_phrase_plan_to_planner(
            planner=planner,
            commit_decision=commit_decision,
            active_phrase_start=active_phrase_start,
            active_phrase_end=active_phrase_end,
            segment_mask=segment_mask,
            pause_segment_mask=pause_segment_mask,
            phrase_speech_budget=phrase_speech_budget,
            phrase_pause_budget=phrase_pause_budget,
            phrase_selection=selection,
        )
        return planner, commit_decision

    def _next_ref_phrase_ptr(
        self,
        *,
        state: StreamingRhythmState,
        ref_conditioning: dict[str, torch.Tensor],
        planner,
        committed: torch.Tensor,
        phase_decoupled_timing: bool = False,
    ) -> torch.Tensor:
        current = self._resolve_ref_phrase_ptr(
            state=state,
            batch_size=planner.boundary_score_unit.size(0),
            device=planner.boundary_score_unit.device,
        )
        valid = ref_conditioning.get("ref_phrase_valid")
        if valid is None:
            return current
        valid_count = valid.long().sum(dim=1).clamp_min(1)
        if phase_decoupled_timing or getattr(planner, "ref_phrase_index", None) is None:
            next_ptr = torch.where(committed.bool(), current + 1, current)
        else:
            selected = planner.ref_phrase_index.long()
            next_ptr = torch.where(committed.bool(), selected + 1, current)
        return torch.minimum(next_ptr, valid_count - 1)

    def build_reference_conditioning(
        self,
        *,
        ref_conditioning: dict[str, torch.Tensor] | None = None,
        ref_rhythm_stats: torch.Tensor | None = None,
        ref_rhythm_trace: torch.Tensor | None = None,
        ref_mel: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if ref_conditioning is not None:
            if "ref_rhythm_stats" not in ref_conditioning or "ref_rhythm_trace" not in ref_conditioning:
                raise ValueError("ref_conditioning must contain ref_rhythm_stats and ref_rhythm_trace.")
            enriched = self.reference_descriptor.from_stats_trace(
                ref_conditioning["ref_rhythm_stats"],
                ref_conditioning["ref_rhythm_trace"],
                selector=self.reference_descriptor.selector,
                include_sidecar=self.reference_descriptor.emit_reference_sidecar,
                runtime_phrase_bank_enable=self.reference_descriptor.runtime_phrase_bank_enable,
                runtime_phrase_bank_max_phrases=self.reference_descriptor.runtime_phrase_bank_max_phrases,
                runtime_phrase_bank_bins=self.reference_descriptor.runtime_phrase_bank_bins,
            )
            enriched = self._merge_reference_sidecars(
                enriched,
                {k: v for k, v in ref_conditioning.items() if v is not None},
            )
            return self._finalize_reference_conditioning(enriched)
        if ref_rhythm_stats is not None and ref_rhythm_trace is not None:
            return self._finalize_reference_conditioning(
                self.reference_descriptor.from_stats_trace(
                    ref_rhythm_stats,
                    ref_rhythm_trace,
                    selector=self.reference_descriptor.selector,
                    include_sidecar=self.reference_descriptor.emit_reference_sidecar,
                    runtime_phrase_bank_enable=self.reference_descriptor.runtime_phrase_bank_enable,
                    runtime_phrase_bank_max_phrases=self.reference_descriptor.runtime_phrase_bank_max_phrases,
                    runtime_phrase_bank_bins=self.reference_descriptor.runtime_phrase_bank_bins,
                )
            )
        if ref_mel is None:
            raise ValueError('Need either (ref_rhythm_stats, ref_rhythm_trace) or ref_mel.')
        return self.encode_reference(ref_mel)

    def sample_trace_window(
        self,
        *,
        ref_conditioning: dict[str, torch.Tensor],
        phase_ptr: torch.Tensor,
        window_size: int,
        horizon: float | None = None,
        visible_sizes: torch.Tensor | None = None,
        anchor_durations: torch.Tensor | None = None,
        commit_frontier: torch.Tensor | None = None,
        lookahead_units: int | None = None,
        active_tail_only: bool = False,
    ) -> torch.Tensor:
        return self.reference_descriptor.sample_trace_window(
            ref_conditioning,
            phase_ptr=phase_ptr,
            window_size=window_size,
            horizon=horizon,
            visible_sizes=visible_sizes,
            anchor_durations=anchor_durations,
            commit_frontier=commit_frontier,
            lookahead_units=lookahead_units,
            active_tail_only=active_tail_only,
        )

    def sample_planner_trace_window(
        self,
        *,
        ref_conditioning: dict[str, torch.Tensor],
        phase_ptr: torch.Tensor,
        window_size: int,
        horizon: float | None = None,
        visible_sizes: torch.Tensor | None = None,
        anchor_durations: torch.Tensor | None = None,
        commit_frontier: torch.Tensor | None = None,
        lookahead_units: int | None = None,
        active_tail_only: bool = False,
    ) -> torch.Tensor:
        return self.reference_descriptor.sample_planner_trace_window(
            ref_conditioning,
            phase_ptr=phase_ptr,
            window_size=window_size,
            horizon=horizon,
            visible_sizes=visible_sizes,
            anchor_durations=anchor_durations,
            commit_frontier=commit_frontier,
            lookahead_units=lookahead_units,
            active_tail_only=active_tail_only,
        )

    def forward(
        self,
        *,
        content_units: torch.Tensor,
        dur_anchor_src: torch.Tensor,
        ref_conditioning: dict[str, torch.Tensor] | None = None,
        ref_rhythm_stats: torch.Tensor | None = None,
        ref_rhythm_trace: torch.Tensor | None = None,
        ref_mel: torch.Tensor | None = None,
        unit_mask: torch.Tensor | None = None,
        open_run_mask: torch.Tensor | None = None,
        sealed_mask: torch.Tensor | None = None,
        sep_hint: torch.Tensor | None = None,
        boundary_confidence: torch.Tensor | None = None,
        state: StreamingRhythmState | None = None,
        trace_horizon: float | None = None,
        trace_active_tail_only: bool | None = None,
        trace_offset_lookahead_units: int | None = None,
        trace_cold_start_min_visible_units: int | None = None,
        trace_cold_start_full_visible_units: int | None = None,
        phase_decoupled_timing: bool | None = None,
        phase_free_timing: bool | None = None,
        projector_reuse_prefix: bool = True,
        projector_force_full_commit: bool = False,
        projector_pause_topk_ratio_override: float | None = None,
        source_boundary_scale_override: float | None = None,
    ):
        ref_conditioning, unit_mask, source_boundary_cue = self._prepare_runtime_inputs(
            dur_anchor_src=dur_anchor_src,
            ref_conditioning=ref_conditioning,
            ref_rhythm_stats=ref_rhythm_stats,
            ref_rhythm_trace=ref_rhythm_trace,
            ref_mel=ref_mel,
            unit_mask=unit_mask,
            sep_hint=sep_hint,
            open_run_mask=open_run_mask,
            sealed_mask=sealed_mask,
            boundary_confidence=boundary_confidence,
            source_boundary_scale_override=source_boundary_scale_override,
        )
        if state is None:
            state = self.init_state(batch_size=content_units.size(0), device=content_units.device)

        effective_trace_horizon = (
            self.reference_descriptor.encoder.trace_horizon if trace_horizon is None else float(trace_horizon)
        )
        effective_phase_decoupled_timing = self.phase_decoupled_timing
        if phase_free_timing is not None:
            effective_phase_decoupled_timing = bool(phase_free_timing)
        if phase_decoupled_timing is not None:
            effective_phase_decoupled_timing = bool(phase_decoupled_timing)
        unit_embed = self.unit_embedding(content_units.long().clamp_min(0))
        chunk_state = None
        if getattr(self.scheduler, "chunk_state_head", None) is not None:
            chunk_state = self.scheduler.chunk_state_head(
                unit_mask=unit_mask,
                state=state,
                source_boundary_cue=source_boundary_cue,
                boundary_score_unit=None,
                sep_hint=sep_hint,
                open_run_mask=open_run_mask,
                sealed_mask=sealed_mask,
                boundary_confidence=boundary_confidence,
            )
        phrase_selection = self._select_scheduler_phrase_bank(
            ref_conditioning=ref_conditioning,
            state=state,
            batch_size=content_units.size(0),
            device=content_units.device,
            chunk_state=(None if effective_phase_decoupled_timing else chunk_state),
            strict_pointer_only=effective_phase_decoupled_timing,
        )
        if effective_phase_decoupled_timing:
            trace_context, planner_trace_context, trace_reliability = self._sample_phase_decoupled_trace_pair(
                ref_conditioning=ref_conditioning,
                window_size=content_units.size(1),
                unit_mask=unit_mask,
                state=state,
                phrase_selection=phrase_selection,
                dur_anchor_src=dur_anchor_src,
                trace_cold_start_min_visible_units=trace_cold_start_min_visible_units,
                trace_cold_start_full_visible_units=trace_cold_start_full_visible_units,
            )
        else:
            trace_context, planner_trace_context, trace_reliability = self._sample_trace_pair(
                ref_conditioning=ref_conditioning,
                phase_ptr=state.phase_ptr,
                window_size=content_units.size(1),
                unit_mask=unit_mask,
                dur_anchor_src=dur_anchor_src,
                horizon=effective_trace_horizon,
                state=state,
                phrase_selection=phrase_selection,
                trace_active_tail_only=trace_active_tail_only,
                trace_offset_lookahead_units=trace_offset_lookahead_units,
                trace_cold_start_min_visible_units=trace_cold_start_min_visible_units,
                trace_cold_start_full_visible_units=trace_cold_start_full_visible_units,
            )
        planner = self.scheduler(
            unit_states=unit_embed,
            dur_anchor_src=dur_anchor_src,
            unit_mask=unit_mask,
            ref_conditioning=ref_conditioning,
            trace_context=trace_context,
            planner_trace_context=planner_trace_context,
            state=state,
            source_boundary_cue=source_boundary_cue,
            sep_hint=sep_hint,
            open_run_mask=open_run_mask,
            sealed_mask=sealed_mask,
            boundary_confidence=boundary_confidence,
            phrase_selection=phrase_selection,
            trace_reliability=trace_reliability,
            trace_exhaustion_final_cell_suppress=self.trace_exhaustion_final_cell_suppress,
            chunk_state=chunk_state,
            phase_free_timing=effective_phase_decoupled_timing,
        )
        if phrase_selection is not None:
            planner.ref_phrase_index = phrase_selection.get("selected_ref_phrase_index")
            planner.ref_phrase_trace = phrase_selection.get("selected_ref_phrase_trace")
            planner.ref_phrase_stats = phrase_selection.get("selected_ref_phrase_stats")
        planner, commit_decision = self._enrich_planner_with_runtime_commit(
            planner=planner,
            ref_conditioning=ref_conditioning,
            dur_anchor_src=dur_anchor_src,
            unit_mask=unit_mask,
            state=state,
            source_boundary_cue=source_boundary_cue,
            boundary_confidence=boundary_confidence,
            sep_hint=sep_hint,
            open_run_mask=open_run_mask,
            sealed_mask=sealed_mask,
            force_full_commit=projector_force_full_commit,
            phrase_selection=phrase_selection,
        )
        execution = self.projector(
            dur_anchor_src=dur_anchor_src,
            unit_mask=unit_mask,
            speech_budget_win=planner.speech_budget_win,
            pause_budget_win=planner.pause_budget_win,
            dur_logratio_unit=planner.dur_logratio_unit,
            pause_weight_unit=planner.pause_weight_unit,
            boundary_score_unit=planner.boundary_score_unit,
            state=state,
            open_run_mask=open_run_mask,
            planner=planner,
            reuse_prefix=projector_reuse_prefix,
            force_full_commit=projector_force_full_commit,
            pause_topk_ratio_override=projector_pause_topk_ratio_override,
        )
        execution.next_state = replace(
            execution.next_state,
            trace_tail_reuse_count=self._next_trace_tail_reuse_count(
                state=execution.next_state,
                horizon=effective_trace_horizon,
            ),
            ref_phrase_ptr=(
                self._next_ref_phrase_ptr(
                    state=state,
                    ref_conditioning=ref_conditioning,
                    planner=execution.planner,
                    committed=commit_decision.committed,
                    phase_decoupled_timing=effective_phase_decoupled_timing,
                )
                if commit_decision is not None and getattr(execution.planner, "ref_phrase_index", None) is not None
                else state.ref_phrase_ptr
            ),
            active_phrase_start=(
                getattr(execution.planner, "active_phrase_start", None)
                if commit_decision is not None
                else state.active_phrase_start
            ),
            active_phrase_end=(
                getattr(execution.planner, "active_phrase_end", None)
                if commit_decision is not None
                else state.active_phrase_end
            ),
            local_rho_unit=getattr(execution.planner, "local_rho_unit", None),
            intra_phrase_alpha=getattr(execution.planner, "intra_phrase_alpha", None),
            commit_confidence=(
                getattr(execution.planner, "commit_confidence", None)
                if commit_decision is not None
                else state.commit_confidence
            ),
        )
        execution.trace_reliability = trace_reliability
        return execution

    def compute_algorithmic_teacher(
        self,
        *,
        content_units: torch.Tensor,
        dur_anchor_src: torch.Tensor,
        ref_conditioning: dict[str, torch.Tensor] | None = None,
        ref_rhythm_stats: torch.Tensor | None = None,
        ref_rhythm_trace: torch.Tensor | None = None,
        ref_mel: torch.Tensor | None = None,
        unit_mask: torch.Tensor | None = None,
        sep_hint: torch.Tensor | None = None,
        open_run_mask: torch.Tensor | None = None,
        sealed_mask: torch.Tensor | None = None,
        boundary_confidence: torch.Tensor | None = None,
        source_boundary_scale_override: float | None = None,
    ) -> RhythmTeacherTargets:
        del content_units
        ref_conditioning, unit_mask, source_boundary_cue = self._prepare_runtime_inputs(
            dur_anchor_src=dur_anchor_src,
            ref_conditioning=ref_conditioning,
            ref_rhythm_stats=ref_rhythm_stats,
            ref_rhythm_trace=ref_rhythm_trace,
            ref_mel=ref_mel,
            unit_mask=unit_mask,
            sep_hint=sep_hint,
            open_run_mask=open_run_mask,
            sealed_mask=sealed_mask,
            boundary_confidence=boundary_confidence,
            source_boundary_scale_override=source_boundary_scale_override,
        )
        return self.teacher(
            dur_anchor_src=dur_anchor_src,
            ref_rhythm_stats=ref_conditioning["ref_rhythm_stats"],
            ref_rhythm_trace=ref_conditioning["ref_rhythm_trace"],
            unit_mask=unit_mask,
            source_boundary_cue=source_boundary_cue,
        )

    def forward_dual(
        self,
        *,
        content_units: torch.Tensor,
        dur_anchor_src: torch.Tensor,
        ref_conditioning: dict[str, torch.Tensor] | None = None,
        ref_rhythm_stats: torch.Tensor | None = None,
        ref_rhythm_trace: torch.Tensor | None = None,
        ref_mel: torch.Tensor | None = None,
        unit_mask: torch.Tensor | None = None,
        open_run_mask: torch.Tensor | None = None,
        sealed_mask: torch.Tensor | None = None,
        sep_hint: torch.Tensor | None = None,
        boundary_confidence: torch.Tensor | None = None,
        state: StreamingRhythmState | None = None,
        offline_content_units: torch.Tensor | None = None,
        offline_dur_anchor_src: torch.Tensor | None = None,
        offline_unit_mask: torch.Tensor | None = None,
        offline_open_run_mask: torch.Tensor | None = None,
        offline_sealed_mask: torch.Tensor | None = None,
        offline_sep_hint: torch.Tensor | None = None,
        offline_boundary_confidence: torch.Tensor | None = None,
        trace_horizon: float | None = None,
        trace_active_tail_only: bool | None = None,
        trace_offset_lookahead_units: int | None = None,
        trace_cold_start_min_visible_units: int | None = None,
        trace_cold_start_full_visible_units: int | None = None,
        phase_decoupled_timing: bool | None = None,
        phase_free_timing: bool | None = None,
        projector_reuse_prefix: bool = True,
        projector_force_full_commit: bool = False,
        projector_pause_topk_ratio_override: float | None = None,
        source_boundary_scale_override: float | None = None,
        teacher_source_boundary_scale_override: float | None = None,
        teacher_projector_force_full_commit: bool = True,
        teacher_projector_soft_pause_selection_override: bool | None = None,
    ) -> dict[str, object]:
        if self.offline_teacher is None:
            raise RuntimeError(
                "forward_dual requires learned offline teacher runtime branch, but it is disabled. "
                "Enable `rhythm_enable_dual_mode_teacher` or `rhythm_runtime_enable_learned_offline_teacher`."
            )
        streaming_execution = self.forward(
            content_units=content_units,
            dur_anchor_src=dur_anchor_src,
            ref_conditioning=ref_conditioning,
            ref_rhythm_stats=ref_rhythm_stats,
            ref_rhythm_trace=ref_rhythm_trace,
            ref_mel=ref_mel,
            unit_mask=unit_mask,
            open_run_mask=open_run_mask,
            sealed_mask=sealed_mask,
            sep_hint=sep_hint,
            boundary_confidence=boundary_confidence,
            state=state,
            trace_horizon=trace_horizon,
            trace_active_tail_only=trace_active_tail_only,
            trace_offset_lookahead_units=trace_offset_lookahead_units,
            trace_cold_start_min_visible_units=trace_cold_start_min_visible_units,
            trace_cold_start_full_visible_units=trace_cold_start_full_visible_units,
            phase_decoupled_timing=phase_decoupled_timing,
            phase_free_timing=phase_free_timing,
            projector_reuse_prefix=projector_reuse_prefix,
            projector_force_full_commit=projector_force_full_commit,
            projector_pause_topk_ratio_override=projector_pause_topk_ratio_override,
            source_boundary_scale_override=source_boundary_scale_override,
        )
        offline_content_units = content_units if offline_content_units is None else offline_content_units
        offline_dur_anchor_src = dur_anchor_src if offline_dur_anchor_src is None else offline_dur_anchor_src
        offline_unit_mask = unit_mask if offline_unit_mask is None else offline_unit_mask
        offline_sep_hint = sep_hint if offline_sep_hint is None else offline_sep_hint
        offline_boundary_confidence = boundary_confidence if offline_boundary_confidence is None else offline_boundary_confidence
        if offline_open_run_mask is None:
            offline_open_run_mask = torch.zeros_like(offline_content_units)
        if offline_sealed_mask is None:
            if offline_unit_mask is None:
                offline_unit_mask = offline_dur_anchor_src.gt(0).float()
            offline_sealed_mask = torch.ones_like(offline_unit_mask).float()
        offline_execution, offline_confidence = self.forward_teacher(
            content_units=offline_content_units,
            dur_anchor_src=offline_dur_anchor_src,
            ref_conditioning=ref_conditioning,
            ref_rhythm_stats=ref_rhythm_stats,
            ref_rhythm_trace=ref_rhythm_trace,
            ref_mel=ref_mel,
            unit_mask=offline_unit_mask,
            open_run_mask=torch.zeros_like(offline_open_run_mask),
            sealed_mask=torch.ones_like(offline_sealed_mask).float(),
            sep_hint=offline_sep_hint,
            boundary_confidence=offline_boundary_confidence,
            projector_pause_topk_ratio_override=projector_pause_topk_ratio_override,
            source_boundary_scale_override=teacher_source_boundary_scale_override,
            projector_force_full_commit=teacher_projector_force_full_commit,
            projector_soft_pause_selection_override=teacher_projector_soft_pause_selection_override,
            trace_horizon=trace_horizon,
            trace_active_tail_only=trace_active_tail_only,
            trace_offset_lookahead_units=trace_offset_lookahead_units,
            trace_cold_start_min_visible_units=trace_cold_start_min_visible_units,
            trace_cold_start_full_visible_units=trace_cold_start_full_visible_units,
            phase_decoupled_timing=phase_decoupled_timing,
            phase_free_timing=phase_free_timing,
        )
        algorithmic_teacher = self.compute_algorithmic_teacher(
            content_units=offline_content_units,
            dur_anchor_src=offline_dur_anchor_src,
            ref_conditioning=ref_conditioning,
            ref_rhythm_stats=ref_rhythm_stats,
            ref_rhythm_trace=ref_rhythm_trace,
            ref_mel=ref_mel,
            unit_mask=offline_unit_mask,
            open_run_mask=torch.zeros_like(offline_open_run_mask),
            sealed_mask=torch.ones_like(offline_sealed_mask).float(),
            sep_hint=offline_sep_hint,
            boundary_confidence=offline_boundary_confidence,
            source_boundary_scale_override=teacher_source_boundary_scale_override,
        )
        return {
            "streaming_execution": streaming_execution,
            "offline_execution": offline_execution,
            "offline_confidence": offline_confidence,
            "algorithmic_teacher": algorithmic_teacher,
        }

    def forward_teacher(
        self,
        *,
        content_units: torch.Tensor,
        dur_anchor_src: torch.Tensor,
        ref_conditioning: dict[str, torch.Tensor] | None = None,
        ref_rhythm_stats: torch.Tensor | None = None,
        ref_rhythm_trace: torch.Tensor | None = None,
        ref_mel: torch.Tensor | None = None,
        unit_mask: torch.Tensor | None = None,
        open_run_mask: torch.Tensor | None = None,
        sealed_mask: torch.Tensor | None = None,
        sep_hint: torch.Tensor | None = None,
        boundary_confidence: torch.Tensor | None = None,
        projector_pause_topk_ratio_override: float | None = None,
        source_boundary_scale_override: float | None = None,
        projector_force_full_commit: bool = True,
        projector_soft_pause_selection_override: bool | None = None,
        trace_horizon: float | None = None,
        trace_active_tail_only: bool | None = None,
        trace_offset_lookahead_units: int | None = None,
        trace_cold_start_min_visible_units: int | None = None,
        trace_cold_start_full_visible_units: int | None = None,
        phase_free_timing: bool | None = None,
    ) -> tuple[object, dict[str, torch.Tensor]]:
        del phase_free_timing
        if self.offline_teacher is None:
            raise RuntimeError(
                "forward_teacher requires learned offline teacher runtime branch, but it is disabled. "
                "Enable `rhythm_runtime_enable_learned_offline_teacher` (or dual-mode teacher)."
            )
        ref_conditioning, unit_mask, source_boundary_cue = self._prepare_runtime_inputs(
            dur_anchor_src=dur_anchor_src,
            ref_conditioning=ref_conditioning,
            ref_rhythm_stats=ref_rhythm_stats,
            ref_rhythm_trace=ref_rhythm_trace,
            ref_mel=ref_mel,
            unit_mask=unit_mask,
            sep_hint=sep_hint,
            open_run_mask=open_run_mask,
            sealed_mask=sealed_mask,
            boundary_confidence=boundary_confidence,
            source_boundary_scale_override=source_boundary_scale_override,
        )
        unit_embed = self.unit_embedding(content_units.long().clamp_min(0))
        teacher_phase_ptr = unit_mask.new_zeros((unit_mask.size(0),))
        teacher_state = self.init_state(batch_size=content_units.size(0), device=content_units.device)
        teacher_phrase_selection = self._select_scheduler_phrase_bank(
            ref_conditioning=ref_conditioning,
            state=teacher_state,
            batch_size=content_units.size(0),
            device=content_units.device,
        )
        trace_context, planner_trace_context, trace_reliability = self._sample_trace_pair(
            ref_conditioning=ref_conditioning,
            phase_ptr=teacher_phase_ptr,
            window_size=content_units.size(1),
            unit_mask=unit_mask,
            dur_anchor_src=dur_anchor_src,
            horizon=(1.0 if trace_horizon is None else float(trace_horizon)),
            state=teacher_state,
            phrase_selection=teacher_phrase_selection,
            trace_active_tail_only=trace_active_tail_only,
            trace_offset_lookahead_units=trace_offset_lookahead_units,
            trace_cold_start_min_visible_units=trace_cold_start_min_visible_units,
            trace_cold_start_full_visible_units=trace_cold_start_full_visible_units,
        )
        planner, confidence = self.offline_teacher(
            unit_states=unit_embed,
            dur_anchor_src=dur_anchor_src,
            unit_mask=unit_mask,
            ref_conditioning=ref_conditioning,
            planner_trace_context=planner_trace_context,
            full_trace_context=trace_context,
            source_boundary_cue=source_boundary_cue,
        )
        planner, _ = self._enrich_planner_with_runtime_commit(
            planner=planner,
            ref_conditioning=ref_conditioning,
            dur_anchor_src=dur_anchor_src,
            unit_mask=unit_mask,
            state=teacher_state,
            source_boundary_cue=source_boundary_cue,
            boundary_confidence=boundary_confidence,
            sep_hint=sep_hint,
            open_run_mask=open_run_mask,
            sealed_mask=sealed_mask,
            force_full_commit=projector_force_full_commit,
            phrase_selection=teacher_phrase_selection,
        )
        execution = self.projector(
            dur_anchor_src=dur_anchor_src,
            unit_mask=unit_mask,
            speech_budget_win=planner.speech_budget_win,
            pause_budget_win=planner.pause_budget_win,
            dur_logratio_unit=planner.dur_logratio_unit,
            pause_weight_unit=planner.pause_weight_unit,
            boundary_score_unit=planner.boundary_score_unit,
            state=teacher_state,
            open_run_mask=torch.zeros_like(content_units) if open_run_mask is None else open_run_mask,
            planner=planner,
            reuse_prefix=False,
            force_full_commit=projector_force_full_commit,
            pause_topk_ratio_override=projector_pause_topk_ratio_override,
            soft_pause_selection_override=projector_soft_pause_selection_override,
        )
        execution.trace_reliability = trace_reliability
        return execution, confidence
