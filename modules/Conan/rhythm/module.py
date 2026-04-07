from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn

from .contracts import RhythmTeacherTargets, StreamingRhythmState, TraceReliabilityBundle
from .offline_teacher import OfflineRhythmTeacherPlanner, OfflineTeacherConfig
from .projector import ProjectorConfig, StreamingRhythmProjector
from .reference_descriptor import RefRhythmDescriptor
from .reference_encoder import REF_RHYTHM_STATS_KEYS, REF_RHYTHM_TRACE_KEYS
from .scheduler import MonotonicRhythmScheduler
from .source_boundary import build_source_boundary_cue
from .teacher import AlgorithmicRhythmTeacher, AlgorithmicTeacherConfig


class StreamingRhythmModule(nn.Module):
    """Minimal strong-rhythm mainline.

    Public contract stays small:
    - content_units
    - dur_anchor_src
    - ref_rhythm_stats / ref_rhythm_trace (or ref_mel)

    Internal execution stays layered but explicit:
    - reference descriptor
    - monotonic scheduler
    - single projector authority
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
        projector_config: ProjectorConfig | None = None,
        teacher_config: AlgorithmicTeacherConfig | None = None,
        offline_teacher_config: OfflineTeacherConfig | None = None,
        enable_learned_offline_teacher: bool = False,
    ) -> None:
        super().__init__()
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

    def _resolve_phase_gap(
        self,
        *,
        phase_ptr: torch.Tensor,
        dur_anchor_src: torch.Tensor | None,
        unit_mask: torch.Tensor,
        state: StreamingRhythmState | None,
    ) -> torch.Tensor | None:
        if state is None:
            return None
        if state.phase_anchor_progress is not None and dur_anchor_src is not None:
            visible_total = (dur_anchor_src.float().clamp_min(0.0) * unit_mask.float()).sum(dim=1).clamp_min(1.0)
            current_progress_ratio = state.phase_anchor_progress.float() / visible_total
            return phase_ptr.float() - current_progress_ratio
        return state.phase_ptr_gap

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

    def _build_trace_reliability(
        self,
        *,
        phase_ptr: torch.Tensor,
        phase_gap: torch.Tensor | None,
        horizon: float,
        tail_reuse_count: torch.Tensor,
    ) -> TraceReliabilityBundle:
        phase_ptr = phase_ptr.float()
        ones = torch.ones_like(phase_ptr)
        zeros = torch.zeros_like(phase_ptr)
        tail_start = max(0.0, 1.0 - float(horizon))
        tail_alpha = ((phase_ptr - tail_start) / max(float(horizon), 1.0e-6)).clamp(0.0, 1.0)
        if phase_gap is None:
            phase_gap = zeros
        phase_gap = phase_gap.float().clamp_min(0.0)
        denom = max(self.trace_exhaustion_gap_end - self.trace_exhaustion_gap_start, 1.0e-6)
        gap_alpha = ((phase_gap - self.trace_exhaustion_gap_start) / denom).clamp(0.0, 1.0)
        reuse_alpha = (
            tail_reuse_count.float() / float(max(self.trace_exhaustion_reuse_full_count, 1))
        ).clamp(0.0, 1.0)
        if self.trace_reliability_enable:
            trace_reliability = 1.0 - tail_alpha * gap_alpha * reuse_alpha
            local_floor = phase_ptr.new_full(trace_reliability.shape, self.trace_exhaustion_local_floor)
            boundary_floor = phase_ptr.new_full(trace_reliability.shape, self.trace_exhaustion_boundary_floor)
            local_trace_path_weight = torch.maximum(trace_reliability, local_floor)
            boundary_trace_path_weight = torch.maximum(trace_reliability.square(), boundary_floor)
        else:
            trace_reliability = ones
            local_trace_path_weight = ones
            boundary_trace_path_weight = ones
        return TraceReliabilityBundle(
            trace_reliability=trace_reliability,
            local_trace_path_weight=local_trace_path_weight,
            boundary_trace_path_weight=boundary_trace_path_weight,
            phase_gap=phase_gap,
            tail_alpha=tail_alpha,
            gap_alpha=gap_alpha,
            reuse_alpha=reuse_alpha,
            tail_reuse_count=tail_reuse_count.float(),
            tail_active=tail_alpha.gt(0.0).float(),
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
    ) -> tuple[torch.Tensor, torch.Tensor, TraceReliabilityBundle]:
        visible_sizes = self._visible_sizes(unit_mask)
        effective_horizon = self.reference_descriptor.encoder.trace_horizon if horizon is None else float(horizon)
        anchor_durations = None
        if self.trace_anchor_aware_sampling and dur_anchor_src is not None:
            anchor_durations = dur_anchor_src.float() * unit_mask.float()
        trace_context = self.sample_trace_window(
            ref_conditioning=ref_conditioning,
            phase_ptr=phase_ptr,
            window_size=window_size,
            horizon=effective_horizon,
            visible_sizes=visible_sizes,
            anchor_durations=anchor_durations,
        )
        planner_trace_context = self.sample_planner_trace_window(
            ref_conditioning=ref_conditioning,
            phase_ptr=phase_ptr,
            window_size=window_size,
            horizon=effective_horizon,
            visible_sizes=visible_sizes,
            anchor_durations=anchor_durations,
        )
        phase_gap = self._resolve_phase_gap(
            phase_ptr=phase_ptr,
            dur_anchor_src=dur_anchor_src,
            unit_mask=unit_mask,
            state=state,
        )
        trace_reliability = self._build_trace_reliability(
            phase_ptr=phase_ptr,
            phase_gap=phase_gap,
            horizon=effective_horizon,
            tail_reuse_count=self._resolve_tail_reuse_count(state, phase_ptr=phase_ptr),
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
            )
            enriched.update({k: v for k, v in ref_conditioning.items() if v is not None})
            return self._finalize_reference_conditioning(enriched)
        if ref_rhythm_stats is not None and ref_rhythm_trace is not None:
            return self._finalize_reference_conditioning(
                self.reference_descriptor.from_stats_trace(
                    ref_rhythm_stats,
                    ref_rhythm_trace,
                    selector=self.reference_descriptor.selector,
                    include_sidecar=self.reference_descriptor.emit_reference_sidecar,
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
    ) -> torch.Tensor:
        return self.reference_descriptor.sample_trace_window(
            ref_conditioning,
            phase_ptr=phase_ptr,
            window_size=window_size,
            horizon=horizon,
            visible_sizes=visible_sizes,
            anchor_durations=anchor_durations,
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
    ) -> torch.Tensor:
        return self.reference_descriptor.sample_planner_trace_window(
            ref_conditioning,
            phase_ptr=phase_ptr,
            window_size=window_size,
            horizon=horizon,
            visible_sizes=visible_sizes,
            anchor_durations=anchor_durations,
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
        unit_embed = self.unit_embedding(content_units.long().clamp_min(0))
        trace_context, planner_trace_context, trace_reliability = self._sample_trace_pair(
            ref_conditioning=ref_conditioning,
            phase_ptr=state.phase_ptr,
            window_size=content_units.size(1),
            unit_mask=unit_mask,
            dur_anchor_src=dur_anchor_src,
            horizon=effective_trace_horizon,
            state=state,
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
            trace_reliability=trace_reliability,
            trace_exhaustion_final_cell_suppress=self.trace_exhaustion_final_cell_suppress,
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
        )
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
    ) -> tuple[object, dict[str, torch.Tensor]]:
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
        trace_context, planner_trace_context, _ = self._sample_trace_pair(
            ref_conditioning=ref_conditioning,
            phase_ptr=teacher_phase_ptr,
            window_size=content_units.size(1),
            unit_mask=unit_mask,
            dur_anchor_src=dur_anchor_src,
            horizon=1.0,
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
        execution = self.projector(
            dur_anchor_src=dur_anchor_src,
            unit_mask=unit_mask,
            speech_budget_win=planner.speech_budget_win,
            pause_budget_win=planner.pause_budget_win,
            dur_logratio_unit=planner.dur_logratio_unit,
            pause_weight_unit=planner.pause_weight_unit,
            boundary_score_unit=planner.boundary_score_unit,
            state=self.init_state(batch_size=content_units.size(0), device=content_units.device),
            open_run_mask=torch.zeros_like(content_units) if open_run_mask is None else open_run_mask,
            planner=planner,
            reuse_prefix=False,
            force_full_commit=projector_force_full_commit,
            pause_topk_ratio_override=projector_pause_topk_ratio_override,
            soft_pause_selection_override=projector_soft_pause_selection_override,
        )
        return execution, confidence
