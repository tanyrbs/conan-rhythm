from __future__ import annotations

import torch
import torch.nn as nn

from .contracts import RhythmTeacherTargets, StreamingRhythmState
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
        min_speech_frames: float = 1.0,
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

    def _sample_trace_pair(
        self,
        *,
        ref_conditioning: dict[str, torch.Tensor],
        phase_ptr: torch.Tensor,
        window_size: int,
        unit_mask: torch.Tensor,
        horizon: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visible_sizes = self._visible_sizes(unit_mask)
        trace_context = self.sample_trace_window(
            ref_conditioning=ref_conditioning,
            phase_ptr=phase_ptr,
            window_size=window_size,
            horizon=horizon,
            visible_sizes=visible_sizes,
        )
        planner_trace_context = self.sample_planner_trace_window(
            ref_conditioning=ref_conditioning,
            phase_ptr=phase_ptr,
            window_size=window_size,
            horizon=horizon,
            visible_sizes=visible_sizes,
        )
        return trace_context, planner_trace_context

    def encode_reference(self, ref_mel: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.reference_descriptor(ref_mel)

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
            if "slow_rhythm_summary" not in enriched and "slow_rhythm_memory" in enriched:
                slow_memory = enriched["slow_rhythm_memory"]
                if slow_memory.dim() == 3:
                    enriched["slow_rhythm_summary"] = slow_memory.mean(dim=1)
            if "planner_slow_rhythm_summary" not in enriched and "planner_slow_rhythm_memory" in enriched:
                planner_slow_memory = enriched["planner_slow_rhythm_memory"]
                if planner_slow_memory.dim() == 3:
                    enriched["planner_slow_rhythm_summary"] = planner_slow_memory.mean(dim=1)
            elif "planner_slow_rhythm_summary" not in enriched and "planner_ref_trace" in enriched:
                enriched["planner_slow_rhythm_summary"] = enriched["planner_ref_trace"].mean(dim=1)
            return enriched
        if ref_rhythm_stats is not None and ref_rhythm_trace is not None:
            return self.reference_descriptor.from_stats_trace(
                ref_rhythm_stats,
                ref_rhythm_trace,
                selector=self.reference_descriptor.selector,
                include_sidecar=self.reference_descriptor.emit_reference_sidecar,
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
    ) -> torch.Tensor:
        return self.reference_descriptor.sample_trace_window(
            ref_conditioning,
            phase_ptr=phase_ptr,
            window_size=window_size,
            horizon=horizon,
            visible_sizes=visible_sizes,
        )

    def sample_planner_trace_window(
        self,
        *,
        ref_conditioning: dict[str, torch.Tensor],
        phase_ptr: torch.Tensor,
        window_size: int,
        horizon: float | None = None,
        visible_sizes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.reference_descriptor.sample_planner_trace_window(
            ref_conditioning,
            phase_ptr=phase_ptr,
            window_size=window_size,
            horizon=horizon,
            visible_sizes=visible_sizes,
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

        unit_embed = self.unit_embedding(content_units.long().clamp_min(0))
        trace_context, planner_trace_context = self._sample_trace_pair(
            ref_conditioning=ref_conditioning,
            phase_ptr=state.phase_ptr,
            window_size=content_units.size(1),
            unit_mask=unit_mask,
            horizon=trace_horizon,
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
        trace_context, planner_trace_context = self._sample_trace_pair(
            ref_conditioning=ref_conditioning,
            phase_ptr=teacher_phase_ptr,
            window_size=content_units.size(1),
            unit_mask=unit_mask,
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
            force_full_commit=True,
            pause_topk_ratio_override=projector_pause_topk_ratio_override,
        )
        return execution, confidence
