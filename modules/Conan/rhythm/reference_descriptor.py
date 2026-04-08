from __future__ import annotations

import torch
import torch.nn as nn

from .reference_encoder import (
    REF_RHYTHM_STATS_KEYS,
    REF_RHYTHM_TRACE_KEYS,
    ReferenceRhythmEncoder,
    build_reference_phrase_bank,
)
from .reference_selector import ReferenceSelector


PLANNER_REF_STATS_KEYS = (
    "global_rate",
    "pause_ratio",
)

PLANNER_REF_TRACE_KEYS = (
    "local_rate_trace",
    "boundary_trace",
)


def mean_speech_frames_to_global_rate(mean_speech_frames: torch.Tensor) -> torch.Tensor:
    mean_speech_frames = mean_speech_frames.float()
    safe_mean_speech = mean_speech_frames.clamp_min(1.0)
    has_speech_evidence = mean_speech_frames > 0.0
    return torch.where(
        has_speech_evidence,
        torch.reciprocal(safe_mean_speech),
        torch.zeros_like(mean_speech_frames),
    )


def global_rate_to_mean_speech_frames(global_rate: torch.Tensor) -> torch.Tensor:
    global_rate = global_rate.float()
    safe_global_rate = global_rate.clamp_min(1.0e-6)
    has_speech_evidence = global_rate > 0.0
    return torch.where(
        has_speech_evidence,
        torch.reciprocal(safe_global_rate),
        torch.zeros_like(global_rate),
    )


class RefRhythmDescriptor(nn.Module):
    """Reference descriptor with a compact planner-facing contract.

    Cache compatibility is preserved through:
      - ref_rhythm_stats [B, 6]
      - ref_rhythm_trace [B, bins, 5]

    The maintained planner consumes only:
      - planner_ref_stats = [global_rate, pause_ratio]
      - planner_ref_trace = [local_rate_trace, boundary_trace]
    """

    def __init__(
        self,
        *,
        trace_bins: int = 24,
        trace_horizon: float = 0.35,
        smooth_kernel: int = 5,
        slow_topk: int = 6,
        selector_cell_size: int = 3,
        maintained_stats_trace_only: bool = True,
        emit_reference_sidecar: bool | None = None,
        runtime_phrase_bank_enable: bool = False,
        runtime_phrase_bank_max_phrases: int | None = None,
        runtime_phrase_bank_bins: int | None = None,
        runtime_phrase_select_window: int = 3,
        runtime_phrase_neighbor_mix_alpha: float = 0.15,
        phrase_selection_boundary_weight: float = 0.28,
        phrase_selection_local_rate_weight: float = 0.28,
        phrase_selection_pause_weight: float = 0.18,
        phrase_selection_voice_weight: float = 0.16,
        phrase_selection_final_bias_weight: float = 0.10,
        phrase_selection_monotonic_bias: float = 0.0,
        phrase_selection_length_bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.maintained_stats_trace_only = bool(maintained_stats_trace_only)
        # Maintained path defaults to the smallest stable contract: stats + trace.
        # Sidecars remain available, but are opt-in unless explicitly overridden.
        self.emit_reference_sidecar = (
            (not self.maintained_stats_trace_only)
            if emit_reference_sidecar is None
            else bool(emit_reference_sidecar)
        )
        self.runtime_phrase_bank_enable = bool(runtime_phrase_bank_enable)
        self.runtime_phrase_bank_max_phrases = max(
            1,
            int(slow_topk if runtime_phrase_bank_max_phrases is None else runtime_phrase_bank_max_phrases),
        )
        self.runtime_phrase_bank_bins = max(
            4,
            int(max(4, selector_cell_size * 2 + 2) if runtime_phrase_bank_bins is None else runtime_phrase_bank_bins),
        )
        self.runtime_phrase_select_window = max(1, int(runtime_phrase_select_window))
        self.runtime_phrase_neighbor_mix_alpha = float(
            max(0.0, min(0.49, runtime_phrase_neighbor_mix_alpha))
        )
        self.encoder = ReferenceRhythmEncoder(
            trace_bins=trace_bins,
            trace_horizon=trace_horizon,
            smooth_kernel=smooth_kernel,
        )
        self.selector = (
            ReferenceSelector(
                slow_topk=slow_topk,
                cell_size=selector_cell_size,
                phrase_select_window=self.runtime_phrase_select_window,
                boundary_weight=phrase_selection_boundary_weight,
                local_rate_weight=phrase_selection_local_rate_weight,
                pause_weight=phrase_selection_pause_weight,
                voiced_weight=phrase_selection_voice_weight,
                final_bias_weight=phrase_selection_final_bias_weight,
                monotonic_bias=phrase_selection_monotonic_bias,
                phrase_length_bias=phrase_selection_length_bias,
            )
            if self.emit_reference_sidecar
            else None
        )

    @staticmethod
    def _validate_cached_contract(ref_rhythm_stats: torch.Tensor, ref_rhythm_trace: torch.Tensor) -> None:
        if ref_rhythm_stats.dim() != 2:
            raise ValueError(
                f"ref_rhythm_stats must be [B, {len(REF_RHYTHM_STATS_KEYS)}], got {tuple(ref_rhythm_stats.shape)}"
            )
        if ref_rhythm_trace.dim() != 3:
            raise ValueError(
                f"ref_rhythm_trace must be [B, bins, {len(REF_RHYTHM_TRACE_KEYS)}], got {tuple(ref_rhythm_trace.shape)}"
            )
        if ref_rhythm_stats.size(-1) != len(REF_RHYTHM_STATS_KEYS):
            raise ValueError(
                f"ref_rhythm_stats dim mismatch: found={ref_rhythm_stats.size(-1)}, "
                f"expected={len(REF_RHYTHM_STATS_KEYS)}"
            )
        if ref_rhythm_trace.size(-1) != len(REF_RHYTHM_TRACE_KEYS):
            raise ValueError(
                f"ref_rhythm_trace dim mismatch: found={ref_rhythm_trace.size(-1)}, "
                f"expected={len(REF_RHYTHM_TRACE_KEYS)}"
            )

    @staticmethod
    def _phase_to_trace_start(phase_ptr: torch.Tensor, horizon: float) -> torch.Tensor:
        if phase_ptr.dim() == 2 and phase_ptr.size(-1) == 1:
            phase_ptr = phase_ptr.squeeze(-1)
        horizon = float(max(0.01, min(1.0, horizon)))
        max_start = max(0.0, 1.0 - horizon)
        phase_ptr = phase_ptr.float().clamp(0.0, 1.0)
        if max_start <= 0.0:
            return torch.zeros_like(phase_ptr)
        return torch.minimum(phase_ptr, phase_ptr.new_full(phase_ptr.shape, max_start))

    @staticmethod
    def _compact_slow_memory(slow_memory: torch.Tensor) -> torch.Tensor:
        if slow_memory.dim() != 3:
            return slow_memory
        if slow_memory.size(-1) >= 3:
            return torch.cat([slow_memory[:, :, 1:2], slow_memory[:, :, 2:3]], dim=-1)
        if slow_memory.size(-1) == 2:
            return slow_memory
        raise ValueError(f"Unexpected slow rhythm memory shape: {tuple(slow_memory.shape)}")

    @staticmethod
    def _weighted_summary(memory: torch.Tensor, score_weight: torch.Tensor) -> torch.Tensor:
        if memory.dim() == 2:
            return memory
        weight = score_weight.clamp_min(0.0).unsqueeze(-1)
        return (memory * (1.0 + weight)).sum(dim=1) / (1.0 + weight).sum(dim=1).clamp_min(1e-6)

    @staticmethod
    def from_stats_trace(
        ref_rhythm_stats: torch.Tensor,
        ref_rhythm_trace: torch.Tensor,
        selector: ReferenceSelector | None = None,
        *,
        include_sidecar: bool = False,
        runtime_phrase_bank_enable: bool = False,
        runtime_phrase_bank_max_phrases: int = 32,
        runtime_phrase_bank_bins: int = 8,
    ) -> dict[str, torch.Tensor]:
        RefRhythmDescriptor._validate_cached_contract(ref_rhythm_stats, ref_rhythm_trace)
        global_rate = mean_speech_frames_to_global_rate(ref_rhythm_stats[:, 2:3])
        pause_ratio = ref_rhythm_stats[:, 0:1].clamp(0.0, 1.0)
        local_rate_trace = ref_rhythm_trace[:, :, 1:2]
        boundary_trace = ref_rhythm_trace[:, :, 2:3]
        planner_ref_stats = torch.cat([global_rate, pause_ratio], dim=-1)
        planner_ref_trace = torch.cat([local_rate_trace, boundary_trace], dim=-1)
        out = {
            "ref_rhythm_stats": ref_rhythm_stats,
            "ref_rhythm_trace": ref_rhythm_trace,
            "global_rate": global_rate,
            "pause_ratio": pause_ratio,
            "local_rate_trace": local_rate_trace,
            "boundary_trace": boundary_trace,
            "planner_ref_stats": planner_ref_stats,
            "planner_ref_trace": planner_ref_trace,
        }
        if include_sidecar or runtime_phrase_bank_enable:
            out.update(
                build_reference_phrase_bank(
                    ref_rhythm_trace=ref_rhythm_trace,
                    max_phrases=runtime_phrase_bank_max_phrases,
                    phrase_trace_bins=runtime_phrase_bank_bins,
                )
            )
        if not include_sidecar:
            return out
        if selector is None:
            slow_memory = ref_rhythm_trace
            slow_indices = torch.arange(ref_rhythm_trace.size(1), device=ref_rhythm_trace.device)[None, :].expand(
                ref_rhythm_trace.size(0), -1
            )
            slow_scores = boundary_trace.squeeze(-1)
            slow_starts = slow_indices
            slow_ends = slow_indices
        else:
            selection = selector(ref_rhythm_trace)
            slow_memory = selection.slow_rhythm_memory
            slow_indices = selection.slow_rhythm_indices
            slow_scores = selection.slow_rhythm_scores
            slow_starts = selection.slow_rhythm_starts
            slow_ends = selection.slow_rhythm_ends
        slow_summary = RefRhythmDescriptor._weighted_summary(slow_memory, slow_scores)
        planner_slow_memory = RefRhythmDescriptor._compact_slow_memory(slow_memory)
        planner_slow_summary = RefRhythmDescriptor._weighted_summary(planner_slow_memory, slow_scores)
        out.update(
            {
                "slow_rhythm_memory": slow_memory,
                "slow_rhythm_summary": slow_summary,
                "planner_slow_rhythm_memory": planner_slow_memory,
                "planner_slow_rhythm_summary": planner_slow_summary,
                "selector_meta_indices": slow_indices,
                "selector_meta_scores": slow_scores,
                "selector_meta_starts": slow_starts,
                "selector_meta_ends": slow_ends,
            }
        )
        return out

    def forward(self, ref_mel: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encoder(ref_mel)
        return self.from_stats_trace(
            encoded["ref_rhythm_stats"],
            encoded["ref_rhythm_trace"],
            selector=self.selector,
            include_sidecar=self.emit_reference_sidecar,
            runtime_phrase_bank_enable=self.runtime_phrase_bank_enable,
            runtime_phrase_bank_max_phrases=self.runtime_phrase_bank_max_phrases,
            runtime_phrase_bank_bins=self.runtime_phrase_bank_bins,
        )

    def select_phrase_bank(
        self,
        ref_conditioning: dict[str, torch.Tensor],
        *,
        ref_phrase_ptr: torch.Tensor,
        query_chunk_summary: torch.Tensor | None = None,
        query_commit_confidence: torch.Tensor | None = None,
        query_phrase_close_prob: torch.Tensor | None = None,
        strict_pointer_only: bool = False,
    ) -> dict[str, torch.Tensor]:
        if "ref_phrase_trace" not in ref_conditioning:
            ref_conditioning = {
                **ref_conditioning,
                **build_reference_phrase_bank(
                    ref_rhythm_trace=ref_conditioning["ref_rhythm_trace"],
                    max_phrases=self.runtime_phrase_bank_max_phrases,
                    phrase_trace_bins=self.runtime_phrase_bank_bins,
                ),
            }
        selector = self.selector
        if selector is not None:
            return selector.select_monotonic_phrase_bank(
                ref_phrase_trace=ref_conditioning["ref_phrase_trace"],
                planner_ref_phrase_trace=ref_conditioning["planner_ref_phrase_trace"],
                ref_phrase_valid=ref_conditioning["ref_phrase_valid"],
                ref_phrase_lengths=ref_conditioning["ref_phrase_lengths"],
                ref_phrase_starts=ref_conditioning["ref_phrase_starts"],
                ref_phrase_ends=ref_conditioning["ref_phrase_ends"],
                ref_phrase_boundary_strength=ref_conditioning["ref_phrase_boundary_strength"],
                ref_phrase_stats=ref_conditioning.get("ref_phrase_stats"),
                ref_phrase_ptr=ref_phrase_ptr,
                query_chunk_summary=query_chunk_summary,
                query_commit_confidence=query_commit_confidence,
                query_phrase_close_prob=query_phrase_close_prob,
                monotonic_window=self.runtime_phrase_select_window,
                strict_pointer_only=strict_pointer_only,
                neighbor_mix_alpha=self.runtime_phrase_neighbor_mix_alpha,
            )
        return ReferenceSelector.select_monotonic_phrase(
            ref_phrase_trace=ref_conditioning["ref_phrase_trace"],
            planner_ref_phrase_trace=ref_conditioning["planner_ref_phrase_trace"],
            ref_phrase_valid=ref_conditioning["ref_phrase_valid"],
            ref_phrase_lengths=ref_conditioning["ref_phrase_lengths"],
            ref_phrase_starts=ref_conditioning["ref_phrase_starts"],
            ref_phrase_ends=ref_conditioning["ref_phrase_ends"],
            ref_phrase_boundary_strength=ref_conditioning["ref_phrase_boundary_strength"],
            ref_phrase_stats=ref_conditioning.get("ref_phrase_stats"),
            ref_phrase_ptr=ref_phrase_ptr,
            query_chunk_summary=query_chunk_summary,
            query_commit_confidence=query_commit_confidence,
            query_phrase_close_prob=query_phrase_close_prob,
            monotonic_window=self.runtime_phrase_select_window,
            strict_pointer_only=strict_pointer_only,
            neighbor_mix_alpha=self.runtime_phrase_neighbor_mix_alpha,
        )

    def sample_trace_window(
        self,
        ref_conditioning: dict[str, torch.Tensor],
        phase_ptr: torch.Tensor,
        window_size: int,
        *,
        horizon: float | None = None,
        visible_sizes: torch.Tensor | None = None,
        anchor_durations: torch.Tensor | None = None,
        commit_frontier: torch.Tensor | None = None,
        lookahead_units: int | None = None,
        active_tail_only: bool = False,
    ) -> torch.Tensor:
        effective_horizon = self.encoder.trace_horizon if horizon is None else horizon
        return self.encoder.sample_trace_window(
            ref_conditioning["ref_rhythm_trace"],
            phase_ptr=self._phase_to_trace_start(phase_ptr, effective_horizon),
            window_size=window_size,
            horizon=effective_horizon,
            visible_sizes=visible_sizes,
            anchor_durations=anchor_durations,
            commit_frontier=commit_frontier,
            lookahead_units=lookahead_units,
            active_tail_only=active_tail_only,
        )

    def sample_planner_trace_window(
        self,
        ref_conditioning: dict[str, torch.Tensor],
        phase_ptr: torch.Tensor,
        window_size: int,
        *,
        horizon: float | None = None,
        visible_sizes: torch.Tensor | None = None,
        anchor_durations: torch.Tensor | None = None,
        commit_frontier: torch.Tensor | None = None,
        lookahead_units: int | None = None,
        active_tail_only: bool = False,
    ) -> torch.Tensor:
        effective_horizon = self.encoder.trace_horizon if horizon is None else horizon
        return self.encoder.sample_trace_window(
            ref_conditioning["planner_ref_trace"],
            phase_ptr=self._phase_to_trace_start(phase_ptr, effective_horizon),
            window_size=window_size,
            horizon=effective_horizon,
            visible_sizes=visible_sizes,
            anchor_durations=anchor_durations,
            commit_frontier=commit_frontier,
            lookahead_units=lookahead_units,
            active_tail_only=active_tail_only,
        )
