from __future__ import annotations

import torch
import torch.nn as nn

from .reference_encoder import (
    REF_RHYTHM_STATS_KEYS,
    REF_RHYTHM_TRACE_KEYS,
    ReferenceRhythmEncoder,
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
        self.encoder = ReferenceRhythmEncoder(
            trace_bins=trace_bins,
            trace_horizon=trace_horizon,
            smooth_kernel=smooth_kernel,
        )
        self.selector = (
            ReferenceSelector(
                slow_topk=slow_topk,
                cell_size=selector_cell_size,
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
    ) -> dict[str, torch.Tensor]:
        RefRhythmDescriptor._validate_cached_contract(ref_rhythm_stats, ref_rhythm_trace)
        global_rate = torch.reciprocal(ref_rhythm_stats[:, 2:3].clamp_min(1.0))
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
        )

    def sample_trace_window(
        self,
        ref_conditioning: dict[str, torch.Tensor],
        phase_ptr: torch.Tensor,
        window_size: int,
        *,
        horizon: float | None = None,
        visible_sizes: torch.Tensor | None = None,
        dur_anchor_src: torch.Tensor | None = None,
        unit_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        effective_horizon = self.encoder.trace_horizon if horizon is None else horizon
        return self.encoder.sample_trace_window(
            ref_conditioning["ref_rhythm_trace"],
            phase_ptr=self._phase_to_trace_start(phase_ptr, effective_horizon),
            window_size=window_size,
            horizon=effective_horizon,
            visible_sizes=visible_sizes,
            dur_anchor_src=dur_anchor_src,
            unit_mask=unit_mask,
        )

    def sample_planner_trace_window(
        self,
        ref_conditioning: dict[str, torch.Tensor],
        phase_ptr: torch.Tensor,
        window_size: int,
        *,
        horizon: float | None = None,
        visible_sizes: torch.Tensor | None = None,
        dur_anchor_src: torch.Tensor | None = None,
        unit_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        effective_horizon = self.encoder.trace_horizon if horizon is None else horizon
        return self.encoder.sample_trace_window(
            ref_conditioning["planner_ref_trace"],
            phase_ptr=self._phase_to_trace_start(phase_ptr, effective_horizon),
            window_size=window_size,
            horizon=effective_horizon,
            visible_sizes=visible_sizes,
            dur_anchor_src=dur_anchor_src,
            unit_mask=unit_mask,
        )
