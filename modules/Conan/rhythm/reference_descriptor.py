from __future__ import annotations

import torch
import torch.nn as nn

from .reference_encoder import ReferenceRhythmEncoder


class RefRhythmDescriptor(nn.Module):
    """Minimal explicit rhythm descriptor.

    Exposes a smaller, more interpretable surface on top of the underlying
    reference rhythm encoder while keeping the cached `ref_rhythm_stats` and
    `ref_rhythm_trace` contract for compatibility.
    """

    def __init__(
        self,
        *,
        trace_bins: int = 24,
        trace_horizon: float = 0.35,
        smooth_kernel: int = 5,
    ) -> None:
        super().__init__()
        self.encoder = ReferenceRhythmEncoder(
            trace_bins=trace_bins,
            trace_horizon=trace_horizon,
            smooth_kernel=smooth_kernel,
        )

    @staticmethod
    def from_stats_trace(
        ref_rhythm_stats: torch.Tensor,
        ref_rhythm_trace: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        global_rate = torch.reciprocal(ref_rhythm_stats[:, 2:3].clamp_min(1.0))
        pause_ratio = ref_rhythm_stats[:, 0:1].clamp(0.0, 1.0)
        local_rate_trace = ref_rhythm_trace[:, :, 1:2]
        boundary_trace = ref_rhythm_trace[:, :, 2:3]
        return {
            "ref_rhythm_stats": ref_rhythm_stats,
            "ref_rhythm_trace": ref_rhythm_trace,
            "global_rate": global_rate,
            "pause_ratio": pause_ratio,
            "local_rate_trace": local_rate_trace,
            "boundary_trace": boundary_trace,
        }

    def forward(self, ref_mel: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encoder(ref_mel)
        return self.from_stats_trace(
            encoded["ref_rhythm_stats"],
            encoded["ref_rhythm_trace"],
        )

    def sample_trace_window(
        self,
        ref_conditioning: dict[str, torch.Tensor],
        phase_ptr: torch.Tensor,
        window_size: int,
        *,
        horizon: float | None = None,
    ) -> torch.Tensor:
        return self.encoder.sample_trace_window(
            ref_conditioning["ref_rhythm_trace"],
            phase_ptr=phase_ptr,
            window_size=window_size,
            horizon=horizon,
        )
