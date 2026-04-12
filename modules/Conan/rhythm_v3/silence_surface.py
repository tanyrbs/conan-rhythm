from __future__ import annotations

import math

import torch


def build_silence_tau_surface(
    *,
    prediction_anchor: torch.Tensor,
    committed_silence_mask: torch.Tensor,
    sep_hint: torch.Tensor | None,
    boundary_cue: torch.Tensor | None,
    max_silence_logstretch: float,
    short_gap_scale: float = 0.35,
    minimal_v1_profile: bool = False,
) -> torch.Tensor:
    silence_mask = committed_silence_mask.float().clamp(0.0, 1.0)
    if silence_mask.numel() <= 0:
        return silence_mask
    max_tau = float(max(1.0e-4, max_silence_logstretch))
    if bool(minimal_v1_profile):
        return torch.full_like(silence_mask, max_tau) * silence_mask

    log_anchor = torch.log(prediction_anchor.float().clamp_min(1.0e-4))
    pause_shape = torch.sigmoid(log_anchor - math.log(3.0))
    edge = (
        boundary_cue.float().clamp(0.0, 1.0)
        if isinstance(boundary_cue, torch.Tensor)
        else torch.zeros_like(pause_shape)
    )
    if isinstance(sep_hint, torch.Tensor):
        edge = torch.maximum(edge, sep_hint.float().clamp(0.0, 1.0))
    silence_shape = torch.maximum(pause_shape, edge)
    scale = float(max(0.0, min(1.0, short_gap_scale)))
    tau = max_tau * (scale + ((1.0 - scale) * silence_shape))
    return tau * silence_mask


__all__ = ["build_silence_tau_surface"]
