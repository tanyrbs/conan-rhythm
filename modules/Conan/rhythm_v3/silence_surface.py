from __future__ import annotations

import math

import torch


def resolve_silence_tau_surface_kind(*, minimal_v1_profile: bool) -> str:
    return "constant_clip" if bool(minimal_v1_profile) else "boundary_aware_clip"


def _build_silence_tau_surface_components(
    *,
    prediction_anchor: torch.Tensor,
    committed_silence_mask: torch.Tensor,
    sep_hint: torch.Tensor | None,
    boundary_cue: torch.Tensor | None,
    max_silence_logstretch: float,
    short_gap_scale: float = 0.35,
    minimal_v1_profile: bool = False,
) -> tuple[torch.Tensor, str, torch.Tensor]:
    silence_mask = committed_silence_mask.float().clamp(0.0, 1.0)
    surface_kind = resolve_silence_tau_surface_kind(minimal_v1_profile=minimal_v1_profile)
    if silence_mask.numel() <= 0:
        return silence_mask, surface_kind, silence_mask
    max_tau = float(max(1.0e-4, max_silence_logstretch))
    if bool(minimal_v1_profile):
        tau = torch.full_like(silence_mask, max_tau) * silence_mask
        return tau, surface_kind, torch.zeros_like(silence_mask)

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
    return tau * silence_mask, surface_kind, silence_shape * silence_mask


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
    tau, _, _ = _build_silence_tau_surface_components(
        prediction_anchor=prediction_anchor,
        committed_silence_mask=committed_silence_mask,
        sep_hint=sep_hint,
        boundary_cue=boundary_cue,
        max_silence_logstretch=max_silence_logstretch,
        short_gap_scale=short_gap_scale,
        minimal_v1_profile=minimal_v1_profile,
    )
    return tau


def build_silence_tau_surface_meta(
    *,
    prediction_anchor: torch.Tensor,
    committed_silence_mask: torch.Tensor,
    sep_hint: torch.Tensor | None,
    boundary_cue: torch.Tensor | None,
    max_silence_logstretch: float,
    short_gap_scale: float = 0.35,
    minimal_v1_profile: bool = False,
) -> dict[str, torch.Tensor | str]:
    tau, surface_kind, boundary_shaping = _build_silence_tau_surface_components(
        prediction_anchor=prediction_anchor,
        committed_silence_mask=committed_silence_mask,
        sep_hint=sep_hint,
        boundary_cue=boundary_cue,
        max_silence_logstretch=max_silence_logstretch,
        short_gap_scale=short_gap_scale,
        minimal_v1_profile=minimal_v1_profile,
    )
    return {
        "silence_tau": tau,
        "silence_surface_kind": surface_kind,
        "silence_boundary_shaping": boundary_shaping,
    }


__all__ = [
    "build_silence_tau_surface",
    "build_silence_tau_surface_meta",
    "resolve_silence_tau_surface_kind",
]
