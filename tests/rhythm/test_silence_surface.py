from __future__ import annotations

import torch

from modules.Conan.rhythm_v3.silence_surface import (
    build_silence_tau_surface,
    build_silence_tau_surface_meta,
)


def test_build_silence_tau_surface_keeps_minimal_profile_constant():
    kwargs = {
        "prediction_anchor": torch.tensor([[2.0, 8.0]], dtype=torch.float32),
        "committed_silence_mask": torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        "sep_hint": torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        "boundary_cue": torch.tensor([[0.0, 0.5]], dtype=torch.float32),
        "max_silence_logstretch": 0.25,
        "short_gap_scale": 0.35,
        "minimal_v1_profile": True,
    }
    tau = build_silence_tau_surface(**kwargs)
    meta = build_silence_tau_surface_meta(**kwargs)
    assert meta["silence_surface_kind"] == "constant_clip"
    assert torch.allclose(meta["silence_tau"], tau)
    assert torch.allclose(meta["silence_boundary_shaping"], torch.zeros_like(tau))
    assert torch.allclose(tau, torch.full_like(tau, 0.25))


def test_build_silence_tau_surface_is_boundary_aware_outside_minimal_profile():
    kwargs = {
        "prediction_anchor": torch.tensor([[1.0, 8.0]], dtype=torch.float32),
        "committed_silence_mask": torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        "sep_hint": torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        "boundary_cue": torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        "max_silence_logstretch": 0.30,
        "short_gap_scale": 0.25,
        "minimal_v1_profile": False,
    }
    tau = build_silence_tau_surface(**kwargs)
    meta = build_silence_tau_surface_meta(**kwargs)
    assert meta["silence_surface_kind"] == "boundary_aware_clip"
    assert torch.allclose(meta["silence_tau"], tau)
    assert float(meta["silence_boundary_shaping"][0, 1]) > float(meta["silence_boundary_shaping"][0, 0])
    assert float(tau[0, 1]) > float(tau[0, 0])
    assert float(tau[0, 0]) >= 0.30 * 0.25
