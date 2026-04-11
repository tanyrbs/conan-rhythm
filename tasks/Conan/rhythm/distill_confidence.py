from __future__ import annotations

import torch

from tasks.Conan.rhythm.confidence_utils import clamp_confidence_preserve_zero
from tasks.Conan.rhythm.rhythm_v2.targets import DistillConfidenceBundle

_DISTILL_CONFIDENCE_OUTPUT_KEYS = {
    "shared": "rhythm_offline_confidence",
    "exec": "rhythm_offline_confidence_exec",
    "budget": "rhythm_offline_confidence_budget",
    "prefix": "rhythm_offline_confidence_prefix",
    "allocation": "rhythm_offline_confidence_allocation",
    "shape": "rhythm_offline_confidence_shape",
}


def _as_confidence_tensor(
    confidence,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(confidence, torch.Tensor):
        return confidence.detach().float().reshape(batch_size, -1)[:, :1].to(device=device)
    return torch.as_tensor(confidence, dtype=torch.float32, device=device).reshape(batch_size, -1)[:, :1]


def normalize_distill_confidence(
    distill_confidence,
    *,
    batch_size: int,
    device: torch.device,
    floor: float,
    power: float,
    preserve_zeros: bool = False,
) -> torch.Tensor:
    if distill_confidence is None:
        confidence = torch.ones((batch_size, 1), device=device)
    else:
        confidence = _as_confidence_tensor(
            distill_confidence,
            batch_size=batch_size,
            device=device,
        )
    confidence = clamp_confidence_preserve_zero(
        confidence,
        floor=float(floor),
        preserve_zeros=preserve_zeros,
    )
    if abs(float(power) - 1.0) > 1e-8:
        positive = confidence > 0.0 if preserve_zeros else None
        if preserve_zeros:
            confidence = torch.where(positive, confidence.pow(float(power)), confidence)
        else:
            confidence = confidence.pow(float(power))
    return confidence


def normalize_component_distill_confidence(
    component_confidence,
    *,
    fallback_confidence: torch.Tensor,
    batch_size: int,
    device: torch.device,
    floor: float,
    power: float,
    preserve_zeros: bool = False,
) -> torch.Tensor:
    if component_confidence is None:
        return fallback_confidence
    return normalize_distill_confidence(
        component_confidence,
        batch_size=batch_size,
        device=device,
        floor=floor,
        power=power,
        preserve_zeros=preserve_zeros,
    )


def build_runtime_distill_confidence_bundle(output) -> DistillConfidenceBundle:
    values = {
        name: output.get(key)
        for name, key in _DISTILL_CONFIDENCE_OUTPUT_KEYS.items()
    }
    if values["shape"] is None:
        values["shape"] = values["exec"]
    return DistillConfidenceBundle(**values)


__all__ = [
    "build_runtime_distill_confidence_bundle",
    "normalize_component_distill_confidence",
    "normalize_distill_confidence",
]
