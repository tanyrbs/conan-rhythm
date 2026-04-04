from __future__ import annotations

import torch

from tasks.Conan.rhythm.targets import DistillConfidenceBundle


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
    hard_zero = confidence <= 0.0
    confidence = confidence.clamp(min=0.0, max=1.0)
    positive = confidence > 0.0
    confidence = torch.where(positive, confidence.clamp(min=float(floor), max=1.0), confidence)
    if abs(float(power) - 1.0) > 1e-8:
        confidence = torch.where(positive, confidence.pow(float(power)), confidence)
    if preserve_zeros:
        confidence = torch.where(hard_zero, torch.zeros_like(confidence), confidence)
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
    return DistillConfidenceBundle(
        shared=output.get("rhythm_offline_confidence"),
        exec=output.get("rhythm_offline_confidence_exec"),
        budget=output.get("rhythm_offline_confidence_budget"),
        prefix=output.get("rhythm_offline_confidence_prefix"),
        allocation=output.get("rhythm_offline_confidence_allocation"),
        shape=output.get("rhythm_offline_confidence_shape", output.get("rhythm_offline_confidence_exec")),
    )


__all__ = [
    "build_runtime_distill_confidence_bundle",
    "normalize_component_distill_confidence",
    "normalize_distill_confidence",
]
