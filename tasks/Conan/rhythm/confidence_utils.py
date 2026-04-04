from __future__ import annotations

import torch


def clamp_confidence_preserve_zero(
    confidence: torch.Tensor,
    *,
    floor: float,
    preserve_zeros: bool = True,
) -> torch.Tensor:
    """Clamp confidence into [0, 1] with optional explicit-zero preservation."""
    confidence = confidence.float().clamp(min=0.0, max=1.0)
    floor = min(max(float(floor), 0.0), 1.0)
    if floor <= 0.0:
        return confidence
    if not preserve_zeros:
        return confidence.clamp(min=floor, max=1.0)
    positive = confidence > 0.0
    floored = confidence.clamp(min=floor, max=1.0)
    return torch.where(positive, floored, confidence)


__all__ = ["clamp_confidence_preserve_zero"]
