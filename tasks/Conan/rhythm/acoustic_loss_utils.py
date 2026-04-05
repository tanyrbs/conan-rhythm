from __future__ import annotations

import torch

from utils.nn.seq_utils import weights_nonzero_speech


def expand_frame_weight(weight: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor:
    """Expand frame-level acoustic weights to a broadcast-safe tensor."""
    if weight is None:
        return weights_nonzero_speech(target)
    weight = weight.float()
    if weight.dim() == 1:
        weight = weight.unsqueeze(0)
    while weight.dim() < target.dim():
        weight = weight.unsqueeze(-1)
    return weight


def reduce_weighted_elementwise_loss(
    loss: torch.Tensor,
    *,
    frame_weight: torch.Tensor | None,
    target: torch.Tensor,
) -> torch.Tensor:
    """Reduce an elementwise acoustic loss with full broadcasted weight mass.

    The correctness point is that frame-level weights like ``[B, T]`` are
    applied to elementwise losses like ``[B, T, C]``. The denominator must use
    the full broadcasted mass, otherwise the reduced loss is inflated by
    roughly the mel/channel dimension.
    """
    weights = expand_frame_weight(frame_weight, target)
    if weights.shape != loss.shape:
        weights = torch.broadcast_to(weights, loss.shape)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


__all__ = [
    "expand_frame_weight",
    "reduce_weighted_elementwise_loss",
]
