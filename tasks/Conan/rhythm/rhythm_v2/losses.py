from __future__ import annotations

from ..common.losses_impl import RhythmLossTargets, build_rhythm_loss_dict


def build_rhythm_v2_loss_dict(execution, targets: RhythmLossTargets):
    return build_rhythm_loss_dict(execution, targets)


__all__ = [
    "RhythmLossTargets",
    "build_rhythm_loss_dict",
    "build_rhythm_v2_loss_dict",
]
