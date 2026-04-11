from __future__ import annotations

from ..common.losses_impl import DurationV3LossTargets, _build_duration_v3_loss_dict


def build_duration_v3_loss_dict(execution, targets: DurationV3LossTargets):
    return _build_duration_v3_loss_dict(execution, targets)


def build_rhythm_loss_dict(execution, targets: DurationV3LossTargets):
    return _build_duration_v3_loss_dict(execution, targets)


__all__ = [
    "DurationV3LossTargets",
    "build_duration_v3_loss_dict",
    "build_rhythm_loss_dict",
]
