from __future__ import annotations

import torch


def build_runtime_teacher_aux_loss_dict(
    *,
    teacher_losses: dict[str, torch.Tensor],
    hparams,
    prefix_state_lambda: float,
    lambda_teacher_aux: float,
) -> dict[str, torch.Tensor]:
    if float(lambda_teacher_aux) <= 0.0:
        return {}
    teacher_exec = (
        teacher_losses["rhythm_exec_speech"] * hparams.get("lambda_rhythm_exec_speech", 1.0)
        + teacher_losses["rhythm_exec_pause"] * hparams.get("lambda_rhythm_exec_pause", 1.0)
    )
    teacher_state = (
        teacher_losses["rhythm_budget"] * hparams.get("lambda_rhythm_budget", 0.25)
        + teacher_losses["rhythm_prefix_state"] * float(prefix_state_lambda)
    )
    teacher_aux_raw = teacher_exec + teacher_state
    teacher_aux_loss = float(lambda_teacher_aux) * teacher_aux_raw
    return {
        "rhythm_teacher_aux_loss": teacher_aux_loss,
        "rhythm_teacher_aux_exec": teacher_exec.detach(),
        "rhythm_teacher_aux_state": teacher_state.detach(),
        "rhythm_teacher_aux": teacher_aux_loss.detach(),
    }


__all__ = ["build_runtime_teacher_aux_loss_dict"]
