from __future__ import annotations

import torch


def _detach_loss_value(losses, key):
    value = losses.get(key)
    if isinstance(value, torch.Tensor):
        losses[key] = value.detach()
    return value


def _compact_base_optimizer_losses(losses, *, mel_loss_names, hparams, schedule_only_stage: bool):
    if schedule_only_stage or not bool(hparams.get("rhythm_compact_joint_loss", True)):
        return
    base_terms = []
    base_keys = []
    for loss_name in mel_loss_names:
        value = losses.get(loss_name)
        if isinstance(value, torch.Tensor):
            base_keys.append(loss_name)
            if value.requires_grad:
                base_terms.append(value)
    if not base_terms:
        return
    losses["base"] = sum(base_terms)
    for key in base_keys:
        _detach_loss_value(losses, key)


def _compact_pitch_optimizer_losses(losses, *, hparams, schedule_only_stage: bool):
    if schedule_only_stage or not bool(hparams.get("rhythm_compact_joint_loss", True)):
        return
    pitch_keys = []
    pitch_terms = []
    for key in ("fdiff", "uv", "pflow", "gdiff", "mdiff"):
        value = losses.get(key)
        if isinstance(value, torch.Tensor):
            pitch_keys.append(key)
            if value.requires_grad:
                pitch_terms.append(value)
    if not pitch_terms:
        return
    losses["pitch"] = sum(pitch_terms)
    for key in pitch_keys:
        _detach_loss_value(losses, key)


def _compact_rhythm_optimizer_losses(losses, *, hparams, schedule_only_stage: bool):
    keep_plan = bool(hparams.get("rhythm_enable_aux_optimizer_losses", False))
    keep_guidance = keep_plan or float(hparams.get("lambda_rhythm_guidance", 0.0) or 0.0) > 0.0
    keep_distill = (
        keep_plan
        or float(hparams.get("lambda_rhythm_distill", 0.0) or 0.0) > 0.0
        or float(hparams.get("lambda_rhythm_teacher_aux", 0.0) or 0.0) > 0.0
    )
    if not keep_plan:
        _detach_loss_value(losses, "rhythm_plan")
    if not keep_guidance:
        _detach_loss_value(losses, "rhythm_guidance")
    if not keep_distill:
        _detach_loss_value(losses, "rhythm_distill")
    if schedule_only_stage or not bool(hparams.get("rhythm_compact_joint_loss", True)):
        return
    exec_terms = []
    exec_keys = []
    for key in ("rhythm_exec_speech", "rhythm_exec_pause"):
        value = losses.get(key)
        if isinstance(value, torch.Tensor):
            exec_keys.append(key)
            if value.requires_grad:
                exec_terms.append(value)
    if exec_terms:
        losses["rhythm_exec"] = sum(exec_terms)
        for key in exec_keys:
            _detach_loss_value(losses, key)
    budget = losses.get("rhythm_budget")
    cumplan = losses.get("rhythm_cumplan", losses.get("rhythm_carry"))
    state_terms = []
    if isinstance(budget, torch.Tensor) and budget.requires_grad:
        state_terms.append(float(hparams.get("rhythm_joint_budget_macro_weight", 0.35)) * budget)
    if isinstance(cumplan, torch.Tensor) and cumplan.requires_grad:
        state_terms.append(float(hparams.get("rhythm_joint_cumplan_macro_weight", 0.65)) * cumplan)
    if state_terms:
        losses["rhythm_stream_state"] = sum(state_terms)
    _detach_loss_value(losses, "rhythm_budget")
    if "rhythm_cumplan" in losses:
        _detach_loss_value(losses, "rhythm_cumplan")
    if "rhythm_carry" in losses:
        _detach_loss_value(losses, "rhythm_carry")


def route_conan_optimizer_losses(losses, *, mel_loss_names, hparams, schedule_only_stage: bool):
    _compact_base_optimizer_losses(
        losses,
        mel_loss_names=mel_loss_names,
        hparams=hparams,
        schedule_only_stage=schedule_only_stage,
    )
    _compact_pitch_optimizer_losses(
        losses,
        hparams=hparams,
        schedule_only_stage=schedule_only_stage,
    )
    _compact_rhythm_optimizer_losses(
        losses,
        hparams=hparams,
        schedule_only_stage=schedule_only_stage,
    )


def update_public_loss_aliases(losses, *, mel_loss_names):
    device = None
    for value in losses.values():
        if isinstance(value, torch.Tensor):
            device = value.device
            break
    zero = torch.tensor(0.0, device=device or "cpu")
    if "rhythm_exec_speech" in losses:
        losses["L_exec_speech"] = losses["rhythm_exec_speech"].detach()
    if "rhythm_exec_pause" in losses:
        losses["L_exec_pause"] = losses["rhythm_exec_pause"].detach()
    losses["L_budget"] = losses.get("rhythm_budget", zero).detach() if isinstance(losses.get("rhythm_budget"), torch.Tensor) else zero
    cumplan = losses.get("rhythm_cumplan", losses.get("rhythm_carry"))
    losses["L_cumplan"] = cumplan.detach() if isinstance(cumplan, torch.Tensor) else zero
    losses["L_prefix_state"] = losses["L_cumplan"]
    losses["L_kd"] = losses.get("rhythm_distill", zero).detach() if isinstance(losses.get("rhythm_distill"), torch.Tensor) else zero
    rhythm_exec = losses.get("rhythm_exec")
    losses["L_rhythm_exec"] = rhythm_exec.detach() if isinstance(rhythm_exec, torch.Tensor) else zero
    stream_state = losses.get("rhythm_stream_state")
    losses["L_stream_state"] = stream_state.detach() if isinstance(stream_state, torch.Tensor) else zero
    pitch_value = losses.get("pitch")
    if not isinstance(pitch_value, torch.Tensor):
        pitch_value = zero
        for loss_name in ("fdiff", "uv", "pflow", "gdiff", "mdiff"):
            value = losses.get(loss_name)
            if isinstance(value, torch.Tensor):
                pitch_value = pitch_value + value.detach()
    else:
        pitch_value = pitch_value.detach()
    losses["L_pitch"] = pitch_value
    losses["L_distill_exec"] = losses.get("rhythm_distill_exec", zero).detach() if isinstance(losses.get("rhythm_distill_exec"), torch.Tensor) else zero
    losses["L_distill_budget"] = losses.get("rhythm_distill_budget", zero).detach() if isinstance(losses.get("rhythm_distill_budget"), torch.Tensor) else zero
    losses["L_distill_prefix"] = losses.get("rhythm_distill_prefix", zero).detach() if isinstance(losses.get("rhythm_distill_prefix"), torch.Tensor) else zero
    losses["L_distill_speech_shape"] = losses.get("rhythm_distill_speech_shape", zero).detach() if isinstance(losses.get("rhythm_distill_speech_shape"), torch.Tensor) else zero
    losses["L_distill_pause_shape"] = losses.get("rhythm_distill_pause_shape", zero).detach() if isinstance(losses.get("rhythm_distill_pause_shape"), torch.Tensor) else zero
    base_value = losses.get("base")
    if isinstance(base_value, torch.Tensor):
        base = base_value.detach()
    else:
        base = zero
        for loss_name in mel_loss_names:
            value = losses.get(loss_name)
            if isinstance(value, torch.Tensor):
                base = base + value.detach()
    losses["L_base"] = base
