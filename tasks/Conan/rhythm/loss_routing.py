from __future__ import annotations

import torch

_PITCH_LOSS_KEYS = ("fdiff", "uv", "pflow", "gdiff", "mdiff")
_STYLE_LOSS_KEYS = ("gloss", "vq_loss")
_PREFIX_STATE_KEYS = ("rhythm_prefix_state", "rhythm_cumplan", "rhythm_carry")
_AUX_REPORTING_KEYS = {
    "rhythm_plan": "lambda_rhythm_plan",
    "rhythm_guidance": "lambda_rhythm_guidance",
    "rhythm_distill": "lambda_rhythm_distill",
    "rhythm_teacher_aux_loss": "lambda_rhythm_teacher_aux",
    "rhythm_descriptor_consistency": "lambda_rhythm_descriptor_consistency",
    "rhythm_pairwise_contrastive": "lambda_rhythm_pairwise_contrastive",
    "rhythm_pairwise_diversity": "lambda_rhythm_pairwise_diversity",
}


def _detach_loss_value(losses, key):
    value = losses.get(key)
    if isinstance(value, torch.Tensor):
        losses[key] = value.detach()
    return value


def _zero_reporting_loss(losses):
    for value in losses.values():
        if isinstance(value, torch.Tensor):
            return value.new_zeros(())
    return torch.tensor(0.0)


def _append_reporting_key(keys, seen, losses, key):
    if key in seen:
        return
    value = losses.get(key)
    if isinstance(value, torch.Tensor):
        keys.append(key)
        seen.add(key)


def _resolve_reporting_prefix_state_key(losses):
    for key in _PREFIX_STATE_KEYS:
        if isinstance(losses.get(key), torch.Tensor):
            return key
    return None


def _resolve_prefix_state_value(losses):
    key = _resolve_reporting_prefix_state_key(losses)
    return losses.get(key) if key is not None else None


def _resolve_aux_optimizer_policy(hparams) -> dict[str, bool]:
    hparams = hparams or {}
    keep_aux = bool(hparams.get("rhythm_enable_aux_optimizer_losses", False))
    return {
        loss_key: keep_aux or float(hparams.get(lambda_key, 0.0) or 0.0) > 0.0
        for loss_key, lambda_key in _AUX_REPORTING_KEYS.items()
    }


def _collect_reporting_total_keys(
    losses,
    *,
    mel_loss_names=(),
    hparams=None,
    schedule_only_stage: bool = False,
):
    hparams = hparams or {}
    keys = []
    seen = set()
    compact_joint = not schedule_only_stage and bool(hparams.get("rhythm_compact_joint_loss", True))
    aux_policy = _resolve_aux_optimizer_policy(hparams)

    if compact_joint and isinstance(losses.get("base"), torch.Tensor):
        _append_reporting_key(keys, seen, losses, "base")
    else:
        for loss_name in mel_loss_names:
            _append_reporting_key(keys, seen, losses, loss_name)

    if compact_joint and isinstance(losses.get("pitch"), torch.Tensor):
        _append_reporting_key(keys, seen, losses, "pitch")
    else:
        for key in _PITCH_LOSS_KEYS:
            _append_reporting_key(keys, seen, losses, key)

    for key in _STYLE_LOSS_KEYS:
        _append_reporting_key(keys, seen, losses, key)

    if compact_joint and isinstance(losses.get("rhythm_exec"), torch.Tensor):
        _append_reporting_key(keys, seen, losses, "rhythm_exec")
    else:
        _append_reporting_key(keys, seen, losses, "rhythm_exec_speech")
        _append_reporting_key(keys, seen, losses, "rhythm_exec_pause")

    if compact_joint and isinstance(losses.get("rhythm_stream_state"), torch.Tensor):
        _append_reporting_key(keys, seen, losses, "rhythm_stream_state")
    else:
        _append_reporting_key(keys, seen, losses, "rhythm_budget")
        prefix_state_key = _resolve_reporting_prefix_state_key(losses)
        if prefix_state_key is not None:
            _append_reporting_key(keys, seen, losses, prefix_state_key)

    for loss_key, enabled in aux_policy.items():
        if not enabled:
            continue
        if loss_key == "rhythm_teacher_aux_loss":
            teacher_aux_key = (
                "rhythm_teacher_aux_loss"
                if isinstance(losses.get("rhythm_teacher_aux_loss"), torch.Tensor)
                else "rhythm_teacher_aux"
            )
            _append_reporting_key(keys, seen, losses, teacher_aux_key)
            continue
        _append_reporting_key(keys, seen, losses, loss_key)

    return keys


def compute_reporting_total_loss(
    losses,
    *,
    mel_loss_names=(),
    hparams=None,
    schedule_only_stage: bool = False,
):
    """Match the optimizer objective while excluding public aliases and diagnostics.

    Validation runs under ``torch.no_grad()``, so a pure ``requires_grad`` filter
    would collapse to zero there. We therefore prefer the real trainable terms
    when gradients are enabled, then fall back to the routed canonical objective
    keys when gradients are disabled.
    """
    trainable_terms = [
        value
        for value in losses.values()
        if isinstance(value, torch.Tensor) and value.requires_grad
    ]
    if trainable_terms:
        total = trainable_terms[0]
        for term in trainable_terms[1:]:
            total = total + term
        return total

    reporting_keys = _collect_reporting_total_keys(
        losses,
        mel_loss_names=mel_loss_names,
        hparams=hparams,
        schedule_only_stage=schedule_only_stage,
    )
    if not reporting_keys:
        return _zero_reporting_loss(losses)
    total = losses[reporting_keys[0]]
    for key in reporting_keys[1:]:
        total = total + losses[key]
    return total


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
    for key in _PITCH_LOSS_KEYS:
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
    aux_policy = _resolve_aux_optimizer_policy(hparams)
    for loss_key, enabled in aux_policy.items():
        if not enabled:
            _detach_loss_value(losses, loss_key)
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
    prefix_state = _resolve_prefix_state_value(losses)
    state_terms = []
    if isinstance(budget, torch.Tensor) and budget.requires_grad:
        state_terms.append(float(hparams.get("rhythm_joint_budget_macro_weight", 0.35)) * budget)
    if isinstance(prefix_state, torch.Tensor) and prefix_state.requires_grad:
        state_terms.append(float(hparams.get("rhythm_joint_cumplan_macro_weight", 0.65)) * prefix_state)
    if state_terms:
        losses["rhythm_stream_state"] = sum(state_terms)
    _detach_loss_value(losses, "rhythm_budget")
    for key in _PREFIX_STATE_KEYS:
        if key in losses:
            _detach_loss_value(losses, key)


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
    prefix_state = _resolve_prefix_state_value(losses)
    losses["L_cumplan"] = prefix_state.detach() if isinstance(prefix_state, torch.Tensor) else zero
    losses["L_prefix_state"] = losses["L_cumplan"]
    losses["L_plan"] = losses.get("rhythm_plan", zero).detach() if isinstance(losses.get("rhythm_plan"), torch.Tensor) else zero
    losses["L_plan_local"] = (
        losses.get("rhythm_plan_local", zero).detach()
        if isinstance(losses.get("rhythm_plan_local"), torch.Tensor)
        else zero
    )
    losses["L_plan_cum"] = (
        losses.get("rhythm_plan_cum", zero).detach()
        if isinstance(losses.get("rhythm_plan_cum"), torch.Tensor)
        else zero
    )
    losses["L_guidance"] = (
        losses.get("rhythm_guidance", zero).detach()
        if isinstance(losses.get("rhythm_guidance"), torch.Tensor)
        else zero
    )
    losses["L_descriptor_consistency"] = (
        losses.get("rhythm_descriptor_consistency", zero).detach()
        if isinstance(losses.get("rhythm_descriptor_consistency"), torch.Tensor)
        else zero
    )
    losses["L_descriptor_global"] = (
        losses.get("rhythm_descriptor_global", zero).detach()
        if isinstance(losses.get("rhythm_descriptor_global"), torch.Tensor)
        else zero
    )
    losses["L_descriptor_pause"] = (
        losses.get("rhythm_descriptor_pause", zero).detach()
        if isinstance(losses.get("rhythm_descriptor_pause"), torch.Tensor)
        else zero
    )
    losses["L_descriptor_local_trace"] = (
        losses.get("rhythm_descriptor_local_trace", zero).detach()
        if isinstance(losses.get("rhythm_descriptor_local_trace"), torch.Tensor)
        else zero
    )
    losses["L_descriptor_boundary_trace"] = (
        losses.get("rhythm_descriptor_boundary_trace", zero).detach()
        if isinstance(losses.get("rhythm_descriptor_boundary_trace"), torch.Tensor)
        else zero
    )
    losses["L_pairwise_contrastive"] = (
        losses.get("rhythm_pairwise_contrastive", zero).detach()
        if isinstance(losses.get("rhythm_pairwise_contrastive"), torch.Tensor)
        else zero
    )
    losses["L_pairwise_diversity"] = (
        losses.get("rhythm_pairwise_diversity", zero).detach()
        if isinstance(losses.get("rhythm_pairwise_diversity"), torch.Tensor)
        else zero
    )
    losses["L_prefix_clock"] = (
        losses.get("rhythm_prefix_clock", zero).detach()
        if isinstance(losses.get("rhythm_prefix_clock"), torch.Tensor)
        else zero
    )
    losses["L_prefix_backlog"] = (
        losses.get("rhythm_prefix_backlog", zero).detach()
        if isinstance(losses.get("rhythm_prefix_backlog"), torch.Tensor)
        else zero
    )
    losses["L_kd"] = losses.get("rhythm_distill", zero).detach() if isinstance(losses.get("rhythm_distill"), torch.Tensor) else zero
    kd_student = losses.get("rhythm_distill_student", losses.get("rhythm_distill"))
    losses["L_kd_student"] = kd_student.detach() if isinstance(kd_student, torch.Tensor) else zero
    teacher_aux = losses.get("rhythm_teacher_aux")
    if not isinstance(teacher_aux, torch.Tensor):
        teacher_aux = losses.get("rhythm_teacher_aux_loss", zero)
    losses["L_teacher_aux"] = teacher_aux.detach() if isinstance(teacher_aux, torch.Tensor) else zero
    rhythm_exec = losses.get("rhythm_exec")
    if not isinstance(rhythm_exec, torch.Tensor):
        rhythm_exec = zero
        for key in ("rhythm_exec_speech", "rhythm_exec_pause"):
            value = losses.get(key)
            if isinstance(value, torch.Tensor):
                rhythm_exec = rhythm_exec + value.detach()
    else:
        rhythm_exec = rhythm_exec.detach()
    losses["L_rhythm_exec"] = rhythm_exec
    stream_state = losses.get("rhythm_stream_state")
    if not isinstance(stream_state, torch.Tensor):
        stream_state = zero
        budget_value = losses.get("rhythm_budget")
        if isinstance(budget_value, torch.Tensor):
            stream_state = stream_state + budget_value.detach()
        if isinstance(prefix_state, torch.Tensor):
            stream_state = stream_state + prefix_state.detach()
    else:
        stream_state = stream_state.detach()
    losses["L_stream_state"] = stream_state
    pitch_value = losses.get("pitch")
    if not isinstance(pitch_value, torch.Tensor):
        pitch_value = zero
        for loss_name in _PITCH_LOSS_KEYS:
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
    losses["L_kd_same_source"] = (
        losses.get("rhythm_distill_same_source_any", zero).detach()
        if isinstance(losses.get("rhythm_distill_same_source_any"), torch.Tensor)
        else zero
    )
    losses["L_kd_same_source_exec"] = (
        losses.get("rhythm_distill_same_source_exec", zero).detach()
        if isinstance(losses.get("rhythm_distill_same_source_exec"), torch.Tensor)
        else zero
    )
    losses["L_kd_same_source_budget"] = (
        losses.get("rhythm_distill_same_source_budget", zero).detach()
        if isinstance(losses.get("rhythm_distill_same_source_budget"), torch.Tensor)
        else zero
    )
    losses["L_kd_same_source_prefix"] = (
        losses.get("rhythm_distill_same_source_prefix", zero).detach()
        if isinstance(losses.get("rhythm_distill_same_source_prefix"), torch.Tensor)
        else zero
    )
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
