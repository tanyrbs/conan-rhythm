from __future__ import annotations

from dataclasses import dataclass

import torch

from modules.Conan.rhythm.bridge import resolve_rhythm_apply_mode
from modules.Conan.rhythm.policy import is_duration_operator_mode, resolve_apply_override
from modules.Conan.rhythm.stages import detect_rhythm_stage, resolve_teacher_as_main

from ..confidence_utils import clamp_confidence_preserve_zero
from .task_config import resolve_task_retimed_target_mode, resolve_task_target_mode


@dataclass(frozen=True)
class TaskRuntimeState:
    effective_global_step: int
    stage: str
    teacher_as_main: bool
    use_reference: bool
    rhythm_apply_override: str | bool | None
    apply_rhythm_render: bool
    retimed_stage_active: bool
    disable_source_pitch_supervision: bool
    disable_acoustic_train_path: bool
    module_only_objective: bool


def resolve_task_apply_override(
    hparams,
    *,
    global_step: int,
    infer: bool,
    test: bool,
    explicit=None,
):
    return resolve_apply_override(
        hparams,
        infer=infer,
        test=test,
        explicit=explicit,
        current_step=int(global_step),
    )


def resolve_task_runtime_state(
    hparams,
    *,
    global_step: int,
    infer: bool,
    test: bool,
    explicit_apply_override=None,
    has_f0: bool,
    has_uv: bool,
) -> TaskRuntimeState:
    rhythm_enable_v2 = bool(hparams.get("rhythm_enable_v2", False))
    rhythm_enable_v3 = bool(
        hparams.get("rhythm_enable_v3", False)
        or is_duration_operator_mode(hparams.get("rhythm_mode", ""))
    )
    rhythm_enabled = bool(rhythm_enable_v2 or rhythm_enable_v3)
    effective_global_step = 200000 if test else int(global_step)
    use_reference = (
        test
        or effective_global_step >= int(hparams["random_speaker_steps"])
        or bool(hparams.get("rhythm_force_reference_conditioning", False))
    )
    rhythm_apply_override = resolve_task_apply_override(
        hparams,
        global_step=effective_global_step,
        infer=infer,
        test=test,
        explicit=explicit_apply_override,
    )
    apply_rhythm_render = resolve_rhythm_apply_mode(
        hparams,
        infer=infer,
        override=rhythm_apply_override,
    )
    retimed_stage_active = bool(
        apply_rhythm_render
        and not infer
        and not test
        and bool(hparams.get("rhythm_use_retimed_target_if_available", False))
        and effective_global_step >= int(hparams.get("rhythm_retimed_target_start_steps", 0) or 0)
    )
    use_retimed_pitch_target = bool(hparams.get("rhythm_use_retimed_pitch_target", False))
    disable_source_pitch_supervision = bool(
        retimed_stage_active
        and (
            not use_retimed_pitch_target
            or not has_f0
            or not has_uv
            or bool(hparams.get("rhythm_disable_pitch_loss_when_retimed", False))
        )
    )
    module_only_objective = bool(
        not test
        and not apply_rhythm_render
        and rhythm_enabled
        and bool(hparams.get("rhythm_optimize_module_only", False))
    )
    disable_acoustic_train_path = bool(
        not infer
        and module_only_objective
        and bool(hparams.get("rhythm_fastpath_disable_acoustic_when_module_only", True))
    )
    disable_source_pitch_supervision = bool(
        disable_source_pitch_supervision or disable_acoustic_train_path
    )
    if rhythm_enable_v3 and not rhythm_enable_v2:
        return TaskRuntimeState(
            effective_global_step=effective_global_step,
            stage="duration_v3",
            teacher_as_main=False,
            use_reference=use_reference,
            rhythm_apply_override=rhythm_apply_override,
            apply_rhythm_render=bool(apply_rhythm_render),
            retimed_stage_active=retimed_stage_active,
            disable_source_pitch_supervision=disable_source_pitch_supervision,
            disable_acoustic_train_path=disable_acoustic_train_path,
            module_only_objective=module_only_objective,
        )
    stage = detect_rhythm_stage(hparams)
    teacher_as_main = resolve_teacher_as_main(hparams, stage=stage, infer=bool(infer))
    return TaskRuntimeState(
        effective_global_step=effective_global_step,
        stage=stage,
        teacher_as_main=teacher_as_main,
        use_reference=use_reference,
        rhythm_apply_override=rhythm_apply_override,
        apply_rhythm_render=bool(apply_rhythm_render),
        retimed_stage_active=retimed_stage_active,
        disable_source_pitch_supervision=disable_source_pitch_supervision,
        disable_acoustic_train_path=disable_acoustic_train_path,
        module_only_objective=module_only_objective,
    )


def merge_retimed_weight(frame_weight, confidence, *, confidence_floor: float = 0.05):
    if frame_weight is None and confidence is None:
        return None
    if confidence is not None:
        confidence = clamp_confidence_preserve_zero(
            confidence.float(),
            floor=float(confidence_floor),
        )
    if frame_weight is None:
        return confidence
    frame_weight = frame_weight.float()
    if confidence is None:
        return frame_weight
    while confidence.dim() < frame_weight.dim():
        confidence = confidence.unsqueeze(-1)
    return frame_weight * confidence


def _resolve_online_retimed_sample_confidence(sample, model_out):
    reference = model_out.get("rhythm_online_retimed_frame_weight") if model_out is not None else None
    if not isinstance(reference, torch.Tensor) and model_out is not None:
        reference = model_out.get("rhythm_online_retimed_mel_tgt")
    reference_device = reference.device if isinstance(reference, torch.Tensor) else None
    for key in (
        "rhythm_retimed_target_confidence",
        "rhythm_teacher_confidence",
        "rhythm_target_confidence",
    ):
        for source in (model_out, sample):
            if source is None:
                continue
            value = source.get(key)
            if value is None:
                continue
            if torch.is_tensor(value):
                value = value.float()
                if reference_device is not None:
                    value = value.to(device=reference_device)
                return value
            return torch.as_tensor(value, dtype=torch.float32, device=reference_device)
    return None


def resolve_acoustic_target_post_model(
    sample,
    model_out,
    *,
    hparams,
    global_step: int,
    apply_rhythm_render: bool,
    infer: bool,
    test: bool,
    current_step=None,
):
    target = sample["mels"]
    frame_weight = None
    is_retimed = False
    source = "source"
    effective_step = int(global_step if current_step is None else current_step)
    start_step = int(hparams.get("rhythm_retimed_target_start_steps", 0) or 0)
    if (
        not bool(apply_rhythm_render)
        or not bool(hparams.get("rhythm_use_retimed_target_if_available", False))
        or effective_step < start_step
    ):
        return target, is_retimed, frame_weight, source

    stage = "test" if test else ("valid" if infer else "train")
    target_mode = resolve_task_retimed_target_mode(hparams)
    online_start = int(hparams.get("rhythm_online_retimed_target_start_steps", start_step) or start_step)
    online_ready = effective_step >= online_start
    prefer_online = target_mode in {"online", "hybrid"} and online_ready

    if prefer_online:
        online_target = model_out.get("rhythm_online_retimed_mel_tgt")
        if online_target is not None:
            online_weight = merge_retimed_weight(
                model_out.get("rhythm_online_retimed_frame_weight"),
                _resolve_online_retimed_sample_confidence(sample, model_out),
                confidence_floor=float(hparams.get("rhythm_retimed_confidence_floor", 0.05)),
            )
            return (
                online_target,
                True,
                online_weight,
                "online",
            )
        if target_mode == "online":
            raise RuntimeError(
                f"Rhythm online retimed target is required for the active render path ({stage}) but is unavailable."
            )

    cached_target = sample.get("rhythm_retimed_mel_tgt")
    if cached_target is not None:
        return (
            cached_target,
            True,
            merge_retimed_weight(
                sample.get("rhythm_retimed_frame_weight"),
                sample.get("rhythm_retimed_target_confidence"),
                confidence_floor=float(hparams.get("rhythm_retimed_confidence_floor", 0.05)),
            ),
            "cached",
        )

    require_retimed = bool(hparams.get("rhythm_require_retimed_cache", False))
    if require_retimed or (not test and resolve_task_target_mode(hparams) == "cached_only"):
        raise RuntimeError(
            "Rhythm retimed target is required for the active render path "
            f"({stage}) but neither online nor cached retimed targets are available."
        )
    return target, is_retimed, frame_weight, source


__all__ = [
    "TaskRuntimeState",
    "merge_retimed_weight",
    "resolve_acoustic_target_post_model",
    "resolve_task_apply_override",
    "resolve_task_runtime_state",
]
