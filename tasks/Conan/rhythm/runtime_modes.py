from __future__ import annotations

from dataclasses import dataclass

import torch

from modules.Conan.rhythm.bridge import resolve_rhythm_apply_mode
from modules.Conan.rhythm.policy import resolve_apply_override
from modules.Conan.rhythm.stages import detect_rhythm_stage, resolve_teacher_as_main
from tasks.Conan.rhythm.confidence_utils import clamp_confidence_preserve_zero
from tasks.Conan.rhythm.task_config import (
    resolve_task_retimed_target_mode,
    resolve_task_target_mode,
)


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
    disable_acoustic_train_path = bool(
        not infer
        and not test
        and not apply_rhythm_render
        and bool(hparams.get("rhythm_enable_v2", False))
        and bool(hparams.get("rhythm_optimize_module_only", False))
        and bool(hparams.get("rhythm_fastpath_disable_acoustic_when_module_only", True))
    )
    disable_source_pitch_supervision = bool(
        disable_source_pitch_supervision or disable_acoustic_train_path
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
    )


def build_rhythm_ref_conditioning(sample, *, explicit=None):
    if explicit is not None:
        return explicit
    ref_stats = sample.get("ref_rhythm_stats")
    ref_trace = sample.get("ref_rhythm_trace")
    if ref_stats is None or ref_trace is None:
        return None
    conditioning = {
        "ref_rhythm_stats": ref_stats,
        "ref_rhythm_trace": ref_trace,
    }
    for extra_key in (
        "global_rate",
        "pause_ratio",
        "local_rate_trace",
        "boundary_trace",
        "planner_ref_stats",
        "planner_ref_trace",
        "slow_rhythm_memory",
        "slow_rhythm_summary",
        "planner_slow_rhythm_memory",
        "planner_slow_rhythm_summary",
        "selector_meta_indices",
        "selector_meta_scores",
        "selector_meta_starts",
        "selector_meta_ends",
    ):
        extra_value = sample.get(extra_key)
        if extra_value is not None:
            conditioning[extra_key] = extra_value
    return conditioning


def collect_planner_runtime_outputs(rhythm_execution) -> dict[str, torch.Tensor]:
    runtime_outputs = {}
    if rhythm_execution is None or getattr(rhythm_execution, "planner", None) is None:
        return runtime_outputs
    planner = rhythm_execution.planner
    for attr_name in (
        "raw_speech_budget_win",
        "raw_pause_budget_win",
        "effective_speech_budget_win",
        "effective_pause_budget_win",
    ):
        attr_value = getattr(planner, attr_name, None)
        if attr_value is not None:
            runtime_outputs[attr_name] = attr_value
    for attr_name in (
        "feasible_speech_budget_delta",
        "feasible_pause_budget_delta",
        "feasible_total_budget_delta",
    ):
        attr_value = getattr(planner, attr_name, None)
        if attr_value is not None:
            runtime_outputs[attr_name] = attr_value
    return runtime_outputs


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
            return (
                online_target,
                True,
                merge_retimed_weight(
                    model_out.get("rhythm_online_retimed_frame_weight"),
                    None,
                    confidence_floor=float(hparams.get("rhythm_retimed_confidence_floor", 0.05)),
                ),
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
