from __future__ import annotations

from .common.runtime_modes import (
    TaskRuntimeState,
    _resolve_online_retimed_sample_confidence,
    merge_retimed_weight,
    resolve_task_apply_override,
    resolve_task_runtime_state,
)
from .common.task_config import (
    resolve_task_retimed_target_mode,
    resolve_task_target_mode,
)
from .duration_v3.runtime_modes import build_duration_v3_ref_conditioning
from .rhythm_v2.runtime_modes import (
    _apply_online_retimed_repair_gate,
    _apply_online_retimed_trace_reliability_gate,
    build_legacy_v2_ref_conditioning,
    collect_legacy_planner_runtime_outputs,
)


def build_rhythm_ref_conditioning(sample, *, explicit=None, backend=None):
    normalized_backend = str(backend or "").strip().lower()
    if normalized_backend == "v3":
        return build_duration_v3_ref_conditioning(sample, explicit=explicit)
    if normalized_backend == "v2":
        return build_legacy_v2_ref_conditioning(sample, explicit=explicit)
    if explicit is not None and not isinstance(explicit, dict):
        return explicit
    source = explicit if isinstance(explicit, dict) else sample
    normalized_v3 = build_duration_v3_ref_conditioning(source)
    if isinstance(normalized_v3, dict) and all(key in normalized_v3 for key in ("global_rate", "operator_coeff")):
        return build_duration_v3_ref_conditioning(sample, explicit=explicit)
    return build_legacy_v2_ref_conditioning(sample, explicit=explicit)


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
            online_weight = _apply_online_retimed_repair_gate(online_weight, model_out)
            online_weight = _apply_online_retimed_trace_reliability_gate(online_weight, model_out)
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
    "build_duration_v3_ref_conditioning",
    "build_legacy_v2_ref_conditioning",
    "build_rhythm_ref_conditioning",
    "collect_legacy_planner_runtime_outputs",
    "merge_retimed_weight",
    "resolve_acoustic_target_post_model",
    "resolve_task_apply_override",
    "resolve_task_runtime_state",
]
