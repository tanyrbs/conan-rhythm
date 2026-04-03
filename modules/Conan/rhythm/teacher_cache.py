from __future__ import annotations

import numpy as np

from .prefix_state import build_prefix_state_from_exec_numpy
from .surface_metadata import (
    RHYTHM_TEACHER_SURFACE_LEARNED_OFFLINE_NAME,
    RHYTHM_TEACHER_TARGET_SOURCE_LEARNED_OFFLINE,
    with_blank_aliases,
)


def _sum_exec_budget(exec_value) -> np.ndarray:
    return np.asarray([float(np.asarray(exec_value, dtype=np.float32).sum())], dtype=np.float32)

def build_prefix_targets_from_exec_numpy(
    speech_exec,
    pause_exec,
    dur_anchor_src,
) -> tuple[np.ndarray, np.ndarray]:
    unit_mask = (np.asarray(dur_anchor_src, dtype=np.float32).reshape(-1) > 0).astype(np.float32)
    return build_prefix_state_from_exec_numpy(
        speech_exec=speech_exec,
        pause_exec=pause_exec,
        dur_anchor_src=dur_anchor_src,
        unit_mask=unit_mask,
    )


def complete_learned_teacher_bundle(
    teacher_bundle_override: dict,
    *,
    source_cache: dict[str, np.ndarray],
    guidance_bundle: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    bundle = with_blank_aliases(dict(teacher_bundle_override or {}))
    if "rhythm_teacher_speech_exec_tgt" not in bundle:
        raise ValueError("learned_offline teacher bundle is missing rhythm_teacher_speech_exec_tgt.")
    if "rhythm_teacher_pause_exec_tgt" not in bundle:
        raise ValueError("learned_offline teacher bundle is missing rhythm_teacher_pause_exec_tgt.")
    expected_units = int(np.asarray(source_cache["dur_anchor_src"]).reshape(-1).shape[0])
    speech_exec = np.asarray(bundle["rhythm_teacher_speech_exec_tgt"], dtype=np.float32).reshape(-1)
    pause_exec = np.asarray(bundle["rhythm_teacher_pause_exec_tgt"], dtype=np.float32).reshape(-1)
    if speech_exec.shape[0] != expected_units or pause_exec.shape[0] != expected_units:
        raise ValueError(
            "learned_offline teacher bundle unit mismatch: "
            f"speech={speech_exec.shape[0]}, pause={pause_exec.shape[0]}, expected={expected_units}."
        )
    bundle["rhythm_teacher_speech_exec_tgt"] = speech_exec.astype(np.float32)
    bundle["rhythm_teacher_pause_exec_tgt"] = pause_exec.astype(np.float32)
    if "rhythm_teacher_speech_budget_tgt" not in bundle:
        bundle["rhythm_teacher_speech_budget_tgt"] = _sum_exec_budget(bundle["rhythm_teacher_speech_exec_tgt"])
    if "rhythm_teacher_pause_budget_tgt" not in bundle:
        bundle["rhythm_teacher_pause_budget_tgt"] = _sum_exec_budget(bundle["rhythm_teacher_pause_exec_tgt"])
    unit_mask = (np.asarray(source_cache["dur_anchor_src"]).reshape(-1) > 0).astype(np.float32)
    for key in (
        "rhythm_teacher_allocation_tgt",
        "rhythm_teacher_prefix_clock_tgt",
        "rhythm_teacher_prefix_backlog_tgt",
    ):
        if key in bundle and np.asarray(bundle[key]).reshape(-1).shape[0] != expected_units:
            raise ValueError(
                f"learned_offline teacher bundle field {key} has length "
                f"{np.asarray(bundle[key]).reshape(-1).shape[0]}, expected={expected_units}."
            )
    if "rhythm_teacher_allocation_tgt" not in bundle:
        allocation = np.zeros_like(unit_mask, dtype=np.float32)
        allocation[:] = (speech_exec + pause_exec) * unit_mask
        bundle["rhythm_teacher_allocation_tgt"] = allocation.astype(np.float32)
    if (
        "rhythm_teacher_prefix_clock_tgt" not in bundle
        or "rhythm_teacher_prefix_backlog_tgt" not in bundle
    ):
        prefix_clock, prefix_backlog = build_prefix_targets_from_exec_numpy(
            bundle["rhythm_teacher_speech_exec_tgt"],
            bundle["rhythm_teacher_pause_exec_tgt"],
            source_cache["dur_anchor_src"],
        )
        if "rhythm_teacher_prefix_clock_tgt" not in bundle:
            bundle["rhythm_teacher_prefix_clock_tgt"] = prefix_clock
        if "rhythm_teacher_prefix_backlog_tgt" not in bundle:
            bundle["rhythm_teacher_prefix_backlog_tgt"] = prefix_backlog
    if "rhythm_teacher_confidence" not in bundle:
        fallback_confidence = 1.0
        if guidance_bundle is not None:
            fallback_confidence = float(
                np.asarray(
                    guidance_bundle.get(
                        "rhythm_guidance_confidence",
                        guidance_bundle.get("rhythm_target_confidence", [1.0]),
                    )
                ).reshape(-1)[0]
            )
        bundle["rhythm_teacher_confidence"] = np.asarray([fallback_confidence], dtype=np.float32)
    return with_blank_aliases(bundle)


def build_learned_offline_teacher_bundle(
    *,
    speech_exec_tgt,
    pause_exec_tgt,
    dur_anchor_src,
    unit_mask=None,
    confidence: float | np.ndarray = 1.0,
) -> dict[str, np.ndarray]:
    dur_anchor_src = np.asarray(dur_anchor_src, dtype=np.float32).reshape(-1)
    if unit_mask is None:
        unit_mask = (dur_anchor_src > 0).astype(np.float32)
    else:
        unit_mask = np.asarray(unit_mask, dtype=np.float32).reshape(-1)
    speech_exec = np.asarray(speech_exec_tgt, dtype=np.float32).reshape(-1)
    pause_exec = np.asarray(pause_exec_tgt, dtype=np.float32).reshape(-1)
    expected_units = int(dur_anchor_src.shape[0])
    if speech_exec.shape[0] != expected_units or pause_exec.shape[0] != expected_units:
        raise ValueError(
            "build_learned_offline_teacher_bundle expects full-length unit surfaces: "
            f"speech={speech_exec.shape[0]}, pause={pause_exec.shape[0]}, expected={expected_units}."
        )
    allocation = ((speech_exec + pause_exec) * unit_mask).astype(np.float32)
    prefix_clock, prefix_backlog = build_prefix_targets_from_exec_numpy(
        speech_exec,
        pause_exec,
        dur_anchor_src,
    )
    confidence_value = float(np.asarray(confidence, dtype=np.float32).reshape(-1)[0])
    return with_blank_aliases({
        "rhythm_teacher_speech_exec_tgt": speech_exec.astype(np.float32),
        "rhythm_teacher_pause_exec_tgt": pause_exec.astype(np.float32),
        "rhythm_teacher_speech_budget_tgt": _sum_exec_budget(speech_exec),
        "rhythm_teacher_pause_budget_tgt": _sum_exec_budget(pause_exec),
        "rhythm_teacher_allocation_tgt": allocation,
        "rhythm_teacher_prefix_clock_tgt": prefix_clock.astype(np.float32),
        "rhythm_teacher_prefix_backlog_tgt": prefix_backlog.astype(np.float32),
        "rhythm_teacher_confidence": np.asarray([confidence_value], dtype=np.float32),
        "rhythm_teacher_surface_name": np.asarray([RHYTHM_TEACHER_SURFACE_LEARNED_OFFLINE_NAME], dtype=np.str_),
        "rhythm_teacher_target_source_id": np.asarray(
            [RHYTHM_TEACHER_TARGET_SOURCE_LEARNED_OFFLINE], dtype=np.int64
        ),
    })


def build_learned_offline_teacher_export_bundle(
    *,
    speech_exec_tgt,
    pause_exec_tgt,
    dur_anchor_src,
    unit_mask=None,
    confidence,
) -> dict[str, np.ndarray]:
    """Stable export contract for learned-offline teacher assets."""
    return build_learned_offline_teacher_bundle(
        speech_exec_tgt=speech_exec_tgt,
        pause_exec_tgt=pause_exec_tgt,
        dur_anchor_src=dur_anchor_src,
        unit_mask=unit_mask,
        confidence=confidence,
    )


__all__ = [
    "build_learned_offline_teacher_bundle",
    "build_learned_offline_teacher_export_bundle",
    "build_prefix_targets_from_exec_numpy",
    "complete_learned_teacher_bundle",
]
