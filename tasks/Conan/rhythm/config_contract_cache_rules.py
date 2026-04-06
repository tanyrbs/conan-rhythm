from __future__ import annotations

"""Cache field expectations and inspected-item checks for rhythm contracts."""

from typing import Any, Mapping, Sequence

import numpy as np

from modules.Conan.rhythm.policy import (
    expected_cache_contract as build_expected_cache_contract,
    normalize_distill_surface,
    normalize_primary_target_surface,
)
from modules.Conan.rhythm.supervision import (
    RHYTHM_GUIDANCE_SURFACE_NAME,
    RHYTHM_RETIMED_SOURCE_GUIDANCE,
    RHYTHM_RETIMED_SOURCE_TEACHER,
    compatible_rhythm_cache_versions,
    is_rhythm_cache_version_compatible,
)

from .config_contract_core import (
    RhythmConfigContractContext,
    RhythmConfigContractReport,
    _dedup_groups,
    _dedup_messages,
    merge_contract_reports,
)


CORE_RHYTHM_FIELDS: tuple[str, ...] = (
    "content_units",
    "dur_anchor_src",
    "open_run_mask",
    "sealed_mask",
    "sep_hint",
    "boundary_confidence",
    "ref_rhythm_stats",
    "ref_rhythm_trace",
)
_CACHED_ONLY_META_FIELD_GROUPS: tuple[tuple[str, ...], ...] = (
    ("rhythm_cache_version",),
    ("rhythm_unit_hop_ms",),
    ("rhythm_trace_hop_ms",),
    ("rhythm_trace_bins",),
    ("rhythm_trace_horizon",),
    ("rhythm_slow_topk",),
    ("rhythm_selector_cell_size",),
    ("rhythm_source_phrase_threshold",),
    ("rhythm_reference_mode_id",),
    ("rhythm_guidance_surface_name",),
)
_GUIDANCE_FIELD_GROUPS: tuple[tuple[str, ...], ...] = (
    ("rhythm_speech_exec_tgt",),
    ("rhythm_blank_exec_tgt", "rhythm_pause_exec_tgt"),
    ("rhythm_speech_budget_tgt",),
    ("rhythm_blank_budget_tgt", "rhythm_pause_budget_tgt"),
    ("rhythm_target_confidence",),
    ("rhythm_guidance_confidence",),
    ("rhythm_guidance_surface_name",),
)
_TEACHER_CORE_FIELD_GROUPS: tuple[tuple[str, ...], ...] = (
    ("rhythm_teacher_speech_exec_tgt",),
    ("rhythm_teacher_blank_exec_tgt", "rhythm_teacher_pause_exec_tgt"),
    ("rhythm_teacher_speech_budget_tgt",),
    ("rhythm_teacher_blank_budget_tgt", "rhythm_teacher_pause_budget_tgt"),
    ("rhythm_teacher_confidence",),
    ("rhythm_teacher_target_source_id",),
    ("rhythm_teacher_surface_name",),
)
_TEACHER_ALLOCATION_FIELD_GROUPS: tuple[tuple[str, ...], ...] = (
    ("rhythm_teacher_allocation_tgt",),
)
_TEACHER_PREFIX_FIELD_GROUPS: tuple[tuple[str, ...], ...] = (
    ("rhythm_teacher_prefix_clock_tgt",),
    ("rhythm_teacher_prefix_backlog_tgt",),
)
_TEACHER_COMPONENT_CONFIDENCE_GROUPS: tuple[tuple[str, ...], ...] = (
    ("rhythm_teacher_confidence_exec",),
    ("rhythm_teacher_confidence_budget",),
    ("rhythm_teacher_confidence_prefix",),
    ("rhythm_teacher_confidence_allocation",),
    ("rhythm_teacher_confidence_shape",),
)
_RETIMED_FIELD_GROUPS: tuple[tuple[str, ...], ...] = (
    ("rhythm_retimed_mel_tgt",),
    ("rhythm_retimed_mel_len",),
    ("rhythm_retimed_frame_weight",),
    ("rhythm_retimed_target_confidence",),
    ("rhythm_retimed_target_source_id",),
    ("rhythm_retimed_target_surface_name",),
)
_REFERENCE_SIDECAR_FIELD_GROUPS: tuple[tuple[str, ...], ...] = (
    ("slow_rhythm_memory",),
    ("slow_rhythm_summary",),
    ("selector_meta_indices",),
    ("selector_meta_scores",),
    ("selector_meta_starts",),
    ("selector_meta_ends",),
)
_PLANNER_REFERENCE_SIDECAR_FIELD_GROUPS: tuple[tuple[str, ...], ...] = (
    ("planner_slow_rhythm_memory",),
    ("planner_slow_rhythm_summary",),
)


def validate_cache_field_contract(
    context: RhythmConfigContractContext,
) -> RhythmConfigContractReport:
    hp = context.hparams
    primary = normalize_primary_target_surface(
        str(hp.get("rhythm_primary_target_surface", "guidance") or "guidance").strip().lower()
    )
    distill = normalize_distill_surface(
        str(hp.get("rhythm_distill_surface", "auto") or "auto").strip().lower()
    )
    cached_only = (
        str(hp.get("rhythm_dataset_target_mode", "prefer_cache") or "prefer_cache").strip().lower()
        == "cached_only"
    )
    target_mode = str(hp.get("rhythm_dataset_target_mode", "prefer_cache") or "prefer_cache").strip().lower()
    runtime_only = target_mode == "runtime_only"
    retimed_source = str(hp.get("rhythm_binarize_retimed_mel_source", "guidance") or "guidance").strip().lower()
    distill_exec_weight = float(hp.get("rhythm_distill_exec_weight", 1.0))
    distill_budget_weight = float(hp.get("rhythm_distill_budget_weight", 0.5))
    distill_allocation_weight = float(hp.get("rhythm_distill_allocation_weight", 0.5))
    distill_prefix_weight = float(hp.get("rhythm_distill_prefix_weight", 0.25))
    distill_speech_shape_weight = float(hp.get("rhythm_distill_speech_shape_weight", 0.0))
    distill_pause_shape_weight = float(hp.get("rhythm_distill_pause_shape_weight", 0.0))

    expected_groups: list[tuple[str, ...]] = [(key,) for key in sorted(CORE_RHYTHM_FIELDS)]
    errors: list[str] = []
    warnings: list[str] = []

    if cached_only:
        expected_groups.extend(_CACHED_ONLY_META_FIELD_GROUPS)
    if primary == "guidance":
        expected_groups.extend(_GUIDANCE_FIELD_GROUPS)
    needs_cached_teacher_core = (
        bool(hp.get("rhythm_require_cached_teacher", False))
        or (not runtime_only and primary == "teacher")
        or (not runtime_only and bool(hp.get("rhythm_binarize_teacher_targets", False)))
        or (not runtime_only and retimed_source == "teacher")
    )
    if (
        needs_cached_teacher_core
    ):
        expected_groups.extend(_TEACHER_CORE_FIELD_GROUPS)
    if (
        bool(hp.get("rhythm_require_retimed_cache", False))
        or bool(hp.get("rhythm_apply_train_override", False))
        or bool(hp.get("rhythm_apply_valid_override", False))
    ):
        expected_groups.extend(_RETIMED_FIELD_GROUPS)
    if bool(hp.get("rhythm_export_debug_sidecars", False)):
        expected_groups.extend(_REFERENCE_SIDECAR_FIELD_GROUPS)
        expected_groups.extend(_PLANNER_REFERENCE_SIDECAR_FIELD_GROUPS)

    if float(hp.get("lambda_rhythm_distill", 0.0)) > 0.0:
        if distill == "none":
            errors.append("lambda_rhythm_distill > 0 but rhythm_distill_surface disables distillation.")
        if distill == "offline" and not bool(hp.get("rhythm_enable_dual_mode_teacher", False)):
            errors.append("Offline distillation requires rhythm_enable_dual_mode_teacher: true.")
        if distill == "cache":
            expected_groups.extend(_TEACHER_CORE_FIELD_GROUPS)
            if distill_allocation_weight > 0.0:
                expected_groups.extend(_TEACHER_ALLOCATION_FIELD_GROUPS)
            if distill_prefix_weight > 0.0:
                expected_groups.extend(_TEACHER_PREFIX_FIELD_GROUPS)
            if distill_exec_weight > 0.0:
                expected_groups.append(("rhythm_teacher_confidence_exec",))
            if distill_budget_weight > 0.0:
                expected_groups.append(("rhythm_teacher_confidence_budget",))
            if distill_prefix_weight > 0.0:
                expected_groups.append(("rhythm_teacher_confidence_prefix",))
            if distill_allocation_weight > 0.0:
                expected_groups.append(("rhythm_teacher_confidence_allocation",))
            if distill_speech_shape_weight > 0.0 or distill_pause_shape_weight > 0.0:
                expected_groups.append(("rhythm_teacher_confidence_shape",))

    if bool(hp.get("rhythm_apply_train_override", False)) and not bool(
        hp.get("rhythm_use_retimed_target_if_available", False)
    ):
        errors.append("Train-time retimed rendering requires rhythm_use_retimed_target_if_available: true.")
    if bool(hp.get("rhythm_schedule_only_stage", False)) and bool(hp.get("rhythm_apply_train_override", False)):
        errors.append("Schedule-only stage should not enable train-time retimed rendering.")
    if cached_only and int(hp.get("rhythm_cache_version", -1)) <= 0:
        errors.append("cached_only requires a positive rhythm_cache_version.")
    if primary == "teacher" and not runtime_only and not bool(hp.get("rhythm_binarize_teacher_targets", False)):
        warnings.append("Primary surface is teacher but rhythm_binarize_teacher_targets is false.")

    return RhythmConfigContractReport(
        profile=context.profile,
        stage=context.stage,
        required_field_groups=_dedup_groups(expected_groups),
        errors=_dedup_messages(errors),
        warnings=_dedup_messages(warnings),
    )


def validate_required_field_presence(
    context: RhythmConfigContractContext,
    items: Sequence[Mapping[str, Any]],
    *,
    split: str,
) -> RhythmConfigContractReport:
    cache_report = validate_cache_field_contract(context)
    inspected = len(items)
    errors: list[str] = []
    if inspected <= 0:
        errors.append(f"Split '{split}' has no inspected items for cache contract validation.")
    else:
        for group in cache_report.required_field_groups:
            have = 0
            for item in items:
                if any(key in item for key in group):
                    have += 1
            if have < inspected:
                label = " | ".join(group)
                errors.append(
                    f"Split '{split}' is missing required field group '{label}' in {inspected - have} inspected items."
                )
    return RhythmConfigContractReport(
        profile=context.profile,
        stage=context.stage,
        required_field_groups=cache_report.required_field_groups,
        errors=_dedup_messages(errors),
    )


def _extract_scalar(value: Any):
    arr = np.asarray(value)
    if arr.size <= 0:
        raise RuntimeError("Encountered empty scalar field while validating rhythm cache metadata.")
    if arr.size != 1:
        raise RuntimeError(
            f"Encountered non-scalar field while validating rhythm cache metadata: shape={tuple(arr.shape)}."
        )
    return arr.reshape(-1)[0]


def validate_inspected_cache_items(
    context: RhythmConfigContractContext,
    items: Sequence[Mapping[str, Any]],
    *,
    split: str,
) -> RhythmConfigContractReport:
    hp = context.hparams
    cached_only = (
        str(hp.get("rhythm_dataset_target_mode", "prefer_cache") or "prefer_cache").strip().lower()
        == "cached_only"
    )
    expected_teacher_surface = context.policy.teacher_surface_name
    expected_teacher_source_id = context.policy.teacher_target_source_id
    need_teacher = (
        normalize_primary_target_surface(
            str(hp.get("rhythm_primary_target_surface", "guidance") or "guidance").strip().lower()
        )
        == "teacher"
        or bool(hp.get("rhythm_require_cached_teacher", False))
        or (
            float(hp.get("lambda_rhythm_distill", 0.0)) > 0.0
            and normalize_distill_surface(
                str(hp.get("rhythm_distill_surface", "auto") or "auto").strip().lower()
            )
            == "cache"
        )
        or str(hp.get("rhythm_binarize_retimed_mel_source", "guidance") or "guidance").strip().lower() == "teacher"
    )
    need_retimed = bool(hp.get("rhythm_require_retimed_cache", False)) or bool(
        hp.get("rhythm_apply_train_override", False)
    ) or bool(hp.get("rhythm_apply_valid_override", False))
    expected_meta = build_expected_cache_contract(hp)

    errors: list[str] = []
    for idx, item in enumerate(items):
        item_name = str(item.get("item_name", f"{split}[{idx}]"))
        if cached_only:
            for key, expected in expected_meta.items():
                if key not in item:
                    errors.append(f"{item_name}: missing cached_only contract key '{key}'.")
                    continue
                found = _extract_scalar(item[key])
                if key == "rhythm_cache_version":
                    if not is_rhythm_cache_version_compatible(int(found), int(expected)):
                        errors.append(
                            f"{item_name}: cache contract mismatch for {key}, found={int(found)}, "
                            f"expected one of {compatible_rhythm_cache_versions(int(expected))}."
                        )
                    continue
                if isinstance(expected, float):
                    if abs(float(found) - expected) > 1e-5:
                        errors.append(
                            f"{item_name}: cache contract mismatch for {key}, found={float(found):.6f}, expected={expected:.6f}."
                        )
                elif int(found) != expected:
                    errors.append(
                        f"{item_name}: cache contract mismatch for {key}, found={int(found)}, expected={expected}."
                    )
            if "rhythm_guidance_surface_name" in item:
                guidance_surface = str(_extract_scalar(item["rhythm_guidance_surface_name"]))
                if guidance_surface != RHYTHM_GUIDANCE_SURFACE_NAME:
                    errors.append(
                        f"{item_name}: rhythm_guidance_surface_name mismatch, found={guidance_surface}, expected={RHYTHM_GUIDANCE_SURFACE_NAME}."
                    )
        if need_teacher:
            if "rhythm_teacher_surface_name" not in item:
                errors.append(f"{item_name}: missing rhythm_teacher_surface_name for teacher-backed training.")
            else:
                teacher_surface = str(_extract_scalar(item["rhythm_teacher_surface_name"]))
                if teacher_surface != expected_teacher_surface:
                    errors.append(
                        f"{item_name}: rhythm_teacher_surface_name mismatch, found={teacher_surface}, expected={expected_teacher_surface}."
                    )
            if "rhythm_teacher_target_source_id" not in item:
                errors.append(f"{item_name}: missing rhythm_teacher_target_source_id for teacher-backed training.")
            else:
                found_source_id = int(_extract_scalar(item["rhythm_teacher_target_source_id"]))
                if found_source_id != expected_teacher_source_id:
                    errors.append(
                        f"{item_name}: rhythm_teacher_target_source_id mismatch, found={found_source_id}, expected={expected_teacher_source_id}."
                    )
        if need_retimed:
            retimed_source = str(hp.get("rhythm_binarize_retimed_mel_source", "guidance") or "guidance").strip().lower()
            expected_retimed_source_id = (
                RHYTHM_RETIMED_SOURCE_TEACHER
                if retimed_source == "teacher"
                else RHYTHM_RETIMED_SOURCE_GUIDANCE
            )
            expected_retimed_surface = (
                expected_teacher_surface if retimed_source == "teacher" else RHYTHM_GUIDANCE_SURFACE_NAME
            )
            if "rhythm_retimed_target_source_id" not in item:
                errors.append(f"{item_name}: missing rhythm_retimed_target_source_id for retimed training.")
            else:
                found_source_id = int(_extract_scalar(item["rhythm_retimed_target_source_id"]))
                if found_source_id != expected_retimed_source_id:
                    errors.append(
                        f"{item_name}: rhythm_retimed_target_source_id mismatch, found={found_source_id}, expected={expected_retimed_source_id}."
                    )
            if "rhythm_retimed_target_surface_name" not in item:
                errors.append(f"{item_name}: missing rhythm_retimed_target_surface_name for retimed training.")
            else:
                found_surface = str(_extract_scalar(item["rhythm_retimed_target_surface_name"]))
                if found_surface != expected_retimed_surface:
                    errors.append(
                        f"{item_name}: rhythm_retimed_target_surface_name mismatch, found={found_surface}, expected={expected_retimed_surface}."
                    )
    return RhythmConfigContractReport(
        profile=context.profile,
        stage=context.stage,
        errors=_dedup_messages(errors),
    )


def collect_cache_contract_report(
    context: RhythmConfigContractContext,
    items: Sequence[Mapping[str, Any]],
    *,
    split: str,
) -> RhythmConfigContractReport:
    return merge_contract_reports(
        validate_cache_field_contract(context),
        validate_required_field_presence(context, items, split=split),
        validate_inspected_cache_items(context, items, split=split),
    )


__all__ = [
    "CORE_RHYTHM_FIELDS",
    "collect_cache_contract_report",
    "validate_cache_field_contract",
    "validate_inspected_cache_items",
    "validate_required_field_presence",
]


