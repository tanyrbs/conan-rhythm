from __future__ import annotations

"""Shared preflight implementation extracted from scripts/preflight_rhythm_v2.py."""

import argparse
import os
import sys
from collections import Counter

import numpy as np


from modules.Conan.rhythm.supervision import (
    RHYTHM_CACHE_VERSION,
    RHYTHM_GUIDANCE_SURFACE_NAME,
    RHYTHM_RETIMED_SOURCE_GUIDANCE,
    RHYTHM_RETIMED_SOURCE_TEACHER,
    compatible_rhythm_cache_versions,
    is_rhythm_cache_version_compatible,
    materialize_rhythm_cache_compat_fields,
)
from modules.Conan.rhythm.policy import (
    build_rhythm_hparams_policy,
    expected_cache_contract as build_expected_cache_contract,
    normalize_distill_surface as policy_normalize_distill_surface,
    normalize_primary_target_surface as policy_normalize_primary_target_surface,
    normalize_retimed_target_mode as policy_normalize_retimed_target_mode,
    resolve_cumplan_lambda as resolve_policy_cumplan_lambda,
    resolve_runtime_offline_teacher_enable as resolve_policy_runtime_offline_teacher_enable,
)
from modules.Conan.rhythm.stages import (
    detect_rhythm_stage,
    resolve_runtime_dual_mode_teacher_enable,
    resolve_teacher_as_main,
)
from modules.Conan.rhythm.validation import collect_rhythm_contract_issues
from utils.commons.hparams import set_hparams
from utils.commons.indexed_datasets import IndexedDataset


CORE_RHYTHM_FIELDS = {
    "content_units",
    "dur_anchor_src",
    "open_run_mask",
    "sealed_mask",
    "sep_hint",
    "boundary_confidence",
    "ref_rhythm_stats",
    "ref_rhythm_trace",
}
CACHED_ONLY_META_FIELD_GROUPS = [
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
]

GUIDANCE_FIELD_GROUPS = [
    ("rhythm_speech_exec_tgt",),
    ("rhythm_blank_exec_tgt", "rhythm_pause_exec_tgt"),
    ("rhythm_speech_budget_tgt",),
    ("rhythm_blank_budget_tgt", "rhythm_pause_budget_tgt"),
    ("rhythm_target_confidence",),
    ("rhythm_guidance_confidence",),
    ("rhythm_guidance_surface_name",),
]

TEACHER_FIELD_GROUPS = [
    ("rhythm_teacher_speech_exec_tgt",),
    ("rhythm_teacher_blank_exec_tgt", "rhythm_teacher_pause_exec_tgt"),
    ("rhythm_teacher_speech_budget_tgt",),
    ("rhythm_teacher_blank_budget_tgt", "rhythm_teacher_pause_budget_tgt"),
    ("rhythm_teacher_allocation_tgt",),
    ("rhythm_teacher_prefix_clock_tgt",),
    ("rhythm_teacher_prefix_backlog_tgt",),
    ("rhythm_teacher_confidence",),
    ("rhythm_teacher_target_source_id",),
    ("rhythm_teacher_surface_name",),
]

RETIMED_FIELD_GROUPS = [
    ("rhythm_retimed_mel_tgt",),
    ("rhythm_retimed_mel_len",),
    ("rhythm_retimed_frame_weight",),
    ("rhythm_retimed_target_confidence",),
    ("rhythm_retimed_target_source_id",),
    ("rhythm_retimed_target_surface_name",),
]


def _normalize_surface(surface: str) -> str:
    return policy_normalize_primary_target_surface(surface)


def _normalize_distill(surface: str) -> str:
    return policy_normalize_distill_surface(surface)


def _normalize_retimed_target_mode(mode: str) -> str:
    return policy_normalize_retimed_target_mode(mode)


def _extract_scalar(value):
    arr = np.asarray(value)
    if arr.size <= 0:
        raise RuntimeError("Encountered empty scalar field while validating rhythm cache metadata.")
    return arr.reshape(-1)[0]


def _expected_cache_contract(hp: dict) -> dict[str, int | float]:
    return build_expected_cache_contract(hp)


def _resolve_expected_teacher_target_source(hp: dict) -> str:
    return build_rhythm_hparams_policy(hp).teacher_target_source


def _resolve_expected_teacher_surface(hp: dict) -> str:
    return build_rhythm_hparams_policy(hp).teacher_surface_name


def _resolve_expected_teacher_target_source_id(hp: dict) -> int:
    return build_rhythm_hparams_policy(hp).teacher_target_source_id


def _detect_stage(hp: dict, config_path: str) -> str:
    return detect_rhythm_stage(hp, config_name=config_path)


def _detect_profile(hp: dict, config_path: str) -> str:
    is_minimal_flag = bool(hp.get("rhythm_minimal_v1_profile", False)) or "minimal_v1" in str(config_path).lower()
    if (
        is_minimal_flag
        and not bool(hp.get("rhythm_enable_dual_mode_teacher", False))
        and not bool(hp.get("rhythm_apply_train_override", False))
        and not bool(hp.get("rhythm_apply_valid_override", False))
        and _normalize_distill(str(hp.get("rhythm_distill_surface", "auto") or "auto").strip().lower()) == "none"
    ):
        return "minimal_v1"
    return "default"


def _resolve_cumplan_lambda(hp: dict) -> float:
    return resolve_policy_cumplan_lambda(hp)


def _validate_profile_contract(
    hp: dict,
    *,
    config_path: str,
    model_dry_run: bool,
) -> tuple[str, list[str], list[str]]:
    profile = _detect_profile(hp, config_path)
    errors: list[str] = []
    warnings: list[str] = []
    if profile != "minimal_v1":
        return profile, errors, warnings

    policy = build_rhythm_hparams_policy(hp, config_path=config_path)
    target_mode = policy.target_mode
    primary = policy.primary_target_surface
    distill = policy.distill_surface
    if target_mode != "cached_only":
        errors.append("minimal_v1 profile requires rhythm_dataset_target_mode: cached_only.")
    if primary != "teacher":
        errors.append("minimal_v1 profile requires rhythm_primary_target_surface: teacher.")
    if distill != "none":
        errors.append("minimal_v1 profile requires rhythm_distill_surface: none.")
    if not bool(hp.get("rhythm_require_cached_teacher", False)):
        errors.append("minimal_v1 profile requires rhythm_require_cached_teacher: true.")
    if not bool(hp.get("rhythm_binarize_teacher_targets", False)):
        errors.append("minimal_v1 profile requires rhythm_binarize_teacher_targets: true.")
    if _resolve_expected_teacher_target_source(hp) != "learned_offline":
        errors.append("minimal_v1 profile requires rhythm_teacher_target_source: learned_offline.")
    if bool(hp.get("rhythm_dataset_build_guidance_from_ref", True)):
        errors.append("minimal_v1 profile should disable runtime guidance target synthesis.")
    if bool(hp.get("rhythm_dataset_build_teacher_from_ref", False)):
        errors.append("minimal_v1 profile should disable runtime teacher target synthesis.")
    if bool(hp.get("rhythm_enable_dual_mode_teacher", False)):
        errors.append("minimal_v1 profile should keep rhythm_enable_dual_mode_teacher: false.")
    if bool(hp.get("rhythm_require_retimed_cache", False)):
        errors.append("minimal_v1 profile should not require retimed mel cache.")
    if bool(hp.get("rhythm_apply_train_override", False)) or bool(hp.get("rhythm_apply_valid_override", False)):
        errors.append("minimal_v1 profile should keep train/valid on source-aligned canvas.")

    if float(hp.get("lambda_rhythm_exec_speech", 0.0)) <= 0.0:
        errors.append("minimal_v1 profile requires lambda_rhythm_exec_speech > 0.")
    if float(hp.get("lambda_rhythm_exec_pause", 0.0)) <= 0.0:
        errors.append("minimal_v1 profile requires lambda_rhythm_exec_pause > 0.")
    if _resolve_cumplan_lambda(hp) <= 0.0:
        errors.append("minimal_v1 profile requires lambda_rhythm_cumplan (or lambda_rhythm_carry) > 0.")
    if float(hp.get("lambda_rhythm_budget", 0.0)) < 0.0:
        errors.append("minimal_v1 profile does not allow negative lambda_rhythm_budget.")
    elif float(hp.get("lambda_rhythm_budget", 0.0)) == 0.0:
        warnings.append("minimal_v1 profile keeps lambda_rhythm_budget at 0; maintained path expects a small positive budget guardrail.")
    if float(hp.get("lambda_rhythm_guidance", 0.0)) > 0.0:
        errors.append("minimal_v1 profile requires lambda_rhythm_guidance: 0.")
    if float(hp.get("lambda_rhythm_plan", 0.0)) > 0.0:
        errors.append("minimal_v1 profile requires lambda_rhythm_plan: 0.")
    if float(hp.get("lambda_rhythm_distill", 0.0)) > 0.0:
        errors.append("minimal_v1 profile requires lambda_rhythm_distill: 0.")

    if not model_dry_run:
        warnings.append("minimal_v1 train-ready preflight should include --model_dry_run before starting training.")
    if "rhythm_min_unit_frames" in hp:
        legacy_min_unit = hp.get("rhythm_min_unit_frames")
        if legacy_min_unit not in {None, "", 0, 0.0, "0", "0.0"}:
            warnings.append(
                "rhythm_min_unit_frames is a legacy/unsupported knob in the maintained path; "
                "remove it instead of assuming it is active."
            )
    return profile, errors, warnings


def _validate_stage_contract(hp: dict, *, config_path: str) -> tuple[str, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    policy = build_rhythm_hparams_policy(hp, config_path=config_path)
    stage = policy.stage
    profile = _detect_profile(hp, config_path)
    target_mode = policy.target_mode
    primary = policy.primary_target_surface
    distill = policy.distill_surface
    require_cached_teacher = policy.require_cached_teacher
    require_retimed_cache = policy.require_retimed_cache
    enable_dual = bool(hp.get("rhythm_enable_dual_mode_teacher", False))
    enable_learned_offline_teacher = bool(hp.get("rhythm_enable_learned_offline_teacher", True))
    explicit_runtime_teacher_enable = hp.get("rhythm_runtime_enable_learned_offline_teacher", None)
    runtime_offline_teacher_enable = policy.runtime_offline_teacher_enabled
    runtime_dual_mode_teacher_enable = resolve_runtime_dual_mode_teacher_enable(
        hp,
        stage=stage,
        infer=False,
    )
    expected_teacher_source = policy.teacher_target_source
    legacy_schedule_flag = bool(hp.get("rhythm_schedule_only_stage", False))
    strict_mainline = policy.strict_mainline
    optimize_module_only = bool(hp.get("rhythm_optimize_module_only", False))
    lambda_distill = float(hp.get("lambda_rhythm_distill", 0.0))
    lambda_teacher_aux = float(hp.get("lambda_rhythm_teacher_aux", 0.0) or 0.0)
    teacher_as_main = policy.teacher_as_main
    apply_train = bool(hp.get("rhythm_apply_train_override", False))
    apply_valid = bool(hp.get("rhythm_apply_valid_override", False))
    use_retimed_target = bool(hp.get("rhythm_use_retimed_target_if_available", False))
    retimed_target_mode = policy.retimed_target_mode
    retimed_target_start = int(hp.get("rhythm_retimed_target_start_steps", 0) or 0)
    online_target_start = int(hp.get("rhythm_online_retimed_target_start_steps", retimed_target_start) or retimed_target_start)
    use_retimed_pitch_target = bool(hp.get("rhythm_use_retimed_pitch_target", False))
    disable_pitch_when_retimed = bool(hp.get("rhythm_disable_pitch_loss_when_retimed", True))
    retimed_source = str(hp.get("rhythm_binarize_retimed_mel_source", "guidance") or "guidance").strip().lower()
    binarize_teacher = bool(hp.get("rhythm_binarize_teacher_targets", False))
    binarize_retimed = bool(hp.get("rhythm_binarize_retimed_mel_targets", False))
    disable_mel_adv_when_retimed = bool(hp.get("rhythm_disable_mel_adv_when_retimed", True))
    lambda_mel_adv = float(hp.get("lambda_mel_adv", 0.0))
    lambda_guidance = float(hp.get("lambda_rhythm_guidance", 0.0))
    distill_budget_weight = float(hp.get("rhythm_distill_budget_weight", 0.5))
    distill_allocation_weight = float(hp.get("rhythm_distill_allocation_weight", 0.5))
    distill_prefix_weight = float(hp.get("rhythm_distill_prefix_weight", 0.25))
    compact_joint_loss = bool(hp.get("rhythm_compact_joint_loss", True))
    pause_topk_ratio = float(hp.get("rhythm_projector_pause_topk_ratio", 0.35))
    pause_topk_ratio_train_start = float(hp.get("rhythm_projector_pause_topk_ratio_train_start", 1.0))
    pause_topk_ratio_train_end = float(hp.get("rhythm_projector_pause_topk_ratio_train_end", pause_topk_ratio))
    pause_topk_ratio_anneal_steps = int(hp.get("rhythm_projector_pause_topk_ratio_anneal_steps", 20000) or 0)
    pause_topk_ratio_warmup_steps = int(hp.get("rhythm_projector_pause_topk_ratio_warmup_steps", 0) or 0)
    distill_conf_floor = float(hp.get("rhythm_distill_confidence_floor", 0.05))
    distill_conf_power = float(hp.get("rhythm_distill_confidence_power", 1.0))
    source_boundary_scale = float(hp.get("rhythm_source_boundary_scale", 1.0))
    source_boundary_scale_train_start = float(hp.get("rhythm_source_boundary_scale_train_start", 1.0))
    source_boundary_scale_train_end = float(hp.get("rhythm_source_boundary_scale_train_end", source_boundary_scale))
    source_boundary_scale_anneal_steps = int(hp.get("rhythm_source_boundary_scale_anneal_steps", 20000) or 0)
    source_boundary_scale_warmup_steps = int(hp.get("rhythm_source_boundary_scale_warmup_steps", 0) or 0)
    teacher_source_boundary_scale = float(hp.get("rhythm_teacher_source_boundary_scale", source_boundary_scale))
    export_debug_sidecars = bool(hp.get("rhythm_export_debug_sidecars", False))
    export_cache_audit_to_sample = bool(hp.get("rhythm_export_cache_audit_to_sample", False))
    public_losses = list(hp.get("rhythm_public_losses", []) or [])
    configured_cache_version = int(hp.get("rhythm_cache_version", RHYTHM_CACHE_VERSION))

    if retimed_target_mode not in {"cached", "online", "hybrid"}:
        errors.append(f"Unsupported rhythm_retimed_target_mode: {retimed_target_mode}")
    if configured_cache_version != int(RHYTHM_CACHE_VERSION):
        errors.append(
            f"rhythm_cache_version mismatch: configured={configured_cache_version}, maintained={int(RHYTHM_CACHE_VERSION)}."
        )
    if retimed_target_start < 0 or online_target_start < 0:
        errors.append("rhythm_retimed_target_start_steps / rhythm_online_retimed_target_start_steps must be >= 0.")
    if online_target_start < retimed_target_start:
        warnings.append("rhythm_online_retimed_target_start_steps < rhythm_retimed_target_start_steps; online target switch may start earlier than retimed stage.")
    if enable_dual and not enable_learned_offline_teacher:
        errors.append("rhythm_enable_dual_mode_teacher requires rhythm_enable_learned_offline_teacher: true.")
    if enable_dual and not runtime_offline_teacher_enable:
        errors.append(
            "rhythm_enable_dual_mode_teacher requires runtime learned offline teacher branch to be enabled "
            "(check rhythm_runtime_enable_learned_offline_teacher / factory resolution)."
        )
    if runtime_dual_mode_teacher_enable and not enable_dual:
        errors.append(
            "Runtime dual-mode teacher resolved enabled while rhythm_enable_dual_mode_teacher is false."
        )
    if explicit_runtime_teacher_enable is True and not enable_learned_offline_teacher and lambda_distill <= 0.0:
        warnings.append(
            "rhythm_runtime_enable_learned_offline_teacher=true while rhythm_enable_learned_offline_teacher=false "
            "and no distillation is active; this adds runtime overhead without a maintained stage objective."
        )
    for name, value in {
        "rhythm_projector_pause_topk_ratio": pause_topk_ratio,
        "rhythm_projector_pause_topk_ratio_train_start": pause_topk_ratio_train_start,
        "rhythm_projector_pause_topk_ratio_train_end": pause_topk_ratio_train_end,
    }.items():
        if not (0.0 <= value <= 1.0):
            errors.append(f"{name} must be in [0, 1].")
    if pause_topk_ratio_anneal_steps < 0:
        errors.append("rhythm_projector_pause_topk_ratio_anneal_steps must be >= 0.")
    if pause_topk_ratio_warmup_steps < 0:
        errors.append("rhythm_projector_pause_topk_ratio_warmup_steps must be >= 0.")
    if pause_topk_ratio_train_start < pause_topk_ratio_train_end:
        warnings.append("pause top-k anneal is configured sparse->dense; maintained path usually uses dense->sparse.")
    if not (0.0 < distill_conf_floor <= 1.0):
        errors.append("rhythm_distill_confidence_floor must be in (0, 1].")
    if distill_conf_power <= 0.0:
        errors.append("rhythm_distill_confidence_power must be > 0.")
    for name, value in {
        "rhythm_source_boundary_scale": source_boundary_scale,
        "rhythm_source_boundary_scale_train_start": source_boundary_scale_train_start,
        "rhythm_source_boundary_scale_train_end": source_boundary_scale_train_end,
        "rhythm_teacher_source_boundary_scale": teacher_source_boundary_scale,
    }.items():
        if value < 0.0:
            errors.append(f"{name} must be >= 0.")
    if source_boundary_scale_anneal_steps < 0:
        errors.append("rhythm_source_boundary_scale_anneal_steps must be >= 0.")
    if source_boundary_scale_warmup_steps < 0:
        errors.append("rhythm_source_boundary_scale_warmup_steps must be >= 0.")
    if source_boundary_scale_train_start < source_boundary_scale_train_end:
        warnings.append("source-boundary prior anneal is configured weak->strong; maintained path usually uses strong->soft.")
    if (apply_train or apply_valid) and not use_retimed_pitch_target and not disable_pitch_when_retimed:
        errors.append(
            "Retimed train/valid rendering must either enable rhythm_use_retimed_pitch_target "
            "or set rhythm_disable_pitch_loss_when_retimed: true."
        )
    if (apply_train or apply_valid) and export_debug_sidecars:
        warnings.append("rhythm_export_debug_sidecars=true on retimed train/valid path increases batch contract complexity.")
    if export_cache_audit_to_sample:
        warnings.append("rhythm_export_cache_audit_to_sample=true adds cache appendix fields to runtime batch; keep it off outside audits.")
    if public_losses:
        required_public_losses = {"L_exec_speech", "L_exec_pause", "L_budget", "L_prefix_state", "L_base"}
        missing_public = sorted(required_public_losses.difference(set(public_losses)))
        if missing_public:
            warnings.append(f"rhythm_public_losses is missing maintained mainline aliases: {missing_public}.")

    if stage == "teacher_offline":
        if strict_mainline:
            warnings.append(
                "teacher_offline is a teacher-asset build path; keep rhythm_strict_mainline=false for the maintained student mainline."
            )
        if legacy_schedule_flag:
            errors.append("teacher_offline stage should keep rhythm_schedule_only_stage: false.")
        if not enable_learned_offline_teacher:
            errors.append("teacher_offline stage requires rhythm_enable_learned_offline_teacher: true.")
        if not runtime_offline_teacher_enable:
            errors.append("teacher_offline stage requires runtime offline teacher branch enabled.")
        if not teacher_as_main:
            errors.append("teacher_offline stage requires rhythm_teacher_as_main: true.")
        if runtime_dual_mode_teacher_enable:
            errors.append("teacher_offline stage should not resolve runtime dual-mode teacher execution.")
        if enable_dual:
            errors.append("teacher_offline stage should keep rhythm_enable_dual_mode_teacher: false.")
        if lambda_distill > 0.0:
            errors.append("teacher_offline stage should keep lambda_rhythm_distill: 0.")
        if lambda_teacher_aux > 0.0:
            errors.append("teacher_offline stage should keep lambda_rhythm_teacher_aux: 0.")
        if primary == "teacher":
            errors.append(
                "teacher_offline stage should not use rhythm_primary_target_surface: teacher; "
                "bootstrap the teacher from guidance/self targets."
            )
        teacher_target_source = _resolve_expected_teacher_target_source(hp)
        if teacher_target_source != "algorithmic":
            warnings.append(
                "teacher_offline stage does not consume learned_offline teacher caches; "
                "prefer rhythm_teacher_target_source: algorithmic for clarity."
            )
    elif stage == "legacy_schedule_only":
        warnings.append(
            "legacy_schedule_only is no longer part of the maintained chain; prefer "
            "teacher_offline -> student_kd -> student_retimed."
        )
        if target_mode != "cached_only":
            errors.append("legacy_schedule_only should use rhythm_dataset_target_mode: cached_only.")
        if primary != "teacher":
            errors.append("legacy_schedule_only should use rhythm_primary_target_surface: teacher.")
        if not require_cached_teacher:
            errors.append("legacy_schedule_only should require cached teacher surfaces.")
        if expected_teacher_source != "learned_offline":
            errors.append("legacy_schedule_only should use rhythm_teacher_target_source: learned_offline.")
        if not legacy_schedule_flag:
            errors.append("legacy_schedule_only should keep rhythm_schedule_only_stage: true.")
        if not optimize_module_only:
            errors.append("legacy_schedule_only should keep rhythm_optimize_module_only: true.")
        if teacher_as_main:
            errors.append("legacy_schedule_only should keep rhythm_teacher_as_main: false.")
        if enable_dual:
            errors.append("legacy_schedule_only should not enable rhythm_enable_dual_mode_teacher.")
        if enable_learned_offline_teacher:
            errors.append(
                "legacy_schedule_only should set rhythm_enable_learned_offline_teacher: false "
                "to avoid unnecessary runtime teacher branch allocation."
            )
        if runtime_offline_teacher_enable:
            errors.append(
                "legacy_schedule_only should keep runtime offline teacher disabled "
                "(rhythm_runtime_enable_learned_offline_teacher should resolve to false)."
            )
        if runtime_dual_mode_teacher_enable:
            errors.append("legacy_schedule_only should not resolve runtime dual-mode teacher execution.")
        if lambda_teacher_aux > 0.0:
            errors.append("legacy_schedule_only should keep lambda_rhythm_teacher_aux: 0.")
        if lambda_distill > 0.0 or distill != "none":
            errors.append("legacy_schedule_only should keep distillation disabled.")
        if apply_train or apply_valid:
            errors.append("legacy_schedule_only should not enable train/valid retimed rendering.")
        if export_debug_sidecars:
            warnings.append("legacy_schedule_only should keep rhythm_export_debug_sidecars: false unless debugging schema.")
        if export_cache_audit_to_sample:
            warnings.append("legacy_schedule_only should keep rhythm_export_cache_audit_to_sample: false unless cache-audit runs.")
    elif stage == "student_kd":
        if target_mode != "cached_only":
            errors.append("student_kd should use rhythm_dataset_target_mode: cached_only.")
        if primary != "teacher":
            errors.append("student_kd should use rhythm_primary_target_surface: teacher.")
        if distill != "cache":
            errors.append("student_kd should use rhythm_distill_surface: cache.")
        if lambda_distill <= 0.0:
            errors.append("student_kd should keep lambda_rhythm_distill > 0.")
        if teacher_as_main:
            errors.append("student_kd should keep rhythm_teacher_as_main: false.")
        if enable_dual:
            errors.append("student_kd should keep rhythm_enable_dual_mode_teacher: false.")
        if enable_learned_offline_teacher:
            errors.append("student_kd should keep rhythm_enable_learned_offline_teacher: false.")
        if runtime_offline_teacher_enable:
            errors.append(
                "student_kd should keep runtime offline teacher disabled "
                "(rhythm_runtime_enable_learned_offline_teacher should resolve false)."
            )
        if runtime_dual_mode_teacher_enable:
            errors.append("student_kd should not resolve runtime dual-mode teacher execution.")
        if not require_cached_teacher:
            errors.append("student_kd should require cached teacher surfaces.")
        if expected_teacher_source != "learned_offline":
            errors.append("student_kd should use rhythm_teacher_target_source: learned_offline.")
        if lambda_guidance > 0.0:
            errors.append("student_kd should keep lambda_rhythm_guidance: 0.")
        if distill_allocation_weight > 0.0:
            errors.append("student_kd should keep rhythm_distill_allocation_weight: 0.")
        if distill_budget_weight > 0.15:
            warnings.append("student_kd usually keeps rhythm_distill_budget_weight <= 0.15.")
        if distill_prefix_weight <= 0.0:
            warnings.append("student_kd usually keeps rhythm_distill_prefix_weight > 0.")
        if lambda_teacher_aux > 0.0:
            errors.append("student_kd should keep lambda_rhythm_teacher_aux: 0.")
        if apply_train or apply_valid:
            errors.append("student_kd should not enable train/valid retimed rendering; that belongs to stage-3.")
        if retimed_target_mode != "cached":
            warnings.append("student_kd usually keeps rhythm_retimed_target_mode: cached (retimed closure starts in stage-3).")
        if export_debug_sidecars:
            warnings.append("student_kd should keep rhythm_export_debug_sidecars: false unless debugging schema.")
        if export_cache_audit_to_sample:
            warnings.append("student_kd should keep rhythm_export_cache_audit_to_sample: false unless cache-audit runs.")
        if not optimize_module_only:
            warnings.append("student_kd usually keeps rhythm_optimize_module_only: true for a short maintained stage-2 path.")
    elif stage == "legacy_dual_mode_kd":
        warnings.append(
            "legacy_dual_mode_kd resolves to a research path. Maintained chain is now "
            "teacher_offline -> student_kd -> student_retimed."
        )
        if teacher_as_main:
            errors.append("legacy_dual_mode_kd should keep rhythm_teacher_as_main: false.")
        if not enable_dual:
            errors.append("legacy_dual_mode_kd requires rhythm_enable_dual_mode_teacher: true.")
        if not enable_learned_offline_teacher:
            errors.append("legacy_dual_mode_kd requires rhythm_enable_learned_offline_teacher: true.")
        if not runtime_offline_teacher_enable:
            errors.append("legacy_dual_mode_kd requires runtime offline teacher branch enabled.")
        if not runtime_dual_mode_teacher_enable:
            errors.append("legacy_dual_mode_kd requires runtime dual-mode teacher execution enabled.")
        if distill != "offline":
            errors.append("legacy_dual_mode_kd requires rhythm_distill_surface: offline.")
        if lambda_teacher_aux <= 0.0:
            warnings.append("legacy_dual_mode_kd usually carries lambda_rhythm_teacher_aux > 0 when kept as a research branch.")
    elif stage == "student_retimed":
        if target_mode != "cached_only":
            errors.append("student_retimed should use rhythm_dataset_target_mode: cached_only.")
        if primary != "teacher":
            errors.append("student_retimed should use rhythm_primary_target_surface: teacher.")
        if expected_teacher_source != "learned_offline":
            errors.append("student_retimed should use rhythm_teacher_target_source: learned_offline.")
        if teacher_as_main:
            errors.append("student_retimed should keep rhythm_teacher_as_main: false.")
        if lambda_distill > 0.0:
            if distill != "cache":
                errors.append("student_retimed with KD enabled should use rhythm_distill_surface: cache.")
            if enable_dual:
                errors.append("student_retimed with KD enabled should keep rhythm_enable_dual_mode_teacher: false.")
            if enable_learned_offline_teacher:
                errors.append(
                    "student_retimed with KD enabled should keep rhythm_enable_learned_offline_teacher: false."
                )
            if runtime_offline_teacher_enable:
                errors.append(
                    "student_retimed with KD enabled should keep runtime offline teacher disabled "
                    "(rhythm_runtime_enable_learned_offline_teacher should resolve false)."
                )
            if runtime_dual_mode_teacher_enable:
                errors.append("student_retimed with KD enabled should not resolve runtime dual-mode teacher execution.")
        else:
            if distill not in {"none", "off", "disable", "disabled", "false"}:
                warnings.append("student_retimed without KD usually keeps rhythm_distill_surface: none.")
            if enable_dual:
                warnings.append("student_retimed without KD usually keeps rhythm_enable_dual_mode_teacher: false.")
            if enable_learned_offline_teacher:
                warnings.append(
                    "student_retimed without KD usually keeps rhythm_enable_learned_offline_teacher: false "
                    "to reduce unused runtime overhead."
                )
            if runtime_offline_teacher_enable:
                warnings.append(
                    "student_retimed without KD usually keeps runtime offline teacher disabled "
                    "to reduce unnecessary branch overhead."
                )
            if runtime_dual_mode_teacher_enable:
                warnings.append(
                    "student_retimed without KD usually keeps runtime dual-mode teacher execution disabled."
                )
        if lambda_teacher_aux > 0.0:
            errors.append("student_retimed should keep lambda_rhythm_teacher_aux: 0.")
        if not require_cached_teacher:
            errors.append("student_retimed should require cached teacher surfaces.")
        if not require_retimed_cache:
            errors.append("student_retimed should require cached retimed mel targets.")
        if not use_retimed_target:
            errors.append("student_retimed requires rhythm_use_retimed_target_if_available: true.")
        if not apply_train or not apply_valid:
            errors.append("student_retimed should enable both train/valid retimed rendering.")
        if retimed_target_start > 0:
            errors.append(
                "student_retimed should set rhythm_retimed_target_start_steps: 0 "
                "for immediate train/infer closure on the retimed canvas."
            )
        if legacy_schedule_flag:
            errors.append("student_retimed should set rhythm_schedule_only_stage: false.")
        if optimize_module_only:
            errors.append("student_retimed should set rhythm_optimize_module_only: false.")
        if not compact_joint_loss:
            errors.append("student_retimed should keep rhythm_compact_joint_loss: true.")
        if retimed_source != "teacher":
            errors.append("student_retimed should use rhythm_binarize_retimed_mel_source: teacher.")
        if not binarize_teacher:
            errors.append("student_retimed requires rhythm_binarize_teacher_targets: true.")
        if not binarize_retimed:
            errors.append("student_retimed requires rhythm_binarize_retimed_mel_targets: true.")
        if lambda_guidance > 0.0:
            errors.append("student_retimed should keep lambda_rhythm_guidance: 0.")
        if distill_allocation_weight > 0.0:
            errors.append("student_retimed should keep rhythm_distill_allocation_weight: 0.")
        if distill_budget_weight > 0.15:
            warnings.append("student_retimed usually keeps rhythm_distill_budget_weight <= 0.15.")
        if distill_prefix_weight <= 0.0 and lambda_distill > 0.0:
            warnings.append("student_retimed usually keeps rhythm_distill_prefix_weight > 0.")
        if not disable_mel_adv_when_retimed:
            errors.append("student_retimed should keep rhythm_disable_mel_adv_when_retimed: true.")
        if lambda_mel_adv > 0.0:
            errors.append("student_retimed should keep lambda_mel_adv: 0.0 on the retimed canvas.")
        if export_debug_sidecars:
            warnings.append("student_retimed should keep rhythm_export_debug_sidecars: false to preserve runtime-minimal batch contract.")
        if export_cache_audit_to_sample:
            warnings.append("student_retimed should keep rhythm_export_cache_audit_to_sample: false except dedicated cache audits.")
        if retimed_target_mode == "online":
            warnings.append("student_retimed uses online-only retimed targets; cached retimed targets will not act as a warm-start fallback.")
    else:
        if profile != "minimal_v1":
            warnings.append(
                "This config resolves to a transitional/prefer_cache path, not the maintained formal chain "
                "(teacher_offline -> student_kd -> student_retimed, with legacy_schedule_only and legacy_dual_mode_kd "
                "kept only as legacy research branches)."
            )
        if (
            not enable_dual
            and lambda_distill <= 0.0
            and (enable_learned_offline_teacher or runtime_offline_teacher_enable)
        ):
            warnings.append(
                "learned offline teacher is enabled but neither dual-mode KD nor distillation is active; "
                "consider disabling rhythm_enable_learned_offline_teacher / runtime teacher branch for this run."
            )
    return stage, errors, warnings


def _expected_fields(hp: dict) -> tuple[list[tuple[str, ...]], list[str], list[str]]:
    expected_groups: list[tuple[str, ...]] = [(key,) for key in sorted(CORE_RHYTHM_FIELDS)]
    errors: list[str] = []
    warnings: list[str] = []

    primary = _normalize_surface(str(hp.get("rhythm_primary_target_surface", "guidance") or "guidance").strip().lower())
    distill = _normalize_distill(str(hp.get("rhythm_distill_surface", "auto") or "auto").strip().lower())
    cached_only = str(hp.get("rhythm_dataset_target_mode", "prefer_cache") or "prefer_cache").strip().lower() == "cached_only"

    if cached_only:
        expected_groups.extend(CACHED_ONLY_META_FIELD_GROUPS)
    if primary == "guidance":
        expected_groups.extend(GUIDANCE_FIELD_GROUPS)
    if (
        primary == "teacher"
        or bool(hp.get("rhythm_require_cached_teacher", False))
        or bool(hp.get("rhythm_binarize_teacher_targets", False))
        or str(hp.get("rhythm_binarize_retimed_mel_source", "guidance") or "guidance").strip().lower() == "teacher"
    ):
        expected_groups.extend(TEACHER_FIELD_GROUPS)
    if (
        bool(hp.get("rhythm_require_retimed_cache", False))
        or bool(hp.get("rhythm_apply_train_override", False))
        or bool(hp.get("rhythm_apply_valid_override", False))
    ):
        expected_groups.extend(RETIMED_FIELD_GROUPS)

    if float(hp.get("lambda_rhythm_distill", 0.0)) > 0.0:
        if distill == "none":
            errors.append("lambda_rhythm_distill > 0 but rhythm_distill_surface disables distillation.")
        if distill == "offline" and not bool(hp.get("rhythm_enable_dual_mode_teacher", False)):
            errors.append("Offline distillation requires rhythm_enable_dual_mode_teacher: true.")
        if distill == "cache":
            expected_groups.extend(TEACHER_FIELD_GROUPS)

    if bool(hp.get("rhythm_apply_train_override", False)) and not bool(hp.get("rhythm_use_retimed_target_if_available", False)):
        errors.append("Train-time retimed rendering requires rhythm_use_retimed_target_if_available: true.")

    if bool(hp.get("rhythm_schedule_only_stage", False)) and bool(hp.get("rhythm_apply_train_override", False)):
        errors.append("Schedule-only stage should not enable train-time retimed rendering.")

    if cached_only and int(hp.get("rhythm_cache_version", -1)) <= 0:
        errors.append("cached_only requires a positive rhythm_cache_version.")

    if primary == "teacher" and not bool(hp.get("rhythm_binarize_teacher_targets", False)):
        warnings.append("Primary surface is teacher but rhythm_binarize_teacher_targets is false.")

    dedup = []
    seen = set()
    for group in expected_groups:
        if group in seen:
            continue
        seen.add(group)
        dedup.append(group)
    return dedup, errors, warnings


def _open_dataset(path_prefix: str):
    idx_path = f"{path_prefix}.idx"
    data_path = f"{path_prefix}.data"
    if not os.path.exists(idx_path) or not os.path.exists(data_path):
        return None
    return IndexedDataset(path_prefix)


def _collect_presence(ds: IndexedDataset, limit: int) -> tuple[list[dict], Counter, list[str]]:
    items: list[dict] = []
    counts: Counter[str] = Counter()
    mismatches: list[str] = []
    num_items = min(len(ds), max(1, limit))
    for idx in range(num_items):
        item = materialize_rhythm_cache_compat_fields(ds[idx])
        items.append(item)
        for key in item.keys():
            counts[key] += 1
        if "rhythm_cache_version" in item:
            try:
                version = int(item["rhythm_cache_version"][0]) if hasattr(item["rhythm_cache_version"], "__len__") else int(item["rhythm_cache_version"])
                counts[f"__cache_version__{version}"] += 1
            except Exception:
                mismatches.append(f"item[{idx}] has unreadable rhythm_cache_version")
    return items, counts, mismatches


def _validate_inspected_items(items: list[dict], hp: dict, *, split: str) -> list[str]:
    errors: list[str] = []
    cached_only = str(hp.get("rhythm_dataset_target_mode", "prefer_cache") or "prefer_cache").strip().lower() == "cached_only"
    expected_teacher_surface = _resolve_expected_teacher_surface(hp)
    expected_teacher_source_id = _resolve_expected_teacher_target_source_id(hp)
    need_teacher = (
        _normalize_surface(str(hp.get("rhythm_primary_target_surface", "guidance") or "guidance").strip().lower()) == "teacher"
        or bool(hp.get("rhythm_require_cached_teacher", False))
        or str(hp.get("rhythm_binarize_retimed_mel_source", "guidance") or "guidance").strip().lower() == "teacher"
    )
    need_retimed = bool(hp.get("rhythm_require_retimed_cache", False)) or bool(hp.get("rhythm_apply_train_override", False)) or bool(hp.get("rhythm_apply_valid_override", False))
    expected_meta = _expected_cache_contract(hp)
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
            if "rhythm_retimed_target_source_id" not in item:
                errors.append(f"{item_name}: missing rhythm_retimed_target_source_id for retimed training.")
            else:
                retimed_source = str(hp.get("rhythm_binarize_retimed_mel_source", "guidance") or "guidance").strip().lower()
                expected_source_id = RHYTHM_RETIMED_SOURCE_TEACHER if retimed_source == "teacher" else RHYTHM_RETIMED_SOURCE_GUIDANCE
                found_source_id = int(_extract_scalar(item["rhythm_retimed_target_source_id"]))
                if found_source_id != expected_source_id:
                    errors.append(
                        f"{item_name}: rhythm_retimed_target_source_id mismatch, found={found_source_id}, expected={expected_source_id}."
                    )
            if "rhythm_retimed_target_surface_name" not in item:
                errors.append(f"{item_name}: missing rhythm_retimed_target_surface_name for retimed training.")
            else:
                expected_surface = expected_teacher_surface if str(hp.get("rhythm_binarize_retimed_mel_source", "guidance") or "guidance").strip().lower() == "teacher" else RHYTHM_GUIDANCE_SURFACE_NAME
                found_surface = str(_extract_scalar(item["rhythm_retimed_target_surface_name"]))
                if found_surface != expected_surface:
                    errors.append(
                        f"{item_name}: rhythm_retimed_target_surface_name mismatch, found={found_surface}, expected={expected_surface}."
                    )
    return errors


def _run_dataset_and_model_dry_run(split: str, *, run_model: bool, hp: dict) -> list[str]:
    errors: list[str] = []
    stage = detect_rhythm_stage(hp)
    try:
        from tasks.Conan.dataset import ConanDataset
    except Exception as exc:
        return [f"Failed to import ConanDataset for split '{split}': {exc}"]
    try:
        ds = ConanDataset(prefix=split, shuffle=False)
    except Exception as exc:
        return [f"Failed to build ConanDataset for split '{split}': {exc}"]
    filtered_len = len(ds)
    print(f"[preflight] dataset_split={split} filtered_items={filtered_len}")
    if filtered_len <= 0:
        return [f"Split '{split}' is empty after ConanDataset filtering."]
    try:
        batch = ds.collater([ds[0]])
    except Exception as exc:
        return [f"Failed to collate split '{split}' sample: {exc}"]
    if not run_model:
        return errors
    try:
        import torch
        from tasks.Conan.Conan import ConanTask
        task = ConanTask()
        task.build_tts_model()
        task.global_step = 0
        with torch.no_grad():
            losses, output = task.run_model(batch, infer=False)
        if "mel_out" not in output:
            errors.append(f"Model dry-run for split '{split}' did not produce mel_out.")
        if "rhythm_execution" not in output:
            errors.append(f"Model dry-run for split '{split}' did not produce rhythm_execution.")
        runtime_teacher_enabled = resolve_policy_runtime_offline_teacher_enable(hp, stage=stage)
        dual_mode_enabled = resolve_runtime_dual_mode_teacher_enable(
            hp,
            stage=stage,
            infer=False,
        )
        teacher_as_main = resolve_teacher_as_main(hp, stage=stage, infer=False)
        reported_stage = str(output.get("rhythm_stage", "") or "").strip().lower()
        if reported_stage and reported_stage != stage:
            errors.append(
                f"Model dry-run for split '{split}' reported rhythm_stage={reported_stage}, expected {stage}."
            )
        offline_execution = output.get("rhythm_offline_execution")
        teacher_as_main_runtime = bool(output.get("rhythm_teacher_as_main", 0.0))
        if teacher_as_main and not teacher_as_main_runtime:
            errors.append(
                f"Model dry-run for split '{split}' expected rhythm_teacher_as_main=true in stage {stage}."
            )
        if not teacher_as_main and teacher_as_main_runtime:
            errors.append(
                f"Model dry-run for split '{split}' unexpectedly reported rhythm_teacher_as_main in stage {stage}."
            )
        if stage == "teacher_offline" and offline_execution is not None:
            errors.append(
                f"Model dry-run for split '{split}' should not emit rhythm_offline_execution in teacher_offline stage."
            )
        if not dual_mode_enabled and offline_execution is not None:
            errors.append(
                f"Model dry-run for split '{split}' unexpectedly produced rhythm_offline_execution while "
                "runtime dual-mode teacher resolves disabled."
            )
        if dual_mode_enabled and offline_execution is None:
            errors.append(
                f"Model dry-run for split '{split}' expected rhythm_offline_execution in dual-mode stage but found none."
            )
        if not runtime_teacher_enabled and offline_execution is not None:
            errors.append(
                f"Model dry-run for split '{split}' unexpectedly produced rhythm_offline_execution while "
                "runtime offline teacher branch resolves disabled."
            )
        if stage == "legacy_schedule_only" and offline_execution is not None:
            errors.append(
                f"Model dry-run for split '{split}' unexpectedly produced rhythm_offline_execution in schedule-only stage."
            )
        apply_split = bool(hp.get("rhythm_apply_train_override", False)) if split == "train" else bool(hp.get("rhythm_apply_valid_override", False))
        retimed_start = int(hp.get("rhythm_retimed_target_start_steps", 0) or 0)
        if apply_split and retimed_start <= 0:
            if not bool(output.get("acoustic_target_is_retimed", False)):
                errors.append(
                    f"Model dry-run for split '{split}' expected retimed acoustic target but got source-aligned target."
                )
            acoustic_target_source = str(output.get("acoustic_target_source", "") or "").strip().lower()
            if not acoustic_target_source:
                errors.append(
                    f"Model dry-run for split '{split}' expected acoustic_target_source when retimed stage is active."
                )
            elif acoustic_target_source in {"source", "source_aligned"}:
                errors.append(
                    f"Model dry-run for split '{split}' got acoustic_target_source={acoustic_target_source}, "
                    "but retimed stage expects cached/online/hybrid retimed target routing."
                )
        print(f"[preflight] model_dry_run split={split} mel_out={tuple(output['mel_out'].shape)} "
              f"units={tuple(output['speech_duration_exec'].shape) if 'speech_duration_exec' in output else 'n/a'} "
              f"loss_keys={sorted(losses.keys())[:8]}")
    except Exception as exc:
        errors.append(f"Model dry-run failed for split '{split}': {exc}")
    return errors


def main():
    parser = argparse.ArgumentParser(description="Preflight check for Rhythm V2 staged training.")
    parser.add_argument("--config", required=True, help="YAML config to validate.")
    parser.add_argument("--exp_name", default="", help="Optional temporary exp name for hparams loading.")
    parser.add_argument("--hparams", default="", help="Extra hparam overrides.")
    parser.add_argument("--binary_data_dir", default="", help="Optional override for binary_data_dir.")
    parser.add_argument("--inspect_items", type=int, default=8, help="How many items to inspect per split.")
    parser.add_argument("--splits", nargs="*", default=["train", "valid"], help="Dataset prefixes to inspect.")
    parser.add_argument("--model_dry_run", action="store_true", help="Also build ConanDataset/ConanTask and run one no-grad batch.")
    args = parser.parse_args()

    hparams_str = args.hparams
    if args.binary_data_dir:
        override = f"binary_data_dir='{args.binary_data_dir}'"
        hparams_str = override if not hparams_str else f"{hparams_str},{override}"
    hp = set_hparams(config=args.config, exp_name=args.exp_name, hparams_str=hparams_str, global_hparams=True, print_hparams=False)

    required_groups, errors, warnings = _expected_fields(hp)
    contract_result = collect_rhythm_contract_issues(
        hp,
        config_path=args.config,
        model_dry_run=args.model_dry_run,
    )
    profile = contract_result.profile
    stage = contract_result.stage
    errors.extend(contract_result.errors)
    warnings.extend(contract_result.warnings)
    binary_dir = hp.get("binary_data_dir", "")
    if not binary_dir:
        errors.append("binary_data_dir is empty.")
    elif not os.path.isdir(binary_dir):
        errors.append(f"binary_data_dir does not exist: {binary_dir}")

    print(f"[preflight] config={args.config}")
    print(f"[preflight] profile={profile}")
    print(f"[preflight] stage={stage}")
    print(f"[preflight] binary_data_dir={binary_dir}")
    printable_groups = [" | ".join(group) for group in required_groups]
    print(f"[preflight] required_field_group_count={len(required_groups)}")
    print(f"[preflight] required_field_groups={printable_groups}")

    for split in args.splits:
        split_path = os.path.join(binary_dir, split)
        ds = _open_dataset(split_path)
        if ds is None:
            errors.append(f"Missing indexed dataset for split '{split}' at {split_path}.data/.idx")
            continue
        if len(ds) <= 0:
            errors.append(f"Indexed dataset for split '{split}' is empty at {split_path}.")
            continue
        items, counts, mismatches = _collect_presence(ds, args.inspect_items)
        inspected = min(len(ds), args.inspect_items)
        print(f"[preflight] split={split} items={len(ds)} inspected={inspected}")
        for group in required_groups:
            have = 0
            for item in items:
                if any(key in item for key in group):
                    have += 1
            label = " | ".join(group)
            print(f"  - {label}: {have}/{inspected}")
            if have < inspected:
                errors.append(f"Split '{split}' is missing required field group '{label}' in {inspected - have} inspected items.")
        expected_version = int(hp.get("rhythm_cache_version", -1))
        seen_versions = [name for name in counts if name.startswith("__cache_version__")]
        if seen_versions:
            print(f"  - cache_versions_seen={[name.replace('__cache_version__', '') for name in seen_versions]}")
            compatible_versions = compatible_rhythm_cache_versions(expected_version)
            compatible_seen = sum(counts.get(f"__cache_version__{version}", 0) for version in compatible_versions)
            if compatible_seen < min(len(ds), args.inspect_items):
                errors.append(
                    f"Split '{split}' has cache version mismatch against expected rhythm_cache_version={expected_version} "
                    f"(accepted compatible versions: {compatible_versions})."
                )
        errors.extend(_validate_inspected_items(items, hp, split=split))
        for mismatch in mismatches:
            errors.append(mismatch)
        if args.model_dry_run:
            errors.extend(_run_dataset_and_model_dry_run(split, run_model=True, hp=hp))

    if warnings:
        print("[preflight] warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    if errors:
        print("[preflight] FAILED")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    print("[preflight] OK")


if __name__ == "__main__":
    main()
