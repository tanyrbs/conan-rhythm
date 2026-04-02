from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from modules.Conan.rhythm.supervision import (
    RHYTHM_CACHE_VERSION,
    RHYTHM_GUIDANCE_SURFACE_NAME,
    RHYTHM_REFERENCE_MODE_STATIC_REF_FULL,
    RHYTHM_RETIMED_SOURCE_GUIDANCE,
    RHYTHM_RETIMED_SOURCE_TEACHER,
    RHYTHM_TRACE_HOP_MS,
    RHYTHM_UNIT_HOP_MS,
    normalize_teacher_target_source,
    resolve_teacher_surface_name,
    resolve_teacher_target_source_id,
)
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
    aliases = {
        "cache_teacher": "teacher",
        "offline": "teacher",
        "offline_teacher": "teacher",
        "teacher_surface": "teacher",
        "guidance_surface": "guidance",
        "self": "guidance",
    }
    return aliases.get(surface, surface)


def _normalize_distill(surface: str) -> str:
    aliases = {
        "off": "none",
        "disable": "none",
        "disabled": "none",
        "false": "none",
        "cache_teacher": "cache",
        "cached_teacher": "cache",
        "full_context": "offline",
        "shared_offline": "offline",
        "algo": "algorithmic",
        "teacher": "cache",
    }
    return aliases.get(surface, surface)


def _normalize_retimed_target_mode(mode: str) -> str:
    aliases = {
        "cache": "cached",
        "cached_only": "cached",
        "teacher": "cached",
        "runtime": "online",
        "online_only": "online",
        "mixed": "hybrid",
    }
    return aliases.get(mode, mode)


def _resolve_runtime_offline_teacher_enable(hp: dict) -> bool:
    """Mirror factory runtime-teacher resolution to catch config/runtime mismatches in preflight."""
    explicit_runtime = hp.get("rhythm_runtime_enable_learned_offline_teacher", None)
    if explicit_runtime is not None:
        return bool(explicit_runtime)

    return bool(hp.get("rhythm_enable_dual_mode_teacher", False)) and bool(
        hp.get("rhythm_enable_learned_offline_teacher", False)
    )


def _extract_scalar(value):
    arr = np.asarray(value)
    if arr.size <= 0:
        raise RuntimeError("Encountered empty scalar field while validating rhythm cache metadata.")
    return arr.reshape(-1)[0]


def _expected_cache_contract(hp: dict) -> dict[str, int | float]:
    return {
        "rhythm_cache_version": int(hp.get("rhythm_cache_version", RHYTHM_CACHE_VERSION)),
        "rhythm_unit_hop_ms": int(hp.get("rhythm_unit_hop_ms", RHYTHM_UNIT_HOP_MS)),
        "rhythm_trace_hop_ms": int(hp.get("rhythm_trace_hop_ms", RHYTHM_TRACE_HOP_MS)),
        "rhythm_trace_bins": int(hp.get("rhythm_trace_bins", 24)),
        "rhythm_trace_horizon": float(hp.get("rhythm_trace_horizon", 0.35)),
        "rhythm_slow_topk": int(hp.get("rhythm_slow_topk", 6)),
        "rhythm_selector_cell_size": int(hp.get("rhythm_selector_cell_size", 3)),
        "rhythm_source_phrase_threshold": float(hp.get("rhythm_source_phrase_threshold", 0.55)),
        "rhythm_reference_mode_id": int(
            hp.get("rhythm_reference_mode_id", RHYTHM_REFERENCE_MODE_STATIC_REF_FULL)
        ),
    }


def _resolve_expected_teacher_target_source(hp: dict) -> str:
    return normalize_teacher_target_source(hp.get("rhythm_teacher_target_source", "algorithmic"))


def _resolve_expected_teacher_surface(hp: dict) -> str:
    return resolve_teacher_surface_name(_resolve_expected_teacher_target_source(hp))


def _resolve_expected_teacher_target_source_id(hp: dict) -> int:
    return resolve_teacher_target_source_id(_resolve_expected_teacher_target_source(hp))


def _detect_stage(hp: dict, config_path: str) -> str:
    distill = _normalize_distill(str(hp.get("rhythm_distill_surface", "auto") or "auto").strip().lower())
    config_name = str(config_path).lower()
    if (
        bool(hp.get("rhythm_apply_train_override", False))
        or bool(hp.get("rhythm_apply_valid_override", False))
        or bool(hp.get("rhythm_require_retimed_cache", False))
        or "retimed_train" in config_name
    ):
        return "retimed_train"
    if (
        "teacher_student_kd" in config_name
        or (
            float(hp.get("lambda_rhythm_distill", 0.0)) > 0.0
            and distill == "cache"
            and not bool(hp.get("rhythm_enable_dual_mode_teacher", False))
        )
    ):
        return "teacher_student_kd"
    if (
        bool(hp.get("rhythm_enable_dual_mode_teacher", False))
        or distill in {"offline", "algorithmic"}
        or "dual_mode_kd" in config_name
    ):
        return "dual_mode_kd"
    if (
        bool(hp.get("rhythm_schedule_only_stage", False))
        or bool(hp.get("rhythm_optimize_module_only", False))
        or "schedule_only" in config_name
    ):
        return "schedule_only"
    return "transitional"


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
    if "lambda_rhythm_cumplan" in hp:
        return float(hp.get("lambda_rhythm_cumplan", 0.0))
    return float(hp.get("lambda_rhythm_carry", 0.0))


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

    target_mode = str(hp.get("rhythm_dataset_target_mode", "prefer_cache") or "prefer_cache").strip().lower()
    primary = _normalize_surface(str(hp.get("rhythm_primary_target_surface", "guidance") or "guidance").strip().lower())
    distill = _normalize_distill(str(hp.get("rhythm_distill_surface", "auto") or "auto").strip().lower())
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
    stage = _detect_stage(hp, config_path)
    profile = _detect_profile(hp, config_path)
    target_mode = str(hp.get("rhythm_dataset_target_mode", "prefer_cache") or "prefer_cache").strip().lower()
    primary = _normalize_surface(str(hp.get("rhythm_primary_target_surface", "guidance") or "guidance").strip().lower())
    distill = _normalize_distill(str(hp.get("rhythm_distill_surface", "auto") or "auto").strip().lower())
    require_cached_teacher = bool(hp.get("rhythm_require_cached_teacher", False))
    require_retimed_cache = bool(hp.get("rhythm_require_retimed_cache", False))
    enable_dual = bool(hp.get("rhythm_enable_dual_mode_teacher", False))
    enable_learned_offline_teacher = bool(hp.get("rhythm_enable_learned_offline_teacher", True))
    explicit_runtime_teacher_enable = hp.get("rhythm_runtime_enable_learned_offline_teacher", None)
    runtime_offline_teacher_enable = _resolve_runtime_offline_teacher_enable(hp)
    expected_teacher_source = _resolve_expected_teacher_target_source(hp)
    expected_teacher_surface = _resolve_expected_teacher_surface(hp)
    schedule_only = bool(hp.get("rhythm_schedule_only_stage", False))
    optimize_module_only = bool(hp.get("rhythm_optimize_module_only", False))
    lambda_distill = float(hp.get("lambda_rhythm_distill", 0.0))
    lambda_teacher_aux = float(hp.get("lambda_rhythm_teacher_aux", 0.0) or 0.0)
    apply_train = bool(hp.get("rhythm_apply_train_override", False))
    apply_valid = bool(hp.get("rhythm_apply_valid_override", False))
    use_retimed_target = bool(hp.get("rhythm_use_retimed_target_if_available", False))
    retimed_target_mode = _normalize_retimed_target_mode(str(hp.get("rhythm_retimed_target_mode", "cached") or "cached").strip().lower())
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

    if stage == "schedule_only":
        if target_mode != "cached_only":
            errors.append("Formal schedule-only stage should use rhythm_dataset_target_mode: cached_only.")
        if primary != "teacher":
            errors.append("Formal schedule-only stage should use rhythm_primary_target_surface: teacher.")
        if not require_cached_teacher:
            errors.append("Formal schedule-only stage should require cached teacher surfaces.")
        if expected_teacher_source != "learned_offline":
            errors.append("Formal schedule-only stage should use rhythm_teacher_target_source: learned_offline.")
        if not schedule_only:
            errors.append("Formal schedule-only stage should keep rhythm_schedule_only_stage: true.")
        if not optimize_module_only:
            errors.append("Formal schedule-only stage should keep rhythm_optimize_module_only: true.")
        if enable_dual:
            errors.append("Stage-1 schedule-only should not enable rhythm_enable_dual_mode_teacher.")
        if enable_learned_offline_teacher:
            errors.append(
                "Stage-1 schedule-only should set rhythm_enable_learned_offline_teacher: false "
                "to avoid unnecessary runtime teacher branch allocation."
            )
        if runtime_offline_teacher_enable:
            errors.append(
                "Stage-1 schedule-only should keep runtime offline teacher disabled "
                "(rhythm_runtime_enable_learned_offline_teacher should resolve to false)."
            )
        if lambda_teacher_aux > 0.0:
            errors.append("Stage-1 schedule-only should keep lambda_rhythm_teacher_aux: 0.")
        if lambda_distill > 0.0 or distill != "none":
            errors.append("Stage-1 schedule-only should keep distillation disabled.")
        if apply_train or apply_valid:
            errors.append("Stage-1 schedule-only should not enable train/valid retimed rendering.")
        if export_debug_sidecars:
            warnings.append("Stage-1 schedule-only should keep rhythm_export_debug_sidecars: false unless debugging schema.")
        if export_cache_audit_to_sample:
            warnings.append("Stage-1 schedule-only should keep rhythm_export_cache_audit_to_sample: false unless cache-audit runs.")
    elif stage == "teacher_student_kd":
        if target_mode != "cached_only":
            errors.append("Formal teacher->student KD stage should use rhythm_dataset_target_mode: cached_only.")
        if primary != "teacher":
            errors.append("Formal teacher->student KD stage should use rhythm_primary_target_surface: teacher.")
        if distill != "cache":
            errors.append("Formal teacher->student KD stage should use rhythm_distill_surface: cache.")
        if lambda_distill <= 0.0:
            errors.append("Formal teacher->student KD stage should keep lambda_rhythm_distill > 0.")
        if enable_dual:
            errors.append("Formal teacher->student KD stage should keep rhythm_enable_dual_mode_teacher: false.")
        if enable_learned_offline_teacher:
            errors.append("Formal teacher->student KD stage should keep rhythm_enable_learned_offline_teacher: false.")
        if runtime_offline_teacher_enable:
            errors.append(
                "Formal teacher->student KD stage should keep runtime offline teacher disabled "
                "(rhythm_runtime_enable_learned_offline_teacher should resolve false)."
            )
        if not require_cached_teacher:
            errors.append("Formal teacher->student KD stage should require cached teacher surfaces.")
        if expected_teacher_source != "learned_offline":
            errors.append("Formal teacher->student KD stage should use rhythm_teacher_target_source: learned_offline.")
        if lambda_guidance > 0.0:
            errors.append("Formal teacher->student KD stage should keep lambda_rhythm_guidance: 0.")
        if distill_allocation_weight > 0.0:
            errors.append("Formal teacher->student KD stage should keep rhythm_distill_allocation_weight: 0.")
        if distill_budget_weight > 0.15:
            warnings.append("Formal teacher->student KD stage usually keeps rhythm_distill_budget_weight <= 0.15.")
        if distill_prefix_weight <= 0.0:
            warnings.append("Formal teacher->student KD stage usually keeps rhythm_distill_prefix_weight > 0.")
        if lambda_teacher_aux > 0.0:
            errors.append("Formal teacher->student KD stage should keep lambda_rhythm_teacher_aux: 0.")
        if apply_train or apply_valid:
            errors.append("Teacher->student KD stage should not enable train/valid retimed rendering; that belongs to stage-3.")
        if retimed_target_mode != "cached":
            warnings.append("Teacher->student KD stage usually keeps rhythm_retimed_target_mode: cached (retimed closure starts in stage-3).")
        if export_debug_sidecars:
            warnings.append("Teacher->student KD stage should keep rhythm_export_debug_sidecars: false unless debugging schema.")
        if export_cache_audit_to_sample:
            warnings.append("Teacher->student KD stage should keep rhythm_export_cache_audit_to_sample: false unless cache-audit runs.")
        if not optimize_module_only:
            warnings.append("Teacher->student KD stage usually keeps rhythm_optimize_module_only: true for a short maintained stage-2 path.")
    elif stage == "dual_mode_kd":
        warnings.append(
            "dual_mode_kd resolves to a legacy research path. Maintained chain is now schedule_only -> teacher_student_kd -> retimed_train."
        )
        if not enable_dual:
            errors.append("Legacy dual-mode KD stage requires rhythm_enable_dual_mode_teacher: true.")
        if not enable_learned_offline_teacher:
            errors.append("Legacy dual-mode KD stage requires rhythm_enable_learned_offline_teacher: true.")
        if not runtime_offline_teacher_enable:
            errors.append("Legacy dual-mode KD stage requires runtime offline teacher branch enabled.")
        if distill != "offline":
            errors.append("Legacy dual-mode KD stage requires rhythm_distill_surface: offline.")
        if lambda_teacher_aux <= 0.0:
            warnings.append("Legacy dual-mode KD stage usually carries lambda_rhythm_teacher_aux > 0 when kept as a research branch.")
    elif stage == "retimed_train":
        if target_mode != "cached_only":
            errors.append("Formal retimed-train stage should use rhythm_dataset_target_mode: cached_only.")
        if primary != "teacher":
            errors.append("Formal retimed-train stage should use rhythm_primary_target_surface: teacher.")
        if expected_teacher_source != "learned_offline":
            errors.append("Formal retimed-train stage should use rhythm_teacher_target_source: learned_offline.")
        if lambda_distill > 0.0:
            if distill != "cache":
                errors.append("Formal retimed-train stage with KD enabled should use rhythm_distill_surface: cache.")
            if enable_dual:
                errors.append("Formal retimed-train stage with KD enabled should keep rhythm_enable_dual_mode_teacher: false.")
            if enable_learned_offline_teacher:
                errors.append(
                    "Formal retimed-train stage with KD enabled should keep rhythm_enable_learned_offline_teacher: false."
                )
            if runtime_offline_teacher_enable:
                errors.append(
                    "Formal retimed-train stage with KD enabled should keep runtime offline teacher disabled "
                    "(rhythm_runtime_enable_learned_offline_teacher should resolve false)."
                )
        else:
            if distill not in {"none", "off", "disable", "disabled", "false"}:
                warnings.append("Formal retimed-train stage without KD usually keeps rhythm_distill_surface: none.")
            if enable_dual:
                warnings.append("Formal retimed-train stage without KD usually keeps rhythm_enable_dual_mode_teacher: false.")
            if enable_learned_offline_teacher:
                warnings.append(
                    "Formal retimed-train stage without KD usually keeps rhythm_enable_learned_offline_teacher: false "
                    "to reduce unused runtime overhead."
                )
            if runtime_offline_teacher_enable:
                warnings.append(
                    "Formal retimed-train stage without KD usually keeps runtime offline teacher disabled "
                    "to reduce unnecessary branch overhead."
                )
        if lambda_teacher_aux > 0.0:
            errors.append("Formal retimed-train stage should keep lambda_rhythm_teacher_aux: 0.")
        if not require_cached_teacher:
            errors.append("Formal retimed-train stage should require cached teacher surfaces.")
        if not require_retimed_cache:
            errors.append("Formal retimed-train stage should require cached retimed mel targets.")
        if not use_retimed_target:
            errors.append("Formal retimed-train stage requires rhythm_use_retimed_target_if_available: true.")
        if not apply_train or not apply_valid:
            errors.append("Formal retimed-train stage should enable both train/valid retimed rendering.")
        if retimed_target_start > 0:
            errors.append(
                "Formal retimed-train stage should set rhythm_retimed_target_start_steps: 0 "
                "for immediate train/infer closure on the retimed canvas."
            )
        if schedule_only:
            errors.append("Formal retimed-train stage should set rhythm_schedule_only_stage: false.")
        if optimize_module_only:
            errors.append("Formal retimed-train stage should set rhythm_optimize_module_only: false.")
        if not compact_joint_loss:
            errors.append("Formal retimed-train stage should keep rhythm_compact_joint_loss: true.")
        if retimed_source != "teacher":
            errors.append("Formal retimed-train stage should use rhythm_binarize_retimed_mel_source: teacher.")
        if not binarize_teacher:
            errors.append("Formal retimed-train stage requires rhythm_binarize_teacher_targets: true.")
        if not binarize_retimed:
            errors.append("Formal retimed-train stage requires rhythm_binarize_retimed_mel_targets: true.")
        if lambda_guidance > 0.0:
            errors.append("Formal retimed-train stage should keep lambda_rhythm_guidance: 0.")
        if distill_allocation_weight > 0.0:
            errors.append("Formal retimed-train stage should keep rhythm_distill_allocation_weight: 0.")
        if distill_budget_weight > 0.15:
            warnings.append("Formal retimed-train stage usually keeps rhythm_distill_budget_weight <= 0.15.")
        if distill_prefix_weight <= 0.0 and lambda_distill > 0.0:
            warnings.append("Formal retimed-train stage usually keeps rhythm_distill_prefix_weight > 0.")
        if not disable_mel_adv_when_retimed:
            errors.append("Formal retimed-train stage should keep rhythm_disable_mel_adv_when_retimed: true.")
        if lambda_mel_adv > 0.0:
            errors.append("Formal retimed-train stage should keep lambda_mel_adv: 0.0 on the retimed canvas.")
        if export_debug_sidecars:
            warnings.append("Formal retimed-train stage should keep rhythm_export_debug_sidecars: false to preserve runtime-minimal batch contract.")
        if export_cache_audit_to_sample:
            warnings.append("Formal retimed-train stage should keep rhythm_export_cache_audit_to_sample: false except dedicated cache audits.")
        if retimed_target_mode == "online":
            warnings.append("Formal retimed-train stage uses online-only retimed targets; cached retimed targets will not act as a warm-start fallback.")
    else:
        if profile != "minimal_v1":
            warnings.append(
                "This config resolves to a transitional/prefer_cache path, not the maintained formal chain "
                "(schedule_only -> teacher_student_kd -> retimed_train, with dual_mode_kd kept only as a legacy research branch)."
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
        item = ds[idx]
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
        runtime_teacher_enabled = _resolve_runtime_offline_teacher_enable(hp)
        dual_mode_enabled = bool(hp.get("rhythm_enable_dual_mode_teacher", False))
        offline_execution = output.get("rhythm_offline_execution")
        if not dual_mode_enabled and offline_execution is not None:
            errors.append(
                f"Model dry-run for split '{split}' unexpectedly produced rhythm_offline_execution while "
                "rhythm_enable_dual_mode_teacher is false."
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
        if bool(hp.get("rhythm_schedule_only_stage", False)) and offline_execution is not None:
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
    profile, profile_errors, profile_warnings = _validate_profile_contract(
        hp,
        config_path=args.config,
        model_dry_run=args.model_dry_run,
    )
    errors.extend(profile_errors)
    warnings.extend(profile_warnings)
    stage, stage_errors, stage_warnings = _validate_stage_contract(hp, config_path=args.config)
    errors.extend(stage_errors)
    warnings.extend(stage_warnings)
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
            if counts.get(f"__cache_version__{expected_version}", 0) < min(len(ds), args.inspect_items):
                errors.append(
                    f"Split '{split}' has cache version mismatch against expected rhythm_cache_version={expected_version}."
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
