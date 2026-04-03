from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from modules.Conan.rhythm.policy import (
    build_rhythm_hparams_policy,
    expected_cache_contract as build_expected_cache_contract,
    normalize_distill_surface,
    normalize_primary_target_surface,
    resolve_cumplan_lambda,
)
from modules.Conan.rhythm.supervision import (
    RHYTHM_CACHE_VERSION,
    RHYTHM_GUIDANCE_SURFACE_NAME,
    RHYTHM_RETIMED_SOURCE_GUIDANCE,
    RHYTHM_RETIMED_SOURCE_TEACHER,
    compatible_rhythm_cache_versions,
    is_rhythm_cache_version_compatible,
)
from modules.Conan.rhythm.stages import (
    resolve_runtime_dual_mode_teacher_enable,
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
_TEACHER_FIELD_GROUPS: tuple[tuple[str, ...], ...] = (
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
)
_RETIMED_FIELD_GROUPS: tuple[tuple[str, ...], ...] = (
    ("rhythm_retimed_mel_tgt",),
    ("rhythm_retimed_mel_len",),
    ("rhythm_retimed_frame_weight",),
    ("rhythm_retimed_target_confidence",),
    ("rhythm_retimed_target_source_id",),
    ("rhythm_retimed_target_surface_name",),
)


@dataclass(frozen=True)
class RhythmConfigContractContext:
    hparams: Mapping[str, Any]
    config_path: str | None
    model_dry_run: bool
    profile: str
    stage: str
    policy: Any


@dataclass(frozen=True)
class RhythmConfigContractReport:
    profile: str
    stage: str
    required_field_groups: tuple[tuple[str, ...], ...] = ()
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def merged(self, *others: "RhythmConfigContractReport") -> "RhythmConfigContractReport":
        return merge_contract_reports(self, *others)


@dataclass(frozen=True)
class RhythmContractValidationResult:
    profile: str
    stage: str
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class RhythmConfigContractEvaluation:
    context: RhythmConfigContractContext
    report: RhythmConfigContractReport


def detect_rhythm_profile(hparams: Mapping[str, Any], config_path: str | None = None) -> str:
    is_minimal_flag = bool(hparams.get("rhythm_minimal_v1_profile", False)) or "minimal_v1" in str(config_path or "").lower()
    if (
        is_minimal_flag
        and not bool(hparams.get("rhythm_enable_dual_mode_teacher", False))
        and not bool(hparams.get("rhythm_apply_train_override", False))
        and not bool(hparams.get("rhythm_apply_valid_override", False))
        and normalize_distill_surface(str(hparams.get("rhythm_distill_surface", "auto") or "auto").strip().lower()) == "none"
    ):
        return "minimal_v1"
    return "default"


def validate_profile_contract(
    hparams: Mapping[str, Any],
    *,
    config_path: str | None = None,
    model_dry_run: bool = True,
) -> tuple[str, list[str], list[str]]:
    profile = detect_rhythm_profile(hparams, config_path)
    errors: list[str] = []
    warnings: list[str] = []
    if profile != "minimal_v1":
        return profile, errors, warnings

    policy = build_rhythm_hparams_policy(hparams, config_path=config_path)
    target_mode = policy.target_mode
    primary = policy.primary_target_surface
    distill = policy.distill_surface
    if target_mode != "cached_only":
        errors.append("minimal_v1 profile requires rhythm_dataset_target_mode: cached_only.")
    if primary != "teacher":
        errors.append("minimal_v1 profile requires rhythm_primary_target_surface: teacher.")
    if distill != "none":
        errors.append("minimal_v1 profile requires rhythm_distill_surface: none.")
    if not bool(hparams.get("rhythm_require_cached_teacher", False)):
        errors.append("minimal_v1 profile requires rhythm_require_cached_teacher: true.")
    if not bool(hparams.get("rhythm_binarize_teacher_targets", False)):
        errors.append("minimal_v1 profile requires rhythm_binarize_teacher_targets: true.")
    if policy.teacher_target_source != "learned_offline":
        errors.append("minimal_v1 profile requires rhythm_teacher_target_source: learned_offline.")
    if bool(hparams.get("rhythm_dataset_build_guidance_from_ref", True)):
        errors.append("minimal_v1 profile should disable runtime guidance target synthesis.")
    if bool(hparams.get("rhythm_dataset_build_teacher_from_ref", False)):
        errors.append("minimal_v1 profile should disable runtime teacher target synthesis.")
    if bool(hparams.get("rhythm_enable_dual_mode_teacher", False)):
        errors.append("minimal_v1 profile should keep rhythm_enable_dual_mode_teacher: false.")
    if bool(hparams.get("rhythm_require_retimed_cache", False)):
        errors.append("minimal_v1 profile should not require retimed mel cache.")
    if bool(hparams.get("rhythm_apply_train_override", False)) or bool(hparams.get("rhythm_apply_valid_override", False)):
        errors.append("minimal_v1 profile should keep train/valid on source-aligned canvas.")

    if float(hparams.get("lambda_rhythm_exec_speech", 0.0)) <= 0.0:
        errors.append("minimal_v1 profile requires lambda_rhythm_exec_speech > 0.")
    if float(hparams.get("lambda_rhythm_exec_pause", 0.0)) <= 0.0:
        errors.append("minimal_v1 profile requires lambda_rhythm_exec_pause > 0.")
    if resolve_cumplan_lambda(hparams) <= 0.0:
        errors.append("minimal_v1 profile requires lambda_rhythm_cumplan (or lambda_rhythm_carry) > 0.")
    if float(hparams.get("lambda_rhythm_budget", 0.0)) < 0.0:
        errors.append("minimal_v1 profile does not allow negative lambda_rhythm_budget.")
    elif float(hparams.get("lambda_rhythm_budget", 0.0)) == 0.0:
        warnings.append("minimal_v1 profile keeps lambda_rhythm_budget at 0; maintained path expects a small positive budget guardrail.")
    if float(hparams.get("lambda_rhythm_guidance", 0.0)) > 0.0:
        errors.append("minimal_v1 profile requires lambda_rhythm_guidance: 0.")
    if float(hparams.get("lambda_rhythm_plan", 0.0)) > 0.0:
        errors.append("minimal_v1 profile requires lambda_rhythm_plan: 0.")
    if float(hparams.get("lambda_rhythm_distill", 0.0)) > 0.0:
        errors.append("minimal_v1 profile requires lambda_rhythm_distill: 0.")

    if not model_dry_run:
        warnings.append("minimal_v1 train-ready preflight should include --model_dry_run before starting training.")
    if "rhythm_min_unit_frames" in hparams:
        legacy_min_unit = hparams.get("rhythm_min_unit_frames")
        if legacy_min_unit not in {None, "", 0, 0.0, "0", "0.0"}:
            warnings.append(
                "rhythm_min_unit_frames is a legacy/unsupported knob in the maintained path; "
                "remove it instead of assuming it is active."
            )
    return profile, errors, warnings


def validate_stage_contract(
    hparams: Mapping[str, Any],
    *,
    config_path: str | None = None,
) -> tuple[str, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    policy = build_rhythm_hparams_policy(hparams, config_path=config_path)
    stage = policy.stage
    profile = detect_rhythm_profile(hparams, config_path)
    target_mode = policy.target_mode
    primary = policy.primary_target_surface
    distill = policy.distill_surface
    require_cached_teacher = policy.require_cached_teacher
    require_retimed_cache = policy.require_retimed_cache
    enable_dual = bool(hparams.get("rhythm_enable_dual_mode_teacher", False))
    enable_learned_offline_teacher = bool(hparams.get("rhythm_enable_learned_offline_teacher", True))
    explicit_runtime_teacher_enable = hparams.get("rhythm_runtime_enable_learned_offline_teacher", None)
    runtime_offline_teacher_enable = policy.runtime_offline_teacher_enabled
    runtime_dual_mode_teacher_enable = resolve_runtime_dual_mode_teacher_enable(hparams, stage=stage, infer=False)
    legacy_schedule_flag = bool(hparams.get("rhythm_schedule_only_stage", False))
    strict_mainline = policy.strict_mainline
    optimize_module_only = bool(hparams.get("rhythm_optimize_module_only", False))
    lambda_distill = float(hparams.get("lambda_rhythm_distill", 0.0))
    lambda_teacher_aux = float(hparams.get("lambda_rhythm_teacher_aux", 0.0) or 0.0)
    teacher_as_main = policy.teacher_as_main
    apply_train = bool(hparams.get("rhythm_apply_train_override", False))
    apply_valid = bool(hparams.get("rhythm_apply_valid_override", False))
    use_retimed_target = bool(hparams.get("rhythm_use_retimed_target_if_available", False))
    retimed_target_mode = policy.retimed_target_mode
    retimed_target_start = int(hparams.get("rhythm_retimed_target_start_steps", 0) or 0)
    online_target_start = int(hparams.get("rhythm_online_retimed_target_start_steps", retimed_target_start) or retimed_target_start)
    use_retimed_pitch_target = bool(hparams.get("rhythm_use_retimed_pitch_target", False))
    disable_pitch_when_retimed = bool(hparams.get("rhythm_disable_pitch_loss_when_retimed", True))
    retimed_source = str(hparams.get("rhythm_binarize_retimed_mel_source", "guidance") or "guidance").strip().lower()
    binarize_teacher = bool(hparams.get("rhythm_binarize_teacher_targets", False))
    disable_mel_adv_when_retimed = bool(hparams.get("rhythm_disable_mel_adv_when_retimed", True))
    lambda_mel_adv = float(hparams.get("lambda_mel_adv", 0.0))
    lambda_guidance = float(hparams.get("lambda_rhythm_guidance", 0.0))
    distill_budget_weight = float(hparams.get("rhythm_distill_budget_weight", 0.5))
    distill_allocation_weight = float(hparams.get("rhythm_distill_allocation_weight", 0.5))
    distill_prefix_weight = float(hparams.get("rhythm_distill_prefix_weight", 0.25))
    compact_joint_loss = bool(hparams.get("rhythm_compact_joint_loss", True))
    pause_topk_ratio = float(hparams.get("rhythm_projector_pause_topk_ratio", 0.35))
    pause_topk_ratio_train_start = float(hparams.get("rhythm_projector_pause_topk_ratio_train_start", 1.0))
    pause_topk_ratio_train_end = float(hparams.get("rhythm_projector_pause_topk_ratio_train_end", pause_topk_ratio))
    pause_topk_ratio_anneal_steps = int(hparams.get("rhythm_projector_pause_topk_ratio_anneal_steps", 20000) or 0)
    pause_topk_ratio_warmup_steps = int(hparams.get("rhythm_projector_pause_topk_ratio_warmup_steps", 0) or 0)
    distill_conf_floor = float(hparams.get("rhythm_distill_confidence_floor", 0.05))
    distill_conf_power = float(hparams.get("rhythm_distill_confidence_power", 1.0))
    source_boundary_scale = float(hparams.get("rhythm_source_boundary_scale", 1.0))
    source_boundary_scale_train_start = float(hparams.get("rhythm_source_boundary_scale_train_start", 1.0))
    source_boundary_scale_train_end = float(hparams.get("rhythm_source_boundary_scale_train_end", source_boundary_scale))
    source_boundary_scale_anneal_steps = int(hparams.get("rhythm_source_boundary_scale_anneal_steps", 20000) or 0)
    source_boundary_scale_warmup_steps = int(hparams.get("rhythm_source_boundary_scale_warmup_steps", 0) or 0)
    teacher_source_boundary_scale = float(hparams.get("rhythm_teacher_source_boundary_scale", source_boundary_scale))
    export_cache_audit_to_sample = bool(hparams.get("rhythm_export_cache_audit_to_sample", False))
    public_losses = list(hparams.get("rhythm_public_losses", []) or [])
    configured_cache_version = int(hparams.get("rhythm_cache_version", RHYTHM_CACHE_VERSION))

    if configured_cache_version != int(RHYTHM_CACHE_VERSION):
        errors.append(
            f"rhythm_cache_version mismatch: configured={configured_cache_version}, maintained={int(RHYTHM_CACHE_VERSION)}."
        )
    if retimed_target_mode not in {"cached", "online", "hybrid"}:
        errors.append(f"Unsupported rhythm_retimed_target_mode: {retimed_target_mode}")
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
        errors.append("Runtime dual-mode teacher resolved enabled while rhythm_enable_dual_mode_teacher is false.")
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
    if export_cache_audit_to_sample:
        warnings.append("rhythm_export_cache_audit_to_sample=true adds cache appendix fields to runtime batch; keep it off outside audits.")
    if public_losses:
        required_public_losses = {"L_exec_speech", "L_exec_pause", "L_budget", "L_prefix_state", "L_base"}
        missing_public = sorted(required_public_losses.difference(set(public_losses)))
        if missing_public:
            warnings.append(f"rhythm_public_losses is missing maintained mainline aliases: {missing_public}.")
    if "lambda_rhythm_carry" in hparams and "lambda_rhythm_cumplan" in hparams:
        lambda_carry = float(hparams.get("lambda_rhythm_carry", 0.0))
        lambda_cumplan = float(hparams.get("lambda_rhythm_cumplan", 0.0))
        if abs(lambda_carry - lambda_cumplan) > 1e-8:
            errors.append("lambda_rhythm_carry and lambda_rhythm_cumplan are both set but disagree.")
        else:
            warnings.append("Both lambda_rhythm_carry and lambda_rhythm_cumplan are set; prefer lambda_rhythm_cumplan.")
    if strict_mainline:
        if lambda_guidance > 0.0:
            errors.append("rhythm_strict_mainline requires lambda_rhythm_guidance: 0.")
        if lambda_distill > 0.0 and distill != "cache":
            errors.append(
                "rhythm_strict_mainline only allows cached teacher distillation; "
                "set rhythm_distill_surface: cache or disable lambda_rhythm_distill."
            )
        if bool(hparams.get("rhythm_enable_algorithmic_teacher", False)):
            errors.append("rhythm_strict_mainline requires rhythm_enable_algorithmic_teacher: false.")
        if enable_dual:
            errors.append("rhythm_strict_mainline requires rhythm_enable_dual_mode_teacher: false.")
        if enable_learned_offline_teacher:
            errors.append("rhythm_strict_mainline requires rhythm_enable_learned_offline_teacher: false.")
        if runtime_offline_teacher_enable:
            errors.append("rhythm_strict_mainline requires rhythm_runtime_enable_learned_offline_teacher: false.")
        if lambda_teacher_aux > 0.0:
            errors.append("rhythm_strict_mainline requires lambda_rhythm_teacher_aux: 0.")
        if teacher_as_main:
            errors.append("rhythm_strict_mainline requires rhythm_teacher_as_main: false.")
    if lambda_distill > 0.0:
        if distill == "none":
            errors.append("lambda_rhythm_distill > 0 but rhythm_distill_surface disables distillation.")
        if distill == "offline" and not enable_dual:
            errors.append("Offline distillation requires rhythm_enable_dual_mode_teacher: true.")
        if distill == "cache" and not require_cached_teacher:
            warnings.append(
                "Cached teacher distillation is enabled without rhythm_require_cached_teacher: true; "
                "maintained student-only KD should require cache-backed teacher surfaces."
            )
    if enable_dual and lambda_distill <= 0.0:
        warnings.append("Dual-mode teacher is enabled but lambda_rhythm_distill == 0.")
    if primary == "teacher" and not binarize_teacher:
        warnings.append("Primary rhythm target surface is teacher but rhythm_binarize_teacher_targets is false.")
    if retimed_source == "teacher" and not binarize_teacher:
        errors.append("Teacher retimed targets require rhythm_binarize_teacher_targets: true.")

    if stage == "teacher_offline":
        if strict_mainline:
            warnings.append("teacher_offline is a teacher-asset build path; keep rhythm_strict_mainline=false for the maintained student mainline.")
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
            errors.append("teacher_offline stage should not use rhythm_primary_target_surface: teacher; bootstrap the teacher from guidance/self targets.")
        if policy.teacher_target_source != "algorithmic":
            warnings.append("teacher_offline stage does not consume learned_offline teacher caches; prefer rhythm_teacher_target_source: algorithmic for clarity.")
    elif stage == "legacy_schedule_only":
        warnings.append("legacy_schedule_only is no longer part of the maintained chain; prefer teacher_offline -> student_kd -> student_retimed.")
        if target_mode != "cached_only":
            errors.append("legacy_schedule_only should use rhythm_dataset_target_mode: cached_only.")
        if primary != "teacher":
            errors.append("legacy_schedule_only should use rhythm_primary_target_surface: teacher.")
        if not require_cached_teacher:
            errors.append("legacy_schedule_only should require cached teacher surfaces.")
        if policy.teacher_target_source != "learned_offline":
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
            errors.append("legacy_schedule_only should set rhythm_enable_learned_offline_teacher: false to avoid unnecessary runtime teacher branch allocation.")
        if runtime_offline_teacher_enable:
            errors.append("legacy_schedule_only should keep runtime offline teacher disabled (rhythm_runtime_enable_learned_offline_teacher should resolve to false).")
        if runtime_dual_mode_teacher_enable:
            errors.append("legacy_schedule_only should not resolve runtime dual-mode teacher execution.")
        if lambda_teacher_aux > 0.0:
            errors.append("legacy_schedule_only should keep lambda_rhythm_teacher_aux: 0.")
        if lambda_distill > 0.0 or distill != "none":
            errors.append("legacy_schedule_only should keep distillation disabled.")
        if apply_train or apply_valid:
            errors.append("legacy_schedule_only should not enable train/valid retimed rendering.")
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
            errors.append("student_kd should keep runtime offline teacher disabled (rhythm_runtime_enable_learned_offline_teacher should resolve false).")
        if runtime_dual_mode_teacher_enable:
            errors.append("student_kd should not resolve runtime dual-mode teacher execution.")
        if not require_cached_teacher:
            errors.append("student_kd should require cached teacher surfaces.")
        if policy.teacher_target_source != "learned_offline":
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
        if not optimize_module_only:
            warnings.append("student_kd usually keeps rhythm_optimize_module_only: true for a short maintained stage-2 path.")
    elif stage == "legacy_dual_mode_kd":
        warnings.append("legacy_dual_mode_kd resolves to a research path. Maintained chain is now teacher_offline -> student_kd -> student_retimed.")
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
        if policy.teacher_target_source != "learned_offline":
            errors.append("student_retimed should use rhythm_teacher_target_source: learned_offline.")
        if teacher_as_main:
            errors.append("student_retimed should keep rhythm_teacher_as_main: false.")
        if lambda_distill > 0.0:
            if distill != "cache":
                errors.append("student_retimed with KD enabled should use rhythm_distill_surface: cache.")
            if enable_dual:
                errors.append("student_retimed with KD enabled should keep rhythm_enable_dual_mode_teacher: false.")
            if enable_learned_offline_teacher:
                errors.append("student_retimed with KD enabled should keep rhythm_enable_learned_offline_teacher: false.")
            if runtime_offline_teacher_enable:
                errors.append("student_retimed with KD enabled should keep runtime offline teacher disabled (rhythm_runtime_enable_learned_offline_teacher should resolve false).")
            if runtime_dual_mode_teacher_enable:
                errors.append("student_retimed with KD enabled should not resolve runtime dual-mode teacher execution.")
        else:
            if distill not in {"none", "off", "disable", "disabled", "false"}:
                warnings.append("student_retimed without KD usually keeps rhythm_distill_surface: none.")
            if enable_dual:
                warnings.append("student_retimed without KD usually keeps rhythm_enable_dual_mode_teacher: false.")
            if enable_learned_offline_teacher:
                warnings.append("student_retimed without KD usually keeps rhythm_enable_learned_offline_teacher: false to reduce unused runtime overhead.")
            if runtime_offline_teacher_enable:
                warnings.append("student_retimed without KD usually keeps runtime offline teacher disabled to reduce unnecessary branch overhead.")
            if runtime_dual_mode_teacher_enable:
                warnings.append("student_retimed without KD usually keeps runtime dual-mode teacher execution disabled.")
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
            errors.append("student_retimed should set rhythm_retimed_target_start_steps: 0 for immediate train/infer closure on the retimed canvas.")
        if legacy_schedule_flag:
            errors.append("student_retimed should set rhythm_schedule_only_stage: false.")
        if optimize_module_only:
            errors.append("student_retimed should set rhythm_optimize_module_only: false.")
        if not compact_joint_loss:
            errors.append("student_retimed should keep rhythm_compact_joint_loss: true.")
        if lambda_guidance > 0.0:
            errors.append("student_retimed should keep lambda_rhythm_guidance: 0.")
        if lambda_mel_adv > 0.0 and not disable_mel_adv_when_retimed:
            warnings.append("student_retimed usually disables mel-adversarial loss on retimed targets.")

    if profile != "minimal_v1" and bool(hparams.get("rhythm_minimal_v1_profile", False)):
        warnings.append("rhythm_minimal_v1_profile=true no longer implies the maintained chain; prefer explicit rhythm_stage={teacher_offline,student_kd,student_retimed}.")
    contract = build_expected_cache_contract(hparams)
    if int(contract["rhythm_cache_version"]) != int(RHYTHM_CACHE_VERSION):
        warnings.append(
            f"expected_cache_contract resolved rhythm_cache_version={int(contract['rhythm_cache_version'])} "
            f"while maintained cache version is {int(RHYTHM_CACHE_VERSION)}."
        )
    return stage, errors, warnings


def collect_rhythm_contract_issues(
    hparams: Mapping[str, Any],
    *,
    config_path: str | None = None,
    model_dry_run: bool = True,
) -> RhythmContractValidationResult:
    profile, profile_errors, profile_warnings = validate_profile_contract(
        hparams,
        config_path=config_path,
        model_dry_run=model_dry_run,
    )
    stage, stage_errors, stage_warnings = validate_stage_contract(
        hparams,
        config_path=config_path,
    )
    return RhythmContractValidationResult(
        profile=profile,
        stage=stage,
        errors=tuple(profile_errors + stage_errors),
        warnings=tuple(profile_warnings + stage_warnings),
    )


def _dedup_groups(groups: Sequence[Sequence[str]]) -> tuple[tuple[str, ...], ...]:
    dedup: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for group in groups:
        normalized = tuple(group)
        if normalized in seen:
            continue
        seen.add(normalized)
        dedup.append(normalized)
    return tuple(dedup)


def _dedup_messages(messages: Sequence[str]) -> tuple[str, ...]:
    dedup: list[str] = []
    seen: set[str] = set()
    for message in messages:
        normalized = str(message)
        if normalized in seen:
            continue
        seen.add(normalized)
        dedup.append(normalized)
    return tuple(dedup)


def build_contract_context(
    hparams: Mapping[str, Any],
    *,
    config_path: str | None = None,
    model_dry_run: bool = True,
) -> RhythmConfigContractContext:
    policy = build_rhythm_hparams_policy(hparams, config_path=config_path)
    return RhythmConfigContractContext(
        hparams=hparams,
        config_path=config_path,
        model_dry_run=model_dry_run,
        profile=detect_rhythm_profile(hparams, config_path=config_path),
        stage=policy.stage,
        policy=policy,
    )


def merge_contract_reports(*reports: RhythmConfigContractReport) -> RhythmConfigContractReport:
    profile = ""
    stage = ""
    groups: list[tuple[str, ...]] = []
    errors: list[str] = []
    warnings: list[str] = []
    for report in reports:
        if report is None:
            continue
        if not profile and report.profile:
            profile = report.profile
        if not stage and report.stage:
            stage = report.stage
        groups.extend(report.required_field_groups)
        errors.extend(report.errors)
        warnings.extend(report.warnings)
    return RhythmConfigContractReport(
        profile=profile,
        stage=stage,
        required_field_groups=_dedup_groups(groups),
        errors=_dedup_messages(errors),
        warnings=_dedup_messages(warnings),
    )


def collect_config_contract_evaluation(
    hparams: Mapping[str, Any],
    *,
    config_path: str | None = None,
    model_dry_run: bool = True,
) -> RhythmConfigContractEvaluation:
    context = build_contract_context(
        hparams,
        config_path=config_path,
        model_dry_run=model_dry_run,
    )
    report = merge_contract_reports(
        validate_stage_profile_contract(context),
        validate_cache_field_contract(context),
    )
    return RhythmConfigContractEvaluation(
        context=context,
        report=report,
    )


def validate_stage_profile_contract(
    context: RhythmConfigContractContext,
) -> RhythmConfigContractReport:
    result = collect_rhythm_contract_issues(
        context.hparams,
        config_path=context.config_path,
        model_dry_run=context.model_dry_run,
    )
    return RhythmConfigContractReport(
        profile=result.profile,
        stage=result.stage,
        errors=tuple(result.errors),
        warnings=tuple(result.warnings),
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

    expected_groups: list[tuple[str, ...]] = [(key,) for key in sorted(CORE_RHYTHM_FIELDS)]
    errors: list[str] = []
    warnings: list[str] = []

    if cached_only:
        expected_groups.extend(_CACHED_ONLY_META_FIELD_GROUPS)
    if primary == "guidance":
        expected_groups.extend(_GUIDANCE_FIELD_GROUPS)
    if (
        primary == "teacher"
        or bool(hp.get("rhythm_require_cached_teacher", False))
        or bool(hp.get("rhythm_binarize_teacher_targets", False))
        or str(hp.get("rhythm_binarize_retimed_mel_source", "guidance") or "guidance").strip().lower() == "teacher"
    ):
        expected_groups.extend(_TEACHER_FIELD_GROUPS)
    if (
        bool(hp.get("rhythm_require_retimed_cache", False))
        or bool(hp.get("rhythm_apply_train_override", False))
        or bool(hp.get("rhythm_apply_valid_override", False))
    ):
        expected_groups.extend(_RETIMED_FIELD_GROUPS)

    if float(hp.get("lambda_rhythm_distill", 0.0)) > 0.0:
        if distill == "none":
            errors.append("lambda_rhythm_distill > 0 but rhythm_distill_surface disables distillation.")
        if distill == "offline" and not bool(hp.get("rhythm_enable_dual_mode_teacher", False)):
            errors.append("Offline distillation requires rhythm_enable_dual_mode_teacher: true.")
        if distill == "cache":
            expected_groups.extend(_TEACHER_FIELD_GROUPS)

    if bool(hp.get("rhythm_apply_train_override", False)) and not bool(
        hp.get("rhythm_use_retimed_target_if_available", False)
    ):
        errors.append("Train-time retimed rendering requires rhythm_use_retimed_target_if_available: true.")
    if bool(hp.get("rhythm_schedule_only_stage", False)) and bool(hp.get("rhythm_apply_train_override", False)):
        errors.append("Schedule-only stage should not enable train-time retimed rendering.")
    if cached_only and int(hp.get("rhythm_cache_version", -1)) <= 0:
        errors.append("cached_only requires a positive rhythm_cache_version.")
    if primary == "teacher" and not bool(hp.get("rhythm_binarize_teacher_targets", False)):
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
    "RhythmContractValidationResult",
    "RhythmConfigContractEvaluation",
    "RhythmConfigContractContext",
    "RhythmConfigContractReport",
    "build_contract_context",
    "collect_config_contract_evaluation",
    "collect_cache_contract_report",
    "collect_rhythm_contract_issues",
    "detect_rhythm_profile",
    "merge_contract_reports",
    "validate_cache_field_contract",
    "validate_inspected_cache_items",
    "validate_profile_contract",
    "validate_required_field_presence",
    "validate_stage_contract",
    "validate_stage_profile_contract",
]
