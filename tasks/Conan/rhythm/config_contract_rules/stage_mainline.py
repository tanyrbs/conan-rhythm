from __future__ import annotations

from .context import RhythmStageValidationContext


def _validate_cached_teacher_student_base(
    ctx: RhythmStageValidationContext,
    errors: list[str],
    *,
    stage_name: str,
) -> None:
    knobs = ctx.knobs
    if knobs.target_mode != "cached_only":
        errors.append(f"{stage_name} should use rhythm_dataset_target_mode: cached_only.")
    if knobs.primary_target_surface != "teacher":
        errors.append(f"{stage_name} should use rhythm_primary_target_surface: teacher.")
    if ctx.policy.teacher_target_source != "learned_offline":
        errors.append(f"{stage_name} should use rhythm_teacher_target_source: learned_offline.")
    if knobs.teacher_as_main:
        errors.append(f"{stage_name} should keep rhythm_teacher_as_main: false.")
    if not knobs.require_cached_teacher:
        errors.append(f"{stage_name} should require cached teacher surfaces.")
    if knobs.lambda_teacher_aux > 0.0:
        errors.append(f"{stage_name} should keep lambda_rhythm_teacher_aux: 0.")


def validate_teacher_offline(
    ctx: RhythmStageValidationContext,
    errors: list[str],
    warnings: list[str],
) -> None:
    knobs = ctx.knobs
    if knobs.strict_mainline:
        warnings.append(
            "teacher_offline is a teacher-asset build path; keep rhythm_strict_mainline=false for the maintained student mainline."
        )
    if knobs.legacy_schedule_flag:
        errors.append("teacher_offline stage should keep rhythm_schedule_only_stage: false.")
    if not knobs.enable_learned_offline_teacher:
        errors.append("teacher_offline stage requires rhythm_enable_learned_offline_teacher: true.")
    if not knobs.runtime_offline_teacher_enabled:
        errors.append("teacher_offline stage requires runtime offline teacher branch enabled.")
    if not knobs.teacher_as_main:
        errors.append("teacher_offline stage requires rhythm_teacher_as_main: true.")
    if knobs.runtime_dual_mode_teacher_enabled:
        errors.append("teacher_offline stage should not resolve runtime dual-mode teacher execution.")
    if knobs.enable_dual_mode_teacher:
        errors.append("teacher_offline stage should keep rhythm_enable_dual_mode_teacher: false.")
    if knobs.lambda_distill > 0.0:
        errors.append("teacher_offline stage should keep lambda_rhythm_distill: 0.")
    if knobs.lambda_teacher_aux > 0.0:
        errors.append("teacher_offline stage should keep lambda_rhythm_teacher_aux: 0.")
    if knobs.primary_target_surface == "teacher":
        errors.append(
            "teacher_offline stage should not use rhythm_primary_target_surface: teacher; bootstrap the teacher from guidance/self targets."
        )
    if ctx.policy.teacher_target_source != "algorithmic":
        warnings.append(
            "teacher_offline stage does not consume learned_offline teacher caches; prefer rhythm_teacher_target_source: algorithmic for clarity."
        )


def validate_student_kd(
    ctx: RhythmStageValidationContext,
    errors: list[str],
    warnings: list[str],
) -> None:
    knobs = ctx.knobs
    _validate_cached_teacher_student_base(ctx, errors, stage_name="student_kd")
    if knobs.distill_surface != "cache":
        errors.append("student_kd should use rhythm_distill_surface: cache.")
    if knobs.lambda_distill <= 0.0:
        errors.append("student_kd should keep lambda_rhythm_distill > 0.")
    if knobs.enable_dual_mode_teacher:
        errors.append("student_kd should keep rhythm_enable_dual_mode_teacher: false.")
    if knobs.enable_learned_offline_teacher:
        errors.append("student_kd should keep rhythm_enable_learned_offline_teacher: false.")
    if knobs.runtime_offline_teacher_enabled:
        errors.append(
            "student_kd should keep runtime offline teacher disabled (rhythm_runtime_enable_learned_offline_teacher should resolve false)."
        )
    if knobs.runtime_dual_mode_teacher_enabled:
        errors.append("student_kd should not resolve runtime dual-mode teacher execution.")
    if knobs.lambda_guidance > 0.0:
        errors.append("student_kd should keep lambda_rhythm_guidance: 0.")
    if knobs.distill_allocation_weight > 0.0:
        errors.append("student_kd should keep rhythm_distill_allocation_weight: 0.")
    if not knobs.dedupe_teacher_primary_cache_distill and knobs.distill_budget_weight > 0.15:
        warnings.append("student_kd usually keeps rhythm_distill_budget_weight <= 0.15.")
    if not knobs.dedupe_teacher_primary_cache_distill and knobs.distill_prefix_weight <= 0.0:
        warnings.append("student_kd usually keeps rhythm_distill_prefix_weight > 0.")
    if knobs.distill_allocation_weight > 0.0 and knobs.shape_distill_active:
        warnings.append(
            "student_kd maintained path should not enable allocation distill together with shape distill."
        )
    if (
        knobs.primary_target_surface == "teacher"
        and knobs.distill_surface == "cache"
        and knobs.lambda_distill > 0.0
    ):
        if knobs.distill_exec_weight > 0.0:
            exec_overlap_msg = (
                "student_kd with rhythm_primary_target_surface: teacher and rhythm_distill_surface: cache "
                "must keep rhythm_distill_exec_weight: 0 so the distill branch does not re-regress the same "
                "cached teacher exec surface."
            )
            if knobs.strict_mainline:
                errors.append(exec_overlap_msg)
            else:
                warnings.append(exec_overlap_msg)
        duplicate_msg = (
            "student_kd reuses cached teacher surfaces for both primary supervision and distillation; "
            "enable rhythm_dedupe_teacher_primary_cache_distill: true (recommended) or switch to a different "
            "distill surface so lambda_rhythm_distill is not just extra teacher reweighting."
        )
        if not knobs.dedupe_teacher_primary_cache_distill:
            if knobs.strict_mainline:
                errors.append(duplicate_msg)
            else:
                warnings.append(duplicate_msg)
        else:
            if not knobs.shape_distill_active:
                errors.append(
                    "student_kd with deduped teacher/cache distill must keep rhythm_distill_speech_shape_weight > 0 "
                    "or rhythm_distill_pause_shape_weight > 0, otherwise lambda_rhythm_distill becomes effectively inactive."
                )
            if (
                knobs.distill_budget_weight > 0.0
                or knobs.distill_prefix_weight > 0.0
                or knobs.distill_allocation_weight > 0.0
            ):
                warnings.append(
                    "student_kd dedupe neutralizes exact duplicate cache-based budget/prefix/allocation distill terms; "
                    "the effective independent KD branch is the remaining shape distill."
                )
    if knobs.apply_train_override or knobs.apply_valid_override:
        errors.append("student_kd should not enable train/valid retimed rendering; that belongs to stage-3.")
    if knobs.retimed_target_mode != "cached":
        warnings.append(
            "student_kd usually keeps rhythm_retimed_target_mode: cached (retimed closure starts in stage-3)."
        )
    if not knobs.optimize_module_only:
        warnings.append(
            "student_kd usually keeps rhythm_optimize_module_only: true for a short maintained stage-2 path."
        )


def validate_student_retimed(
    ctx: RhythmStageValidationContext,
    errors: list[str],
    warnings: list[str],
) -> None:
    knobs = ctx.knobs
    _validate_cached_teacher_student_base(ctx, errors, stage_name="student_retimed")
    if knobs.lambda_distill > 0.0:
        if knobs.distill_surface != "cache":
            errors.append("student_retimed with KD enabled should use rhythm_distill_surface: cache.")
        if knobs.enable_dual_mode_teacher:
            errors.append(
                "student_retimed with KD enabled should keep rhythm_enable_dual_mode_teacher: false."
            )
        if knobs.enable_learned_offline_teacher:
            errors.append(
                "student_retimed with KD enabled should keep rhythm_enable_learned_offline_teacher: false."
            )
        if knobs.runtime_offline_teacher_enabled:
            errors.append(
                "student_retimed with KD enabled should keep runtime offline teacher disabled (rhythm_runtime_enable_learned_offline_teacher should resolve false)."
            )
        if knobs.runtime_dual_mode_teacher_enabled:
            errors.append(
                "student_retimed with KD enabled should not resolve runtime dual-mode teacher execution."
            )
    else:
        if knobs.distill_surface != "none":
            warnings.append("student_retimed without KD usually keeps rhythm_distill_surface: none.")
        if knobs.enable_dual_mode_teacher:
            warnings.append("student_retimed without KD usually keeps rhythm_enable_dual_mode_teacher: false.")
        if knobs.enable_learned_offline_teacher:
            warnings.append(
                "student_retimed without KD usually keeps rhythm_enable_learned_offline_teacher: false to reduce unused runtime overhead."
            )
        if knobs.runtime_offline_teacher_enabled:
            warnings.append(
                "student_retimed without KD usually keeps runtime offline teacher disabled to reduce unnecessary branch overhead."
            )
        if knobs.runtime_dual_mode_teacher_enabled:
            warnings.append(
                "student_retimed without KD usually keeps runtime dual-mode teacher execution disabled."
            )
    if not knobs.require_retimed_cache:
        errors.append("student_retimed should require cached retimed mel targets.")
    if not knobs.use_retimed_target_if_available:
        errors.append("student_retimed requires rhythm_use_retimed_target_if_available: true.")
    if not knobs.apply_train_override or not knobs.apply_valid_override:
        errors.append("student_retimed should enable both train/valid retimed rendering.")
    if knobs.retimed_target_start_steps > 0:
        errors.append(
            "student_retimed should set rhythm_retimed_target_start_steps: 0 for immediate train/infer closure on the retimed canvas."
        )
    if knobs.legacy_schedule_flag:
        errors.append("student_retimed should set rhythm_schedule_only_stage: false.")
    if knobs.optimize_module_only:
        errors.append("student_retimed should set rhythm_optimize_module_only: false.")
    if not knobs.compact_joint_loss:
        errors.append("student_retimed should keep rhythm_compact_joint_loss: true.")
    if knobs.lambda_guidance > 0.0:
        errors.append("student_retimed should keep lambda_rhythm_guidance: 0.")
    if knobs.lambda_mel_adv > 0.0 and not knobs.disable_mel_adv_when_retimed:
        warnings.append("student_retimed usually disables mel-adversarial loss on retimed targets.")


def validate_student_ref_bootstrap(
    ctx: RhythmStageValidationContext,
    errors: list[str],
    warnings: list[str],
) -> None:
    knobs = ctx.knobs
    if knobs.strict_mainline:
        errors.append(
            "student_ref_bootstrap is an experimental external-reference stage; keep rhythm_strict_mainline=false."
        )
    if knobs.target_mode != "runtime_only":
        errors.append("student_ref_bootstrap should use rhythm_dataset_target_mode: runtime_only.")
    if knobs.cached_reference_policy not in {"sample_ref", "paired", "external"}:
        errors.append(
            "student_ref_bootstrap should use an external rhythm reference policy "
            "(sample_ref / paired / external)."
        )
    if not bool(ctx.hparams.get("rhythm_require_external_reference", False)):
        errors.append(
            "student_ref_bootstrap should set rhythm_require_external_reference: true "
            "so singleton speaker pools fail fast instead of silently collapsing to self-reference."
        )
    if knobs.primary_target_surface != "teacher":
        errors.append("student_ref_bootstrap should use rhythm_primary_target_surface: teacher.")
    if ctx.policy.teacher_target_source != "algorithmic":
        errors.append("student_ref_bootstrap should use rhythm_teacher_target_source: algorithmic.")
    if knobs.distill_surface != "none":
        errors.append("student_ref_bootstrap should keep rhythm_distill_surface: none.")
    if knobs.lambda_distill > 0.0:
        errors.append("student_ref_bootstrap should keep lambda_rhythm_distill: 0.")
    if knobs.lambda_teacher_aux > 0.0:
        errors.append("student_ref_bootstrap should keep lambda_rhythm_teacher_aux: 0.")
    if knobs.lambda_guidance > 0.0:
        errors.append("student_ref_bootstrap should keep lambda_rhythm_guidance: 0.")
    if knobs.require_cached_teacher:
        errors.append("student_ref_bootstrap should keep rhythm_require_cached_teacher: false.")
    if knobs.require_retimed_cache:
        errors.append("student_ref_bootstrap should keep rhythm_require_retimed_cache: false.")
    if knobs.apply_train_override or knobs.apply_valid_override:
        errors.append("student_ref_bootstrap should keep train/valid on the source-aligned canvas.")
    if not knobs.optimize_module_only:
        warnings.append(
            "student_ref_bootstrap usually keeps rhythm_optimize_module_only: true to isolate "
            "reference-driven control learning before stage-3 acoustic closure."
        )


__all__ = [
    "validate_teacher_offline",
    "validate_student_kd",
    "validate_student_retimed",
    "validate_student_ref_bootstrap",
]
