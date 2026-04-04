from __future__ import annotations

from .context import RhythmStageValidationContext


def validate_legacy_schedule_only(
    ctx: RhythmStageValidationContext,
    errors: list[str],
    warnings: list[str],
) -> None:
    knobs = ctx.knobs
    warnings.append(
        "legacy_schedule_only is no longer part of the maintained chain; prefer teacher_offline -> student_kd -> student_retimed."
    )
    if knobs.target_mode != "cached_only":
        errors.append("legacy_schedule_only should use rhythm_dataset_target_mode: cached_only.")
    if knobs.primary_target_surface != "teacher":
        errors.append("legacy_schedule_only should use rhythm_primary_target_surface: teacher.")
    if not knobs.require_cached_teacher:
        errors.append("legacy_schedule_only should require cached teacher surfaces.")
    if ctx.policy.teacher_target_source != "learned_offline":
        errors.append("legacy_schedule_only should use rhythm_teacher_target_source: learned_offline.")
    if not knobs.legacy_schedule_flag:
        errors.append("legacy_schedule_only should keep rhythm_schedule_only_stage: true.")
    if not knobs.optimize_module_only:
        errors.append("legacy_schedule_only should keep rhythm_optimize_module_only: true.")
    if knobs.teacher_as_main:
        errors.append("legacy_schedule_only should keep rhythm_teacher_as_main: false.")
    if knobs.enable_dual_mode_teacher:
        errors.append("legacy_schedule_only should not enable rhythm_enable_dual_mode_teacher.")
    if knobs.enable_learned_offline_teacher:
        errors.append(
            "legacy_schedule_only should set rhythm_enable_learned_offline_teacher: false to avoid unnecessary runtime teacher branch allocation."
        )
    if knobs.runtime_offline_teacher_enabled:
        errors.append(
            "legacy_schedule_only should keep runtime offline teacher disabled (rhythm_runtime_enable_learned_offline_teacher should resolve to false)."
        )
    if knobs.runtime_dual_mode_teacher_enabled:
        errors.append("legacy_schedule_only should not resolve runtime dual-mode teacher execution.")
    if knobs.lambda_teacher_aux > 0.0:
        errors.append("legacy_schedule_only should keep lambda_rhythm_teacher_aux: 0.")
    if knobs.lambda_distill > 0.0 or knobs.distill_surface != "none":
        errors.append("legacy_schedule_only should keep distillation disabled.")
    if knobs.apply_train_override or knobs.apply_valid_override:
        errors.append("legacy_schedule_only should not enable train/valid retimed rendering.")


def validate_legacy_dual_mode_kd(
    ctx: RhythmStageValidationContext,
    errors: list[str],
    warnings: list[str],
) -> None:
    knobs = ctx.knobs
    warnings.append(
        "legacy_dual_mode_kd resolves to a research path. Maintained chain is now teacher_offline -> student_kd -> student_retimed."
    )
    if knobs.teacher_as_main:
        errors.append("legacy_dual_mode_kd should keep rhythm_teacher_as_main: false.")
    if not knobs.enable_dual_mode_teacher:
        errors.append("legacy_dual_mode_kd requires rhythm_enable_dual_mode_teacher: true.")
    if not knobs.enable_learned_offline_teacher:
        errors.append("legacy_dual_mode_kd requires rhythm_enable_learned_offline_teacher: true.")
    if not knobs.runtime_offline_teacher_enabled:
        errors.append("legacy_dual_mode_kd requires runtime offline teacher branch enabled.")
    if not knobs.runtime_dual_mode_teacher_enabled:
        errors.append("legacy_dual_mode_kd requires runtime dual-mode teacher execution enabled.")
    if knobs.distill_surface != "offline":
        errors.append("legacy_dual_mode_kd requires rhythm_distill_surface: offline.")
    if knobs.lambda_teacher_aux <= 0.0:
        warnings.append(
            "legacy_dual_mode_kd usually carries lambda_rhythm_teacher_aux > 0 when kept as a research branch."
        )


__all__ = [
    "validate_legacy_schedule_only",
    "validate_legacy_dual_mode_kd",
]
