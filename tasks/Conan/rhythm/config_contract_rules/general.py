from __future__ import annotations

from modules.Conan.rhythm.surface_metadata import RHYTHM_CACHE_VERSION, normalize_teacher_target_source

from .compat import (
    validate_pause_boundary_alias_consistency,
    validate_prefix_lambda_alias_consistency,
)
from .context import RhythmStageValidationContext

_REQUIRED_PUBLIC_LOSSES = {
    "L_exec_speech",
    "L_exec_pause",
    "L_budget",
    "L_prefix_state",
    "L_base",
}
_MAINLINE_STAGES = {"teacher_offline", "student_kd", "student_retimed"}


def _validate_cache_version(ctx: RhythmStageValidationContext, errors: list[str]) -> None:
    configured_cache_version = ctx.knobs.configured_cache_version
    if configured_cache_version != int(RHYTHM_CACHE_VERSION):
        errors.append(
            f"rhythm_cache_version mismatch: configured={configured_cache_version}, maintained={int(RHYTHM_CACHE_VERSION)}."
        )


def _validate_nonnegative_weights(ctx: RhythmStageValidationContext, errors: list[str]) -> None:
    knobs = ctx.knobs
    for name, value in {
        "rhythm_budget_raw_weight": knobs.budget_raw_weight,
        "rhythm_budget_exec_weight": knobs.budget_exec_weight,
        "rhythm_plan_local_weight": knobs.plan_local_weight,
        "rhythm_plan_cum_weight": knobs.plan_cum_weight,
        "rhythm_pause_boundary_weight": knobs.pause_boundary_weight,
        "rhythm_distill_exec_weight": knobs.distill_exec_weight,
        "rhythm_distill_budget_weight": knobs.distill_budget_weight,
        "rhythm_distill_allocation_weight": knobs.distill_allocation_weight,
        "rhythm_distill_prefix_weight": knobs.distill_prefix_weight,
        "rhythm_distill_speech_shape_weight": knobs.distill_speech_shape_weight,
        "rhythm_distill_pause_shape_weight": knobs.distill_pause_shape_weight,
    }.items():
        if value < 0.0:
            errors.append(f"{name} must be >= 0.")


def _validate_objective_activation(ctx: RhythmStageValidationContext, errors: list[str]) -> None:
    knobs = ctx.knobs
    if knobs.lambda_budget > 0.0 and knobs.total_budget_component_weight <= 0.0:
        errors.append(
            "lambda_rhythm_budget > 0 requires rhythm_budget_raw_weight + rhythm_budget_exec_weight > 0."
        )
    if knobs.lambda_plan > 0.0 and knobs.total_plan_component_weight <= 0.0:
        errors.append(
            "lambda_rhythm_plan > 0 requires rhythm_plan_local_weight + rhythm_plan_cum_weight > 0."
        )
    if knobs.lambda_distill > 0.0 and knobs.total_distill_component_weight <= 0.0:
        errors.append(
            "lambda_rhythm_distill > 0 requires at least one active distillation component weight."
        )


def _validate_inactive_distill_config_clutter(
    ctx: RhythmStageValidationContext,
    warnings: list[str],
) -> None:
    knobs = ctx.knobs
    if knobs.lambda_distill <= 0.0 and any(
        value > 0.0 for value in knobs.explicit_inactive_distill_weights
    ):
        warnings.append(
            "lambda_rhythm_distill <= 0 while distill component weights remain nonzero; "
            "this is inactive KD config clutter and usually indicates a stale stage setting."
        )


def _validate_runtime_teacher_configuration(
    ctx: RhythmStageValidationContext,
    errors: list[str],
    warnings: list[str],
) -> None:
    knobs = ctx.knobs
    if knobs.enable_dual_mode_teacher and not knobs.enable_learned_offline_teacher:
        errors.append(
            "rhythm_enable_dual_mode_teacher requires rhythm_enable_learned_offline_teacher: true."
        )
    if knobs.enable_dual_mode_teacher and not knobs.runtime_offline_teacher_enabled:
        errors.append(
            "rhythm_enable_dual_mode_teacher requires runtime learned offline teacher branch to be enabled "
            "(check rhythm_runtime_enable_learned_offline_teacher / factory resolution)."
        )
    if knobs.runtime_dual_mode_teacher_enabled and not knobs.enable_dual_mode_teacher:
        errors.append(
            "Runtime dual-mode teacher resolved enabled while rhythm_enable_dual_mode_teacher is false."
        )
    if (
        knobs.explicit_runtime_teacher_enable is True
        and not knobs.enable_learned_offline_teacher
        and knobs.lambda_distill <= 0.0
    ):
        warnings.append(
            "rhythm_runtime_enable_learned_offline_teacher=true while rhythm_enable_learned_offline_teacher=false "
            "and no distillation is active; this adds runtime overhead without a maintained stage objective."
        )


def _validate_projector_schedule_ranges(
    ctx: RhythmStageValidationContext,
    errors: list[str],
    warnings: list[str],
) -> None:
    knobs = ctx.knobs
    for name, value in {
        "rhythm_projector_pause_topk_ratio": knobs.pause_topk_ratio,
        "rhythm_projector_pause_topk_ratio_train_start": knobs.pause_topk_ratio_train_start,
        "rhythm_projector_pause_topk_ratio_train_end": knobs.pause_topk_ratio_train_end,
    }.items():
        if not (0.0 <= value <= 1.0):
            errors.append(f"{name} must be in [0, 1].")
    if knobs.pause_topk_ratio_anneal_steps < 0:
        errors.append("rhythm_projector_pause_topk_ratio_anneal_steps must be >= 0.")
    if knobs.pause_topk_ratio_warmup_steps < 0:
        errors.append("rhythm_projector_pause_topk_ratio_warmup_steps must be >= 0.")
    if knobs.pause_topk_ratio_train_start < knobs.pause_topk_ratio_train_end:
        warnings.append(
            "pause top-k anneal is configured sparse->dense; maintained path usually uses dense->sparse."
        )
    if not (0.0 <= knobs.distill_conf_floor <= 1.0):
        errors.append("rhythm_distill_confidence_floor must be in [0, 1].")
    if knobs.distill_conf_power <= 0.0:
        errors.append("rhythm_distill_confidence_power must be > 0.")


def _validate_pause_recall_support_controls(
    ctx: RhythmStageValidationContext,
    warnings: list[str],
) -> None:
    knobs = ctx.knobs
    pause_recall_aux_enabled = (
        knobs.pause_event_weight > 0.0
        or knobs.pause_support_weight > 0.0
    )
    if not pause_recall_aux_enabled:
        return
    if knobs.pause_selection_mode != "sparse":
        warnings.append(
            "pause-recall auxiliaries are enabled but rhythm_projector_pause_selection_mode is not 'sparse'; "
            "the projector will not use top-k support selection, so recall-oriented capacity tuning may not take effect."
        )
    if knobs.pause_topk_ratio_train_end < 0.40:
        warnings.append(
            "pause-recall auxiliaries are enabled while rhythm_projector_pause_topk_ratio_train_end < 0.40; "
            "sparse projector capacity may still cap pause recall."
        )
    if knobs.pause_soft_temperature <= 0.12:
        warnings.append(
            "pause-recall auxiliaries are enabled while rhythm_projector_pause_soft_temperature <= 0.12; "
            "the sparse gate remains sharp, so near-threshold pause candidates may still get weak gradients."
        )
    if (
        ctx.stage == "teacher_offline"
        and knobs.teacher_projector_force_full_commit
        and not knobs.teacher_projector_soft_pause_selection
    ):
        warnings.append(
            "pause-recall auxiliaries are enabled during teacher_offline, but the teacher projector still runs "
            "with force_full_commit and no soft pause-selection override; sparse support losers may receive little "
            "gradient. Consider setting rhythm_teacher_projector_soft_pause_selection: true for recall ablations."
        )
    if (
        knobs.pause_boundary_weight >= 0.45
        and knobs.pause_boundary_bias_weight >= 0.20
    ):
        warnings.append(
            "pause-recall auxiliaries are enabled while both rhythm_pause_boundary_weight and "
            "rhythm_projector_pause_boundary_bias_weight remain aggressive; if recall stays low, inspect "
            "pre-vs-post-projector recall before increasing boundary weighting further."
        )


def _validate_supervision_overlap(
    ctx: RhythmStageValidationContext,
    errors: list[str],
    warnings: list[str],
) -> None:
    knobs = ctx.knobs
    if knobs.distill_allocation_weight > 0.0 and knobs.shape_distill_active:
        message = (
            "rhythm_distill_allocation_weight > 0 while shape distillation is also enabled; "
            "this double-constrains the same mass split and is not part of the maintained mainline."
        )
        if knobs.strict_mainline:
            errors.append(message)
        else:
            warnings.append(message)
    if knobs.lambda_plan > 0.0:
        warnings.append(
            "lambda_rhythm_plan > 0 re-enables the optional planner proxy loss; maintained mainline keeps it at 0."
        )
        if knobs.primary_target_surface == "teacher":
            warnings.append(
                "lambda_rhythm_plan > 0 with rhythm_primary_target_surface: teacher duplicates teacher timing-shape supervision; "
                "keep it for ablations only."
            )
        if knobs.lambda_distill > 0.0 and knobs.distill_allocation_weight > 0.0:
            warnings.append(
                "lambda_rhythm_plan > 0 together with distill allocation reintroduces duplicate total-allocation supervision."
            )


def _validate_source_boundary_controls(
    ctx: RhythmStageValidationContext,
    errors: list[str],
    warnings: list[str],
) -> None:
    knobs = ctx.knobs
    for name, value in {
        "rhythm_source_boundary_scale": knobs.source_boundary_scale,
        "rhythm_source_boundary_scale_train_start": knobs.source_boundary_scale_train_start,
        "rhythm_source_boundary_scale_train_end": knobs.source_boundary_scale_train_end,
        "rhythm_teacher_source_boundary_scale": knobs.teacher_source_boundary_scale,
    }.items():
        if value < 0.0:
            errors.append(f"{name} must be >= 0.")
    if knobs.source_boundary_scale_anneal_steps < 0:
        errors.append("rhythm_source_boundary_scale_anneal_steps must be >= 0.")
    if knobs.source_boundary_scale_warmup_steps < 0:
        errors.append("rhythm_source_boundary_scale_warmup_steps must be >= 0.")
    if knobs.source_boundary_scale_train_start < knobs.source_boundary_scale_train_end:
        warnings.append(
            "source-boundary prior anneal is configured weak->strong; maintained path usually uses strong->soft."
        )


def _validate_retimed_target_controls(
    ctx: RhythmStageValidationContext,
    errors: list[str],
    warnings: list[str],
) -> None:
    knobs = ctx.knobs
    if knobs.retimed_target_start_steps < 0 or knobs.online_retimed_target_start_steps < 0:
        errors.append(
            "rhythm_retimed_target_start_steps / rhythm_online_retimed_target_start_steps must be >= 0."
        )
    if knobs.online_retimed_target_start_steps < knobs.retimed_target_start_steps:
        warnings.append(
            "rhythm_online_retimed_target_start_steps < rhythm_retimed_target_start_steps; online target switch may start earlier than retimed stage."
        )
    if (
        knobs.has_train_or_valid_override
        and not knobs.use_retimed_pitch_target
        and not knobs.disable_pitch_when_retimed
    ):
        errors.append(
            "Retimed train/valid rendering must either enable rhythm_use_retimed_pitch_target "
            "or set rhythm_disable_pitch_loss_when_retimed: true."
        )
    if knobs.retimed_mel_source == "teacher" and not knobs.binarize_teacher_targets:
        errors.append("Teacher retimed targets require rhythm_binarize_teacher_targets: true.")


def _validate_reporting_and_export_hints(
    ctx: RhythmStageValidationContext,
    warnings: list[str],
) -> None:
    knobs = ctx.knobs
    if knobs.export_cache_audit_to_sample:
        warnings.append(
            "rhythm_export_cache_audit_to_sample=true adds cache appendix fields to runtime batch; keep it off outside audits."
        )
    if knobs.public_losses:
        missing_public = sorted(_REQUIRED_PUBLIC_LOSSES.difference(set(knobs.public_losses)))
        if missing_public:
            warnings.append(
                f"rhythm_public_losses is missing maintained mainline aliases: {missing_public}."
            )
        if knobs.pause_allocation_weight > 0.0 and "L_pause_allocation" not in set(knobs.public_losses):
            warnings.append(
                "rhythm_pause_allocation_weight > 0 but rhythm_public_losses omits L_pause_allocation; "
                "the structural pause-allocation diagnostic term will be harder to monitor in logs."
            )


def _validate_strict_mainline(
    ctx: RhythmStageValidationContext,
    errors: list[str],
) -> None:
    knobs = ctx.knobs
    if not knobs.strict_mainline:
        return
    if knobs.lambda_guidance > 0.0:
        errors.append("rhythm_strict_mainline requires lambda_rhythm_guidance: 0.")
    if knobs.lambda_distill > 0.0 and knobs.distill_surface != "cache":
        errors.append(
            "rhythm_strict_mainline only allows cached teacher distillation; "
            "set rhythm_distill_surface: cache or disable lambda_rhythm_distill."
        )
    if bool(ctx.hparams.get("rhythm_enable_algorithmic_teacher", False)):
        errors.append("rhythm_strict_mainline requires rhythm_enable_algorithmic_teacher: false.")
    if knobs.enable_dual_mode_teacher:
        errors.append("rhythm_strict_mainline requires rhythm_enable_dual_mode_teacher: false.")
    if knobs.enable_learned_offline_teacher:
        errors.append(
            "rhythm_strict_mainline requires rhythm_enable_learned_offline_teacher: false."
        )
    if knobs.runtime_offline_teacher_enabled:
        errors.append(
            "rhythm_strict_mainline requires rhythm_runtime_enable_learned_offline_teacher: false."
        )
    if knobs.lambda_teacher_aux > 0.0:
        errors.append("rhythm_strict_mainline requires lambda_rhythm_teacher_aux: 0.")
    if knobs.lambda_plan > 0.0:
        errors.append("rhythm_strict_mainline requires lambda_rhythm_plan: 0.")
    if knobs.teacher_as_main:
        errors.append("rhythm_strict_mainline requires rhythm_teacher_as_main: false.")


def _validate_stage_hints(
    ctx: RhythmStageValidationContext,
    errors: list[str],
    warnings: list[str],
) -> None:
    knobs = ctx.knobs
    if ctx.stage == "transitional":
        warnings.append(
            "transitional is a migration/debug stage. Prefer teacher_offline / student_kd / student_retimed for maintained runs."
        )
    if knobs.lambda_distill > 0.0:
        if knobs.distill_surface == "none":
            errors.append("lambda_rhythm_distill > 0 but rhythm_distill_surface disables distillation.")
        if knobs.distill_surface == "offline" and not knobs.enable_dual_mode_teacher:
            errors.append("Offline distillation requires rhythm_enable_dual_mode_teacher: true.")
        if knobs.distill_surface == "cache" and not knobs.require_cached_teacher:
            warnings.append(
                "Cached teacher distillation is enabled without rhythm_require_cached_teacher: true; "
                "maintained student-only KD should require cache-backed teacher surfaces."
            )
    if knobs.enable_dual_mode_teacher and knobs.lambda_distill <= 0.0:
        warnings.append("Dual-mode teacher is enabled but lambda_rhythm_distill == 0.")
    if (
        knobs.primary_target_surface == "teacher"
        and not knobs.binarize_teacher_targets
        and knobs.target_mode != "runtime_only"
    ):
        warnings.append(
            "Primary rhythm target surface is teacher but rhythm_binarize_teacher_targets is false."
        )
    if ctx.stage in _MAINLINE_STAGES and knobs.lambda_plan > 0.0:
        warnings.append(
            f"{ctx.stage} usually keeps lambda_rhythm_plan: 0; treat rhythm_plan as an ablation/debug term rather than the maintained objective."
        )


def _validate_external_reference_supervision_consistency(
    ctx: RhythmStageValidationContext,
    errors: list[str],
    warnings: list[str],
) -> None:
    policy = str(ctx.hparams.get("rhythm_cached_reference_policy", "self") or "self").strip().lower()
    if policy not in {"sample_ref", "paired", "external"}:
        return

    knobs = ctx.knobs
    teacher_source = normalize_teacher_target_source(
        ctx.hparams.get("rhythm_teacher_target_source", "algorithmic")
    )
    if knobs.target_mode != "runtime_only":
        errors.append(
            "External rhythm reference policy currently requires rhythm_dataset_target_mode: runtime_only. "
            "cached_only/prefer_cache would pair sampled-ref conditioning with source-item cached surfaces and "
            "create supervision conflict that teaches the model to ignore the reference."
        )
    else:
        if bool(ctx.hparams.get("rhythm_binarize_teacher_targets", False)):
            warnings.append(
                "runtime_only + external rhythm reference usually should not keep rhythm_binarize_teacher_targets=true. "
                "That precomputes self-conditioned teacher surfaces that this stage does not consume and can blur the intended training semantics."
            )
        if bool(ctx.hparams.get("rhythm_binarize_retimed_mel_targets", False)):
            warnings.append(
                "runtime_only + external rhythm reference usually should not keep rhythm_binarize_retimed_mel_targets=true. "
                "This stage is module-only and does not need cached retimed acoustic targets."
            )
    if knobs.target_mode == "runtime_only" and knobs.primary_target_surface == "teacher" and teacher_source != "algorithmic":
        errors.append(
            "runtime_only + external rhythm reference currently requires rhythm_teacher_target_source: algorithmic. "
            "learned_offline teacher surfaces are cache-backed and cannot be regenerated online from the sampled ref."
        )


def validate_general_stage_rules(
    ctx: RhythmStageValidationContext,
    errors: list[str],
    warnings: list[str],
) -> None:
    _validate_cache_version(ctx, errors)
    _validate_nonnegative_weights(ctx, errors)
    _validate_objective_activation(ctx, errors)
    _validate_inactive_distill_config_clutter(ctx, warnings)
    validate_pause_boundary_alias_consistency(ctx.hparams, errors, warnings)
    validate_prefix_lambda_alias_consistency(ctx.hparams, errors, warnings)
    _validate_runtime_teacher_configuration(ctx, errors, warnings)
    _validate_projector_schedule_ranges(ctx, errors, warnings)
    _validate_pause_recall_support_controls(ctx, warnings)
    _validate_supervision_overlap(ctx, errors, warnings)
    _validate_source_boundary_controls(ctx, errors, warnings)
    _validate_retimed_target_controls(ctx, errors, warnings)
    _validate_reporting_and_export_hints(ctx, warnings)
    _validate_strict_mainline(ctx, errors)
    _validate_stage_hints(ctx, errors, warnings)
    _validate_external_reference_supervision_consistency(ctx, errors, warnings)


__all__ = ["validate_general_stage_rules"]
