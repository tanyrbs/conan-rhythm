from __future__ import annotations

from typing import Any, Mapping

from modules.Conan.rhythm.policy import (
    build_rhythm_hparams_policy,
    normalize_distill_surface,
    resolve_cumplan_lambda,
)
from modules.Conan.rhythm.stages import normalize_rhythm_stage

from .compat import warn_legacy_min_unit_frames


def detect_rhythm_profile(hparams: Mapping[str, Any], config_path: str | None = None) -> str:
    explicit_stage = hparams.get("rhythm_stage", None)
    stage_is_minimal = False
    if explicit_stage not in {None, ""}:
        stage_is_minimal = normalize_rhythm_stage(explicit_stage) == "minimal_v1"
    if stage_is_minimal:
        return "minimal_v1"

    config_name = str(config_path or "").lower()
    is_minimal_flag = (
        bool(hparams.get("rhythm_minimal_v1_profile", False))
        or "minimal_v1" in config_name
        or "cached_only" in config_name
    )
    if (
        is_minimal_flag
        and not bool(hparams.get("rhythm_enable_dual_mode_teacher", False))
        and not bool(hparams.get("rhythm_apply_train_override", False))
        and not bool(hparams.get("rhythm_apply_valid_override", False))
        and normalize_distill_surface(
            str(hparams.get("rhythm_distill_surface", "auto") or "auto").strip().lower()
        )
        == "none"
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
    if bool(hparams.get("rhythm_apply_train_override", False)) or bool(
        hparams.get("rhythm_apply_valid_override", False)
    ):
        errors.append("minimal_v1 profile should keep train/valid on source-aligned canvas.")

    if float(hparams.get("lambda_rhythm_exec_speech", 0.0)) <= 0.0:
        errors.append("minimal_v1 profile requires lambda_rhythm_exec_speech > 0.")
    if float(hparams.get("lambda_rhythm_exec_pause", 0.0)) <= 0.0:
        errors.append("minimal_v1 profile requires lambda_rhythm_exec_pause > 0.")
    if resolve_cumplan_lambda(hparams) <= 0.0:
        errors.append(
            "minimal_v1 profile requires lambda_rhythm_cumplan (or lambda_rhythm_carry) > 0."
        )
    if float(hparams.get("lambda_rhythm_budget", 0.0)) < 0.0:
        errors.append("minimal_v1 profile does not allow negative lambda_rhythm_budget.")
    elif float(hparams.get("lambda_rhythm_budget", 0.0)) == 0.0:
        warnings.append(
            "minimal_v1 profile keeps lambda_rhythm_budget at 0; maintained path expects a small positive budget guardrail."
        )
    if float(hparams.get("lambda_rhythm_guidance", 0.0)) > 0.0:
        errors.append("minimal_v1 profile requires lambda_rhythm_guidance: 0.")
    if float(hparams.get("lambda_rhythm_plan", 0.0)) > 0.0:
        errors.append("minimal_v1 profile requires lambda_rhythm_plan: 0.")
    if float(hparams.get("lambda_rhythm_distill", 0.0)) > 0.0:
        errors.append("minimal_v1 profile requires lambda_rhythm_distill: 0.")

    if not model_dry_run:
        warnings.append(
            "minimal_v1 train-ready preflight should include --model_dry_run before starting training."
        )
    warn_legacy_min_unit_frames(hparams, warnings)
    return profile, errors, warnings


__all__ = ["detect_rhythm_profile", "validate_profile_contract"]
