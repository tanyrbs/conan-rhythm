from __future__ import annotations

from typing import Any, Mapping


def resolve_duplicate_primary_distill_dedupe_flag(
    hparams: Mapping[str, Any],
    *,
    default: bool = True,
) -> bool:
    if "rhythm_dedupe_teacher_primary_cache_distill" in hparams:
        return bool(hparams.get("rhythm_dedupe_teacher_primary_cache_distill", default))
    return bool(hparams.get("rhythm_suppress_duplicate_primary_distill", default))


def validate_pause_boundary_alias_consistency(
    hparams: Mapping[str, Any],
    errors: list[str],
    warnings: list[str],
) -> None:
    has_legacy = "rhythm_pause_exec_boundary_boost" in hparams
    has_public = "rhythm_pause_boundary_weight" in hparams
    if has_legacy and has_public:
        legacy_pause_boundary = float(hparams.get("rhythm_pause_exec_boundary_boost", 0.75))
        public_pause_boundary = float(hparams.get("rhythm_pause_boundary_weight", 0.35))
        if abs(legacy_pause_boundary - public_pause_boundary) > 1e-8:
            errors.append(
                "rhythm_pause_exec_boundary_boost and rhythm_pause_boundary_weight are both set but disagree; "
                "remove the legacy key."
            )
        else:
            warnings.append(
                "Both rhythm_pause_exec_boundary_boost and rhythm_pause_boundary_weight are set; "
                "prefer rhythm_pause_boundary_weight and remove the legacy key."
            )
    elif has_legacy:
        warnings.append(
            "rhythm_pause_exec_boundary_boost is a legacy alias; prefer rhythm_pause_boundary_weight."
        )


def validate_prefix_lambda_alias_consistency(
    hparams: Mapping[str, Any],
    errors: list[str],
    warnings: list[str],
) -> None:
    if "lambda_rhythm_carry" not in hparams or "lambda_rhythm_cumplan" not in hparams:
        return
    lambda_carry = float(hparams.get("lambda_rhythm_carry", 0.0))
    lambda_cumplan = float(hparams.get("lambda_rhythm_cumplan", 0.0))
    if abs(lambda_carry - lambda_cumplan) > 1e-8:
        errors.append("lambda_rhythm_carry and lambda_rhythm_cumplan are both set but disagree.")
    else:
        warnings.append(
            "Both lambda_rhythm_carry and lambda_rhythm_cumplan are set; prefer lambda_rhythm_cumplan."
        )


def warn_legacy_min_unit_frames(
    hparams: Mapping[str, Any],
    warnings: list[str],
) -> None:
    if "rhythm_min_unit_frames" not in hparams:
        return
    legacy_min_unit = hparams.get("rhythm_min_unit_frames")
    if legacy_min_unit in {None, "", 0, 0.0, "0", "0.0"}:
        return
    warnings.append(
        "rhythm_min_unit_frames is a legacy/unsupported knob in the maintained path; "
        "remove it instead of assuming it is active."
    )


__all__ = [
    "resolve_duplicate_primary_distill_dedupe_flag",
    "validate_pause_boundary_alias_consistency",
    "validate_prefix_lambda_alias_consistency",
    "warn_legacy_min_unit_frames",
]
