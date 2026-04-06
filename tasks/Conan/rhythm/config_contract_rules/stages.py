from __future__ import annotations

from collections.abc import Callable

from .context import RhythmStageValidationContext
from .stage_legacy import validate_legacy_dual_mode_kd, validate_legacy_schedule_only
from .stage_mainline import (
    validate_student_kd,
    validate_student_ref_bootstrap,
    validate_student_retimed,
    validate_teacher_offline,
)

StageValidator = Callable[[RhythmStageValidationContext, list[str], list[str]], None]

MAINTAINED_STAGE_VALIDATORS: dict[str, StageValidator] = {
    "teacher_offline": validate_teacher_offline,
    "student_kd": validate_student_kd,
    "student_retimed": validate_student_retimed,
}

LEGACY_STAGE_VALIDATORS: dict[str, StageValidator] = {
    "legacy_schedule_only": validate_legacy_schedule_only,
    "legacy_dual_mode_kd": validate_legacy_dual_mode_kd,
}

SPECIAL_STAGE_VALIDATORS: dict[str, StageValidator] = {
    "student_ref_bootstrap": validate_student_ref_bootstrap,
}

STAGE_VALIDATORS: dict[str, StageValidator] = {
    **MAINTAINED_STAGE_VALIDATORS,
    **LEGACY_STAGE_VALIDATORS,
    **SPECIAL_STAGE_VALIDATORS,
}


def validate_stage_specific_rules(
    ctx: RhythmStageValidationContext,
    errors: list[str],
    warnings: list[str],
) -> None:
    validator = STAGE_VALIDATORS.get(ctx.stage)
    if validator is None:
        return
    validator(ctx, errors, warnings)


__all__ = [
    "MAINTAINED_STAGE_VALIDATORS",
    "LEGACY_STAGE_VALIDATORS",
    "SPECIAL_STAGE_VALIDATORS",
    "STAGE_VALIDATORS",
    "validate_stage_specific_rules",
]
