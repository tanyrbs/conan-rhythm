from __future__ import annotations

"""Public facade for rhythm profile/stage contract validation."""

from typing import Any, Mapping

from .config_contract_core import RhythmContractValidationResult
from .config_contract_rules import (
    detect_rhythm_profile,
    validate_general_stage_rules,
    validate_profile_contract,
    validate_stage_post_rules,
    validate_stage_specific_rules,
)
from .config_contract_rules.context import build_stage_validation_context


def validate_stage_contract(
    hparams: Mapping[str, Any],
    *,
    config_path: str | None = None,
) -> tuple[str, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    ctx = build_stage_validation_context(hparams, config_path=config_path)
    validate_general_stage_rules(ctx, errors, warnings)
    validate_stage_specific_rules(ctx, errors, warnings)
    validate_stage_post_rules(ctx, warnings)
    return ctx.stage, errors, warnings


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


__all__ = [
    "collect_rhythm_contract_issues",
    "detect_rhythm_profile",
    "validate_profile_contract",
    "validate_stage_contract",
]
