from __future__ import annotations

"""Shared facade for rhythm config contract validation.

`config_contract.py` remains the stable import surface used by task config and
preflight code, while the implementation now lives in focused modules:

- config_contract_core.py: reports, contexts, merge helpers
- stage_policy_validation.py: profile/stage validation rules
- config_contract_cache_rules.py: cache field expectations and sample checks
"""

from typing import Any, Mapping

from modules.Conan.rhythm.policy import build_rhythm_hparams_policy

from .config_contract_cache_rules import (
    CORE_RHYTHM_FIELDS,
    collect_cache_contract_report,
    validate_cache_field_contract,
    validate_inspected_cache_items,
    validate_required_field_presence,
)
from .config_contract_core import (
    RhythmConfigContractContext,
    RhythmConfigContractEvaluation,
    RhythmConfigContractReport,
    RhythmContractValidationResult,
    merge_contract_reports,
)
from .stage_policy_validation import (
    collect_rhythm_contract_issues,
    detect_rhythm_profile,
    validate_profile_contract,
    validate_stage_contract,
)


def _build_stage_report(
    *,
    profile: str,
    stage: str,
    errors: list[str],
    warnings: list[str],
) -> RhythmConfigContractReport:
    return RhythmConfigContractReport(
        profile=profile,
        stage=stage,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


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
    profile, profile_errors, profile_warnings = validate_profile_contract(
        context.hparams,
        config_path=context.config_path,
        model_dry_run=context.model_dry_run,
    )
    stage, stage_errors, stage_warnings = validate_stage_contract(
        context.hparams,
        config_path=context.config_path,
    )
    return merge_contract_reports(
        _build_stage_report(
            profile=profile,
            stage=stage,
            errors=profile_errors,
            warnings=profile_warnings,
        ),
        _build_stage_report(
            profile=profile,
            stage=stage,
            errors=stage_errors,
            warnings=stage_warnings,
        ),
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
