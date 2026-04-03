from __future__ import annotations

from modules.Conan.rhythm.policy import (
    normalize_distill_surface,
    normalize_primary_target_surface,
    normalize_retimed_target_mode,
    normalize_rhythm_target_mode,
    parse_optional_bool,
    resolve_pause_boundary_weight,
)
from tasks.Conan.rhythm.config_contract import (
    collect_config_contract_evaluation,
)


def validate_rhythm_training_hparams(hparams) -> None:
    if not bool(hparams.get("rhythm_enable_v2", False)):
        return
    report = collect_config_contract_evaluation(hparams, model_dry_run=False).report
    if report.errors:
        raise ValueError("Invalid Rhythm V2 training config:\n- " + "\n- ".join(report.errors))
    if report.warnings:
        print("| Rhythm V2 config warnings:")
        for warning in report.warnings:
            print(f"|   - {warning}")


def resolve_task_pause_boundary_weight(hparams) -> float:
    return resolve_pause_boundary_weight(hparams)


def parse_task_optional_bool(value):
    return parse_optional_bool(value)


def resolve_task_target_mode(hparams) -> str:
    return normalize_rhythm_target_mode(hparams.get("rhythm_dataset_target_mode", "prefer_cache"))


def resolve_task_primary_target_surface(hparams) -> str:
    return normalize_primary_target_surface(hparams.get("rhythm_primary_target_surface", "guidance"))


def resolve_task_distill_surface(hparams) -> str:
    return normalize_distill_surface(hparams.get("rhythm_distill_surface", "auto"))


def resolve_task_retimed_target_mode(hparams) -> str:
    return normalize_retimed_target_mode(hparams.get("rhythm_retimed_target_mode", "cached"))
