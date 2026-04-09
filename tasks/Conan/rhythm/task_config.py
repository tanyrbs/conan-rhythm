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
    rhythm_enable_v2 = bool(hparams.get("rhythm_enable_v2", False))
    rhythm_enable_v3 = bool(
        hparams.get("rhythm_enable_v3", False)
        or str(hparams.get("rhythm_mode", "") or "").strip().lower() == "duration_ref_memory"
    )
    if rhythm_enable_v2 and rhythm_enable_v3:
        raise ValueError("Enable only one rhythm backend: rhythm_enable_v2 or rhythm_enable_v3.")
    if rhythm_enable_v3 and not rhythm_enable_v2:
        if int(hparams.get("rhythm_role_codebook_size", 12) or 0) <= 0:
            raise ValueError("rhythm_role_codebook_size must be > 0 for rhythm_v3.")
        if int(hparams.get("rhythm_role_dim", 64) or 0) <= 0:
            raise ValueError("rhythm_role_dim must be > 0 for rhythm_v3.")
        for key in ("lambda_rhythm_dur", "lambda_rhythm_mem", "lambda_rhythm_pref", "lambda_rhythm_anti"):
            if float(hparams.get(key, 0.0) or 0.0) < 0.0:
                raise ValueError(f"{key} must be >= 0 for rhythm_v3.")
        return
    if not rhythm_enable_v2:
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
