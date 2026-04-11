from __future__ import annotations

from modules.Conan.rhythm.policy import is_duration_operator_mode

from .config_contract import collect_config_contract_evaluation
from .common.task_config import (
    parse_task_optional_bool,
    resolve_task_distill_surface,
    resolve_task_pause_boundary_weight,
    resolve_task_primary_target_surface,
    resolve_task_retimed_target_mode,
    resolve_task_target_mode,
)
from .duration_v3.task_config import (
    is_duration_v3_prompt_summary_backbone,
    normalize_duration_v3_backbone_mode,
    validate_duration_v3_training_hparams,
)
from .rhythm_v2.task_config import validate_rhythm_v2_training_hparams


def validate_rhythm_training_hparams(hparams) -> None:
    rhythm_enable_v2 = bool(hparams.get("rhythm_enable_v2", False))
    rhythm_enable_v3 = bool(
        hparams.get("rhythm_enable_v3", False)
        or is_duration_operator_mode(hparams.get("rhythm_mode", ""))
    )
    if rhythm_enable_v2 and rhythm_enable_v3:
        raise ValueError("Enable only one rhythm backend: rhythm_enable_v2 or rhythm_enable_v3.")
    if rhythm_enable_v3 and not rhythm_enable_v2:
        return validate_duration_v3_training_hparams(hparams)
    if rhythm_enable_v2:
        return validate_rhythm_v2_training_hparams(hparams)
    return None


__all__ = [
    "collect_config_contract_evaluation",
    "is_duration_v3_prompt_summary_backbone",
    "normalize_duration_v3_backbone_mode",
    "parse_task_optional_bool",
    "resolve_task_distill_surface",
    "resolve_task_pause_boundary_weight",
    "resolve_task_primary_target_surface",
    "resolve_task_retimed_target_mode",
    "resolve_task_target_mode",
    "validate_duration_v3_training_hparams",
    "validate_rhythm_v2_training_hparams",
    "validate_rhythm_training_hparams",
]
