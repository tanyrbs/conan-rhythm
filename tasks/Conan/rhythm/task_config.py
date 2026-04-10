from __future__ import annotations

from modules.Conan.rhythm.policy import (
    is_duration_operator_mode,
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

_DEPRECATED_V3_HPARAM_RENAMES = {
    "lambda_rhythm_mem": "lambda_rhythm_op",
    "lambda_rhythm_anti": "lambda_rhythm_zero",
}
_REMOVED_V3_HPARAMS = (
    "rhythm_anti_pos_bins",
    "rhythm_anti_pos_grl_scale",
)


def validate_rhythm_training_hparams(hparams) -> None:
    rhythm_enable_v2 = bool(hparams.get("rhythm_enable_v2", False))
    rhythm_enable_v3 = bool(
        hparams.get("rhythm_enable_v3", False)
        or is_duration_operator_mode(hparams.get("rhythm_mode", ""))
    )
    if rhythm_enable_v2 and rhythm_enable_v3:
        raise ValueError("Enable only one rhythm backend: rhythm_enable_v2 or rhythm_enable_v3.")
    if rhythm_enable_v3 and not rhythm_enable_v2:
        for old_key, new_key in _DEPRECATED_V3_HPARAM_RENAMES.items():
            if old_key in hparams:
                raise ValueError(f"{old_key} has been removed. Use {new_key} instead.")
        for removed_key in _REMOVED_V3_HPARAMS:
            if removed_key in hparams:
                raise ValueError(f"{removed_key} has been removed from rhythm_v3.")
        if int(hparams.get("rhythm_response_rank", 12) or 0) <= 0:
            raise ValueError("rhythm_response_rank must be > 0 for rhythm_v3.")
        for key in (
            "lambda_rhythm_dur",
            "lambda_rhythm_op",
            "lambda_rhythm_pref",
            "lambda_rhythm_cons",
            "lambda_rhythm_zero",
        ):
            if float(hparams.get(key, 0.0) or 0.0) < 0.0:
                raise ValueError(f"{key} must be >= 0 for rhythm_v3.")
        streaming_mode = str(hparams.get("rhythm_streaming_mode", "strict") or "strict").strip().lower()
        if streaming_mode not in {"strict", "micro_lookahead"}:
            raise ValueError("rhythm_streaming_mode must be one of: strict, micro_lookahead")
        response_window_right = int(hparams.get("rhythm_response_window_right", 0) or 0)
        micro_lookahead_units = hparams.get("rhythm_micro_lookahead_units")
        if streaming_mode == "strict":
            if response_window_right != 0:
                raise ValueError("rhythm_response_window_right must be 0 when rhythm_streaming_mode='strict'.")
            if micro_lookahead_units is not None and int(micro_lookahead_units) != 0:
                raise ValueError("rhythm_micro_lookahead_units must be 0/None when rhythm_streaming_mode='strict'.")
        else:
            effective_lookahead = response_window_right if micro_lookahead_units is None else int(micro_lookahead_units)
            if effective_lookahead <= 0:
                raise ValueError("micro_lookahead mode requires positive rhythm_micro_lookahead_units or rhythm_response_window_right.")
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
