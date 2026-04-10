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
from modules.Conan.rhythm_v3.hparam_aliases import (
    resolve_progress_bins,
    resolve_progress_support_tau,
)
from tasks.Conan.rhythm.config_contract import (
    collect_config_contract_evaluation,
)

_DEPRECATED_V3_HPARAM_RENAMES = {
    "lambda_rhythm_mem": "lambda_rhythm_op",
    "lambda_rhythm_anti": "lambda_rhythm_zero",
}
_REMOVED_V3_HPARAMS = (
    "rhythm_v3_mem",
    "rhythm_v3_anti",
    "rhythm_v3_allow_proxy_infer",
    "rhythm_role_dim",
    "rhythm_role_codebook_size",
    "rhythm_anti_pos_bins",
    "rhythm_anti_pos_grl_scale",
    "rhythm_baseline_struct_enable",
    "rhythm_baseline_struct_scale_init",
)
_REQUIRED_V3_PUBLIC_INPUTS = (
    "content_units",
    "dur_anchor_src",
    "unit_anchor_base",
    "prompt_content_units",
    "prompt_duration_obs",
    "prompt_unit_mask",
)
_FORBIDDEN_V3_PUBLIC_INPUTS = (
    "ref_rhythm_stats",
    "ref_rhythm_trace",
)
_REQUIRED_V3_PUBLIC_OUTPUTS = (
    "speech_duration_exec",
    "rhythm_frame_plan",
    "commit_frontier",
    "rhythm_state_next",
)
_FORBIDDEN_V3_PUBLIC_OUTPUTS = ("pause_after_exec",)
_REQUIRED_V3_PUBLIC_LOSSES = (
    "rhythm_total",
    "rhythm_v3_dur",
    "rhythm_v3_op",
    "rhythm_v3_pref",
    "rhythm_v3_zero",
)
_FORBIDDEN_V3_PUBLIC_LOSSES = (
    "L_exec_speech",
    "L_exec_stretch",
    "L_prefix_state",
    "L_rhythm_exec",
    "L_stream_state",
    "rhythm_v3_break",
)


def _normalize_public_surface(value, *, key: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = tuple(part.strip() for part in value.split(",") if part.strip())
    elif isinstance(value, (list, tuple, set)):
        normalized = tuple(str(part).strip() for part in value if str(part).strip())
    else:
        raise ValueError(f"{key} must be a list/tuple/set or comma-separated string for rhythm_v3.")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{key} contains duplicate entries for rhythm_v3.")
    return normalized


def _validate_required_public_surface(
    values: tuple[str, ...] | None,
    *,
    key: str,
    required: tuple[str, ...],
    forbidden: tuple[str, ...] = (),
) -> None:
    if values is None:
        return
    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError(f"{key} is missing required rhythm_v3 entries: {missing}")
    stale = [name for name in forbidden if name in values]
    if stale:
        raise ValueError(f"{key} contains legacy rhythm_v2/slot entries that are removed for rhythm_v3: {stale}")


def _normalize_optional_path(value, *, key: str) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    return normalized


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
        baseline_train_mode = str(hparams.get("rhythm_v3_baseline_train_mode", "joint") or "joint").strip().lower()
        if baseline_train_mode not in {"joint", "frozen", "pretrain"}:
            raise ValueError("rhythm_v3_baseline_train_mode must be one of: joint, frozen, pretrain")
        baseline_target_mode = str(
            hparams.get("rhythm_v3_baseline_target_mode", "deglobalized") or "deglobalized"
        ).strip().lower()
        if baseline_target_mode not in {"raw", "deglobalized"}:
            raise ValueError("rhythm_v3_baseline_target_mode must be one of: raw, deglobalized")
        _normalize_optional_path(hparams.get("rhythm_v3_baseline_ckpt"), key="rhythm_v3_baseline_ckpt")
        _normalize_optional_path(
            hparams.get("rhythm_baseline_table_prior_path"),
            key="rhythm_baseline_table_prior_path",
        )
        if "rhythm_v3_ablation" in hparams:
            raise ValueError(
                "rhythm_v3_ablation has been removed. "
                "Use rhythm_v3_backbone/rhythm_v3_warp_mode/rhythm_v3_allow_hybrid instead."
            )
        backbone_mode = str(hparams.get("rhythm_v3_backbone", "global_only") or "global_only").strip().lower()
        warp_mode = str(hparams.get("rhythm_v3_warp_mode", "none") or "none").strip().lower()
        allow_hybrid = bool(hparams.get("rhythm_v3_allow_hybrid", False))
        if backbone_mode not in {"global_only", "operator"}:
            raise ValueError("rhythm_v3_backbone must be one of: global_only, operator")
        if warp_mode not in {"none", "progress", "detector"}:
            raise ValueError("rhythm_v3_warp_mode must be one of: none, progress, detector")
        if backbone_mode == "global_only" and allow_hybrid:
            raise ValueError("rhythm_v3_allow_hybrid is only valid when rhythm_v3_backbone='operator'.")
        if backbone_mode == "operator" and warp_mode == "progress" and not allow_hybrid:
            raise ValueError(
                "rhythm_v3_backbone='operator' with rhythm_v3_warp_mode='progress' "
                "requires rhythm_v3_allow_hybrid=true."
            )
        if backbone_mode == "operator" and warp_mode == "detector":
            raise ValueError(
                "rhythm_v3_warp_mode='detector' is only valid with rhythm_v3_backbone='global_only'."
            )
        source_residual_gain = float(hparams.get("rhythm_v3_source_residual_gain", 0.0) or 0.0)
        if source_residual_gain > 0.0 and backbone_mode != "operator":
            raise ValueError(
                "rhythm_v3_source_residual_gain requires rhythm_v3_backbone='operator'."
            )
        if source_residual_gain > 0.0 and warp_mode == "progress":
            raise ValueError(
                "rhythm_v3_source_residual_gain cannot be combined with hybrid operator+progress warp. "
                "Disable progress warp or set source residual gain back to 0."
            )
        runtime_mode = (
            "progress_only" if backbone_mode == "global_only" and warp_mode == "progress"
            else "detector_only" if backbone_mode == "global_only" and warp_mode == "detector"
            else "global_only" if backbone_mode == "global_only"
            else "operator_progress" if warp_mode == "progress"
            else "operator_srcres" if source_residual_gain > 0.0
            else "operator"
        )
        for key in (
            "lambda_rhythm_dur",
            "lambda_rhythm_op",
            "lambda_rhythm_pref",
            "lambda_rhythm_base",
            "lambda_rhythm_cons",
            "lambda_rhythm_zero",
            "lambda_rhythm_ortho",
            "rhythm_global_shrink_tau",
            "rhythm_operator_support_tau",
            "rhythm_operator_holdout_ratio",
            "rhythm_operator_min_support_factor",
        ):
            if float(hparams.get(key, 0.0) or 0.0) < 0.0:
                raise ValueError(f"{key} must be >= 0 for rhythm_v3.")
        if resolve_progress_support_tau(hparams, default=8.0) < 0.0:
            raise ValueError("rhythm_progress_support_tau / rhythm_coarse_support_tau must be >= 0 for rhythm_v3.")
        if resolve_progress_bins(hparams, default=4) <= 0:
            raise ValueError("rhythm_progress_bins / rhythm_coarse_bins must be > 0 for rhythm_v3.")
        holdout_ratio = float(hparams.get("rhythm_operator_holdout_ratio", 0.30) or 0.0)
        if holdout_ratio >= 1.0:
            raise ValueError("rhythm_operator_holdout_ratio must be < 1 for rhythm_v3.")
        if baseline_train_mode == "pretrain" and float(hparams.get("lambda_rhythm_base", 0.0) or 0.0) <= 0.0:
            raise ValueError("rhythm_v3_baseline_train_mode='pretrain' requires lambda_rhythm_base > 0.")
        source_residual_gain = float(hparams.get("rhythm_v3_source_residual_gain", 0.0) or 0.0)
        if source_residual_gain < 0.0:
            raise ValueError("rhythm_v3_source_residual_gain must be >= 0 for rhythm_v3.")
        if runtime_mode in {"global_only", "progress_only", "detector_only"}:
            for key in ("lambda_rhythm_op", "lambda_rhythm_zero", "lambda_rhythm_ortho"):
                if float(hparams.get(key, 0.0) or 0.0) > 0.0:
                    raise ValueError(f"{key} must be 0 when rhythm_v3 runtime mode is '{runtime_mode}'.")
            if source_residual_gain > 0.0:
                raise ValueError(
                    f"rhythm_v3_source_residual_gain must be 0 when rhythm_v3 runtime mode is '{runtime_mode}'."
                )
        elif runtime_mode != "operator_srcres" and source_residual_gain > 0.0:
            raise ValueError(
                "rhythm_v3_source_residual_gain is only valid when rhythm_v3 runtime mode is 'operator_srcres'."
            )
        public_inputs = _normalize_public_surface(hparams.get("rhythm_public_inputs"), key="rhythm_public_inputs")
        public_outputs = _normalize_public_surface(hparams.get("rhythm_public_outputs"), key="rhythm_public_outputs")
        public_losses = _normalize_public_surface(hparams.get("rhythm_public_losses"), key="rhythm_public_losses")
        _validate_required_public_surface(
            public_inputs,
            key="rhythm_public_inputs",
            required=_REQUIRED_V3_PUBLIC_INPUTS,
            forbidden=_FORBIDDEN_V3_PUBLIC_INPUTS,
        )
        _validate_required_public_surface(
            public_outputs,
            key="rhythm_public_outputs",
            required=_REQUIRED_V3_PUBLIC_OUTPUTS,
            forbidden=_FORBIDDEN_V3_PUBLIC_OUTPUTS,
        )
        _validate_required_public_surface(
            public_losses,
            key="rhythm_public_losses",
            required=_REQUIRED_V3_PUBLIC_LOSSES,
            forbidden=_FORBIDDEN_V3_PUBLIC_LOSSES,
        )
        if public_losses is not None:
            if float(hparams.get("lambda_rhythm_base", 0.0) or 0.0) > 0.0 and "rhythm_v3_base" not in public_losses:
                raise ValueError("rhythm_public_losses must include rhythm_v3_base when lambda_rhythm_base > 0.")
            if float(hparams.get("lambda_rhythm_cons", 0.0) or 0.0) > 0.0 and "rhythm_v3_cons" not in public_losses:
                raise ValueError("rhythm_public_losses must include rhythm_v3_cons when lambda_rhythm_cons > 0.")
            if float(hparams.get("lambda_rhythm_ortho", 0.0) or 0.0) > 0.0 and "rhythm_v3_ortho" not in public_losses:
                raise ValueError("rhythm_public_losses must include rhythm_v3_ortho when lambda_rhythm_ortho > 0.")
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
