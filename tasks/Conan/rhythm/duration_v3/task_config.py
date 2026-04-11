from __future__ import annotations

from modules.Conan.rhythm.policy import is_duration_operator_mode

from ..common.task_config import (
    _normalize_optional_path,
    _normalize_public_surface,
    _validate_required_public_surface,
)

_DEPRECATED_V3_HPARAM_RENAMES = {
    "lambda_rhythm_anti": "lambda_rhythm_zero",
}
_REMOVED_V3_HPARAMS = (
    "rhythm_v3_mem",
    "rhythm_v3_anti",
    "rhythm_v3_allow_proxy_infer",
    "rhythm_role_codebook_size",
    "rhythm_anti_pos_bins",
    "rhythm_anti_pos_grl_scale",
    "rhythm_baseline_struct_enable",
    "rhythm_baseline_struct_scale_init",
    "rhythm_coarse_bins",
    "rhythm_coarse_support_tau",
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
    "rhythm_v3_pref",
)
_FORBIDDEN_V3_PUBLIC_LOSSES = (
    "L_exec_speech",
    "L_exec_stretch",
    "L_prefix_state",
    "L_rhythm_exec",
    "L_stream_state",
    "rhythm_v3_break",
)


def normalize_duration_v3_backbone_mode(value) -> str:
    normalized = str(value or "global_only").strip().lower()
    if normalized == "role_memory":
        return "prompt_summary"
    return normalized


def is_duration_v3_prompt_summary_backbone(value) -> bool:
    return normalize_duration_v3_backbone_mode(value) == "prompt_summary"


def validate_duration_v3_training_hparams(hparams) -> None:
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
    backbone_mode = normalize_duration_v3_backbone_mode(hparams.get("rhythm_v3_backbone", "global_only"))
    warp_mode = str(hparams.get("rhythm_v3_warp_mode", "none") or "none").strip().lower()
    allow_hybrid = bool(hparams.get("rhythm_v3_allow_hybrid", False))
    if backbone_mode not in {"global_only", "operator", "prompt_summary"}:
        raise ValueError(
            "rhythm_v3_backbone must be one of: global_only, operator, prompt_summary (legacy alias: role_memory)"
        )
    if warp_mode not in {"none", "progress", "detector"}:
        raise ValueError("rhythm_v3_warp_mode must be one of: none, progress, detector")
    if backbone_mode == "global_only" and allow_hybrid:
        raise ValueError("rhythm_v3_allow_hybrid is only valid when rhythm_v3_backbone='operator'.")
    if backbone_mode == "prompt_summary":
        if warp_mode != "none":
            raise ValueError(
                "rhythm_v3_backbone='prompt_summary' (legacy alias: 'role_memory') only supports rhythm_v3_warp_mode='none'."
            )
        if allow_hybrid:
            raise ValueError(
                "rhythm_v3_allow_hybrid is not used when rhythm_v3_backbone='prompt_summary' (legacy alias: 'role_memory')."
            )
    if backbone_mode == "operator" and warp_mode == "progress" and not allow_hybrid:
        raise ValueError(
            "rhythm_v3_backbone='operator' with rhythm_v3_warp_mode='progress' "
            "requires rhythm_v3_allow_hybrid=true."
        )
    if backbone_mode == "operator" and warp_mode == "detector":
        raise ValueError("rhythm_v3_warp_mode='detector' is only valid with rhythm_v3_backbone='global_only'.")
    source_residual_gain = float(hparams.get("rhythm_v3_source_residual_gain", 0.0) or 0.0)
    if source_residual_gain > 0.0 and backbone_mode != "operator":
        raise ValueError("rhythm_v3_source_residual_gain requires rhythm_v3_backbone='operator'.")
    anchor_mode = str(hparams.get("rhythm_v3_anchor_mode", "baseline") or "baseline").strip().lower()
    if anchor_mode not in {"baseline", "source_observed"}:
        raise ValueError("rhythm_v3_anchor_mode must be one of: baseline, source_observed")
    if backbone_mode == "prompt_summary" and anchor_mode != "source_observed":
        raise ValueError(
            "rhythm_v3_backbone='prompt_summary' (legacy alias: 'role_memory') requires rhythm_v3_anchor_mode='source_observed'."
        )
    if source_residual_gain > 0.0 and warp_mode == "progress":
        raise ValueError(
            "rhythm_v3_source_residual_gain cannot be combined with hybrid operator+progress warp. "
            "Disable progress warp or set source residual gain back to 0."
        )
    runtime_mode = (
        "prompt_summary"
        if backbone_mode == "prompt_summary"
        else "progress_only"
        if backbone_mode == "global_only" and warp_mode == "progress"
        else "detector_only"
        if backbone_mode == "global_only" and warp_mode == "detector"
        else "global_only"
        if backbone_mode == "global_only"
        else "operator_progress"
        if warp_mode == "progress"
        else "operator_srcres"
        if source_residual_gain > 0.0
        else "operator"
    )
    lambda_bias = float(
        hparams.get(
            "lambda_rhythm_bias",
            0.20 if runtime_mode == "prompt_summary" else 0.0,
        )
        or 0.0
    )
    for key in (
        "lambda_rhythm_dur",
        "lambda_rhythm_op",
        "lambda_rhythm_mem",
        "lambda_rhythm_summary",
        "lambda_rhythm_bias",
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
    if float(hparams.get("rhythm_progress_support_tau", 8.0) or 0.0) < 0.0:
        raise ValueError("rhythm_progress_support_tau must be >= 0 for rhythm_v3.")
    if int(hparams.get("rhythm_progress_bins", 4) or 0) <= 0:
        raise ValueError("rhythm_progress_bins must be > 0 for rhythm_v3.")
    holdout_ratio = float(hparams.get("rhythm_operator_holdout_ratio", 0.30) or 0.0)
    if holdout_ratio >= 1.0:
        raise ValueError("rhythm_operator_holdout_ratio must be < 1 for rhythm_v3.")
    if baseline_train_mode == "pretrain" and float(hparams.get("lambda_rhythm_base", 0.0) or 0.0) <= 0.0:
        raise ValueError("rhythm_v3_baseline_train_mode='pretrain' requires lambda_rhythm_base > 0.")
    if source_residual_gain < 0.0:
        raise ValueError("rhythm_v3_source_residual_gain must be >= 0 for rhythm_v3.")
    if runtime_mode in {"global_only", "progress_only", "detector_only", "prompt_summary"}:
        for key in ("lambda_rhythm_op", "lambda_rhythm_mem", "lambda_rhythm_summary", "lambda_rhythm_zero", "lambda_rhythm_ortho"):
            value = float(hparams.get(key, 0.0) or 0.0)
            if runtime_mode == "prompt_summary" and key in {"lambda_rhythm_op", "lambda_rhythm_mem", "lambda_rhythm_summary"}:
                continue
            if value > 0.0:
                raise ValueError(f"{key} must be 0 when rhythm_v3 runtime mode is '{runtime_mode}'.")
        if source_residual_gain > 0.0:
            raise ValueError(
                f"rhythm_v3_source_residual_gain must be 0 when rhythm_v3 runtime mode is '{runtime_mode}'."
            )
    if runtime_mode != "prompt_summary" and lambda_bias > 0.0:
        raise ValueError("lambda_rhythm_bias is only valid when rhythm_v3_backbone='prompt_summary'.")
    if runtime_mode == "prompt_summary":
        if float(hparams.get("lambda_rhythm_zero", 0.0) or 0.0) > 0.0:
            raise ValueError("lambda_rhythm_zero must be 0 when rhythm_v3_backbone='prompt_summary'.")
        if float(hparams.get("lambda_rhythm_ortho", 0.0) or 0.0) > 0.0:
            raise ValueError("lambda_rhythm_ortho must be 0 when rhythm_v3_backbone='prompt_summary'.")
    elif (
        float(hparams.get("lambda_rhythm_mem", 0.0) or 0.0) > 0.0
        or float(hparams.get("lambda_rhythm_summary", 0.0) or 0.0) > 0.0
    ):
        raise ValueError(
            "lambda_rhythm_summary (legacy alias: lambda_rhythm_mem) is only valid when rhythm_v3_backbone='prompt_summary'."
        )
    elif runtime_mode != "operator_srcres" and source_residual_gain > 0.0:
        raise ValueError("rhythm_v3_source_residual_gain is only valid when rhythm_v3 runtime mode is 'operator_srcres'.")
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
        if runtime_mode == "prompt_summary":
            lambda_summary = max(
                float(hparams.get("lambda_rhythm_op", 0.0) or 0.0),
                float(hparams.get("lambda_rhythm_mem", 0.0) or 0.0),
                float(hparams.get("lambda_rhythm_summary", 0.0) or 0.0),
            )
            if lambda_summary > 0.0 and not any(key in public_losses for key in ("rhythm_v3_summary", "rhythm_v3_mem", "rhythm_v3_op")):
                raise ValueError(
                    "rhythm_public_losses must include rhythm_v3_summary "
                    "(legacy aliases: rhythm_v3_mem / rhythm_v3_op) when rhythm_v3_backbone='prompt_summary'."
                )
        else:
            if "rhythm_v3_op" not in public_losses:
                raise ValueError("rhythm_public_losses must include rhythm_v3_op for non-prompt-summary rhythm_v3 modes.")
            if float(hparams.get("lambda_rhythm_zero", 0.0) or 0.0) > 0.0 and "rhythm_v3_zero" not in public_losses:
                raise ValueError("rhythm_public_losses must include rhythm_v3_zero when lambda_rhythm_zero > 0.")
        if float(hparams.get("lambda_rhythm_base", 0.0) or 0.0) > 0.0 and "rhythm_v3_base" not in public_losses:
            raise ValueError("rhythm_public_losses must include rhythm_v3_base when lambda_rhythm_base > 0.")
        if lambda_bias > 0.0 and "rhythm_v3_bias" not in public_losses:
            raise ValueError("rhythm_public_losses must include rhythm_v3_bias when lambda_rhythm_bias > 0.")
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


def validate_rhythm_training_hparams(hparams) -> None:
    validate_duration_v3_training_hparams(hparams)


__all__ = [
    "is_duration_v3_prompt_summary_backbone",
    "normalize_duration_v3_backbone_mode",
    "validate_duration_v3_training_hparams",
    "validate_rhythm_training_hparams",
]
