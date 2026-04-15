from __future__ import annotations

import json
import math
from pathlib import Path

from modules.Conan.rhythm.policy import is_duration_operator_mode

from ..common.task_config import (
    _normalize_optional_path,
    _normalize_public_surface,
    _validate_optional_existing_path,
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
_PROMPT_SUMMARY_REQUIRED_V3_PUBLIC_INPUTS = (
    "prompt_speech_mask",
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
_LEGACY_PROMPT_SUMMARY_BACKBONES = {"role_memory", "unit_run"}
_MINIMAL_V1_GLOBAL_BACKBONES = {"minimal_v1_global", "v1g_minimal"}
_UNIT_NORM_G_VARIANTS = {"unit_norm", "unit_normalized"}
_GATE_STATUS_FINGERPRINT_KEYS = (
    "rhythm_v3_g_variant",
    "rhythm_v3_g_trim_ratio",
    "rhythm_v3_drop_edge_runs_for_g",
    "rhythm_v3_min_boundary_confidence_for_g",
    "rhythm_v3_min_prompt_speech_ratio",
    "rhythm_v3_min_prompt_ref_len_sec",
    "rhythm_v3_max_prompt_ref_len_sec",
    "rhythm_v3_disallow_same_text_reference",
    "rhythm_v3_disallow_same_text_paired_target",
    "rhythm_v3_require_same_text_paired_target",
    "rhythm_v3_strict_eval_invalid_g",
    "rhythm_v3_alignment_prefilter_bad_samples",
    "rhythm_v3_alignment_prefilter_max_attempts",
    "rhythm_v3_alignment_unmatched_speech_ratio_max",
    "rhythm_v3_alignment_mean_local_confidence_speech_min",
    "rhythm_v3_alignment_mean_coarse_confidence_speech_min",
    "rhythm_v3_alignment_local_margin_p10_min",
    "rhythm_v3_prompt_domain_mode",
    "rhythm_v3_prompt_require_clean_support",
    "rhythm_v3_prompt_g_variant",
    "rhythm_v3_src_g_variant",
    "rhythm_v3_src_prefix_stat_mode",
    "rhythm_v3_src_prefix_min_support",
    "rhythm_v3_src_rate_init_mode",
    "rhythm_v3_use_src_gap_in_coarse_head",
    "rhythm_v3_analytic_gap_clip",
    "rhythm_v3_prefix_budget_pos",
    "rhythm_v3_prefix_budget_neg",
    "rhythm_v3_dynamic_budget_ratio",
    "rhythm_v3_min_prefix_budget",
    "rhythm_v3_max_prefix_budget",
    "rhythm_v3_budget_mode",
    "rhythm_v3_boundary_carry_decay",
    "rhythm_v3_boundary_offset_decay",
    "rhythm_v3_boundary_reset_thresh",
    "rhythm_v3_integer_projection_mode",
    "rhythm_v3_integer_projection_anchor_mode",
    "rhythm_v3_projection_repair_max_steps",
    "rhythm_v3_projection_repair_speech_bonus",
    "rhythm_v3_projection_repair_boundary_penalty",
    "rhythm_v3_use_continuous_alignment",
    "rhythm_v3_alignment_mode",
    "rhythm_v3_minimal_v1_profile",
    "rhythm_v3_strict_minimal_claim_profile",
)


def _is_unit_norm_variant(value) -> bool:
    return str(value or "raw_median").strip().lower() in _UNIT_NORM_G_VARIANTS


def normalize_duration_v3_backbone_mode(value) -> str:
    normalized = str(value or "global_only").strip().lower()
    if normalized in _LEGACY_PROMPT_SUMMARY_BACKBONES | _MINIMAL_V1_GLOBAL_BACKBONES:
        return "prompt_summary"
    return normalized


def normalize_duration_v3_rate_mode(value) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"", "none", "auto"}:
        return ""
    if normalized in {"simple", "simple_global", "global_stat", "global_stats"}:
        return "simple_global"
    if normalized in {"log_base", "normalized", "content_normalized"}:
        return "log_base"
    return normalized


def normalize_duration_v3_alignment_mode(value) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"", "auto"}:
        return "continuous_viterbi_v1"
    aliases = {
        "precomputed": "continuous_precomputed",
        "continuous": "continuous_viterbi_v1",
        "viterbi": "continuous_viterbi_v1",
        "run_state_viterbi": "continuous_viterbi_v1",
    }
    return aliases.get(normalized, normalized)


def _normalize_prompt_domain_mode(value) -> str:
    mode = str(value or "minimal_strict").strip().lower()
    aliases = {
        "strict": "minimal_strict",
        "meaningful": "meaningful_reference",
        "semantic": "meaningful_reference",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"minimal_strict", "meaningful_reference"}:
        raise ValueError(
            f"Unsupported rhythm_v3_prompt_domain_mode={value!r}. "
            "Expected one of: minimal_strict, meaningful_reference."
        )
    return mode


def resolve_duration_v3_rate_mode(hparams) -> str:
    explicit = normalize_duration_v3_rate_mode(hparams.get("rhythm_v3_rate_mode", ""))
    if explicit:
        return explicit
    if _is_enabled_flag(hparams.get("rhythm_v3_simple_global_stats", False)):
        return "simple_global"
    return "log_base"


def is_duration_v3_prompt_summary_backbone(value) -> bool:
    return normalize_duration_v3_backbone_mode(value) == "prompt_summary"


def _is_enabled_flag(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _get_hparam_alias(hparams, primary: str, alias: str, default=None):
    if primary in hparams:
        return hparams.get(primary)
    if alias in hparams:
        return hparams.get(alias)
    return default


_GATE_STATUS_BOOL_KEYS = {
    "rhythm_v3_alignment_prefilter_bad_samples",
    "rhythm_v3_disallow_same_text_paired_target",
    "rhythm_v3_disallow_same_text_reference",
    "rhythm_v3_prompt_require_clean_support",
    "rhythm_v3_require_same_text_paired_target",
    "rhythm_v3_strict_eval_invalid_g",
    "rhythm_v3_use_src_gap_in_coarse_head",
    "rhythm_v3_use_continuous_alignment",
    "rhythm_v3_minimal_v1_profile",
    "rhythm_v3_strict_minimal_claim_profile",
}
_GATE_STATUS_INT_KEYS = {
    "rhythm_v3_alignment_prefilter_max_attempts",
    "rhythm_v3_drop_edge_runs_for_g",
    "rhythm_v3_src_prefix_min_support",
    "rhythm_v3_prefix_budget_pos",
    "rhythm_v3_prefix_budget_neg",
    "rhythm_v3_min_prefix_budget",
    "rhythm_v3_max_prefix_budget",
    "rhythm_v3_projection_repair_max_steps",
}
_GATE_STATUS_FLOAT_KEYS = {
    "rhythm_v3_alignment_local_margin_p10_min",
    "rhythm_v3_alignment_mean_coarse_confidence_speech_min",
    "rhythm_v3_alignment_mean_local_confidence_speech_min",
    "rhythm_v3_alignment_unmatched_speech_ratio_max",
    "rhythm_v3_g_trim_ratio",
    "rhythm_v3_min_boundary_confidence_for_g",
    "rhythm_v3_max_prompt_ref_len_sec",
    "rhythm_v3_min_prompt_ref_len_sec",
    "rhythm_v3_min_prompt_speech_ratio",
    "rhythm_v3_analytic_gap_clip",
    "rhythm_v3_dynamic_budget_ratio",
    "rhythm_v3_boundary_carry_decay",
    "rhythm_v3_boundary_offset_decay",
    "rhythm_v3_boundary_reset_thresh",
    "rhythm_v3_projection_repair_speech_bonus",
    "rhythm_v3_projection_repair_boundary_penalty",
}


def _normalize_gate_fingerprint_value(key: str, value):
    if value is None:
        return None
    if key in _GATE_STATUS_BOOL_KEYS:
        return _is_enabled_flag(value)
    if key in _GATE_STATUS_INT_KEYS:
        return int(value)
    if key in _GATE_STATUS_FLOAT_KEYS:
        return float(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, (int, bool, str)) or value is None:
        return value
    return str(value)


def _resolve_gate_contract_hparam(hparams, key: str):
    if key == "rhythm_v3_g_variant":
        return str(hparams.get(key, "raw_median") or "raw_median")
    if key == "rhythm_v3_g_trim_ratio":
        return float(hparams.get(key, 0.2) or 0.2)
    if key == "rhythm_v3_drop_edge_runs_for_g":
        return int(hparams.get(key, 0) or 0)
    if key == "rhythm_v3_min_boundary_confidence_for_g":
        return hparams.get(key, None)
    if key == "rhythm_v3_prompt_domain_mode":
        return _normalize_prompt_domain_mode(hparams.get(key, "minimal_strict"))
    if key == "rhythm_v3_prompt_require_clean_support":
        return _is_enabled_flag(hparams.get(key, True))
    if key == "rhythm_v3_prompt_g_variant":
        return str(hparams.get(key, hparams.get("rhythm_v3_g_variant", "raw_median")) or "raw_median")
    if key == "rhythm_v3_src_g_variant":
        return str(hparams.get(key, hparams.get("rhythm_v3_g_variant", "raw_median")) or "raw_median")
    if key == "rhythm_v3_src_prefix_stat_mode":
        return str(hparams.get(key, "ema") or "ema").strip().lower()
    if key == "rhythm_v3_src_prefix_min_support":
        return int(hparams.get(key, 3) or 3)
    if key == "rhythm_v3_src_rate_init_mode":
        return str(hparams.get(key, "first_speech") or "first_speech").strip().lower()
    if key == "rhythm_v3_use_src_gap_in_coarse_head":
        return _is_enabled_flag(hparams.get(key, False))
    if key == "rhythm_v3_analytic_gap_clip":
        return float(hparams.get(key, 0.35) or 0.0)
    if key == "rhythm_v3_prefix_budget_pos":
        return int(hparams.get(key, hparams.get("rhythm_v3_unit_budget_pos", 24)) or 24)
    if key == "rhythm_v3_prefix_budget_neg":
        return int(hparams.get(key, hparams.get("rhythm_v3_unit_budget_neg", 24)) or 24)
    if key == "rhythm_v3_dynamic_budget_ratio":
        return float(hparams.get(key, 0.0) or 0.0)
    if key == "rhythm_v3_min_prefix_budget":
        return int(hparams.get(key, 0) or 0)
    if key == "rhythm_v3_max_prefix_budget":
        return int(hparams.get(key, 0) or 0)
    if key == "rhythm_v3_budget_mode":
        return str(hparams.get(key, "total") or "total").strip().lower()
    if key == "rhythm_v3_boundary_carry_decay":
        return float(hparams.get(key, 0.25) or 0.25)
    if key == "rhythm_v3_boundary_offset_decay":
        value = hparams.get(key, None)
        if value is None:
            value = hparams.get("rhythm_v3_boundary_carry_decay", 0.25)
        return float(value)
    if key == "rhythm_v3_boundary_reset_thresh":
        return float(hparams.get(key, 0.5) or 0.5)
    if key == "rhythm_v3_integer_projection_mode":
        return str(
            _get_hparam_alias(
                hparams,
                "rhythm_v3_integer_projection_mode",
                "rhythm_v3_projection_mode",
                "greedy",
            )
            or "greedy"
        ).strip().lower()
    if key == "rhythm_v3_integer_projection_anchor_mode":
        return str(hparams.get(key, "rounded") or "rounded").strip().lower()
    if key == "rhythm_v3_projection_repair_max_steps":
        return int(hparams.get(key, 0) or 0)
    if key == "rhythm_v3_projection_repair_speech_bonus":
        return float(hparams.get(key, 1.0) or 0.0)
    if key == "rhythm_v3_projection_repair_boundary_penalty":
        return float(hparams.get(key, 0.35) or 0.0)
    if key == "rhythm_v3_use_continuous_alignment":
        return _is_enabled_flag(hparams.get(key, False))
    if key == "rhythm_v3_alignment_mode":
        return normalize_duration_v3_alignment_mode(hparams.get(key, "continuous_viterbi_v1"))
    if key == "rhythm_v3_minimal_v1_profile":
        return _is_enabled_flag(hparams.get(key, False))
    if key == "rhythm_v3_strict_minimal_claim_profile":
        return _is_enabled_flag(hparams.get(key, True))
    return hparams.get(key)


def _build_gate_contract_fingerprint(hparams) -> dict[str, object]:
    return {
        key: _normalize_gate_fingerprint_value(key, _resolve_gate_contract_hparam(hparams, key))
        for key in _GATE_STATUS_FINGERPRINT_KEYS
    }


def _validate_rhythm_v3_gate_status_json(
    hparams,
    *,
    gate_status_json,
) -> None:
    gate_path = Path(str(gate_status_json))
    try:
        payload = json.loads(gate_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(
            f"rhythm_v3_required_gate_status_json must contain valid JSON: {gate_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            "rhythm_v3_required_gate_status_json must contain a JSON object with gate pass keys."
        )
    required = ("gate0_pass", "gate1_pass")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(
            "rhythm_v3_required_gate_status_json is missing required keys: "
            + ", ".join(missing)
        )
    payload_fp = payload.get("contract_fingerprint")
    if not isinstance(payload_fp, dict):
        raise ValueError(
            "rhythm_v3_required_gate_status_json must contain contract_fingerprint for strict gate reuse."
        )
    current_fp = _build_gate_contract_fingerprint(hparams)
    payload_keys = set(payload_fp.keys())
    required_keys = set(_GATE_STATUS_FINGERPRINT_KEYS)
    missing_fp_keys = sorted(required_keys - payload_keys)
    extra_fp_keys = sorted(payload_keys - required_keys)
    if missing_fp_keys or extra_fp_keys:
        problems: list[str] = []
        if missing_fp_keys:
            problems.append("missing_keys=" + ",".join(missing_fp_keys))
        if extra_fp_keys:
            problems.append("extra_keys=" + ",".join(extra_fp_keys))
        raise ValueError(
            "rhythm_v3_required_gate_status_json contract_fingerprint schema mismatch. "
            + "; ".join(problems)
        )
    mismatches = []
    for key in _GATE_STATUS_FINGERPRINT_KEYS:
        payload_value = _normalize_gate_fingerprint_value(key, payload_fp.get(key))
        if payload_value != current_fp.get(key):
            mismatches.append(
                f"{key}: payload={payload_value!r} current={current_fp.get(key)!r}"
            )
    if mismatches:
        raise ValueError(
            "rhythm_v3_required_gate_status_json contract mismatch with current hparams. "
            "Regenerate gate status for the current g/prefix/alignment contract. "
            + "; ".join(mismatches)
        )
    if not bool(payload["gate0_pass"]) or not bool(payload["gate1_pass"]):
        raise ValueError(
            "rhythm_v3_gate_quality_strict requires gate0_pass=true and gate1_pass=true in "
            "rhythm_v3_required_gate_status_json."
        )
    if _is_enabled_flag(hparams.get("rhythm_v3_require_gate2_for_official_train", False)) and not bool(
        payload.get("gate2_pass", False)
    ):
        raise ValueError(
            "rhythm_v3_require_gate2_for_official_train requires gate2_pass=true in "
            "rhythm_v3_required_gate_status_json."
        )
    if _is_enabled_flag(hparams.get("rhythm_v3_require_gate3_for_prefix_finetune", False)) and not bool(
        payload.get("gate3_pass", False)
    ):
        raise ValueError(
            "rhythm_v3_require_gate3_for_prefix_finetune requires gate3_pass=true in "
            "rhythm_v3_required_gate_status_json."
        )


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
    unit_prior_path = _validate_optional_existing_path(
        hparams.get("rhythm_v3_unit_prior_path"),
        key="rhythm_v3_unit_prior_path",
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
            "rhythm_v3_backbone must be one of: global_only, operator, prompt_summary "
            "(public minimal alias: minimal_v1_global; legacy aliases: role_memory, unit_run)"
        )
    if warp_mode not in {"none", "progress", "detector"}:
        raise ValueError("rhythm_v3_warp_mode must be one of: none, progress, detector")
    if backbone_mode == "global_only" and allow_hybrid:
        raise ValueError("rhythm_v3_allow_hybrid is only valid when rhythm_v3_backbone='operator'.")
    if backbone_mode == "prompt_summary":
        if warp_mode != "none":
            raise ValueError(
                "rhythm_v3_backbone='prompt_summary' (public minimal alias: 'minimal_v1_global'; legacy aliases: 'role_memory', 'unit_run') only supports rhythm_v3_warp_mode='none'."
            )
        if allow_hybrid:
            raise ValueError(
                "rhythm_v3_allow_hybrid is not used when rhythm_v3_backbone='prompt_summary' (public minimal alias: 'minimal_v1_global'; legacy aliases: 'role_memory', 'unit_run')."
            )
        if not bool(hparams.get("rhythm_v3_emit_silence_runs", True)):
            raise ValueError(
                "rhythm_v3_backbone='prompt_summary' (public minimal alias: 'minimal_v1_global'; legacy aliases: 'role_memory', 'unit_run') "
                "requires rhythm_v3_emit_silence_runs=true."
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
    rate_mode = resolve_duration_v3_rate_mode(hparams)
    if rate_mode not in {"simple_global", "log_base"}:
        raise ValueError("rhythm_v3_rate_mode must be one of: simple_global, log_base")
    if backbone_mode == "prompt_summary" and anchor_mode != "source_observed":
        raise ValueError(
            "rhythm_v3_backbone='prompt_summary' (public minimal alias: 'minimal_v1_global'; legacy aliases: 'role_memory', 'unit_run') requires rhythm_v3_anchor_mode='source_observed'."
        )
    minimal_v1_profile = _is_enabled_flag(hparams.get("rhythm_v3_minimal_v1_profile", False))
    strict_minimal_claim_profile = _is_enabled_flag(
        hparams.get("rhythm_v3_strict_minimal_claim_profile", True)
    )
    prompt_domain_mode = _normalize_prompt_domain_mode(
        hparams.get("rhythm_v3_prompt_domain_mode", "minimal_strict")
    )
    if minimal_v1_profile:
        if "rhythm_v3_detach_global_term_in_local_head" not in hparams:
            hparams["rhythm_v3_detach_global_term_in_local_head"] = True
        if "rhythm_v3_freeze_src_rate_init" not in hparams:
            hparams["rhythm_v3_freeze_src_rate_init"] = True
        backbone_value = hparams.get("rhythm_v3_backbone")
        backbone_text = str(backbone_value or "").strip().lower()
        normalized_backbone_value = normalize_duration_v3_backbone_mode(backbone_value)
        if backbone_value is None:
            raise ValueError(
                "rhythm_v3_minimal_v1_profile requires explicit rhythm_v3_backbone='minimal_v1_global'."
            )
        if backbone_text in _LEGACY_PROMPT_SUMMARY_BACKBONES:
            raise ValueError(
                "rhythm_v3_minimal_v1_profile forbids legacy rhythm_v3_backbone aliases "
                "'role_memory' and 'unit_run'; use 'minimal_v1_global' instead."
            )
        if normalized_backbone_value != "prompt_summary":
            raise ValueError(
                "rhythm_v3_minimal_v1_profile requires rhythm_v3_backbone='minimal_v1_global' "
                "(or 'prompt_summary' only for compatibility)."
            )
        for key in ("rhythm_num_summary_slots", "rhythm_num_role_slots", "num_summary_slots", "num_role_slots"):
            if key in hparams and int(hparams.get(key, 1) or 1) != 1:
                raise ValueError("rhythm_v3_minimal_v1_profile requires rhythm_num_summary_slots=1.")
        if int(hparams.get("rhythm_num_summary_slots", hparams.get("rhythm_num_role_slots", 1)) or 1) != 1:
            raise ValueError("rhythm_v3_minimal_v1_profile requires rhythm_num_summary_slots=1.")
        for key in ("rhythm_v3_summary_use_unit_embedding", "summary_use_unit_embedding"):
            if _is_enabled_flag(hparams.get(key, False)):
                raise ValueError(
                    "rhythm_v3_minimal_v1_profile requires rhythm_v3_summary_use_unit_embedding=false."
                )
        for key in ("rhythm_v3_summary_pool_speech_only", "summary_pool_speech_only"):
            if key in hparams and not _is_enabled_flag(hparams.get(key, True)):
                raise ValueError(
                    "rhythm_v3_minimal_v1_profile requires rhythm_v3_summary_pool_speech_only=true."
                )
        if not _is_enabled_flag(hparams.get("rhythm_v3_disallow_same_text_reference", True)):
            raise ValueError(
                "rhythm_v3_minimal_v1_profile requires rhythm_v3_disallow_same_text_reference=true."
            )
        if _is_enabled_flag(hparams.get("rhythm_v3_disallow_same_text_paired_target", False)):
            raise ValueError(
                "rhythm_v3_minimal_v1_profile requires rhythm_v3_disallow_same_text_paired_target=false."
            )
        if not _is_enabled_flag(hparams.get("rhythm_v3_require_same_text_paired_target", True)):
            raise ValueError(
                "rhythm_v3_minimal_v1_profile requires rhythm_v3_require_same_text_paired_target=true."
            )
        if _is_enabled_flag(hparams.get("rhythm_v3_allow_source_self_target_fallback", False)):
            raise ValueError(
                "rhythm_v3_minimal_v1_profile requires rhythm_v3_allow_source_self_target_fallback=false."
            )
        if not _is_enabled_flag(hparams.get("rhythm_v3_use_continuous_alignment", False)):
            raise ValueError(
                "rhythm_v3_minimal_v1_profile requires rhythm_v3_use_continuous_alignment=true "
                "unless unit_duration_tgt is already explicitly cached with continuous provenance."
            )
        alignment_mode = normalize_duration_v3_alignment_mode(
            hparams.get("rhythm_v3_alignment_mode", "continuous_viterbi_v1")
        )
        if alignment_mode not in {"continuous_precomputed", "continuous_viterbi_v1"}:
            raise ValueError(
                "rhythm_v3_minimal_v1_profile requires rhythm_v3_alignment_mode to be "
                "continuous_precomputed or continuous_viterbi_v1."
            )
        for key in ("rhythm_v3_simple_global_stats", "simple_global_stats"):
            if key in hparams and not _is_enabled_flag(hparams.get(key, True)):
                raise ValueError("rhythm_v3_minimal_v1_profile requires rhythm_v3_simple_global_stats=true.")
        if rate_mode != "simple_global":
            raise ValueError("rhythm_v3_minimal_v1_profile requires rhythm_v3_rate_mode=simple_global.")
        for key in ("rhythm_v3_rate_mode", "rate_mode"):
            if key in hparams and normalize_duration_v3_rate_mode(hparams.get(key, "")) != "simple_global":
                raise ValueError("rhythm_v3_minimal_v1_profile requires rhythm_v3_rate_mode=simple_global.")
        for key in ("rhythm_v3_use_log_base_rate", "use_log_base_rate"):
            if _is_enabled_flag(hparams.get(key, False)):
                raise ValueError("rhythm_v3_minimal_v1_profile requires rhythm_v3_use_log_base_rate=false.")
        for key in ("rhythm_v3_use_reference_summary", "use_reference_summary"):
            if _is_enabled_flag(hparams.get(key, False)):
                raise ValueError("rhythm_v3_minimal_v1_profile requires rhythm_v3_use_reference_summary=false.")
        for key in ("rhythm_v3_use_learned_residual_gate", "use_learned_residual_gate"):
            if _is_enabled_flag(hparams.get(key, False)):
                raise ValueError("rhythm_v3_minimal_v1_profile requires rhythm_v3_use_learned_residual_gate=false.")
        for key in ("rhythm_v3_disable_learned_gate", "disable_learned_gate"):
            if (
                strict_minimal_claim_profile
                and key in hparams
                and not _is_enabled_flag(hparams.get(key, True))
            ):
                raise ValueError("strict minimal claim profile requires rhythm_v3_disable_learned_gate=true.")
        if not _is_enabled_flag(hparams.get("rhythm_v3_detach_global_term_in_local_head", True)):
            raise ValueError(
                "rhythm_v3_minimal_v1_profile requires rhythm_v3_detach_global_term_in_local_head=true."
            )
        if not _is_enabled_flag(hparams.get("rhythm_v3_freeze_src_rate_init", True)):
            raise ValueError(
                "rhythm_v3_minimal_v1_profile requires rhythm_v3_freeze_src_rate_init=true."
            )
        if str(hparams.get("rhythm_streaming_mode", "strict") or "strict").strip().lower() != "strict":
            raise ValueError("rhythm_v3_minimal_v1_profile requires rhythm_streaming_mode='strict'.")
        if int(hparams.get("rhythm_response_window_right", 0) or 0) != 0:
            raise ValueError("rhythm_v3_minimal_v1_profile requires rhythm_response_window_right=0.")
        micro_lookahead = hparams.get("rhythm_micro_lookahead_units")
        if micro_lookahead is not None and int(micro_lookahead or 0) != 0:
            raise ValueError("rhythm_v3_minimal_v1_profile requires rhythm_micro_lookahead_units=0 or unset.")
        for key in ("lambda_rhythm_summary", "lambda_rhythm_mem"):
            if float(hparams.get(key, 0.0) or 0.0) > 0.0:
                raise ValueError("rhythm_v3_minimal_v1_profile requires lambda_rhythm_summary=0.")
        if float(hparams.get("lambda_rhythm_base", 0.0) or 0.0) > 0.0:
            raise ValueError("rhythm_v3_minimal_v1_profile requires lambda_rhythm_base=0.")
        if float(hparams.get("rhythm_v3_silence_coarse_weight", 0.0) or 0.0) > 0.0:
            raise ValueError("rhythm_v3_minimal_v1_profile requires rhythm_v3_silence_coarse_weight=0.")
        for key in ("rhythm_v3_short_gap_silence_scale", "short_gap_silence_scale"):
            if abs(float(hparams.get(key, 0.35) or 0.35) - 0.35) > 1.0e-6:
                raise ValueError(
                    "rhythm_v3_minimal_v1_profile requires rhythm_v3_short_gap_silence_scale=0.35."
                )
        for key in ("rhythm_v3_leading_silence_scale", "leading_silence_scale"):
            if abs(float(hparams.get(key, 0.0) or 0.0) - 0.0) > 1.0e-6:
                raise ValueError(
                    "rhythm_v3_minimal_v1_profile requires rhythm_v3_leading_silence_scale=0.0."
                )
        if int(hparams.get("rhythm_v3_drop_edge_runs_for_g", 0) or 0) < 1:
            raise ValueError("rhythm_v3_minimal_v1_profile requires rhythm_v3_drop_edge_runs_for_g >= 1.")
        if strict_minimal_claim_profile and _is_enabled_flag(hparams.get("rhythm_v3_alignment_soft_repair", False)):
            raise ValueError(
                "strict minimal claim profile requires rhythm_v3_alignment_soft_repair=false."
            )
        if _is_enabled_flag(hparams.get("rhythm_v3_allow_silence_aux", False)):
            raise ValueError("rhythm_v3_minimal_v1_profile requires rhythm_v3_allow_silence_aux=false.")
        if prompt_domain_mode == "minimal_strict":
            min_prompt_ref_len_sec = float(hparams.get("rhythm_v3_min_prompt_ref_len_sec", 3.0) or 0.0)
            max_prompt_ref_len_sec = float(hparams.get("rhythm_v3_max_prompt_ref_len_sec", 8.0) or 0.0)
            if abs(min_prompt_ref_len_sec - 3.0) > 1.0e-6:
                raise ValueError(
                    "strict minimal claim profile requires rhythm_v3_min_prompt_ref_len_sec=3.0."
                )
            if abs(max_prompt_ref_len_sec - 8.0) > 1.0e-6:
                raise ValueError(
                    "strict minimal claim profile requires rhythm_v3_max_prompt_ref_len_sec=8.0."
                )
        if float(hparams.get("rhythm_prompt_dropout", 0.0) or 0.0) > 0.0:
            raise ValueError("rhythm_v3_minimal_v1_profile requires rhythm_prompt_dropout=0.0.")
        if float(hparams.get("rhythm_prompt_truncation", 0.0) or 0.0) > 0.0:
            raise ValueError("rhythm_v3_minimal_v1_profile requires rhythm_prompt_truncation=0.0.")
        for key in ("rhythm_v3_alignment_allow_source_skip", "rhythm_v3_align_allow_source_skip"):
            if _is_enabled_flag(hparams.get(key, False)):
                raise ValueError(
                    "rhythm_v3_minimal_v1_profile requires rhythm_v3_alignment_allow_source_skip=false."
                )
        if strict_minimal_claim_profile and _is_enabled_flag(hparams.get("rhythm_v3_use_src_gap_in_coarse_head", False)):
            raise ValueError(
                "strict minimal claim profile requires rhythm_v3_use_src_gap_in_coarse_head=false."
            )
    if _is_enabled_flag(hparams.get("rhythm_v3_disable_learned_gate", False)) and _is_enabled_flag(
        hparams.get("rhythm_v3_use_learned_residual_gate", False)
    ):
        raise ValueError(
            "rhythm_v3_disable_learned_gate=true is incompatible with rhythm_v3_use_learned_residual_gate=true."
        )
    if _is_enabled_flag(hparams.get("rhythm_v3_require_same_text_paired_target", False)) and _is_enabled_flag(
        hparams.get("rhythm_v3_disallow_same_text_paired_target", False)
    ):
        raise ValueError(
            "rhythm_v3_require_same_text_paired_target=true is incompatible with "
            "rhythm_v3_disallow_same_text_paired_target=true."
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
        "rhythm_v3_silence_coarse_weight",
        "rhythm_v3_silence_max_logstretch",
        "rhythm_v3_analytic_gap_clip",
        "rhythm_v3_local_short_run_min_duration",
        "rhythm_global_shrink_tau",
        "rhythm_operator_support_tau",
        "rhythm_operator_holdout_ratio",
        "rhythm_operator_min_support_factor",
    ):
        if float(hparams.get(key, 0.0) or 0.0) < 0.0:
            raise ValueError(f"{key} must be >= 0 for rhythm_v3.")
    for key in (
        "rhythm_v3_short_gap_silence_scale",
        "rhythm_v3_leading_silence_scale",
        "rhythm_v3_local_rate_decay",
        "rhythm_v3_local_rate_decay_fast",
        "rhythm_v3_local_rate_decay_slow",
        "rhythm_v3_local_rate_slow_mix",
        "rhythm_v3_boundary_carry_decay",
        "rhythm_v3_boundary_reset_thresh",
    ):
        value = float(hparams.get(key, 0.0) or 0.0)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{key} must be within [0, 1] for rhythm_v3.")
    dynamic_budget_ratio = float(hparams.get("rhythm_v3_dynamic_budget_ratio", 0.0) or 0.0)
    if dynamic_budget_ratio < 0.0 or dynamic_budget_ratio > 1.0:
        raise ValueError("rhythm_v3_dynamic_budget_ratio must be within [0, 1] for rhythm_v3.")
    min_prefix_budget = int(hparams.get("rhythm_v3_min_prefix_budget", 0) or 0)
    max_prefix_budget = int(hparams.get("rhythm_v3_max_prefix_budget", 0) or 0)
    if min_prefix_budget < 0:
        raise ValueError("rhythm_v3_min_prefix_budget must be >= 0 for rhythm_v3.")
    if max_prefix_budget < 0:
        raise ValueError("rhythm_v3_max_prefix_budget must be >= 0 for rhythm_v3.")
    if max_prefix_budget > 0 and max_prefix_budget < min_prefix_budget:
        raise ValueError("rhythm_v3_max_prefix_budget must be >= rhythm_v3_min_prefix_budget.")
    budget_mode = str(hparams.get("rhythm_v3_budget_mode", "total") or "total").strip().lower()
    if budget_mode not in {"total", "speech_only", "hybrid"}:
        raise ValueError("rhythm_v3_budget_mode must be one of: total, speech_only, hybrid.")
    integer_projection_mode = str(
        hparams.get(
            "rhythm_v3_integer_projection_mode",
            hparams.get("rhythm_v3_projection_mode", "greedy"),
        )
        or "greedy"
    ).strip().lower()
    integer_projection_aliases = {
        "default": "greedy",
        "nearest": "greedy",
        "recurrent": "greedy",
        "repair": "greedy_repair",
        "greedyrepair": "greedy_repair",
        "dp": "prefix_optimal",
        "prefix": "prefix_optimal",
        "prefix_dp": "prefix_optimal",
        "closed_prefix": "prefix_optimal",
        "closed_prefix_optimal": "prefix_optimal",
    }
    integer_projection_mode = integer_projection_aliases.get(integer_projection_mode, integer_projection_mode)
    if integer_projection_mode not in {"greedy", "greedy_repair", "prefix_optimal"}:
        raise ValueError("rhythm_v3_integer_projection_mode must be one of: greedy, greedy_repair, prefix_optimal.")
    integer_projection_anchor_mode = (
        str(hparams.get("rhythm_v3_integer_projection_anchor_mode", "rounded") or "rounded").strip().lower()
    )
    integer_projection_anchor_aliases = {
        "default": "rounded",
        "round": "rounded",
        "source_rounded": "rounded",
        "raw": "continuous",
        "float": "continuous",
        "source": "continuous",
        "source_continuous": "continuous",
    }
    integer_projection_anchor_mode = integer_projection_anchor_aliases.get(
        integer_projection_anchor_mode,
        integer_projection_anchor_mode,
    )
    if integer_projection_anchor_mode not in {"rounded", "continuous"}:
        raise ValueError("rhythm_v3_integer_projection_anchor_mode must be one of: rounded, continuous.")
    if int(hparams.get("rhythm_v3_prefix_projection_candidate_radius", 2) or 0) < 0:
        raise ValueError("rhythm_v3_prefix_projection_candidate_radius must be >= 0 for rhythm_v3.")
    if int(hparams.get("rhythm_v3_prefix_projection_max_states", 256) or 0) <= 0:
        raise ValueError("rhythm_v3_prefix_projection_max_states must be > 0 for rhythm_v3.")
    for key in (
        "rhythm_v3_prefix_projection_terminal_carry_weight",
        "rhythm_v3_prefix_projection_terminal_offset_weight",
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
    if int(hparams.get("rhythm_v3_drop_edge_runs_for_g", 0) or 0) < 0:
        raise ValueError("rhythm_v3_drop_edge_runs_for_g must be >= 0 for rhythm_v3.")
    min_prompt_speech_ratio = float(hparams.get("rhythm_v3_min_prompt_speech_ratio", 0.6) or 0.0)
    if min_prompt_speech_ratio < 0.0 or min_prompt_speech_ratio > 1.0:
        raise ValueError("rhythm_v3_min_prompt_speech_ratio must be within [0, 1] for rhythm_v3.")
    min_prompt_ref_len_sec = float(hparams.get("rhythm_v3_min_prompt_ref_len_sec", 3.0) or 0.0)
    max_prompt_ref_len_sec = float(hparams.get("rhythm_v3_max_prompt_ref_len_sec", 8.0) or 0.0)
    if min_prompt_ref_len_sec <= 0.0:
        raise ValueError("rhythm_v3_min_prompt_ref_len_sec must be > 0 for rhythm_v3.")
    if max_prompt_ref_len_sec <= 0.0:
        raise ValueError("rhythm_v3_max_prompt_ref_len_sec must be > 0 for rhythm_v3.")
    if max_prompt_ref_len_sec < min_prompt_ref_len_sec:
        raise ValueError(
            "rhythm_v3_max_prompt_ref_len_sec must be >= rhythm_v3_min_prompt_ref_len_sec for rhythm_v3."
        )
    src_rate_init_mode = str(hparams.get("rhythm_v3_src_rate_init_mode", "auto") or "auto").strip().lower()
    if src_rate_init_mode not in {"auto", "learned", "zero", "first_speech"}:
        raise ValueError(
            "rhythm_v3_src_rate_init_mode must be one of: auto, learned, zero, first_speech."
        )
    src_prefix_stat_mode = str(hparams.get("rhythm_v3_src_prefix_stat_mode", "ema") or "ema").strip().lower()
    if src_prefix_stat_mode not in {
        "ema",
        "dual_timescale",
        "dual",
        "dual_scale",
        "dualtime",
        "family_hybrid",
        "robust",
        "hybrid",
        "family",
        "exact_global_family",
        "exact",
        "exact_family",
        "global_family",
    }:
        raise ValueError(
            "rhythm_v3_src_prefix_stat_mode must be one of: "
            "ema, dual_timescale, family_hybrid, exact_global_family."
        )
    coarse_delta_scale = float(hparams.get("rhythm_v3_coarse_delta_scale", 0.20) or 0.0)
    if coarse_delta_scale < 0.0:
        raise ValueError("rhythm_v3_coarse_delta_scale must be >= 0 for rhythm_v3.")
    local_residual_scale = float(hparams.get("rhythm_v3_local_residual_scale", 0.35) or 0.0)
    if local_residual_scale < 0.0:
        raise ValueError("rhythm_v3_local_residual_scale must be >= 0 for rhythm_v3.")
    boundary_offset_decay = hparams.get("rhythm_v3_boundary_offset_decay", None)
    if boundary_offset_decay is not None:
        boundary_offset_decay = float(boundary_offset_decay)
        if boundary_offset_decay < 0.0 or boundary_offset_decay > 1.0:
            raise ValueError("rhythm_v3_boundary_offset_decay must be within [0, 1] for rhythm_v3.")
    src_rate_init_value = float(hparams.get("rhythm_v3_src_rate_init_value", 0.0) or 0.0)
    if not math.isfinite(src_rate_init_value):
        raise ValueError("rhythm_v3_src_rate_init_value must be finite for rhythm_v3.")
    min_boundary_confidence_for_g = hparams.get("rhythm_v3_min_boundary_confidence_for_g", None)
    if min_boundary_confidence_for_g is not None:
        min_boundary_confidence_for_g = float(min_boundary_confidence_for_g)
        if min_boundary_confidence_for_g < 0.0 or min_boundary_confidence_for_g > 1.0:
            raise ValueError("rhythm_v3_min_boundary_confidence_for_g must be within [0, 1] for rhythm_v3.")
    for key in (
        "rhythm_v3_min_support_log_iqr_for_g",
        "rhythm_v3_min_support_log_span_for_g",
        "rhythm_v3_beta1_min_var",
    ):
        if float(hparams.get(key, 0.0) or 0.0) < 0.0:
            raise ValueError(f"{key} must be >= 0 for rhythm_v3.")
    if int(hparams.get("rhythm_v3_min_support_unique_for_g", 1) or 0) <= 0:
        raise ValueError("rhythm_v3_min_support_unique_for_g must be >= 1 for rhythm_v3.")
    beta1_min = float(hparams.get("rhythm_v3_beta1_min", 0.7) or 0.7)
    beta1_max = float(hparams.get("rhythm_v3_beta1_max", 1.3) or 1.3)
    if beta1_min > beta1_max:
        raise ValueError("rhythm_v3_beta1_min must be <= rhythm_v3_beta1_max for rhythm_v3.")
    if int(hparams.get("rhythm_v3_beta1_min_points", 24) or 0) <= 0:
        raise ValueError("rhythm_v3_beta1_min_points must be > 0 for rhythm_v3.")
    g_variant = str(hparams.get("rhythm_v3_g_variant", "raw_median") or "raw_median").strip().lower()
    if _is_unit_norm_variant(g_variant):
        if minimal_v1_profile and strict_minimal_claim_profile:
            raise ValueError(
                "rhythm_v3_minimal_v1_profile with strict minimal claim profile forbids rhythm_v3_g_variant=unit_norm."
            )
        if unit_prior_path is None:
            raise ValueError("rhythm_v3_g_variant=unit_norm requires rhythm_v3_unit_prior_path.")
    strict_gate = _is_enabled_flag(hparams.get("rhythm_v3_gate_quality_strict", False))
    gate_status_json = hparams.get("rhythm_v3_required_gate_status_json")
    if minimal_v1_profile and strict_gate:
        if not _is_enabled_flag(hparams.get("rhythm_v3_debug_export", True)):
            raise ValueError("rhythm_v3_gate_quality_strict requires rhythm_v3_debug_export=true.")
        if not _is_enabled_flag(hparams.get("rhythm_v3_export_projector_telemetry", True)):
            raise ValueError(
                "rhythm_v3_gate_quality_strict requires rhythm_v3_export_projector_telemetry=true."
            )
        if not _is_enabled_flag(hparams.get("rhythm_v3_strict_eval_invalid_g", True)):
            raise ValueError(
                "rhythm_v3_gate_quality_strict requires rhythm_v3_strict_eval_invalid_g=true."
            )
        if not _is_enabled_flag(hparams.get("rhythm_v3_detach_global_term_in_local_head", True)):
            raise ValueError(
                "rhythm_v3_gate_quality_strict requires rhythm_v3_detach_global_term_in_local_head=true."
            )
        if not gate_status_json:
            raise ValueError(
                "rhythm_v3_gate_quality_strict requires rhythm_v3_required_gate_status_json."
            )
        _validate_optional_existing_path(
            gate_status_json, key="rhythm_v3_required_gate_status_json"
        )
        _validate_rhythm_v3_gate_status_json(
            hparams,
            gate_status_json=gate_status_json,
        )
    for key in (
        "rhythm_v3_alignment_unmatched_speech_ratio_max",
        "rhythm_v3_alignment_mean_local_confidence_speech_min",
        "rhythm_v3_alignment_mean_coarse_confidence_speech_min",
    ):
        if key in hparams:
            value = float(hparams.get(key, 0.0) or 0.0)
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{key} must be within [0, 1] for rhythm_v3.")
    for key in (
        "rhythm_v3_alignment_local_margin_p10_min",
        "rhythm_v3_align_local_margin_p10_min",
    ):
        if key in hparams and float(hparams.get(key, 0.0) or 0.0) < 0.0:
            raise ValueError(f"{key} must be >= 0 for rhythm_v3.")
    for key, alias in (
        ("rhythm_v3_alignment_lambda_emb", "rhythm_v3_align_lambda_emb"),
        ("rhythm_v3_alignment_lambda_type", "rhythm_v3_align_lambda_type"),
        ("rhythm_v3_alignment_lambda_band", "rhythm_v3_align_lambda_band"),
        ("rhythm_v3_alignment_lambda_unit", "rhythm_v3_align_lambda_unit"),
        ("rhythm_v3_alignment_bad_cost_threshold", "rhythm_v3_align_bad_cost_threshold"),
        ("rhythm_v3_alignment_min_dp_weight", "rhythm_v3_align_min_dp_weight"),
        ("rhythm_v3_alignment_skip_penalty", "rhythm_v3_align_skip_penalty"),
    ):
        value = _get_hparam_alias(hparams, key, alias, None)
        if value is not None and float(value or 0.0) < 0.0:
            raise ValueError(f"{key} must be >= 0 for rhythm_v3.")
    band_ratio_value = _get_hparam_alias(
        hparams,
        "rhythm_v3_alignment_band_ratio",
        "rhythm_v3_align_band_ratio",
        None,
    )
    if band_ratio_value is not None:
        band_ratio = float(band_ratio_value or 0.0)
        if band_ratio < 0.0:
            raise ValueError("rhythm_v3_alignment_band_ratio must be >= 0 for rhythm_v3.")
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
    allow_silence_aux = _is_enabled_flag(hparams.get("rhythm_v3_allow_silence_aux", False))
    if not allow_silence_aux and float(hparams.get("rhythm_v3_silence_coarse_weight", 0.0) or 0.0) > 0.0:
        raise ValueError("rhythm_v3_silence_coarse_weight > 0 requires rhythm_v3_allow_silence_aux=true.")
    if not allow_silence_aux and float(hparams.get("lambda_rhythm_silence_aux", 0.0) or 0.0) > 0.0:
        raise ValueError("lambda_rhythm_silence_aux > 0 requires rhythm_v3_allow_silence_aux=true.")
    public_inputs = _normalize_public_surface(hparams.get("rhythm_public_inputs"), key="rhythm_public_inputs")
    public_outputs = _normalize_public_surface(hparams.get("rhythm_public_outputs"), key="rhythm_public_outputs")
    public_losses = _normalize_public_surface(hparams.get("rhythm_public_losses"), key="rhythm_public_losses")
    _validate_required_public_surface(
        public_inputs,
        key="rhythm_public_inputs",
        required=_REQUIRED_V3_PUBLIC_INPUTS,
        forbidden=_FORBIDDEN_V3_PUBLIC_INPUTS,
    )
    if public_inputs is not None and runtime_mode == "prompt_summary":
        missing_prompt_summary_inputs = [
            key for key in _PROMPT_SUMMARY_REQUIRED_V3_PUBLIC_INPUTS if key not in public_inputs
        ]
        if missing_prompt_summary_inputs:
            raise ValueError(
                "rhythm_v3 prompt_summary public surface requires explicit speech-only prompt inputs: "
                + ", ".join(missing_prompt_summary_inputs)
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
    "normalize_duration_v3_rate_mode",
    "resolve_duration_v3_rate_mode",
    "validate_duration_v3_training_hparams",
    "validate_rhythm_training_hparams",
]
