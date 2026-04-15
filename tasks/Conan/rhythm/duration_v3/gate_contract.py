from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class GateThresholds:
    gate1_monotonicity_rate_min: float = 0.95
    gate1_transfer_slope_min: float = 0.10
    gate1_effect_size_min: float = 0.02
    gate1_tie_rate_max: float = 0.05
    gate2_final_prefix_offset_abs_max: float = 0.25
    gate2_cumulative_prefix_offset_abs_max: float = 0.25


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
    "rhythm_v3_prompt_g_trim_ratio",
    "rhythm_v3_prompt_g_drop_edge_runs",
    "rhythm_v3_prompt_min_boundary_confidence_for_g",
    "rhythm_v3_src_g_variant",
    "rhythm_v3_src_g_trim_ratio",
    "rhythm_v3_src_g_drop_edge_runs",
    "rhythm_v3_src_min_boundary_confidence_for_g",
    "rhythm_v3_prompt_ref_len_contract_active",
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
    "rhythm_v3_projection_mode",
    "rhythm_v3_integer_projection_mode",
    "rhythm_v3_integer_projection_anchor_mode",
    "rhythm_v3_prefix_optimal_step_weight",
    "rhythm_v3_prefix_optimal_prefix_weight",
    "rhythm_v3_prefix_optimal_terminal_weight",
    "rhythm_v3_prefix_optimal_boundary_weight",
    "rhythm_v3_prefix_optimal_coarse_weight",
    "rhythm_v3_prefix_optimal_phrase_final_boost",
    "rhythm_v3_prefix_optimal_max_window",
    "rhythm_v3_prefix_optimal_max_states",
    "rhythm_v3_projection_repair_max_steps",
    "rhythm_v3_projection_repair_speech_bonus",
    "rhythm_v3_projection_repair_boundary_penalty",
    "rhythm_v3_use_continuous_alignment",
    "rhythm_v3_alignment_mode",
    "rhythm_v3_minimal_v1_profile",
    "rhythm_v3_strict_minimal_claim_profile",
)

_GATE_STATUS_BOOL_KEYS = {
    "rhythm_v3_alignment_prefilter_bad_samples",
    "rhythm_v3_disallow_same_text_paired_target",
    "rhythm_v3_disallow_same_text_reference",
    "rhythm_v3_prompt_require_clean_support",
    "rhythm_v3_require_same_text_paired_target",
    "rhythm_v3_prompt_ref_len_contract_active",
    "rhythm_v3_strict_eval_invalid_g",
    "rhythm_v3_use_src_gap_in_coarse_head",
    "rhythm_v3_use_continuous_alignment",
    "rhythm_v3_minimal_v1_profile",
    "rhythm_v3_strict_minimal_claim_profile",
}

_GATE_STATUS_INT_KEYS = {
    "rhythm_v3_alignment_prefilter_max_attempts",
    "rhythm_v3_drop_edge_runs_for_g",
    "rhythm_v3_prompt_g_drop_edge_runs",
    "rhythm_v3_src_g_drop_edge_runs",
    "rhythm_v3_src_prefix_min_support",
    "rhythm_v3_prefix_budget_pos",
    "rhythm_v3_prefix_budget_neg",
    "rhythm_v3_min_prefix_budget",
    "rhythm_v3_max_prefix_budget",
    "rhythm_v3_prefix_optimal_max_window",
    "rhythm_v3_prefix_optimal_max_states",
    "rhythm_v3_projection_repair_max_steps",
}

_GATE_STATUS_FLOAT_KEYS = {
    "rhythm_v3_alignment_local_margin_p10_min",
    "rhythm_v3_alignment_mean_coarse_confidence_speech_min",
    "rhythm_v3_alignment_mean_local_confidence_speech_min",
    "rhythm_v3_alignment_unmatched_speech_ratio_max",
    "rhythm_v3_g_trim_ratio",
    "rhythm_v3_min_boundary_confidence_for_g",
    "rhythm_v3_prompt_g_trim_ratio",
    "rhythm_v3_prompt_min_boundary_confidence_for_g",
    "rhythm_v3_src_g_trim_ratio",
    "rhythm_v3_src_min_boundary_confidence_for_g",
    "rhythm_v3_max_prompt_ref_len_sec",
    "rhythm_v3_min_prompt_ref_len_sec",
    "rhythm_v3_min_prompt_speech_ratio",
    "rhythm_v3_analytic_gap_clip",
    "rhythm_v3_dynamic_budget_ratio",
    "rhythm_v3_boundary_carry_decay",
    "rhythm_v3_boundary_offset_decay",
    "rhythm_v3_boundary_reset_thresh",
    "rhythm_v3_prefix_optimal_step_weight",
    "rhythm_v3_prefix_optimal_prefix_weight",
    "rhythm_v3_prefix_optimal_terminal_weight",
    "rhythm_v3_prefix_optimal_boundary_weight",
    "rhythm_v3_prefix_optimal_coarse_weight",
    "rhythm_v3_prefix_optimal_phrase_final_boost",
    "rhythm_v3_projection_repair_speech_bonus",
    "rhythm_v3_projection_repair_boundary_penalty",
}


def _is_enabled_flag(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


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


def _normalize_alignment_mode(value) -> str:
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


def normalize_projection_mode(value) -> str:
    mode = str(value or "greedy").strip().lower()
    aliases = {
        "default": "greedy",
        "nearest": "greedy",
        "recurrent": "greedy",
        "repair": "greedy_repair",
        "greedyrepair": "greedy_repair",
        "optimal": "prefix_optimal",
        "dp": "prefix_optimal",
        "prefix_dp": "prefix_optimal",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"greedy", "greedy_repair", "prefix_optimal"}:
        raise ValueError(
            f"rhythm_v3_projection_mode must be one of: greedy, greedy_repair, prefix_optimal; got {value!r}."
        )
    return mode


def normalize_gate_fingerprint_value(key: str, value):
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


def prompt_ref_len_contract_active_from_hparams(hparams) -> bool:
    return _normalize_prompt_domain_mode(
        hparams.get("rhythm_v3_prompt_domain_mode", "minimal_strict")
    ) == "minimal_strict"


def resolve_gate_contract_hparam(hparams, key: str):
    if key == "rhythm_v3_g_variant":
        return str(hparams.get(key, "raw_median") or "raw_median")
    if key == "rhythm_v3_g_trim_ratio":
        return float(hparams.get(key, 0.2) or 0.2)
    if key == "rhythm_v3_drop_edge_runs_for_g":
        return int(hparams.get(key, 0) or 0)
    if key == "rhythm_v3_min_boundary_confidence_for_g":
        return hparams.get(key, None)
    if key == "rhythm_v3_min_prompt_ref_len_sec":
        if not prompt_ref_len_contract_active_from_hparams(hparams):
            return None
        return float(hparams.get(key, 3.0) or 0.0)
    if key == "rhythm_v3_max_prompt_ref_len_sec":
        if not prompt_ref_len_contract_active_from_hparams(hparams):
            return None
        return float(hparams.get(key, 8.0) or 0.0)
    if key == "rhythm_v3_prompt_domain_mode":
        return _normalize_prompt_domain_mode(hparams.get(key, "minimal_strict"))
    if key == "rhythm_v3_prompt_require_clean_support":
        return _is_enabled_flag(hparams.get(key, True))
    if key == "rhythm_v3_prompt_g_variant":
        return str(hparams.get(key, hparams.get("rhythm_v3_g_variant", "raw_median")) or "raw_median")
    if key == "rhythm_v3_prompt_g_trim_ratio":
        return float(hparams.get(key, hparams.get("rhythm_v3_g_trim_ratio", 0.2)) or 0.2)
    if key == "rhythm_v3_prompt_g_drop_edge_runs":
        return int(hparams.get(key, hparams.get("rhythm_v3_drop_edge_runs_for_g", 0)) or 0)
    if key == "rhythm_v3_prompt_min_boundary_confidence_for_g":
        return hparams.get(
            key,
            hparams.get("rhythm_v3_min_boundary_confidence_for_g", None),
        )
    if key == "rhythm_v3_src_g_variant":
        return str(hparams.get(key, hparams.get("rhythm_v3_g_variant", "raw_median")) or "raw_median")
    if key == "rhythm_v3_src_g_trim_ratio":
        return float(hparams.get(key, hparams.get("rhythm_v3_g_trim_ratio", 0.2)) or 0.2)
    if key == "rhythm_v3_src_g_drop_edge_runs":
        return int(hparams.get(key, hparams.get("rhythm_v3_drop_edge_runs_for_g", 0)) or 0)
    if key == "rhythm_v3_src_min_boundary_confidence_for_g":
        return hparams.get(
            key,
            hparams.get("rhythm_v3_min_boundary_confidence_for_g", None),
        )
    if key == "rhythm_v3_prompt_ref_len_contract_active":
        return prompt_ref_len_contract_active_from_hparams(hparams)
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
    if key == "rhythm_v3_projection_mode":
        return normalize_projection_mode(
            hparams.get(
                key,
                hparams.get("rhythm_v3_integer_projection_mode", "greedy"),
            )
        )
    if key == "rhythm_v3_integer_projection_mode":
        return normalize_projection_mode(
            hparams.get(
                "rhythm_v3_projection_mode",
                hparams.get(key, "greedy"),
            )
        )
    if key == "rhythm_v3_integer_projection_anchor_mode":
        mode = str(hparams.get(key, "rounded") or "rounded").strip().lower()
        aliases = {
            "default": "rounded",
            "round": "rounded",
            "source_rounded": "rounded",
            "raw": "continuous",
            "float": "continuous",
            "source": "continuous",
            "source_continuous": "continuous",
        }
        return aliases.get(mode, mode)
    if key == "rhythm_v3_prefix_optimal_step_weight":
        return float(hparams.get(key, 0.10) or 0.10)
    if key == "rhythm_v3_prefix_optimal_prefix_weight":
        return float(hparams.get(key, 1.00) or 1.00)
    if key == "rhythm_v3_prefix_optimal_terminal_weight":
        return float(hparams.get(key, 1.00) or 1.00)
    if key == "rhythm_v3_prefix_optimal_boundary_weight":
        return float(hparams.get(key, 0.75) or 0.75)
    if key == "rhythm_v3_prefix_optimal_coarse_weight":
        return float(hparams.get(key, 0.50) or 0.50)
    if key == "rhythm_v3_prefix_optimal_phrase_final_boost":
        return float(hparams.get(key, 1.50) or 1.50)
    if key == "rhythm_v3_prefix_optimal_max_window":
        return int(hparams.get(key, 96) or 96)
    if key == "rhythm_v3_prefix_optimal_max_states":
        return int(hparams.get(key, 97) or 97)
    if key == "rhythm_v3_projection_repair_max_steps":
        return int(hparams.get(key, 0) or 0)
    if key == "rhythm_v3_projection_repair_speech_bonus":
        return float(hparams.get(key, 1.0) or 0.0)
    if key == "rhythm_v3_projection_repair_boundary_penalty":
        return float(hparams.get(key, 0.35) or 0.0)
    if key == "rhythm_v3_use_continuous_alignment":
        return _is_enabled_flag(hparams.get(key, False))
    if key == "rhythm_v3_alignment_mode":
        return _normalize_alignment_mode(hparams.get(key, "continuous_viterbi_v1"))
    if key == "rhythm_v3_minimal_v1_profile":
        return _is_enabled_flag(hparams.get(key, False))
    if key == "rhythm_v3_strict_minimal_claim_profile":
        return _is_enabled_flag(hparams.get(key, True))
    return hparams.get(key)


def build_gate_contract_fingerprint(hparams) -> dict[str, object]:
    return {
        key: normalize_gate_fingerprint_value(key, resolve_gate_contract_hparam(hparams, key))
        for key in _GATE_STATUS_FINGERPRINT_KEYS
    }


def build_runtime_contract_fingerprint_from_values(values: dict[str, object]) -> dict[str, object]:
    return {
        key: normalize_gate_fingerprint_value(key, values.get(key))
        for key in _GATE_STATUS_FINGERPRINT_KEYS
    }


def build_runtime_contract_fingerprint_json_from_values(values: dict[str, object]) -> str:
    fingerprint = build_runtime_contract_fingerprint_from_values(values)
    return json.dumps(
        fingerprint,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def build_runtime_contract_id_from_values(values: dict[str, object]) -> str:
    fingerprint_json = build_runtime_contract_fingerprint_json_from_values(values)
    return hashlib.sha1(fingerprint_json.encode("utf-8")).hexdigest()[:16]


def build_runtime_contract_id_from_hparams(hparams) -> str:
    return build_runtime_contract_id_from_values(build_gate_contract_fingerprint(hparams))


def resolve_train_stage(hparams) -> str:
    explicit = str(hparams.get("rhythm_v3_train_stage", "") or "").strip().lower()
    if explicit in {"analytic", "coarse_only", "learned", "prefix_finetune"}:
        return explicit
    eval_mode = str(hparams.get("rhythm_v3_eval_mode", "learned") or "learned").strip().lower()
    if eval_mode == "analytic":
        return "analytic"
    if eval_mode == "coarse_only":
        return "coarse_only"
    if eval_mode == "learned":
        return "learned"
    lambda_pref = float(hparams.get("lambda_rhythm_pref", 0.0) or 0.0)
    lambda_cons = float(hparams.get("lambda_rhythm_cons", 0.0) or 0.0)
    if lambda_pref > 0.0 or lambda_cons > 0.0:
        return "prefix_finetune"
    return "learned"


def required_stage_gate_keys(stage: str) -> tuple[str, ...]:
    normalized = str(stage or "").strip().lower()
    if normalized in {"analytic", "coarse_only"}:
        return ("gate0_pass", "gate1_pass")
    if normalized == "learned":
        return ("gate0_pass", "gate1_pass", "gate2_pass")
    if normalized == "prefix_finetune":
        return ("gate0_pass", "gate1_pass", "gate2_pass", "gate3_pass")
    return ("gate0_pass", "gate1_pass")


def attach_gate_permissions(payload: dict[str, object]) -> dict[str, object]:
    gate0a = bool(payload.get("gate0a_pass", False))
    gate0b = bool(payload.get("gate0b_pass", False))
    gate0c = bool(payload.get("gate0c_pass", False))
    gate1 = bool(payload.get("gate1_pass", False))
    gate2 = bool(payload.get("gate2_pass", False))
    payload["gate0a_prompt_domain_pass"] = gate0a
    payload["gate0b_alignment_surface_pass"] = gate0b
    payload["gate0c_signal_explainability_pass"] = gate0c
    payload["gate1_eligible"] = gate0a and gate0b and gate0c
    payload["gate2_eligible"] = gate1
    payload["gate3_eligible"] = gate2
    return payload


__all__ = [
    "GateThresholds",
    "_GATE_STATUS_FINGERPRINT_KEYS",
    "attach_gate_permissions",
    "build_gate_contract_fingerprint",
    "build_runtime_contract_fingerprint_from_values",
    "build_runtime_contract_fingerprint_json_from_values",
    "build_runtime_contract_id_from_hparams",
    "build_runtime_contract_id_from_values",
    "normalize_gate_fingerprint_value",
    "normalize_projection_mode",
    "prompt_ref_len_contract_active_from_hparams",
    "required_stage_gate_keys",
    "resolve_gate_contract_hparam",
    "resolve_train_stage",
]
