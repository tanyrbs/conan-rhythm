from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from tasks.Conan.rhythm.task_config import validate_rhythm_training_hparams


ROOT = Path(__file__).resolve().parents[2]


def _minimal_v3_hparams():
    return {
        "rhythm_enable_v2": False,
        "rhythm_enable_v3": True,
        "rhythm_v3_backbone": "operator",
        "rhythm_v3_warp_mode": "progress",
        "rhythm_v3_allow_hybrid": True,
        "rhythm_response_rank": 4,
        "lambda_rhythm_dur": 1.0,
        "lambda_rhythm_bias": 0.0,
        "lambda_rhythm_op": 0.25,
        "lambda_rhythm_pref": 0.20,
        "lambda_rhythm_cons": 0.0,
        "lambda_rhythm_zero": 0.05,
        "rhythm_streaming_mode": "strict",
        "rhythm_response_window_right": 0,
    }


def _minimal_prompt_summary_v1_hparams():
    hparams = _minimal_v3_hparams()
    hparams.update(
        {
            "rhythm_v3_minimal_v1_profile": True,
            "rhythm_v3_backbone": "minimal_v1_global",
            "rhythm_v3_warp_mode": "none",
            "rhythm_v3_allow_hybrid": False,
            "rhythm_v3_anchor_mode": "source_observed",
            "lambda_rhythm_op": 0.0,
            "lambda_rhythm_zero": 0.0,
            "rhythm_v3_simple_global_stats": True,
            "rhythm_v3_rate_mode": "simple_global",
            "rhythm_v3_use_log_base_rate": False,
            "rhythm_v3_use_reference_summary": False,
            "rhythm_v3_use_learned_residual_gate": False,
            "rhythm_v3_disable_learned_gate": True,
            "rhythm_v3_drop_edge_runs_for_g": 1,
            "rhythm_v3_summary_pool_speech_only": True,
            "rhythm_v3_disallow_same_text_reference": True,
            "rhythm_v3_disallow_same_text_paired_target": False,
            "rhythm_v3_require_same_text_paired_target": True,
            "rhythm_v3_allow_source_self_target_fallback": False,
        }
    )
    return hparams


def test_validate_rhythm_training_hparams_rejects_v2_v3_mutual_exclusion():
    with pytest.raises(ValueError, match="Enable only one rhythm backend"):
        validate_rhythm_training_hparams(
            {
                **_minimal_v3_hparams(),
                "rhythm_enable_v2": True,
            }
        )


def test_validate_rhythm_training_hparams_rejects_minimal_v1_stage_on_v2_backend():
    with pytest.raises(ValueError, match="minimal_v1 must run on rhythm_v3 only"):
        validate_rhythm_training_hparams(
            {
                "rhythm_stage": "minimal_v1",
                "rhythm_enable_v2": True,
                "rhythm_enable_v3": False,
                "rhythm_response_rank": 4,
            }
        )


def test_validate_rhythm_training_hparams_rejects_v2_with_v3_alias_mutual_exclusion():
    with pytest.raises(ValueError, match="Enable only one rhythm backend"):
        validate_rhythm_training_hparams(
            {
                "rhythm_enable_v2": True,
                "rhythm_mode": "duration_operator",
                "rhythm_response_rank": 4,
            }
        )


def test_validate_rhythm_training_hparams_rejects_removed_duration_ref_memory_alias():
    with pytest.raises(ValueError, match="duration_ref_memory"):
        validate_rhythm_training_hparams(
            {
                "rhythm_enable_v2": False,
                "rhythm_mode": "duration_ref_memory",
                "rhythm_response_rank": 4,
            }
        )


@pytest.mark.parametrize(
    "bad_key,bad_value,match",
    [
        ("rhythm_response_rank", 0, "rhythm_response_rank must be > 0"),
        ("lambda_rhythm_op", -0.1, "lambda_rhythm_op must be >= 0"),
        ("lambda_rhythm_cons", -0.1, "lambda_rhythm_cons must be >= 0"),
    ],
)
def test_validate_rhythm_training_hparams_rejects_invalid_v3_values(bad_key, bad_value, match):
    hparams = _minimal_v3_hparams()
    hparams[bad_key] = bad_value
    with pytest.raises(ValueError, match=match):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_negative_silence_coarse_weight():
    hparams = _minimal_v3_hparams()
    hparams["rhythm_v3_silence_coarse_weight"] = -0.1
    with pytest.raises(ValueError, match="rhythm_v3_silence_coarse_weight"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_negative_silence_max_logstretch():
    hparams = _minimal_v3_hparams()
    hparams["rhythm_v3_silence_max_logstretch"] = -0.1
    with pytest.raises(ValueError, match="rhythm_v3_silence_max_logstretch"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_accepts_positive_silence_settings():
    validate_rhythm_training_hparams(
        {
            **_minimal_v3_hparams(),
            "rhythm_v3_allow_silence_aux": True,
            "rhythm_v3_silence_coarse_weight": 0.5,
            "rhythm_v3_silence_max_logstretch": 0.45,
        }
    )


def test_validate_rhythm_training_hparams_rejects_negative_analytic_gap_clip():
    hparams = _minimal_v3_hparams()
    hparams["rhythm_v3_analytic_gap_clip"] = -0.1
    with pytest.raises(ValueError, match="rhythm_v3_analytic_gap_clip"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_accepts_positive_analytic_gap_clip():
    validate_rhythm_training_hparams(
        {
            **_minimal_v3_hparams(),
            "rhythm_v3_analytic_gap_clip": 0.35,
        }
    )


def test_validate_rhythm_training_hparams_rejects_invalid_dynamic_budget_ratio():
    hparams = _minimal_v3_hparams()
    hparams["rhythm_v3_dynamic_budget_ratio"] = 1.5
    with pytest.raises(ValueError, match="rhythm_v3_dynamic_budget_ratio"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_max_prefix_budget_below_min():
    hparams = _minimal_v3_hparams()
    hparams["rhythm_v3_min_prefix_budget"] = 12
    hparams["rhythm_v3_max_prefix_budget"] = 8
    with pytest.raises(ValueError, match="rhythm_v3_max_prefix_budget"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_accepts_dynamic_prefix_budget_surface():
    validate_rhythm_training_hparams(
        {
            **_minimal_v3_hparams(),
            "rhythm_v3_dynamic_budget_ratio": 0.15,
            "rhythm_v3_min_prefix_budget": 12,
            "rhythm_v3_max_prefix_budget": 48,
        }
    )


def test_validate_rhythm_training_hparams_rejects_invalid_boundary_reset_thresh():
    hparams = _minimal_v3_hparams()
    hparams["rhythm_v3_boundary_reset_thresh"] = 1.2
    with pytest.raises(ValueError, match="rhythm_v3_boundary_reset_thresh"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_prompt_summary_without_explicit_silence_runs():
    hparams = _minimal_v3_hparams()
    hparams["rhythm_v3_backbone"] = "prompt_summary"
    hparams["rhythm_v3_warp_mode"] = "none"
    hparams["rhythm_v3_allow_hybrid"] = False
    hparams["rhythm_v3_anchor_mode"] = "source_observed"
    hparams["lambda_rhythm_op"] = 0.0
    hparams["lambda_rhythm_zero"] = 0.0
    hparams["rhythm_v3_emit_silence_runs"] = False
    with pytest.raises(ValueError, match="rhythm_v3_emit_silence_runs=true"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_minimal_v1_profile_without_simple_global_stats():
    hparams = _minimal_v3_hparams()
    hparams.update(
        {
            "rhythm_v3_minimal_v1_profile": True,
            "rhythm_v3_backbone": "prompt_summary",
            "rhythm_v3_warp_mode": "none",
            "rhythm_v3_allow_hybrid": False,
            "rhythm_v3_anchor_mode": "source_observed",
            "lambda_rhythm_op": 0.0,
            "lambda_rhythm_zero": 0.0,
            "rhythm_v3_use_continuous_alignment": True,
            "rhythm_v3_simple_global_stats": False,
        }
    )
    with pytest.raises(ValueError, match="rhythm_v3_simple_global_stats=true"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_minimal_v1_profile_when_same_text_paired_target_is_disabled():
    hparams = _minimal_v3_hparams()
    hparams.update(
        {
            "rhythm_v3_minimal_v1_profile": True,
            "rhythm_v3_backbone": "prompt_summary",
            "rhythm_v3_warp_mode": "none",
            "rhythm_v3_allow_hybrid": False,
            "rhythm_v3_anchor_mode": "source_observed",
            "lambda_rhythm_op": 0.0,
            "lambda_rhythm_zero": 0.0,
            "rhythm_v3_use_continuous_alignment": True,
            "rhythm_v3_simple_global_stats": True,
            "rhythm_v3_disallow_same_text_paired_target": True,
        }
    )
    with pytest.raises(ValueError, match="rhythm_v3_disallow_same_text_paired_target=false"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_minimal_v1_profile_without_same_text_paired_target_requirement():
    hparams = _minimal_v3_hparams()
    hparams.update(
        {
            "rhythm_v3_minimal_v1_profile": True,
            "rhythm_v3_backbone": "prompt_summary",
            "rhythm_v3_warp_mode": "none",
            "rhythm_v3_allow_hybrid": False,
            "rhythm_v3_anchor_mode": "source_observed",
            "lambda_rhythm_op": 0.0,
            "lambda_rhythm_zero": 0.0,
            "rhythm_v3_use_continuous_alignment": True,
            "rhythm_v3_simple_global_stats": True,
            "rhythm_v3_require_same_text_paired_target": False,
        }
    )
    with pytest.raises(ValueError, match="rhythm_v3_require_same_text_paired_target=true"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_minimal_v1_profile_with_reference_summary_enabled():
    hparams = _minimal_v3_hparams()
    hparams.update(
        {
            "rhythm_v3_minimal_v1_profile": True,
            "rhythm_v3_backbone": "prompt_summary",
            "rhythm_v3_warp_mode": "none",
            "rhythm_v3_allow_hybrid": False,
            "rhythm_v3_anchor_mode": "source_observed",
            "lambda_rhythm_op": 0.0,
            "lambda_rhythm_zero": 0.0,
            "rhythm_v3_use_continuous_alignment": True,
            "rhythm_v3_simple_global_stats": True,
            "rhythm_v3_use_reference_summary": True,
        }
    )
    with pytest.raises(ValueError, match="rhythm_v3_use_reference_summary=false"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_minimal_v1_profile_with_log_base_rate_enabled():
    hparams = _minimal_v3_hparams()
    hparams.update(
        {
            "rhythm_v3_minimal_v1_profile": True,
            "rhythm_v3_backbone": "prompt_summary",
            "rhythm_v3_warp_mode": "none",
            "rhythm_v3_allow_hybrid": False,
            "rhythm_v3_anchor_mode": "source_observed",
            "lambda_rhythm_op": 0.0,
            "lambda_rhythm_zero": 0.0,
            "rhythm_v3_use_continuous_alignment": True,
            "rhythm_v3_simple_global_stats": True,
            "rhythm_v3_use_log_base_rate": True,
        }
    )
    with pytest.raises(ValueError, match="rhythm_v3_use_log_base_rate=false"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_minimal_v1_profile_with_non_simple_rate_mode():
    hparams = _minimal_v3_hparams()
    hparams.update(
        {
            "rhythm_v3_minimal_v1_profile": True,
            "rhythm_v3_backbone": "prompt_summary",
            "rhythm_v3_warp_mode": "none",
            "rhythm_v3_allow_hybrid": False,
            "rhythm_v3_anchor_mode": "source_observed",
            "lambda_rhythm_op": 0.0,
            "lambda_rhythm_zero": 0.0,
            "rhythm_v3_use_continuous_alignment": True,
            "rhythm_v3_simple_global_stats": True,
            "rhythm_v3_rate_mode": "log_base",
        }
    )
    with pytest.raises(ValueError, match="rhythm_v3_rate_mode=simple_global"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_minimal_v1_profile_without_speech_only_summary_pooling():
    hparams = _minimal_v3_hparams()
    hparams.update(
        {
            "rhythm_v3_minimal_v1_profile": True,
            "rhythm_v3_backbone": "prompt_summary",
            "rhythm_v3_warp_mode": "none",
            "rhythm_v3_allow_hybrid": False,
            "rhythm_v3_anchor_mode": "source_observed",
            "lambda_rhythm_op": 0.0,
            "lambda_rhythm_zero": 0.0,
            "rhythm_v3_use_continuous_alignment": True,
            "rhythm_v3_simple_global_stats": True,
            "rhythm_v3_summary_pool_speech_only": False,
        }
    )
    with pytest.raises(ValueError, match="rhythm_v3_summary_pool_speech_only=true"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_minimal_v1_profile_without_cross_text_reference_contract():
    hparams = _minimal_v3_hparams()
    hparams.update(
        {
            "rhythm_v3_minimal_v1_profile": True,
            "rhythm_v3_backbone": "prompt_summary",
            "rhythm_v3_warp_mode": "none",
            "rhythm_v3_allow_hybrid": False,
            "rhythm_v3_anchor_mode": "source_observed",
            "lambda_rhythm_op": 0.0,
            "lambda_rhythm_zero": 0.0,
            "rhythm_v3_use_continuous_alignment": True,
            "rhythm_v3_simple_global_stats": True,
            "rhythm_v3_disallow_same_text_reference": False,
        }
    )
    with pytest.raises(ValueError, match="rhythm_v3_disallow_same_text_reference=true"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_minimal_v1_profile_without_continuous_alignment():
    with pytest.raises(ValueError, match="rhythm_v3_use_continuous_alignment=true"):
        validate_rhythm_training_hparams(_minimal_prompt_summary_v1_hparams())


def test_validate_rhythm_training_hparams_rejects_minimal_v1_profile_with_trainable_src_rate_init():
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams["rhythm_v3_freeze_src_rate_init"] = False
    with pytest.raises(ValueError, match="rhythm_v3_freeze_src_rate_init=true"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_accepts_minimal_v1_global_backbone_surface():
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True

    validate_rhythm_training_hparams(hparams)


@pytest.mark.parametrize("backbone", ["role_memory", "unit_run"])
def test_validate_rhythm_training_hparams_rejects_legacy_backbone_aliases_under_minimal_v1(backbone):
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_backbone"] = backbone
    hparams["rhythm_v3_use_continuous_alignment"] = True

    with pytest.raises(ValueError, match="forbids legacy rhythm_v3_backbone aliases"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_unit_norm_without_prior_path():
    hparams = _minimal_v3_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams["rhythm_v3_g_variant"] = "unit_norm"
    with pytest.raises(ValueError, match="rhythm_v3_unit_prior_path"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_minimal_v1_profile_with_unit_norm():
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams["rhythm_v3_g_variant"] = "unit_norm"
    hparams["rhythm_v3_unit_prior_path"] = str(ROOT / "tests" / "rhythm" / "fixtures" / "dummy_unit_prior.npz")
    with pytest.raises(
        ValueError,
        match="rhythm_v3_minimal_v1_profile with strict minimal claim profile forbids rhythm_v3_g_variant=unit_norm",
    ):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_allows_unit_norm_when_strict_claim_disabled():
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams["rhythm_v3_g_variant"] = "unit_norm"
    hparams["rhythm_v3_unit_prior_path"] = str(ROOT / "tests" / "rhythm" / "fixtures" / "dummy_unit_prior.npz")
    hparams["rhythm_v3_strict_minimal_claim_profile"] = False
    validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_minimal_v1_profile_with_src_gap_in_coarse_head():
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams["rhythm_v3_use_src_gap_in_coarse_head"] = True
    with pytest.raises(ValueError, match="rhythm_v3_use_src_gap_in_coarse_head=false"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_allows_src_gap_in_coarse_head_when_strict_claim_disabled():
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams["rhythm_v3_strict_minimal_claim_profile"] = False
    hparams["rhythm_v3_use_src_gap_in_coarse_head"] = True
    validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_allows_disable_learned_gate_false_when_strict_claim_disabled():
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams["rhythm_v3_strict_minimal_claim_profile"] = False
    hparams["rhythm_v3_disable_learned_gate"] = False
    validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_alignment_soft_repair_under_strict_claim():
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams["rhythm_v3_alignment_soft_repair"] = True
    with pytest.raises(ValueError, match="rhythm_v3_alignment_soft_repair=false"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_allows_alignment_soft_repair_when_strict_claim_disabled():
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams["rhythm_v3_strict_minimal_claim_profile"] = False
    hparams["rhythm_v3_alignment_soft_repair"] = True
    validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_strict_gate_without_status_path():
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams["rhythm_v3_gate_quality_strict"] = True
    with pytest.raises(
        ValueError,
        match="rhythm_v3_gate_quality_strict requires rhythm_v3_required_gate_status_json",
    ):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_strict_gate_with_missing_status_file(tmp_path):
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams["rhythm_v3_gate_quality_strict"] = True
    hparams["rhythm_v3_required_gate_status_json"] = str(tmp_path / "gate_status.json")
    with pytest.raises(ValueError, match="rhythm_v3_required_gate_status_json"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_strict_gate_without_debug_export(tmp_path):
    gate_status = tmp_path / "gate_status.json"
    gate_status.write_text('{"gate0_pass": false, "gate1_pass": false, "gate2_pass": false}', encoding="utf-8")
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams["rhythm_v3_gate_quality_strict"] = True
    hparams["rhythm_v3_required_gate_status_json"] = str(gate_status)
    hparams["rhythm_v3_debug_export"] = False
    with pytest.raises(ValueError, match="rhythm_v3_debug_export=true"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_strict_gate_without_strict_invalid_g(tmp_path):
    gate_status = tmp_path / "gate_status.json"
    gate_status.write_text('{"gate0_pass": true, "gate1_pass": true, "gate2_pass": false}', encoding="utf-8")
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams["rhythm_v3_gate_quality_strict"] = True
    hparams["rhythm_v3_required_gate_status_json"] = str(gate_status)
    hparams["rhythm_v3_strict_eval_invalid_g"] = False
    with pytest.raises(ValueError, match="rhythm_v3_strict_eval_invalid_g=true"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_strict_gate_when_gate_status_fails(tmp_path):
    gate_status = tmp_path / "gate_status.json"
    gate_status.write_text('{"gate0_pass": true, "gate1_pass": false, "gate2_pass": true}', encoding="utf-8")
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams["rhythm_v3_gate_quality_strict"] = True
    hparams["rhythm_v3_required_gate_status_json"] = str(gate_status)
    with pytest.raises(ValueError, match="gate0_pass=true and gate1_pass=true"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_strict_gate_when_gate_status_missing_required_keys(tmp_path):
    gate_status = tmp_path / "gate_status.json"
    gate_status.write_text('{"gate0_pass": true}', encoding="utf-8")
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams["rhythm_v3_gate_quality_strict"] = True
    hparams["rhythm_v3_required_gate_status_json"] = str(gate_status)
    with pytest.raises(ValueError, match="missing required keys: gate1_pass"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_missing_gate2_when_official_train_requires_it(tmp_path):
    gate_status = tmp_path / "gate_status.json"
    gate_status.write_text('{"gate0_pass": true, "gate1_pass": true, "gate2_pass": false}', encoding="utf-8")
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams["rhythm_v3_gate_quality_strict"] = True
    hparams["rhythm_v3_required_gate_status_json"] = str(gate_status)
    hparams["rhythm_v3_require_gate2_for_official_train"] = True
    with pytest.raises(ValueError, match="requires gate2_pass=true"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_gate_status_contract_mismatch(tmp_path):
    gate_status = tmp_path / "gate_status.json"
    gate_status.write_text(
        json.dumps(
            {
                "gate0_pass": True,
                "gate1_pass": True,
                "gate2_pass": True,
                "contract_fingerprint": {
                    "rhythm_v3_g_variant": "weighted_median",
                    "rhythm_v3_src_prefix_stat_mode": "exact_global_family",
                },
            }
        ),
        encoding="utf-8",
    )
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams["rhythm_v3_gate_quality_strict"] = True
    hparams["rhythm_v3_debug_export"] = True
    hparams["rhythm_v3_strict_eval_invalid_g"] = True
    hparams["rhythm_v3_required_gate_status_json"] = str(gate_status)
    hparams["rhythm_v3_g_variant"] = "raw_median"
    hparams["rhythm_v3_src_prefix_stat_mode"] = "ema"
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_missing_gate3_when_prefix_finetune_requires_it(tmp_path):
    gate_status = tmp_path / "gate_status.json"
    gate_status.write_text(
        '{"gate0_pass": true, "gate1_pass": true, "gate2_pass": true, "gate3_pass": false}',
        encoding="utf-8",
    )
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams["rhythm_v3_gate_quality_strict"] = True
    hparams["rhythm_v3_required_gate_status_json"] = str(gate_status)
    hparams["rhythm_v3_require_gate3_for_prefix_finetune"] = True
    with pytest.raises(ValueError, match="requires gate3_pass=true"):
        validate_rhythm_training_hparams(hparams)


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("rhythm_prompt_dropout", 0.1),
        ("rhythm_prompt_truncation", 0.5),
    ),
)
def test_validate_rhythm_training_hparams_rejects_minimal_v1_profile_with_prompt_augmentation(key, value):
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams[key] = value
    with pytest.raises(ValueError, match=key):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_nonexistent_unit_prior_path():
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams["rhythm_v3_unit_prior_path"] = str(ROOT / "tests" / "rhythm" / "fixtures" / "missing_unit_prior.npz")
    with pytest.raises(ValueError, match="rhythm_v3_unit_prior_path"):
        validate_rhythm_training_hparams(hparams)


@pytest.mark.parametrize("bad_ratio", [-0.1, 1.1])
def test_validate_rhythm_training_hparams_rejects_out_of_range_min_prompt_speech_ratio(bad_ratio):
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams["rhythm_v3_min_prompt_speech_ratio"] = bad_ratio
    with pytest.raises(ValueError, match="rhythm_v3_min_prompt_speech_ratio"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_invalid_budget_mode():
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams["rhythm_v3_budget_mode"] = "bad_mode"
    with pytest.raises(ValueError, match="rhythm_v3_budget_mode"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_accepts_prefix_optimal_integer_projection():
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams["rhythm_v3_integer_projection_mode"] = "prefix_optimal"
    hparams["rhythm_v3_integer_projection_anchor_mode"] = "continuous"
    hparams["rhythm_v3_prefix_projection_candidate_radius"] = 2
    hparams["rhythm_v3_prefix_projection_max_states"] = 32
    validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_invalid_integer_projection_mode():
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams["rhythm_v3_integer_projection_mode"] = "bad_mode"
    with pytest.raises(ValueError, match="rhythm_v3_integer_projection_mode"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_invalid_integer_projection_anchor_mode():
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams["rhythm_v3_integer_projection_anchor_mode"] = "bad_mode"
    with pytest.raises(ValueError, match="rhythm_v3_integer_projection_anchor_mode"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_minimal_v1_profile_with_discrete_alignment_mode():
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams["rhythm_v3_alignment_mode"] = "discrete"
    with pytest.raises(ValueError, match="rhythm_v3_alignment_mode"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_negative_short_alignment_aliases():
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams["rhythm_v3_align_lambda_emb"] = -0.1
    with pytest.raises(ValueError, match="rhythm_v3_alignment_lambda_emb"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_minimal_v1_profile_with_self_target_fallback():
    hparams = _minimal_v3_hparams()
    hparams.update(
        {
            "rhythm_v3_minimal_v1_profile": True,
            "rhythm_v3_backbone": "prompt_summary",
            "rhythm_v3_warp_mode": "none",
            "rhythm_v3_allow_hybrid": False,
            "rhythm_v3_anchor_mode": "source_observed",
            "lambda_rhythm_op": 0.0,
            "lambda_rhythm_zero": 0.0,
            "rhythm_v3_use_continuous_alignment": True,
            "rhythm_v3_simple_global_stats": True,
            "rhythm_v3_allow_source_self_target_fallback": True,
        }
    )
    with pytest.raises(ValueError, match="rhythm_v3_allow_source_self_target_fallback=false"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_minimal_v1_profile_with_alignment_source_skip_enabled():
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams["rhythm_v3_alignment_allow_source_skip"] = True
    with pytest.raises(ValueError, match="rhythm_v3_alignment_allow_source_skip=false"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_minimal_v1_profile_with_alignment_source_skip_short_alias_enabled():
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams["rhythm_v3_align_allow_source_skip"] = True
    with pytest.raises(ValueError, match="rhythm_v3_alignment_allow_source_skip=false"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_minimal_v1_profile_with_noncanonical_short_gap_silence_scale():
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams["rhythm_v3_short_gap_silence_scale"] = 0.5
    with pytest.raises(ValueError, match="rhythm_v3_short_gap_silence_scale=0.35"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_minimal_v1_profile_with_noncanonical_leading_silence_scale():
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams["rhythm_v3_leading_silence_scale"] = 0.1
    with pytest.raises(ValueError, match="rhythm_v3_leading_silence_scale=0.0"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_minimal_v1_profile_with_noncanonical_prompt_min_ref_len():
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams["rhythm_v3_min_prompt_ref_len_sec"] = 2.0
    with pytest.raises(ValueError, match="rhythm_v3_min_prompt_ref_len_sec=3.0"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_minimal_v1_profile_with_noncanonical_prompt_max_ref_len():
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams["rhythm_v3_max_prompt_ref_len_sec"] = 10.0
    with pytest.raises(ValueError, match="rhythm_v3_max_prompt_ref_len_sec=8.0"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_invalid_prompt_ref_len_order():
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams["rhythm_v3_min_prompt_ref_len_sec"] = 8.0
    hparams["rhythm_v3_max_prompt_ref_len_sec"] = 3.0
    with pytest.raises(ValueError, match="prompt_ref_len_sec"):
        validate_rhythm_training_hparams(hparams)


@pytest.mark.parametrize("key", ["rhythm_v3_alignment_min_dp_weight", "rhythm_v3_align_min_dp_weight"])
def test_validate_rhythm_training_hparams_rejects_negative_alignment_min_dp_weight(key):
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams[key] = -0.1
    with pytest.raises(ValueError, match="rhythm_v3_alignment_min_dp_weight"):
        validate_rhythm_training_hparams(hparams)


@pytest.mark.parametrize("key", ["rhythm_v3_alignment_local_margin_p10_min", "rhythm_v3_align_local_margin_p10_min"])
def test_validate_rhythm_training_hparams_rejects_negative_alignment_local_margin_threshold(key):
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams[key] = -0.01
    with pytest.raises(ValueError, match="local_margin_p10_min"):
        validate_rhythm_training_hparams(hparams)


@pytest.mark.parametrize(
    "key,bad_value,match",
    [
        ("rhythm_v3_coarse_delta_scale", -0.01, "rhythm_v3_coarse_delta_scale"),
        ("rhythm_v3_local_residual_scale", -0.01, "rhythm_v3_local_residual_scale"),
    ],
)
def test_validate_rhythm_training_hparams_rejects_negative_minimal_v1_scale_knobs(key, bad_value, match):
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams[key] = bad_value
    with pytest.raises(ValueError, match=match):
        validate_rhythm_training_hparams(hparams)


@pytest.mark.parametrize("bad_value", [-0.1, 1.1])
def test_validate_rhythm_training_hparams_rejects_out_of_range_boundary_offset_decay(bad_value):
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams["rhythm_v3_boundary_offset_decay"] = bad_value
    with pytest.raises(ValueError, match="rhythm_v3_boundary_offset_decay"):
        validate_rhythm_training_hparams(hparams)


@pytest.mark.parametrize("bad_value", [-0.1, 1.1])
def test_validate_rhythm_training_hparams_rejects_out_of_range_min_boundary_confidence_for_g(bad_value):
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams["rhythm_v3_min_boundary_confidence_for_g"] = bad_value
    with pytest.raises(ValueError, match="rhythm_v3_min_boundary_confidence_for_g"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_silence_aux_lambda_without_enable_flag():
    hparams = _minimal_v3_hparams()
    hparams["lambda_rhythm_silence_aux"] = 0.01
    with pytest.raises(ValueError, match="lambda_rhythm_silence_aux > 0 requires rhythm_v3_allow_silence_aux=true"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_accepts_silence_aux_lambda_when_enable_flag_is_set():
    validate_rhythm_training_hparams(
        {
            **_minimal_v3_hparams(),
            "rhythm_v3_allow_silence_aux": True,
            "lambda_rhythm_silence_aux": 0.01,
        }
    )


def test_validate_rhythm_training_hparams_rejects_minimal_v1_profile_with_silence_aux_weight():
    hparams = _minimal_v3_hparams()
    hparams.update(
        {
            "rhythm_v3_minimal_v1_profile": True,
            "rhythm_v3_backbone": "prompt_summary",
            "rhythm_v3_warp_mode": "none",
            "rhythm_v3_allow_hybrid": False,
            "rhythm_v3_anchor_mode": "source_observed",
            "rhythm_num_summary_slots": 1,
            "rhythm_v3_summary_use_unit_embedding": False,
            "rhythm_v3_summary_pool_speech_only": True,
            "rhythm_v3_disallow_same_text_reference": True,
            "rhythm_v3_disallow_same_text_paired_target": False,
            "rhythm_v3_require_same_text_paired_target": True,
            "rhythm_v3_allow_source_self_target_fallback": False,
            "rhythm_v3_use_continuous_alignment": True,
            "rhythm_v3_simple_global_stats": True,
            "rhythm_v3_rate_mode": "simple_global",
            "rhythm_v3_use_log_base_rate": False,
            "rhythm_v3_use_reference_summary": False,
            "rhythm_v3_use_learned_residual_gate": False,
            "rhythm_v3_disable_learned_gate": True,
            "rhythm_v3_silence_coarse_weight": 0.1,
        }
    )
    with pytest.raises(ValueError, match="rhythm_v3_silence_coarse_weight=0"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_minimal_v1_profile_with_silence_aux_enabled():
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams["rhythm_v3_allow_silence_aux"] = True
    with pytest.raises(ValueError, match="rhythm_v3_allow_silence_aux=false"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_invalid_src_rate_init_mode():
    hparams = _minimal_prompt_summary_v1_hparams()
    hparams["rhythm_v3_use_continuous_alignment"] = True
    hparams["rhythm_v3_src_rate_init_mode"] = "bad_mode"
    with pytest.raises(ValueError, match="rhythm_v3_src_rate_init_mode"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_minimal_v1_profile_with_non_strict_streaming():
    hparams = _minimal_v3_hparams()
    hparams.update(
        {
            "rhythm_v3_minimal_v1_profile": True,
            "rhythm_v3_backbone": "prompt_summary",
            "rhythm_v3_warp_mode": "none",
            "rhythm_v3_allow_hybrid": False,
            "rhythm_v3_anchor_mode": "source_observed",
            "lambda_rhythm_op": 0.0,
            "lambda_rhythm_zero": 0.0,
            "rhythm_v3_use_continuous_alignment": True,
            "rhythm_v3_simple_global_stats": True,
            "rhythm_streaming_mode": "lookahead",
        }
    )
    with pytest.raises(ValueError, match="rhythm_streaming_mode='strict'"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_unknown_v3_baseline_train_mode():
    hparams = _minimal_v3_hparams()
    hparams["rhythm_v3_baseline_train_mode"] = "alternating"
    with pytest.raises(ValueError, match="rhythm_v3_baseline_train_mode must be one of"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_accepts_frozen_baseline_lifecycle_surface():
    validate_rhythm_training_hparams(
        {
            **_minimal_v3_hparams(),
            "rhythm_v3_baseline_train_mode": "frozen",
            "rhythm_v3_baseline_ckpt": "baseline.ckpt",
            "rhythm_baseline_table_prior_path": "prior.pt",
        }
    )


def test_validate_rhythm_training_hparams_accepts_baseline_pretrain_surface():
    validate_rhythm_training_hparams(
        {
            **_minimal_v3_hparams(),
            "rhythm_v3_baseline_train_mode": "pretrain",
            "rhythm_v3_baseline_target_mode": "deglobalized",
            "lambda_rhythm_base": 1.0,
            "rhythm_public_losses": [
                "rhythm_total",
                "rhythm_v3_base",
                "rhythm_v3_dur",
                "rhythm_v3_op",
                "rhythm_v3_pref",
                "rhythm_v3_zero",
            ],
        }
    )


def test_validate_rhythm_training_hparams_rejects_pretrain_without_baseline_loss():
    with pytest.raises(ValueError, match="requires lambda_rhythm_base > 0"):
        validate_rhythm_training_hparams(
            {
                **_minimal_v3_hparams(),
                "rhythm_v3_baseline_train_mode": "pretrain",
                "lambda_rhythm_base": 0.0,
            }
        )


def test_validate_rhythm_training_hparams_rejects_unknown_v3_baseline_target_mode():
    with pytest.raises(ValueError, match="rhythm_v3_baseline_target_mode must be one of"):
        validate_rhythm_training_hparams(
            {
                **_minimal_v3_hparams(),
                "rhythm_v3_baseline_target_mode": "teacher_defined",
            }
        )


@pytest.mark.parametrize(
    "deprecated_key,new_key",
    [
        ("lambda_rhythm_anti", "lambda_rhythm_zero"),
    ],
)
def test_validate_rhythm_training_hparams_rejects_removed_v3_hparam_aliases(deprecated_key, new_key):
    hparams = _minimal_v3_hparams()
    hparams[deprecated_key] = 0.1
    with pytest.raises(ValueError, match=new_key):
        validate_rhythm_training_hparams(hparams)


@pytest.mark.parametrize(
    "removed_key",
    [
        "rhythm_v3_mem",
        "rhythm_v3_anti",
        "rhythm_role_codebook_size",
        "rhythm_anti_pos_bins",
        "rhythm_anti_pos_grl_scale",
        "rhythm_baseline_struct_enable",
        "rhythm_baseline_struct_scale_init",
    ],
)
def test_validate_rhythm_training_hparams_rejects_removed_v3_surface_keys(removed_key):
    hparams = _minimal_v3_hparams()
    hparams[removed_key] = 1
    with pytest.raises(ValueError, match=removed_key):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_accepts_prompt_summary_backbone_surface():
    validate_rhythm_training_hparams(
        {
            **_minimal_v3_hparams(),
            "rhythm_v3_backbone": "prompt_summary",
            "rhythm_v3_warp_mode": "none",
            "rhythm_v3_allow_hybrid": False,
            "rhythm_v3_anchor_mode": "source_observed",
            "lambda_rhythm_bias": 0.20,
            "rhythm_role_dim": 48,
            "rhythm_num_role_slots": 12,
            "lambda_rhythm_op": 0.25,
            "lambda_rhythm_zero": 0.0,
            "lambda_rhythm_ortho": 0.0,
        }
    )


def test_validate_rhythm_training_hparams_accepts_prompt_summary_public_surface_without_zero():
    validate_rhythm_training_hparams(
        {
            **_minimal_v3_hparams(),
            "rhythm_v3_backbone": "prompt_summary",
            "rhythm_v3_warp_mode": "none",
            "rhythm_v3_allow_hybrid": False,
            "rhythm_v3_anchor_mode": "source_observed",
            "lambda_rhythm_bias": 0.20,
            "lambda_rhythm_op": 0.0,
            "lambda_rhythm_summary": 0.25,
            "lambda_rhythm_zero": 0.0,
            "lambda_rhythm_ortho": 0.0,
            "rhythm_public_losses": [
                "rhythm_total",
                "rhythm_v3_dur",
                "rhythm_v3_bias",
                "rhythm_v3_summary",
                "rhythm_v3_pref",
            ],
        }
    )


def test_validate_rhythm_training_hparams_rejects_prompt_summary_without_source_anchor():
    with pytest.raises(ValueError, match="requires rhythm_v3_anchor_mode='source_observed'"):
        validate_rhythm_training_hparams(
            {
                **_minimal_v3_hparams(),
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_warp_mode": "none",
                "rhythm_v3_allow_hybrid": False,
                "rhythm_v3_anchor_mode": "baseline",
                "lambda_rhythm_zero": 0.0,
                "lambda_rhythm_ortho": 0.0,
            }
        )


def test_validate_rhythm_training_hparams_valid_v3_skips_legacy_contract_evaluation():
    with mock.patch("tasks.Conan.rhythm.task_config.collect_config_contract_evaluation") as mocked:
        validate_rhythm_training_hparams(_minimal_v3_hparams())
    mocked.assert_not_called()


def test_validate_rhythm_training_hparams_requires_prompt_unit_public_inputs_when_surface_is_declared():
    hparams = _minimal_v3_hparams()
    hparams["rhythm_public_inputs"] = ["content_units", "dur_anchor_src", "unit_anchor_base"]
    with pytest.raises(ValueError, match="rhythm_public_inputs is missing required rhythm_v3 entries"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_requires_prompt_speech_mask_for_prompt_summary_public_surface():
    hparams = {
        **_minimal_v3_hparams(),
        "rhythm_v3_backbone": "prompt_summary",
        "rhythm_v3_warp_mode": "none",
        "rhythm_v3_allow_hybrid": False,
        "rhythm_v3_anchor_mode": "source_observed",
        "lambda_rhythm_bias": 0.20,
        "lambda_rhythm_op": 0.0,
        "lambda_rhythm_zero": 0.0,
        "lambda_rhythm_ortho": 0.0,
        "rhythm_public_inputs": [
            "content_units",
            "dur_anchor_src",
            "unit_anchor_base",
            "prompt_content_units",
            "prompt_duration_obs",
            "prompt_unit_mask",
        ],
        "rhythm_public_losses": [
            "rhythm_total",
            "rhythm_v3_dur",
            "rhythm_v3_bias",
            "rhythm_v3_pref",
        ],
    }
    with pytest.raises(ValueError, match="prompt_speech_mask"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_legacy_public_inputs_for_v3():
    hparams = _minimal_v3_hparams()
    hparams["rhythm_public_inputs"] = [
        "content_units",
        "dur_anchor_src",
        "unit_anchor_base",
        "prompt_content_units",
        "prompt_duration_obs",
        "prompt_unit_mask",
        "ref_rhythm_trace",
    ]
    with pytest.raises(ValueError, match="rhythm_public_inputs contains legacy"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_rejects_legacy_public_output_for_v3():
    hparams = _minimal_v3_hparams()
    hparams["rhythm_public_outputs"] = [
        "speech_duration_exec",
        "rhythm_frame_plan",
        "commit_frontier",
        "rhythm_state_next",
        "pause_after_exec",
    ]
    with pytest.raises(ValueError, match="rhythm_public_outputs contains legacy"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_requires_consistency_loss_surface_when_enabled():
    hparams = _minimal_v3_hparams()
    hparams["lambda_rhythm_cons"] = 0.10
    hparams["rhythm_public_losses"] = [
        "rhythm_total",
        "rhythm_v3_dur",
        "rhythm_v3_op",
        "rhythm_v3_pref",
        "rhythm_v3_zero",
    ]
    with pytest.raises(ValueError, match="rhythm_v3_cons"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_requires_summary_surface_for_prompt_summary_public_losses():
    hparams = {
        **_minimal_v3_hparams(),
        "rhythm_v3_backbone": "prompt_summary",
        "rhythm_v3_warp_mode": "none",
        "rhythm_v3_allow_hybrid": False,
        "rhythm_v3_anchor_mode": "source_observed",
        "lambda_rhythm_bias": 0.20,
        "lambda_rhythm_op": 0.0,
        "lambda_rhythm_summary": 0.25,
        "lambda_rhythm_zero": 0.0,
        "lambda_rhythm_ortho": 0.0,
        "rhythm_public_losses": [
            "rhythm_total",
            "rhythm_v3_dur",
            "rhythm_v3_pref",
        ],
    }
    with pytest.raises(ValueError, match="rhythm_v3_summary"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_requires_baseline_loss_surface_when_enabled():
    hparams = _minimal_v3_hparams()
    hparams["lambda_rhythm_base"] = 0.10
    hparams["rhythm_public_losses"] = [
        "rhythm_total",
        "rhythm_v3_dur",
        "rhythm_v3_op",
        "rhythm_v3_pref",
        "rhythm_v3_zero",
    ]
    with pytest.raises(ValueError, match="rhythm_v3_base"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_accepts_role_memory_legacy_alias_surface():
    validate_rhythm_training_hparams(
        {
            **_minimal_v3_hparams(),
            "rhythm_v3_backbone": "role_memory",
            "rhythm_v3_warp_mode": "none",
            "rhythm_v3_allow_hybrid": False,
            "rhythm_v3_anchor_mode": "source_observed",
            "lambda_rhythm_bias": 0.20,
            "lambda_rhythm_mem": 0.25,
            "lambda_rhythm_zero": 0.0,
            "lambda_rhythm_ortho": 0.0,
            "rhythm_public_losses": [
                "rhythm_total",
                "rhythm_v3_dur",
                "rhythm_v3_bias",
                "rhythm_v3_mem",
                "rhythm_v3_pref",
            ],
        }
    )


def test_validate_rhythm_training_hparams_accepts_unit_run_alias_surface():
    validate_rhythm_training_hparams(
        {
            **_minimal_v3_hparams(),
            "rhythm_v3_backbone": "unit_run",
            "rhythm_v3_warp_mode": "none",
            "rhythm_v3_allow_hybrid": False,
            "rhythm_v3_anchor_mode": "source_observed",
            "lambda_rhythm_bias": 0.20,
            "lambda_rhythm_mem": 0.25,
            "lambda_rhythm_zero": 0.0,
            "lambda_rhythm_ortho": 0.0,
            "rhythm_public_losses": [
                "rhythm_total",
                "rhythm_v3_dur",
                "rhythm_v3_bias",
                "rhythm_v3_mem",
                "rhythm_v3_pref",
            ],
        }
    )


def test_validate_rhythm_training_hparams_allows_prompt_summary_public_losses_without_summary_when_disabled():
    validate_rhythm_training_hparams(
        {
            **_minimal_v3_hparams(),
            "rhythm_v3_backbone": "prompt_summary",
            "rhythm_v3_warp_mode": "none",
            "rhythm_v3_allow_hybrid": False,
            "rhythm_v3_anchor_mode": "source_observed",
            "lambda_rhythm_bias": 0.20,
            "lambda_rhythm_summary": 0.0,
            "lambda_rhythm_op": 0.0,
            "lambda_rhythm_zero": 0.0,
            "lambda_rhythm_ortho": 0.0,
            "rhythm_public_losses": [
                "rhythm_total",
                "rhythm_v3_dur",
                "rhythm_v3_bias",
                "rhythm_v3_pref",
            ],
        }
    )


def test_validate_rhythm_training_hparams_requires_bias_surface_for_prompt_summary_public_losses():
    hparams = {
        **_minimal_v3_hparams(),
        "rhythm_v3_backbone": "prompt_summary",
        "rhythm_v3_warp_mode": "none",
        "rhythm_v3_allow_hybrid": False,
        "rhythm_v3_anchor_mode": "source_observed",
        "lambda_rhythm_bias": 0.20,
        "lambda_rhythm_op": 0.0,
        "lambda_rhythm_summary": 0.25,
        "lambda_rhythm_zero": 0.0,
        "lambda_rhythm_ortho": 0.0,
        "rhythm_public_losses": [
            "rhythm_total",
            "rhythm_v3_dur",
            "rhythm_v3_summary",
            "rhythm_v3_pref",
        ],
    }
    with pytest.raises(ValueError, match="rhythm_v3_bias"):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_accepts_compact_example_public_surface():
    validate_rhythm_training_hparams(
        {
            **_minimal_v3_hparams(),
            "rhythm_public_inputs": [
                "content_units",
                "dur_anchor_src",
                "unit_anchor_base",
                "prompt_content_units",
                "prompt_duration_obs",
                "prompt_unit_mask",
            ],
            "rhythm_public_outputs": [
                "speech_duration_exec",
                "rhythm_frame_plan",
                "commit_frontier",
                "rhythm_state_next",
            ],
            "rhythm_public_losses": [
                "rhythm_total",
                "rhythm_v3_dur",
                "rhythm_v3_op",
                "rhythm_v3_pref",
                "rhythm_v3_zero",
            ],
        }
    )


def test_validate_rhythm_training_hparams_accepts_global_only_surface_when_operator_losses_are_disabled():
    validate_rhythm_training_hparams(
        {
            **_minimal_v3_hparams(),
            "rhythm_v3_backbone": "global_only",
            "rhythm_v3_warp_mode": "none",
            "rhythm_v3_allow_hybrid": False,
            "lambda_rhythm_op": 0.0,
            "lambda_rhythm_zero": 0.0,
            "lambda_rhythm_ortho": 0.0,
        }
    )


def test_validate_rhythm_training_hparams_accepts_progress_surface_when_operator_losses_are_disabled():
    validate_rhythm_training_hparams(
        {
            **_minimal_v3_hparams(),
            "rhythm_v3_backbone": "global_only",
            "rhythm_v3_warp_mode": "progress",
            "rhythm_v3_allow_hybrid": False,
            "lambda_rhythm_op": 0.0,
            "lambda_rhythm_zero": 0.0,
            "lambda_rhythm_ortho": 0.0,
        }
    )


def test_validate_rhythm_training_hparams_accepts_detector_surface_when_operator_losses_are_disabled():
    validate_rhythm_training_hparams(
        {
            **_minimal_v3_hparams(),
            "rhythm_v3_backbone": "global_only",
            "rhythm_v3_warp_mode": "detector",
            "rhythm_v3_allow_hybrid": False,
            "lambda_rhythm_op": 0.0,
            "lambda_rhythm_zero": 0.0,
            "lambda_rhythm_ortho": 0.0,
        }
    )


def test_validate_rhythm_training_hparams_accepts_operator_progress_surface():
    validate_rhythm_training_hparams(_minimal_v3_hparams())


def test_validate_rhythm_training_hparams_accepts_detector_candidate_surface():
    validate_rhythm_training_hparams(
        {
            **_minimal_v3_hparams(),
            "rhythm_v3_backbone": "global_only",
            "rhythm_v3_warp_mode": "detector",
            "rhythm_v3_allow_hybrid": False,
            "lambda_rhythm_op": 0.0,
            "lambda_rhythm_zero": 0.0,
            "lambda_rhythm_ortho": 0.0,
        }
    )


def test_validate_rhythm_training_hparams_accepts_progress_alias_hparams():
    validate_rhythm_training_hparams(
        {
            **_minimal_v3_hparams(),
            "rhythm_v3_backbone": "global_only",
            "rhythm_v3_warp_mode": "progress",
            "rhythm_v3_allow_hybrid": False,
            "rhythm_progress_bins": 4,
            "rhythm_progress_support_tau": 8.0,
            "lambda_rhythm_op": 0.0,
            "lambda_rhythm_zero": 0.0,
            "lambda_rhythm_ortho": 0.0,
        }
    )


def test_validate_rhythm_training_hparams_rejects_removed_coarse_progress_key():
    with pytest.raises(ValueError, match="rhythm_coarse_bins"):
        validate_rhythm_training_hparams(
            {
                **_minimal_v3_hparams(),
                "rhythm_coarse_bins": 4,
            }
        )


def test_validate_rhythm_training_hparams_rejects_removed_legacy_ablation_surface():
    with pytest.raises(ValueError, match="rhythm_v3_ablation has been removed"):
        validate_rhythm_training_hparams(
            {
                **_minimal_v3_hparams(),
                "rhythm_v3_ablation": "global_only",
            }
        )


def test_validate_rhythm_training_hparams_rejects_operator_detector_combo():
    with pytest.raises(ValueError, match="only valid with rhythm_v3_backbone='global_only'"):
        validate_rhythm_training_hparams(
            {
                **_minimal_v3_hparams(),
                "rhythm_v3_backbone": "operator",
                "rhythm_v3_warp_mode": "detector",
                "rhythm_v3_allow_hybrid": False,
            }
        )


def test_validate_rhythm_training_hparams_rejects_operator_progress_without_explicit_hybrid_enable():
    with pytest.raises(ValueError, match="requires rhythm_v3_allow_hybrid=true"):
        validate_rhythm_training_hparams(
            {
                **_minimal_v3_hparams(),
                "rhythm_v3_backbone": "operator",
                "rhythm_v3_warp_mode": "progress",
                "rhythm_v3_allow_hybrid": False,
            }
        )


def test_validate_rhythm_training_hparams_rejects_removed_proxy_infer_surface():
    with pytest.raises(ValueError, match="rhythm_v3_allow_proxy_infer"):
        validate_rhythm_training_hparams(
            {
                **_minimal_v3_hparams(),
                "rhythm_v3_allow_proxy_infer": True,
            }
        )


def test_validate_rhythm_training_hparams_rejects_global_only_with_operator_loss_budget():
    with pytest.raises(ValueError, match="runtime mode is 'global_only'"):
        validate_rhythm_training_hparams(
            {
                **_minimal_v3_hparams(),
                "rhythm_v3_backbone": "global_only",
                "rhythm_v3_warp_mode": "none",
                "rhythm_v3_allow_hybrid": False,
            }
        )


def test_validate_rhythm_training_hparams_rejects_progress_only_with_operator_loss_budget():
    with pytest.raises(ValueError, match="runtime mode is 'progress_only'"):
        validate_rhythm_training_hparams(
            {
                **_minimal_v3_hparams(),
                "rhythm_v3_backbone": "global_only",
                "rhythm_v3_warp_mode": "progress",
                "rhythm_v3_allow_hybrid": False,
            }
        )


def test_validate_rhythm_training_hparams_rejects_source_residual_gain_outside_srcres_runtime():
    with pytest.raises(ValueError, match="requires rhythm_v3_backbone='operator'"):
        validate_rhythm_training_hparams(
            {
                **_minimal_v3_hparams(),
                "rhythm_v3_backbone": "global_only",
                "rhythm_v3_warp_mode": "none",
                "rhythm_v3_allow_hybrid": False,
                "rhythm_v3_source_residual_gain": 0.5,
            }
        )


def test_validate_rhythm_training_hparams_accepts_operator_srcres_surface():
    validate_rhythm_training_hparams(
        {
            **_minimal_v3_hparams(),
            "rhythm_v3_backbone": "operator",
            "rhythm_v3_warp_mode": "none",
            "rhythm_v3_allow_hybrid": False,
            "rhythm_v3_source_residual_gain": 0.5,
        }
    )


def test_validate_rhythm_training_hparams_rejects_unknown_streaming_mode():
    with pytest.raises(ValueError, match="rhythm_streaming_mode must be one of"):
        validate_rhythm_training_hparams(
            {
                **_minimal_v3_hparams(),
                "rhythm_streaming_mode": "future_cheat",
            }
        )


def test_validate_rhythm_training_hparams_strict_mode_rejects_right_lookahead():
    with pytest.raises(ValueError, match="rhythm_response_window_right must be 0"):
        validate_rhythm_training_hparams(
            {
                **_minimal_v3_hparams(),
                "rhythm_streaming_mode": "strict",
                "rhythm_response_window_right": 1,
            }
        )


def test_validate_rhythm_training_hparams_strict_mode_rejects_micro_lookahead_units():
    with pytest.raises(ValueError, match="rhythm_micro_lookahead_units must be 0/None"):
        validate_rhythm_training_hparams(
            {
                **_minimal_v3_hparams(),
                "rhythm_streaming_mode": "strict",
                "rhythm_micro_lookahead_units": 1,
            }
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"rhythm_streaming_mode": "micro_lookahead", "rhythm_response_window_right": 1},
        {
            "rhythm_streaming_mode": "micro_lookahead",
            "rhythm_response_window_right": 0,
            "rhythm_micro_lookahead_units": 1,
        },
    ],
)
def test_validate_rhythm_training_hparams_accepts_explicit_micro_lookahead(updates):
    validate_rhythm_training_hparams(
        {
            **_minimal_v3_hparams(),
            **updates,
        }
    )


def test_validate_rhythm_training_hparams_micro_lookahead_requires_positive_lookahead():
    with pytest.raises(ValueError, match="micro_lookahead mode requires positive"):
        validate_rhythm_training_hparams(
            {
                **_minimal_v3_hparams(),
                "rhythm_streaming_mode": "micro_lookahead",
                "rhythm_response_window_right": 0,
                "rhythm_micro_lookahead_units": 0,
            }
        )


def test_validate_rhythm_training_hparams_rejects_non_positive_progress_bins():
    with pytest.raises(ValueError, match="rhythm_progress_bins must be > 0"):
        validate_rhythm_training_hparams(
            {
                **_minimal_v3_hparams(),
                "rhythm_progress_bins": 0,
            }
        )


def test_maintained_v3_yaml_defaults_to_minimal_v1_global_stats_surface():
    source = (ROOT / "egs" / "conan_emformer_rhythm_v3.yaml").read_text(encoding="utf-8")
    assert "rhythm_v3_minimal_v1_profile: true" in source
    assert "rhythm_v3_backbone: minimal_v1_global" in source
    assert "rhythm_v3_rate_mode: simple_global" in source
    assert "rhythm_v3_simple_global_stats: true" in source
    assert "rhythm_v3_use_log_base_rate: false" in source
    assert "rhythm_v3_disable_learned_gate: true" in source
    assert "rhythm_v3_eval_mode: learned" in source
    assert "rhythm_v3_g_variant: weighted_median" in source
    assert "rhythm_v3_prompt_domain_mode: meaningful_reference" in source
    assert "rhythm_v3_prompt_g_variant: weighted_median" in source
    assert "rhythm_v3_src_g_variant: weighted_median" in source
    assert "rhythm_v3_prompt_require_clean_support: false" in source
    assert "rhythm_v3_drop_edge_runs_for_g: 1" in source
    assert "rhythm_v3_min_boundary_confidence_for_g: 0.5" in source
    assert "rhythm_v3_use_continuous_alignment: true" in source
    assert "rhythm_v3_alignment_mode: continuous_viterbi_v1" in source
    assert "rhythm_v3_alignment_lambda_emb: 1.0" in source
    assert "rhythm_v3_alignment_lambda_type: 0.5" in source
    assert "rhythm_v3_alignment_lambda_band: 0.2" in source
    assert "rhythm_v3_alignment_band_ratio: 0.08" in source
    assert "rhythm_v3_alignment_unmatched_speech_ratio_max: 0.15" in source
    assert "rhythm_v3_alignment_mean_local_confidence_speech_min: 0.55" in source
    assert "rhythm_v3_alignment_mean_coarse_confidence_speech_min: 0.60" in source
    assert "rhythm_v3_alignment_local_margin_p10_min: 0.02" in source
    assert "rhythm_v3_budget_mode: total" in source
    assert "rhythm_v3_detach_global_term_in_local_head: true" in source
    assert "rhythm_v3_freeze_src_rate_init: true" in source
    assert "rhythm_v3_debug_export: true" in source
    assert "rhythm_v3_strict_minimal_claim_profile: true" in source
    assert "rhythm_v3_required_gate_status_json: egs/overrides/rhythm_v3_gate_status.json" in source
    assert "rhythm_v3_require_gate2_for_official_train: true" in source
    assert "rhythm_v3_strict_eval_invalid_g: true" in source
    assert "rhythm_v3_use_src_gap_in_coarse_head: false" in source
    assert "rhythm_v3_min_prompt_speech_ratio: 0.60" in source
    assert "rhythm_v3_min_prompt_ref_len_sec: 3.0" in source
    assert "rhythm_v3_max_prompt_ref_len_sec: 8.0" in source
    assert "rhythm_v3_coarse_delta_scale: 0.20" in source
    assert "rhythm_v3_local_residual_scale: 0.35" in source
    assert "rhythm_v3_boundary_offset_decay: 0.60" in source
    assert "rhythm_v3_src_rate_init_mode: first_speech" in source
    assert "rhythm_v3_src_rate_init_value: 0.0" in source
    assert "rhythm_v3_export_projector_telemetry: true" in source
    assert "rhythm_v3_alignment_prefilter_bad_samples: true" in source
    assert "rhythm_v3_alignment_prefilter_max_attempts: 2" in source
    assert "rhythm_prompt_dropout: 0.0" in source
    assert "rhythm_prompt_truncation: 0.0" in source
    assert "rhythm_v3_silence_coarse_weight: 0.0" in source
    assert "rhythm_v3_disallow_same_text_reference: true" in source
    assert "rhythm_v3_disallow_same_text_paired_target: false" in source
    assert "rhythm_v3_require_same_text_paired_target: true" in source
    assert "prompt_speech_mask" in source


def test_deprecated_v2_minimal_v1_yaml_now_aliases_v3_contract():
    source = (ROOT / "egs" / "conan_emformer_rhythm_v2_minimal_v1.yaml").read_text(encoding="utf-8")
    assert "base_config: egs/conan_emformer_rhythm_v3.yaml" in source
    assert "rhythm_enable_v2: false" in source
    assert "rhythm_enable_v3: true" in source
    assert "rhythm_v3_backbone: minimal_v1_global" in source
    assert "rhythm_v3_rate_mode: simple_global" in source
