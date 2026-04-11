from __future__ import annotations

from unittest import mock

import pytest

from tasks.Conan.rhythm.task_config import validate_rhythm_training_hparams


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


def test_validate_rhythm_training_hparams_rejects_v2_v3_mutual_exclusion():
    with pytest.raises(ValueError, match="Enable only one rhythm backend"):
        validate_rhythm_training_hparams(
            {
                **_minimal_v3_hparams(),
                "rhythm_enable_v2": True,
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
