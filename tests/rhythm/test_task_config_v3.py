from __future__ import annotations

from unittest import mock

import pytest

from tasks.Conan.rhythm.task_config import validate_rhythm_training_hparams


def _minimal_v3_hparams():
    return {
        "rhythm_enable_v2": False,
        "rhythm_enable_v3": True,
        "rhythm_response_rank": 4,
        "lambda_rhythm_dur": 1.0,
        "lambda_rhythm_op": 0.25,
        "lambda_rhythm_pref": 0.20,
        "lambda_rhythm_cons": 0.10,
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


@pytest.mark.parametrize(
    "deprecated_key,new_key",
    [
        ("lambda_rhythm_mem", "lambda_rhythm_op"),
        ("lambda_rhythm_anti", "lambda_rhythm_zero"),
    ],
)
def test_validate_rhythm_training_hparams_rejects_removed_v3_hparam_aliases(deprecated_key, new_key):
    hparams = _minimal_v3_hparams()
    hparams[deprecated_key] = 0.1
    with pytest.raises(ValueError, match=new_key):
        validate_rhythm_training_hparams(hparams)


def test_validate_rhythm_training_hparams_valid_v3_skips_legacy_contract_evaluation():
    with mock.patch("tasks.Conan.rhythm.task_config.collect_config_contract_evaluation") as mocked:
        validate_rhythm_training_hparams(_minimal_v3_hparams())
    mocked.assert_not_called()


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
