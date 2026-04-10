from __future__ import annotations

import pytest
import torch

from modules.Conan.rhythm_v3.runtime_adapter import ConanDurationAdapter
from tasks.Conan.rhythm.loss_routing import (
    compute_reporting_total_loss,
    route_conan_optimizer_losses,
    update_public_loss_aliases,
)
from tasks.Conan.rhythm.losses import build_rhythm_loss_dict
from tasks.Conan.rhythm.targets import DurationV3TargetBuildConfig, build_duration_v3_loss_targets


def _build_hparams():
    return {
        "silent_token": 57,
        "rhythm_separator_aware": True,
        "rhythm_tail_open_units": 1,
        "rhythm_anchor_hidden_size": 32,
        "rhythm_anchor_min_frames": 1.0,
        "rhythm_anchor_max_frames": 6.0,
        "rhythm_hidden_size": 64,
        "rhythm_response_rank": 4,
        "rhythm_response_window_left": 2,
        "rhythm_response_window_right": 0,
        "rhythm_trace_bins": 8,
        "rhythm_ref_coverage_floor": 0.05,
        "rhythm_prefix_drift_gain": 0.25,
        "rhythm_prefix_drift_clip": 0.05,
        "rhythm_max_logstretch": 0.8,
        "rhythm_phrase_dim": 12,
        "rhythm_max_pause_frames": 4.0,
        "rhythm_apply_mode": "always",
    }


def test_rhythm_v3_loss_builder_returns_compact_losses():
    adapter = ConanDurationAdapter(_build_hparams(), hidden_size=32, vocab_size=128)
    content = torch.tensor([[1, 1, 2, 2, 3, 4, 4, 5]])
    ref = torch.randn(1, 24, 80)
    ret = {}
    adapter(
        ret=ret,
        content=content,
        ref=ref,
        target=None,
        f0=None,
        uv=None,
        infer=False,
        global_steps=0,
        content_embed=torch.randn(content.size(0), content.size(1), 32),
        tgt_nonpadding=torch.ones(content.size(0), content.size(1), 1),
        content_lengths=torch.full((content.size(0),), int(content.size(1)), dtype=torch.long),
        rhythm_state=None,
        rhythm_ref_conditioning=None,
        rhythm_apply_override=None,
        rhythm_runtime_overrides=None,
        rhythm_source_cache=None,
        rhythm_offline_source_cache=None,
        speech_state_fn=lambda x: torch.randn(x.size(0), x.size(1), 32),
    )
    sample = {
        "unit_duration_tgt": ret["speech_duration_exec"].detach() + 0.25,
    }
    targets = build_duration_v3_loss_targets(
        sample=sample,
        output=ret,
        config=DurationV3TargetBuildConfig(
            lambda_dur=1.0,
            lambda_op=0.0,
            lambda_pref=0.20,
            lambda_cons=0.10,
            lambda_zero=0.0,
        ),
    )
    assert targets is not None
    ref_memory = ret["rhythm_ref_conditioning"]
    assert torch.allclose(targets.prompt_random_target_tgt, ref_memory.prompt_random_target)
    assert torch.allclose(targets.prompt_mask, ref_memory.prompt_mask)
    assert torch.allclose(targets.prompt_operator_fit_pred, ref_memory.prompt_operator_fit)
    losses = build_rhythm_loss_dict(ret["rhythm_execution"], targets)
    expected_keys = {
        "rhythm_exec_speech",
        "rhythm_exec_stretch",
        "rhythm_prefix_state",
        "rhythm_v3_dur",
        "rhythm_v3_op",
        "rhythm_v3_pref",
        "rhythm_v3_cons",
        "rhythm_v3_stream",
        "rhythm_v3_zero",
        "rhythm_is_v3_bundle",
        "rhythm_total",
    }
    assert set(losses.keys()) == expected_keys
    for key in expected_keys:
        assert key in losses
        assert torch.is_tensor(losses[key])
        assert torch.isfinite(losses[key]).all()
    for legacy_key in ("rhythm_v3_break", "rhythm_exec_pause", "rhythm_budget", "rhythm_prefix_clock", "rhythm_prefix_backlog"):
        assert legacy_key not in losses
    assert torch.equal(losses["rhythm_is_v3_bundle"], torch.tensor(1.0, device=losses["rhythm_is_v3_bundle"].device))


def test_rhythm_v3_loss_routing_keeps_single_trainable_total():
    adapter = ConanDurationAdapter(_build_hparams(), hidden_size=32, vocab_size=128)
    content = torch.tensor([[1, 1, 2, 2, 3, 4, 4, 5]])
    ref = torch.randn(1, 24, 80)
    ret = {}
    adapter(
        ret=ret,
        content=content,
        ref=ref,
        target=None,
        f0=None,
        uv=None,
        infer=False,
        global_steps=0,
        content_embed=torch.randn(content.size(0), content.size(1), 32),
        tgt_nonpadding=torch.ones(content.size(0), content.size(1), 1),
        content_lengths=torch.full((content.size(0),), int(content.size(1)), dtype=torch.long),
        rhythm_state=None,
        rhythm_ref_conditioning=None,
        rhythm_apply_override=None,
        rhythm_runtime_overrides=None,
        rhythm_source_cache=None,
        rhythm_offline_source_cache=None,
        speech_state_fn=lambda x: torch.randn(x.size(0), x.size(1), 32),
    )
    targets = build_duration_v3_loss_targets(
        sample={
            "unit_duration_tgt": ret["speech_duration_exec"].detach() + 0.1,
        },
        output=ret,
        config=DurationV3TargetBuildConfig(
            lambda_dur=1.0,
            lambda_op=0.0,
            lambda_pref=0.20,
            lambda_cons=0.10,
            lambda_zero=0.0,
        ),
    )
    losses = build_rhythm_loss_dict(ret["rhythm_execution"], targets)
    route_conan_optimizer_losses(
        losses,
        mel_loss_names=(),
        hparams={"rhythm_compact_joint_loss": True},
        schedule_only_stage=False,
    )
    trainable = [
        key for key, value in losses.items()
        if isinstance(value, torch.Tensor) and value.requires_grad
    ]
    assert trainable == ["rhythm_total"]
    report_total = compute_reporting_total_loss(
        losses,
        mel_loss_names=(),
        hparams={"rhythm_compact_joint_loss": True},
        schedule_only_stage=False,
    )
    assert torch.allclose(report_total, losses["rhythm_total"])


def test_rhythm_v3_public_aliases_do_not_reintroduce_legacy_rhythm_surface():
    losses = {
        "rhythm_exec_speech": torch.tensor(1.0),
        "rhythm_exec_stretch": torch.tensor(2.0),
        "rhythm_prefix_state": torch.tensor(3.0),
        "rhythm_v3_dur": torch.tensor(0.1),
        "rhythm_v3_op": torch.tensor(0.2),
        "rhythm_v3_pref": torch.tensor(0.3),
        "rhythm_v3_cons": torch.tensor(0.4),
        "rhythm_v3_stream": torch.tensor(0.5),
        "rhythm_v3_zero": torch.tensor(0.6),
        "rhythm_is_v3_bundle": torch.tensor(1.0),
        "rhythm_total": torch.tensor(6.0),
    }
    update_public_loss_aliases(losses, mel_loss_names=())
    for legacy_key in (
        "L_exec_speech",
        "L_exec_stretch",
        "L_prefix_state",
        "L_rhythm_exec",
        "L_stream_state",
    ):
        assert legacy_key not in losses


def test_rhythm_v3_targets_require_explicit_duration_target_by_default():
    ret = {
        "rhythm_execution": object(),
        "rhythm_unit_batch": type("DummyBatch", (), {"unit_mask": torch.ones((1, 3)), "unit_anchor_base": torch.ones((1, 3))})(),
    }
    with pytest.raises(ValueError, match="explicit unit duration target under unit_duration_tgt"):
        build_duration_v3_loss_targets(
            sample={},
            output=ret,
            config=DurationV3TargetBuildConfig(
                lambda_dur=1.0,
                lambda_op=0.25,
                lambda_pref=0.20,
                lambda_cons=0.10,
                lambda_zero=0.05,
            ),
        )


def test_rhythm_v3_targets_require_prompt_targets_when_memory_loss_enabled():
    ret = {
        "rhythm_execution": object(),
        "rhythm_unit_batch": type("DummyBatch", (), {"unit_mask": torch.ones((1, 3)), "unit_anchor_base": torch.ones((1, 3))})(),
        "rhythm_ref_conditioning": object(),
    }
    with pytest.raises(ValueError, match="output\\['rhythm_ref_conditioning'\\]"):
        build_duration_v3_loss_targets(
            sample={"unit_duration_tgt": torch.ones((1, 3))},
            output=ret,
            config=DurationV3TargetBuildConfig(
                lambda_dur=1.0,
                lambda_op=0.25,
                lambda_pref=0.20,
                lambda_cons=0.10,
                lambda_zero=0.05,
            ),
        )


def test_rhythm_v3_targets_reject_missing_unit_duration_target_even_with_executor():
    ret = {
        "rhythm_execution": object(),
        "rhythm_unit_batch": type("DummyBatch", (), {"unit_mask": torch.ones((1, 3)), "unit_anchor_base": torch.ones((1, 3))})(),
        "speech_duration_exec": torch.ones((1, 3)),
    }
    with pytest.raises(ValueError, match="explicit unit duration target under unit_duration_tgt"):
        build_duration_v3_loss_targets(
            sample={},
            output=ret,
            config=DurationV3TargetBuildConfig(
                lambda_dur=1.0,
                lambda_op=0.0,
                lambda_pref=0.20,
                lambda_cons=0.10,
                lambda_zero=0.05,
            ),
        )
