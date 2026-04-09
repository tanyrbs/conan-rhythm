from __future__ import annotations

import torch

from modules.Conan.rhythm_v3.runtime_adapter import ConanDurationAdapter
from tasks.Conan.rhythm.loss_routing import compute_reporting_total_loss, route_conan_optimizer_losses
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
        "rhythm_role_dim": 16,
        "rhythm_role_codebook_size": 4,
        "rhythm_role_window_left": 2,
        "rhythm_role_window_right": 0,
        "rhythm_trace_bins": 8,
        "rhythm_ref_coverage_floor": 0.05,
        "rhythm_prefix_drift_gain": 0.25,
        "rhythm_prefix_drift_clip": 0.05,
        "rhythm_max_logstretch": 0.8,
        "rhythm_anti_pos_bins": 4,
        "rhythm_anti_pos_grl_scale": 1.0,
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
        "rhythm_speech_exec_tgt": ret["speech_duration_exec"].detach() + 0.25,
    }
    targets = build_duration_v3_loss_targets(
        sample=sample,
        output=ret,
        config=DurationV3TargetBuildConfig(
            lambda_dur=1.0,
            lambda_mem=0.25,
            lambda_pref=0.20,
            lambda_anti=0.05,
            anti_pos_bins=4,
        ),
    )
    assert targets is not None
    losses = build_rhythm_loss_dict(ret["rhythm_execution"], targets)
    for key in ("rhythm_exec_speech", "rhythm_exec_stretch", "rhythm_prefix_state", "rhythm_total"):
        assert key in losses
        assert torch.is_tensor(losses[key])
        assert torch.isfinite(losses[key]).all()


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
        sample={"rhythm_speech_exec_tgt": ret["speech_duration_exec"].detach() + 0.1},
        output=ret,
        config=DurationV3TargetBuildConfig(
            lambda_dur=1.0,
            lambda_mem=0.25,
            lambda_pref=0.20,
            lambda_anti=0.05,
            anti_pos_bins=4,
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


def test_rhythm_v3_targets_require_explicit_duration_target_by_default():
    ret = {
        "rhythm_execution": object(),
        "rhythm_unit_batch": type("DummyBatch", (), {"unit_mask": torch.ones((1, 3)), "unit_anchor_base": torch.ones((1, 3))})(),
        "unit_anchor_base": torch.ones((1, 3)),
    }
    try:
        build_duration_v3_loss_targets(
            sample={"dur_anchor_src": torch.ones((1, 3))},
            output=ret,
            config=DurationV3TargetBuildConfig(
                lambda_dur=1.0,
                lambda_mem=0.25,
                lambda_pref=0.20,
                lambda_anti=0.05,
                anti_pos_bins=4,
            ),
        )
        raise AssertionError("Expected ValueError when explicit duration target is missing.")
    except ValueError:
        pass
