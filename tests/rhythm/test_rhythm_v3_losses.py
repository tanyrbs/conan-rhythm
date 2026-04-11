from __future__ import annotations

import pytest
import torch

from modules.Conan.rhythm_v3.contracts import (
    PromptConditioningEvidence,
    ReferenceDurationMemory,
    StructuredDurationOperatorMemory,
)
from modules.Conan.rhythm_v3.runtime_adapter import ConanDurationAdapter
from tasks.Conan.rhythm.loss_routing import (
    compute_reporting_total_loss,
    route_conan_optimizer_losses,
    update_public_loss_aliases,
)
from tasks.Conan.rhythm.losses import build_rhythm_loss_dict
from tasks.Conan.rhythm.losses import DurationV3LossTargets
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
        "rhythm_ref_coverage_floor": 0.05,
        "rhythm_prefix_drift_gain": 0.25,
        "rhythm_prefix_drift_clip": 0.05,
        "rhythm_max_logstretch": 0.8,
        "rhythm_phrase_dim": 12,
        "rhythm_max_pause_frames": 4.0,
        "rhythm_v3_backbone": "operator",
        "rhythm_v3_warp_mode": "progress",
        "rhythm_v3_allow_hybrid": True,
        "rhythm_apply_mode": "always",
    }


def _build_prompt_summary_hparams():
    hparams = _build_hparams()
    hparams.update(
        {
            "rhythm_v3_backbone": "prompt_summary",
            "rhythm_v3_warp_mode": "none",
            "rhythm_v3_allow_hybrid": False,
            "rhythm_v3_anchor_mode": "source_observed",
            "rhythm_role_dim": 32,
            "rhythm_num_role_slots": 8,
            "lambda_rhythm_zero": 0.0,
            "lambda_rhythm_ortho": 0.0,
        }
    )
    return hparams


def _build_prompt_conditioning():
    return {
        "prompt_content_units": torch.tensor([[1, 2, 3, 4, 0]], dtype=torch.long),
        "prompt_duration_obs": torch.tensor([[3.0, 4.0, 2.0, 5.0, 0.0]], dtype=torch.float32),
        "prompt_unit_mask": torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0]], dtype=torch.float32),
    }


def test_rhythm_v3_loss_builder_returns_compact_losses():
    adapter = ConanDurationAdapter(_build_hparams(), hidden_size=32, vocab_size=128)
    content = torch.tensor([[1, 1, 2, 2, 3, 4, 4, 5]])
    ret = {}
    adapter(
        ret=ret,
        content=content,
        ref=None,
        target=None,
        f0=None,
        uv=None,
        infer=False,
        global_steps=0,
        content_embed=torch.randn(content.size(0), content.size(1), 32),
        tgt_nonpadding=torch.ones(content.size(0), content.size(1), 1),
        content_lengths=torch.full((content.size(0),), int(content.size(1)), dtype=torch.long),
        rhythm_state=None,
        rhythm_ref_conditioning=_build_prompt_conditioning(),
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
            lambda_cons=0.0,
            lambda_zero=0.0,
        ),
    )
    assert targets is not None
    ref_memory = ret["rhythm_ref_conditioning"]
    assert torch.allclose(targets.prompt_random_target_tgt, ref_memory.prompt_random_target)
    assert torch.allclose(targets.prompt_mask, ref_memory.prompt_mask)
    assert torch.allclose(targets.prompt_operator_fit_pred, ref_memory.prompt_operator_fit)
    assert torch.allclose(targets.prompt_operator_cv_fit_pred, ref_memory.prompt_operator_cv_fit)
    losses = build_rhythm_loss_dict(ret["rhythm_execution"], targets)
    expected_keys = {
        "rhythm_exec_speech",
        "rhythm_exec_stretch",
        "rhythm_prefix_state",
        "rhythm_v3_base",
        "rhythm_v3_dur",
        "rhythm_v3_op",
        "rhythm_v3_summary",
        "rhythm_v3_ortho",
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
    assert torch.allclose(losses["rhythm_v3_summary"], losses["rhythm_v3_op"])


def test_rhythm_v3_prompt_summary_targets_use_source_anchor_and_prompt_memory_loss():
    adapter = ConanDurationAdapter(_build_prompt_summary_hparams(), hidden_size=32, vocab_size=128)
    content = torch.tensor([[1, 1, 2, 2, 3, 4, 4, 5]], dtype=torch.long)
    ret = {}
    adapter(
        ret=ret,
        content=content,
        ref=None,
        target=None,
        f0=None,
        uv=None,
        infer=False,
        global_steps=0,
        content_embed=torch.randn(content.size(0), content.size(1), 32),
        tgt_nonpadding=torch.ones(content.size(0), content.size(1), 1),
        content_lengths=torch.full((content.size(0),), int(content.size(1)), dtype=torch.long),
        rhythm_state=None,
        rhythm_ref_conditioning=_build_prompt_conditioning(),
        rhythm_apply_override=None,
        rhythm_runtime_overrides=None,
        rhythm_source_cache=None,
        rhythm_offline_source_cache=None,
        speech_state_fn=lambda x: torch.randn(x.size(0), x.size(1), 32),
    )
    sample = {
        "unit_duration_tgt": ret["speech_duration_exec"].detach() + 0.2,
    }
    targets = build_duration_v3_loss_targets(
        sample=sample,
        output=ret,
        config=DurationV3TargetBuildConfig(
            lambda_dur=1.0,
            lambda_op=0.25,
            lambda_pref=0.20,
            lambda_cons=0.0,
            lambda_zero=0.0,
            lambda_ortho=0.0,
            anchor_mode="source_observed",
        ),
    )
    assert targets is not None
    unit_batch = ret["rhythm_unit_batch"]
    speech_mask = unit_batch.unit_mask.float() * (1.0 - unit_batch.sep_mask.float())
    assert torch.allclose(
        targets.prediction_anchor,
        torch.where(
            unit_batch.source_duration_obs.float() * speech_mask > 0.0,
            unit_batch.source_duration_obs.float() * speech_mask,
            unit_batch.unit_anchor_base.float() * speech_mask,
        ),
    )
    assert targets.prompt_role_attn is not None
    assert targets.prompt_role_value is not None
    assert targets.prompt_role_var is not None
    losses = build_rhythm_loss_dict(ret["rhythm_execution"], targets)
    assert "rhythm_v3_summary" in losses
    assert torch.allclose(losses["rhythm_v3_summary"], losses["rhythm_v3_op"])
    assert "rhythm_v3_mem" in losses
    assert torch.allclose(losses["rhythm_v3_mem"], losses["rhythm_v3_summary"])
    assert torch.isfinite(losses["rhythm_v3_summary"]).all()


def test_rhythm_v3_loss_routing_keeps_single_trainable_total():
    adapter = ConanDurationAdapter(_build_hparams(), hidden_size=32, vocab_size=128)
    content = torch.tensor([[1, 1, 2, 2, 3, 4, 4, 5]])
    ret = {}
    adapter(
        ret=ret,
        content=content,
        ref=None,
        target=None,
        f0=None,
        uv=None,
        infer=False,
        global_steps=0,
        content_embed=torch.randn(content.size(0), content.size(1), 32),
        tgt_nonpadding=torch.ones(content.size(0), content.size(1), 1),
        content_lengths=torch.full((content.size(0),), int(content.size(1)), dtype=torch.long),
        rhythm_state=None,
        rhythm_ref_conditioning=_build_prompt_conditioning(),
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
            lambda_cons=0.0,
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
        "rhythm_v3_base": torch.tensor(0.05),
        "rhythm_v3_dur": torch.tensor(0.1),
        "rhythm_v3_op": torch.tensor(0.2),
        "rhythm_v3_pref": torch.tensor(0.3),
        "rhythm_v3_cons": torch.tensor(0.4),
        "rhythm_v3_stream": torch.tensor(0.5),
        "rhythm_v3_zero": torch.tensor(0.6),
        "rhythm_v3_ortho": torch.tensor(0.7),
        "rhythm_is_v3_bundle": torch.tensor(1.0),
        "rhythm_total": torch.tensor(6.0),
    }
    update_public_loss_aliases(losses, mel_loss_names=())
    assert "rhythm_v3_summary" in losses
    assert torch.allclose(losses["rhythm_v3_summary"], losses["rhythm_v3_op"])
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
                lambda_cons=0.0,
                lambda_zero=0.05,
            ),
        )


def test_rhythm_v3_targets_require_prompt_targets_when_prompt_conditioning_loss_enabled():
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
                lambda_cons=0.0,
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
                lambda_cons=0.0,
                lambda_zero=0.05,
            ),
        )


def test_rhythm_v3_training_rejects_trace_proxy_without_prompt_units():
    adapter = ConanDurationAdapter(_build_hparams(), hidden_size=32, vocab_size=128)
    content = torch.tensor([[1, 1, 2, 2]], dtype=torch.long)
    with pytest.raises(ValueError, match="prompt_content_units / prompt_duration_obs / prompt_unit_mask"):
        adapter(
            ret={},
            content=content,
            ref=torch.randn(1, 12, 80),
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


def test_rhythm_v3_training_rejects_prebuilt_operator_dict_without_prompt_units():
    adapter = ConanDurationAdapter(_build_hparams(), hidden_size=32, vocab_size=128)
    content = torch.tensor([[1, 1, 2, 2]], dtype=torch.long)
    with pytest.raises(ValueError, match="prompt_content_units / prompt_duration_obs / prompt_unit_mask"):
        adapter(
            ret={},
            content=content,
            ref=None,
            target=None,
            f0=None,
            uv=None,
            infer=False,
            global_steps=0,
            content_embed=torch.randn(content.size(0), content.size(1), 32),
            tgt_nonpadding=torch.ones(content.size(0), content.size(1), 1),
            content_lengths=torch.full((content.size(0),), int(content.size(1)), dtype=torch.long),
            rhythm_state=None,
            rhythm_ref_conditioning={
                "global_rate": torch.zeros((1, 1)),
                "operator_coeff": torch.zeros((1, 4)),
            },
            rhythm_apply_override=None,
            rhythm_runtime_overrides=None,
            rhythm_source_cache=None,
            rhythm_offline_source_cache=None,
            speech_state_fn=lambda x: torch.randn(x.size(0), x.size(1), 32),
        )


def test_rhythm_v3_training_rejects_proxy_built_reference_memory_without_prompt_durations():
    adapter = ConanDurationAdapter(_build_hparams(), hidden_size=32, vocab_size=128)
    content = torch.tensor([[1, 1, 2, 2]], dtype=torch.long)
    proxy_memory = ReferenceDurationMemory(
        global_rate=torch.zeros((1, 1)),
        operator=StructuredDurationOperatorMemory(operator_coeff=torch.zeros((1, 4))),
        prompt=PromptConditioningEvidence(
            prompt_basis_activation=torch.zeros((1, 3, 4)),
            prompt_random_target=torch.zeros((1, 3)),
            prompt_mask=torch.ones((1, 3)),
            prompt_log_duration=None,
        ),
    )
    with pytest.raises(ValueError, match="prompt_content_units / prompt_duration_obs / prompt_unit_mask"):
        adapter(
            ret={},
            content=content,
            ref=None,
            target=None,
            f0=None,
            uv=None,
            infer=False,
            global_steps=0,
            content_embed=torch.randn(content.size(0), content.size(1), 32),
            tgt_nonpadding=torch.ones(content.size(0), content.size(1), 1),
            content_lengths=torch.full((content.size(0),), int(content.size(1)), dtype=torch.long),
            rhythm_state=None,
            rhythm_ref_conditioning=proxy_memory,
            rhythm_apply_override=None,
            rhythm_runtime_overrides=None,
            rhythm_source_cache=None,
            rhythm_offline_source_cache=None,
            speech_state_fn=lambda x: torch.randn(x.size(0), x.size(1), 32),
        )


def test_rhythm_v3_training_rejects_prompt_backed_reference_memory_even_with_prompt_logs():
    adapter = ConanDurationAdapter(_build_hparams(), hidden_size=32, vocab_size=128)
    content = torch.tensor([[1, 1, 2, 2]], dtype=torch.long)
    prompt_memory = ReferenceDurationMemory(
        global_rate=torch.zeros((1, 1)),
        operator=StructuredDurationOperatorMemory(operator_coeff=torch.zeros((1, 4))),
        prompt=PromptConditioningEvidence(
            prompt_basis_activation=torch.zeros((1, 3, 4)),
            prompt_random_target=torch.zeros((1, 3)),
            prompt_mask=torch.ones((1, 3)),
            prompt_log_duration=torch.zeros((1, 3)),
        ),
    )
    with pytest.raises(ValueError, match="prompt_content_units / prompt_duration_obs / prompt_unit_mask"):
        adapter(
            ret={},
            content=content,
            ref=None,
            target=None,
            f0=None,
            uv=None,
            infer=False,
            global_steps=0,
            content_embed=torch.randn(content.size(0), content.size(1), 32),
            tgt_nonpadding=torch.ones(content.size(0), content.size(1), 1),
            content_lengths=torch.full((content.size(0),), int(content.size(1)), dtype=torch.long),
            rhythm_state=None,
            rhythm_ref_conditioning=prompt_memory,
            rhythm_apply_override=None,
            rhythm_runtime_overrides=None,
            rhythm_source_cache=None,
            rhythm_offline_source_cache=None,
            speech_state_fn=lambda x: torch.randn(x.size(0), x.size(1), 32),
        )


def test_rhythm_v3_consistency_targets_tolerate_shorter_cached_prefix_state():
    unit_mask = torch.ones((1, 5), dtype=torch.float32)
    output = {
        "rhythm_execution": type("DummyExec", (), {"commit_mask": unit_mask})(),
        "rhythm_unit_batch": type("DummyBatch", (), {"unit_mask": unit_mask, "unit_anchor_base": torch.ones((1, 5))})(),
        "rhythm_ref_conditioning": type(
            "DummyRef",
            (),
            {
                "global_rate": torch.zeros((1, 1)),
                "prompt_basis_activation": torch.zeros((1, 4, 2)),
                "prompt_random_target": torch.zeros((1, 4)),
                "prompt_mask": torch.ones((1, 4)),
                "prompt_fit_mask": torch.ones((1, 4)),
                "prompt_eval_mask": torch.zeros((1, 4)),
                "prompt_operator_fit": torch.zeros((1, 4)),
                "prompt_operator_cv_fit": torch.zeros((1, 4)),
            },
        )(),
        "rhythm_state_prev": type(
            "DummyState",
            (),
            {
                "cached_duration_exec": torch.ones((1, 3)),
                "committed_units": torch.tensor([3]),
            },
        )(),
    }
    targets = build_duration_v3_loss_targets(
        sample={"unit_duration_tgt": torch.ones((1, 5))},
        output=output,
        config=DurationV3TargetBuildConfig(
            lambda_dur=1.0,
            lambda_op=0.25,
            lambda_pref=0.20,
            lambda_cons=0.10,
            lambda_zero=0.05,
        ),
    )
    assert targets is not None
    assert targets.consistency_duration_tgt is not None
    assert targets.consistency_duration_tgt.shape == (1, 5)
    assert targets.consistency_mask is not None
    assert targets.consistency_mask.shape == (1, 5)


def test_rhythm_v3_consistency_targets_are_skipped_when_loss_is_disabled():
    unit_mask = torch.ones((1, 5), dtype=torch.float32)
    output = {
        "rhythm_execution": type("DummyExec", (), {"commit_mask": unit_mask})(),
        "rhythm_unit_batch": type("DummyBatch", (), {"unit_mask": unit_mask, "unit_anchor_base": torch.ones((1, 5))})(),
        "rhythm_ref_conditioning": type(
            "DummyRef",
            (),
            {
                "global_rate": torch.zeros((1, 1)),
                "prompt_basis_activation": torch.zeros((1, 4, 2)),
                "prompt_random_target": torch.zeros((1, 4)),
                "prompt_mask": torch.ones((1, 4)),
                "prompt_fit_mask": torch.ones((1, 4)),
                "prompt_eval_mask": torch.zeros((1, 4)),
                "prompt_operator_fit": torch.zeros((1, 4)),
                "prompt_operator_cv_fit": torch.zeros((1, 4)),
            },
        )(),
        "rhythm_state_prev": type(
            "DummyState",
            (),
            {
                "cached_duration_exec": torch.ones((1, 3)),
                "committed_units": torch.tensor([3]),
            },
        )(),
    }
    targets = build_duration_v3_loss_targets(
        sample={"unit_duration_tgt": torch.ones((1, 5))},
        output=output,
        config=DurationV3TargetBuildConfig(
            lambda_dur=1.0,
            lambda_op=0.25,
            lambda_pref=0.20,
            lambda_cons=0.0,
            lambda_zero=0.05,
        ),
    )
    assert targets is not None
    assert targets.consistency_duration_tgt is None
    assert targets.consistency_mask is None


def test_rhythm_v3_loss_builder_ignores_separator_units_in_supervision():
    adapter = ConanDurationAdapter(_build_hparams(), hidden_size=32, vocab_size=128)
    content = torch.tensor([[1, 1, 2, 2, 3, 4, 4, 5]])
    ret = {}
    adapter(
        ret=ret,
        content=content,
        ref=None,
        target=None,
        f0=None,
        uv=None,
        infer=False,
        global_steps=0,
        content_embed=torch.randn(content.size(0), content.size(1), 32),
        tgt_nonpadding=torch.ones(content.size(0), content.size(1), 1),
        content_lengths=torch.full((content.size(0),), int(content.size(1)), dtype=torch.long),
        rhythm_state=None,
        rhythm_ref_conditioning=_build_prompt_conditioning(),
        rhythm_apply_override=None,
        rhythm_runtime_overrides=None,
        rhythm_source_cache=None,
        rhythm_offline_source_cache=None,
        speech_state_fn=lambda x: torch.randn(x.size(0), x.size(1), 32),
    )
    committed = (ret["rhythm_execution"].commit_mask > 0.5).nonzero(as_tuple=False)
    assert committed.numel() > 0
    sep_mask = ret["rhythm_unit_batch"].sep_mask.clone()
    sep_mask[committed[0, 0], committed[0, 1]] = 1.0
    ret["rhythm_unit_batch"].sep_mask = sep_mask
    baseline_target = ret["speech_duration_exec"].detach().clone()
    baseline_targets = build_duration_v3_loss_targets(
        sample={"unit_duration_tgt": baseline_target},
        output=ret,
        config=DurationV3TargetBuildConfig(
            lambda_dur=1.0,
            lambda_op=0.0,
            lambda_pref=0.20,
            lambda_cons=0.0,
            lambda_zero=0.0,
        ),
    )
    baseline_losses = build_rhythm_loss_dict(ret["rhythm_execution"], baseline_targets)
    masked_target = baseline_target.clone()
    masked_target[committed[0, 0], committed[0, 1]] = masked_target[committed[0, 0], committed[0, 1]] + 100.0
    masked_targets = build_duration_v3_loss_targets(
        sample={"unit_duration_tgt": masked_target},
        output=ret,
        config=DurationV3TargetBuildConfig(
            lambda_dur=1.0,
            lambda_op=0.0,
            lambda_pref=0.20,
            lambda_cons=0.0,
            lambda_zero=0.0,
        ),
    )
    masked_losses = build_rhythm_loss_dict(ret["rhythm_execution"], masked_targets)
    assert torch.allclose(masked_losses["rhythm_v3_dur"], baseline_losses["rhythm_v3_dur"])
    assert torch.allclose(masked_losses["rhythm_v3_pref"], baseline_losses["rhythm_v3_pref"])


def test_rhythm_v3_prefix_uses_sg_baseline_and_consistency_prefers_raw_duration_when_available():
    execution = type(
        "DummyExec",
        (),
        {
            "unit_logstretch": torch.zeros((1, 2), dtype=torch.float32),
            "unit_duration_raw": torch.full((1, 2), 2.0, dtype=torch.float32),
            "speech_duration_exec": torch.full((1, 2), 7.0, dtype=torch.float32),
            "basis_activation": None,
        },
    )()
    targets = DurationV3LossTargets(
        unit_duration_tgt=torch.full((1, 2), 2.0, dtype=torch.float32),
        unit_anchor_base=torch.full((1, 2), 2.0, dtype=torch.float32),
        unit_mask=torch.ones((1, 2), dtype=torch.float32),
        committed_mask=torch.ones((1, 2), dtype=torch.float32),
        consistency_duration_tgt=torch.full((1, 2), 2.0, dtype=torch.float32),
        consistency_mask=torch.ones((1, 2), dtype=torch.float32),
        lambda_dur=1.0,
        lambda_op=0.0,
        lambda_pref=0.20,
        lambda_cons=0.10,
        lambda_zero=0.0,
        lambda_ortho=0.0,
    )
    losses = build_rhythm_loss_dict(execution, targets)
    assert torch.allclose(losses["rhythm_v3_pref"], torch.tensor(0.0))
    assert torch.allclose(losses["rhythm_v3_cons"], torch.tensor(0.0))


def test_rhythm_v3_baseline_targets_can_be_deglobalized():
    unit_mask = torch.ones((1, 3), dtype=torch.float32)
    output = {
        "rhythm_execution": type("DummyExec", (), {"commit_mask": unit_mask})(),
        "rhythm_unit_batch": type(
            "DummyBatch",
            (),
            {
                "unit_mask": unit_mask,
                "sep_mask": torch.zeros((1, 3), dtype=torch.float32),
                "unit_anchor_base": torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float32),
            },
        )(),
        "rhythm_ref_conditioning": type(
            "DummyRef",
            (),
            {
                "global_rate": torch.zeros((1, 1)),
                "prompt_basis_activation": None,
                "prompt_random_target": None,
                "prompt_mask": None,
                "prompt_fit_mask": None,
                "prompt_eval_mask": None,
                "prompt_operator_fit": None,
                "prompt_operator_cv_fit": None,
            },
        )(),
    }
    targets = build_duration_v3_loss_targets(
        sample={"unit_duration_tgt": torch.tensor([[4.0, 4.0, 4.0]], dtype=torch.float32)},
        output=output,
        config=DurationV3TargetBuildConfig(
            lambda_dur=1.0,
            lambda_op=0.0,
            lambda_pref=0.0,
            lambda_base=1.0,
            baseline_target_mode="deglobalized",
        ),
    )
    assert targets is not None
    assert torch.allclose(targets.baseline_global_tgt, torch.full((1, 1), torch.log(torch.tensor(2.0))))
    assert torch.allclose(targets.baseline_duration_tgt, torch.full((1, 3), 2.0))


def test_rhythm_v3_baseline_pretrain_routes_total_to_baseline_only():
    execution = type(
        "DummyExec",
        (),
        {
            "unit_logstretch": torch.zeros((1, 2), dtype=torch.float32),
            "speech_duration_exec": torch.full((1, 2), 3.0, dtype=torch.float32),
            "basis_activation": None,
        },
    )()
    targets = DurationV3LossTargets(
        unit_duration_tgt=torch.full((1, 2), 3.0, dtype=torch.float32),
        unit_anchor_base=torch.full((1, 2), 2.0, dtype=torch.float32),
        unit_mask=torch.ones((1, 2), dtype=torch.float32),
        committed_mask=torch.ones((1, 2), dtype=torch.float32),
        baseline_duration_tgt=torch.full((1, 2), 2.0, dtype=torch.float32),
        baseline_mask=torch.ones((1, 2), dtype=torch.float32),
        lambda_dur=1.0,
        lambda_op=0.25,
        lambda_pref=0.20,
        lambda_base=1.0,
        lambda_zero=0.05,
        lambda_ortho=0.01,
        baseline_pretrain_only=True,
    )
    losses = build_rhythm_loss_dict(execution, targets)
    assert torch.allclose(losses["rhythm_total"], torch.tensor(0.0))
    assert torch.allclose(losses["rhythm_exec_speech"], torch.tensor(0.0))
    assert torch.allclose(losses["rhythm_exec_stretch"], torch.tensor(0.0))
    assert torch.allclose(losses["rhythm_prefix_state"], torch.tensor(0.0))
    assert torch.allclose(losses["rhythm_v3_base"], torch.tensor(0.0))
