from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from modules.Conan.rhythm_v3.contracts import (
    PromptConditioningEvidence,
    ReferenceDurationMemory,
    StructuredDurationOperatorMemory,
)
from modules.Conan.rhythm_v3.contracts import move_source_unit_batch
from modules.Conan.rhythm_v3.global_condition import PromptGlobalConditionEncoderV1G
from modules.Conan.rhythm_v3.runtime_adapter import ConanDurationAdapter
from modules.Conan.rhythm_v3.summary_memory import PromptDurationMemoryEncoder
from tasks.Conan.rhythm.loss_routing import (
    compute_reporting_total_loss,
    route_conan_optimizer_losses,
    update_public_loss_aliases,
)
from tasks.Conan.rhythm.losses import (
    DurationV3LossTargets,
    _build_duration_v3_stream_losses,
    _resolve_duration_v3_prefix_target_surface,
    build_rhythm_loss_dict,
)
from tasks.Conan.rhythm.targets import DurationV3TargetBuildConfig, build_duration_v3_loss_targets
from tasks.Conan.rhythm.common.targets_impl import _build_duration_v3_silence_tau


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
            "lambda_rhythm_bias": 0.20,
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
        "prompt_valid_mask": torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0]], dtype=torch.float32),
        "prompt_speech_mask": torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0]], dtype=torch.float32),
        "prompt_closed_mask": torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0]], dtype=torch.float32),
        "prompt_boundary_confidence": torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0]], dtype=torch.float32),
        "prompt_ref_len_sec": torch.tensor([[4.0]], dtype=torch.float32),
        "prompt_speech_ratio_scalar": torch.tensor([[1.0]], dtype=torch.float32),
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
    expected_core_keys = {
        "rhythm_exec_speech",
        "rhythm_exec_stretch",
        "rhythm_prefix_state",
        "rhythm_v3_base",
        "rhythm_v3_dur",
        "rhythm_v3_bias",
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
    diagnostic_keys = {
        "rhythm_v3_committed_exec_prefix_discrepancy",
        "rhythm_v3_projected_exec_prefix_discrepancy",
        "rhythm_v3_coarse_explained_ratio",
        "rhythm_v3_local_residual_abs_mean",
        "rhythm_v3_local_residual_mean",
        "rhythm_v3_silence_local_leak_rate",
        "rhythm_v3_global_bias_tgt_support_mass",
        "rhythm_v3_global_bias_tgt_support_count",
        "rhythm_v3_coarse_target_speech_conf_mean",
        "rhythm_v3_silence_coarse_logstretch_tgt_abs_mean",
    }
    assert expected_core_keys.issubset(losses.keys())
    assert diagnostic_keys.issubset(losses.keys())
    for key in expected_core_keys | diagnostic_keys:
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
    silence_mask = getattr(unit_batch, "source_silence_mask", None)
    speech_mask = unit_batch.unit_mask.float()
    if isinstance(silence_mask, torch.Tensor):
        speech_mask = speech_mask * (1.0 - silence_mask.float())
    assert torch.allclose(
        targets.prediction_anchor,
        torch.where(
            unit_batch.source_duration_obs.float() * speech_mask > 0.0,
            unit_batch.source_duration_obs.float() * speech_mask,
            unit_batch.unit_anchor_base.float() * speech_mask,
        ),
    )
    assert targets is not None
    ref_memory = ret["rhythm_ref_conditioning"]
    if isinstance(targets.prompt_random_target_tgt, torch.Tensor) and isinstance(getattr(ref_memory, "prompt_random_target", None), torch.Tensor):
        assert torch.allclose(targets.prompt_random_target_tgt, ref_memory.prompt_random_target)
    if isinstance(targets.prompt_mask, torch.Tensor) and isinstance(getattr(ref_memory, "prompt_mask", None), torch.Tensor):
        assert torch.allclose(targets.prompt_mask, ref_memory.prompt_mask)
    if isinstance(targets.prompt_operator_fit_pred, torch.Tensor) and isinstance(getattr(ref_memory, "prompt_operator_fit", None), torch.Tensor):
        assert torch.allclose(targets.prompt_operator_fit_pred, ref_memory.prompt_operator_fit)
    if isinstance(targets.prompt_operator_cv_fit_pred, torch.Tensor) and isinstance(getattr(ref_memory, "prompt_operator_cv_fit", None), torch.Tensor):
        assert torch.allclose(targets.prompt_operator_cv_fit_pred, ref_memory.prompt_operator_cv_fit)
    losses = build_rhythm_loss_dict(ret["rhythm_execution"], targets)
    assert "rhythm_v3_bias" in losses
    assert torch.isfinite(losses["rhythm_v3_bias"]).all()
    assert "rhythm_v3_summary" in losses
    assert torch.allclose(losses["rhythm_v3_summary"], losses["rhythm_v3_op"])
    assert "rhythm_v3_mem" in losses
    assert torch.allclose(losses["rhythm_v3_mem"], losses["rhythm_v3_summary"])
    assert torch.isfinite(losses["rhythm_v3_summary"]).all()


def test_rhythm_v3_source_observed_prediction_anchor_masks_out_silence_runs():
    adapter = ConanDurationAdapter(_build_prompt_summary_hparams(), hidden_size=32, vocab_size=128)
    content = torch.tensor([[1, 1, 57, 57, 2, 2]], dtype=torch.long)
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
        sample={"unit_duration_tgt": ret["speech_duration_exec"].detach().clamp_min(1.0) + 0.2},
        output=ret,
        config=DurationV3TargetBuildConfig(
            lambda_dur=1.0,
            lambda_op=0.0,
            lambda_pref=0.0,
            lambda_cons=0.0,
            lambda_zero=0.0,
            lambda_ortho=0.0,
            anchor_mode="source_observed",
        ),
    )
    assert targets is not None
    silence_mask = ret["rhythm_unit_batch"].source_silence_mask.float()
    silence_positions = (silence_mask > 0.5).nonzero(as_tuple=False)
    assert silence_positions.numel() > 0
    for row, col in silence_positions.tolist():
        assert torch.allclose(targets.prediction_anchor[row, col], torch.tensor(0.0))


def test_rhythm_v3_minimal_profile_target_builder_rejects_baseline_loss_surface():
    adapter = ConanDurationAdapter(_build_prompt_summary_hparams(), hidden_size=32, vocab_size=128)
    content = torch.tensor([[1, 1, 2, 2]], dtype=torch.long)
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
    with pytest.raises(ValueError, match="forbids baseline/log-base loss"):
        build_duration_v3_loss_targets(
            sample={"unit_duration_tgt": ret["speech_duration_exec"].detach() + 0.2},
            output=ret,
            config=DurationV3TargetBuildConfig(
                lambda_dur=1.0,
                lambda_op=0.0,
                lambda_pref=0.0,
                lambda_base=1.0,
                minimal_v1_profile=True,
            ),
        )


def test_prompt_summary_global_rate_uses_raw_log_by_default():
    encoder = PromptDurationMemoryEncoder(
        vocab_size=16,
        dim=32,
        num_slots=4,
        operator_rank=2,
        coverage_floor=0.05,
        summary_pool_speech_only=True,
    )
    prompt_units = torch.tensor([[10, 11, 12, 13]], dtype=torch.long)
    prompt_duration_obs = torch.tensor([[6.0, 4.0, 10.0, 8.0]], dtype=torch.float32)
    prompt_anchor_base = torch.tensor([[3.0, 2.0, 2.0, 4.0]], dtype=torch.float32)
    valid_mask = torch.ones_like(prompt_duration_obs)
    speech_mask = torch.tensor([[1.0, 1.0, 0.0, 1.0]], dtype=torch.float32)
    memory = encoder(
        prompt_content_units=prompt_units,
        prompt_duration_obs=prompt_duration_obs,
        prompt_mask=valid_mask,
        prompt_valid_mask=valid_mask,
        prompt_speech_mask=speech_mask,
        prompt_unit_anchor_base=prompt_anchor_base,
    )
    speech_values = torch.log(prompt_duration_obs)[speech_mask.bool()]
    expected = torch.median(speech_values).reshape(1, 1)
    assert torch.allclose(memory.global_rate, expected)


def test_prompt_summary_global_rate_can_use_prompt_log_base_when_enabled():
    encoder = PromptDurationMemoryEncoder(
        vocab_size=16,
        dim=32,
        num_slots=4,
        operator_rank=2,
        coverage_floor=0.05,
        summary_pool_speech_only=True,
        use_log_base_rate=True,
    )
    prompt_units = torch.tensor([[10, 11, 12, 13]], dtype=torch.long)
    prompt_duration_obs = torch.tensor([[6.0, 4.0, 10.0, 8.0]], dtype=torch.float32)
    prompt_anchor_base = torch.tensor([[3.0, 2.0, 2.0, 4.0]], dtype=torch.float32)
    valid_mask = torch.ones_like(prompt_duration_obs)
    speech_mask = torch.tensor([[1.0, 1.0, 0.0, 1.0]], dtype=torch.float32)
    memory = encoder(
        prompt_content_units=prompt_units,
        prompt_duration_obs=prompt_duration_obs,
        prompt_mask=valid_mask,
        prompt_valid_mask=valid_mask,
        prompt_speech_mask=speech_mask,
        prompt_unit_anchor_base=prompt_anchor_base,
    )
    normalized_logdur = torch.log(prompt_duration_obs) - torch.log(prompt_anchor_base)
    speech_values = normalized_logdur[speech_mask.bool()]
    expected = torch.median(speech_values).reshape(1, 1)
    assert torch.allclose(memory.global_rate, expected)


def test_prompt_summary_pool_speech_only_outputs_zero_when_no_speech_runs():
    encoder_filter = PromptDurationMemoryEncoder(
        vocab_size=16,
        dim=32,
        num_slots=4,
        operator_rank=2,
        coverage_floor=0.05,
        summary_pool_speech_only=True,
    )
    encoder_all = PromptDurationMemoryEncoder(
        vocab_size=16,
        dim=32,
        num_slots=4,
        operator_rank=2,
        coverage_floor=0.05,
        summary_pool_speech_only=False,
    )
    prompt_units = torch.tensor([[5, 6, 7]], dtype=torch.long)
    valid_mask = torch.ones((1, 3), dtype=torch.float32)
    speech_mask = torch.zeros((1, 3), dtype=torch.float32)
    prompt_anchor_base = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
    durations = torch.tensor([[3.0, 4.0, 5.0]], dtype=torch.float32)
    common_kwargs = {
        "prompt_content_units": prompt_units,
        "prompt_mask": valid_mask,
        "prompt_valid_mask": valid_mask,
        "prompt_speech_mask": speech_mask,
        "prompt_unit_anchor_base": prompt_anchor_base,
    }
    summary_a = encoder_filter(
        **{**common_kwargs, "prompt_duration_obs": durations},
    ).summary_state
    summary_b = encoder_filter(
        **{**common_kwargs, "prompt_duration_obs": durations + torch.tensor([[1.0, -1.0, 3.0]], dtype=torch.float32)},
    ).summary_state
    delta_filter = torch.max(torch.abs(summary_a - summary_b))
    summary_c = encoder_all(
        **{**common_kwargs, "prompt_duration_obs": durations},
    ).summary_state
    summary_d = encoder_all(
        **{**common_kwargs, "prompt_duration_obs": durations + torch.tensor([[1.0, -1.0, 3.0]], dtype=torch.float32)},
    ).summary_state
    memory_filter = encoder_filter(
        **{**common_kwargs, "prompt_duration_obs": durations},
    )
    delta_all = torch.max(torch.abs(summary_c - summary_d))
    assert memory_filter.summary_state is not None
    assert torch.isfinite(memory_filter.global_rate).all()
    assert torch.allclose(memory_filter.global_rate, torch.zeros_like(memory_filter.global_rate))
    assert delta_filter < 5e-2
    assert delta_all > 1e-4


def test_prompt_summary_pool_speech_only_handles_mixed_speech_and_all_silence_batch():
    encoder = PromptDurationMemoryEncoder(
        vocab_size=16,
        dim=32,
        num_slots=4,
        operator_rank=2,
        coverage_floor=0.05,
        summary_pool_speech_only=True,
    )
    memory = encoder(
        prompt_content_units=torch.tensor([[5, 6, 7], [8, 9, 10]], dtype=torch.long),
        prompt_duration_obs=torch.tensor([[3.0, 4.0, 5.0], [6.0, 2.0, 1.0]], dtype=torch.float32),
        prompt_mask=torch.ones((2, 3), dtype=torch.float32),
        prompt_valid_mask=torch.ones((2, 3), dtype=torch.float32),
        prompt_speech_mask=torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=torch.float32),
        prompt_unit_anchor_base=torch.ones((2, 3), dtype=torch.float32),
    )
    assert memory.summary_state is not None
    assert torch.isfinite(memory.global_rate).all()
    assert torch.allclose(memory.global_rate[1], torch.zeros_like(memory.global_rate[1]))
    assert torch.count_nonzero(memory.prompt_mask[1]).item() == 3


def test_prompt_summary_strict_clean_global_support_uses_closed_boundary_clean_runs():
    encoder = PromptDurationMemoryEncoder(
        vocab_size=16,
        dim=32,
        num_slots=4,
        operator_rank=2,
        coverage_floor=0.05,
        summary_pool_speech_only=True,
        strict_clean_global_support=True,
        min_boundary_confidence=0.8,
    )
    encoder.eval()
    memory = encoder(
        prompt_content_units=torch.tensor([[5, 6, 7, 8, 9], [10, 11, 12, 13, 14]], dtype=torch.long),
        prompt_duration_obs=torch.tensor(
            [[2.0, 16.0, 8.0, 4.0, 32.0], [3.0, 5.0, 7.0, 11.0, 13.0]],
            dtype=torch.float32,
        ),
        prompt_mask=torch.ones((2, 5), dtype=torch.float32),
        prompt_valid_mask=torch.ones((2, 5), dtype=torch.float32),
        prompt_speech_mask=torch.ones((2, 5), dtype=torch.float32),
        prompt_unit_anchor_base=torch.ones((2, 5), dtype=torch.float32),
        prompt_closed_mask=torch.tensor(
            [[1.0, 0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        prompt_boundary_confidence=torch.tensor(
            [[0.95, 0.95, 0.92, 0.85, 0.30], [0.20, 0.30, 0.40, 0.10, 0.25]],
            dtype=torch.float32,
        ),
    )
    expected_clean_rate = torch.log(torch.tensor([[4.0]], dtype=torch.float32))
    assert memory.summary_state is not None
    assert torch.allclose(memory.global_rate[0], expected_clean_rate[0], atol=1.0e-5)
    assert torch.allclose(memory.global_rate[1], torch.zeros_like(memory.global_rate[1]))
    assert torch.isfinite(memory.summary_state).all()
    assert torch.isfinite(memory.prompt_role_attn).all()
    assert memory.prompt_g_support_mask is not None
    assert memory.prompt_g_clean_mask is not None
    assert memory.prompt_g_domain_valid is not None
    assert torch.equal(
        memory.prompt_g_support_mask[0].bool(),
        torch.tensor([True, False, True, True, False]),
    )
    assert torch.count_nonzero(memory.prompt_g_support_mask[1]).item() == 0
    assert torch.allclose(memory.prompt_g_domain_valid, torch.tensor([[1.0], [0.0]], dtype=torch.float32))
    assert torch.allclose(
        memory.prompt_g_support_ratio_vs_speech,
        torch.tensor([[3.0 / 5.0], [0.0]], dtype=torch.float32),
        atol=1.0e-6,
    )
    assert torch.allclose(
        memory.prompt_g_clean_ratio_vs_valid,
        torch.tensor([[3.0 / 5.0], [0.0]], dtype=torch.float32),
        atol=1.0e-6,
    )


def test_minimal_prompt_global_condition_encoder_keeps_invalid_support_rows_empty():
    encoder = PromptGlobalConditionEncoderV1G(
        operator_rank=2,
        min_speech_ratio=0.6,
        min_ref_len_sec=3.0,
        max_ref_len_sec=8.0,
        drop_edge_runs_for_g=1,
        min_boundary_confidence=0.8,
    )
    encoder.eval()
    memory = encoder(
        prompt_content_units=torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long),
        prompt_duration_obs=torch.tensor([[4.0, 8.0, 16.0, 32.0], [2.0, 3.0, 5.0, 7.0]], dtype=torch.float32),
        prompt_mask=torch.ones((2, 4), dtype=torch.float32),
        prompt_valid_mask=torch.ones((2, 4), dtype=torch.float32),
        prompt_speech_mask=torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        prompt_closed_mask=torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        prompt_boundary_confidence=torch.tensor([[0.95, 0.90, 0.92, 0.88], [0.10, 0.10, 0.10, 0.10]], dtype=torch.float32),
        prompt_ref_len_sec=torch.tensor([[5.0], [5.0]], dtype=torch.float32),
        prompt_speech_ratio_scalar=torch.tensor([[1.0], [2.0 / 17.0]], dtype=torch.float32),
    )
    assert torch.isfinite(memory.global_rate).all()
    assert torch.count_nonzero(memory.prompt_g_support_mask[1]).item() == 0
    assert torch.count_nonzero(memory.prompt_log_residual[1]).item() == 0
    assert torch.allclose(memory.global_rate[1], torch.zeros_like(memory.global_rate[1]))
    assert torch.allclose(memory.prompt_g_domain_valid, torch.tensor([[1.0], [0.0]], dtype=torch.float32))
    assert torch.allclose(
        memory.prompt_g_support_ratio_vs_valid,
        torch.tensor([[0.5], [0.0]], dtype=torch.float32),
        atol=1.0e-6,
    )


def test_prompt_summary_pool_speech_only_zeros_role_attention_on_silence():
    encoder = PromptDurationMemoryEncoder(
        vocab_size=16,
        dim=32,
        num_slots=4,
        operator_rank=2,
        coverage_floor=0.05,
        summary_pool_speech_only=True,
    )
    memory = encoder(
        prompt_content_units=torch.tensor([[5, 6, 7]], dtype=torch.long),
        prompt_duration_obs=torch.tensor([[3.0, 4.0, 5.0]], dtype=torch.float32),
        prompt_mask=torch.ones((1, 3), dtype=torch.float32),
        prompt_valid_mask=torch.ones((1, 3), dtype=torch.float32),
        prompt_speech_mask=torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32),
        prompt_unit_anchor_base=torch.ones((1, 3), dtype=torch.float32),
    )
    assert memory.prompt_role_attn is not None
    assert torch.allclose(memory.prompt_role_attn[:, 1, :], torch.zeros_like(memory.prompt_role_attn[:, 1, :]))


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
        "rhythm_v3_bias": torch.tensor(0.12),
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


def test_rhythm_v3_silence_runs_use_coarse_only_targets():
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
    silence_mask = ret["rhythm_unit_batch"].unit_mask.new_zeros(ret["rhythm_unit_batch"].unit_mask.shape)
    silence_mask[committed[0, 0], committed[0, 1]] = 1.0
    ret["rhythm_unit_batch"].source_silence_mask = silence_mask
    baseline_target = ret["speech_duration_exec"].detach().clone()
    config = DurationV3TargetBuildConfig(
        lambda_dur=1.0,
        lambda_op=0.0,
        lambda_pref=0.20,
        lambda_cons=0.0,
        lambda_zero=0.0,
        silence_coarse_weight=0.5,
        silence_logstretch_max=0.25,
    )
    baseline_targets = build_duration_v3_loss_targets(
        sample={"unit_duration_tgt": baseline_target},
        output=ret,
        config=config,
    )
    baseline_losses = build_rhythm_loss_dict(ret["rhythm_execution"], baseline_targets)
    masked_target = baseline_target.clone()
    masked_target[committed[0, 0], committed[0, 1]] = masked_target[committed[0, 0], committed[0, 1]] + 100.0
    masked_targets = build_duration_v3_loss_targets(
        sample={"unit_duration_tgt": masked_target},
        output=ret,
        config=config,
    )
    masked_losses = build_rhythm_loss_dict(ret["rhythm_execution"], masked_targets)
    assert baseline_targets is not None
    assert masked_targets is not None
    assert torch.allclose(baseline_targets.local_residual_tgt, masked_targets.local_residual_tgt)
    assert torch.allclose(baseline_targets.silence_mask, masked_targets.silence_mask)
    assert masked_targets.silence_coarse_weight == config.silence_coarse_weight
    assert isinstance(masked_targets.silence_coarse_logstretch_tgt, torch.Tensor)
    assert isinstance(masked_targets.global_bias_tgt_support_mass, torch.Tensor)
    assert isinstance(masked_targets.global_bias_tgt_support_count, torch.Tensor)
    assert isinstance(masked_targets.coarse_target_speech_conf_mean, torch.Tensor)
    row = int(committed[0, 0].item())
    col = int(committed[0, 1].item())
    zeros = masked_targets.local_residual_tgt.new_zeros(1)
    assert torch.allclose(masked_targets.local_residual_tgt[row, col].unsqueeze(0), zeros)
    expected_silence = baseline_targets.coarse_logstretch_tgt[row, col]
    assert abs(float(expected_silence.item())) <= float(config.silence_logstretch_max) + 1.0e-6
    assert torch.allclose(masked_targets.coarse_logstretch_tgt[row, col], expected_silence)
    expected_prefix_duration = (
        masked_targets.prediction_anchor[row, col].float()
        * torch.exp(masked_targets.coarse_logstretch_tgt[row, col].float())
    )
    assert torch.allclose(masked_targets.coarse_duration_tgt[row, col], expected_prefix_duration)
    assert torch.allclose(masked_targets.prefix_duration_tgt[row, col], expected_prefix_duration)
    assert torch.allclose(
        masked_targets.silence_coarse_logstretch_tgt[row, col],
        masked_targets.coarse_logstretch_tgt[row, col],
    )
    assert float(masked_targets.global_bias_tgt_support_mass.item()) > 0.0
    assert float(masked_targets.global_bias_tgt_support_count.item()) > 0.0
    assert float(masked_targets.coarse_target_speech_conf_mean.item()) > 0.0
    assert torch.allclose(masked_losses["rhythm_v3_dur"], baseline_losses["rhythm_v3_dur"])
    assert torch.allclose(masked_losses["rhythm_v3_bias"], baseline_losses["rhythm_v3_bias"])
    assert torch.allclose(masked_losses["rhythm_v3_pref"], baseline_losses["rhythm_v3_pref"])


def test_rhythm_v3_target_builder_applies_explicit_analytic_gap_clip():
    adapter = ConanDurationAdapter(_build_prompt_summary_hparams(), hidden_size=32, vocab_size=128)
    content = torch.tensor([[1, 1, 2, 2, 3, 4]], dtype=torch.long)
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
    ret["rhythm_ref_conditioning"].global_rate = torch.full((1, 1), 3.0, dtype=torch.float32)
    targets = build_duration_v3_loss_targets(
        sample={"unit_duration_tgt": ret["speech_duration_exec"].detach().clone()},
        output=ret,
        config=DurationV3TargetBuildConfig(
            lambda_dur=1.0,
            lambda_op=0.0,
            lambda_pref=0.20,
            lambda_cons=0.0,
            lambda_zero=0.0,
            analytic_gap_clip=0.25,
        ),
    )
    assert targets is not None
    assert targets.global_shift_tgt is not None
    assert float(targets.global_shift_tgt.abs().max().item()) <= 0.25 + 1.0e-6


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
        unit_duration_tgt=torch.full((1, 2), 5.0, dtype=torch.float32),
        unit_anchor_base=torch.full((1, 2), 2.0, dtype=torch.float32),
        unit_mask=torch.ones((1, 2), dtype=torch.float32),
        committed_mask=torch.ones((1, 2), dtype=torch.float32),
        coarse_duration_tgt=torch.full((1, 2), 2.0, dtype=torch.float32),
        unit_confidence_local_tgt=torch.ones((1, 2), dtype=torch.float32),
        unit_confidence_coarse_tgt=torch.ones((1, 2), dtype=torch.float32),
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


def test_rhythm_v3_loss_builder_exports_exec_prefix_and_role_diagnostics():
    execution = type(
        "DummyExec",
        (),
        {
            "unit_logstretch": torch.zeros((1, 3), dtype=torch.float32),
            "unit_duration_raw": torch.tensor([[2.0, 1.5, 1.0]], dtype=torch.float32),
            "speech_duration_exec": torch.tensor([[2.0, 1.0, 1.0]], dtype=torch.float32),
            "local_residual": torch.tensor([[0.10, -0.20, 0.05]], dtype=torch.float32),
            "coarse_correction": torch.tensor([[0.30, 0.30, 0.30]], dtype=torch.float32),
            "coarse_logstretch": torch.tensor([[0.40, 0.10, 0.30]], dtype=torch.float32),
            "basis_activation": None,
        },
    )()
    targets = DurationV3LossTargets(
        unit_duration_tgt=torch.tensor([[2.0, 1.0, 4.0]], dtype=torch.float32),
        unit_anchor_base=torch.ones((1, 3), dtype=torch.float32),
        prediction_anchor=torch.ones((1, 3), dtype=torch.float32),
        unit_mask=torch.ones((1, 3), dtype=torch.float32),
        committed_mask=torch.ones((1, 3), dtype=torch.float32),
        speech_mask=torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32),
        silence_mask=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
        committed_speech_mask=torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32),
        committed_silence_mask=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
        local_residual_tgt=torch.zeros((1, 3), dtype=torch.float32),
        coarse_logstretch_tgt=torch.tensor([[0.30, 0.30, 0.20]], dtype=torch.float32),
        coarse_duration_tgt=torch.tensor([[2.0, 2.0, 1.5]], dtype=torch.float32),
        silence_coarse_logstretch_tgt=torch.tensor([[0.0, 0.0, 0.20]], dtype=torch.float32),
        global_bias_tgt=torch.tensor([[0.30]], dtype=torch.float32),
        global_bias_tgt_support_mass=torch.tensor([[1.5]], dtype=torch.float32),
        global_bias_tgt_support_count=torch.tensor([[2.0]], dtype=torch.float32),
        coarse_target_speech_conf_mean=torch.tensor([[0.75]], dtype=torch.float32),
        local_residual_tgt_center=torch.tensor([[0.10]], dtype=torch.float32),
        local_residual_tgt_abs_mean=torch.tensor([[0.25]], dtype=torch.float32),
        unit_confidence_local_tgt=torch.ones((1, 3), dtype=torch.float32),
        unit_confidence_coarse_tgt=torch.ones((1, 3), dtype=torch.float32),
        lambda_dur=1.0,
        lambda_op=0.0,
        lambda_pref=0.20,
        lambda_bias=1.0,
        lambda_cons=0.0,
        lambda_zero=0.0,
        lambda_ortho=0.0,
    )
    losses = build_rhythm_loss_dict(execution, targets)
    assert losses["rhythm_v3_committed_exec_prefix_discrepancy"] > 0.0
    assert losses["rhythm_v3_projected_exec_prefix_discrepancy"] > 0.0
    assert 0.0 < float(losses["rhythm_v3_coarse_explained_ratio"].item()) < 1.0
    assert torch.allclose(losses["rhythm_v3_local_residual_mean"], torch.tensor(-0.05))
    assert torch.allclose(losses["rhythm_v3_local_residual_abs_mean"], torch.tensor(0.15))
    assert torch.allclose(losses["rhythm_v3_local_residual_tgt_center_abs"], torch.tensor(0.10))
    assert torch.allclose(losses["rhythm_v3_local_residual_tgt_abs_mean"], torch.tensor(0.25))
    assert torch.allclose(losses["rhythm_v3_local_residual_center_gap"], torch.tensor(0.15))
    assert torch.allclose(losses["rhythm_v3_silence_local_leak_rate"], torch.tensor(1.0))
    assert torch.allclose(losses["rhythm_v3_global_bias_tgt_support_mass"], torch.tensor(1.5))
    assert torch.allclose(losses["rhythm_v3_global_bias_tgt_support_count"], torch.tensor(2.0))
    assert torch.allclose(losses["rhythm_v3_coarse_target_speech_conf_mean"], torch.tensor(0.75))
    assert torch.allclose(losses["rhythm_v3_silence_coarse_logstretch_tgt_abs_mean"], torch.tensor(0.20))


def test_rhythm_v3_silence_aux_defaults_off_even_with_available_targets():
    execution = type(
        "DummyExec",
        (),
        {
            "unit_logstretch": torch.tensor([[0.0, 0.0]], dtype=torch.float32),
            "unit_logstretch_raw": torch.tensor([[0.0, 0.8]], dtype=torch.float32),
            "speech_duration_exec": torch.ones((1, 2), dtype=torch.float32),
            "basis_activation": None,
        },
    )()
    targets = DurationV3LossTargets(
        unit_duration_tgt=torch.ones((1, 2), dtype=torch.float32),
        unit_anchor_base=torch.ones((1, 2), dtype=torch.float32),
        prediction_anchor=torch.ones((1, 2), dtype=torch.float32),
        unit_mask=torch.ones((1, 2), dtype=torch.float32),
        committed_mask=torch.ones((1, 2), dtype=torch.float32),
        silence_mask=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        committed_silence_mask=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        silence_coarse_logstretch_tgt=torch.tensor([[0.0, 0.1]], dtype=torch.float32),
        unit_confidence_coarse_tgt=torch.ones((1, 2), dtype=torch.float32),
        lambda_dur=0.0,
        lambda_op=0.0,
        lambda_pref=0.0,
        lambda_bias=0.0,
        lambda_cons=0.0,
        lambda_zero=0.0,
        lambda_ortho=0.0,
        lambda_silence_aux=0.0,
    )
    losses = build_rhythm_loss_dict(execution, targets)
    assert torch.allclose(losses["rhythm_v3_silence_aux"], torch.tensor(0.0))
    assert torch.allclose(losses["rhythm_total"], torch.tensor(0.0))


def test_rhythm_v3_silence_aux_uses_raw_logstretch_when_enabled():
    execution = type(
        "DummyExec",
        (),
        {
            "unit_logstretch": torch.tensor([[0.0, 0.1]], dtype=torch.float32),
            "unit_logstretch_raw": torch.tensor([[0.0, 0.6]], dtype=torch.float32),
            "speech_duration_exec": torch.ones((1, 2), dtype=torch.float32),
            "basis_activation": None,
        },
    )()
    targets = DurationV3LossTargets(
        unit_duration_tgt=torch.ones((1, 2), dtype=torch.float32),
        unit_anchor_base=torch.ones((1, 2), dtype=torch.float32),
        prediction_anchor=torch.ones((1, 2), dtype=torch.float32),
        unit_mask=torch.ones((1, 2), dtype=torch.float32),
        committed_mask=torch.ones((1, 2), dtype=torch.float32),
        silence_mask=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        committed_silence_mask=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        silence_coarse_logstretch_tgt=torch.tensor([[0.0, 0.1]], dtype=torch.float32),
        unit_confidence_coarse_tgt=torch.ones((1, 2), dtype=torch.float32),
        lambda_dur=0.0,
        lambda_op=0.0,
        lambda_pref=0.0,
        lambda_bias=0.0,
        lambda_cons=0.0,
        lambda_zero=0.0,
        lambda_ortho=0.0,
        lambda_silence_aux=1.0,
    )
    losses = build_rhythm_loss_dict(execution, targets)
    expected = torch.nn.functional.smooth_l1_loss(
        torch.tensor([0.6], dtype=torch.float32),
        torch.tensor([0.1], dtype=torch.float32),
        beta=0.25,
        reduction="mean",
    )
    assert torch.allclose(losses["rhythm_v3_silence_aux"], expected)
    assert torch.allclose(losses["rhythm_total"], expected)


def test_rhythm_v3_loss_routes_local_and_coarse_confidence_separately():
    execution = type(
        "DummyExec",
        (),
        {
            "unit_logstretch": torch.zeros((1, 2), dtype=torch.float32),
            "speech_duration_exec": torch.ones((1, 2), dtype=torch.float32),
            "local_residual": torch.zeros((1, 2), dtype=torch.float32),
            "coarse_logstretch": torch.zeros((1, 2), dtype=torch.float32),
            "basis_activation": None,
        },
    )()
    base_targets = DurationV3LossTargets(
        unit_duration_tgt=torch.ones((1, 2), dtype=torch.float32),
        unit_anchor_base=torch.ones((1, 2), dtype=torch.float32),
        prediction_anchor=torch.ones((1, 2), dtype=torch.float32),
        unit_mask=torch.ones((1, 2), dtype=torch.float32),
        committed_mask=torch.ones((1, 2), dtype=torch.float32),
        speech_mask=torch.ones((1, 2), dtype=torch.float32),
        committed_speech_mask=torch.ones((1, 2), dtype=torch.float32),
        local_residual_tgt=torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        coarse_logstretch_tgt=torch.tensor([[0.25, 0.25]], dtype=torch.float32),
        coarse_duration_tgt=torch.exp(torch.tensor([[0.25, 0.25]], dtype=torch.float32)),
        unit_confidence_local_tgt=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        unit_confidence_coarse_tgt=torch.ones((1, 2), dtype=torch.float32),
        lambda_dur=1.0,
        lambda_op=0.0,
        lambda_pref=0.0,
        lambda_bias=1.0,
        lambda_cons=0.0,
        lambda_zero=0.0,
        lambda_ortho=0.0,
    )
    local_changed = replace(
        base_targets,
        local_residual_tgt=torch.tensor([[5.0, 0.5]], dtype=torch.float32),
    )
    coarse_changed = replace(
        base_targets,
        coarse_logstretch_tgt=torch.tensor([[1.0, 0.25]], dtype=torch.float32),
    )
    base_losses = build_rhythm_loss_dict(execution, base_targets)
    local_losses = build_rhythm_loss_dict(execution, local_changed)
    coarse_losses = build_rhythm_loss_dict(execution, coarse_changed)
    assert torch.allclose(local_losses["rhythm_v3_dur"], base_losses["rhythm_v3_dur"])
    assert not torch.allclose(coarse_losses["rhythm_v3_bias"], base_losses["rhythm_v3_bias"])


def test_rhythm_v3_bias_loss_prefers_global_bias_scalar_when_available():
    execution = type(
        "DummyExec",
        (),
        {
            "unit_logstretch": torch.zeros((1, 2), dtype=torch.float32),
            "speech_duration_exec": torch.ones((1, 2), dtype=torch.float32),
            "local_residual": torch.zeros((1, 2), dtype=torch.float32),
            "coarse_logstretch": torch.tensor([[3.0, 3.0]], dtype=torch.float32),
            "global_bias_scalar": torch.tensor([[0.25]], dtype=torch.float32),
            "basis_activation": None,
        },
    )()
    targets = DurationV3LossTargets(
        unit_duration_tgt=torch.ones((1, 2), dtype=torch.float32),
        unit_anchor_base=torch.ones((1, 2), dtype=torch.float32),
        prediction_anchor=torch.ones((1, 2), dtype=torch.float32),
        unit_mask=torch.ones((1, 2), dtype=torch.float32),
        committed_mask=torch.ones((1, 2), dtype=torch.float32),
        speech_mask=torch.ones((1, 2), dtype=torch.float32),
        committed_speech_mask=torch.ones((1, 2), dtype=torch.float32),
        local_residual_tgt=torch.zeros((1, 2), dtype=torch.float32),
        coarse_logstretch_tgt=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        coarse_duration_tgt=torch.ones((1, 2), dtype=torch.float32),
        global_bias_tgt=torch.tensor([[0.25]], dtype=torch.float32),
        unit_confidence_local_tgt=torch.ones((1, 2), dtype=torch.float32),
        unit_confidence_coarse_tgt=torch.ones((1, 2), dtype=torch.float32),
        lambda_dur=1.0,
        lambda_op=0.0,
        lambda_pref=0.0,
        lambda_bias=1.0,
        lambda_cons=0.0,
        lambda_zero=0.0,
        lambda_ortho=0.0,
    )
    losses = build_rhythm_loss_dict(execution, targets)
    assert torch.allclose(losses["rhythm_v3_bias"], torch.tensor(0.0))


def test_rhythm_v3_bias_loss_downweights_low_coarse_confidence_utterances():
    execution = type(
        "DummyExec",
        (),
        {
            "unit_logstretch": torch.zeros((2, 2), dtype=torch.float32),
            "speech_duration_exec": torch.ones((2, 2), dtype=torch.float32),
            "local_residual": torch.zeros((2, 2), dtype=torch.float32),
            "coarse_logstretch": torch.zeros((2, 2), dtype=torch.float32),
            "global_bias_scalar": torch.zeros((2, 1), dtype=torch.float32),
            "basis_activation": None,
        },
    )()
    base_targets = DurationV3LossTargets(
        unit_duration_tgt=torch.ones((2, 2), dtype=torch.float32),
        unit_anchor_base=torch.ones((2, 2), dtype=torch.float32),
        prediction_anchor=torch.ones((2, 2), dtype=torch.float32),
        unit_mask=torch.ones((2, 2), dtype=torch.float32),
        committed_mask=torch.ones((2, 2), dtype=torch.float32),
        committed_speech_mask=torch.ones((2, 2), dtype=torch.float32),
        global_bias_tgt=torch.tensor([[1.0], [4.0]], dtype=torch.float32),
        unit_confidence_local_tgt=torch.ones((2, 2), dtype=torch.float32),
        unit_confidence_coarse_tgt=torch.ones((2, 2), dtype=torch.float32),
        lambda_dur=1.0,
        lambda_op=0.0,
        lambda_pref=0.0,
        lambda_bias=1.0,
        lambda_cons=0.0,
        lambda_zero=0.0,
        lambda_ortho=0.0,
    )
    downweighted_targets = replace(
        base_targets,
        unit_confidence_coarse_tgt=torch.tensor([[1.0, 1.0], [0.1, 0.1]], dtype=torch.float32),
    )
    base_losses = build_rhythm_loss_dict(execution, base_targets)
    downweighted_losses = build_rhythm_loss_dict(execution, downweighted_targets)
    assert downweighted_losses["rhythm_v3_bias"] < base_losses["rhythm_v3_bias"]


def test_rhythm_v3_bias_loss_multiplies_sample_and_coarse_confidence():
    execution = type(
        "DummyExec",
        (),
        {
            "unit_logstretch": torch.zeros((2, 2), dtype=torch.float32),
            "speech_duration_exec": torch.ones((2, 2), dtype=torch.float32),
            "local_residual": torch.zeros((2, 2), dtype=torch.float32),
            "coarse_logstretch": torch.zeros((2, 2), dtype=torch.float32),
            "global_bias_scalar": torch.zeros((2, 1), dtype=torch.float32),
            "basis_activation": None,
        },
    )()
    targets = DurationV3LossTargets(
        unit_duration_tgt=torch.ones((2, 2), dtype=torch.float32),
        unit_anchor_base=torch.ones((2, 2), dtype=torch.float32),
        prediction_anchor=torch.ones((2, 2), dtype=torch.float32),
        unit_mask=torch.ones((2, 2), dtype=torch.float32),
        committed_mask=torch.ones((2, 2), dtype=torch.float32),
        committed_speech_mask=torch.ones((2, 2), dtype=torch.float32),
        global_bias_tgt=torch.tensor([[1.0], [4.0]], dtype=torch.float32),
        unit_confidence_local_tgt=torch.ones((2, 2), dtype=torch.float32),
        unit_confidence_coarse_tgt=torch.ones((2, 2), dtype=torch.float32),
        sample_confidence=torch.tensor([[1.0], [0.0]], dtype=torch.float32),
        lambda_dur=1.0,
        lambda_op=0.0,
        lambda_pref=0.0,
        lambda_bias=1.0,
        lambda_cons=0.0,
        lambda_zero=0.0,
        lambda_ortho=0.0,
    )
    losses = build_rhythm_loss_dict(execution, targets)
    expected = torch.nn.functional.smooth_l1_loss(
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        beta=0.25,
        reduction="mean",
    )
    assert torch.allclose(losses["rhythm_v3_bias"], expected)


def test_rhythm_v3_minimal_profile_loss_builder_rejects_prompt_operator_surface():
    execution = type(
        "DummyExec",
        (),
        {
            "unit_logstretch": torch.zeros((1, 2), dtype=torch.float32),
            "speech_duration_exec": torch.ones((1, 2), dtype=torch.float32),
            "basis_activation": None,
        },
    )()
    targets = DurationV3LossTargets(
        unit_duration_tgt=torch.ones((1, 2), dtype=torch.float32),
        unit_anchor_base=torch.ones((1, 2), dtype=torch.float32),
        unit_mask=torch.ones((1, 2), dtype=torch.float32),
        committed_mask=torch.ones((1, 2), dtype=torch.float32),
        lambda_dur=1.0,
        lambda_op=0.25,
        lambda_pref=0.0,
        minimal_v1_profile=True,
    )
    with pytest.raises(ValueError, match="forbids prompt summary/operator loss"):
        build_rhythm_loss_dict(execution, targets)


def test_rhythm_v3_minimal_profile_prefix_target_surface_requires_committed_silence_mask():
    targets = DurationV3LossTargets(
        unit_duration_tgt=torch.ones((1, 2), dtype=torch.float32),
        unit_anchor_base=torch.ones((1, 2), dtype=torch.float32),
        unit_mask=torch.ones((1, 2), dtype=torch.float32),
        committed_mask=torch.ones((1, 2), dtype=torch.float32),
        speech_mask=torch.ones((1, 2), dtype=torch.float32),
        committed_speech_mask=torch.ones((1, 2), dtype=torch.float32),
        minimal_v1_profile=True,
    )
    with pytest.raises(RuntimeError, match="prefix target surface"):
        _resolve_duration_v3_prefix_target_surface(targets)


def test_rhythm_v3_minimal_profile_prefix_consistency_requires_committed_silence_mask():
    execution = type(
        "DummyExec",
        (),
        {
            "unit_logstretch": torch.zeros((1, 2), dtype=torch.float32),
            "unit_logstretch_raw": torch.zeros((1, 2), dtype=torch.float32),
            "speech_duration_exec": torch.ones((1, 2), dtype=torch.float32),
            "local_residual": torch.zeros((1, 2), dtype=torch.float32),
            "unit_duration_raw": torch.ones((1, 2), dtype=torch.float32),
        },
    )()
    targets = DurationV3LossTargets(
        unit_duration_tgt=torch.ones((1, 2), dtype=torch.float32),
        unit_anchor_base=torch.ones((1, 2), dtype=torch.float32),
        unit_mask=torch.ones((1, 2), dtype=torch.float32),
        committed_mask=torch.ones((1, 2), dtype=torch.float32),
        speech_mask=torch.ones((1, 2), dtype=torch.float32),
        committed_speech_mask=torch.ones((1, 2), dtype=torch.float32),
        lambda_cons=0.5,
        consistency_duration_tgt=torch.ones((1, 2), dtype=torch.float32),
        consistency_mask=torch.ones((1, 2), dtype=torch.float32),
        minimal_v1_profile=True,
    )
    with pytest.raises(RuntimeError, match="prefix stream losses"):
        _build_duration_v3_stream_losses(
            pred_speech=torch.ones((1, 2), dtype=torch.float32),
            execution=execution,
            targets=targets,
            committed_mask=torch.ones((1, 2), dtype=torch.float32),
        )


def test_rhythm_v3_minimal_profile_prefix_target_surface_forbids_anchor_fallback_for_committed_silence():
    targets = DurationV3LossTargets(
        unit_duration_tgt=torch.tensor([[2.0, 3.0]], dtype=torch.float32),
        unit_anchor_base=torch.ones((1, 2), dtype=torch.float32),
        unit_mask=torch.ones((1, 2), dtype=torch.float32),
        committed_mask=torch.ones((1, 2), dtype=torch.float32),
        speech_mask=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        committed_speech_mask=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        committed_silence_mask=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        minimal_v1_profile=True,
    )
    with pytest.raises(RuntimeError, match="coarse-derived silence target"):
        _resolve_duration_v3_prefix_target_surface(targets)


def test_rhythm_v3_minimal_profile_prefix_consistency_forbids_anchor_fallback_for_committed_silence():
    execution = type(
        "DummyExec",
        (),
        {
            "unit_logstretch": torch.zeros((1, 2), dtype=torch.float32),
            "unit_logstretch_raw": torch.zeros((1, 2), dtype=torch.float32),
            "speech_duration_exec": torch.ones((1, 2), dtype=torch.float32),
            "local_residual": torch.zeros((1, 2), dtype=torch.float32),
            "unit_duration_raw": torch.ones((1, 2), dtype=torch.float32),
        },
    )()
    targets = DurationV3LossTargets(
        unit_duration_tgt=torch.tensor([[2.0, 3.0]], dtype=torch.float32),
        unit_anchor_base=torch.ones((1, 2), dtype=torch.float32),
        unit_mask=torch.ones((1, 2), dtype=torch.float32),
        committed_mask=torch.ones((1, 2), dtype=torch.float32),
        speech_mask=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        committed_speech_mask=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        committed_silence_mask=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        lambda_cons=0.5,
        consistency_duration_tgt=torch.ones((1, 2), dtype=torch.float32),
        consistency_mask=torch.ones((1, 2), dtype=torch.float32),
        minimal_v1_profile=True,
    )
    with pytest.raises(RuntimeError, match="coarse-derived silence target"):
        _build_duration_v3_stream_losses(
            pred_speech=torch.ones((1, 2), dtype=torch.float32),
            execution=execution,
            targets=targets,
            committed_mask=torch.ones((1, 2), dtype=torch.float32),
        )


def test_rhythm_v3_minimal_profile_target_builder_requires_global_rate():
    hparams = _build_prompt_summary_hparams()
    hparams["rhythm_v3_minimal_v1_profile"] = True
    hparams["rhythm_v3_rate_mode"] = "simple_global"
    hparams["rhythm_v3_simple_global_stats"] = True
    hparams["rhythm_v3_use_log_base_rate"] = False
    hparams["rhythm_v3_use_reference_summary"] = False
    hparams["rhythm_v3_use_learned_residual_gate"] = False
    hparams["rhythm_v3_disable_learned_gate"] = True
    adapter = ConanDurationAdapter(hparams, hidden_size=32, vocab_size=128)
    content = torch.tensor([[1, 1, 57, 57, 2, 2]], dtype=torch.long)
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
    broken = dict(ret)
    broken["rhythm_ref_conditioning"] = object()
    with pytest.raises(ValueError, match="requires rhythm_ref_conditioning.global_rate"):
        build_duration_v3_loss_targets(
            sample={"unit_duration_tgt": ret["speech_duration_exec"].detach() + 0.1},
            output=broken,
            config=DurationV3TargetBuildConfig(
                lambda_dur=1.0,
                lambda_op=0.0,
                lambda_pref=0.10,
                lambda_bias=0.20,
                lambda_zero=0.0,
                lambda_ortho=0.0,
                minimal_v1_profile=True,
            ),
        )


def test_rhythm_v3_minimal_profile_target_builder_omits_prompt_and_silence_aux_surfaces():
    hparams = _build_prompt_summary_hparams()
    hparams["rhythm_v3_minimal_v1_profile"] = True
    hparams["rhythm_v3_rate_mode"] = "simple_global"
    hparams["rhythm_v3_simple_global_stats"] = True
    hparams["rhythm_v3_use_log_base_rate"] = False
    hparams["rhythm_v3_use_reference_summary"] = False
    hparams["rhythm_v3_use_learned_residual_gate"] = False
    hparams["rhythm_v3_disable_learned_gate"] = True
    adapter = ConanDurationAdapter(hparams, hidden_size=32, vocab_size=128)
    content = torch.tensor([[1, 1, 57, 57, 2, 2]], dtype=torch.long)
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
        sample={"unit_duration_tgt": ret["speech_duration_exec"].detach() + 0.1},
        output=ret,
        config=DurationV3TargetBuildConfig(
            lambda_dur=1.0,
            lambda_op=0.0,
            lambda_pref=0.10,
            lambda_bias=0.20,
            lambda_zero=0.0,
            lambda_ortho=0.0,
            silence_coarse_weight=0.45,
            minimal_v1_profile=True,
        ),
    )
    assert targets is not None
    assert targets.silence_coarse_weight == pytest.approx(0.0)
    assert targets.baseline_duration_tgt is None
    assert targets.baseline_mask is None
    assert targets.baseline_global_tgt is None
    assert targets.prompt_basis_activation is None
    assert targets.prompt_random_target_tgt is None
    assert targets.prompt_mask is None
    assert targets.prompt_role_attn is None
    assert targets.prompt_log_residual is None


def test_prompt_summary_runtime_uses_learned_source_rate_init_without_history():
    adapter = ConanDurationAdapter(_build_prompt_summary_hparams(), hidden_size=32, vocab_size=128)
    adapter.module.duration_head.src_rate_init.data.fill_(1.2345)
    content = torch.tensor([[1, 1, 2, 2, 3, 4]], dtype=torch.long)
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
    source_rate_seq = ret["rhythm_execution"].source_rate_seq
    assert source_rate_seq is not None
    assert torch.allclose(source_rate_seq[0, 0], source_rate_seq.new_tensor(1.2345), atol=1e-4)


def test_prompt_summary_reference_global_rate_uses_prompt_log_base_normalization_when_enabled():
    hparams = _build_prompt_summary_hparams()
    hparams["rhythm_v3_use_log_base_rate"] = True
    adapter = ConanDurationAdapter(hparams, hidden_size=32, vocab_size=128)
    ref_memory = adapter.module.build_reference_conditioning(
        ref_conditioning={
            "prompt_content_units": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "prompt_duration_obs": torch.tensor([[4.0, 8.0, 4.0]], dtype=torch.float32),
            "prompt_unit_mask": torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
            "prompt_valid_mask": torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
            "prompt_speech_mask": torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
            "prompt_unit_anchor_base": torch.tensor([[2.0, 4.0, 2.0]], dtype=torch.float32),
        }
    )
    expected = torch.log(torch.tensor([[2.0]], dtype=torch.float32))
    assert torch.allclose(ref_memory.global_rate, expected, atol=1.0e-4)
    assert ref_memory.prompt_log_base is not None
    assert torch.allclose(
        ref_memory.prompt_log_base,
        torch.log(torch.tensor([[2.0, 4.0, 2.0]], dtype=torch.float32)),
        atol=1.0e-4,
    )


def test_prompt_summary_runtime_retimes_explicit_silence_runs_via_coarse_only_branch():
    hparams = _build_prompt_summary_hparams()
    hparams["rhythm_v3_emit_silence_runs"] = True
    hparams["rhythm_v3_silence_max_logstretch"] = 0.35
    adapter = ConanDurationAdapter(hparams, hidden_size=32, vocab_size=128)
    content = torch.tensor([[1, 1, 57, 57, 2, 2, 3, 3]], dtype=torch.long)
    ret = {}
    prompt = {
        "prompt_content_units": torch.tensor([[1, 2, 3, 0]], dtype=torch.long),
        "prompt_duration_obs": torch.tensor([[10.0, 10.0, 10.0, 0.0]], dtype=torch.float32),
        "prompt_unit_mask": torch.tensor([[1.0, 1.0, 1.0, 0.0]], dtype=torch.float32),
        "prompt_valid_mask": torch.tensor([[1.0, 1.0, 1.0, 0.0]], dtype=torch.float32),
        "prompt_speech_mask": torch.tensor([[1.0, 1.0, 1.0, 0.0]], dtype=torch.float32),
    }
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
        rhythm_ref_conditioning=prompt,
        rhythm_apply_override=None,
        rhythm_runtime_overrides=None,
        rhythm_source_cache=None,
        rhythm_offline_source_cache=None,
        speech_state_fn=lambda x: torch.randn(x.size(0), x.size(1), 32),
    )
    unit_batch = ret["rhythm_unit_batch"]
    execution = ret["rhythm_execution"]
    silence_positions = ((unit_batch.source_silence_mask > 0.5) & (execution.commit_mask > 0.5)).nonzero(as_tuple=False)
    assert silence_positions.numel() > 0
    row = int(silence_positions[0, 0].item())
    col = int(silence_positions[0, 1].item())
    assert torch.allclose(execution.local_residual[row, col].unsqueeze(0), torch.zeros(1))
    silence_tau = _build_duration_v3_silence_tau(
        prediction_anchor=unit_batch.source_duration_obs.float(),
        committed_silence_mask=(unit_batch.source_silence_mask.float() * execution.commit_mask.float()),
        sep_hint=unit_batch.sep_mask.float(),
        boundary_cue=getattr(unit_batch, "source_boundary_cue", None),
        max_silence_logstretch=float(hparams["rhythm_v3_silence_max_logstretch"]),
        short_gap_scale=float(hparams.get("rhythm_v3_short_gap_silence_scale", 0.35)),
    )
    expected = execution.coarse_logstretch[row, col].clamp(
        min=-silence_tau[row, col],
        max=silence_tau[row, col],
    )
    assert torch.allclose(execution.unit_logstretch[row, col], expected, atol=1.0e-5)
    expected_duration = unit_batch.source_duration_obs[row, col].float() * torch.exp(expected.float())
    if isinstance(getattr(execution, "unit_duration_raw", None), torch.Tensor):
        assert torch.allclose(execution.unit_duration_raw[row, col], expected_duration, atol=1.0e-5)
    assert float(execution.unit_duration_exec[row, col].item()) >= 1.0


def test_prompt_summary_runtime_cold_start_gates_first_speech_local_residual():
    adapter = ConanDurationAdapter(_build_prompt_summary_hparams(), hidden_size=32, vocab_size=128)
    content = torch.tensor([[1, 1, 2, 2, 3, 3]], dtype=torch.long)
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
    execution = ret["rhythm_execution"]
    assert torch.allclose(execution.local_residual[0, 0], torch.tensor(0.0))


def test_prompt_summary_runtime_accepts_reference_memory_without_role_sidecars():
    adapter = ConanDurationAdapter(_build_prompt_summary_hparams(), hidden_size=32, vocab_size=128)
    content = torch.tensor([[1, 1, 2, 2, 3, 4]], dtype=torch.long)
    source_batch = move_source_unit_batch(
        adapter._build_source_batch(
            content=content,
            content_lengths=torch.full((1,), int(content.size(1)), dtype=torch.long),
            rhythm_source_cache=None,
            infer=False,
        ),
        device=content.device,
    )
    ref_memory = adapter.module.build_reference_conditioning(ref_conditioning=_build_prompt_conditioning())
    ref_memory.role = None
    execution = adapter.module(
        source_batch=source_batch,
        ref_memory=ref_memory,
        state=None,
    )
    assert execution.global_bias_scalar is not None
    assert execution.source_rate_seq is not None


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
