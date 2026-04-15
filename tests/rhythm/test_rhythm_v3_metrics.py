from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from modules.Conan.rhythm_v3.runtime_adapter import ConanDurationAdapter
from tasks.Conan.rhythm.duration_v3.metrics import (
    local_silence_delta_share,
    monotonic_triplet_table,
    residual_bias_share,
    residual_target_stats,
    silence_leakage,
    speech_weighted_mae,
    tempo_explainability,
    tempo_monotonicity,
    tempo_tie_rate,
)
from tasks.Conan.rhythm.metrics import build_rhythm_metric_dict, build_rhythm_metric_sections
from tasks.Conan.rhythm.common.targets_impl import (
    DurationV3TargetBuildConfig,
    _build_duration_v3_silence_tau,
    build_duration_v3_loss_targets,
)


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
        "rhythm_max_logstretch": 0.8,
        "rhythm_streaming_mode": "strict",
        "rhythm_v3_backbone": "operator",
        "rhythm_v3_warp_mode": "progress",
        "rhythm_v3_allow_hybrid": True,
        "rhythm_apply_mode": "always",
    }


def _run_adapter():
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
        infer=True,
        global_steps=0,
        content_embed=torch.randn(content.size(0), content.size(1), 32),
        tgt_nonpadding=torch.ones(content.size(0), content.size(1), 1),
        content_lengths=torch.full((content.size(0),), int(content.size(1)), dtype=torch.long),
        rhythm_state=None,
        rhythm_ref_conditioning={
            "prompt_content_units": torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long),
            "prompt_duration_obs": torch.full((1, 6), 3.0, dtype=torch.float32),
            "prompt_unit_mask": torch.ones((1, 6), dtype=torch.float32),
        },
        rhythm_apply_override=None,
        rhythm_runtime_overrides=None,
        rhythm_source_cache=None,
        rhythm_offline_source_cache=None,
        speech_state_fn=lambda x: torch.randn(x.size(0), x.size(1), 32),
    )
    ret["rhythm_v3_dur"] = torch.tensor(0.1)
    ret["rhythm_v3_base"] = torch.tensor(0.05)
    ret["rhythm_v3_bias"] = torch.tensor(0.07)
    ret["rhythm_v3_op"] = torch.tensor(0.2)
    ret["rhythm_v3_zero"] = torch.tensor(0.25)
    ret["rhythm_v3_ortho"] = torch.tensor(0.28)
    ret["rhythm_v3_pref"] = torch.tensor(0.3)
    ret["rhythm_v3_cons"] = torch.tensor(0.35)
    ret["rhythm_v3_stream"] = torch.tensor(0.37)
    ret["rhythm_total"] = torch.tensor(1.47)
    return ret


def test_rhythm_v3_metric_sections_cover_committed_duration_path_only():
    output = _run_adapter()
    commit_mask = output["rhythm_execution"].commit_mask > 0.5
    visible_mask = output["rhythm_unit_batch"].unit_mask > 0.5
    assert torch.any(visible_mask & (~commit_mask))
    target = output["speech_duration_exec"].detach().clone()
    target[visible_mask & (~commit_mask)] = target[visible_mask & (~commit_mask)] + 100.0
    sample = {"unit_duration_tgt": target}
    sections = build_rhythm_metric_sections(output, sample=sample)
    metrics = build_rhythm_metric_dict(output, sample=sample)
    assert "plan_surfaces" in sections
    assert "runtime_state" in sections
    for key in (
        "rhythm_metric_committed_units_mean",
        "rhythm_metric_commit_ratio_mean",
        "rhythm_metric_unit_duration_mean",
        "rhythm_metric_logstretch_abs_mean",
        "rhythm_metric_basis_activation_abs_mean",
        "rhythm_metric_frame_plan_present",
        "rhythm_metric_global_rate_mean",
        "rhythm_metric_global_stretch_mean",
        "rhythm_metric_progress_response_abs_mean",
        "rhythm_metric_progress_profile_abs_mean",
        "rhythm_metric_local_response_abs_mean",
        "rhythm_metric_operator_coeff_abs_mean",
        "rhythm_metric_operator_support_mean",
        "rhythm_metric_short_prompt_fallback_rate",
        "rhythm_metric_operator_coeff_norm_mean",
        "rhythm_metric_operator_condition_number_mean",
        "rhythm_metric_source_residual_gain",
        "rhythm_metric_commit_frontier_mean",
        "rhythm_metric_rounding_residual_mean",
        "rhythm_metric_rounding_residual_abs_mean",
        "rhythm_metric_exec_speech_l1",
        "rhythm_metric_prefix_drift_l1",
        "rhythm_metric_rhythm_total",
        "rhythm_metric_rhythm_v3_base",
        "rhythm_metric_rhythm_v3_dur",
        "rhythm_metric_rhythm_v3_bias",
        "rhythm_metric_rhythm_v3_summary",
        "rhythm_metric_rhythm_v3_op",
        "rhythm_metric_rhythm_v3_zero",
        "rhythm_metric_rhythm_v3_ortho",
        "rhythm_metric_rhythm_v3_pref",
        "rhythm_metric_rhythm_v3_cons",
        "rhythm_metric_rhythm_v3_stream",
    ):
        assert key in metrics
        assert torch.isfinite(metrics[key]).all()
    assert torch.allclose(metrics["rhythm_metric_exec_speech_l1"], torch.tensor(0.0))
    assert torch.allclose(metrics["rhythm_metric_prefix_drift_l1"], torch.tensor(0.0))
    for key in (
        "rhythm_metric_pause_duration_mean",
        "rhythm_metric_pause_event_rate",
        "rhythm_metric_phrase_state_norm",
        "rhythm_metric_rhythm_v3_break",
        "rhythm_metric_L_exec_speech",
        "rhythm_metric_L_exec_stretch",
        "rhythm_metric_L_prefix_state",
        "rhythm_metric_L_rhythm_exec",
        "rhythm_metric_L_stream_state",
    ):
        assert key not in metrics


def test_rhythm_v3_metric_path_works_when_version_flag_is_missing():
    output = _run_adapter()
    output.pop("rhythm_version", None)
    sample = {"unit_duration_tgt": output["speech_duration_exec"].detach()}
    metrics = build_rhythm_metric_dict(output, sample=sample)
    assert "rhythm_metric_basis_activation_abs_mean" in metrics
    assert "rhythm_metric_exec_speech_l1" in metrics
    assert "rhythm_metric_pause_duration_mean" not in metrics


def test_rhythm_v3_metrics_ignore_silence_units_in_speech_supervision():
    output = _run_adapter()
    commit_mask = output["rhythm_execution"].commit_mask > 0.5
    committed = commit_mask.nonzero(as_tuple=False)
    assert committed.numel() > 0
    silence_mask = output["rhythm_unit_batch"].unit_mask.new_zeros(output["rhythm_unit_batch"].unit_mask.shape)
    silence_mask[committed[0, 0], committed[0, 1]] = 1.0
    output["rhythm_unit_batch"].source_silence_mask = silence_mask
    target = output["speech_duration_exec"].detach().clone()
    target[committed[0, 0], committed[0, 1]] = target[committed[0, 0], committed[0, 1]] + 100.0
    metrics = build_rhythm_metric_dict(output, sample={"unit_duration_tgt": target})
    assert torch.allclose(metrics["rhythm_metric_exec_speech_l1"], torch.tensor(0.0))
    assert torch.allclose(metrics["rhythm_metric_prefix_drift_l1"], torch.tensor(0.0))


def test_rhythm_v3_metrics_report_silence_follow_statistics():
    output = _run_adapter()
    execution = output["rhythm_execution"]
    commit_mask = execution.commit_mask.float()
    committed = (commit_mask > 0.5).nonzero(as_tuple=False)
    assert committed.numel() > 0
    row = int(committed[0, 0].item())
    col = int(committed[0, 1].item())
    silence_mask = torch.zeros_like(commit_mask)
    silence_mask[row, col] = 1.0
    output["rhythm_unit_batch"].source_silence_mask = silence_mask
    target = output["speech_duration_exec"].detach().clone()
    target[row, col] = target[row, col] + 2.0
    metrics = build_rhythm_metric_dict(output, sample={"unit_duration_tgt": target})
    expected_logstretch = execution.unit_logstretch[row, col].abs()
    expected_ratio = (
        output["speech_duration_exec"][row, col]
        / output["rhythm_unit_batch"].source_duration_obs[row, col].clamp_min(1.0e-6)
    )
    assert torch.allclose(metrics["rhythm_metric_silence_logstretch_abs_mean"], expected_logstretch)
    assert torch.allclose(metrics["rhythm_metric_silence_exec_ratio_mean"], expected_ratio)
    assert torch.allclose(metrics["rhythm_metric_silence_prefix_drift_l1"], torch.tensor(2.0))


def test_rhythm_v3_metrics_source_rate_seq_mean_uses_speech_mask():
    output = _run_adapter()
    execution = output["rhythm_execution"]
    commit_mask = execution.commit_mask.float()
    silence_mask = torch.zeros_like(commit_mask)
    silence_mask[0, 1] = 1.0
    unit_batch = output["rhythm_unit_batch"]
    unit_batch.source_silence_mask = silence_mask
    seq_len = execution.unit_logstretch.size(1)
    source_rate_seq = torch.linspace(0.1, 0.2, seq_len, dtype=torch.float32).unsqueeze(0)
    execution.source_rate_seq = source_rate_seq
    speech_mask = commit_mask * (1.0 - silence_mask)
    sample = {"unit_duration_tgt": output["speech_duration_exec"].detach()}
    metrics = build_rhythm_metric_dict(output, sample=sample)
    expected = (source_rate_seq * speech_mask).sum(dim=1) / speech_mask.sum(dim=1).clamp_min(1.0e-6)
    assert torch.allclose(metrics["rhythm_metric_source_rate_seq_mean"], expected)


def test_gate3_local_metrics_capture_speech_gain_and_purity():
    pred = torch.tensor([[0.30, 0.10, 0.0]], dtype=torch.float32)
    target = torch.tensor([[0.20, 0.20, 0.0]], dtype=torch.float32)
    speech_mask = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32)
    weight = torch.tensor([[1.0, 2.0, 0.0]], dtype=torch.float32)
    mae = speech_weighted_mae(pred, target, speech_mask, weight)
    expected = (0.10 * 1.0 + 0.10 * 2.0) / 3.0
    assert torch.allclose(mae, torch.tensor(expected, dtype=torch.float32))

    residual_pred = torch.tensor([[0.08, -0.04, 0.0]], dtype=torch.float32)
    residual_target = torch.tensor([[0.10, -0.05, 0.0]], dtype=torch.float32)
    stats = residual_target_stats(residual_pred, residual_target, speech_mask)
    assert stats["count"] == pytest.approx(2.0)
    assert stats["spearman"] == pytest.approx(1.0)

    bias_share = residual_bias_share(residual_pred, speech_mask, torch.tensor([[0.20]], dtype=torch.float32))
    assert torch.isfinite(bias_share)
    assert float(bias_share.item()) < 0.30


def test_gate3_local_silence_delta_share_stays_zero_when_delta_is_speech_only():
    learned = torch.tensor([[0.18, 0.12, 0.0]], dtype=torch.float32)
    coarse = torch.tensor([[0.10, 0.10, 0.0]], dtype=torch.float32)
    speech_mask = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32)
    silence_mask = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    share = local_silence_delta_share(learned, coarse, speech_mask, silence_mask)
    assert torch.allclose(share, torch.tensor(0.0))


def test_silence_metrics_return_nan_when_speech_denominator_is_too_small():
    delta = torch.tensor([[0.0, 0.0, 1.0e-5]], dtype=torch.float32)
    speech_mask = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32)
    silence_mask = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)

    leak = silence_leakage(delta, speech_mask, silence_mask)
    share = local_silence_delta_share(delta, torch.zeros_like(delta), speech_mask, silence_mask)

    assert torch.isnan(leak)
    assert torch.isnan(share)


def test_duration_v3_loss_targets_clip_silence_only_coarse():
    unit_mask = torch.ones((1, 3))
    sample = {"unit_duration_tgt": torch.tensor([[5.0, 3.0, 25.0]])}
    unit_batch = SimpleNamespace(
        unit_mask=unit_mask,
        unit_anchor_base=torch.ones((1, 3)),
        source_silence_mask=torch.tensor([[0.0, 0.0, 1.0]]),
    )
    execution = SimpleNamespace(commit_mask=unit_mask.clone())
    output = {
        "rhythm_execution": execution,
        "rhythm_unit_batch": unit_batch,
        "rhythm_ref_conditioning": SimpleNamespace(global_rate=torch.tensor([[0.2]])),
        "rhythm_v3_source_rate_init": torch.tensor([0.1]),
    }
    config = DurationV3TargetBuildConfig(
        lambda_dur=1.0,
        lambda_op=0.0,
        lambda_pref=0.0,
        silence_coarse_weight=0.75,
        silence_logstretch_max=0.25,
    )
    targets = build_duration_v3_loss_targets(sample=sample, output=output, config=config)
    assert targets is not None
    full_logstretch = (
        torch.log(sample["unit_duration_tgt"])
        - torch.log(targets.prediction_anchor.float().clamp_min(1.0e-6))
    )
    silence_tau = _build_duration_v3_silence_tau(
        prediction_anchor=targets.prediction_anchor.float(),
        committed_silence_mask=targets.committed_silence_mask.float(),
        sep_hint=None,
        boundary_cue=None,
        max_silence_logstretch=float(config.silence_logstretch_max),
        short_gap_scale=float(config.silence_short_gap_scale),
    )
    silence_idx = 2
    expected_coarse_duration = targets.prediction_anchor[0, silence_idx] * torch.exp(targets.coarse_logstretch_tgt[0, silence_idx])
    assert torch.allclose(targets.coarse_duration_tgt[0, silence_idx], expected_coarse_duration)
    assert torch.allclose(
        targets.prefix_duration_tgt[0, silence_idx],
        expected_coarse_duration,
    )
    assert abs(float(targets.coarse_logstretch_tgt[0, silence_idx].item())) <= float(silence_tau[0, silence_idx].item()) + 1.0e-6
    assert not torch.allclose(targets.coarse_logstretch_tgt[0, silence_idx], full_logstretch[0, silence_idx])
    assert torch.allclose(targets.local_residual_tgt[0, silence_idx], torch.tensor(0.0))


def test_rhythm_v3_global_only_metrics_report_zero_local_response():
    output = _run_adapter()
    output["rhythm_execution"].unit_logstretch = output["rhythm_ref_conditioning"].global_rate.expand_as(
        output["rhythm_execution"].unit_logstretch
    ) * output["rhythm_execution"].commit_mask
    output["rhythm_execution"].local_response = torch.zeros_like(output["rhythm_execution"].unit_logstretch)
    output["rhythm_execution"].progress_response = torch.zeros_like(output["rhythm_execution"].unit_logstretch)
    output["rhythm_v3_source_residual_gain"] = 0.0
    metrics = build_rhythm_metric_dict(
        output,
        sample={"unit_duration_tgt": output["speech_duration_exec"].detach()},
    )
    assert torch.allclose(metrics["rhythm_metric_local_response_abs_mean"], torch.tensor(0.0))
    assert torch.allclose(metrics["rhythm_metric_progress_response_abs_mean"], torch.tensor(0.0))


def test_rhythm_v3_progress_only_metrics_keep_local_response_zero():
    adapter = ConanDurationAdapter(
        {
            **_build_hparams(),
            "rhythm_v3_backbone": "global_only",
            "rhythm_v3_warp_mode": "progress",
            "rhythm_v3_allow_hybrid": False,
            "lambda_rhythm_op": 0.0,
            "lambda_rhythm_zero": 0.0,
            "lambda_rhythm_ortho": 0.0,
        },
        hidden_size=32,
        vocab_size=128,
    )
    content = torch.tensor([[1, 1, 2, 2, 3, 4, 4, 5]])
    ret = {}
    adapter(
        ret=ret,
        content=content,
        ref=None,
        target=None,
        f0=None,
        uv=None,
        infer=True,
        global_steps=0,
        content_embed=torch.randn(content.size(0), content.size(1), 32),
        tgt_nonpadding=torch.ones(content.size(0), content.size(1), 1),
        content_lengths=torch.full((content.size(0),), int(content.size(1)), dtype=torch.long),
        rhythm_state=None,
        rhythm_ref_conditioning={
            "global_rate": torch.tensor([[0.1]], dtype=torch.float32),
            "progress_profile": torch.tensor([[0.10, 0.20, 0.15, 0.05]], dtype=torch.float32),
            "operator_coeff": torch.full((1, 4), 9.0, dtype=torch.float32),
        },
        rhythm_apply_override=None,
        rhythm_runtime_overrides=None,
        rhythm_source_cache=None,
        rhythm_offline_source_cache=None,
        speech_state_fn=lambda x: torch.randn(x.size(0), x.size(1), 32),
    )
    metrics = build_rhythm_metric_dict(ret, sample={"unit_duration_tgt": ret["speech_duration_exec"].detach()})
    assert torch.allclose(metrics["rhythm_metric_local_response_abs_mean"], torch.tensor(0.0))
    assert float(metrics["rhythm_metric_progress_response_abs_mean"].item()) > 0.0
    assert float(metrics["rhythm_metric_progress_profile_abs_mean"].item()) > 0.0


def test_rhythm_v3_detector_only_metrics_report_detector_response():
    adapter = ConanDurationAdapter(
        {
            **_build_hparams(),
            "rhythm_v3_backbone": "global_only",
            "rhythm_v3_warp_mode": "detector",
            "rhythm_v3_allow_hybrid": False,
            "lambda_rhythm_op": 0.0,
            "lambda_rhythm_zero": 0.0,
            "lambda_rhythm_ortho": 0.0,
        },
        hidden_size=32,
        vocab_size=128,
    )
    content = torch.tensor([[1, 1, 2, 2, 3, 4, 4, 5]])
    ret = {}
    adapter(
        ret=ret,
        content=content,
        ref=None,
        target=None,
        f0=None,
        uv=None,
        infer=True,
        global_steps=0,
        content_embed=torch.randn(content.size(0), content.size(1), 32),
        tgt_nonpadding=torch.ones(content.size(0), content.size(1), 1),
        content_lengths=torch.full((content.size(0),), int(content.size(1)), dtype=torch.long),
        rhythm_state=None,
        rhythm_ref_conditioning={
            "global_rate": torch.tensor([[0.1]], dtype=torch.float32),
            "detector_coeff": torch.tensor([[0.10, 0.25, 0.05, -0.10]], dtype=torch.float32),
            "operator_coeff": torch.full((1, 4), 9.0, dtype=torch.float32),
        },
        rhythm_apply_override=None,
        rhythm_runtime_overrides=None,
        rhythm_source_cache=None,
        rhythm_offline_source_cache=None,
        speech_state_fn=lambda x: torch.randn(x.size(0), x.size(1), 32),
    )
    metrics = build_rhythm_metric_dict(ret, sample={"unit_duration_tgt": ret["speech_duration_exec"].detach()})
    assert torch.allclose(metrics["rhythm_metric_local_response_abs_mean"], torch.tensor(0.0))
    assert float(metrics["rhythm_metric_detector_response_abs_mean"].item()) > 0.0
    assert float(metrics["rhythm_metric_detector_coeff_abs_mean"].item()) > 0.0


def test_tempo_explainability_uses_tie_aware_spearman():
    metrics = tempo_explainability(
        torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32),
        torch.tensor([0.0, 0.0, 2.0, 2.0], dtype=torch.float32),
    )
    assert metrics["count"] == 4.0
    assert abs(metrics["spearman"] - 1.0) < 1.0e-6
    assert abs(metrics["robust_slope"] - 2.0) < 1.0e-6
    assert abs(metrics["r2_like"] - 1.0) < 1.0e-6


def test_tempo_explainability_accepts_pandas_series_inputs():
    metrics = tempo_explainability(
        pd.Series([0.0, 0.5, 1.0], dtype="float32"),
        pd.Series([0.0, 1.0, 2.0], dtype="float32"),
    )
    assert metrics["count"] == 3.0
    assert abs(metrics["spearman"] - 1.0) < 1.0e-6
    assert abs(metrics["robust_slope"] - 2.0) < 1.0e-6
    assert abs(metrics["r2_like"] - 1.0) < 1.0e-6


def test_tempo_monotonicity_supports_margin_threshold():
    slow = torch.tensor([1.0, 1.0], dtype=torch.float32)
    mid = torch.tensor([1.05, 1.3], dtype=torch.float32)
    fast = torch.tensor([1.2, 1.7], dtype=torch.float32)
    strict = tempo_monotonicity(slow, mid, fast)
    margin = tempo_monotonicity(slow, mid, fast, margin=0.1)
    assert torch.allclose(strict, torch.tensor(1.0))
    assert torch.allclose(margin, torch.tensor(0.5))


def test_tempo_monotonicity_supports_direction_for_duration_like_metrics():
    slow = torch.tensor([3.0, 3.0, 3.0], dtype=torch.float32)
    mid = torch.tensor([2.0, 2.7, 2.8], dtype=torch.float32)
    fast = torch.tensor([1.0, 2.6, 3.1], dtype=torch.float32)

    increasing = tempo_monotonicity(slow, mid, fast, increasing=True)
    decreasing = tempo_monotonicity(slow, mid, fast, increasing=False)
    decreasing_margin = tempo_monotonicity(slow, mid, fast, margin=0.2, increasing=False)

    assert torch.allclose(increasing, torch.tensor(0.0))
    assert torch.allclose(decreasing, torch.tensor(2.0 / 3.0))
    assert torch.allclose(decreasing_margin, torch.tensor(1.0 / 3.0))


def test_monotonic_triplet_table_supports_direction():
    sample_ids = ["a", "b", "c"]
    slow = torch.tensor([1.0, 3.0, 3.0], dtype=torch.float32)
    mid = torch.tensor([2.0, 2.0, 2.8], dtype=torch.float32)
    fast = torch.tensor([3.0, 1.0, 3.1], dtype=torch.float32)

    inc = monotonic_triplet_table(sample_ids, slow, mid, fast, increasing=True)
    dec = monotonic_triplet_table(sample_ids, slow, mid, fast, increasing=False)

    assert inc["sample_id"] == sample_ids
    assert inc["mono_ok"] == [1.0, 0.0, 0.0]
    assert dec["mono_ok"] == [0.0, 1.0, 0.0]
    assert inc["valid"] == [1.0, 1.0, 1.0]


def test_tempo_tie_rate_counts_near_ties_without_marking_all_invalid():
    slow = torch.tensor([1.0, 1.0, float("nan")], dtype=torch.float32)
    mid = torch.tensor([1.0, 1.2, 1.5], dtype=torch.float32)
    fast = torch.tensor([1.4, 1.2 + 5.0e-5, 1.8], dtype=torch.float32)
    tie_rate = tempo_tie_rate(slow, mid, fast, atol=1.0e-4)
    assert torch.allclose(tie_rate, torch.tensor(1.0))


def test_monotonic_triplet_table_supports_margin_for_decreasing_duration_like_metrics():
    sample_ids = ["a", "b", "c"]
    slow = torch.tensor([3.0, 3.0, 3.0], dtype=torch.float32)
    mid = torch.tensor([2.9, 2.7, 2.0], dtype=torch.float32)
    fast = torch.tensor([2.8, 2.6, 1.0], dtype=torch.float32)

    table = monotonic_triplet_table(
        sample_ids,
        slow,
        mid,
        fast,
        increasing=False,
        margin=0.2,
    )

    assert table["mono_ok_strict"] == [1.0, 1.0, 1.0]
    assert table["mono_ok_margin"] == [0.0, 0.0, 1.0]


def test_monotonic_triplet_table_reports_ties_and_validity():
    sample_ids = ["a", "b", "c"]
    slow = torch.tensor([1.0, 1.0, float("nan")], dtype=torch.float32)
    mid = torch.tensor([1.0, 1.2, 1.5], dtype=torch.float32)
    fast = torch.tensor([1.4, 1.20005, 1.8], dtype=torch.float32)

    table = monotonic_triplet_table(
        sample_ids,
        slow,
        mid,
        fast,
        increasing=True,
        tie_atol=1.0e-4,
    )

    assert table["valid"] == [1.0, 1.0, 0.0]
    assert table["tie_sm"] == [1.0, 0.0, 0.0]
    assert table["tie_mf"] == [0.0, 1.0, 0.0]
    assert table["tie_any"] == [1.0, 1.0, 0.0]


def test_tempo_monotonicity_returns_nan_when_no_valid_triplets():
    slow = torch.tensor([float("nan")], dtype=torch.float32)
    mid = torch.tensor([1.0], dtype=torch.float32)
    fast = torch.tensor([2.0], dtype=torch.float32)

    value = tempo_monotonicity(slow, mid, fast)

    assert torch.isnan(value)


def test_tempo_tie_rate_respects_atol_boundary():
    slow = torch.tensor([1.0], dtype=torch.float32)
    mid = torch.tensor([1.0002], dtype=torch.float32)
    fast = torch.tensor([1.5], dtype=torch.float32)

    tie_rate = tempo_tie_rate(slow, mid, fast, atol=1.0e-4)

    assert torch.allclose(tie_rate, torch.tensor(0.0))
