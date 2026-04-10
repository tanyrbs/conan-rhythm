from __future__ import annotations

import torch

from modules.Conan.rhythm_v3.runtime_adapter import ConanDurationAdapter
from tasks.Conan.rhythm.metrics import build_rhythm_metric_dict, build_rhythm_metric_sections


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
        "rhythm_max_logstretch": 0.8,
        "rhythm_streaming_mode": "strict",
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
        rhythm_ref_conditioning=None,
        rhythm_apply_override=None,
        rhythm_runtime_overrides=None,
        rhythm_source_cache=None,
        rhythm_offline_source_cache=None,
        speech_state_fn=lambda x: torch.randn(x.size(0), x.size(1), 32),
    )
    ret["rhythm_v3_dur"] = torch.tensor(0.1)
    ret["rhythm_v3_op"] = torch.tensor(0.2)
    ret["rhythm_v3_zero"] = torch.tensor(0.25)
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
        "rhythm_metric_operator_coeff_abs_mean",
        "rhythm_metric_commit_frontier_mean",
        "rhythm_metric_rounding_residual_mean",
        "rhythm_metric_rounding_residual_abs_mean",
        "rhythm_metric_exec_speech_l1",
        "rhythm_metric_prefix_drift_l1",
        "rhythm_metric_rhythm_total",
        "rhythm_metric_rhythm_v3_dur",
        "rhythm_metric_rhythm_v3_op",
        "rhythm_metric_rhythm_v3_zero",
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
