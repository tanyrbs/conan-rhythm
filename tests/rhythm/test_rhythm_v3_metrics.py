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
    ret["rhythm_v3_dur"] = torch.tensor(0.1)
    ret["rhythm_v3_mem"] = torch.tensor(0.2)
    ret["rhythm_v3_pref"] = torch.tensor(0.3)
    ret["rhythm_v3_anti"] = torch.tensor(0.4)
    return ret


def test_rhythm_v3_metric_sections_cover_duration_path():
    output = _run_adapter()
    sample = {"rhythm_speech_exec_tgt": output["speech_duration_exec"].detach() + 0.1}
    sections = build_rhythm_metric_sections(output, sample=sample)
    metrics = build_rhythm_metric_dict(output, sample=sample)
    assert "plan_surfaces" in sections
    for key in (
        "rhythm_metric_exec_total_mean",
        "rhythm_metric_unit_duration_mean",
        "rhythm_metric_logstretch_abs_mean",
        "rhythm_metric_role_entropy",
        "rhythm_metric_frame_plan_present",
        "rhythm_metric_global_rate_mean",
        "rhythm_metric_role_coverage_mean",
        "rhythm_metric_exec_speech_l1",
        "rhythm_metric_prefix_drift_l1",
        "rhythm_metric_rhythm_v3_dur",
    ):
        assert key in metrics
        assert torch.isfinite(metrics[key]).all()


def test_rhythm_v3_metric_path_works_when_version_flag_is_missing():
    output = _run_adapter()
    output.pop("rhythm_version", None)
    sample = {"rhythm_speech_exec_tgt": output["speech_duration_exec"].detach()}
    metrics = build_rhythm_metric_dict(output, sample=sample)
    assert "rhythm_metric_exec_total_mean" in metrics
    assert "rhythm_metric_role_entropy" in metrics
