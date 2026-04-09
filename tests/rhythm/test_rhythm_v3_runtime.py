from __future__ import annotations

import torch

from modules.Conan.rhythm_v3.runtime_adapter import ConanDurationAdapter


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


def _run_adapter(adapter, *, content, ref, state=None, ref_conditioning=None, ref_lengths=None):
    ret = {}
    hidden = 32
    adapter(
        ret=ret,
        content=content,
        ref=ref,
        target=None,
        f0=None,
        uv=None,
        infer=True,
        global_steps=0,
        content_embed=torch.randn(content.size(0), content.size(1), hidden),
        tgt_nonpadding=torch.ones(content.size(0), content.size(1), 1),
        content_lengths=torch.full((content.size(0),), int(content.size(1)), dtype=torch.long),
        ref_lengths=ref_lengths,
        rhythm_state=state,
        rhythm_ref_conditioning=ref_conditioning,
        rhythm_apply_override=None,
        rhythm_runtime_overrides=None,
        rhythm_source_cache=None,
        rhythm_offline_source_cache=None,
        speech_state_fn=lambda x: torch.randn(x.size(0), x.size(1), hidden),
    )
    return ret


def test_rhythm_v3_adapter_emits_static_memory_runtime():
    adapter = ConanDurationAdapter(_build_hparams(), hidden_size=32, vocab_size=128)
    content = torch.tensor([[1, 1, 2, 2, 3, 4, 4, 5]])
    ref = torch.randn(1, 24, 80)
    ret = _run_adapter(adapter, content=content, ref=ref)
    assert ret["rhythm_version"] == "v3"
    assert ret["role_value"].shape == (1, 4)
    assert ret["role_coverage"].shape == (1, 4)
    assert ret["speech_duration_exec"].shape[1] == ret["rhythm_unit_batch"].content_units.shape[1]
    assert ret["rhythm_execution"].planner is None
    assert ret["rhythm_frame_plan"] is not None
    assert torch.all(ret["speech_duration_exec"] >= 0.0)


def test_rhythm_v3_reference_memory_is_reusable_across_chunks():
    adapter = ConanDurationAdapter(_build_hparams(), hidden_size=32, vocab_size=128)
    ref = torch.randn(1, 24, 80)
    first = _run_adapter(adapter, content=torch.tensor([[1, 1, 2, 2]]), ref=ref)
    second = _run_adapter(
        adapter,
        content=torch.tensor([[1, 1, 2, 2, 3, 3, 4]]),
        ref=None,
        state=first["rhythm_state_next"],
        ref_conditioning=first["rhythm_ref_conditioning"],
    )
    assert second["rhythm_version"] == "v3"
    assert second["rhythm_ref_conditioning"] is first["rhythm_ref_conditioning"]
    assert int(second["commit_frontier"][0].item()) >= int(first["commit_frontier"][0].item())


def test_rhythm_v3_freezes_committed_prefix_across_chunks():
    adapter = ConanDurationAdapter(_build_hparams(), hidden_size=32, vocab_size=128)
    ref = torch.randn(1, 24, 80)
    first = _run_adapter(adapter, content=torch.tensor([[1, 1, 2, 2, 3, 3]]), ref=ref)
    second = _run_adapter(
        adapter,
        content=torch.tensor([[1, 1, 2, 2, 3, 3, 4, 4, 5]]),
        ref=None,
        state=first["rhythm_state_next"],
        ref_conditioning=first["rhythm_ref_conditioning"],
    )
    frontier = int(first["commit_frontier"][0].item())
    assert frontier > 0
    assert torch.allclose(
        second["speech_duration_exec"][:, :frontier],
        first["speech_duration_exec"][:, :frontier],
    )
    assert torch.isfinite(second["rhythm_state_next"].cumulative_pred_frames).all()
    assert torch.isfinite(second["rhythm_state_next"].clock_delta).all()
    assert torch.isfinite(second["rhythm_state_next"].backlog).all()


def test_rhythm_v3_handles_single_frame_reference_with_lengths():
    adapter = ConanDurationAdapter(_build_hparams(), hidden_size=32, vocab_size=128)
    ret = _run_adapter(
        adapter,
        content=torch.tensor([[1, 1, 2, 2]]),
        ref=torch.randn(1, 1, 80),
        ref_lengths=torch.tensor([1]),
    )
    for key in ("global_rate", "role_value", "role_coverage", "speech_duration_exec"):
        assert torch.isfinite(ret[key]).all()


def test_rhythm_v3_handles_empty_source_chunk_without_crashing():
    adapter = ConanDurationAdapter(_build_hparams(), hidden_size=32, vocab_size=128)
    first = _run_adapter(
        adapter,
        content=torch.tensor([[1, 1, 2, 2]]),
        ref=torch.randn(1, 4, 80),
        ref_lengths=torch.tensor([4]),
    )
    ret = {}
    adapter(
        ret=ret,
        content=torch.zeros((1, 0), dtype=torch.long),
        ref=None,
        target=None,
        f0=None,
        uv=None,
        infer=True,
        global_steps=0,
        content_embed=torch.zeros((1, 0, 32)),
        tgt_nonpadding=torch.zeros((1, 0, 1)),
        content_lengths=torch.tensor([0]),
        ref_lengths=torch.tensor([4]),
        rhythm_state=None,
        rhythm_ref_conditioning=first["rhythm_ref_conditioning"],
        rhythm_apply_override=False,
        rhythm_runtime_overrides=None,
        rhythm_source_cache=None,
        rhythm_offline_source_cache=None,
        speech_state_fn=lambda x: torch.zeros((x.size(0), x.size(1), 32)),
    )
    assert ret["speech_duration_exec"].shape == (1, 0)
    assert ret["rhythm_apply_render"] == 0.0
