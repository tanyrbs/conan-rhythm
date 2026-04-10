from __future__ import annotations

import pytest
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
        "rhythm_response_rank": 4,
        "rhythm_response_window_left": 2,
        "rhythm_response_window_right": 0,
        "rhythm_trace_bins": 8,
        "rhythm_ref_coverage_floor": 0.05,
        "rhythm_max_logstretch": 0.8,
        "rhythm_anti_pos_bins": 4,
        "rhythm_anti_pos_grl_scale": 1.0,
        "rhythm_streaming_mode": "strict",
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
    ref_memory = ret["rhythm_ref_conditioning"]
    assert ret["rhythm_version"] == "v3"
    assert ref_memory.operator_coeff.shape == (1, 4)
    assert ref_memory.global_rate.shape == (1, 1)
    assert ret["speech_duration_exec"].shape[1] == ret["rhythm_unit_batch"].content_units.shape[1]
    assert ret["rhythm_execution"].planner is None
    assert ret["rhythm_frame_plan"] is not None
    assert torch.all(ret["speech_duration_exec"] >= 0.0)
    assert torch.all(ret["rhythm_execution"].commit_mask >= 0.0)
    assert torch.allclose(ret["speech_duration_exec"], ret["rhythm_state_next"].cached_duration_exec)
    prompt_rel = getattr(getattr(ref_memory, "prompt", None), "prompt_rel_stretch", None)
    prompt_mask = getattr(getattr(ref_memory, "prompt", None), "prompt_mask", None)
    if isinstance(prompt_rel, torch.Tensor):
        if isinstance(prompt_mask, torch.Tensor):
            visible = prompt_mask > 0.1
            residual_center = []
            for batch_idx in range(prompt_rel.size(0)):
                if bool(visible[batch_idx].any().item()):
                    residual_center.append(prompt_rel[batch_idx][visible[batch_idx]].median())
                else:
                    residual_center.append(prompt_rel.new_zeros(()))
            residual_center = torch.stack(residual_center, dim=0)
        else:
            residual_center = prompt_rel.median(dim=1).values
        assert torch.allclose(residual_center, torch.zeros_like(residual_center), atol=1e-5)
    committed = ret["rhythm_execution"].commit_mask > 0.5
    assert torch.allclose(
        ret["speech_duration_exec"][committed],
        torch.round(ret["speech_duration_exec"][committed]),
    )
    assert "global_rate" not in ret
    assert "role_value" not in ret
    assert "blank_duration_exec" not in ret
    assert "rhythm_render_slot_index" not in ret
    assert "rhythm_render_phase_features" not in ret


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


def test_rhythm_v3_freezes_committed_prefix_across_chunks_and_updates_rounding_state():
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
    assert torch.allclose(first["rhythm_state_next"].cached_duration_exec[:, :frontier], first["speech_duration_exec"][:, :frontier])
    assert torch.allclose(second["rhythm_state_next"].cached_duration_exec[:, :frontier], second["speech_duration_exec"][:, :frontier])
    assert torch.isfinite(second["rhythm_state_next"].rounding_residual).all()


def test_rhythm_v3_commits_only_sealed_units():
    adapter = ConanDurationAdapter(_build_hparams(), hidden_size=32, vocab_size=128)
    ret = _run_adapter(
        adapter,
        content=torch.tensor([[1, 1, 2, 2, 3]]),
        ref=torch.randn(1, 12, 80),
    )
    frontier = int(ret["commit_frontier"][0].item())
    total_units = int(ret["rhythm_unit_batch"].unit_mask.sum().item())
    assert frontier < total_units
    assert torch.allclose(ret["speech_duration_exec"][:, frontier:], torch.zeros_like(ret["speech_duration_exec"][:, frontier:]))


def test_rhythm_v3_handles_single_frame_reference_with_lengths():
    adapter = ConanDurationAdapter(_build_hparams(), hidden_size=32, vocab_size=128)
    ret = _run_adapter(
        adapter,
        content=torch.tensor([[1, 1, 2, 2]]),
        ref=torch.randn(1, 1, 80),
        ref_lengths=torch.tensor([1]),
    )
    ref_memory = ret["rhythm_ref_conditioning"]
    for value in (ref_memory.global_rate, ref_memory.operator_coeff, ret["speech_duration_exec"]):
        assert torch.isfinite(value).all()


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


def test_rhythm_v3_skips_render_for_empty_chunk_even_when_requested():
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
        ref_lengths=None,
        rhythm_state=None,
        rhythm_ref_conditioning=first["rhythm_ref_conditioning"],
        rhythm_apply_override=True,
        rhythm_runtime_overrides=None,
        rhythm_source_cache=None,
        rhythm_offline_source_cache=None,
        speech_state_fn=lambda x: torch.zeros((x.size(0), x.size(1), 32)),
    )
    assert ret["speech_duration_exec"].shape == (1, 0)
    assert ret["rhythm_apply_render"] == 0.0
    assert ret["rhythm_render_skipped_empty"] == 1.0


def test_rhythm_v3_rejects_runtime_state_batch_mismatch():
    adapter = ConanDurationAdapter(_build_hparams(), hidden_size=32, vocab_size=128)
    ref = torch.randn(1, 24, 80)
    first = _run_adapter(adapter, content=torch.tensor([[1, 1, 2, 2]]), ref=ref)
    with pytest.raises(ValueError, match="DurationRuntimeState batch mismatch"):
        _run_adapter(
            adapter,
            content=torch.tensor([[1, 1, 2, 2], [3, 3, 4, 4]]),
            ref=None,
            state=first["rhythm_state_next"],
            ref_conditioning=first["rhythm_ref_conditioning"],
        )


def test_rhythm_v3_rejects_invalid_ref_conditioning_shape_early():
    adapter = ConanDurationAdapter(_build_hparams(), hidden_size=32, vocab_size=128)
    with pytest.raises(ValueError, match="StructuredDurationOperatorMemory.operator_coeff batch mismatch"):
        _run_adapter(
            adapter,
            content=torch.tensor([[1, 1, 2, 2]]),
            ref=None,
            ref_conditioning={
                "global_rate": torch.zeros((1, 1)),
                "operator_coeff": torch.zeros((2, 4)),
            },
        )


def test_rhythm_v3_rejects_invalid_precomputed_cache_shape_early():
    adapter = ConanDurationAdapter(_build_hparams(), hidden_size=32, vocab_size=128)
    ret = {}
    with pytest.raises(ValueError, match="Precomputed source cache shape mismatch"):
        adapter(
            ret=ret,
            content=torch.tensor([[1, 1, 2, 2]]),
            ref=torch.randn(1, 4, 80),
            target=None,
            f0=None,
            uv=None,
            infer=True,
            global_steps=0,
            content_embed=torch.randn(1, 4, 32),
            tgt_nonpadding=torch.ones(1, 4, 1),
            content_lengths=torch.tensor([4]),
            ref_lengths=torch.tensor([4]),
            rhythm_state=None,
            rhythm_ref_conditioning=None,
            rhythm_apply_override=False,
            rhythm_runtime_overrides=None,
            rhythm_source_cache={
                "content_units": torch.tensor([[1, 2, 3]]),
                "source_duration_obs": torch.tensor([[1, 1, 1, 1]]),
            },
            rhythm_offline_source_cache=None,
            speech_state_fn=lambda x: torch.randn(x.size(0), x.size(1), 32),
        )


def test_rhythm_v3_handles_zero_length_reference_without_lengths():
    adapter = ConanDurationAdapter(_build_hparams(), hidden_size=32, vocab_size=128)
    ret = _run_adapter(
        adapter,
        content=torch.tensor([[1, 1, 2, 2]]),
        ref=torch.zeros((1, 0, 80)),
        ref_lengths=None,
    )
    ref_memory = ret["rhythm_ref_conditioning"]
    for value in (ref_memory.global_rate, ref_memory.operator_coeff, ret["speech_duration_exec"]):
        assert torch.isfinite(value).all()
