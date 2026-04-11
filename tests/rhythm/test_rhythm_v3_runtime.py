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
        "rhythm_ref_coverage_floor": 0.05,
        "rhythm_max_logstretch": 0.8,
        "rhythm_streaming_mode": "strict",
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
            "rhythm_v3_source_residual_gain": 0.0,
        }
    )
    return hparams


def _run_adapter(adapter, *, content, ref, state=None, ref_conditioning=None, ref_lengths=None, auto_prompt_from_ref=True):
    if auto_prompt_from_ref and ref_conditioning is None and ref is not None:
        ref_conditioning = _build_prompt_conditioning()
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


def _build_prompt_conditioning(*, prompt_units: int = 6):
    content = torch.arange(1, prompt_units + 1, dtype=torch.long).unsqueeze(0)
    duration = torch.full((1, prompt_units), 3.0, dtype=torch.float32)
    mask = torch.ones((1, prompt_units), dtype=torch.float32)
    return {
        "prompt_content_units": content,
        "prompt_duration_obs": duration,
        "prompt_unit_mask": mask,
    }


def test_rhythm_v3_adapter_emits_prompt_conditioned_operator_runtime():
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
    assert ret["rhythm_v3_runtime_mode"] == "operator_progress"
    assert ret["rhythm_execution"].progress_response is not None
    assert torch.isfinite(ret["rhythm_execution"].progress_response).all()
    assert ret["rhythm_execution"].local_response is not None
    assert torch.isfinite(ret["rhythm_execution"].local_response).all()
    assert torch.allclose(ret["speech_duration_exec"], ret["rhythm_state_next"].cached_duration_exec)
    prompt_rel = getattr(getattr(ref_memory, "prompt", None), "prompt_random_target", None)
    prompt_mask = getattr(getattr(ref_memory, "prompt", None), "prompt_mask", None)
    if isinstance(prompt_rel, torch.Tensor):
        assert torch.isfinite(prompt_rel).all()
    prompt_cv_fit = getattr(getattr(ref_memory, "prompt", None), "prompt_operator_cv_fit", None)
    if isinstance(prompt_cv_fit, torch.Tensor):
        assert torch.isfinite(prompt_cv_fit).all()
    prompt_eval_mask = getattr(getattr(ref_memory, "prompt", None), "prompt_eval_mask", None)
    if isinstance(prompt_eval_mask, torch.Tensor):
        assert prompt_eval_mask.shape == prompt_mask.shape
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


def test_rhythm_v3_prompt_summary_runtime_uses_static_prompt_memory_and_source_anchor():
    adapter = ConanDurationAdapter(_build_prompt_summary_hparams(), hidden_size=32, vocab_size=128)
    content = torch.tensor([[1, 1, 2, 2, 3, 4, 4, 5]], dtype=torch.long)
    ret = _run_adapter(adapter, content=content, ref=None, ref_conditioning=_build_prompt_conditioning())
    ref_memory = ret["rhythm_ref_conditioning"]
    execution = ret["rhythm_execution"]
    assert ret["rhythm_v3_runtime_mode"] == "prompt_summary"
    assert ref_memory.role_value is not None
    assert ref_memory.role_var is not None
    assert ref_memory.role_coverage is not None
    assert ref_memory.prompt_role_attn is not None
    assert execution.role_conf_unit is not None
    assert torch.isfinite(execution.role_conf_unit).all()
    assert execution.progress_response is None
    assert execution.detector_response is None
    assert execution.local_response is not None
    assert ret["speech_duration_exec"].shape[1] == ret["rhythm_unit_batch"].content_units.shape[1]
    assert torch.isfinite(ret["speech_duration_exec"]).all()


def test_rhythm_v3_baseline_pretrain_can_run_without_reference_prompt():
    hparams = {
        **_build_hparams(),
        "rhythm_v3_baseline_train_mode": "pretrain",
        "lambda_rhythm_base": 1.0,
    }
    adapter = ConanDurationAdapter(hparams, hidden_size=32, vocab_size=128)
    ret = {}
    content = torch.tensor([[1, 1, 2, 2, 3, 4]], dtype=torch.long)
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
        rhythm_ref_conditioning=None,
        rhythm_apply_override=None,
        rhythm_runtime_overrides=None,
        rhythm_source_cache=None,
        rhythm_offline_source_cache=None,
        speech_state_fn=lambda x: torch.randn(x.size(0), x.size(1), 32),
    )
    assert ret["rhythm_version"] == "v3"
    assert ret["rhythm_v3_baseline_train_mode"] == "pretrain"
    ref_memory = ret["rhythm_ref_conditioning"]
    assert torch.allclose(ref_memory.global_rate, torch.zeros_like(ref_memory.global_rate))


def test_rhythm_v3_accepts_explicit_prompt_units_without_cached_anchor_base():
    adapter = ConanDurationAdapter(_build_hparams(), hidden_size=32, vocab_size=128)
    content = torch.tensor([[1, 1, 2, 2, 3, 4]], dtype=torch.long)
    ret = _run_adapter(
        adapter,
        content=content,
        ref=None,
        ref_conditioning={
            "prompt_content_units": torch.tensor([[1, 2, 3, 0]], dtype=torch.long),
            "prompt_duration_obs": torch.tensor([[3.0, 4.0, 2.0, 0.0]], dtype=torch.float32),
            "prompt_unit_mask": torch.tensor([[1.0, 1.0, 1.0, 0.0]], dtype=torch.float32),
        },
    )
    ref_memory = ret["rhythm_ref_conditioning"]
    assert ref_memory.operator_coeff.shape == (1, 4)
    assert ref_memory.prompt_log_base is not None
    assert torch.isfinite(ref_memory.prompt_log_base).all()


def test_rhythm_v3_prompt_conditioning_is_reusable_across_chunks():
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


def test_rhythm_v3_infer_rejects_mel_only_reference_without_prompt_units():
    adapter = ConanDurationAdapter(_build_hparams(), hidden_size=32, vocab_size=128)
    with pytest.raises(ValueError, match="requires explicit prompt units"):
        _run_adapter(
            adapter,
            content=torch.tensor([[1, 1, 2, 2]], dtype=torch.long),
            ref=torch.randn(1, 12, 80),
            ref_conditioning=None,
            auto_prompt_from_ref=False,
        )


def test_rhythm_v3_short_prompt_falls_back_to_global_only():
    hparams = {
        **_build_hparams(),
        "rhythm_operator_min_support_factor": 1.0,
    }
    adapter = ConanDurationAdapter(hparams, hidden_size=32, vocab_size=128)
    ret = _run_adapter(
        adapter,
        content=torch.tensor([[1, 1, 2, 2, 3, 4]], dtype=torch.long),
        ref=None,
        ref_conditioning=_build_prompt_conditioning(prompt_units=2),
    )
    ref_memory = ret["rhythm_ref_conditioning"]
    execution = ret["rhythm_execution"]
    prompt = ref_memory.prompt
    assert prompt is not None
    assert torch.allclose(ref_memory.operator_coeff, torch.zeros_like(ref_memory.operator_coeff))
    assert isinstance(prompt.prompt_short_fallback, torch.Tensor)
    assert torch.all(prompt.prompt_short_fallback == 1.0)
    assert isinstance(prompt.prompt_operator_support, torch.Tensor)
    assert float(prompt.prompt_operator_support[0, 0].item()) == 2.0
    assert torch.allclose(execution.local_response, torch.zeros_like(execution.local_response))


def test_rhythm_v3_new_backbone_surface_maps_to_progress_warp_candidate():
    hparams = {
        **_build_hparams(),
        "rhythm_v3_backbone": "global_only",
        "rhythm_v3_warp_mode": "progress",
        "rhythm_v3_allow_hybrid": False,
    }
    adapter = ConanDurationAdapter(hparams, hidden_size=32, vocab_size=128)
    assert adapter.module.backbone_mode == "global_only"
    assert adapter.module.warp_mode == "progress"
    assert adapter.module.allow_hybrid is False
    assert adapter.module.runtime_mode == "progress_only"


def test_rhythm_v3_new_backbone_surface_maps_to_detector_candidate():
    hparams = {
        **_build_hparams(),
        "rhythm_v3_backbone": "global_only",
        "rhythm_v3_warp_mode": "detector",
        "rhythm_v3_allow_hybrid": False,
        "lambda_rhythm_op": 0.0,
        "lambda_rhythm_zero": 0.0,
        "lambda_rhythm_ortho": 0.0,
    }
    adapter = ConanDurationAdapter(hparams, hidden_size=32, vocab_size=128)
    assert adapter.module.backbone_mode == "global_only"
    assert adapter.module.warp_mode == "detector"
    assert adapter.module.allow_hybrid is False
    assert adapter.module.runtime_mode == "detector_only"


def test_rhythm_v3_global_only_ignores_operator_coeff():
    hparams = {
        **_build_hparams(),
        "rhythm_v3_backbone": "global_only",
        "rhythm_v3_warp_mode": "none",
        "rhythm_v3_allow_hybrid": False,
        "lambda_rhythm_op": 0.0,
        "lambda_rhythm_zero": 0.0,
        "lambda_rhythm_ortho": 0.0,
    }
    adapter = ConanDurationAdapter(hparams, hidden_size=32, vocab_size=128)
    ret = _run_adapter(
        adapter,
        content=torch.tensor([[1, 1, 2, 2, 3, 4]], dtype=torch.long),
        ref=None,
        ref_conditioning={
            "global_rate": torch.full((1, 1), 0.35, dtype=torch.float32),
            "operator_coeff": torch.full((1, 4), 9.0, dtype=torch.float32),
        },
    )
    execution = ret["rhythm_execution"]
    committed = execution.commit_mask > 0.5
    expected = ret["rhythm_ref_conditioning"].global_rate.expand_as(execution.unit_logstretch)
    assert torch.allclose(
        execution.unit_logstretch[committed],
        expected[committed],
        atol=1.0e-5,
    )


def test_rhythm_v3_global_only_accepts_flat_memory_without_operator_coeff():
    hparams = {
        **_build_hparams(),
        "rhythm_v3_backbone": "global_only",
        "rhythm_v3_warp_mode": "none",
        "rhythm_v3_allow_hybrid": False,
        "lambda_rhythm_op": 0.0,
        "lambda_rhythm_zero": 0.0,
        "lambda_rhythm_ortho": 0.0,
    }
    adapter = ConanDurationAdapter(hparams, hidden_size=32, vocab_size=128)
    ret = _run_adapter(
        adapter,
        content=torch.tensor([[1, 1, 2, 2, 3, 4]], dtype=torch.long),
        ref=None,
        ref_conditioning={
            "global_rate": torch.full((1, 1), 0.20, dtype=torch.float32),
        },
    )
    ref_memory = ret["rhythm_ref_conditioning"]
    assert ref_memory.operator_coeff.shape == (1, 4)
    assert torch.allclose(ref_memory.operator_coeff, torch.zeros_like(ref_memory.operator_coeff))


def test_rhythm_v3_progress_only_ignores_operator_coeff():
    hparams = {
        **_build_hparams(),
        "rhythm_v3_backbone": "global_only",
        "rhythm_v3_warp_mode": "progress",
        "rhythm_v3_allow_hybrid": False,
        "lambda_rhythm_op": 0.0,
        "lambda_rhythm_zero": 0.0,
        "lambda_rhythm_ortho": 0.0,
    }
    adapter = ConanDurationAdapter(hparams, hidden_size=32, vocab_size=128)
    base_conditioning = {
        "global_rate": torch.full((1, 1), 0.10, dtype=torch.float32),
        "progress_profile": torch.tensor([[0.0, 0.25, -0.10, 0.15]], dtype=torch.float32),
    }
    ret_a = _run_adapter(
        adapter,
        content=torch.tensor([[1, 1, 2, 2, 3, 4]], dtype=torch.long),
        ref=None,
        ref_conditioning={
            **base_conditioning,
            "operator_coeff": torch.zeros((1, 4), dtype=torch.float32),
        },
    )
    ret_b = _run_adapter(
        adapter,
        content=torch.tensor([[1, 1, 2, 2, 3, 4]], dtype=torch.long),
        ref=None,
        ref_conditioning={
            **base_conditioning,
            "operator_coeff": torch.full((1, 4), 9.0, dtype=torch.float32),
        },
    )
    assert torch.allclose(
        ret_a["rhythm_execution"].unit_logstretch,
        ret_b["rhythm_execution"].unit_logstretch,
        atol=1.0e-5,
    )
    assert torch.allclose(
        ret_a["rhythm_execution"].local_response,
        torch.zeros_like(ret_a["rhythm_execution"].local_response),
        atol=1.0e-6,
    )
    assert torch.allclose(
        ret_a["rhythm_ref_conditioning"].progress_profile,
        base_conditioning["progress_profile"],
    )


def test_rhythm_v3_zero_progress_profile_matches_global_only():
    global_adapter = ConanDurationAdapter(
        {
            **_build_hparams(),
            "rhythm_v3_backbone": "global_only",
            "rhythm_v3_warp_mode": "none",
            "rhythm_v3_allow_hybrid": False,
            "lambda_rhythm_op": 0.0,
            "lambda_rhythm_zero": 0.0,
            "lambda_rhythm_ortho": 0.0,
        },
        hidden_size=32,
        vocab_size=128,
    )
    progress_adapter = ConanDurationAdapter(
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
    conditioning = {
        "global_rate": torch.full((1, 1), 0.20, dtype=torch.float32),
        "progress_profile": torch.zeros((1, 4), dtype=torch.float32),
        "operator_coeff": torch.full((1, 4), 5.0, dtype=torch.float32),
    }
    ret_global = _run_adapter(
        global_adapter,
        content=torch.tensor([[1, 1, 2, 2, 3, 4]], dtype=torch.long),
        ref=None,
        ref_conditioning=conditioning,
    )
    ret_progress = _run_adapter(
        progress_adapter,
        content=torch.tensor([[1, 1, 2, 2, 3, 4]], dtype=torch.long),
        ref=None,
        ref_conditioning=conditioning,
    )
    assert torch.allclose(
        ret_global["rhythm_execution"].unit_logstretch,
        ret_progress["rhythm_execution"].unit_logstretch,
        atol=1.0e-5,
    )


def test_rhythm_v3_detector_only_ignores_operator_coeff_and_emits_detector_response():
    hparams = {
        **_build_hparams(),
        "rhythm_v3_backbone": "global_only",
        "rhythm_v3_warp_mode": "detector",
        "rhythm_v3_allow_hybrid": False,
        "lambda_rhythm_op": 0.0,
        "lambda_rhythm_zero": 0.0,
        "lambda_rhythm_ortho": 0.0,
    }
    adapter = ConanDurationAdapter(hparams, hidden_size=32, vocab_size=128)
    base_conditioning = {
        "global_rate": torch.full((1, 1), 0.10, dtype=torch.float32),
        "detector_coeff": torch.tensor([[0.20, 0.50, -0.10, 0.30]], dtype=torch.float32),
    }
    ret_a = _run_adapter(
        adapter,
        content=torch.tensor([[1, 1, 2, 2, 3, 4]], dtype=torch.long),
        ref=None,
        ref_conditioning={
            **base_conditioning,
            "operator_coeff": torch.zeros((1, 4), dtype=torch.float32),
        },
    )
    ret_b = _run_adapter(
        adapter,
        content=torch.tensor([[1, 1, 2, 2, 3, 4]], dtype=torch.long),
        ref=None,
        ref_conditioning={
            **base_conditioning,
            "operator_coeff": torch.full((1, 4), 9.0, dtype=torch.float32),
        },
    )
    assert torch.allclose(
        ret_a["rhythm_execution"].unit_logstretch,
        ret_b["rhythm_execution"].unit_logstretch,
        atol=1.0e-5,
    )
    assert torch.allclose(
        ret_a["rhythm_execution"].local_response,
        torch.zeros_like(ret_a["rhythm_execution"].local_response),
        atol=1.0e-6,
    )
    assert ret_a["rhythm_execution"].detector_response is not None
    assert torch.isfinite(ret_a["rhythm_execution"].detector_response).all()
    assert float(ret_a["rhythm_execution"].detector_response.abs().sum().item()) > 0.0


def test_rhythm_v3_operator_srcres_adds_centered_source_residual():
    hparams = {
        **_build_hparams(),
        "rhythm_v3_warp_mode": "none",
        "rhythm_v3_allow_hybrid": False,
        "rhythm_v3_source_residual_gain": 1.0,
    }
    adapter = ConanDurationAdapter(hparams, hidden_size=32, vocab_size=128)
    content = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
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
        content_embed=torch.randn(1, 4, 32),
        tgt_nonpadding=torch.ones(1, 4, 1),
        content_lengths=torch.tensor([4], dtype=torch.long),
        ref_lengths=None,
        rhythm_state=None,
        rhythm_ref_conditioning={
            "global_rate": torch.zeros((1, 1), dtype=torch.float32),
            "operator_coeff": torch.zeros((1, 4), dtype=torch.float32),
        },
        rhythm_apply_override=None,
        rhythm_runtime_overrides=None,
        rhythm_source_cache={
            "content_units": content,
            "source_duration_obs": torch.tensor([[2.0, 4.0, 8.0, 16.0]], dtype=torch.float32),
            "unit_mask": torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32),
            "sealed_mask": torch.tensor([[1.0, 1.0, 1.0, 0.0]], dtype=torch.float32),
            "sep_mask": torch.zeros((1, 4), dtype=torch.float32),
            "unit_anchor_base": torch.ones((1, 4), dtype=torch.float32),
        },
        rhythm_offline_source_cache=None,
        speech_state_fn=lambda x: torch.randn(x.size(0), x.size(1), 32),
    )
    expected_raw = torch.log(torch.tensor([[2.0, 4.0, 8.0]], dtype=torch.float32))
    expected_mean = torch.cumsum(expected_raw, dim=1) / torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    expected = expected_raw - expected_mean
    assert torch.allclose(
        ret["rhythm_execution"].unit_logstretch[:, :3],
        expected,
        atol=1.0e-5,
    )


def test_rhythm_v3_projector_committed_speech_units_keep_at_least_one_frame():
    adapter = ConanDurationAdapter(
        {
            **_build_hparams(),
            "rhythm_v3_backbone": "global_only",
            "rhythm_v3_warp_mode": "none",
            "rhythm_v3_allow_hybrid": False,
        },
        hidden_size=32,
        vocab_size=128,
    )
    projected, residual = adapter.module.projector._project_duration_prefix(
        unit_duration_exec=torch.tensor([[0.20, 0.20]], dtype=torch.float32),
        commit_mask=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        speech_commit_mask=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        residual_prev=torch.zeros((1, 1), dtype=torch.float32),
        committed_units_prev=torch.zeros((1,), dtype=torch.long),
        cached_duration_exec_prev=None,
    )
    assert torch.all(projected >= 1.0)
    assert float(residual[0, 0].item()) < 0.0
