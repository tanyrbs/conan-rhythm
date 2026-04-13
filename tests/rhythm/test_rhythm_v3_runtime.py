from __future__ import annotations

import pytest
import torch

from modules.Conan.rhythm_v3.contracts import (
    ReferenceDurationMemory,
    StructuredDurationOperatorMemory,
    StructuredRoleDurationMemory,
)
import modules.Conan.rhythm_v3.runtime_adapter as runtime_adapter_mod
from modules.Conan.rhythm_v3.minimal_head import MinimalStreamingDurationHeadV1G
from modules.Conan.rhythm_v3.projector import StreamingDurationProjector
from modules.Conan.rhythm_v3.runtime_adapter import ConanDurationAdapter
from modules.Conan.rhythm_v3.summary_memory import StreamingDurationHead
from modules.Conan.rhythm_v3.unitizer import build_compressed_sequence
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


def _build_prompt_conditioning(*, prompt_units: int = 6, prompt_ref_len_sec: float = 5.0):
    content = torch.arange(1, prompt_units + 1, dtype=torch.long).unsqueeze(0)
    duration = torch.full((1, prompt_units), 3.0, dtype=torch.float32)
    mask = torch.ones((1, prompt_units), dtype=torch.float32)
    return {
        "prompt_content_units": content,
        "prompt_duration_obs": duration,
        "prompt_unit_mask": mask,
        "prompt_valid_mask": mask,
        "prompt_speech_mask": mask,
        "prompt_closed_mask": mask,
        "prompt_boundary_confidence": torch.ones((1, prompt_units), dtype=torch.float32),
        "prompt_ref_len_sec": torch.tensor([[prompt_ref_len_sec]], dtype=torch.float32),
        "prompt_speech_ratio_scalar": torch.ones((1, 1), dtype=torch.float32),
    }


def _build_prebuilt_minimal_ref_memory(
    adapter,
    *,
    with_summary_state: bool = False,
    with_role: bool = False,
):
    operator_rank = int(getattr(adapter.module.prompt_memory_encoder, "operator_rank", 4))
    summary_state = None
    if with_summary_state:
        summary_state = torch.zeros((1, operator_rank), dtype=torch.float32)
    role = None
    if with_role:
        role = StructuredRoleDurationMemory(
            role_value=torch.zeros((1, operator_rank), dtype=torch.float32),
            role_var=torch.zeros((1, operator_rank), dtype=torch.float32),
            role_coverage=torch.ones((1, operator_rank), dtype=torch.float32),
        )
    return ReferenceDurationMemory(
        global_rate=torch.zeros((1, 1), dtype=torch.float32),
        operator=StructuredDurationOperatorMemory(
            operator_coeff=torch.zeros((1, operator_rank), dtype=torch.float32),
        ),
        role=role,
        summary_state=summary_state,
    )


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


def test_rhythm_v3_operator_runtime_exports_source_prefix_diagnostics():
    adapter = ConanDurationAdapter(_build_hparams(), hidden_size=32, vocab_size=128)
    content = torch.tensor([[1, 1, 2, 2, 3, 4, 4, 5]])
    ref = torch.randn(1, 24, 80)
    ret = _run_adapter(adapter, content=content, ref=ref)
    execution = ret["rhythm_execution"]
    assert execution.g_src_utt is not None
    assert execution.g_src_prefix is not None
    assert execution.source_rate_seq is not None
    assert execution.g_src_prefix_mean is not None
    assert execution.next_state.local_rate_ema is not None
    assert torch.allclose(execution.source_rate_seq, execution.g_src_prefix)
    assert torch.allclose(ret["rhythm_g_src_utt"], execution.g_src_utt)
    assert torch.allclose(ret["rhythm_g_src_prefix_mean"], execution.g_src_prefix_mean)
    assert execution.g_src_prefix.shape == ret["speech_duration_exec"].shape


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


def test_rhythm_v3_nonminimal_prompt_summary_keeps_shared_duration_head_path():
    adapter = ConanDurationAdapter(_build_prompt_summary_hparams(), hidden_size=32, vocab_size=128)
    assert isinstance(adapter.module.duration_head, StreamingDurationHead)
    assert not isinstance(adapter.module.duration_head, MinimalStreamingDurationHeadV1G)


def test_rhythm_v3_prompt_summary_threads_minimal_v1_global_stat_switches():
    hparams = _build_prompt_summary_hparams()
    hparams["rhythm_v3_minimal_v1_profile"] = True
    hparams["rhythm_v3_rate_mode"] = "simple_global"
    hparams["rhythm_v3_simple_global_stats"] = True
    hparams["rhythm_v3_use_log_base_rate"] = False
    hparams["rhythm_v3_use_reference_summary"] = False
    hparams["rhythm_v3_use_learned_residual_gate"] = False
    hparams["rhythm_v3_disable_learned_gate"] = True
    hparams["rhythm_v3_coarse_delta_scale"] = 0.30
    hparams["rhythm_v3_local_residual_scale"] = 0.15
    hparams["rhythm_v3_src_rate_init_mode"] = "learned"
    hparams["rhythm_v3_src_rate_init_value"] = 0.75
    hparams["rhythm_v3_freeze_src_rate_init"] = True
    adapter = ConanDurationAdapter(hparams, hidden_size=32, vocab_size=128)
    assert adapter.unit_frontend.rate_mode == "simple_global"
    assert adapter.unit_frontend.simple_global_stats is True
    assert adapter.module.rate_mode == "simple_global"
    assert adapter.module.simple_global_stats is True
    assert adapter.module.use_log_base_rate is False
    assert adapter.module.use_reference_summary is False
    assert adapter.module.use_learned_residual_gate is False
    assert adapter.module.prompt_memory_encoder.rate_mode == "simple_global"
    assert adapter.module.prompt_memory_encoder.simple_global_stats is True
    assert adapter.module.prompt_memory_encoder.use_log_base_rate is False
    assert adapter.module.duration_head.rate_mode == "simple_global"
    assert adapter.module.duration_head.simple_global_stats is True
    assert adapter.module.duration_head.use_log_base_rate is False
    assert adapter.module.duration_head.use_learned_residual_gate is False
    assert adapter.module.duration_head.src_rate_init_mode == "learned"
    assert adapter.module.duration_head.coarse_delta_scale == pytest.approx(0.30)
    assert adapter.module.duration_head.local_residual_scale == pytest.approx(0.15)
    assert adapter.module.duration_head.freeze_src_rate_init is True
    assert adapter.module.duration_head.src_rate_init.requires_grad is False
    assert float(adapter.module.duration_head.src_rate_init.detach().item()) == pytest.approx(0.75)


def test_rhythm_v3_minimal_prompt_summary_defaults_to_frozen_src_rate_init():
    hparams = _build_prompt_summary_hparams()
    hparams["rhythm_v3_minimal_v1_profile"] = True
    hparams["rhythm_v3_rate_mode"] = "simple_global"
    hparams["rhythm_v3_simple_global_stats"] = True
    hparams["rhythm_v3_use_log_base_rate"] = False
    hparams["rhythm_v3_use_reference_summary"] = False
    hparams["rhythm_v3_use_learned_residual_gate"] = False
    hparams["rhythm_v3_disable_learned_gate"] = True
    hparams.pop("rhythm_v3_freeze_src_rate_init", None)
    adapter = ConanDurationAdapter(hparams, hidden_size=32, vocab_size=128)
    assert adapter.module.freeze_src_rate_init is True
    assert adapter.module.duration_head.freeze_src_rate_init is True
    assert adapter.module.duration_head.src_rate_init.requires_grad is False


def test_rhythm_v3_minimal_prompt_summary_uses_closed_boundary_clean_global_support():
    hparams = _build_prompt_summary_hparams()
    hparams["rhythm_v3_minimal_v1_profile"] = True
    hparams["rhythm_v3_rate_mode"] = "simple_global"
    hparams["rhythm_v3_simple_global_stats"] = True
    hparams["rhythm_v3_use_log_base_rate"] = False
    hparams["rhythm_v3_use_reference_summary"] = False
    hparams["rhythm_v3_use_learned_residual_gate"] = False
    hparams["rhythm_v3_disable_learned_gate"] = True
    hparams["rhythm_v3_min_boundary_confidence_for_g"] = 0.8
    adapter = ConanDurationAdapter(hparams, hidden_size=32, vocab_size=128)
    ref_memory = adapter.module.build_reference_conditioning(
        ref_conditioning={
            "prompt_content_units": torch.tensor([[5, 6, 7]], dtype=torch.long),
            "prompt_duration_obs": torch.tensor([[2.0, 100.0, 4.0]], dtype=torch.float32),
            "prompt_unit_mask": torch.ones((1, 3), dtype=torch.float32),
            "prompt_valid_mask": torch.ones((1, 3), dtype=torch.float32),
            "prompt_speech_mask": torch.ones((1, 3), dtype=torch.float32),
            "prompt_closed_mask": torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32),
            "prompt_boundary_confidence": torch.tensor([[0.9, 0.1, 0.95]], dtype=torch.float32),
            "prompt_ref_len_sec": torch.tensor([[5.0]], dtype=torch.float32),
            "prompt_speech_ratio_scalar": torch.ones((1, 1), dtype=torch.float32),
        }
    )
    expected = 0.5 * (
        torch.log(torch.tensor(2.0, dtype=torch.float32))
        + torch.log(torch.tensor(4.0, dtype=torch.float32))
    )
    assert torch.allclose(ref_memory.global_rate, expected.reshape(1, 1))


def test_rhythm_v3_minimal_prompt_summary_exports_falsification_debug_contract():
    hparams = _build_prompt_summary_hparams()
    hparams["rhythm_v3_minimal_v1_profile"] = True
    hparams["rhythm_v3_rate_mode"] = "simple_global"
    hparams["rhythm_v3_simple_global_stats"] = True
    hparams["rhythm_v3_use_log_base_rate"] = False
    hparams["rhythm_v3_use_reference_summary"] = False
    hparams["rhythm_v3_use_learned_residual_gate"] = False
    hparams["rhythm_v3_disable_learned_gate"] = True
    hparams["rhythm_v3_debug_export"] = True
    hparams["rhythm_v3_drop_edge_runs_for_g"] = 1
    adapter = ConanDurationAdapter(hparams, hidden_size=32, vocab_size=128)
    ret = _run_adapter(
        adapter,
        content=torch.tensor([[1, 57, 2, 2]], dtype=torch.long),
        ref=None,
        ref_conditioning={
            **_build_prompt_conditioning(prompt_units=3),
            "prompt_content_units": torch.tensor([[5, 57, 6]], dtype=torch.long),
            "prompt_duration_obs": torch.tensor([[4.0, 2.0, 8.0]], dtype=torch.float32),
            "prompt_speech_mask": torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32),
            "prompt_speech_ratio_scalar": torch.tensor([[6.0 / 7.0]], dtype=torch.float32),
        },
    )
    execution = ret["rhythm_execution"]
    debug = ret["rhythm_v3_debug"]
    assert isinstance(debug, dict)
    assert execution.g_ref is not None
    assert execution.g_src_prefix is not None
    assert execution.g_src_utt is not None
    assert execution.g_src_prefix_mean is not None
    assert execution.eval_mode == "learned"
    assert torch.allclose(debug["g_ref"], execution.g_ref)
    assert torch.allclose(debug["g_src_prefix"], execution.g_src_prefix)
    assert torch.allclose(debug["g_src_utt"], execution.g_src_utt)
    assert torch.allclose(debug["g_src_prefix_mean"], execution.g_src_prefix_mean)
    assert debug["g_variant"] == "raw_median"
    assert torch.allclose(debug["g_ref_scalar"], execution.g_ref)
    assert torch.allclose(debug["g_src_prefix_seq"], execution.g_src_prefix)
    assert debug["projector_prefix_offset"] is not None
    assert debug["projector_prefix_drift"] is not None
    assert debug["projector_rounding_residual"] is not None
    assert "projector_boundary_hit" in debug
    assert "projector_boundary_decay_applied" in debug
    assert "projector_budget_pos_used" in debug
    assert "projector_budget_neg_used" in debug
    assert "projector_budget_hit_mask" in debug
    assert "commit_closed_prefix_ok" in debug
    assert "open_tail_commit_violation" in debug
    assert "open_tail_commit_violation_count" in debug
    assert "projected_prefix_cumsum" in debug
    assert "source_prefix_cumsum" in debug
    assert "projector_preclamp_duration_exec" in debug
    assert "projector_clamp_delta" in debug
    assert "projector_projection_regret" in debug
    assert "projector_preclamp_prefix_cumsum" in debug
    assert debug["projector_budget_mode"] == "total"
    assert "coarse_path_logstretch" in debug
    assert "analytic_logstretch" in debug
    assert "coarse_delta" in debug
    assert "coarse_correction_used" in debug
    assert "coarse_correction_pred" in debug
    assert "residual_logstretch" in debug
    assert "residual_logstretch_used" in debug
    assert "residual_logstretch_pred" in debug
    assert "coarse_scalar_raw" in debug
    assert "global_term_before_local" in debug
    assert "unit_residual_gate" in debug
    assert "unit_residual_cold_gate" in debug
    assert "unit_residual_short_gate" in debug
    assert "unit_residual_gate_stability" in debug
    assert "residual_gate_cold" in debug
    assert "residual_gate_short" in debug
    assert "residual_gate_stability" in debug
    assert "unit_runtime_stability" in debug
    assert "residual_gate_mean" in debug
    assert "detach_global_term_in_local_head" in debug
    assert "speech_pred" in debug
    assert "silence_pred" in debug
    assert "projector_since_last_boundary" in debug
    assert "rhythm_debug_projector_boundary_hit" in ret
    assert "rhythm_debug_projector_boundary_decay" in ret
    assert "rhythm_debug_projector_budget_hit_mask" in ret
    assert "rhythm_debug_projector_prefix_drift" in ret
    assert "rhythm_debug_projected_prefix_cumsum" in ret
    assert "rhythm_debug_source_prefix_cumsum" in ret
    assert "rhythm_debug_projector_preclamp_duration_exec" in ret
    assert "rhythm_debug_projector_clamp_delta" in ret
    assert "rhythm_debug_projector_projection_regret" in ret
    assert "rhythm_debug_projector_preclamp_prefix_cumsum" in ret
    assert "rhythm_debug_commit_closed_prefix_ok" in ret
    assert "rhythm_debug_open_tail_commit_violation" in ret
    assert "rhythm_debug_open_tail_commit_violation_count" in ret
    assert "rhythm_debug_analytic_logstretch" in ret
    assert "rhythm_debug_coarse_delta" in ret
    assert "rhythm_debug_coarse_path" in ret
    assert "rhythm_debug_coarse_scalar_raw" in ret
    assert "rhythm_debug_global_term_before_local" in ret
    assert "rhythm_debug_residual_logstretch" in ret
    assert "rhythm_debug_unit_residual_gate" in ret
    assert "rhythm_debug_unit_residual_cold_gate" in ret
    assert "rhythm_debug_unit_residual_short_gate" in ret
    assert "rhythm_debug_unit_residual_gate_stability" in ret
    assert "rhythm_debug_residual_gate_cold" in ret
    assert "rhythm_debug_residual_gate_short" in ret
    assert "rhythm_debug_residual_gate_stability" in ret
    assert "rhythm_debug_unit_runtime_stability" in ret
    assert "rhythm_debug_residual_gate_mean" in ret
    assert "rhythm_debug_detach_global_term_in_local_head" in ret
    assert "rhythm_debug_speech_pred" in ret
    assert "rhythm_debug_silence_pred" in ret
    assert "rhythm_debug_prompt_ref_len_sec" in ret
    assert torch.allclose(ret["rhythm_debug_prompt_ref_len_sec"], torch.tensor([[5.0]], dtype=torch.float32))
    assert "rhythm_prompt_ref_len_sec" in ret
    assert torch.allclose(ret["rhythm_prompt_ref_len_sec"], torch.tensor([[5.0]], dtype=torch.float32))
    assert "rhythm_debug_projector_since_last_boundary" in ret
    assert "rhythm_debug_budget_hit_mask" in ret
    assert torch.allclose(ret["rhythm_g_src_utt"], execution.g_src_utt)
    assert torch.allclose(ret["rhythm_g_src_prefix_mean"], execution.g_src_prefix_mean)
    assert torch.allclose(execution.prompt_valid_len, torch.tensor([[3.0]], dtype=torch.float32))
    assert torch.allclose(execution.prompt_speech_ratio, torch.tensor([[6.0 / 7.0]], dtype=torch.float32))
    assert torch.allclose(ret["rhythm_prompt_valid_len"], execution.prompt_valid_len)
    assert torch.allclose(ret["rhythm_prompt_speech_ratio"], execution.prompt_speech_ratio)
    assert torch.allclose(debug["g_support_count"], torch.tensor([[2.0]], dtype=torch.float32))
    assert torch.allclose(debug["g_speech_count"], torch.tensor([[2.0]], dtype=torch.float32))
    assert torch.allclose(debug["g_valid_count"], torch.tensor([[3.0]], dtype=torch.float32))
    assert torch.allclose(debug["g_support_ratio_vs_speech"], torch.tensor([[1.0]], dtype=torch.float32))
    assert torch.allclose(debug["g_support_ratio_vs_valid"], torch.tensor([[2.0 / 3.0]], dtype=torch.float32))
    assert torch.allclose(debug["prompt_g_support_ratio_vs_speech"], torch.tensor([[1.0]], dtype=torch.float32))
    assert torch.allclose(debug["prompt_g_support_ratio_vs_valid"], torch.tensor([[2.0 / 3.0]], dtype=torch.float32))
    assert torch.allclose(debug["prompt_g_clean_ratio_vs_speech"], torch.tensor([[1.0]], dtype=torch.float32))
    assert torch.allclose(debug["prompt_g_clean_ratio_vs_valid"], torch.tensor([[2.0 / 3.0]], dtype=torch.float32))
    assert torch.allclose(debug["g_valid"], torch.tensor([[1.0]], dtype=torch.float32))
    assert torch.allclose(debug["g_valid_support"], torch.tensor([[1.0]], dtype=torch.float32))
    assert torch.allclose(debug["g_domain_valid"], torch.tensor([[1.0]], dtype=torch.float32))
    assert torch.allclose(debug["g_edge_runs_dropped"], torch.tensor([[0.0]], dtype=torch.float32))
    assert torch.allclose(debug["g_ref_len_valid"], torch.tensor([[1.0]], dtype=torch.float32))
    assert torch.allclose(debug["g_min_speech_ratio"], torch.tensor([[0.6]], dtype=torch.float32))
    assert torch.allclose(debug["prompt_speech_ratio"], torch.tensor([[6.0 / 7.0]], dtype=torch.float32))
    assert torch.allclose(ret["rhythm_debug_g_support_count"], debug["g_support_count"])
    assert torch.allclose(ret["rhythm_debug_g_support_ratio_vs_valid"], debug["g_support_ratio_vs_valid"])
    assert debug["eval_mode"] == "learned"
    assert execution.coarse_path_logstretch is not None
    assert execution.coarse_correction_pred is not None
    assert execution.local_residual_pred is not None
    assert torch.allclose(execution.coarse_path_logstretch, execution.coarse_logstretch)
    assert torch.allclose(debug["analytic_logstretch"], execution.global_shift_analytic)
    assert torch.allclose(debug["coarse_correction_used"], execution.coarse_correction)
    assert torch.allclose(debug["coarse_correction_pred"], execution.coarse_correction_pred)
    assert torch.allclose(debug["coarse_delta"], execution.coarse_correction)
    assert torch.allclose(debug["coarse_path_logstretch"], execution.coarse_path_logstretch)
    assert torch.allclose(debug["residual_logstretch_used"], execution.local_residual)
    assert torch.allclose(debug["residual_logstretch_pred"], execution.local_residual_pred)
    assert torch.allclose(debug["residual_logstretch"], execution.local_residual)
    assert torch.allclose(debug["coarse_scalar_raw"], execution.coarse_scalar_raw)
    assert torch.allclose(debug["global_term_before_local"], execution.global_term_before_local)
    assert torch.allclose(debug["unit_residual_gate"], execution.unit_residual_gate)
    assert torch.allclose(debug["unit_residual_cold_gate"], execution.unit_residual_cold_gate)
    assert torch.allclose(debug["unit_residual_short_gate"], execution.unit_residual_short_gate)
    assert torch.allclose(debug["unit_residual_gate_stability"], execution.unit_residual_gate_stability)
    assert torch.allclose(debug["residual_gate_cold"], execution.unit_residual_cold_gate)
    assert torch.allclose(debug["residual_gate_short"], execution.unit_residual_short_gate)
    assert torch.allclose(debug["residual_gate_stability"], execution.unit_residual_gate_stability)
    assert torch.allclose(ret["rhythm_debug_residual_gate_cold"], execution.unit_residual_cold_gate)
    assert torch.allclose(ret["rhythm_debug_residual_gate_short"], execution.unit_residual_short_gate)
    assert torch.allclose(ret["rhythm_debug_residual_gate_stability"], execution.unit_residual_gate_stability)
    assert torch.allclose(debug["unit_runtime_stability"], execution.unit_runtime_stability)
    assert torch.allclose(debug["residual_gate_mean"], execution.residual_gate_mean)
    assert torch.allclose(debug["detach_global_term_in_local_head"], torch.ones((1, 1), dtype=torch.float32))
    assert torch.allclose(ret["rhythm_debug_residual_used"], execution.local_residual)
    assert torch.allclose(ret["rhythm_debug_residual_pred"], execution.local_residual_pred)
    assert torch.allclose(debug["commit_closed_prefix_ok"], torch.ones((1, 1), dtype=torch.float32))
    assert torch.allclose(debug["open_tail_commit_violation"], torch.zeros_like(execution.commit_mask))
    assert torch.allclose(debug["open_tail_commit_violation_count"], torch.zeros((1, 1), dtype=torch.float32))
    assert torch.allclose(debug["projector_preclamp_duration_exec"], execution.projector_preclamp_duration_exec)
    assert torch.allclose(debug["projector_clamp_delta"], execution.projector_clamp_delta)
    assert torch.allclose(debug["projector_projection_regret"], execution.projector_projection_regret)
    assert torch.allclose(debug["projector_preclamp_prefix_cumsum"], execution.projector_preclamp_prefix_cumsum)
    assert torch.allclose(ret["rhythm_v3_commit_closed_prefix_ok"], debug["commit_closed_prefix_ok"])
    assert torch.allclose(ret["rhythm_v3_open_tail_commit_violation"], debug["open_tail_commit_violation"])
    assert torch.allclose(
        ret["rhythm_v3_open_tail_commit_violation_count"],
        debug["open_tail_commit_violation_count"],
    )


def test_rhythm_v3_non_debug_runtime_skips_g_debug_support_stats(monkeypatch):
    hparams = _build_prompt_summary_hparams()
    hparams["rhythm_v3_minimal_v1_profile"] = True
    hparams["rhythm_v3_rate_mode"] = "simple_global"
    hparams["rhythm_v3_simple_global_stats"] = True
    hparams["rhythm_v3_use_log_base_rate"] = False
    hparams["rhythm_v3_use_reference_summary"] = False
    hparams["rhythm_v3_use_learned_residual_gate"] = False
    hparams["rhythm_v3_disable_learned_gate"] = True
    hparams["rhythm_v3_debug_export"] = False
    adapter = ConanDurationAdapter(hparams, hidden_size=32, vocab_size=128)

    def _fail_if_debug_only_support_stats_are_built(*args, **kwargs):
        raise AssertionError("g debug support stats should not be built when debug_export is false")

    monkeypatch.setattr(
        runtime_adapter_mod,
        "summarize_global_rate_support",
        _fail_if_debug_only_support_stats_are_built,
    )
    ret = _run_adapter(
        adapter,
        content=torch.tensor([[1, 57, 2, 2]], dtype=torch.long),
        ref=None,
        ref_conditioning={
            **_build_prompt_conditioning(prompt_units=3),
            "prompt_content_units": torch.tensor([[5, 57, 6]], dtype=torch.long),
            "prompt_duration_obs": torch.tensor([[4.0, 2.0, 8.0]], dtype=torch.float32),
            "prompt_speech_mask": torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32),
            "prompt_speech_ratio_scalar": torch.tensor([[6.0 / 7.0]], dtype=torch.float32),
        },
    )
    assert "rhythm_v3_debug" not in ret
    assert "rhythm_debug_g_support_count" not in ret
    assert "rhythm_prompt_speech_ratio" in ret


def test_rhythm_v3_minimal_prompt_summary_threads_detach_global_term_switch():
    hparams = _build_prompt_summary_hparams()
    hparams["rhythm_v3_minimal_v1_profile"] = True
    hparams["rhythm_v3_rate_mode"] = "simple_global"
    hparams["rhythm_v3_simple_global_stats"] = True
    hparams["rhythm_v3_use_log_base_rate"] = False
    hparams["rhythm_v3_use_reference_summary"] = False
    hparams["rhythm_v3_use_learned_residual_gate"] = False
    hparams["rhythm_v3_disable_learned_gate"] = True
    hparams["rhythm_v3_debug_export"] = True
    hparams["rhythm_v3_detach_global_term_in_local_head"] = True
    adapter = ConanDurationAdapter(hparams, hidden_size=32, vocab_size=128)
    assert adapter.module.detach_global_term_in_local_head is True
    assert adapter.module.duration_head.detach_global_term_in_local_head is True
    ret = _run_adapter(
        adapter,
        content=torch.tensor([[1, 57, 2]], dtype=torch.long),
        ref=None,
        ref_conditioning={
            **_build_prompt_conditioning(prompt_units=3),
            "prompt_content_units": torch.tensor([[5, 57, 6]], dtype=torch.long),
            "prompt_duration_obs": torch.tensor([[4.0, 2.0, 8.0]], dtype=torch.float32),
            "prompt_speech_mask": torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32),
            "prompt_speech_ratio_scalar": torch.tensor([[6.0 / 7.0]], dtype=torch.float32),
        },
    )
    debug = ret["rhythm_v3_debug"]
    assert torch.allclose(debug["detach_global_term_in_local_head"], torch.ones((1, 1), dtype=torch.float32))
    assert ret["rhythm_v3_detach_global_term_in_local_head"] == 1.0


def test_rhythm_v3_minimal_prompt_summary_rejects_out_of_domain_prompt_ref_len_sec():
    hparams = _build_prompt_summary_hparams()
    hparams["rhythm_v3_minimal_v1_profile"] = True
    hparams["rhythm_v3_rate_mode"] = "simple_global"
    hparams["rhythm_v3_simple_global_stats"] = True
    hparams["rhythm_v3_use_log_base_rate"] = False
    hparams["rhythm_v3_use_reference_summary"] = False
    hparams["rhythm_v3_use_learned_residual_gate"] = False
    hparams["rhythm_v3_disable_learned_gate"] = True
    adapter = ConanDurationAdapter(hparams, hidden_size=32, vocab_size=128)
    with pytest.raises(ValueError, match="3-8s reference duration"):
        _run_adapter(
            adapter,
            content=torch.tensor([[1, 2, 3]], dtype=torch.long),
            ref=None,
            ref_conditioning=_build_prompt_conditioning(prompt_units=3, prompt_ref_len_sec=2.5),
        )


def test_rhythm_v3_debug_g_support_stats_respect_clean_support_sidecars():
    hparams = _build_prompt_summary_hparams()
    hparams["rhythm_v3_minimal_v1_profile"] = True
    hparams["rhythm_v3_rate_mode"] = "simple_global"
    hparams["rhythm_v3_simple_global_stats"] = True
    hparams["rhythm_v3_use_log_base_rate"] = False
    hparams["rhythm_v3_use_reference_summary"] = False
    hparams["rhythm_v3_use_learned_residual_gate"] = False
    hparams["rhythm_v3_disable_learned_gate"] = True
    hparams["rhythm_v3_debug_export"] = True
    hparams["rhythm_v3_min_boundary_confidence_for_g"] = 0.8
    adapter = ConanDurationAdapter(hparams, hidden_size=32, vocab_size=128)
    ret = _run_adapter(
        adapter,
        content=torch.tensor([[1, 2, 3]], dtype=torch.long),
        ref=None,
        ref_conditioning={
            "prompt_content_units": torch.tensor([[5, 6, 7]], dtype=torch.long),
            "prompt_duration_obs": torch.tensor([[2.0, 100.0, 4.0]], dtype=torch.float32),
            "prompt_unit_mask": torch.ones((1, 3), dtype=torch.float32),
            "prompt_valid_mask": torch.ones((1, 3), dtype=torch.float32),
            "prompt_speech_mask": torch.ones((1, 3), dtype=torch.float32),
            "prompt_closed_mask": torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32),
            "prompt_boundary_confidence": torch.tensor([[0.9, 0.1, 0.95]], dtype=torch.float32),
            "prompt_ref_len_sec": torch.tensor([[5.0]], dtype=torch.float32),
            "prompt_speech_ratio_scalar": torch.ones((1, 1), dtype=torch.float32),
        },
    )
    debug = ret["rhythm_v3_debug"]
    assert torch.allclose(debug["g_support_count"], torch.tensor([[2.0]], dtype=torch.float32))
    assert torch.allclose(debug["g_clean_count"], torch.tensor([[2.0]], dtype=torch.float32))
    assert torch.allclose(debug["g_support_ratio_vs_valid"], torch.tensor([[2.0 / 3.0]], dtype=torch.float32))
    assert torch.allclose(debug["g_ref_len_valid"], torch.tensor([[1.0]], dtype=torch.float32))
    assert torch.allclose(debug["g_domain_valid"], torch.tensor([[1.0]], dtype=torch.float32))


def test_rhythm_v3_debug_g_domain_valid_respects_prompt_ref_len_gate():
    hparams = _build_prompt_summary_hparams()
    hparams["rhythm_v3_minimal_v1_profile"] = True
    hparams["rhythm_v3_rate_mode"] = "simple_global"
    hparams["rhythm_v3_simple_global_stats"] = True
    hparams["rhythm_v3_use_log_base_rate"] = False
    hparams["rhythm_v3_use_reference_summary"] = False
    hparams["rhythm_v3_use_learned_residual_gate"] = False
    hparams["rhythm_v3_disable_learned_gate"] = True
    hparams["rhythm_v3_debug_export"] = True
    adapter = ConanDurationAdapter(hparams, hidden_size=32, vocab_size=128)
    adapter.eval()
    ret = _run_adapter(
        adapter,
        content=torch.tensor([[1, 2, 3]], dtype=torch.long),
        ref=None,
        ref_conditioning={
            **_build_prompt_conditioning(prompt_units=3, prompt_ref_len_sec=2.5),
            "prompt_content_units": torch.tensor([[5, 6, 7]], dtype=torch.long),
            "prompt_duration_obs": torch.tensor([[3.0, 3.0, 3.0]], dtype=torch.float32),
            "prompt_speech_mask": torch.ones((1, 3), dtype=torch.float32),
            "prompt_speech_ratio_scalar": torch.ones((1, 1), dtype=torch.float32),
        },
    )
    debug = ret["rhythm_v3_debug"]
    assert torch.allclose(debug["g_ref_len_valid"], torch.tensor([[0.0]], dtype=torch.float32))
    assert torch.allclose(debug["g_domain_valid"], torch.tensor([[0.0]], dtype=torch.float32))
    assert torch.allclose(ret["rhythm_prompt_domain_valid"], torch.tensor([[0.0]], dtype=torch.float32))
    assert ret["rhythm_domain_invalid_any"] == 1.0
    assert ret["rhythm_render_skipped_invalid_prompt"] == 1.0


def test_rhythm_v3_minimal_head_applies_explicit_analytic_gap_clip():
    head = MinimalStreamingDurationHeadV1G(
        vocab_size=32,
        dim=16,
        num_slots=1,
        simple_global_stats=True,
        use_log_base_rate=False,
        use_learned_residual_gate=False,
        analytic_gap_clip=0.25,
        eval_mode="analytic",
        disable_coarse_bias=True,
        disable_local_residual=True,
    )
    output = head(
        content_units=torch.tensor([[1, 2]], dtype=torch.long),
        log_anchor=torch.zeros((1, 2), dtype=torch.float32),
        unit_mask=torch.ones((1, 2), dtype=torch.float32),
        sealed_mask=torch.ones((1, 2), dtype=torch.float32),
        sep_hint=torch.zeros((1, 2), dtype=torch.float32),
        edge_cue=torch.zeros((1, 2), dtype=torch.float32),
        global_rate=torch.full((1, 1), 2.0, dtype=torch.float32),
        local_rate_ema=torch.zeros((1, 1), dtype=torch.float32),
        silence_mask=torch.zeros((1, 2), dtype=torch.float32),
    )
    expected = torch.full((1, 2), 0.25, dtype=torch.float32)
    assert torch.allclose(output["unit_analytic_logstretch"], expected)
    assert torch.allclose(output["unit_coarse_path_logstretch"], expected)
    assert torch.allclose(output["unit_logstretch"], expected)
    assert torch.allclose(output["unit_global_term_before_local"], expected)
    assert torch.allclose(output["unit_residual_gate"], torch.zeros((1, 2), dtype=torch.float32))
    assert output["coarse_scalar_raw"].shape == (1, 1)
    assert torch.isfinite(output["coarse_scalar_raw"]).all()
    assert torch.allclose(output["detach_global_term_in_local_head"], torch.zeros((1, 1), dtype=torch.float32))
    assert torch.allclose(output["residual_gate_mean"], torch.zeros((1, 1), dtype=torch.float32))
    assert output["unit_residual_cold_gate"].shape == (1, 2)
    assert output["unit_residual_short_gate"].shape == (1, 2)
    assert torch.isfinite(output["unit_residual_cold_gate"]).all()
    assert torch.isfinite(output["unit_residual_short_gate"]).all()
    assert torch.allclose(output["unit_runtime_stability"], torch.ones((1, 2), dtype=torch.float32))
    assert torch.allclose(output["residual_gate_cold"], output["unit_residual_cold_gate"])
    assert torch.allclose(output["residual_gate_short"], output["unit_residual_short_gate"])
    assert torch.allclose(output["residual_gate_stability"], output["unit_residual_gate_stability"])
    assert output["unit_silence_tau_surface_kind"] == "constant_clip"
    assert torch.allclose(output["unit_boundary_shaping"], torch.zeros((1, 2), dtype=torch.float32))
    assert torch.allclose(output["unit_leading_gate"], torch.ones((1, 2), dtype=torch.float32))


def test_rhythm_v3_minimal_head_first_speech_init_uses_first_speech_anchor():
    head = MinimalStreamingDurationHeadV1G(
        vocab_size=32,
        dim=16,
        num_slots=1,
        simple_global_stats=True,
        use_log_base_rate=False,
        use_learned_residual_gate=False,
        src_rate_init_mode="first_speech",
        disable_coarse_bias=True,
        disable_local_residual=True,
    )
    output = head(
        content_units=torch.tensor([[1, 31, 2]], dtype=torch.long),
        log_anchor=torch.tensor([[0.2, 1.5, 0.7]], dtype=torch.float32),
        unit_mask=torch.ones((1, 3), dtype=torch.float32),
        sealed_mask=torch.ones((1, 3), dtype=torch.float32),
        sep_hint=torch.zeros((1, 3), dtype=torch.float32),
        edge_cue=torch.zeros((1, 3), dtype=torch.float32),
        global_rate=torch.zeros((1, 1), dtype=torch.float32),
        local_rate_ema=None,
        silence_mask=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
    )
    expected = torch.tensor([[1.5, 1.5, 1.5]], dtype=torch.float32)
    assert torch.allclose(output["local_rate_seq"], expected)


def test_rhythm_v3_minimal_head_rejects_log_base_inputs():
    head = MinimalStreamingDurationHeadV1G(
        vocab_size=32,
        dim=16,
        num_slots=1,
        simple_global_stats=True,
        use_log_base_rate=False,
        use_learned_residual_gate=False,
    )
    with pytest.raises(ValueError, match="log_base=None"):
        head(
            content_units=torch.tensor([[1, 2]], dtype=torch.long),
            log_anchor=torch.zeros((1, 2), dtype=torch.float32),
            log_base=torch.zeros((1, 2), dtype=torch.float32),
            unit_mask=torch.ones((1, 2), dtype=torch.float32),
            sealed_mask=torch.ones((1, 2), dtype=torch.float32),
            sep_hint=torch.zeros((1, 2), dtype=torch.float32),
            edge_cue=torch.zeros((1, 2), dtype=torch.float32),
            global_rate=torch.zeros((1, 1), dtype=torch.float32),
            local_rate_ema=torch.zeros((1, 1), dtype=torch.float32),
            silence_mask=torch.zeros((1, 2), dtype=torch.float32),
        )


def test_rhythm_v3_streaming_head_exports_boundary_aware_silence_debug_surface():
    head = StreamingDurationHead(
        vocab_size=32,
        dim=16,
        num_slots=2,
        short_gap_silence_scale=0.25,
        leading_silence_scale=0.4,
        disable_coarse_bias=True,
        disable_local_residual=True,
        eval_mode="analytic",
    )
    output = head(
        content_units=torch.tensor([[1, 2, 3]], dtype=torch.long),
        log_anchor=torch.log(torch.tensor([[2.0, 2.0, 8.0]], dtype=torch.float32)),
        log_base=None,
        unit_mask=torch.ones((1, 3), dtype=torch.float32),
        sealed_mask=torch.ones((1, 3), dtype=torch.float32),
        sep_hint=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
        edge_cue=torch.zeros((1, 3), dtype=torch.float32),
        global_rate=torch.zeros((1, 1), dtype=torch.float32),
        summary_state=torch.zeros((1, 16), dtype=torch.float32),
        spk_embed=torch.zeros((1, 16), dtype=torch.float32),
        role_value=None,
        role_var=None,
        role_coverage=None,
        local_rate_ema=torch.zeros((1, 1), dtype=torch.float32),
        silence_mask=torch.tensor([[0.0, 1.0, 1.0]], dtype=torch.float32),
        run_stability=torch.ones((1, 3), dtype=torch.float32),
    )
    assert output["unit_silence_tau_surface_kind"] == "boundary_aware_clip"
    assert torch.allclose(output["unit_leading_gate"], torch.tensor([[0.4, 1.0, 1.0]], dtype=torch.float32))
    assert float(output["unit_boundary_shaping"][0, 2]) > float(output["unit_boundary_shaping"][0, 1])
    assert torch.allclose(output["unit_global_term_before_local"], output["unit_coarse_path_logstretch"])


def test_rhythm_v3_minimal_prompt_summary_uses_global_only_prompt_memory():
    hparams = _build_prompt_summary_hparams()
    hparams["rhythm_v3_minimal_v1_profile"] = True
    hparams["rhythm_v3_rate_mode"] = "simple_global"
    hparams["rhythm_v3_simple_global_stats"] = True
    hparams["rhythm_v3_use_log_base_rate"] = False
    hparams["rhythm_v3_use_reference_summary"] = False
    hparams["rhythm_v3_use_learned_residual_gate"] = False
    hparams["rhythm_v3_disable_learned_gate"] = True
    adapter = ConanDurationAdapter(hparams, hidden_size=32, vocab_size=128)
    prompt_a = {
        **_build_prompt_conditioning(prompt_units=3),
        "prompt_content_units": torch.tensor([[5, 57, 6]], dtype=torch.long),
        "prompt_duration_obs": torch.tensor([[4.0, 2.0, 8.0]], dtype=torch.float32),
        "prompt_speech_mask": torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32),
        "prompt_speech_ratio_scalar": torch.tensor([[6.0 / 7.0]], dtype=torch.float32),
    }
    prompt_b = {
        **prompt_a,
        "prompt_duration_obs": torch.tensor([[4.0, 8.0, 8.0]], dtype=torch.float32),
        "prompt_speech_ratio_scalar": torch.tensor([[0.6]], dtype=torch.float32),
    }
    memory_a = adapter.module.build_reference_conditioning(ref_conditioning=prompt_a)
    memory_b = adapter.module.build_reference_conditioning(ref_conditioning=prompt_b)
    assert torch.allclose(memory_a.global_rate, memory_b.global_rate)
    assert adapter.module.summary_codebook is None
    assert adapter.module.role_codebook is None
    assert isinstance(adapter.module.duration_head, MinimalStreamingDurationHeadV1G)
    assert memory_a.summary_state is None
    assert memory_a.role_value is None
    assert memory_a.role_var is None
    assert memory_a.role_coverage is None
    assert memory_a.prompt_role_attn is None


def test_rhythm_v3_minimal_prompt_summary_ignores_spurious_prompt_log_base():
    hparams = _build_prompt_summary_hparams()
    hparams["rhythm_v3_minimal_v1_profile"] = True
    hparams["rhythm_v3_rate_mode"] = "simple_global"
    hparams["rhythm_v3_simple_global_stats"] = True
    hparams["rhythm_v3_use_log_base_rate"] = False
    hparams["rhythm_v3_use_reference_summary"] = False
    hparams["rhythm_v3_use_learned_residual_gate"] = False
    hparams["rhythm_v3_disable_learned_gate"] = True
    adapter = ConanDurationAdapter(hparams, hidden_size=32, vocab_size=128)
    prompt = {
        **_build_prompt_conditioning(prompt_units=3),
        "prompt_content_units": torch.tensor([[5, 57, 6]], dtype=torch.long),
        "prompt_duration_obs": torch.tensor([[4.0, 2.0, 8.0]], dtype=torch.float32),
        "prompt_speech_mask": torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32),
        "prompt_speech_ratio_scalar": torch.tensor([[6.0 / 7.0]], dtype=torch.float32),
    }
    with_base = {
        **prompt,
        "prompt_log_base": torch.tensor([[9.0, 9.0, 9.0]], dtype=torch.float32),
        "prompt_unit_anchor_base": torch.tensor([[99.0, 99.0, 99.0]], dtype=torch.float32),
    }
    memory_plain = adapter.module.build_reference_conditioning(ref_conditioning=prompt)
    memory_with_base = adapter.module.build_reference_conditioning(ref_conditioning=with_base)
    assert torch.allclose(memory_plain.global_rate, memory_with_base.global_rate)
    assert memory_with_base.prompt_log_base is not None
    assert torch.allclose(memory_with_base.prompt_log_base, torch.zeros_like(memory_with_base.prompt_log_base))


def test_rhythm_v3_minimal_prompt_summary_rejects_prebuilt_summary_state_memory():
    hparams = _build_prompt_summary_hparams()
    hparams["rhythm_v3_minimal_v1_profile"] = True
    hparams["rhythm_v3_rate_mode"] = "simple_global"
    hparams["rhythm_v3_simple_global_stats"] = True
    hparams["rhythm_v3_use_log_base_rate"] = False
    hparams["rhythm_v3_use_reference_summary"] = False
    hparams["rhythm_v3_use_learned_residual_gate"] = False
    hparams["rhythm_v3_disable_learned_gate"] = True
    adapter = ConanDurationAdapter(hparams, hidden_size=32, vocab_size=128)
    bad_memory = _build_prebuilt_minimal_ref_memory(adapter, with_summary_state=True)
    with pytest.raises(RuntimeError, match="summary_state"):
        adapter.module.build_reference_conditioning(ref_conditioning=bad_memory)


def test_rhythm_v3_minimal_prompt_summary_forward_rejects_role_memory():
    hparams = _build_prompt_summary_hparams()
    hparams["rhythm_v3_minimal_v1_profile"] = True
    hparams["rhythm_v3_rate_mode"] = "simple_global"
    hparams["rhythm_v3_simple_global_stats"] = True
    hparams["rhythm_v3_use_log_base_rate"] = False
    hparams["rhythm_v3_use_reference_summary"] = False
    hparams["rhythm_v3_use_learned_residual_gate"] = False
    hparams["rhythm_v3_disable_learned_gate"] = True
    adapter = ConanDurationAdapter(hparams, hidden_size=32, vocab_size=128)
    source_batch = adapter.unit_frontend.from_precomputed(
        content_units=torch.tensor([[1, 57, 2]], dtype=torch.long),
        source_duration_obs=torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float32),
        unit_anchor_base=torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float32),
        unit_mask=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
        sealed_mask=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
        source_silence_mask=torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32),
    )
    bad_memory = _build_prebuilt_minimal_ref_memory(adapter, with_role=True)
    with pytest.raises(RuntimeError, match="forward: role"):
        adapter.module(source_batch=source_batch, ref_memory=bad_memory)


def test_rhythm_v3_minimal_prompt_summary_uses_constant_silence_clip():
    hparams = _build_prompt_summary_hparams()
    hparams["rhythm_v3_minimal_v1_profile"] = True
    hparams["rhythm_v3_rate_mode"] = "simple_global"
    hparams["rhythm_v3_simple_global_stats"] = True
    hparams["rhythm_v3_use_log_base_rate"] = False
    hparams["rhythm_v3_use_reference_summary"] = False
    hparams["rhythm_v3_use_learned_residual_gate"] = False
    hparams["rhythm_v3_disable_learned_gate"] = True
    hparams["rhythm_v3_silence_max_logstretch"] = 0.22
    adapter = ConanDurationAdapter(hparams, hidden_size=32, vocab_size=128)
    ret = _run_adapter(
        adapter,
        content=torch.tensor([[1, 57, 2]], dtype=torch.long),
        ref=None,
        ref_conditioning={
            **_build_prompt_conditioning(prompt_units=3),
            "prompt_content_units": torch.tensor([[1, 57, 2]], dtype=torch.long),
            "prompt_duration_obs": torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float32),
            "prompt_speech_mask": torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32),
            "prompt_ref_len_sec": torch.tensor([[3.0]], dtype=torch.float32),
            "prompt_speech_ratio_scalar": torch.tensor([[2.0 / 3.0]], dtype=torch.float32),
        },
    )
    execution = ret["rhythm_execution"]
    assert torch.allclose(execution.local_residual[0, 1], torch.tensor(0.0))
    expected = execution.coarse_logstretch[0, 1].clamp(min=-0.22, max=0.22)
    assert torch.allclose(execution.unit_logstretch[0, 1], expected)


def test_rhythm_v3_prompt_summary_prediction_anchor_keeps_open_visible_units():
    adapter = ConanDurationAdapter(_build_prompt_summary_hparams(), hidden_size=32, vocab_size=128)
    source_batch = adapter.unit_frontend.from_precomputed(
        content_units=torch.tensor([[1, 57, 2]], dtype=torch.long),
        source_duration_obs=torch.tensor([[2.0, 3.0, 4.0]], dtype=torch.float32),
        unit_anchor_base=torch.tensor([[9.0, 1.0, 1.0]], dtype=torch.float32),
        unit_mask=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
        sealed_mask=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        source_silence_mask=torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32),
    )
    commit_mask = source_batch.unit_mask.float() * source_batch.sealed_mask.float()
    speech_commit_mask = commit_mask * (1.0 - source_batch.source_silence_mask.float())
    anchor = adapter.module._resolve_prediction_anchor(
        source_batch=source_batch,
        speech_commit_mask=speech_commit_mask,
        commit_mask=commit_mask,
    )
    assert torch.allclose(anchor, source_batch.source_duration_obs.float())


def test_rhythm_v3_prompt_summary_runtime_tracks_incremental_frontend_state_without_precomputed_cache():
    adapter = ConanDurationAdapter(_build_prompt_summary_hparams(), hidden_size=32, vocab_size=128)
    first = _run_adapter(
        adapter,
        content=torch.tensor([[1, 1, 57, 57]], dtype=torch.long),
        ref=None,
        ref_conditioning=_build_prompt_conditioning(),
    )
    first_state = first["rhythm_state_next"]
    assert first_state.frontend_state is not None
    assert torch.equal(first_state.consumed_content_steps, torch.tensor([[4]], dtype=torch.long))

    second_content = torch.tensor([[1, 1, 57, 57, 2, 2]], dtype=torch.long)
    second = _run_adapter(
        adapter,
        content=second_content,
        ref=None,
        state=first_state,
        ref_conditioning=first["rhythm_ref_conditioning"],
    )
    second_state = second["rhythm_state_next"]
    assert second_state.frontend_state is not None
    assert torch.equal(second_state.consumed_content_steps, torch.tensor([[6]], dtype=torch.long))
    offline = adapter.unit_frontend.from_content_tensor(
        second_content,
        content_lengths=torch.tensor([6], dtype=torch.long),
        mark_last_open=True,
    )
    assert torch.equal(second["rhythm_unit_batch"].content_units, offline.content_units)
    assert torch.allclose(second["rhythm_unit_batch"].source_duration_obs, offline.source_duration_obs)

    third = _run_adapter(
        adapter,
        content=second_content,
        ref=None,
        state=second_state,
        ref_conditioning=second["rhythm_ref_conditioning"],
    )
    assert torch.equal(third["rhythm_state_next"].consumed_content_steps, torch.tensor([[6]], dtype=torch.long))
    assert torch.equal(third["rhythm_unit_batch"].content_units, second["rhythm_unit_batch"].content_units)
    assert torch.allclose(third["rhythm_unit_batch"].source_duration_obs, second["rhythm_unit_batch"].source_duration_obs)


def test_rhythm_v3_prompt_summary_runtime_tracks_batched_incremental_frontend_state_without_precomputed_cache():
    adapter = ConanDurationAdapter(_build_prompt_summary_hparams(), hidden_size=32, vocab_size=128)
    first_content = torch.tensor(
        [
            [1, 1, 57, 57],
            [3, 3, 4, 4],
        ],
        dtype=torch.long,
    )
    prompt_conditioning = {
        "prompt_content_units": torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long),
        "prompt_duration_obs": torch.full((2, 3), 3.0, dtype=torch.float32),
        "prompt_unit_mask": torch.ones((2, 3), dtype=torch.float32),
        "prompt_valid_mask": torch.ones((2, 3), dtype=torch.float32),
        "prompt_speech_mask": torch.ones((2, 3), dtype=torch.float32),
    }
    first = _run_adapter(
        adapter,
        content=first_content,
        ref=None,
        ref_conditioning=prompt_conditioning,
    )
    first_state = first["rhythm_state_next"]
    assert torch.equal(first_state.consumed_content_steps, torch.tensor([[4], [4]], dtype=torch.long))

    second_content = torch.tensor(
        [
            [1, 1, 57, 57, 2, 2],
            [3, 3, 4, 4, 5, 5],
        ],
        dtype=torch.long,
    )
    second = _run_adapter(
        adapter,
        content=second_content,
        ref=None,
        state=first_state,
        ref_conditioning=first["rhythm_ref_conditioning"],
    )
    second_state = second["rhythm_state_next"]
    assert torch.equal(second_state.consumed_content_steps, torch.tensor([[6], [6]], dtype=torch.long))
    offline = adapter.unit_frontend.from_content_tensor(
        second_content,
        content_lengths=torch.tensor([6, 6], dtype=torch.long),
        mark_last_open=True,
    )
    assert torch.equal(second["rhythm_unit_batch"].content_units, offline.content_units)
    assert torch.allclose(second["rhythm_unit_batch"].source_duration_obs, offline.source_duration_obs)


def test_rhythm_v3_prompt_summary_runtime_rejects_trimmed_incremental_content_without_source_cache():
    adapter = ConanDurationAdapter(_build_prompt_summary_hparams(), hidden_size=32, vocab_size=128)
    first = _run_adapter(
        adapter,
        content=torch.tensor([[1, 1, 2, 2]], dtype=torch.long),
        ref=None,
        ref_conditioning=_build_prompt_conditioning(),
    )
    with pytest.raises(ValueError, match="non-decreasing content lengths"):
        _run_adapter(
            adapter,
            content=torch.tensor([[2, 2]], dtype=torch.long),
            ref=None,
            state=first["rhythm_state_next"],
            ref_conditioning=first["rhythm_ref_conditioning"],
        )


def test_rhythm_v3_prompt_summary_can_disable_prompt_diagnostics_when_reference_summary_is_unused():
    hparams = _build_prompt_summary_hparams()
    hparams["rhythm_v3_rate_mode"] = "simple_global"
    hparams["rhythm_v3_simple_global_stats"] = True
    hparams["rhythm_v3_use_log_base_rate"] = False
    hparams["rhythm_v3_emit_prompt_diagnostics"] = False
    hparams["rhythm_v3_use_reference_summary"] = False
    adapter = ConanDurationAdapter(hparams, hidden_size=32, vocab_size=128)
    ret = _run_adapter(
        adapter,
        content=torch.tensor([[1, 1, 2, 2]], dtype=torch.long),
        ref=None,
        ref_conditioning=_build_prompt_conditioning(),
    )
    ref_memory = ret["rhythm_ref_conditioning"]
    assert ref_memory.summary_state is None
    assert ref_memory.role_value is None
    assert ref_memory.role_var is None
    assert ref_memory.role_coverage is None
    assert ref_memory.prompt_role_attn is None
    assert ref_memory.prompt_role_fit is None
    assert ref_memory.prompt_log_residual is not None
    assert torch.isfinite(ref_memory.global_rate).all()


def test_rhythm_v3_prompt_summary_coarse_correction_is_scalar_broadcast():
    adapter = ConanDurationAdapter(_build_prompt_summary_hparams(), hidden_size=32, vocab_size=128)
    content = torch.tensor([[1, 1, 2, 2, 3, 3]], dtype=torch.long)
    ret = _run_adapter(adapter, content=content, ref=None, ref_conditioning=_build_prompt_conditioning())
    execution = ret["rhythm_execution"]
    committed = execution.commit_mask[0] > 0.5
    coarse = execution.coarse_correction[0, committed]
    assert coarse.numel() > 0
    assert torch.allclose(coarse, coarse[:1].expand_as(coarse), atol=1.0e-6)
    assert execution.global_bias_scalar is not None
    assert torch.allclose(execution.global_bias_scalar[0, 0], coarse[0], atol=1.0e-6)


def test_rhythm_v3_prompt_summary_requires_explicit_silence_runs():
    hparams = _build_prompt_summary_hparams()
    hparams["rhythm_v3_emit_silence_runs"] = False
    with pytest.raises(ValueError, match="rhythm_v3_emit_silence_runs=true"):
        ConanDurationAdapter(hparams, hidden_size=32, vocab_size=128)


def test_rhythm_v3_non_minimal_prompt_summary_compat_derives_prompt_speech_mask_from_explicit_silence_units():
    adapter = ConanDurationAdapter(_build_prompt_summary_hparams(), hidden_size=32, vocab_size=128)
    content = torch.tensor([[1, 1, 57, 57, 2, 2]], dtype=torch.long)
    ret = _run_adapter(
        adapter,
        content=content,
        ref=None,
        ref_conditioning={
            "prompt_content_units": torch.tensor([[1, 57, 2]], dtype=torch.long),
            "prompt_duration_obs": torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float32),
            "prompt_unit_mask": torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
            "prompt_valid_mask": torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
        },
    )
    ref_memory = ret["rhythm_ref_conditioning"]
    assert ref_memory.prompt_speech_mask is not None
    assert torch.allclose(ref_memory.prompt_speech_mask, torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32))


def test_rhythm_v3_minimal_prompt_summary_requires_explicit_prompt_speech_mask():
    hparams = _build_prompt_summary_hparams()
    hparams["rhythm_v3_minimal_v1_profile"] = True
    hparams["rhythm_v3_rate_mode"] = "simple_global"
    hparams["rhythm_v3_simple_global_stats"] = True
    hparams["rhythm_v3_use_log_base_rate"] = False
    hparams["rhythm_v3_use_reference_summary"] = False
    hparams["rhythm_v3_use_learned_residual_gate"] = False
    hparams["rhythm_v3_disable_learned_gate"] = True
    adapter = ConanDurationAdapter(hparams, hidden_size=32, vocab_size=128)
    with pytest.raises(ValueError, match="explicit prompt_speech_mask"):
        _run_adapter(
            adapter,
            content=torch.tensor([[1, 57, 2]], dtype=torch.long),
            ref=None,
            ref_conditioning={
                "prompt_content_units": torch.tensor([[1, 57, 2]], dtype=torch.long),
                "prompt_duration_obs": torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float32),
                "prompt_unit_mask": torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
                "prompt_valid_mask": torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
                "prompt_silence_mask": torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32),
            },
        )


def test_rhythm_v3_minimal_module_contract_rejects_non_default_silence_scaling_knobs():
    hparams = _build_prompt_summary_hparams()
    hparams["rhythm_v3_minimal_v1_profile"] = True
    hparams["rhythm_v3_rate_mode"] = "simple_global"
    hparams["rhythm_v3_simple_global_stats"] = True
    hparams["rhythm_v3_use_log_base_rate"] = False
    hparams["rhythm_v3_use_reference_summary"] = False
    hparams["rhythm_v3_use_learned_residual_gate"] = False
    hparams["rhythm_v3_disable_learned_gate"] = True
    hparams["rhythm_v3_short_gap_silence_scale"] = 0.10
    with pytest.raises(ValueError, match="short_gap_silence_scale"):
        ConanDurationAdapter(hparams, hidden_size=32, vocab_size=128)


def test_rhythm_v3_minimal_module_contract_rejects_non_simple_global_runtime():
    hparams = _build_prompt_summary_hparams()
    hparams["rhythm_v3_minimal_v1_profile"] = True
    hparams["rhythm_v3_rate_mode"] = "log_base"
    hparams["rhythm_v3_simple_global_stats"] = False
    hparams["rhythm_v3_use_log_base_rate"] = True
    hparams["rhythm_v3_use_reference_summary"] = True
    with pytest.raises(ValueError, match="must be false|runtime contract violation"):
        ConanDurationAdapter(hparams, hidden_size=32, vocab_size=128)


def test_rhythm_v3_prompt_summary_local_response_cold_start_blocks_initial_speech():
    hparams = _build_prompt_summary_hparams()
    hparams["rhythm_v3_local_cold_start_runs"] = 3
    adapter = ConanDurationAdapter(hparams, hidden_size=32, vocab_size=128)
    content = torch.tensor([[1, 1, 2, 2, 3, 3]], dtype=torch.long)
    conditioning = _build_prompt_conditioning(prompt_units=6)
    ret = _run_adapter(adapter, content=content, ref=None, ref_conditioning=conditioning)
    local_response = ret["rhythm_execution"].local_response
    assert isinstance(local_response, torch.Tensor)
    assert torch.isfinite(local_response).all()
    assert torch.allclose(local_response[0, 0], torch.zeros(1))


def test_rhythm_v3_prompt_summary_runtime_run_stability_gates_local_residuals():
    hparams = _build_prompt_summary_hparams()
    hparams["rhythm_v3_local_cold_start_runs"] = 0
    hparams["rhythm_v3_local_short_run_min_duration"] = 1.0
    adapter = ConanDurationAdapter(hparams, hidden_size=32, vocab_size=128)
    content = torch.tensor([[1, 1, 2, 2, 3, 3]], dtype=torch.long)
    base_cache = {
        "content_units": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "source_duration_obs": torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float32),
        "unit_mask": torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
        "sealed_mask": torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
        "sep_mask": torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        "source_silence_mask": torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
    }

    def _run_with_stability(stability: torch.Tensor):
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
            content_embed=torch.zeros(content.size(0), content.size(1), 32),
            tgt_nonpadding=torch.ones(content.size(0), content.size(1), 1),
            content_lengths=torch.full((content.size(0),), int(content.size(1)), dtype=torch.long),
            rhythm_state=None,
            rhythm_ref_conditioning=_build_prompt_conditioning(prompt_units=6),
            rhythm_apply_override=None,
            rhythm_runtime_overrides=None,
            rhythm_source_cache={**base_cache, "source_run_stability": stability},
            rhythm_offline_source_cache=None,
            speech_state_fn=lambda x: torch.zeros((x.size(0), x.size(1), 32)),
        )
        return ret

    low = _run_with_stability(torch.zeros((1, 3), dtype=torch.float32))
    high = _run_with_stability(torch.ones((1, 3), dtype=torch.float32))
    assert torch.allclose(low["rhythm_execution"].local_response, torch.zeros_like(low["rhythm_execution"].local_response))
    assert torch.any(high["rhythm_execution"].local_response.abs() > 1.0e-6)


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


def test_rhythm_v3_prompt_summary_cached_source_requires_silence_mask():
    adapter = ConanDurationAdapter(_build_prompt_summary_hparams(), hidden_size=32, vocab_size=128)
    ret = {}
    with pytest.raises(ValueError, match="source_silence_mask"):
        adapter(
            ret=ret,
            content=torch.tensor([[1, 1, 2, 2]], dtype=torch.long),
            ref=None,
            target=None,
            f0=None,
            uv=None,
            infer=True,
            global_steps=0,
            content_embed=torch.randn(1, 4, 32),
            tgt_nonpadding=torch.ones(1, 4, 1),
            content_lengths=torch.tensor([4], dtype=torch.long),
            rhythm_state=None,
            rhythm_ref_conditioning=_build_prompt_conditioning(prompt_units=3),
            rhythm_apply_override=None,
            rhythm_runtime_overrides=None,
            rhythm_source_cache={
                "content_units": torch.tensor([[1, 57, 2]], dtype=torch.long),
                "source_duration_obs": torch.tensor([[2.0, 1.0, 2.0]], dtype=torch.float32),
                "unit_mask": torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
            },
            rhythm_offline_source_cache=None,
            speech_state_fn=lambda x: torch.randn(x.size(0), x.size(1), 32),
        )


def test_rhythm_v3_minimal_prompt_summary_cached_source_requires_cache_meta():
    hparams = _build_prompt_summary_hparams()
    hparams["rhythm_v3_minimal_v1_profile"] = True
    hparams["rhythm_v3_rate_mode"] = "simple_global"
    hparams["rhythm_v3_simple_global_stats"] = True
    hparams["rhythm_v3_use_log_base_rate"] = False
    hparams["rhythm_v3_use_reference_summary"] = False
    hparams["rhythm_v3_use_learned_residual_gate"] = False
    hparams["rhythm_v3_disable_learned_gate"] = True
    adapter = ConanDurationAdapter(hparams, hidden_size=32, vocab_size=128)
    ret = {}
    with pytest.raises(ValueError, match="rhythm_v3_cache_meta"):
        adapter(
            ret=ret,
            content=torch.tensor([[1, 1, 57, 57]], dtype=torch.long),
            ref=None,
            target=None,
            f0=None,
            uv=None,
            infer=True,
            global_steps=0,
            content_embed=torch.randn(1, 4, 32),
            tgt_nonpadding=torch.ones(1, 4, 1),
            content_lengths=torch.tensor([4], dtype=torch.long),
            rhythm_state=None,
            rhythm_ref_conditioning=_build_prompt_conditioning(prompt_units=3),
            rhythm_apply_override=None,
            rhythm_runtime_overrides=None,
            rhythm_source_cache={
                "content_units": torch.tensor([[1, 57, 2]], dtype=torch.long),
                "source_duration_obs": torch.tensor([[2.0, 1.0, 2.0]], dtype=torch.float32),
                "unit_mask": torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
                "sealed_mask": torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
                "sep_mask": torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
                "source_silence_mask": torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32),
                "source_run_stability": torch.ones((1, 3), dtype=torch.float32),
                "source_boundary_cue": torch.zeros((1, 3), dtype=torch.float32),
                "unit_anchor_base": torch.tensor([[2.0, 1.0, 2.0]], dtype=torch.float32),
            },
            rhythm_offline_source_cache=None,
            speech_state_fn=lambda x: torch.randn(x.size(0), x.size(1), 32),
        )


def test_rhythm_v3_build_compressed_sequence_debounce_preserves_open_tail():
    compressed = build_compressed_sequence(
        [1, 1, 2, 1],
        silent_token=57,
        separator_aware=False,
        tail_open_units=1,
        mark_last_open=True,
        emit_silence_runs=False,
        debounce_min_run_frames=2,
    )
    assert compressed.units == [1, 2, 1]
    assert compressed.durations == [2, 1, 1]
    assert compressed.open_run_mask == [0, 0, 1]
    assert compressed.sealed_mask == [1, 1, 0]
    assert compressed.tail_buffer == [1]


def test_rhythm_v3_build_compressed_sequence_merges_short_silence_flicker_when_closed():
    compressed = build_compressed_sequence(
        [1, 57, 1],
        silent_token=57,
        separator_aware=False,
        tail_open_units=2,
        mark_last_open=False,
        emit_silence_runs=True,
        debounce_min_run_frames=2,
    )
    assert compressed.units == [1]
    assert compressed.durations == [3]
    assert compressed.silence_mask == [0]
    assert compressed.sep_hint == [0]


def test_rhythm_v3_build_compressed_sequence_keeps_short_silence_flicker_in_open_tail():
    compressed = build_compressed_sequence(
        [1, 57, 1],
        silent_token=57,
        separator_aware=False,
        tail_open_units=1,
        mark_last_open=True,
        emit_silence_runs=True,
        debounce_min_run_frames=2,
    )
    assert compressed.units == [1, 57, 1]
    assert compressed.durations == [1, 1, 1]
    assert compressed.open_run_mask == [0, 0, 1]
    assert compressed.sealed_mask == [1, 1, 0]


def test_rhythm_v3_projector_dynamic_budget_clamps_short_prefix_more_tightly():
    projector_static = StreamingDurationProjector(
        prefix_budget_pos=24,
        prefix_budget_neg=24,
    )
    projector_dynamic = StreamingDurationProjector(
        prefix_budget_pos=24,
        prefix_budget_neg=24,
        dynamic_budget_ratio=0.10,
        min_prefix_budget=2,
        max_prefix_budget=6,
    )
    kwargs = dict(
        unit_logstretch=torch.zeros((1, 1), dtype=torch.float32),
        unit_duration_exec=torch.tensor([[30.0]], dtype=torch.float32),
        basis_activation=torch.zeros((1, 1, 1), dtype=torch.float32),
        source_duration_obs=torch.tensor([[10.0]], dtype=torch.float32),
        unit_mask=torch.tensor([[1.0]], dtype=torch.float32),
        sealed_mask=torch.tensor([[1.0]], dtype=torch.float32),
        speech_commit_mask=torch.tensor([[1.0]], dtype=torch.float32),
        state=None,
    )
    static_execution = projector_static.finalize_execution(**kwargs)
    dynamic_execution = projector_dynamic.finalize_execution(**kwargs)
    assert torch.allclose(static_execution.unit_duration_exec, torch.tensor([[30.0]], dtype=torch.float32))
    assert torch.allclose(dynamic_execution.unit_duration_exec, torch.tensor([[12.0]], dtype=torch.float32))
    assert dynamic_execution.projector_budget_pos_used is not None
    assert dynamic_execution.projector_budget_neg_used is not None
    assert torch.allclose(dynamic_execution.projector_budget_pos_used, torch.tensor([[2.0]], dtype=torch.float32))
    assert torch.allclose(dynamic_execution.projector_budget_neg_used, torch.tensor([[2.0]], dtype=torch.float32))
    assert dynamic_execution.projector_budget_hit_pos is not None
    assert bool(dynamic_execution.projector_budget_hit_pos[0, 0].item()) is True
    assert static_execution.projector_budget_hit_pos is not None
    assert bool(static_execution.projector_budget_hit_pos[0, 0].item()) is False


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


def test_rhythm_v3_minimal_v1_invalid_prompt_prefers_skip_and_reports_status():
    hparams = _build_prompt_summary_hparams()
    hparams["rhythm_v3_minimal_v1_profile"] = True
    hparams["rhythm_v3_rate_mode"] = "simple_global"
    hparams["rhythm_v3_simple_global_stats"] = True
    hparams["rhythm_v3_use_log_base_rate"] = False
    hparams["rhythm_v3_use_reference_summary"] = False
    hparams["rhythm_v3_use_learned_residual_gate"] = False
    hparams["rhythm_v3_disable_learned_gate"] = True
    adapter = ConanDurationAdapter(hparams, hidden_size=32, vocab_size=128)
    adapter.eval()
    invalid_prompt = {
        **_build_prompt_conditioning(),
        "prompt_speech_mask": torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        "prompt_speech_ratio_scalar": torch.tensor([[1.0 / 6.0]], dtype=torch.float32),
    }
    ret = _run_adapter(
        adapter,
        content=torch.tensor([[1, 1, 2, 2, 3, 4]]),
        ref=None,
        ref_conditioning=invalid_prompt,
        auto_prompt_from_ref=False,
    )
    assert ret["rhythm_domain_invalid_any"] == 1.0
    assert torch.allclose(ret["rhythm_prompt_domain_valid"], torch.zeros_like(ret["rhythm_prompt_domain_valid"]))
    assert torch.all(ret["rhythm_domain_invalid"] > 0.5)
    assert ret["rhythm_apply_render"] == 0.0
    assert ret["rhythm_render_skipped_invalid_prompt"] == 1.0
    execution = ret["rhythm_execution"]
    assert torch.allclose(execution.unit_logstretch, torch.zeros_like(execution.unit_logstretch))
    assert torch.allclose(execution.coarse_logstretch, torch.zeros_like(execution.coarse_logstretch))
    assert torch.allclose(execution.local_residual, torch.zeros_like(execution.local_residual))


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
    projected, residual, prefix_offset, _, _ = adapter.module.projector._project_duration_prefix(
        unit_duration_exec=torch.tensor([[0.20, 0.20]], dtype=torch.float32),
        source_duration_obs=torch.tensor([[2.0, 2.0]], dtype=torch.float32),
        commit_mask=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        speech_commit_mask=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        coarse_only_commit_mask=None,
        residual_prev=torch.zeros((1, 1), dtype=torch.float32),
        prefix_unit_offset_prev=torch.zeros((1, 1), dtype=torch.float32),
        committed_units_prev=torch.zeros((1,), dtype=torch.long),
        cached_duration_exec_prev=None,
        budget_pos=24,
        budget_neg=24,
    )
    assert torch.all(projected >= 1.0)
    assert float(residual[0, 0].item()) < 0.0
    assert float(prefix_offset[0, 0].item()) == -2.0


def test_rhythm_v3_projector_applies_prefix_unit_budget_clamp():
    adapter = ConanDurationAdapter(
        {
            **_build_hparams(),
            "rhythm_v3_backbone": "global_only",
            "rhythm_v3_warp_mode": "none",
            "rhythm_v3_allow_hybrid": False,
            "rhythm_v3_prefix_budget_pos": 1,
            "rhythm_v3_prefix_budget_neg": 1,
        },
        hidden_size=32,
        vocab_size=128,
    )
    projected, residual, prefix_offset, _, _ = adapter.module.projector._project_duration_prefix(
        unit_duration_exec=torch.tensor([[20.0, 20.0, 20.0]], dtype=torch.float32),
        source_duration_obs=torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float32),
        commit_mask=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
        speech_commit_mask=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
        coarse_only_commit_mask=None,
        residual_prev=torch.zeros((1, 1), dtype=torch.float32),
        prefix_unit_offset_prev=torch.zeros((1, 1), dtype=torch.float32),
        committed_units_prev=torch.zeros((1,), dtype=torch.long),
        cached_duration_exec_prev=None,
        budget_pos=1,
        budget_neg=1,
    )
    assert torch.equal(projected, torch.tensor([[3.0, 2.0, 2.0]], dtype=torch.float32))
    assert float(prefix_offset[0, 0].item()) == 1.0
    assert float(residual[0, 0].item()) >= 0.0


def test_rhythm_v3_projector_rejects_noncontiguous_visible_prefix_commit_mask():
    projector = StreamingDurationProjector(prefix_budget_pos=24, prefix_budget_neg=24)
    with pytest.raises(ValueError, match="contiguous visible prefix"):
        projector.finalize_execution(
            unit_logstretch=torch.zeros((1, 3), dtype=torch.float32),
            unit_duration_exec=torch.ones((1, 3), dtype=torch.float32),
            basis_activation=torch.zeros((1, 3, 1), dtype=torch.float32),
            source_duration_obs=torch.ones((1, 3), dtype=torch.float32),
            unit_mask=torch.ones((1, 3), dtype=torch.float32),
            sealed_mask=torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32),
            speech_commit_mask=torch.ones((1, 3), dtype=torch.float32),
            state=None,
        )


def test_rhythm_v3_projector_rejects_committing_open_tail_units():
    with pytest.raises(ValueError, match="open-tail"):
        StreamingDurationProjector._validate_prefix_commit_mask(
            unit_mask=torch.ones((1, 3), dtype=torch.float32),
            commit_mask=torch.ones((1, 3), dtype=torch.float32),
            sealed_mask=torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32),
        )


def test_rhythm_v3_projector_exports_closed_prefix_commit_telemetry():
    projector = StreamingDurationProjector(prefix_budget_pos=24, prefix_budget_neg=24)
    execution = projector.finalize_execution(
        unit_logstretch=torch.zeros((1, 3), dtype=torch.float32),
        unit_duration_exec=torch.ones((1, 3), dtype=torch.float32),
        basis_activation=torch.zeros((1, 3, 1), dtype=torch.float32),
        source_duration_obs=torch.ones((1, 3), dtype=torch.float32),
        unit_mask=torch.ones((1, 3), dtype=torch.float32),
        sealed_mask=torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32),
        speech_commit_mask=torch.ones((1, 3), dtype=torch.float32),
        state=None,
    )
    assert torch.allclose(execution.commit_mask, torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32))
    assert torch.allclose(execution.commit_closed_prefix_ok, torch.ones((1, 1), dtype=torch.float32))
    assert torch.allclose(execution.open_tail_commit_violation, torch.zeros((1, 3), dtype=torch.float32))
    assert torch.allclose(execution.open_tail_commit_violation_count, torch.zeros((1, 1), dtype=torch.float32))


def test_rhythm_v3_projector_resets_carry_across_phrase_boundary():
    projector_no_reset = StreamingDurationProjector(
        prefix_budget_pos=24,
        prefix_budget_neg=24,
        boundary_carry_decay=1.0,
        boundary_reset_thresh=0.5,
    )
    projector_reset = StreamingDurationProjector(
        prefix_budget_pos=24,
        prefix_budget_neg=24,
        boundary_carry_decay=0.0,
        boundary_reset_thresh=0.5,
    )
    kwargs = dict(
        unit_duration_exec=torch.tensor([[2.6, 2.6]], dtype=torch.float32),
        source_duration_obs=torch.tensor([[2.0, 2.0]], dtype=torch.float32),
        commit_mask=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        speech_commit_mask=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        coarse_only_commit_mask=None,
        source_boundary_cue=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        phrase_final_mask=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        residual_prev=torch.zeros((1, 1), dtype=torch.float32),
        prefix_unit_offset_prev=torch.zeros((1, 1), dtype=torch.float32),
        committed_units_prev=torch.zeros((1,), dtype=torch.long),
        cached_duration_exec_prev=None,
        budget_pos=24,
        budget_neg=24,
        boundary_reset_thresh=0.5,
    )
    projected_no_reset, _, _, boundary_hit_no_reset, boundary_decay_no_reset = projector_no_reset._project_duration_prefix(
        **kwargs,
        boundary_carry_decay=1.0,
    )
    projected_reset, _, _, boundary_hit_reset, boundary_decay_reset = projector_reset._project_duration_prefix(
        **kwargs,
        boundary_carry_decay=0.0,
    )
    assert torch.equal(projected_no_reset, torch.tensor([[3.0, 2.0]], dtype=torch.float32))
    assert torch.equal(projected_reset, torch.tensor([[3.0, 3.0]], dtype=torch.float32))
    assert float(boundary_hit_no_reset.sum().item()) >= 1.0
    assert float(boundary_decay_no_reset.sum().item()) == 0.0
    assert float(boundary_hit_reset.sum().item()) >= 1.0
    assert float(boundary_decay_reset.sum().item()) >= 1.0


def test_rhythm_v3_projector_tracks_since_last_boundary_state():
    projector = StreamingDurationProjector(
        prefix_budget_pos=24,
        prefix_budget_neg=24,
        boundary_carry_decay=0.25,
        boundary_reset_thresh=0.5,
    )
    execution = projector.finalize_execution(
        unit_logstretch=torch.zeros((1, 3), dtype=torch.float32),
        unit_duration_exec=torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float32),
        basis_activation=torch.zeros((1, 3, 1), dtype=torch.float32),
        source_duration_obs=torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float32),
        unit_mask=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
        sealed_mask=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
        speech_commit_mask=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
        source_boundary_cue=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        phrase_final_mask=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        state=None,
    )
    assert execution.next_state.since_last_boundary is not None
    assert torch.allclose(execution.next_state.since_last_boundary, torch.tensor([[2.0]], dtype=torch.float32))
    assert execution.projector_since_last_boundary is not None
    assert torch.allclose(execution.projector_since_last_boundary, torch.tensor([[2.0]], dtype=torch.float32))
    assert execution.projector_boundary_hit is not None
    assert float(execution.projector_boundary_hit.sum().item()) >= 1.0
    assert execution.projector_boundary_decay_applied is not None
    assert float(execution.projector_boundary_decay_applied.sum().item()) >= 1.0


def test_rhythm_v3_projector_tensor_budget_matches_scalar_budget_resolution():
    projector = StreamingDurationProjector(
        prefix_budget_pos=3,
        prefix_budget_neg=3,
        dynamic_budget_ratio=0.5,
        min_prefix_budget=1,
        max_prefix_budget=4,
    )
    source_duration_obs = torch.tensor(
        [[2.0, 2.0, 2.0], [3.0, 1.0, 5.0]],
        dtype=torch.float32,
    )
    committed_len = torch.tensor([2, 3], dtype=torch.long)
    budget_tensor = projector._resolve_prefix_budget_tensor(
        source_duration_obs=source_duration_obs,
        speech_commit_mask=torch.ones_like(source_duration_obs),
        committed_len=committed_len,
        static_budget=projector.prefix_budget_pos,
        dynamic_budget_ratio=projector.dynamic_budget_ratio,
        min_prefix_budget=projector.min_prefix_budget,
        max_prefix_budget=projector.max_prefix_budget,
        budget_mode=projector.budget_mode,
    )
    expected = torch.tensor(
        [
                projector._resolve_prefix_budget(
                    source_duration_obs=source_duration_obs[0],
                    speech_commit_mask=torch.ones_like(source_duration_obs[0]),
                    committed_len=2,
                    static_budget=projector.prefix_budget_pos,
                    dynamic_budget_ratio=projector.dynamic_budget_ratio,
                    min_prefix_budget=projector.min_prefix_budget,
                    max_prefix_budget=projector.max_prefix_budget,
                    budget_mode=projector.budget_mode,
                ),
                projector._resolve_prefix_budget(
                    source_duration_obs=source_duration_obs[1],
                    speech_commit_mask=torch.ones_like(source_duration_obs[1]),
                    committed_len=3,
                    static_budget=projector.prefix_budget_pos,
                    dynamic_budget_ratio=projector.dynamic_budget_ratio,
                    min_prefix_budget=projector.min_prefix_budget,
                    max_prefix_budget=projector.max_prefix_budget,
                    budget_mode=projector.budget_mode,
                ),
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(budget_tensor, expected)


def test_rhythm_v3_frame_plan_uses_real_source_timeline_for_explicit_silence_runs():
    adapter = ConanDurationAdapter(_build_prompt_summary_hparams(), hidden_size=32, vocab_size=128)
    ret = {}
    content = torch.tensor([[1, 57, 2]], dtype=torch.long)
    adapter(
        ret=ret,
        content=content,
        ref=None,
        target=None,
        f0=None,
        uv=None,
        infer=True,
        global_steps=0,
        content_embed=torch.randn(1, 3, 32),
        tgt_nonpadding=torch.ones(1, 3, 1),
        content_lengths=torch.tensor([3], dtype=torch.long),
        ref_lengths=None,
        rhythm_state=None,
        rhythm_ref_conditioning={
            **_build_prompt_conditioning(prompt_units=3),
            "prompt_content_units": torch.tensor([[1, 57, 2]], dtype=torch.long),
            "prompt_duration_obs": torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float32),
            "prompt_speech_mask": torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32),
            "prompt_ref_len_sec": torch.tensor([[3.0]], dtype=torch.float32),
            "prompt_speech_ratio_scalar": torch.tensor([[2.0 / 3.0]], dtype=torch.float32),
        },
        rhythm_apply_override=False,
        rhythm_runtime_overrides=None,
        rhythm_source_cache={
            "content_units": content,
            "source_duration_obs": torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float32),
            "unit_mask": torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
            "sealed_mask": torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
            "sep_mask": torch.zeros((1, 3), dtype=torch.float32),
            "source_silence_mask": torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32),
            "unit_anchor_base": torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float32),
        },
        rhythm_offline_source_cache=None,
        speech_state_fn=lambda x: torch.randn(x.size(0), x.size(1), 32),
    )
    plan = ret["rhythm_frame_plan"]
    total = plan.total_mask[0] > 0.5
    unit_index = plan.frame_unit_index[0, total]
    src_index = plan.frame_src_index[0, total]
    silence_src = src_index[unit_index == 1]
    assert silence_src.numel() > 0
    assert int(silence_src[0].item()) == 2
    assert torch.all(silence_src[1:] >= silence_src[:-1])
    assert silence_src.numel() == int(round(float(ret["rhythm_execution"].speech_duration_exec[0, 1].item())))
    silence_tau = _build_duration_v3_silence_tau(
        prediction_anchor=ret["rhythm_unit_batch"].source_duration_obs.float(),
        committed_silence_mask=(
            ret["rhythm_unit_batch"].source_silence_mask.float()
            * ret["rhythm_execution"].commit_mask.float()
        ),
        sep_hint=ret["rhythm_unit_batch"].sep_mask.float(),
        boundary_cue=getattr(ret["rhythm_unit_batch"], "source_boundary_cue", None),
        max_silence_logstretch=float(adapter.hparams.get("rhythm_v3_silence_max_logstretch", 0.35)),
        short_gap_scale=float(adapter.hparams.get("rhythm_v3_short_gap_silence_scale", 0.35)),
    )
    expected = ret["rhythm_execution"].coarse_logstretch[0, 1].clamp(
        min=-silence_tau[0, 1],
        max=silence_tau[0, 1],
    )
    assert torch.allclose(ret["rhythm_execution"].local_residual[0, 1], torch.tensor(0.0))
    assert torch.allclose(ret["rhythm_execution"].unit_logstretch[0, 1], expected)


def test_rhythm_v3_render_keeps_raw_open_tail_after_committed_prefix():
    adapter = ConanDurationAdapter(
        {
            **_build_hparams(),
            "rhythm_v3_backbone": "global_only",
            "rhythm_v3_warp_mode": "none",
            "rhythm_v3_allow_hybrid": False,
            "rhythm_apply_mode": "always",
        },
        hidden_size=32,
        vocab_size=128,
    )
    ret = {}
    content = torch.tensor([[1, 57, 2, 3]], dtype=torch.long)
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
        rhythm_ref_conditioning={"global_rate": torch.zeros((1, 1), dtype=torch.float32)},
        rhythm_apply_override=True,
        rhythm_runtime_overrides=None,
        rhythm_source_cache={
            "content_units": content,
            "source_duration_obs": torch.tensor([[2.0, 1.0, 2.0, 4.0]], dtype=torch.float32),
            "unit_mask": torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32),
            "sealed_mask": torch.tensor([[1.0, 1.0, 1.0, 0.0]], dtype=torch.float32),
            "sep_mask": torch.zeros((1, 4), dtype=torch.float32),
            "source_silence_mask": torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32),
            "unit_anchor_base": torch.tensor([[2.0, 1.0, 2.0, 4.0]], dtype=torch.float32),
        },
        rhythm_offline_source_cache=None,
        speech_state_fn=lambda x: torch.randn(x.size(0), x.size(1), 32),
    )
    rendered = ret["content"]
    assert rendered.shape[1] >= 4
    assert torch.equal(rendered[0, -4:], torch.tensor([3, 3, 3, 3], dtype=torch.long))
    assert int(ret["rhythm_render_unit_index"][0, -1].item()) == 3
    assert ret["rhythm_v3_has_uncommitted_tail"] == 1.0
    assert ret["rhythm_v3_render_frame_plan_contains_uncommitted_tail"] == 1.0
    assert ret["rhythm_v3_render_open_tail_frame_count"] >= 4.0
    assert torch.allclose(ret["rhythm_v3_open_tail_commit_violation_count"], torch.zeros((1, 1)))
