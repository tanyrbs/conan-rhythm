from __future__ import annotations

import torch

from tasks.Conan.rhythm.runtime_modes import (
    build_duration_v3_ref_conditioning,
    build_legacy_v2_ref_conditioning,
    build_rhythm_ref_conditioning,
    resolve_task_runtime_state,
)


def test_build_duration_v3_ref_conditioning_normalizes_rhythm_ref_conditioning_owner():
    rhythm_ref_conditioning = {
        "global_rate": torch.tensor([[0.25]], dtype=torch.float32),
        "operator_coeff": torch.randn(1, 4),
        "prompt_basis_activation": torch.randn(1, 6, 4),
        "prompt_random_target": torch.randn(1, 6),
        "prompt_mask": torch.ones(1, 6),
        "prompt_operator_fit": torch.randn(1, 6),
        "ref_rhythm_stats": torch.randn(1, 6),
        "ref_rhythm_trace": torch.randn(1, 8, 5),
        "planner_ref_stats": torch.randn(1, 2),
        "selector_meta_scores": torch.randn(1, 3),
        "ref_phrase_valid": torch.ones(1, 2),
    }
    sample = {
        "rhythm_ref_conditioning": rhythm_ref_conditioning,
    }
    conditioning = build_duration_v3_ref_conditioning(
        sample,
        explicit=sample["rhythm_ref_conditioning"],
    )
    assert set(("global_rate", "operator_coeff")).issubset(conditioning.keys())
    assert "prompt_random_target" in conditioning
    assert "prompt_mask" in conditioning
    assert "prompt_operator_fit" in conditioning
    assert "planner_ref_stats" not in conditioning
    assert "selector_meta_scores" not in conditioning
    assert "ref_phrase_valid" not in conditioning


def test_build_duration_v3_ref_conditioning_accepts_explicit_prompt_units_without_anchor_base():
    sample = {
        "rhythm_ref_conditioning": {
            "prompt_content_units": torch.tensor([[1, 2, 3, 0]], dtype=torch.long),
            "prompt_duration_obs": torch.tensor([[3.0, 4.0, 2.0, 0.0]], dtype=torch.float32),
            "prompt_unit_mask": torch.tensor([[1.0, 1.0, 1.0, 0.0]], dtype=torch.float32),
            "ref_rhythm_stats": torch.randn(1, 6),
            "ref_rhythm_trace": torch.randn(1, 8, 5),
        }
    }
    conditioning = build_duration_v3_ref_conditioning(
        sample,
        explicit=sample["rhythm_ref_conditioning"],
    )
    assert "prompt_content_units" in conditioning
    assert "prompt_duration_obs" in conditioning
    assert "prompt_unit_mask" in conditioning


def test_build_legacy_v2_ref_conditioning_keeps_planner_sidecars():
    sample = {
        "ref_rhythm_stats": torch.randn(1, 6),
        "ref_rhythm_trace": torch.randn(1, 8, 5),
        "planner_ref_stats": torch.randn(1, 2),
        "planner_ref_trace": torch.randn(1, 8, 2),
        "selector_meta_scores": torch.randn(1, 3),
        "ref_phrase_valid": torch.ones(1, 2),
    }
    conditioning = build_legacy_v2_ref_conditioning(sample)
    assert "planner_ref_stats" in conditioning
    assert "planner_ref_trace" in conditioning
    assert "selector_meta_scores" in conditioning
    assert "ref_phrase_valid" in conditioning


def test_build_rhythm_ref_conditioning_dispatches_by_backend():
    rhythm_ref_conditioning = {
        "global_rate": torch.tensor([[0.1]], dtype=torch.float32),
        "operator_coeff": torch.zeros((1, 4)),
        "planner_ref_stats": torch.randn(1, 2),
    }
    sample = {
        "rhythm_ref_conditioning": rhythm_ref_conditioning,
    }
    conditioning = build_rhythm_ref_conditioning(
        sample,
        explicit=sample["rhythm_ref_conditioning"],
        backend="v3",
    )
    assert "global_rate" in conditioning
    assert "planner_ref_stats" not in conditioning


def test_resolve_task_runtime_state_supports_duration_operator_mode():
    runtime_state = resolve_task_runtime_state(
        {
            "rhythm_enable_v2": False,
            "rhythm_mode": "duration_operator",
            "random_speaker_steps": 0,
            "rhythm_optimize_module_only": True,
            "rhythm_fastpath_disable_acoustic_when_module_only": True,
            "rhythm_apply_mode": "never",
        },
        global_step=0,
        infer=False,
        test=False,
        explicit_apply_override=False,
        has_f0=False,
        has_uv=False,
    )
    assert runtime_state.module_only_objective is True
    assert runtime_state.disable_acoustic_train_path is True


def test_resolve_task_runtime_state_keeps_infer_path_out_of_acoustic_short_circuit():
    runtime_state = resolve_task_runtime_state(
        {
            "rhythm_enable_v3": True,
            "random_speaker_steps": 0,
            "rhythm_optimize_module_only": True,
            "rhythm_fastpath_disable_acoustic_when_module_only": True,
            "rhythm_apply_mode": "never",
        },
        global_step=0,
        infer=True,
        test=False,
        explicit_apply_override=False,
        has_f0=False,
        has_uv=False,
    )
    assert runtime_state.module_only_objective is True
    assert runtime_state.disable_acoustic_train_path is False
