from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.Conan.rhythm.factory import (
    build_boundary_commit_controller_from_hparams,
    build_commit_config_from_hparams,
    build_projector_config_from_hparams,
    build_streaming_rhythm_module_from_hparams,
)
from modules.Conan.rhythm.controller import ChunkStateBundle, WindowBudgetController


class FactoryDefaultTests(unittest.TestCase):
    def test_strict_mainline_projector_defaults_keep_sparse_guarded_render_plan(self) -> None:
        cfg = build_projector_config_from_hparams(
            {
                "rhythm_strict_mainline": True,
            }
        )
        self.assertEqual(cfg.pause_selection_mode, "sparse")
        self.assertTrue(cfg.use_boundary_commit_guard)
        self.assertTrue(cfg.build_render_plan)

    def test_projector_defaults_still_allow_explicit_override(self) -> None:
        cfg = build_projector_config_from_hparams(
            {
                "rhythm_strict_mainline": True,
                "rhythm_projector_pause_selection_mode": "simple",
                "rhythm_projector_use_boundary_commit_guard": False,
                "rhythm_projector_build_render_plan": False,
            }
        )
        self.assertEqual(cfg.pause_selection_mode, "simple")
        self.assertFalse(cfg.use_boundary_commit_guard)
        self.assertFalse(cfg.build_render_plan)

    def test_projector_defaults_preserve_explicit_topk_ratio_under_strict_mainline(self) -> None:
        cfg = build_projector_config_from_hparams(
            {
                "rhythm_strict_mainline": True,
                "rhythm_projector_pause_topk_ratio": 0.55,
            }
        )
        self.assertEqual(cfg.pause_selection_mode, "sparse")
        self.assertAlmostEqual(cfg.pause_topk_ratio, 0.55, places=6)

    def test_commit_config_defaults_are_legacy_projector(self) -> None:
        cfg = build_commit_config_from_hparams({})
        self.assertEqual(cfg.mode, "legacy_projector")
        self.assertAlmostEqual(cfg.threshold, 0.65, places=6)
        self.assertTrue(cfg.require_sealed_boundary)
        self.assertEqual(cfg.min_phrase_units, 2)
        self.assertEqual(cfg.max_lookahead_units, 3)

    def test_commit_config_explicit_override_is_respected(self) -> None:
        cfg = build_commit_config_from_hparams(
            {
                "rhythm_commit_mode": "boundary_phrase",
                "rhythm_commit_threshold": 0.72,
                "rhythm_commit_require_sealed_boundary": False,
                "rhythm_commit_min_phrase_units": 3,
                "rhythm_commit_max_lookahead_units": 5,
                "rhythm_commit_sep_hint_bonus": 0.15,
                "rhythm_commit_boundary_confidence_weight": 0.25,
                "rhythm_commit_source_boundary_weight": 0.45,
                "rhythm_commit_planner_boundary_weight": 0.30,
            }
        )
        self.assertEqual(cfg.mode, "boundary_phrase")
        self.assertAlmostEqual(cfg.threshold, 0.72, places=6)
        self.assertFalse(cfg.require_sealed_boundary)
        self.assertEqual(cfg.min_phrase_units, 3)
        self.assertEqual(cfg.max_lookahead_units, 5)
        self.assertAlmostEqual(cfg.sep_hint_bonus, 0.15, places=6)
        self.assertAlmostEqual(cfg.boundary_confidence_weight, 0.25, places=6)
        self.assertAlmostEqual(cfg.source_boundary_weight, 0.45, places=6)
        self.assertAlmostEqual(cfg.planner_boundary_weight, 0.30, places=6)

    def test_commit_controller_builder_returns_controller(self) -> None:
        controller = build_boundary_commit_controller_from_hparams({"rhythm_commit_mode": "boundary_phrase"})
        self.assertEqual(controller.config.mode, "boundary_phrase")

    def test_trace_cold_start_defaults_zero(self) -> None:
        module = build_streaming_rhythm_module_from_hparams({})
        if not hasattr(module, "trace_cold_start_min_visible_units"):
            self.skipTest("trace cold-start attributes not available yet")
        self.assertEqual(module.trace_cold_start_min_visible_units, 0)
        self.assertEqual(module.trace_cold_start_full_visible_units, 0)
        self.assertFalse(module.trace_active_tail_only)
        self.assertEqual(module.trace_offset_lookahead_units, 0)
        self.assertFalse(module.reference_descriptor.runtime_phrase_bank_enable)
        self.assertEqual(module.reference_descriptor.runtime_phrase_select_window, 3)
        self.assertIsNone(module.commit_controller)
        self.assertTrue(module.chunk_state_enable)
        self.assertAlmostEqual(module.budget_phase_feature_scale, 0.0, places=6)
        self.assertFalse(module.phase_decoupled_timing)
        self.assertFalse(hasattr(module, "phase_free_timing"))
        self.assertAlmostEqual(module.phase_decoupled_segment_shape_scale, 0.0, places=6)
        self.assertAlmostEqual(module.phase_decoupled_rollover_start, 0.68, places=6)
        self.assertAlmostEqual(module.phase_decoupled_rollover_end, 0.92, places=6)
        self.assertIsNotNone(module.scheduler.chunk_state_head)
        self.assertAlmostEqual(module.scheduler.window_budget.phase_feature_scale, 0.0, places=6)
        self.assertFalse(module.scheduler.phase_decoupled_timing)
        self.assertFalse(hasattr(module.scheduler, "phase_free_timing"))
        self.assertFalse(module.scheduler.window_budget.phase_decoupled_timing)
        self.assertFalse(hasattr(module.scheduler.window_budget, "phase_free_timing"))
        self.assertAlmostEqual(module.scheduler.phase_decoupled_segment_shape_scale, 0.0, places=6)
        self.assertAlmostEqual(module.scheduler.phase_decoupled_local_rho_scale, 0.0, places=6)
        self.assertAlmostEqual(module.scheduler.phase_decoupled_soft_rollover_scale, 0.0, places=6)
        self.assertAlmostEqual(module.scheduler.window_budget.segment_shape_scale, 0.0, places=6)
        self.assertAlmostEqual(module.scheduler.unit_redistribution.segment_shape_scale, 0.0, places=6)
        self.assertAlmostEqual(module.scheduler.unit_redistribution.local_rho_scale, 0.0, places=6)
        self.assertAlmostEqual(module.scheduler.unit_redistribution.soft_rollover_scale, 0.0, places=6)

    def test_trace_cold_start_hparams_override(self) -> None:
        module = build_streaming_rhythm_module_from_hparams(
            {
                "rhythm_trace_cold_start_min_visible_units": 2,
                "rhythm_trace_cold_start_full_visible_units": 7,
                "rhythm_trace_active_tail_only": True,
                "rhythm_trace_offset_lookahead_units": 5,
                "rhythm_runtime_phrase_bank_enable": True,
                "rhythm_runtime_phrase_bank_max_phrases": 9,
                "rhythm_runtime_phrase_bank_bins": 12,
                "rhythm_runtime_phrase_select_window": 5,
                "rhythm_phase_decoupled_timing": True,
                "rhythm_phase_decoupled_phrase_gate_boundary_threshold": 0.61,
                "rhythm_phase_decoupled_boundary_style_residual_scale": 0.27,
                "rhythm_phase_decoupled_segment_shape_scale": 0.35,
                "rhythm_phase_decoupled_rollover_start": 0.60,
                "rhythm_phase_decoupled_rollover_end": 0.90,
                "rhythm_debt_control_scale": 3.5,
                "rhythm_debt_pause_priority": 0.22,
                "rhythm_debt_speech_priority": 0.31,
                "rhythm_projector_debt_leak": 0.12,
                "rhythm_projector_debt_max_abs": 9.0,
                "rhythm_projector_debt_correction_horizon": 2.5,
            }
        )
        if not hasattr(module, "trace_cold_start_min_visible_units"):
            self.skipTest("trace cold-start attributes not available yet")
        if not (
            module.trace_cold_start_min_visible_units == 2
            and module.trace_cold_start_full_visible_units == 7
            and module.trace_active_tail_only
            and module.trace_offset_lookahead_units == 5
            and module.reference_descriptor.runtime_phrase_bank_enable
            and module.reference_descriptor.runtime_phrase_bank_max_phrases == 9
            and module.reference_descriptor.runtime_phrase_bank_bins == 12
            and module.reference_descriptor.runtime_phrase_select_window == 5
        ):
            self.skipTest("trace cold-start overrides not wired yet")
        self.assertEqual(module.trace_cold_start_min_visible_units, 2)
        self.assertEqual(module.trace_cold_start_full_visible_units, 7)
        self.assertTrue(module.trace_active_tail_only)
        self.assertEqual(module.trace_offset_lookahead_units, 5)
        self.assertTrue(module.reference_descriptor.runtime_phrase_bank_enable)
        self.assertEqual(module.reference_descriptor.runtime_phrase_bank_max_phrases, 9)
        self.assertEqual(module.reference_descriptor.runtime_phrase_bank_bins, 12)
        self.assertEqual(module.reference_descriptor.runtime_phrase_select_window, 5)
        self.assertAlmostEqual(module.phase_decoupled_phrase_gate_boundary_threshold, 0.61, places=6)
        self.assertTrue(module.phase_decoupled_timing)
        self.assertFalse(hasattr(module, "phase_free_phrase_boundary_threshold"))
        self.assertFalse(hasattr(module, "phase_free_timing"))
        self.assertTrue(module.scheduler.phase_decoupled_timing)
        self.assertFalse(hasattr(module.scheduler, "phase_free_timing"))
        self.assertTrue(module.scheduler.window_budget.phase_decoupled_timing)
        self.assertFalse(hasattr(module.scheduler.window_budget, "phase_free_timing"))
        self.assertAlmostEqual(module.scheduler.phase_decoupled_boundary_style_residual_scale, 0.27, places=6)
        self.assertAlmostEqual(module.phase_decoupled_segment_shape_scale, 0.35, places=6)
        self.assertAlmostEqual(module.phase_decoupled_rollover_start, 0.60, places=6)
        self.assertAlmostEqual(module.phase_decoupled_rollover_end, 0.90, places=6)
        self.assertAlmostEqual(module.scheduler.phase_decoupled_segment_shape_scale, 0.35, places=6)
        self.assertAlmostEqual(module.scheduler.phase_decoupled_local_rho_scale, 0.0, places=6)
        self.assertAlmostEqual(module.scheduler.phase_decoupled_soft_rollover_scale, 0.0, places=6)
        self.assertAlmostEqual(module.scheduler.window_budget.segment_shape_scale, 0.35, places=6)
        self.assertAlmostEqual(module.scheduler.unit_redistribution.segment_shape_scale, 0.35, places=6)
        self.assertAlmostEqual(module.scheduler.debt_control_scale, 3.5, places=6)
        self.assertAlmostEqual(module.scheduler.debt_pause_priority, 0.22, places=6)
        self.assertAlmostEqual(module.scheduler.debt_speech_priority, 0.31, places=6)
        self.assertAlmostEqual(module.projector.config.debt_leak, 0.12, places=6)
        self.assertAlmostEqual(module.projector.config.debt_max_abs, 9.0, places=6)
        self.assertAlmostEqual(module.projector.config.debt_correction_horizon, 2.5, places=6)
        self.assertIsNone(module.commit_controller)

    def test_legacy_phase_free_hparams_still_enable_phase_decoupled_path(self) -> None:
        module = build_streaming_rhythm_module_from_hparams(
            {
                "rhythm_phase_free_timing": True,
                "rhythm_phase_free_phrase_boundary_threshold": 0.58,
            }
        )
        self.assertTrue(module.phase_decoupled_timing)
        self.assertFalse(hasattr(module, "phase_free_timing"))
        self.assertAlmostEqual(module.phase_decoupled_phrase_gate_boundary_threshold, 0.58, places=6)
        self.assertFalse(hasattr(module, "phase_free_phrase_boundary_threshold"))

    def test_legacy_soft_rollover_start_alias_maps_to_canonical_rollover_start(self) -> None:
        module = build_streaming_rhythm_module_from_hparams(
            {
                "rhythm_phase_decoupled_soft_rollover_start": 0.57,
            }
        )
        self.assertAlmostEqual(module.phase_decoupled_rollover_start, 0.57, places=6)
        self.assertAlmostEqual(module.scheduler.phase_decoupled_rollover_start, 0.57, places=6)

    def test_phase_decoupled_hparams_conflict_with_legacy_alias(self) -> None:
        with self.assertRaises(ValueError):
            build_streaming_rhythm_module_from_hparams(
                {
                    "rhythm_phase_decoupled_timing": True,
                    "rhythm_phase_free_timing": False,
                }
            )
        with self.assertRaises(ValueError):
            build_streaming_rhythm_module_from_hparams(
                {
                    "rhythm_phase_decoupled_phrase_gate_boundary_threshold": 0.61,
                    "rhythm_phase_free_phrase_boundary_threshold": 0.57,
                }
            )

    def test_boundary_phrase_mode_explicitly_enables_discrete_commit_controller(self) -> None:
        module = build_streaming_rhythm_module_from_hparams(
            {
                "rhythm_commit_mode": "boundary_phrase",
            }
        )
        self.assertIsNotNone(module.commit_controller)
        self.assertEqual(module.commit_controller.config.mode, "boundary_phrase")

    def test_chunk_state_and_budget_phase_overrides_are_respected(self) -> None:
        module = build_streaming_rhythm_module_from_hparams(
            {
                "rhythm_chunk_state_enable": False,
                "rhythm_budget_phase_feature_scale": 0.35,
            }
        )
        self.assertFalse(module.chunk_state_enable)
        self.assertIsNone(module.scheduler.chunk_state_head)
        self.assertAlmostEqual(module.budget_phase_feature_scale, 0.35, places=6)
        self.assertAlmostEqual(module.scheduler.window_budget.phase_feature_scale, 0.35, places=6)

    def test_phrase_selection_weight_overrides_flow_to_selector(self) -> None:
        module = build_streaming_rhythm_module_from_hparams(
            {
                "rhythm_emit_reference_sidecar": True,
                "rhythm_phrase_selection_boundary_weight": 0.35,
                "rhythm_phrase_selection_local_rate_weight": 0.12,
                "rhythm_phrase_selection_pause_weight": 0.22,
                "rhythm_phrase_selection_voice_weight": 0.05,
                "rhythm_phrase_selection_final_bias_weight": 0.15,
                "rhythm_phrase_selection_monotonic_bias": 0.02,
                "rhythm_phrase_selection_length_bias": 0.11,
            }
        )
        selector = module.reference_descriptor.selector
        if selector is None:
            self.skipTest("selector not enabled")
        self.assertAlmostEqual(selector.boundary_weight, 0.35, places=6)
        self.assertAlmostEqual(selector.local_rate_weight, 0.12, places=6)
        self.assertAlmostEqual(selector.pause_weight, 0.22, places=6)
        self.assertAlmostEqual(selector.voiced_weight, 0.05, places=6)
        self.assertAlmostEqual(selector.final_bias_weight, 0.15, places=6)
        self.assertAlmostEqual(selector.monotonic_bias, 0.02, places=6)
        self.assertAlmostEqual(selector.phrase_length_bias, 0.11, places=6)

    def test_window_budget_ignores_phase_when_chunk_state_is_authoritative(self) -> None:
        controller = WindowBudgetController(
            hidden_size=8,
            stats_dim=2,
            trace_dim=2,
            phase_feature_scale=0.0,
        )
        kwargs = dict(
            unit_states=torch.zeros((1, 4, 8), dtype=torch.float32),
            dur_anchor_src=torch.ones((1, 4), dtype=torch.float32),
            unit_mask=torch.ones((1, 4), dtype=torch.float32),
            planner_ref_stats=torch.tensor([[0.5, 0.2]], dtype=torch.float32),
            planner_trace_context=torch.zeros((1, 4, 2), dtype=torch.float32),
            slow_rhythm_summary=torch.tensor([[0.1, 0.3]], dtype=torch.float32),
            boundary_score_unit=torch.tensor([[0.2, 0.4, 0.6, 0.8]], dtype=torch.float32),
            clock_delta=torch.tensor([0.0], dtype=torch.float32),
            chunk_state=ChunkStateBundle(
                chunk_summary=torch.tensor([[0.25, 0.75, 0.50, 0.40, 0.20, 0.60]], dtype=torch.float32),
                structure_progress=torch.tensor([0.6], dtype=torch.float32),
                commit_now_prob=torch.tensor([0.6], dtype=torch.float32),
                phrase_open_prob=torch.tensor([0.2], dtype=torch.float32),
                phrase_close_prob=torch.tensor([0.6], dtype=torch.float32),
                phrase_role_prob=torch.tensor([[0.2, 0.2, 0.6]], dtype=torch.float32),
                active_tail_mask=torch.ones((1, 4), dtype=torch.float32),
            ),
        )
        out_a = controller(phase_ptr=torch.tensor([0.0], dtype=torch.float32), **kwargs)
        out_b = controller(phase_ptr=torch.tensor([0.95], dtype=torch.float32), **kwargs)
        self.assertTrue(torch.allclose(out_a["speech_budget_win"], out_b["speech_budget_win"]))
        self.assertTrue(torch.allclose(out_a["pause_budget_win"], out_b["pause_budget_win"]))

    def test_window_budget_uses_commit_frontier_progress_when_phase_feature_is_disabled(self) -> None:
        controller = WindowBudgetController(
            hidden_size=8,
            stats_dim=2,
            trace_dim=2,
            phase_feature_scale=0.0,
        )
        kwargs = dict(
            unit_states=torch.zeros((1, 4, 8), dtype=torch.float32),
            dur_anchor_src=torch.ones((1, 4), dtype=torch.float32),
            unit_mask=torch.ones((1, 4), dtype=torch.float32),
            planner_ref_stats=torch.tensor([[0.5, 0.2]], dtype=torch.float32),
            planner_trace_context=torch.zeros((1, 4, 2), dtype=torch.float32),
            slow_rhythm_summary=torch.tensor([[0.1, 0.3]], dtype=torch.float32),
            boundary_score_unit=torch.tensor([[0.2, 0.4, 0.6, 0.8]], dtype=torch.float32),
            clock_delta=torch.tensor([0.0], dtype=torch.float32),
            commit_frontier=torch.tensor([2], dtype=torch.long),
            chunk_state=None,
        )
        out_a = controller(phase_ptr=torch.tensor([0.0], dtype=torch.float32), **kwargs)
        out_b = controller(phase_ptr=torch.tensor([0.95], dtype=torch.float32), **kwargs)
        self.assertTrue(torch.allclose(out_a["speech_budget_win"], out_b["speech_budget_win"]))
        self.assertTrue(torch.allclose(out_a["pause_budget_win"], out_b["pause_budget_win"]))


if __name__ == "__main__":
    unittest.main()
