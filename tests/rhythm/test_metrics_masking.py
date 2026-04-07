from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.Conan.rhythm.metrics import build_rhythm_metric_dict, build_rhythm_metric_sections


class RhythmMetricMaskingTests(unittest.TestCase):
    def test_masked_factorization_metrics_ignore_padding(self) -> None:
        unit_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
        dur_anchor = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
        speech_exec = torch.tensor([[1.0, 2.0, 100.0, 100.0]], dtype=torch.float32)
        pause_exec = torch.tensor([[0.0, 1.0, 100.0, 100.0]], dtype=torch.float32)
        planner = SimpleNamespace(
            speech_budget_win=torch.tensor([[3.0]], dtype=torch.float32),
            pause_budget_win=torch.tensor([[1.0]], dtype=torch.float32),
            raw_speech_budget_win=torch.tensor([[3.0]], dtype=torch.float32),
            raw_pause_budget_win=torch.tensor([[1.0]], dtype=torch.float32),
            dur_shape_unit=torch.tensor([[1.0, 3.0, 100.0, 100.0]], dtype=torch.float32),
            pause_shape_unit=torch.tensor([[0.25, 0.75, 0.0, 0.0]], dtype=torch.float32),
            boundary_score_unit=torch.tensor([[0.2, 0.4, 100.0, 100.0]], dtype=torch.float32),
            trace_context=torch.zeros((1, 4, 3), dtype=torch.float32),
        )
        execution = SimpleNamespace(
            speech_duration_exec=speech_exec,
            blank_duration_exec=pause_exec,
            pause_after_exec=pause_exec,
            planner=planner,
            commit_frontier=torch.tensor([2], dtype=torch.long),
        )
        output = {
            "rhythm_execution": execution,
            "rhythm_unit_batch": SimpleNamespace(unit_mask=unit_mask, dur_anchor_src=dur_anchor),
            "source_boundary_cue": torch.tensor([[0.1, 0.3, 100.0, 100.0]], dtype=torch.float32),
        }

        metrics = build_rhythm_metric_dict(output)

        self.assertTrue(torch.allclose(metrics["rhythm_metric_dur_shape_abs_mean"], torch.tensor(2.0)))
        self.assertTrue(torch.allclose(metrics["rhythm_metric_boundary_score_mean"], torch.tensor(0.3)))
        self.assertTrue(torch.allclose(metrics["rhythm_metric_prefix_clock_abs_mean"], torch.tensor(1.0)))
        self.assertTrue(torch.allclose(metrics["rhythm_metric_prefix_backlog_mean"], torch.tensor(1.0)))
        self.assertTrue(torch.allclose(metrics["rhythm_metric_source_boundary_mean"], torch.tensor(0.2)))
        self.assertIn("rhythm_metric_pause_shape_entropy_norm", metrics)
        expected_entropy = -(0.25 * math.log(0.25) + 0.75 * math.log(0.75))
        expected_entropy_norm = expected_entropy / math.log(2.0)
        self.assertAlmostEqual(
            float(metrics["rhythm_metric_pause_shape_entropy"].item()),
            expected_entropy,
            places=4,
        )
        self.assertAlmostEqual(
            float(metrics["rhythm_metric_pause_shape_entropy_norm"].item()),
            expected_entropy_norm,
            places=4,
        )

    def test_budget_repair_metrics_expose_raw_vs_exec_gap(self) -> None:
        unit_mask = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        dur_anchor = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        speech_exec = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        pause_exec = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        planner = SimpleNamespace(
            speech_budget_win=torch.tensor([[3.0]], dtype=torch.float32),
            pause_budget_win=torch.tensor([[0.5]], dtype=torch.float32),
            raw_speech_budget_win=torch.tensor([[2.0]], dtype=torch.float32),
            raw_pause_budget_win=torch.tensor([[1.0]], dtype=torch.float32),
            feasible_total_budget_delta=torch.tensor([[0.5]], dtype=torch.float32),
            dur_shape_unit=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
            pause_shape_unit=torch.tensor([[0.5, 0.5]], dtype=torch.float32),
            boundary_score_unit=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
            trace_context=torch.zeros((1, 2, 3), dtype=torch.float32),
        )
        execution = SimpleNamespace(
            speech_duration_exec=speech_exec,
            blank_duration_exec=pause_exec,
            pause_after_exec=pause_exec,
            planner=planner,
            commit_frontier=torch.tensor([2], dtype=torch.long),
        )
        output = {
            "rhythm_execution": execution,
            "rhythm_unit_batch": SimpleNamespace(unit_mask=unit_mask, dur_anchor_src=dur_anchor),
        }

        metrics = build_rhythm_metric_dict(output)

        self.assertTrue(torch.allclose(metrics["rhythm_metric_budget_raw_exec_gap_mean"], torch.tensor(1.5)))
        self.assertTrue(torch.allclose(metrics["rhythm_metric_budget_raw_exec_gap_ratio_mean"], torch.tensor(0.5)))
        self.assertTrue(
            torch.allclose(
                metrics["rhythm_metric_budget_repair_ratio_mean"],
                torch.tensor(0.5 / 3.0),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                metrics["rhythm_metric_budget_projection_redistribution_ratio_mean"],
                torch.tensor(0.5 / 3.0),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                metrics["rhythm_metric_budget_projection_repair_ratio_mean"],
                torch.tensor(1.0 / 3.0),
                atol=1e-6,
            )
        )
        self.assertTrue(torch.allclose(metrics["rhythm_metric_budget_repair_active_rate"], torch.tensor(1.0)))
        self.assertTrue(torch.allclose(metrics["rhythm_metric_budget_projection_repair_active_rate"], torch.tensor(1.0)))

    def test_alias_metrics_export_same_source_kd_diagnostics(self) -> None:
        unit_mask = torch.tensor([[1.0]], dtype=torch.float32)
        dur_anchor = torch.tensor([[1.0]], dtype=torch.float32)
        planner = SimpleNamespace(
            speech_budget_win=torch.tensor([[1.0]], dtype=torch.float32),
            pause_budget_win=torch.tensor([[0.0]], dtype=torch.float32),
            raw_speech_budget_win=torch.tensor([[1.0]], dtype=torch.float32),
            raw_pause_budget_win=torch.tensor([[0.0]], dtype=torch.float32),
            dur_shape_unit=torch.tensor([[0.0]], dtype=torch.float32),
            pause_shape_unit=torch.tensor([[1.0]], dtype=torch.float32),
            boundary_score_unit=torch.tensor([[0.0]], dtype=torch.float32),
            trace_context=torch.zeros((1, 1, 3), dtype=torch.float32),
        )
        execution = SimpleNamespace(
            speech_duration_exec=torch.tensor([[1.0]], dtype=torch.float32),
            blank_duration_exec=torch.tensor([[0.0]], dtype=torch.float32),
            pause_after_exec=torch.tensor([[0.0]], dtype=torch.float32),
            planner=planner,
            commit_frontier=torch.tensor([1], dtype=torch.long),
        )
        metrics = build_rhythm_metric_dict(
            {
                "rhythm_execution": execution,
                "rhythm_unit_batch": SimpleNamespace(unit_mask=unit_mask, dur_anchor_src=dur_anchor),
                "L_kd_same_source": torch.tensor(1.0),
                "L_kd_same_source_exec": torch.tensor(1.0),
                "L_kd_same_source_budget": torch.tensor(0.0),
                "L_kd_same_source_prefix": torch.tensor(1.0),
            }
        )
        self.assertTrue(torch.allclose(metrics["rhythm_metric_alias_L_kd_same_source"], torch.tensor(1.0)))
        self.assertTrue(torch.allclose(metrics["rhythm_metric_alias_L_kd_same_source_exec"], torch.tensor(1.0)))
        self.assertTrue(torch.allclose(metrics["rhythm_metric_alias_L_kd_same_source_budget"], torch.tensor(0.0)))
        self.assertTrue(torch.allclose(metrics["rhythm_metric_alias_L_kd_same_source_prefix"], torch.tensor(1.0)))

    def test_acoustic_target_metrics_expose_length_mismatch_and_alignment_mode(self) -> None:
        unit_mask = torch.tensor([[1.0]], dtype=torch.float32)
        dur_anchor = torch.tensor([[1.0]], dtype=torch.float32)
        planner = SimpleNamespace(
            speech_budget_win=torch.tensor([[1.0]], dtype=torch.float32),
            pause_budget_win=torch.tensor([[0.0]], dtype=torch.float32),
            raw_speech_budget_win=torch.tensor([[1.0]], dtype=torch.float32),
            raw_pause_budget_win=torch.tensor([[0.0]], dtype=torch.float32),
            dur_shape_unit=torch.tensor([[0.0]], dtype=torch.float32),
            pause_shape_unit=torch.tensor([[1.0]], dtype=torch.float32),
            boundary_score_unit=torch.tensor([[0.0]], dtype=torch.float32),
            trace_context=torch.zeros((1, 1, 3), dtype=torch.float32),
        )
        execution = SimpleNamespace(
            speech_duration_exec=torch.tensor([[1.0]], dtype=torch.float32),
            blank_duration_exec=torch.tensor([[0.0]], dtype=torch.float32),
            pause_after_exec=torch.tensor([[0.0]], dtype=torch.float32),
            planner=planner,
            commit_frontier=torch.tensor([1], dtype=torch.long),
        )
        metrics = build_rhythm_metric_dict(
            {
                "rhythm_execution": execution,
                "rhythm_unit_batch": SimpleNamespace(unit_mask=unit_mask, dur_anchor_src=dur_anchor),
                "acoustic_target_is_retimed": True,
                "acoustic_target_source": "cached",
                "acoustic_target_length_frames_before_align": 17.0,
                "acoustic_output_length_frames_before_align": 12.0,
                "acoustic_target_length_delta_before_align": 5.0,
                "acoustic_target_length_mismatch_abs_before_align": 5.0,
                "acoustic_target_length_mismatch_present_before_align": 1.0,
                "acoustic_target_length_mismatch_ratio_before_align": 5.0 / 17.0,
                "acoustic_target_resampled_to_output": 1.0,
                "acoustic_target_trimmed_to_output": 0.0,
                "acoustic_target_length_frames_after_align": 12.0,
                "acoustic_output_length_frames_after_align": 12.0,
            }
        )
        self.assertTrue(
            torch.allclose(
                metrics["rhythm_metric_acoustic_target_length_frames_before_align"],
                torch.tensor(17.0),
            )
        )
        self.assertTrue(
            torch.allclose(
                metrics["rhythm_metric_acoustic_output_length_frames_before_align"],
                torch.tensor(12.0),
            )
        )
        self.assertTrue(
            torch.allclose(
                metrics["rhythm_metric_acoustic_target_length_delta_before_align"],
                torch.tensor(5.0),
            )
        )
        self.assertTrue(
            torch.allclose(
                metrics["rhythm_metric_acoustic_target_length_mismatch_abs_before_align"],
                torch.tensor(5.0),
            )
        )
        self.assertTrue(
            torch.allclose(
                metrics["rhythm_metric_acoustic_target_length_mismatch_present_before_align"],
                torch.tensor(1.0),
            )
        )
        self.assertTrue(
            torch.allclose(
                metrics["rhythm_metric_acoustic_target_length_mismatch_ratio_before_align"],
                torch.tensor(5.0 / 17.0),
            )
        )
        self.assertTrue(
            torch.allclose(
                metrics["rhythm_metric_acoustic_target_resampled_to_output"],
                torch.tensor(1.0),
            )
        )
        self.assertTrue(
            torch.allclose(
                metrics["rhythm_metric_acoustic_target_trimmed_to_output"],
                torch.tensor(0.0),
            )
        )
        self.assertTrue(
            torch.allclose(
                metrics["rhythm_metric_acoustic_target_length_frames_after_align"],
                torch.tensor(12.0),
            )
        )
        self.assertTrue(
            torch.allclose(
                metrics["rhythm_metric_acoustic_output_length_frames_after_align"],
                torch.tensor(12.0),
            )
        )
        self.assertTrue(
            torch.allclose(
                metrics["rhythm_metric_acoustic_target_source_is_cached"],
                torch.tensor(1.0),
            )
        )

    def test_runtime_objective_and_pitch_guard_metrics_are_observable(self) -> None:
        unit_mask = torch.tensor([[1.0]], dtype=torch.float32)
        dur_anchor = torch.tensor([[1.0]], dtype=torch.float32)
        planner = SimpleNamespace(
            speech_budget_win=torch.tensor([[1.0]], dtype=torch.float32),
            pause_budget_win=torch.tensor([[0.0]], dtype=torch.float32),
            raw_speech_budget_win=torch.tensor([[1.0]], dtype=torch.float32),
            raw_pause_budget_win=torch.tensor([[0.0]], dtype=torch.float32),
            dur_shape_unit=torch.tensor([[0.0]], dtype=torch.float32),
            pause_shape_unit=torch.tensor([[1.0]], dtype=torch.float32),
            boundary_score_unit=torch.tensor([[0.0]], dtype=torch.float32),
            trace_context=torch.zeros((1, 1, 3), dtype=torch.float32),
        )
        execution = SimpleNamespace(
            speech_duration_exec=torch.tensor([[1.0]], dtype=torch.float32),
            blank_duration_exec=torch.tensor([[0.0]], dtype=torch.float32),
            pause_after_exec=torch.tensor([[0.0]], dtype=torch.float32),
            planner=planner,
            commit_frontier=torch.tensor([1], dtype=torch.long),
        )
        metrics = build_rhythm_metric_dict(
            {
                "rhythm_execution": execution,
                "rhythm_unit_batch": SimpleNamespace(unit_mask=unit_mask, dur_anchor_src=dur_anchor),
                "disable_acoustic_train_path": 1.0,
                "rhythm_module_only_objective": 1.0,
                "rhythm_skip_acoustic_objective": 1.0,
                "rhythm_pitch_supervision_disabled": 1.0,
                "rhythm_missing_retimed_pitch_target": 1.0,
                "rhythm_stage3_acoustic_loss_scale": 0.5,
                "rhythm_retimed_acoustic_loss_scale": 0.5,
                "rhythm_stage3_pitch_loss_scale": 0.25,
                "rhythm_retimed_pitch_loss_scale": 0.25,
            }
        )
        self.assertTrue(torch.allclose(metrics["rhythm_metric_disable_acoustic_train_path"], torch.tensor(1.0)))
        self.assertTrue(torch.allclose(metrics["rhythm_metric_module_only_objective"], torch.tensor(1.0)))
        self.assertTrue(torch.allclose(metrics["rhythm_metric_skip_acoustic_objective"], torch.tensor(1.0)))
        self.assertTrue(torch.allclose(metrics["rhythm_metric_pitch_supervision_disabled"], torch.tensor(1.0)))
        self.assertTrue(torch.allclose(metrics["rhythm_metric_missing_retimed_pitch_target"], torch.tensor(1.0)))
        self.assertTrue(torch.allclose(metrics["rhythm_metric_stage3_acoustic_loss_scale"], torch.tensor(0.5)))
        self.assertTrue(torch.allclose(metrics["rhythm_metric_retimed_acoustic_loss_scale"], torch.tensor(0.5)))
        self.assertTrue(torch.allclose(metrics["rhythm_metric_stage3_pitch_loss_scale"], torch.tensor(0.25)))
        self.assertTrue(torch.allclose(metrics["rhythm_metric_retimed_pitch_loss_scale"], torch.tensor(0.25)))

    def test_metric_sections_flatten_without_loss(self) -> None:
        unit_mask = torch.tensor([[1.0]], dtype=torch.float32)
        dur_anchor = torch.tensor([[1.0]], dtype=torch.float32)
        planner = SimpleNamespace(
            speech_budget_win=torch.tensor([[1.0]], dtype=torch.float32),
            pause_budget_win=torch.tensor([[0.0]], dtype=torch.float32),
            raw_speech_budget_win=torch.tensor([[1.0]], dtype=torch.float32),
            raw_pause_budget_win=torch.tensor([[0.0]], dtype=torch.float32),
            dur_shape_unit=torch.tensor([[0.0]], dtype=torch.float32),
            pause_shape_unit=torch.tensor([[1.0]], dtype=torch.float32),
            boundary_score_unit=torch.tensor([[0.0]], dtype=torch.float32),
            trace_context=torch.zeros((1, 1, 3), dtype=torch.float32),
        )
        execution = SimpleNamespace(
            speech_duration_exec=torch.tensor([[1.0]], dtype=torch.float32),
            blank_duration_exec=torch.tensor([[0.0]], dtype=torch.float32),
            pause_after_exec=torch.tensor([[0.0]], dtype=torch.float32),
            planner=planner,
            commit_frontier=torch.tensor([1], dtype=torch.long),
        )
        output = {
            "rhythm_execution": execution,
            "rhythm_unit_batch": SimpleNamespace(unit_mask=unit_mask, dur_anchor_src=dur_anchor),
            "L_kd_same_source": torch.tensor(1.0),
        }

        sections = build_rhythm_metric_sections(output)
        flattened = build_rhythm_metric_dict(output)

        self.assertEqual(
            list(sections.keys()),
            ["plan_surfaces", "runtime_state", "teacher_targets", "observability"],
        )
        merged: dict[str, torch.Tensor] = {}
        for section_metrics in sections.values():
            merged.update(section_metrics)
        self.assertEqual(set(flattened.keys()), set(merged.keys()))
        for key, expected in merged.items():
            self.assertTrue(torch.allclose(flattened[key], expected), key)


if __name__ == "__main__":
    unittest.main()
