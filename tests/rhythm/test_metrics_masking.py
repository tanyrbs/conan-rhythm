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

from tasks.Conan.rhythm.metrics import build_rhythm_metric_dict


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
        self.assertTrue(torch.allclose(metrics["rhythm_metric_budget_repair_active_rate"], torch.tensor(1.0)))


if __name__ == "__main__":
    unittest.main()
