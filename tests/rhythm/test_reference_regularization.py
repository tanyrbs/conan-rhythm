from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.Conan.rhythm.reference_regularization import (
    CompactReferenceDescriptor,
    _compute_group_gap_scales,
    build_predicted_compact_reference_descriptor,
    build_target_compact_reference_descriptor,
    compute_descriptor_consistency_loss,
    compute_group_reference_contrastive_loss,
)


class ReferenceRegularizationTests(unittest.TestCase):
    @staticmethod
    def _build_output(boundary_score=None, blank_duration=None):
        planner = type(
            "Planner",
            (),
            {
                "boundary_score_unit": (
                    torch.tensor([[0.0, 1.0, 0.2]], dtype=torch.float32)
                    if boundary_score is None
                    else torch.tensor(boundary_score, dtype=torch.float32)
                ),
            },
        )()
        blank_duration = (
            torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
            if blank_duration is None
            else torch.tensor(blank_duration, dtype=torch.float32)
        )
        execution = type(
            "Execution",
            (),
            {
                "speech_duration_exec": torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
                "blank_duration_exec": blank_duration,
                "pause_after_exec": blank_duration,
                "planner": planner,
            },
        )()
        unit_batch = type(
            "UnitBatch",
            (),
            {
                "unit_mask": torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
                "dur_anchor_src": torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
            },
        )()
        return {"rhythm_execution": execution, "rhythm_unit_batch": unit_batch}

    def test_descriptor_consistency_is_near_zero_when_target_matches_prediction(self) -> None:
        output = self._build_output()
        pred = build_predicted_compact_reference_descriptor(output, trace_bins=6)
        self.assertIsNotNone(pred)
        ref_stats = torch.zeros((1, 6), dtype=torch.float32)
        ref_stats[:, 0:1] = pred.pause_ratio
        ref_stats[:, 2:3] = torch.reciprocal(pred.global_rate.clamp_min(1.0e-6))
        ref_trace = torch.zeros((1, 6, 5), dtype=torch.float32)
        ref_trace[:, :, 1:2] = pred.local_rate_trace
        ref_trace[:, :, 2:3] = pred.boundary_trace
        target = build_target_compact_reference_descriptor(
            {"ref_rhythm_stats": ref_stats, "ref_rhythm_trace": ref_trace}
        )
        losses = compute_descriptor_consistency_loss(pred, target)
        self.assertLess(float(losses["stats"]), 1.0e-6)
        self.assertLess(float(losses["local_trace"]), 1.0e-6)
        self.assertLess(float(losses["boundary_trace"]), 1.0e-6)

    def test_predicted_boundary_descriptor_ignores_planner_boundary_score_shortcut(self) -> None:
        output_a = self._build_output(
            boundary_score=[[0.0, 1.0, 0.2]],
            blank_duration=[[0.0, 1.0, 0.0]],
        )
        output_b = self._build_output(
            boundary_score=[[1.0, 0.0, 1.0]],
            blank_duration=[[0.0, 1.0, 0.0]],
        )

        pred_a = build_predicted_compact_reference_descriptor(output_a, trace_bins=6)
        pred_b = build_predicted_compact_reference_descriptor(output_b, trace_bins=6)

        self.assertIsNotNone(pred_a)
        self.assertIsNotNone(pred_b)
        self.assertTrue(torch.allclose(pred_a.boundary_trace, pred_b.boundary_trace))

    def test_predicted_boundary_descriptor_tracks_executed_pause_structure(self) -> None:
        output = self._build_output(
            boundary_score=[[0.0, 0.0, 0.0]],
            blank_duration=[[0.0, 2.5, 0.0]],
        )
        pred = build_predicted_compact_reference_descriptor(output, trace_bins=6)

        self.assertIsNotNone(pred)
        self.assertGreater(float(pred.boundary_trace.max().item()), 0.5)
        self.assertLess(float(pred.boundary_trace[:, 0].max().item()), 0.2)
        self.assertLess(float(pred.boundary_trace[:, -1].max().item()), 0.2)

    def test_group_contrastive_prefers_matching_reference(self) -> None:
        pred = CompactReferenceDescriptor(
            global_rate=torch.tensor([[1.0], [3.0]], dtype=torch.float32),
            pause_ratio=torch.tensor([[0.1], [0.4]], dtype=torch.float32),
            local_rate_trace=torch.tensor(
                [[[0.0], [0.2]], [[1.0], [0.8]]],
                dtype=torch.float32,
            ),
            boundary_trace=torch.tensor(
                [[[0.0], [1.0]], [[1.0], [0.0]]],
                dtype=torch.float32,
            ),
        )
        target = CompactReferenceDescriptor(
            global_rate=pred.global_rate.clone(),
            pause_ratio=pred.pause_ratio.clone(),
            local_rate_trace=pred.local_rate_trace.clone(),
            boundary_trace=pred.boundary_trace.clone(),
        )
        swapped = CompactReferenceDescriptor(
            global_rate=target.global_rate.flip(0),
            pause_ratio=target.pause_ratio.flip(0),
            local_rate_trace=target.local_rate_trace.flip(0),
            boundary_trace=target.boundary_trace.flip(0),
        )
        group_ids = torch.tensor([[0], [0]], dtype=torch.long)
        loss_match = compute_group_reference_contrastive_loss(pred, target, group_ids, temperature=0.05)
        loss_swap = compute_group_reference_contrastive_loss(pred, swapped, group_ids, temperature=0.05)
        self.assertIsNotNone(loss_match)
        self.assertIsNotNone(loss_swap)
        self.assertLess(float(loss_match), float(loss_swap))

    def test_group_gap_scale_is_larger_for_more_separated_targets(self) -> None:
        near = torch.tensor(
            [[1.0, 0.0], [0.98, 0.02], [0.97, 0.03]],
            dtype=torch.float32,
        )
        near = torch.nn.functional.normalize(near, dim=-1)
        far = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            dtype=torch.float32,
        )
        far = torch.nn.functional.normalize(far, dim=-1)

        near_scale = _compute_group_gap_scales(near, gap_floor=0.10, min_scale=0.50, gap_power=1.0)
        far_scale = _compute_group_gap_scales(far, gap_floor=0.10, min_scale=0.50, gap_power=1.0)

        self.assertLess(float(near_scale.mean()), float(far_scale.mean()))
        self.assertGreaterEqual(float(near_scale.min()), 0.50)
        self.assertLessEqual(float(far_scale.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
