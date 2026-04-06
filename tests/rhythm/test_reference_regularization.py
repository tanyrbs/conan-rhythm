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
    build_predicted_compact_reference_descriptor,
    build_target_compact_reference_descriptor,
    compute_descriptor_consistency_loss,
    compute_group_reference_contrastive_loss,
)


class ReferenceRegularizationTests(unittest.TestCase):
    @staticmethod
    def _build_output():
        planner = type(
            "Planner",
            (),
            {
                "boundary_score_unit": torch.tensor([[0.0, 1.0, 0.2]], dtype=torch.float32),
            },
        )()
        execution = type(
            "Execution",
            (),
            {
                "speech_duration_exec": torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
                "blank_duration_exec": torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32),
                "pause_after_exec": torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32),
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


if __name__ == "__main__":
    unittest.main()
