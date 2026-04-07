from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.Conan.rhythm.reference_descriptor import RefRhythmDescriptor
from modules.Conan.rhythm.reference_encoder import _resolve_progress_from_speech_mask
from tasks.Conan.rhythm.reference_regularization import (
    build_predicted_compact_reference_descriptor,
    build_target_compact_reference_descriptor,
)


class ReferenceDescriptorEdgeCaseTests(unittest.TestCase):
    def test_from_stats_trace_maps_zero_mean_speech_to_zero_global_rate(self) -> None:
        ref_stats = torch.tensor(
            [[1.0, 4.0, 0.0, 0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        ref_trace = torch.zeros((1, 8, 5), dtype=torch.float32)

        descriptor = RefRhythmDescriptor.from_stats_trace(ref_stats, ref_trace)

        self.assertEqual(float(descriptor["global_rate"].item()), 0.0)
        self.assertEqual(float(descriptor["planner_ref_stats"][0, 0].item()), 0.0)

    def test_target_compact_descriptor_sanitizes_stale_global_rate_when_stats_have_no_speech(self) -> None:
        ref_stats = torch.tensor(
            [[1.0, 3.0, 0.0, 0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        ref_trace = torch.zeros((1, 6, 5), dtype=torch.float32)

        target = build_target_compact_reference_descriptor(
            {
                "ref_rhythm_stats": ref_stats,
                "ref_rhythm_trace": ref_trace,
                "global_rate": torch.ones((1, 1), dtype=torch.float32),
            }
        )

        self.assertIsNotNone(target)
        self.assertEqual(float(target.global_rate.item()), 0.0)

    def test_predicted_compact_descriptor_zero_speech_keeps_global_rate_zero(self) -> None:
        planner = type(
            "Planner",
            (),
            {
                "boundary_score_unit": torch.tensor([[0.2, 0.8, 0.6]], dtype=torch.float32),
            },
        )()
        execution = type(
            "Execution",
            (),
            {
                "speech_duration_exec": torch.zeros((1, 3), dtype=torch.float32),
                "blank_duration_exec": torch.tensor([[1.0, 2.0, 1.0]], dtype=torch.float32),
                "pause_after_exec": torch.tensor([[1.0, 2.0, 1.0]], dtype=torch.float32),
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

        pred = build_predicted_compact_reference_descriptor(
            {"rhythm_execution": execution, "rhythm_unit_batch": unit_batch},
            trace_bins=6,
        )

        self.assertIsNotNone(pred)
        self.assertEqual(float(pred.global_rate.item()), 0.0)
        self.assertEqual(float(pred.pause_ratio.item()), 1.0)
        self.assertTrue(torch.isfinite(pred.local_rate_trace).all())
        self.assertTrue(torch.isfinite(pred.boundary_trace).all())

    def test_zero_speech_progress_falls_back_to_uniform(self) -> None:
        speech_mask = torch.zeros((2, 5), dtype=torch.bool)

        progress, uniform = _resolve_progress_from_speech_mask(speech_mask)

        self.assertTrue(torch.allclose(progress, uniform))
        self.assertTrue(
            torch.allclose(
                progress,
                torch.tensor(
                    [
                        [0.0, 0.25, 0.5, 0.75, 1.0],
                        [0.0, 0.25, 0.5, 0.75, 1.0],
                    ],
                    dtype=torch.float32,
                ),
            )
        )


if __name__ == "__main__":
    unittest.main()
