from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.Conan.rhythm.frame_plan import (
    build_frame_plan,
    build_frame_plan_from_execution,
    sample_tensor_by_frame_plan,
)


class RhythmFramePlanTests(unittest.TestCase):
    def test_zero_duration_speech_slot_advances_source_cursor(self) -> None:
        plan = build_frame_plan_from_execution(
            dur_anchor_src=torch.tensor([[2.0, 3.0]], dtype=torch.float32),
            speech_exec=torch.tensor([[0.0, 3.0]], dtype=torch.float32),
            pause_exec=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
            unit_mask=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        )
        self.assertEqual(plan.frame_src_index[0, :3].tolist(), [2, 3, 4])

    def test_sample_tensor_by_frame_plan_supports_batched_gather_and_blank_fill(self) -> None:
        plan = build_frame_plan_from_execution(
            dur_anchor_src=torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=torch.float32),
            speech_exec=torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=torch.float32),
            pause_exec=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
            unit_mask=torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32),
        )
        source = torch.tensor(
            [
                [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
                [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]],
            ],
            dtype=torch.float32,
        )
        blank_fill = torch.tensor([[0.5, 0.25], [0.75, 0.5]], dtype=torch.float32)

        gathered = sample_tensor_by_frame_plan(source, plan, blank_fill=blank_fill)

        self.assertEqual(tuple(gathered.shape), (2, plan.frame_src_index.size(1), 2))
        self.assertTrue(torch.allclose(gathered[0, 0], torch.tensor([1.0, 10.0])))
        self.assertTrue(torch.allclose(gathered[0, 2], torch.tensor([0.5, 0.25])))
        self.assertTrue(torch.allclose(gathered[1, 0], torch.tensor([4.0, 40.0])))
        self.assertTrue(torch.allclose(gathered[1, 3], torch.tensor([0.75, 0.5])))

    def test_build_frame_plan_uses_explicit_active_slot_positions(self) -> None:
        plan = build_frame_plan(
            dur_anchor_src=torch.tensor([[2.0, 2.0]], dtype=torch.float32),
            slot_duration_exec=torch.tensor([[2.0, 5.0, 2.0, 1.0]], dtype=torch.float32),
            slot_mask=torch.tensor([[1.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
            slot_is_blank=torch.tensor([[0, 1, 0, 1]], dtype=torch.long),
            slot_unit_index=torch.tensor([[0, 0, 1, 1]], dtype=torch.long),
        )

        self.assertEqual(plan.frame_src_index[0, :5].tolist(), [0, 1, 2, 3, -1])
        self.assertEqual(plan.frame_is_blank[0, :5].tolist(), [0, 0, 0, 0, 1])
        self.assertEqual(plan.frame_slot_index[0, :5].tolist(), [0, 0, 2, 2, 3])
        self.assertEqual(plan.total_mask[0, :5].tolist(), [1.0, 1.0, 1.0, 1.0, 1.0])
        self.assertTrue(torch.all(plan.total_mask[0, 5:] == 0.0))

    def test_fractional_slot_rounding_preserves_group_totals(self) -> None:
        plan = build_frame_plan_from_execution(
            dur_anchor_src=torch.tensor([[0.4, 0.4, 0.4]], dtype=torch.float32),
            speech_exec=torch.tensor([[0.4, 0.4, 0.4]], dtype=torch.float32),
            pause_exec=torch.tensor([[0.4, 0.4, 0.4]], dtype=torch.float32),
            unit_mask=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
        )
        self.assertEqual(int(plan.total_mask[0].sum().item()), 2)
        self.assertEqual(int(plan.speech_mask[0].sum().item()), 1)
        self.assertEqual(int(plan.blank_mask[0].sum().item()), 1)

    def test_fractional_anchor_rounding_preserves_source_mass(self) -> None:
        plan = build_frame_plan_from_execution(
            dur_anchor_src=torch.tensor([[0.4, 0.4, 0.4]], dtype=torch.float32),
            speech_exec=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
            pause_exec=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
            unit_mask=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
        )
        speech_indices = plan.frame_src_index[0, plan.speech_mask[0] > 0.5]
        self.assertEqual(speech_indices.tolist(), [0])


if __name__ == "__main__":
    unittest.main()
