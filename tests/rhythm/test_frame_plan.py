from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.Conan.rhythm.frame_plan import build_frame_plan_from_execution, sample_tensor_by_frame_plan


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


if __name__ == "__main__":
    unittest.main()
