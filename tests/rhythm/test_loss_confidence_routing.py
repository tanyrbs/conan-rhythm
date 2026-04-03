from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.Conan.rhythm.losses import RhythmLossTargets, build_rhythm_loss_dict


class RhythmLossConfidenceRoutingTests(unittest.TestCase):
    def test_shape_distill_uses_dedicated_shape_confidence(self) -> None:
        execution = SimpleNamespace(
            speech_duration_exec=torch.tensor([[2.0, 0.0]], dtype=torch.float32),
            pause_after_exec=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
            planner=SimpleNamespace(
                speech_budget_win=torch.tensor([[2.0]], dtype=torch.float32),
                pause_budget_win=torch.tensor([[0.0]], dtype=torch.float32),
                raw_speech_budget_win=torch.tensor([[2.0]], dtype=torch.float32),
                raw_pause_budget_win=torch.tensor([[0.0]], dtype=torch.float32),
                boundary_score_unit=torch.zeros((1, 2), dtype=torch.float32),
                trace_context=torch.zeros((1, 2, 3), dtype=torch.float32),
                source_boundary_cue=torch.zeros((1, 2), dtype=torch.float32),
                feasible_total_budget_delta=torch.zeros((1, 1), dtype=torch.float32),
            ),
        )
        targets = RhythmLossTargets(
            speech_exec_tgt=torch.tensor([[2.0, 0.0]], dtype=torch.float32),
            pause_exec_tgt=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
            speech_budget_tgt=torch.tensor([[2.0]], dtype=torch.float32),
            pause_budget_tgt=torch.tensor([[0.0]], dtype=torch.float32),
            unit_mask=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
            dur_anchor_src=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
            distill_speech_tgt=torch.tensor([[0.0, 2.0]], dtype=torch.float32),
            distill_pause_tgt=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
            distill_exec_confidence=torch.tensor([[1.0]], dtype=torch.float32),
            distill_shape_confidence=torch.tensor([[0.0]], dtype=torch.float32),
            distill_speech_shape_weight=1.0,
            distill_pause_shape_weight=0.0,
            distill_allocation_weight=0.0,
            pause_boundary_weight=0.0,
        )

        losses = build_rhythm_loss_dict(execution, targets)

        self.assertGreater(float(losses["rhythm_distill_exec"].item()), 0.0)
        self.assertTrue(torch.allclose(losses["rhythm_distill_speech_shape"], torch.tensor(0.0)))
        self.assertTrue(torch.allclose(losses["rhythm_distill"], losses["rhythm_distill_exec"]))


if __name__ == "__main__":
    unittest.main()
