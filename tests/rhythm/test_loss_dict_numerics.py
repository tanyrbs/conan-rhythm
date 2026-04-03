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


class LossDictNumericsTests(unittest.TestCase):
    def test_plan_and_distill_terms_remain_finite_with_masked_tail(self) -> None:
        speech_exec = torch.tensor([[2.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
        pause_exec = torch.tensor([[0.5, 0.5, 0.0, 0.0]], dtype=torch.float32)
        unit_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
        execution = SimpleNamespace(
            speech_duration_exec=speech_exec,
            blank_duration_exec=pause_exec,
            pause_after_exec=pause_exec,
            planner=SimpleNamespace(
                speech_budget_win=speech_exec.sum(dim=1, keepdim=True),
                pause_budget_win=pause_exec.sum(dim=1, keepdim=True),
                raw_speech_budget_win=speech_exec.sum(dim=1, keepdim=True),
                raw_pause_budget_win=pause_exec.sum(dim=1, keepdim=True),
                boundary_score_unit=torch.zeros_like(speech_exec),
                source_boundary_cue=torch.zeros_like(speech_exec),
            ),
        )
        targets = RhythmLossTargets(
            speech_exec_tgt=speech_exec.clone(),
            pause_exec_tgt=pause_exec.clone(),
            speech_budget_tgt=speech_exec.sum(dim=1, keepdim=True),
            pause_budget_tgt=pause_exec.sum(dim=1, keepdim=True),
            unit_mask=unit_mask,
            dur_anchor_src=torch.tensor([[1.0, 2.0, 0.0, 0.0]], dtype=torch.float32),
            distill_speech_tgt=torch.flip(speech_exec, dims=[1]),
            distill_pause_tgt=torch.flip(pause_exec, dims=[1]),
            distill_allocation_weight=1.0,
            distill_speech_shape_weight=1.0,
            distill_pause_shape_weight=1.0,
        )
        losses = build_rhythm_loss_dict(execution, targets)
        for key in (
            "rhythm_plan_local",
            "rhythm_plan",
            "rhythm_distill_allocation",
            "rhythm_distill_speech_shape",
            "rhythm_distill_pause_shape",
            "rhythm_distill",
            "rhythm_total",
        ):
            self.assertTrue(torch.isfinite(losses[key]), key)
        self.assertIn("rhythm_prefix_clock", losses)
        self.assertIn("rhythm_prefix_backlog", losses)
        self.assertTrue(
            torch.allclose(
                losses["rhythm_prefix_state"],
                losses["rhythm_prefix_clock"] + losses["rhythm_prefix_backlog"],
            )
        )


if __name__ == "__main__":
    unittest.main()
