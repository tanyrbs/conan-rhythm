from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.Conan.rhythm.runtime_teacher_supervision import slice_runtime_teacher_execution


class RuntimeTeacherSupervisionTests(unittest.TestCase):
    def test_slice_budget_view_keeps_grad_path(self) -> None:
        speech_exec = torch.tensor([[2.0, 3.0, 5.0]], dtype=torch.float32, requires_grad=True)
        blank_exec = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32, requires_grad=True)
        speech_budget = torch.tensor([[10.0]], dtype=torch.float32, requires_grad=True)
        pause_budget = torch.tensor([[6.0]], dtype=torch.float32, requires_grad=True)
        raw_speech_budget = torch.tensor([[8.0]], dtype=torch.float32, requires_grad=True)
        raw_pause_budget = torch.tensor([[4.0]], dtype=torch.float32, requires_grad=True)
        feasible_speech_delta = torch.tensor([[2.0]], dtype=torch.float32, requires_grad=True)
        feasible_pause_delta = torch.tensor([[2.0]], dtype=torch.float32, requires_grad=True)
        execution = SimpleNamespace(
            speech_duration_exec=speech_exec,
            blank_duration_exec=blank_exec,
            pause_after_exec=blank_exec,
            planner=SimpleNamespace(
                speech_budget_win=speech_budget,
                pause_budget_win=pause_budget,
                raw_speech_budget_win=raw_speech_budget,
                raw_pause_budget_win=raw_pause_budget,
                effective_speech_budget_win=speech_budget,
                effective_pause_budget_win=pause_budget,
                feasible_speech_budget_delta=feasible_speech_delta,
                feasible_pause_budget_delta=feasible_pause_delta,
                boundary_score_unit=torch.zeros((1, 3), dtype=torch.float32),
                source_boundary_cue=torch.zeros((1, 3), dtype=torch.float32),
            ),
        )

        sliced = slice_runtime_teacher_execution(execution, teacher_units=2)
        loss = (
            sliced.planner.raw_speech_budget_win.sum()
            + sliced.planner.speech_budget_win.sum()
            + sliced.planner.feasible_total_budget_delta.sum()
        )
        loss.backward()
        self.assertIsNotNone(raw_speech_budget.grad)
        self.assertIsNotNone(speech_budget.grad)
        self.assertIsNotNone(feasible_speech_delta.grad)

    def test_slice_preserves_budget_repair_semantics_proportionally(self) -> None:
        execution = SimpleNamespace(
            speech_duration_exec=torch.tensor([[2.0, 3.0, 5.0]], dtype=torch.float32),
            blank_duration_exec=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
            pause_after_exec=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
            planner=SimpleNamespace(
                speech_budget_win=torch.tensor([[10.0]], dtype=torch.float32),
                pause_budget_win=torch.tensor([[6.0]], dtype=torch.float32),
                raw_speech_budget_win=torch.tensor([[8.0]], dtype=torch.float32),
                raw_pause_budget_win=torch.tensor([[4.0]], dtype=torch.float32),
                effective_speech_budget_win=torch.tensor([[10.0]], dtype=torch.float32),
                effective_pause_budget_win=torch.tensor([[6.0]], dtype=torch.float32),
                feasible_speech_budget_delta=torch.tensor([[2.0]], dtype=torch.float32),
                feasible_pause_budget_delta=torch.tensor([[2.0]], dtype=torch.float32),
                feasible_total_budget_delta=torch.tensor([[4.0]], dtype=torch.float32),
                boundary_score_unit=torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
                source_boundary_cue=torch.tensor([[0.4, 0.5, 0.6]], dtype=torch.float32),
            ),
        )

        sliced = slice_runtime_teacher_execution(execution, teacher_units=2)
        self.assertEqual(sliced.planner.runtime_budget_slice_mode, "proportional_prefix")
        self.assertTrue(torch.allclose(sliced.planner.speech_budget_win, torch.tensor([[5.0]])))
        self.assertTrue(torch.allclose(sliced.planner.pause_budget_win, torch.tensor([[3.0]])))
        self.assertTrue(torch.allclose(sliced.planner.raw_speech_budget_win, torch.tensor([[4.0]])))
        self.assertTrue(torch.allclose(sliced.planner.raw_pause_budget_win, torch.tensor([[2.0]])))
        self.assertTrue(torch.allclose(sliced.planner.feasible_speech_budget_delta, torch.tensor([[1.0]])))
        self.assertTrue(torch.allclose(sliced.planner.feasible_pause_budget_delta, torch.tensor([[1.0]])))
        self.assertTrue(torch.allclose(sliced.planner.feasible_total_budget_delta, torch.tensor([[2.0]])))
        self.assertTrue(torch.allclose(sliced.planner.boundary_score_unit, torch.tensor([[0.1, 0.2]])))
        self.assertTrue(torch.allclose(sliced.planner.source_boundary_cue, torch.tensor([[0.4, 0.5]])))

    def test_no_slice_returns_original_execution(self) -> None:
        execution = SimpleNamespace(
            speech_duration_exec=torch.ones((1, 2), dtype=torch.float32),
            blank_duration_exec=torch.zeros((1, 2), dtype=torch.float32),
            pause_after_exec=torch.zeros((1, 2), dtype=torch.float32),
            planner=SimpleNamespace(),
        )
        self.assertIs(slice_runtime_teacher_execution(execution, teacher_units=2), execution)


if __name__ == "__main__":
    unittest.main()
