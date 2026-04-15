from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.Conan.rhythm.common.losses_impl import (
    _batch_kl_div,
    _budget_surface_loss,
    _compute_budget_supervision,
    _reduce_batch_loss_with_scale,
)


class BudgetSurfaceTests(unittest.TestCase):
    def test_compact_budget_loss_zero_on_exact_match(self) -> None:
        loss, total_loss, pause_share_loss = _budget_surface_loss(
            pred_speech_budget=torch.tensor([[3.0]], dtype=torch.float32),
            pred_pause_budget=torch.tensor([[1.0]], dtype=torch.float32),
            tgt_speech_budget=torch.tensor([[3.0]], dtype=torch.float32),
            tgt_pause_budget=torch.tensor([[1.0]], dtype=torch.float32),
        )
        self.assertTrue(torch.allclose(loss, torch.tensor(0.0)))
        self.assertTrue(torch.allclose(total_loss, torch.tensor(0.0)))
        self.assertTrue(torch.allclose(pause_share_loss, torch.tensor(0.0)))

    def test_pure_total_error_does_not_leak_into_pause_share_surface(self) -> None:
        loss, total_loss, pause_share_loss = _budget_surface_loss(
            pred_speech_budget=torch.tensor([[2.0]], dtype=torch.float32),
            pred_pause_budget=torch.tensor([[1.0]], dtype=torch.float32),
            tgt_speech_budget=torch.tensor([[4.0]], dtype=torch.float32),
            tgt_pause_budget=torch.tensor([[2.0]], dtype=torch.float32),
        )
        self.assertGreater(float(total_loss.item()), 0.0)
        self.assertTrue(torch.allclose(pause_share_loss, torch.tensor(0.0)))
        self.assertTrue(torch.allclose(loss, total_loss))

    def test_pure_pause_share_error_does_not_leak_into_total_surface(self) -> None:
        loss, total_loss, pause_share_loss = _budget_surface_loss(
            pred_speech_budget=torch.tensor([[3.0]], dtype=torch.float32),
            pred_pause_budget=torch.tensor([[1.0]], dtype=torch.float32),
            tgt_speech_budget=torch.tensor([[1.0]], dtype=torch.float32),
            tgt_pause_budget=torch.tensor([[3.0]], dtype=torch.float32),
        )
        self.assertTrue(torch.allclose(total_loss, torch.tensor(0.0)))
        self.assertGreater(float(pause_share_loss.item()), 0.0)
        self.assertTrue(torch.allclose(loss, pause_share_loss))

    def test_budget_supervision_keeps_raw_exec_split_but_uses_compact_surfaces(self) -> None:
        execution = SimpleNamespace(
            planner=SimpleNamespace(
                raw_speech_budget_win=torch.tensor([[2.0]], dtype=torch.float32),
                raw_pause_budget_win=torch.tensor([[1.0]], dtype=torch.float32),
                speech_budget_win=torch.tensor([[3.0]], dtype=torch.float32),
                pause_budget_win=torch.tensor([[1.5]], dtype=torch.float32),
            )
        )
        total_loss, raw_surface, exec_surface, total_surface, pause_share_surface = _compute_budget_supervision(
            execution,
            speech_budget_tgt=torch.tensor([[4.0]], dtype=torch.float32),
            pause_budget_tgt=torch.tensor([[2.0]], dtype=torch.float32),
            raw_weight=1.0,
            exec_weight=0.25,
        )
        self.assertGreater(float(raw_surface.item()), 0.0)
        self.assertGreater(float(exec_surface.item()), 0.0)
        self.assertGreater(float(total_surface.item()), 0.0)
        self.assertLess(float(exec_surface.item()), float(raw_surface.item()))
        self.assertTrue(torch.allclose(pause_share_surface, torch.tensor(0.0)))
        self.assertTrue(torch.allclose(total_loss, total_surface + pause_share_surface))

    def test_reduce_batch_loss_with_scale_does_not_normalize_away_sample_gate(self) -> None:
        loss = torch.tensor([2.0, 2.0], dtype=torch.float32)
        scaled = _reduce_batch_loss_with_scale(
            loss,
            batch_weight=torch.ones((2, 1), dtype=torch.float32),
            loss_scale=torch.tensor([[1.0], [0.0]], dtype=torch.float32),
        )
        self.assertTrue(torch.allclose(scaled, torch.tensor(1.0)))

    def test_budget_supervision_soft_gates_exec_total_when_feasibility_lifts_total_budget(self) -> None:
        execution = SimpleNamespace(
            planner=SimpleNamespace(
                raw_speech_budget_win=torch.tensor([[4.0]], dtype=torch.float32),
                raw_pause_budget_win=torch.tensor([[2.0]], dtype=torch.float32),
                speech_budget_win=torch.tensor([[6.0]], dtype=torch.float32),
                pause_budget_win=torch.tensor([[2.0]], dtype=torch.float32),
                feasible_total_budget_delta=torch.tensor([[2.0]], dtype=torch.float32),
            )
        )
        total_loss, raw_surface, exec_surface, total_surface, pause_share_surface = _compute_budget_supervision(
            execution,
            speech_budget_tgt=torch.tensor([[4.0]], dtype=torch.float32),
            pause_budget_tgt=torch.tensor([[2.0]], dtype=torch.float32),
            raw_weight=1.0,
            exec_weight=0.25,
        )
        ungated_exec_total = torch.abs(torch.log1p(torch.tensor(8.0)) - torch.log1p(torch.tensor(6.0)))
        expected_gate = torch.tensor(6.0 / 8.0)
        expected_exec_total = 0.25 * ungated_exec_total * expected_gate
        expected_pause_share = 0.25 * torch.tensor(abs((2.0 / 8.0) - (2.0 / 6.0))) * expected_gate
        self.assertTrue(torch.allclose(raw_surface, torch.tensor(0.0)))
        self.assertTrue(torch.allclose(total_surface, expected_exec_total, atol=1e-6))
        self.assertTrue(torch.allclose(pause_share_surface, expected_pause_share, atol=1e-6))
        self.assertTrue(torch.allclose(exec_surface, expected_exec_total + expected_pause_share, atol=1e-6))
        self.assertTrue(torch.allclose(total_loss, exec_surface, atol=1e-6))

    def test_masked_kl_stays_finite_with_zeroed_tail(self) -> None:
        pred = torch.tensor([[2.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
        tgt = torch.tensor([[2.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
        loss = _batch_kl_div(pred, tgt, mask)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.allclose(loss, torch.tensor(0.0)))

    def test_budget_supervision_soft_gates_exec_pause_share_when_repair_only_redistributes_budget(self) -> None:
        execution = SimpleNamespace(
            planner=SimpleNamespace(
                raw_speech_budget_win=torch.tensor([[1.0]], dtype=torch.float32),
                raw_pause_budget_win=torch.tensor([[4.0]], dtype=torch.float32),
                speech_budget_win=torch.tensor([[4.0]], dtype=torch.float32),
                pause_budget_win=torch.tensor([[1.0]], dtype=torch.float32),
                feasible_total_budget_delta=torch.tensor([[0.0]], dtype=torch.float32),
            )
        )
        total_loss, raw_surface, exec_surface, total_surface, pause_share_surface = _compute_budget_supervision(
            execution,
            speech_budget_tgt=torch.tensor([[1.0]], dtype=torch.float32),
            pause_budget_tgt=torch.tensor([[4.0]], dtype=torch.float32),
            raw_weight=1.0,
            exec_weight=0.25,
        )
        expected_gate = torch.tensor(5.0 / 8.0)
        expected_exec_pause_share = 0.25 * torch.tensor(abs((1.0 / 5.0) - (4.0 / 5.0))) * expected_gate
        self.assertTrue(torch.allclose(raw_surface, torch.tensor(0.0)))
        self.assertTrue(torch.allclose(total_surface, torch.tensor(0.0)))
        self.assertTrue(torch.allclose(pause_share_surface, expected_exec_pause_share, atol=1e-6))
        self.assertTrue(torch.allclose(exec_surface, expected_exec_pause_share, atol=1e-6))
        self.assertTrue(torch.allclose(total_loss, pause_share_surface, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
