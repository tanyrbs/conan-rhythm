from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.Conan.rhythm.projector import _apply_boundary_pause_constraints
from modules.Conan.rhythm.source_boundary import (
    BOUNDARY_TYPE_JOIN,
    BOUNDARY_TYPE_PHRASE,
    BOUNDARY_TYPE_WEAK,
)


class ProjectorBoundaryTypeTests(unittest.TestCase):
    def test_boundary_pause_constraints_zero_join_cap_weak_and_redistribute_to_phrase(self) -> None:
        pause_after_exec = torch.tensor([[0.0, 3.0, 2.5, 0.5]], dtype=torch.float32)
        boundary_type_unit = torch.tensor(
            [[BOUNDARY_TYPE_JOIN, BOUNDARY_TYPE_WEAK, BOUNDARY_TYPE_PHRASE, BOUNDARY_TYPE_JOIN]],
            dtype=torch.long,
        )
        constrained = _apply_boundary_pause_constraints(
            pause_after_exec=pause_after_exec,
            boundary_type_unit=boundary_type_unit,
            unit_mask=torch.ones_like(pause_after_exec),
            commit_frontier=torch.tensor([0], dtype=torch.long),
            reuse_prefix=False,
            weak_boundary_pause_cap=1.0,
        )
        self.assertAlmostEqual(float(constrained[0, 0].item()), 0.0, places=6)
        self.assertAlmostEqual(float(constrained[0, 1].item()), 1.0, places=6)
        self.assertAlmostEqual(float(constrained[0, 3].item()), 0.0, places=6)
        self.assertGreater(float(constrained[0, 2].item()), float(pause_after_exec[0, 2].item()))
        self.assertAlmostEqual(float(constrained.sum().item()), float(pause_after_exec.sum().item()), places=6)

    def test_boundary_pause_constraints_do_not_rewrite_reused_prefix(self) -> None:
        pause_after_exec = torch.tensor([[0.5, 2.0, 3.0]], dtype=torch.float32)
        boundary_type_unit = torch.tensor(
            [[BOUNDARY_TYPE_JOIN, BOUNDARY_TYPE_WEAK, BOUNDARY_TYPE_PHRASE]],
            dtype=torch.long,
        )
        constrained = _apply_boundary_pause_constraints(
            pause_after_exec=pause_after_exec,
            boundary_type_unit=boundary_type_unit,
            unit_mask=torch.ones_like(pause_after_exec),
            commit_frontier=torch.tensor([2], dtype=torch.long),
            reuse_prefix=True,
            weak_boundary_pause_cap=1.0,
        )
        self.assertTrue(torch.allclose(constrained[:, :2], pause_after_exec[:, :2]))
        self.assertAlmostEqual(float(constrained[0, 2].item()), float(pause_after_exec[0, 2].item()), places=6)


if __name__ == "__main__":
    unittest.main()
