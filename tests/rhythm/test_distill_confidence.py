from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.Conan.rhythm.distill_confidence import (
    normalize_component_distill_confidence,
    normalize_distill_confidence,
)


class DistillConfidenceTests(unittest.TestCase):
    def test_component_normalization_preserves_explicit_zero_gate(self) -> None:
        confidence = normalize_component_distill_confidence(
            torch.tensor([[0.0], [0.2]], dtype=torch.float32),
            fallback_confidence=torch.ones((2, 1), dtype=torch.float32),
            batch_size=2,
            device=torch.device("cpu"),
            floor=0.05,
            power=1.0,
            preserve_zeros=True,
        )
        self.assertTrue(torch.allclose(confidence, torch.tensor([[0.0], [0.2]], dtype=torch.float32)))

    def test_shared_normalization_still_uses_floor(self) -> None:
        confidence = normalize_distill_confidence(
            torch.tensor([[0.01]], dtype=torch.float32),
            batch_size=1,
            device=torch.device("cpu"),
            floor=0.05,
            power=1.0,
        )
        self.assertTrue(torch.allclose(confidence, torch.tensor([[0.05]], dtype=torch.float32)))


if __name__ == "__main__":
    unittest.main()
