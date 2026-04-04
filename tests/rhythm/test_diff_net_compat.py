from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.Conan.diff.net import silu


class DiffNetCompatTests(unittest.TestCase):
    def test_silu_matches_eager_functional_and_is_not_scripted(self) -> None:
        x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=torch.float32)
        self.assertTrue(torch.allclose(silu(x), F.silu(x)))
        self.assertFalse(hasattr(silu, "graph"))


if __name__ == "__main__":
    unittest.main()
