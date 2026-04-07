import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.Conan.rhythm.reference_encoder import ReferenceRhythmEncoder


class ReferenceRhythmEncoderTests(unittest.TestCase):
    def test_boundary_trace_uses_soft_strength_while_stats_keep_binary_ratio(self) -> None:
        encoder = ReferenceRhythmEncoder(
            trace_bins=8,
            smooth_kernel=3,
            boundary_quantile=0.60,
        )
        energy = torch.tensor(
            [[-1.0, 0.0, 2.0, 0.0, 2.0, 0.0, -1.0, 0.0]],
            dtype=torch.float32,
        )
        ref_mel = energy.unsqueeze(-1).expand(-1, -1, 80)

        encoded = encoder(ref_mel)

        boundary_trace = encoded["ref_rhythm_trace"][0, :, 2]
        boundary_ratio = encoded["ref_rhythm_stats"][0, 4]

        self.assertTrue(torch.all(boundary_trace >= 0.0))
        self.assertTrue(torch.all(boundary_trace <= 1.0))
        self.assertTrue(torch.any((boundary_trace > 0.0) & (boundary_trace < 1.0)))
        self.assertFalse(torch.allclose(boundary_trace, boundary_trace.round()))
        self.assertGreaterEqual(float(boundary_ratio.item()), 0.0)
        self.assertLessEqual(float(boundary_ratio.item()), 1.0)


if __name__ == "__main__":
    unittest.main()
