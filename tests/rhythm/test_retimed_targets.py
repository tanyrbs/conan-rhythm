from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.Conan.rhythm.retimed_targets import _infer_silence_frame


class RetimedTargetSilenceFrameTests(unittest.TestCase):
    def test_infer_silence_frame_averages_bottom_k_pool(self) -> None:
        mel = torch.tensor(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [9.0, 9.0], [10.0, 10.0], [11.0, 11.0], [12.0, 12.0], [13.0, 13.0], [14.0, 14.0], [15.0, 15.0]],
            dtype=torch.float32,
        )
        silence = _infer_silence_frame(mel)
        expected = mel[:3].mean(dim=0)
        self.assertTrue(torch.allclose(silence, expected))

    def test_infer_silence_frame_caps_bottom_k_by_sequence_length(self) -> None:
        mel = torch.tensor([[2.0, 4.0], [0.0, 2.0]], dtype=torch.float32)
        silence = _infer_silence_frame(mel)
        expected = mel.mean(dim=0)
        self.assertTrue(torch.allclose(silence, expected))


if __name__ == "__main__":
    unittest.main()
