from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.Conan.rhythm.acoustic_loss_utils import (
    expand_frame_weight,
    reduce_weighted_elementwise_loss,
)
from utils.nn.seq_utils import weights_nonzero_speech


class AcousticLossUtilsTests(unittest.TestCase):
    def test_expand_frame_weight_defaults_to_nonzero_speech_mask(self) -> None:
        target = torch.tensor(
            [
                [[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]],
            ],
            dtype=torch.float32,
        )
        expected = weights_nonzero_speech(target)
        actual = expand_frame_weight(None, target)
        self.assertTrue(torch.equal(actual, expected))

    def test_reduce_weighted_loss_matches_unweighted_mean_for_uniform_weights(self) -> None:
        pred = torch.tensor(
            [
                [[1.0, 3.0], [2.0, 4.0]],
            ],
            dtype=torch.float32,
        )
        target = torch.zeros_like(pred)
        loss = F.l1_loss(pred, target, reduction="none")
        frame_weight = torch.ones((pred.size(0), pred.size(1)), dtype=torch.float32)

        reduced = reduce_weighted_elementwise_loss(
            loss,
            frame_weight=frame_weight,
            target=target,
        )
        expected = F.l1_loss(pred, target)
        self.assertTrue(torch.isclose(reduced, expected))

    def test_reduce_weighted_loss_uses_full_broadcasted_weight_mass(self) -> None:
        pred = torch.tensor(
            [
                [[1.0, 2.0], [10.0, 20.0]],
            ],
            dtype=torch.float32,
        )
        target = torch.zeros_like(pred)
        loss = F.l1_loss(pred, target, reduction="none")
        frame_weight = torch.tensor([[1.0, 0.25]], dtype=torch.float32)

        reduced = reduce_weighted_elementwise_loss(
            loss,
            frame_weight=frame_weight,
            target=target,
        )
        expanded = torch.tensor(
            [
                [[1.0, 1.0], [0.25, 0.25]],
            ],
            dtype=torch.float32,
        )
        expected = (loss * expanded).sum() / expanded.sum()
        self.assertTrue(torch.isclose(reduced, expected))


if __name__ == "__main__":
    unittest.main()
