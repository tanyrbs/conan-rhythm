from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.Conan.rhythm.task_mixin import RhythmConanTaskMixin
from utils.metrics.ssim import ssim


class _DummyTask(RhythmConanTaskMixin):
    pass


class WeightedAcousticLossTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task = _DummyTask.__new__(_DummyTask)

    def test_expand_frame_weight_matches_target_shape(self) -> None:
        target = torch.zeros((2, 3, 4), dtype=torch.float32)
        frame_weight = torch.tensor([[1.0, 0.5, 0.0], [0.25, 1.0, 0.75]], dtype=torch.float32)
        expanded = self.task._expand_frame_weight(frame_weight, target)
        self.assertEqual(expanded.shape, target.shape)
        self.assertTrue(torch.allclose(expanded[:, :, 0], frame_weight))
        self.assertTrue(torch.allclose(expanded[:, :, -1], frame_weight))

    def test_weighted_l1_matches_manual_channel_normalization(self) -> None:
        decoder_output = torch.tensor([[[1.0, 3.0], [2.0, 6.0]]], dtype=torch.float32)
        target = torch.tensor([[[0.0, 1.0], [1.0, 5.0]]], dtype=torch.float32)
        frame_weight = torch.tensor([[1.0, 0.5]], dtype=torch.float32)
        expanded = frame_weight[:, :, None].expand_as(target)
        expected = (torch.abs(decoder_output - target) * expanded).sum() / expanded.sum()
        actual = self.task._weighted_l1_loss(decoder_output, target, frame_weight)
        self.assertTrue(torch.allclose(actual, expected))

    def test_weighted_mse_matches_manual_channel_normalization(self) -> None:
        decoder_output = torch.tensor([[[1.0, 3.0], [2.0, 6.0]]], dtype=torch.float32)
        target = torch.tensor([[[0.0, 1.0], [1.0, 5.0]]], dtype=torch.float32)
        frame_weight = torch.tensor([[1.0, 0.5]], dtype=torch.float32)
        expanded = frame_weight[:, :, None].expand_as(target)
        expected = (((decoder_output - target) ** 2) * expanded).sum() / expanded.sum()
        actual = self.task._weighted_mse_loss(decoder_output, target, frame_weight)
        self.assertTrue(torch.allclose(actual, expected))

    def test_weighted_ssim_matches_manual_channel_normalization(self) -> None:
        torch.manual_seed(0)
        decoder_output = torch.randn((1, 12, 6), dtype=torch.float32)
        target = torch.randn((1, 12, 6), dtype=torch.float32)
        frame_weight = torch.tensor([[1.0] * 6 + [0.5] * 6], dtype=torch.float32)
        ssim_loss = 1 - ssim(decoder_output[:, None] + 6.0, target[:, None] + 6.0, size_average=False)
        expanded = frame_weight[:, :, None].expand_as(target)
        expected = (ssim_loss * expanded).sum() / expanded.sum()
        actual = self.task._weighted_ssim_loss(decoder_output, target, frame_weight)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
