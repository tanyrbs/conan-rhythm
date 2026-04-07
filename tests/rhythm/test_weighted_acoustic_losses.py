from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.Conan.rhythm.task_mixin import RhythmConanTaskMixin
from utils.commons.hparams import hparams
from utils.metrics.ssim import ssim


class _DummyTask(RhythmConanTaskMixin):
    pass


class WeightedAcousticLossTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task = _DummyTask.__new__(_DummyTask)
        self.task.global_step = 0
        self.task.mel_losses = {"l1": 1.0}
        self._hparams_backup = dict(hparams)

    def tearDown(self) -> None:
        hparams.clear()
        hparams.update(self._hparams_backup)

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

    def test_resolve_acoustic_loss_scale_ramps_only_for_training_retimed_targets(self) -> None:
        hparams.clear()
        hparams.update(
            {
                "rhythm_stage3_acoustic_weight_end": 1.0,
                "rhythm_stage3_acoustic_weight_start": 0.25,
                "rhythm_stage3_acoustic_ramp_steps": 10000,
            }
        )
        self.task.global_step = 0
        self.assertAlmostEqual(
            self.task._resolve_stage3_acoustic_loss_scale(
                stage="student_retimed",
                retimed_stage_active=True,
                acoustic_target_is_retimed=True,
                infer=False,
                test=False,
            ),
            0.25,
        )
        self.task.global_step = 5000
        self.assertAlmostEqual(
            self.task._resolve_stage3_acoustic_loss_scale(
                stage="student_retimed",
                retimed_stage_active=True,
                acoustic_target_is_retimed=True,
                infer=False,
                test=False,
            ),
            0.625,
        )
        self.task.global_step = 20000
        self.assertAlmostEqual(
            self.task._resolve_stage3_acoustic_loss_scale(
                stage="student_retimed",
                retimed_stage_active=True,
                acoustic_target_is_retimed=True,
                infer=False,
                test=False,
            ),
            1.0,
        )
        self.assertEqual(
            self.task._resolve_stage3_acoustic_loss_scale(
                stage="student_retimed",
                retimed_stage_active=True,
                acoustic_target_is_retimed=True,
                infer=True,
                test=False,
            ),
            1.0,
        )
        self.assertEqual(
            self.task._resolve_stage3_acoustic_loss_scale(
                stage="student_kd",
                retimed_stage_active=True,
                acoustic_target_is_retimed=True,
                infer=False,
                test=False,
            ),
            1.0,
        )
        self.assertEqual(
            self.task._resolve_stage3_acoustic_loss_scale(
                stage="student_retimed",
                retimed_stage_active=True,
                acoustic_target_is_retimed=False,
                infer=False,
                test=False,
            ),
            1.0,
        )

    def test_resolve_acoustic_loss_scale_anchors_to_stage3_start_even_after_warm_start(self) -> None:
        hparams.clear()
        hparams.update(
            {
                "rhythm_stage3_acoustic_weight_end": 1.0,
                "rhythm_stage3_acoustic_weight_start": 0.25,
                "rhythm_stage3_acoustic_ramp_steps": 10000,
            }
        )
        self.task.global_step = 40000
        self.assertAlmostEqual(
            self.task._resolve_stage3_acoustic_loss_scale(
                stage="student_retimed",
                retimed_stage_active=True,
                acoustic_target_is_retimed=True,
                infer=False,
                test=False,
            ),
            0.25,
        )
        self.task.global_step = 45000
        self.assertAlmostEqual(
            self.task._resolve_stage3_acoustic_loss_scale(
                stage="student_retimed",
                retimed_stage_active=True,
                acoustic_target_is_retimed=True,
                infer=False,
                test=False,
            ),
            0.625,
        )
        self.task.global_step = 52000
        self.assertAlmostEqual(
            self.task._resolve_stage3_acoustic_loss_scale(
                stage="student_retimed",
                retimed_stage_active=True,
                acoustic_target_is_retimed=True,
                infer=False,
                test=False,
            ),
            1.0,
        )

    def test_add_acoustic_loss_applies_true_scalar_after_weighted_reduction(self) -> None:
        mel_out = torch.ones((1, 2, 2), dtype=torch.float32)
        target = torch.zeros_like(mel_out)
        frame_weight = torch.full((1, 2), 0.5, dtype=torch.float32)
        losses = {}
        self.task._add_acoustic_loss(
            mel_out,
            target,
            losses,
            frame_weight=frame_weight,
            loss_scale=0.25,
        )
        self.assertTrue(torch.allclose(losses["l1"], torch.tensor(0.25)))

    def test_add_acoustic_loss_scales_unweighted_mel_terms_after_add_mel_loss(self) -> None:
        def _fake_add_mel_loss(mel_out, target, losses):
            losses["l1"] = torch.tensor(2.0)
            losses["mse"] = torch.tensor(3.0)

        self.task.mel_losses = {"l1": 1.0, "mse": 1.0}
        self.task.add_mel_loss = _fake_add_mel_loss
        losses = {}
        self.task._add_acoustic_loss(
            torch.zeros((1, 1, 1), dtype=torch.float32),
            torch.zeros((1, 1, 1), dtype=torch.float32),
            losses,
            frame_weight=None,
            loss_scale=0.5,
        )
        self.assertTrue(torch.allclose(losses["l1"], torch.tensor(1.0)))
        self.assertTrue(torch.allclose(losses["mse"], torch.tensor(1.5)))

    def test_add_pitch_loss_applies_stage3_scalar_to_pitch_terms(self) -> None:
        hparams.clear()
        hparams.update({"f0_gen": "diff", "lambda_uv": 1.0})
        output = {
            "content": torch.ones((1, 4), dtype=torch.long),
            "tgt_nonpadding": torch.ones((1, 4), dtype=torch.float32),
            "fdiff": torch.tensor(4.0),
            "uv_pred": torch.zeros((1, 4, 1), dtype=torch.float32),
        }
        sample = {
            "content": torch.ones((1, 4), dtype=torch.long),
            "f0": torch.zeros((1, 4), dtype=torch.float32),
            "uv": torch.zeros((1, 4), dtype=torch.float32),
        }
        full_losses = {}
        scaled_losses = {}
        self.task.add_pitch_loss(output, sample, full_losses, loss_scale=1.0)
        self.task.add_pitch_loss(output, sample, scaled_losses, loss_scale=0.25)
        self.assertTrue(torch.allclose(scaled_losses["fdiff"], full_losses["fdiff"] * 0.25))
        self.assertTrue(torch.allclose(scaled_losses["uv"], full_losses["uv"] * 0.25))

    def test_observability_passthrough_includes_retimed_acoustic_scale(self) -> None:
        observability = self.task._collect_runtime_observability_outputs(
            {
                "rhythm_stage3_acoustic_loss_scale": torch.tensor([0.25, 0.75], dtype=torch.float32),
                "rhythm_retimed_acoustic_loss_scale": torch.tensor([0.25, 0.75], dtype=torch.float32),
                "rhythm_stage3_pitch_loss_scale": torch.tensor([0.25, 0.75], dtype=torch.float32),
                "rhythm_retimed_pitch_loss_scale": torch.tensor([0.25, 0.75], dtype=torch.float32),
            }
        )
        self.assertAlmostEqual(observability["rhythm_metric_stage3_acoustic_loss_scale"], 0.5)
        self.assertAlmostEqual(observability["rhythm_metric_retimed_acoustic_loss_scale"], 0.5)
        self.assertAlmostEqual(observability["rhythm_metric_stage3_pitch_loss_scale"], 0.5)
        self.assertAlmostEqual(observability["rhythm_metric_retimed_pitch_loss_scale"], 0.5)


if __name__ == "__main__":
    unittest.main()
