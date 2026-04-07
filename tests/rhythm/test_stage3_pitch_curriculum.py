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


class _DummyTask(RhythmConanTaskMixin):
    def __init__(self):
        self.global_step = 0
        self.mel_losses = {"l1": 1.0}


class Stage3PitchCurriculumTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = dict(hparams)
        hparams.clear()
        hparams.update(
            {
                "f0_gen": "diff",
                "lambda_uv": 1.0,
                "rhythm_stage3_scale_pitch_loss": True,
            }
        )
        self.task = _DummyTask()

    def tearDown(self) -> None:
        hparams.clear()
        hparams.update(self._saved)

    def test_add_pitch_loss_scales_pitch_terms(self) -> None:
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

    def test_pitch_loss_scale_defaults_to_acoustic_scale(self) -> None:
        output = {
            "rhythm_stage": "student_retimed",
            "acoustic_target_is_retimed": True,
        }
        scale = self.task._resolve_pitch_loss_scale(output, infer=False, test=False)
        acoustic_scale = self.task._resolve_acoustic_loss_scale(output, infer=False, test=False)
        self.assertAlmostEqual(scale, acoustic_scale)

    def test_explicit_pitch_scale_override_wins(self) -> None:
        hparams["rhythm_stage3_pitch_loss_scale"] = 0.4
        output = {
            "rhythm_stage": "student_retimed",
            "acoustic_target_is_retimed": True,
        }
        scale = self.task._resolve_pitch_loss_scale(output, infer=False, test=False)
        self.assertAlmostEqual(scale, 0.4)


if __name__ == "__main__":
    unittest.main()
