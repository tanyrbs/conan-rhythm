from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.Conan.rhythm.task_runtime_support import RhythmTaskRuntimeSupport
from tasks.Conan.rhythm.task_mixin import RhythmConanTaskMixin


class PitchSupervisionRuntimeTests(unittest.TestCase):
    def test_missing_source_pitch_raises_when_pitch_embed_is_enabled(self) -> None:
        with mock.patch.dict(
            "tasks.Conan.rhythm.task_mixin.hparams",
            {
                "use_pitch_embed": True,
                "rhythm_fail_fast_missing_pitch_supervision": True,
            },
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "Pitch supervision is missing"):
                RhythmConanTaskMixin._assert_pitch_supervision_ready(
                    {
                        "disable_acoustic_train_path": 0.0,
                        "rhythm_pitch_supervision_disabled": 0.0,
                        "acoustic_target_is_retimed": 0.0,
                    },
                    {"f0": None, "uv": None},
                    infer=False,
                    test=False,
                    retimed_stage_active=False,
                )

    def test_retimed_training_requires_matched_retimed_pitch_targets(self) -> None:
        with mock.patch.dict(
            "tasks.Conan.rhythm.task_mixin.hparams",
            {
                "use_pitch_embed": True,
                "rhythm_fail_fast_missing_pitch_supervision": True,
            },
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "retimed training is missing usable pitch supervision"):
                RhythmConanTaskMixin._assert_pitch_supervision_ready(
                    {
                        "disable_acoustic_train_path": 0.0,
                        "rhythm_pitch_supervision_disabled": 0.0,
                        "acoustic_target_is_retimed": 1.0,
                    },
                    {"f0": [1.0], "uv": [0.0]},
                    infer=False,
                    test=False,
                    retimed_stage_active=True,
                )

    def test_retimed_length_mismatch_raises_before_pitch_loss_runs(self) -> None:
        with mock.patch.dict(
            "tasks.Conan.rhythm.task_mixin.hparams",
            {
                "use_pitch_embed": True,
                "rhythm_fail_fast_missing_pitch_supervision": True,
            },
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "length-aligned"):
                RhythmConanTaskMixin._assert_pitch_supervision_ready(
                    {
                        "disable_acoustic_train_path": 0.0,
                        "rhythm_pitch_supervision_disabled": 0.0,
                        "acoustic_target_is_retimed": 1.0,
                        "mel_out": torch.zeros((1, 4, 80), dtype=torch.float32),
                        "retimed_f0_tgt": torch.ones((1, 6), dtype=torch.float32),
                        "retimed_uv_tgt": torch.zeros((1, 6), dtype=torch.float32),
                    },
                    {"f0": [1.0, 2.0, 3.0, 4.0], "uv": [0.0, 0.0, 0.0, 0.0]},
                    infer=False,
                    test=False,
                    retimed_stage_active=True,
                )

    def test_escape_hatch_allows_debug_runs_without_pitch_targets(self) -> None:
        with mock.patch.dict(
            "tasks.Conan.rhythm.task_mixin.hparams",
            {
                "use_pitch_embed": True,
                "rhythm_fail_fast_missing_pitch_supervision": False,
            },
            clear=True,
        ):
            RhythmConanTaskMixin._assert_pitch_supervision_ready(
                {
                    "disable_acoustic_train_path": 0.0,
                    "rhythm_pitch_supervision_disabled": 1.0,
                    "acoustic_target_is_retimed": 1.0,
                },
                {"f0": None, "uv": None},
                infer=False,
                test=False,
                retimed_stage_active=True,
            )

    def test_source_pitch_supervision_passes_when_non_retimed_target_is_active(self) -> None:
        with mock.patch.dict(
            "tasks.Conan.rhythm.task_mixin.hparams",
            {
                "use_pitch_embed": True,
                "rhythm_fail_fast_missing_pitch_supervision": True,
            },
            clear=True,
        ):
            RhythmConanTaskMixin._assert_pitch_supervision_ready(
                {
                    "disable_acoustic_train_path": 0.0,
                    "rhythm_pitch_supervision_disabled": 0.0,
                    "acoustic_target_is_retimed": 0.0,
                },
                {"f0": [120.0, 121.0], "uv": [0.0, 0.0]},
                infer=False,
                test=False,
                retimed_stage_active=False,
            )

    def test_attach_bundle_then_add_pitch_loss_avoids_length_mismatch_on_retimed_targets(self) -> None:
        class _DummyOwner:
            mel_losses = {"l1": 1.0}

            @staticmethod
            def _align_acoustic_target_to_output(mel_out, acoustic_target, acoustic_weight):
                target_len = mel_out.size(1)
                return mel_out, acoustic_target[:, :target_len], acoustic_weight[:, :target_len]

        class _DummyTask(RhythmConanTaskMixin):
            pass

        support = RhythmTaskRuntimeSupport(_DummyOwner())
        task = _DummyTask()
        output = {
            "mel_out": torch.zeros((1, 4, 80), dtype=torch.float32),
            "retimed_f0_tgt": torch.arange(6, dtype=torch.float32).unsqueeze(0),
            "retimed_uv_tgt": torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]], dtype=torch.float32),
            "uv_pred": torch.zeros((1, 4, 1), dtype=torch.float32),
            "fdiff": torch.tensor(0.0),
        }
        sample = {
            "content": torch.ones((1, 4), dtype=torch.long),
            "f0": torch.zeros((1, 4), dtype=torch.float32),
            "uv": torch.zeros((1, 4), dtype=torch.float32),
        }
        with mock.patch.dict(
            "tasks.Conan.rhythm.task_runtime_support.hparams",
            {"rhythm_resample_retimed_target_to_output": False},
            clear=True,
        ):
            support.attach_acoustic_target_bundle(
                output,
                acoustic_target=torch.zeros((1, 6, 80), dtype=torch.float32),
                acoustic_target_is_retimed=True,
                acoustic_weight=torch.ones((1, 6), dtype=torch.float32),
                acoustic_target_source="cached",
                disable_source_pitch_supervision=False,
                disable_acoustic_train_path=False,
            )
        losses = {}
        with mock.patch.dict(
            "tasks.Conan.rhythm.task_mixin.hparams",
            {"f0_gen": "diff", "lambda_uv": 1.0},
            clear=True,
        ):
            task.add_pitch_loss(output, sample, losses)
        self.assertIn("uv", losses)
        self.assertTrue(torch.isfinite(losses["uv"]))


if __name__ == "__main__":
    unittest.main()
