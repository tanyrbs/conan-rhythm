from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


if __name__ == "__main__":
    unittest.main()
