from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.Conan.Conan import ConanTask


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rhythm_enable_v2 = True
        self.core = nn.Linear(2, 2, bias=False)
        self.extra = nn.Linear(2, 2, bias=False)


class StageParamFreezeGuardTests(unittest.TestCase):
    @staticmethod
    def _make_task() -> ConanTask:
        task = ConanTask.__new__(ConanTask)
        nn.Module.__init__(task)
        return task

    def test_module_only_stage_raises_when_collector_returns_empty(self) -> None:
        task = self._make_task()
        with mock.patch("tasks.Conan.Conan.Conan", return_value=_DummyModel()):
            with mock.patch("tasks.Conan.Conan.detect_rhythm_stage", return_value="student_kd"):
                with mock.patch.dict(
                    "tasks.Conan.Conan.hparams",
                    {"rhythm_optimize_module_only": True},
                    clear=True,
                ):
                    with mock.patch.object(ConanTask, "_collect_rhythm_gen_params", return_value=[]):
                        with self.assertRaisesRegex(RuntimeError, "zero trainable params"):
                            ConanTask.build_tts_model(task)

    def test_teacher_only_stage_raises_when_collector_returns_empty(self) -> None:
        task = self._make_task()
        with mock.patch("tasks.Conan.Conan.Conan", return_value=_DummyModel()):
            with mock.patch("tasks.Conan.Conan.detect_rhythm_stage", return_value="teacher_offline"):
                with mock.patch.dict(
                    "tasks.Conan.Conan.hparams",
                    {"rhythm_optimize_module_only": False},
                    clear=True,
                ):
                    with mock.patch.object(ConanTask, "_collect_offline_teacher_gen_params", return_value=[]):
                        with self.assertRaisesRegex(RuntimeError, "zero trainable params"):
                            ConanTask.build_tts_model(task)

    def test_freeze_keeps_only_collected_params_trainable(self) -> None:
        task = self._make_task()
        model = _DummyModel()
        chosen = [model.core.weight]
        with mock.patch("tasks.Conan.Conan.Conan", return_value=model):
            with mock.patch("tasks.Conan.Conan.detect_rhythm_stage", return_value="student_kd"):
                with mock.patch.dict(
                    "tasks.Conan.Conan.hparams",
                    {"rhythm_optimize_module_only": True},
                    clear=True,
                ):
                    with mock.patch.object(ConanTask, "_collect_rhythm_gen_params", return_value=chosen):
                        ConanTask.build_tts_model(task)
        self.assertEqual(task.gen_params, chosen)
        self.assertTrue(model.core.weight.requires_grad)
        self.assertFalse(model.extra.weight.requires_grad)

    def test_freeze_raises_when_collector_returns_foreign_param(self) -> None:
        task = self._make_task()
        model = _DummyModel()
        foreign = nn.Parameter(torch.ones(1))
        with mock.patch("tasks.Conan.Conan.Conan", return_value=model):
            with mock.patch("tasks.Conan.Conan.detect_rhythm_stage", return_value="student_kd"):
                with mock.patch.dict(
                    "tasks.Conan.Conan.hparams",
                    {"rhythm_optimize_module_only": True},
                    clear=True,
                ):
                    with mock.patch.object(
                        ConanTask,
                        "_collect_rhythm_gen_params",
                        return_value=[model.core.weight, foreign],
                    ):
                        with self.assertRaisesRegex(RuntimeError, "do not belong to the active model"):
                            ConanTask.build_tts_model(task)


if __name__ == "__main__":
    unittest.main()
