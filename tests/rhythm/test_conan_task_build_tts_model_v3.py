from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.Conan.Conan import ConanTask


class _DummyV3Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rhythm_enable_v2 = False
        self.rhythm_enable_v3 = True
        self.core = nn.Linear(2, 2, bias=False)
        self.extra = nn.Linear(2, 2, bias=False)
        self.rhythm_unit_frontend = mock.Mock()


class ConanTaskBuildTTSModelV3Tests(unittest.TestCase):
    @staticmethod
    def _make_task() -> ConanTask:
        task = ConanTask.__new__(ConanTask)
        nn.Module.__init__(task)
        return task

    def test_teacher_offline_stage_does_not_use_teacher_only_branch_for_v3(self) -> None:
        task = self._make_task()
        model = _DummyV3Model()
        with mock.patch("tasks.Conan.Conan.Conan", return_value=model):
            with mock.patch("tasks.Conan.Conan.detect_rhythm_stage", return_value="teacher_offline"):
                with mock.patch.dict(
                    "tasks.Conan.Conan.hparams",
                    {"rhythm_optimize_module_only": False},
                    clear=True,
                ):
                    with mock.patch.object(
                        ConanTask,
                        "_collect_offline_teacher_gen_params",
                        side_effect=AssertionError("v3 should not use offline teacher params"),
                    ):
                        ConanTask.build_tts_model(task)
        self.assertEqual(task.gen_params, [p for p in model.parameters() if p.requires_grad])

    def test_module_only_stage_uses_rhythm_param_collector_for_v3(self) -> None:
        task = self._make_task()
        model = _DummyV3Model()
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

    def test_default_v3_stage_keeps_all_trainable_params(self) -> None:
        task = self._make_task()
        model = _DummyV3Model()
        with mock.patch("tasks.Conan.Conan.Conan", return_value=model):
            with mock.patch("tasks.Conan.Conan.detect_rhythm_stage", return_value="student_kd"):
                with mock.patch.dict(
                    "tasks.Conan.Conan.hparams",
                    {"rhythm_optimize_module_only": False},
                    clear=True,
                ):
                    ConanTask.build_tts_model(task)
        self.assertEqual(task.gen_params, [p for p in model.parameters() if p.requires_grad])

    def test_build_tts_model_warmstarts_and_freezes_v3_baseline_when_requested(self) -> None:
        task = self._make_task()
        model = _DummyV3Model()
        with mock.patch("tasks.Conan.Conan.Conan", return_value=model):
            with mock.patch("tasks.Conan.Conan.detect_rhythm_stage", return_value="student_kd"):
                with mock.patch.dict(
                    "tasks.Conan.Conan.hparams",
                    {
                        "rhythm_optimize_module_only": False,
                        "rhythm_baseline_table_prior_path": "prior.pt",
                        "rhythm_v3_baseline_ckpt": "baseline.ckpt",
                        "rhythm_v3_baseline_train_mode": "frozen",
                    },
                    clear=True,
                ):
                    ConanTask.build_tts_model(task)
        model.rhythm_unit_frontend.load_table_prior_file.assert_called_once_with("prior.pt")
        model.rhythm_unit_frontend.load_baseline_checkpoint.assert_called_once_with("baseline.ckpt", strict=True)
        model.rhythm_unit_frontend.freeze_baseline.assert_called_once_with()
        model.rhythm_unit_frontend.unfreeze_baseline.assert_not_called()

    def test_build_tts_model_unfreezes_v3_baseline_for_joint_mode(self) -> None:
        task = self._make_task()
        model = _DummyV3Model()
        with mock.patch("tasks.Conan.Conan.Conan", return_value=model):
            with mock.patch("tasks.Conan.Conan.detect_rhythm_stage", return_value="student_kd"):
                with mock.patch.dict(
                    "tasks.Conan.Conan.hparams",
                    {
                        "rhythm_optimize_module_only": False,
                        "rhythm_v3_baseline_train_mode": "joint",
                        "rhythm_v3_freeze_baseline": False,
                    },
                    clear=True,
                ):
                    ConanTask.build_tts_model(task)
        model.rhythm_unit_frontend.unfreeze_baseline.assert_called_once_with()
        model.rhythm_unit_frontend.freeze_baseline.assert_not_called()

    def test_build_tts_model_baseline_pretrain_freezes_to_baseline_params_only(self) -> None:
        task = self._make_task()
        model = _DummyV3Model()
        baseline_params = [model.core.weight]
        with mock.patch("tasks.Conan.Conan.Conan", return_value=model):
            with mock.patch.dict(
                "tasks.Conan.Conan.hparams",
                {
                    "rhythm_v3_baseline_train_mode": "pretrain",
                    "lambda_rhythm_base": 1.0,
                },
                clear=True,
            ):
                with mock.patch.object(
                    ConanTask,
                    "_collect_rhythm_v3_baseline_only_params",
                    return_value=baseline_params,
                ):
                    ConanTask.build_tts_model(task)
        self.assertEqual(task.gen_params, baseline_params)
        self.assertTrue(model.core.weight.requires_grad)
        self.assertFalse(model.extra.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
