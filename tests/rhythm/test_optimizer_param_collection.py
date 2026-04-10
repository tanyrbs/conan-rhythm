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

from tasks.Conan.rhythm.task_mixin import RhythmConanTaskMixin


class _DummyOfflineTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.core = nn.Linear(2, 2, bias=False)
        self.confidence_trunk = nn.Linear(2, 2, bias=False)
        self.confidence_heads = nn.Linear(2, 1, bias=False)


class _DummyRhythmModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.core = nn.Linear(2, 2, bias=False)
        self.unit_embedding = nn.Embedding(4, 2)
        self.reference_descriptor = nn.Linear(2, 2, bias=False)
        self.offline_teacher = _DummyOfflineTeacher()


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rhythm_enable_v2 = True
        self.rhythm_module = _DummyRhythmModule()
        self.rhythm_pause_state = nn.Parameter(torch.ones(1))
        self.rhythm_render_phase_mlp = nn.Linear(2, 2, bias=False)
        self.rhythm_render_phase_gain = nn.Parameter(torch.ones(1))


class _DummyV3Frontend(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.baseline = nn.Linear(2, 2, bias=False)

    def get_baseline_module(self):
        return self.baseline


class _DummyV3Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rhythm_enable_v2 = False
        self.rhythm_enable_v3 = True
        self.rhythm_enabled = True
        self.rhythm_module = _DummyRhythmModule()
        self.rhythm_unit_frontend = _DummyV3Frontend()
        self.rhythm_pause_state = None
        self.rhythm_render_phase_mlp = None
        self.rhythm_render_phase_gain = None


class _DummyTask(RhythmConanTaskMixin):
    def __init__(self) -> None:
        self.model = _DummyModel()


class _DummyV3Task(RhythmConanTaskMixin):
    def __init__(self) -> None:
        self.model = _DummyV3Model()


class OptimizerParamCollectionTests(unittest.TestCase):
    def test_collect_rhythm_gen_params_skips_offline_confidence_heads_by_default(self) -> None:
        task = _DummyTask()
        confidence_ids = {
            id(param)
            for param in list(task.model.rhythm_module.offline_teacher.confidence_trunk.parameters())
            + list(task.model.rhythm_module.offline_teacher.confidence_heads.parameters())
        }
        with mock.patch.dict(
            "tasks.Conan.rhythm.task_mixin.hparams",
            {
                "rhythm_optimize_pause_state": False,
                "rhythm_train_offline_confidence_heads": False,
                "rhythm_apply_train_override": False,
            },
            clear=True,
        ):
            params = task._collect_rhythm_gen_params()
        self.assertGreater(len(params), 0)
        self.assertTrue(all(id(param) not in confidence_ids for param in params))

    def test_collect_rhythm_gen_params_can_include_pause_and_render_params(self) -> None:
        task = _DummyTask()
        pause_state_id = id(task.model.rhythm_pause_state)
        render_ids = {
            id(task.model.rhythm_render_phase_gain),
            *(id(param) for param in task.model.rhythm_render_phase_mlp.parameters()),
        }
        with mock.patch.dict(
            "tasks.Conan.rhythm.task_mixin.hparams",
            {
                "rhythm_optimize_pause_state": True,
                "rhythm_train_offline_confidence_heads": False,
                "rhythm_apply_train_override": True,
            },
            clear=True,
        ):
            params = task._collect_rhythm_gen_params()
        ids = {id(param) for param in params}
        self.assertIn(pause_state_id, ids)
        self.assertTrue(render_ids.issubset(ids))

    def test_collect_offline_teacher_gen_params_stays_teacher_scoped(self) -> None:
        task = _DummyTask()
        confidence_ids = {
            id(param)
            for param in list(task.model.rhythm_module.offline_teacher.confidence_trunk.parameters())
            + list(task.model.rhythm_module.offline_teacher.confidence_heads.parameters())
        }
        with mock.patch.dict(
            "tasks.Conan.rhythm.task_mixin.hparams",
            {
                "rhythm_train_offline_confidence_heads": False,
            },
            clear=True,
        ):
            params = task._collect_offline_teacher_gen_params()
        ids = {id(param) for param in params}
        self.assertTrue(ids)
        self.assertIn(id(task.model.rhythm_module.unit_embedding.weight), ids)
        self.assertTrue(
            all(id(param) not in confidence_ids for param in params),
        )

    def test_collect_rhythm_gen_params_includes_v3_baseline_when_joint_train_mode(self) -> None:
        task = _DummyV3Task()
        baseline_ids = {id(param) for param in task.model.rhythm_unit_frontend.baseline.parameters()}
        with mock.patch.dict(
            "tasks.Conan.rhythm.task_mixin.hparams",
            {
                "rhythm_v3_baseline_train_mode": "joint",
                "rhythm_v3_freeze_baseline": False,
                "rhythm_train_offline_confidence_heads": False,
            },
            clear=True,
        ):
            params = task._collect_rhythm_gen_params()
        ids = {id(param) for param in params}
        self.assertTrue(baseline_ids.issubset(ids))

    def test_collect_rhythm_gen_params_skips_v3_baseline_when_frozen(self) -> None:
        task = _DummyV3Task()
        baseline_ids = {id(param) for param in task.model.rhythm_unit_frontend.baseline.parameters()}
        with mock.patch.dict(
            "tasks.Conan.rhythm.task_mixin.hparams",
            {
                "rhythm_v3_baseline_train_mode": "frozen",
                "rhythm_v3_freeze_baseline": True,
                "rhythm_train_offline_confidence_heads": False,
            },
            clear=True,
        ):
            params = task._collect_rhythm_gen_params()
        ids = {id(param) for param in params}
        self.assertTrue(ids.isdisjoint(baseline_ids))

    def test_collect_rhythm_v3_baseline_only_params_is_baseline_scoped(self) -> None:
        task = _DummyV3Task()
        baseline_ids = {id(param) for param in task.model.rhythm_unit_frontend.baseline.parameters()}
        params = task._collect_rhythm_v3_baseline_only_params()
        ids = {id(param) for param in params}
        self.assertEqual(ids, baseline_ids)


if __name__ == "__main__":
    unittest.main()
