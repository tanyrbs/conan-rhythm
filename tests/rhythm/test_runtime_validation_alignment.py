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


class _DummyCallableModel:
    rhythm_enable_v2 = True
    rhythm_minimal_style_only = False

    def __call__(self, content, **kwargs):
        batch, steps = content.shape
        return {
            "mel_out": torch.zeros(batch, steps, 80),
        }


class _ModuleOnlyValidationTask(RhythmConanTaskMixin):
    def __init__(self) -> None:
        self.model = _DummyCallableModel()
        self.global_step = 0
        self.mel_losses = {"l1": 1.0}
        self._warned_retimed_pitch_supervision = False

    def _add_acoustic_loss(self, mel_out, target, losses, *, frame_weight=None):
        losses["acoustic_called"] = torch.tensor(1.0)

    def add_pitch_loss(self, output, sample, losses):
        losses["pitch_called"] = torch.tensor(1.0)

    def add_rhythm_loss(self, output, sample, losses):
        losses["rhythm_exec_speech"] = torch.tensor(1.0)


class _TeacherOfflineEvalTask(RhythmConanTaskMixin):
    def __init__(self) -> None:
        self.model = _DummyCallableModel()
        self.global_step = 0
        self.mel_losses = {"l1": 1.0}
        self._warned_retimed_pitch_supervision = False

    def _run_offline_teacher_model(self, sample, *, infer: bool, test: bool, **kwargs):
        return {"rhythm_exec_speech": torch.tensor(1.0)}, {"rhythm_teacher_only_stage": 1.0}


class _SilentLogger:
    def add_audio(self, *args, **kwargs):
        return None

    def add_figure(self, *args, **kwargs):
        return None


class _ValidationWithoutMelTask(RhythmConanTaskMixin):
    def __init__(self) -> None:
        self.global_step = 0
        self.mel_losses = {"l1": 1.0}
        self.logger = _SilentLogger()
        self.vocoder = object()

    def run_model(self, sample, infer=False, test=False, **kwargs):
        return {"rhythm_exec_speech": torch.tensor(1.0)}, {"rhythm_teacher_only_stage": 1.0}


class _TrainingObservabilityTask(RhythmConanTaskMixin):
    def __init__(self) -> None:
        self.global_step = 0
        self.mel_disc = None
        self.model = object()

    def run_model(self, sample, infer=False, test=False, **kwargs):
        return (
            {"L_base": torch.tensor(1.0, requires_grad=True)},
            {
                "disable_acoustic_train_path": 1.0,
                "rhythm_module_only_objective": 1.0,
                "rhythm_skip_acoustic_objective": 1.0,
                "rhythm_pitch_supervision_disabled": 1.0,
                "rhythm_missing_retimed_pitch_target": 1.0,
                "acoustic_target_is_retimed": 1.0,
                "acoustic_target_length_delta_before_align": 12.0,
                "acoustic_target_length_mismatch_abs_before_align": 12.0,
                "acoustic_target_resampled_to_output": 0.0,
                "acoustic_target_trimmed_to_output": 1.0,
                "retimed_pitch_target_length_mismatch_abs_before_align": 7.0,
                "retimed_pitch_target_resampled_to_output": 1.0,
                "retimed_pitch_target_trimmed_to_output": 0.0,
            },
        )


class RuntimeValidationAlignmentTests(unittest.TestCase):
    @staticmethod
    def _sample():
        return {
            "content": torch.ones(1, 4, dtype=torch.long),
            "mels": torch.zeros(1, 4, 80),
            "ref_mels": torch.zeros(1, 4, 80),
            "mel_lengths": torch.tensor([4]),
            "nsamples": 1,
            "f0": None,
            "uv": None,
        }

    def test_module_only_validation_skips_acoustic_and_pitch_objectives(self) -> None:
        task = _ModuleOnlyValidationTask()
        with mock.patch.dict(
            "tasks.Conan.rhythm.task_mixin.hparams",
            {
                "random_speaker_steps": 0,
                "rhythm_stage": "student_kd",
                "rhythm_enable_v2": True,
                "rhythm_optimize_module_only": True,
                "rhythm_fastpath_disable_acoustic_when_module_only": True,
                "rhythm_apply_train_override": False,
                "rhythm_apply_valid_override": False,
                "rhythm_use_retimed_target_if_available": False,
                "rhythm_force_reference_conditioning": False,
                "style": False,
                "rhythm_compact_joint_loss": False,
            },
            clear=True,
        ):
            losses, output = task.run_model(self._sample(), infer=True)
        self.assertIn("rhythm_exec_speech", losses)
        self.assertNotIn("acoustic_called", losses)
        self.assertNotIn("pitch_called", losses)
        self.assertEqual(output["rhythm_module_only_objective"], 1.0)
        self.assertEqual(output["rhythm_skip_acoustic_objective"], 1.0)

    def test_teacher_offline_validation_uses_teacher_only_branch(self) -> None:
        task = _TeacherOfflineEvalTask()
        with mock.patch.dict(
            "tasks.Conan.rhythm.task_mixin.hparams",
            {
                "random_speaker_steps": 0,
                "rhythm_stage": "teacher_offline",
                "rhythm_enable_v2": True,
                "rhythm_optimize_module_only": True,
                "rhythm_fastpath_disable_acoustic_when_module_only": True,
                "rhythm_apply_train_override": False,
                "rhythm_apply_valid_override": False,
                "rhythm_use_retimed_target_if_available": False,
                "rhythm_force_reference_conditioning": False,
                "style": False,
            },
            clear=True,
        ):
            losses, output = task.run_model(self._sample(), infer=True)
        self.assertIn("rhythm_exec_speech", losses)
        self.assertEqual(output["rhythm_teacher_only_stage"], 1.0)

    def test_validation_step_tolerates_teacher_only_output_without_mel(self) -> None:
        task = _ValidationWithoutMelTask()
        with mock.patch.dict(
            "tasks.Conan.rhythm.task_mixin.hparams",
            {
                "num_valid_plots": 1,
                "audio_sample_rate": 16000,
            },
            clear=True,
        ):
            with mock.patch("tasks.Conan.rhythm.task_mixin.build_rhythm_metric_dict", return_value={}):
                with mock.patch("tasks.Conan.rhythm.task_mixin.compute_reporting_total_loss", return_value=1.0):
                    outputs = task.validation_step(self._sample(), batch_idx=0)
        self.assertEqual(outputs["total_loss"], 1.0)

    def test_training_step_exposes_runtime_observability_scalars(self) -> None:
        task = _TrainingObservabilityTask()
        with mock.patch.dict(
            "tasks.Conan.rhythm.task_mixin.hparams",
            {
                "disc_start_steps": 999999,
                "lambda_mel_adv": 0.0,
            },
            clear=True,
        ):
            total_loss, loss_output = task._training_step(self._sample(), batch_idx=0, optimizer_idx=0)
        self.assertAlmostEqual(float(total_loss.detach()), 1.0)
        self.assertEqual(loss_output["batch_size"], 1)
        self.assertAlmostEqual(float(loss_output["rhythm_metric_disable_acoustic_train_path"]), 1.0)
        self.assertAlmostEqual(float(loss_output["rhythm_metric_module_only_objective"]), 1.0)
        self.assertAlmostEqual(float(loss_output["rhythm_metric_skip_acoustic_objective"]), 1.0)
        self.assertAlmostEqual(float(loss_output["rhythm_metric_pitch_supervision_disabled"]), 1.0)
        self.assertAlmostEqual(float(loss_output["rhythm_metric_missing_retimed_pitch_target"]), 1.0)
        self.assertAlmostEqual(float(loss_output["rhythm_metric_acoustic_target_is_retimed"]), 1.0)
        self.assertAlmostEqual(float(loss_output["rhythm_metric_acoustic_target_length_delta_before_align"]), 12.0)
        self.assertAlmostEqual(
            float(loss_output["rhythm_metric_acoustic_target_length_mismatch_abs_before_align"]),
            12.0,
        )
        self.assertAlmostEqual(float(loss_output["rhythm_metric_acoustic_target_resampled_to_output"]), 0.0)
        self.assertAlmostEqual(float(loss_output["rhythm_metric_acoustic_target_trimmed_to_output"]), 1.0)
        self.assertAlmostEqual(
            float(loss_output["rhythm_metric_retimed_pitch_target_length_mismatch_abs_before_align"]),
            7.0,
        )
        self.assertAlmostEqual(float(loss_output["rhythm_metric_retimed_pitch_target_resampled_to_output"]), 1.0)
        self.assertAlmostEqual(float(loss_output["rhythm_metric_retimed_pitch_target_trimmed_to_output"]), 0.0)


if __name__ == "__main__":
    unittest.main()
