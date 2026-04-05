from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.Conan import base_gen_task as conan_base_gen_task
from tasks.Emformer import base_gen_task as emformer_base_gen_task


class _SilentLogger:
    def add_audio(self, *args, **kwargs):
        return None

    def add_figure(self, *args, **kwargs):
        return None


class _FakeVocoder:
    def spec2wav(self, mel, f0=None):
        return np.zeros(16, dtype=np.float32)


class _FakePool:
    def __init__(self) -> None:
        self.jobs = []

    def add_job(self, fn, args=None, kwargs=None):
        self.jobs.append((fn, args or [], kwargs or {}))


class BaseGenTaskPitchGuardTests(unittest.TestCase):
    @staticmethod
    def _sample(*, include_source_pitch: bool):
        sample = {
            "mels": torch.zeros(1, 4, 80),
            "item_name": ["demo item"],
            "nsamples": 1,
        }
        if include_source_pitch:
            sample["f0"] = torch.ones(1, 4)
            sample["uv"] = torch.zeros(1, 4)
        else:
            sample["f0"] = None
            sample["uv"] = None
        return sample

    def _make_task(self, module):
        task = module.AuxDecoderMIDITask.__new__(module.AuxDecoderMIDITask)
        task.global_step = 0
        task.logger = _SilentLogger()
        task.vocoder = _FakeVocoder()
        task.plot_mel = lambda *args, **kwargs: None
        task.gen_dir = "tmp"
        task.saving_result_pool = _FakePool()
        task.run_model = lambda sample, infer=True: ({}, {"mel_out": torch.zeros(1, 4, 80)})
        return task

    def _exercise_module_without_predicted_f0(self, module) -> None:
        task = self._make_task(module)
        for include_source_pitch in (True, False):
            task = self._make_task(module)
            sample = self._sample(include_source_pitch=include_source_pitch)
            with mock.patch.dict(
                module.hparams,
                {
                    "num_valid_plots": 1,
                    "audio_sample_rate": 16000,
                    "save_gt": False,
                },
                clear=True,
            ):
                with mock.patch.object(module, "denorm_f0", side_effect=lambda f0, uv: f0):
                    with mock.patch.object(module, "f0_to_figure", return_value=None):
                        validation_outputs = task.validation_step(sample, batch_idx=0)
                        test_outputs = task.test_step(sample, batch_idx=0)
            self.assertIn("losses", validation_outputs)
            self.assertEqual(test_outputs, {})
            self.assertEqual(len(task.saving_result_pool.jobs), 1)

    def test_conan_base_gen_task_tolerates_missing_predicted_f0(self) -> None:
        self._exercise_module_without_predicted_f0(conan_base_gen_task)

    def test_emformer_base_gen_task_tolerates_missing_predicted_f0(self) -> None:
        self._exercise_module_without_predicted_f0(emformer_base_gen_task)


if __name__ == "__main__":
    unittest.main()
