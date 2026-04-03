from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.Conan.rhythm.dataset_target_builder import RhythmDatasetTargetBuilder


class _DummyOwner:
    _RHYTHM_TARGET_KEYS = (
        "rhythm_speech_exec_tgt",
        "rhythm_pause_exec_tgt",
        "rhythm_teacher_speech_exec_tgt",
        "rhythm_teacher_pause_exec_tgt",
        "rhythm_teacher_allocation_tgt",
        "rhythm_teacher_prefix_clock_tgt",
        "rhythm_teacher_prefix_backlog_tgt",
        "rhythm_speech_budget_tgt",
        "rhythm_pause_budget_tgt",
        "rhythm_teacher_speech_budget_tgt",
        "rhythm_teacher_pause_budget_tgt",
    )
    _RHYTHM_META_KEYS = ("rhythm_retimed_target_source_id",)

    def __init__(self) -> None:
        self.hparams = {
            "rhythm_retimed_pause_frame_weight": 0.2,
            "rhythm_retimed_stretch_weight_min": 0.35,
        }


class RhythmDatasetTargetBuilderTests(unittest.TestCase):
    def test_adapt_cached_targets_to_prefix_refreshes_budgets_and_teacher_prefix(self) -> None:
        builder = RhythmDatasetTargetBuilder(_DummyOwner())
        item = {
            "dur_anchor_src": np.asarray([3.0, 4.0, 5.0], dtype=np.float32),
        }
        cached_targets = {
            "rhythm_speech_exec_tgt": np.asarray([3.0, 4.0, 5.0], dtype=np.float32),
            "rhythm_pause_exec_tgt": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            "rhythm_teacher_speech_exec_tgt": np.asarray([3.0, 4.0, 5.0], dtype=np.float32),
            "rhythm_teacher_pause_exec_tgt": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            "rhythm_teacher_allocation_tgt": np.asarray([0.2, 0.3, 0.5], dtype=np.float32),
        }
        source_cache = {
            "dur_anchor_src": np.asarray([3.0, 2.0], dtype=np.float32),
        }
        sample = {"mel": torch.zeros((5, 80), dtype=torch.float32)}

        adapted = builder.adapt_cached_targets_to_prefix(
            item=item,
            cached_targets=cached_targets,
            source_cache=source_cache,
            sample=sample,
        )

        self.assertTrue(np.allclose(adapted["rhythm_speech_exec_tgt"], np.asarray([3.0, 2.0], dtype=np.float32)))
        self.assertTrue(np.allclose(adapted["rhythm_pause_exec_tgt"], np.asarray([1.0, 1.0], dtype=np.float32)))
        self.assertTrue(np.allclose(adapted["rhythm_speech_budget_tgt"], np.asarray([5.0], dtype=np.float32)))
        self.assertTrue(np.allclose(adapted["rhythm_pause_budget_tgt"], np.asarray([2.0], dtype=np.float32)))
        self.assertTrue(np.allclose(adapted["rhythm_blank_budget_tgt"], adapted["rhythm_pause_budget_tgt"]))
        self.assertAlmostEqual(float(adapted["rhythm_teacher_allocation_tgt"].sum()), 1.0, places=6)
        self.assertEqual(adapted["rhythm_teacher_prefix_clock_tgt"].shape[0], 2)
        self.assertEqual(adapted["rhythm_teacher_prefix_backlog_tgt"].shape[0], 2)


if __name__ == "__main__":
    unittest.main()
