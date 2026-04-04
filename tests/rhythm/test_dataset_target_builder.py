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

    def test_prefix_alignment_tolerates_float_noise(self) -> None:
        alignment = RhythmDatasetTargetBuilder._resolve_prefix_alignment(
            {"dur_anchor_src": np.asarray([2.0, 3.0], dtype=np.float32)},
            {"dur_anchor_src": np.asarray([2.0, 3.0 + 1e-5], dtype=np.float32)},
        )
        self.assertFalse(alignment["is_truncated"])

    def test_prefix_tail_ratio_is_clamped_to_unit_interval(self) -> None:
        builder = RhythmDatasetTargetBuilder(_DummyOwner())
        adapted = builder.adapt_cached_targets_to_prefix(
            item={"dur_anchor_src": np.asarray([3.0, 2.0], dtype=np.float32)},
            cached_targets={
                "rhythm_speech_exec_tgt": np.asarray([3.0, 4.0], dtype=np.float32),
                "rhythm_pause_exec_tgt": np.asarray([1.0, 2.0], dtype=np.float32),
                "rhythm_teacher_speech_exec_tgt": np.asarray([3.0, 4.0], dtype=np.float32),
                "rhythm_teacher_pause_exec_tgt": np.asarray([1.0, 2.0], dtype=np.float32),
                "rhythm_teacher_allocation_tgt": np.asarray([0.3, 0.7], dtype=np.float32),
            },
            source_cache={"dur_anchor_src": np.asarray([3.0, 5.0], dtype=np.float32)},
            sample={"mel": torch.zeros((8, 80), dtype=torch.float32)},
        )
        self.assertTrue(np.all(adapted["rhythm_speech_exec_tgt"] <= np.asarray([3.0, 4.0], dtype=np.float32) + 1e-6))
        self.assertTrue(np.all(adapted["rhythm_pause_exec_tgt"] <= np.asarray([1.0, 2.0], dtype=np.float32) + 1e-6))

    def test_prefix_tail_ratio_negative_is_clamped_to_zero(self) -> None:
        adapted = {"rhythm_speech_exec_tgt": np.asarray([1.0, 2.0], dtype=np.float32)}
        RhythmDatasetTargetBuilder._apply_prefix_tail_ratio(adapted, tail_ratio=-0.5)
        self.assertTrue(np.allclose(adapted["rhythm_speech_exec_tgt"], np.asarray([1.0, 0.0], dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
