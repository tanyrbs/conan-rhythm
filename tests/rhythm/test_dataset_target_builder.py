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

    def _resolve_primary_target_surface(self):
        return str(self.hparams.get("rhythm_primary_target_surface", "guidance"))

    def _resolve_distill_surface(self):
        return str(self.hparams.get("rhythm_distill_surface", "none"))

    def _resolve_teacher_target_source(self):
        return str(self.hparams.get("rhythm_teacher_target_source", "algorithmic"))


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

    def test_runtime_only_pairwise_teacher_builds_online_teacher_targets_from_reference(self) -> None:
        owner = _DummyOwner()
        owner.hparams.update(
            {
                "rhythm_primary_target_surface": "teacher",
                "rhythm_distill_surface": "none",
                "rhythm_teacher_target_source": "algorithmic",
                "rhythm_dataset_build_teacher_from_ref": True,
                "rhythm_dataset_build_guidance_from_ref": False,
                "rhythm_require_cached_teacher": False,
                "rhythm_use_retimed_target_if_available": False,
                "lambda_rhythm_guidance": 0.0,
                "lambda_rhythm_distill": 0.0,
            }
        )
        builder = RhythmDatasetTargetBuilder(owner)
        source_cache = {
            "dur_anchor_src": np.asarray([3.0, 2.0, 4.0], dtype=np.float32),
            "boundary_confidence": np.asarray([0.0, 1.0, 0.5], dtype=np.float32),
        }
        ref_conditioning = {
            "ref_rhythm_stats": np.asarray([0.20, 2.0, 4.0, 0.10, 0.30, 0.80], dtype=np.float32),
            "ref_rhythm_trace": np.stack(
                [
                    np.full((8,), 0.10, dtype=np.float32),
                    np.linspace(0.0, 1.0, 8, dtype=np.float32),
                    np.linspace(1.0, 0.0, 8, dtype=np.float32),
                    np.linspace(-0.2, 0.2, 8, dtype=np.float32),
                    np.ones((8,), dtype=np.float32),
                ],
                axis=-1,
            ),
        }

        runtime_targets = builder.build_runtime_rhythm_targets(source_cache, ref_conditioning)

        self.assertIn("rhythm_teacher_speech_exec_tgt", runtime_targets)
        self.assertIn("rhythm_teacher_pause_exec_tgt", runtime_targets)
        self.assertIn("rhythm_teacher_speech_budget_tgt", runtime_targets)
        self.assertIn("rhythm_teacher_pause_budget_tgt", runtime_targets)
        self.assertIn("rhythm_teacher_confidence", runtime_targets)
        self.assertNotIn("rhythm_guidance_speech_tgt", runtime_targets)


if __name__ == "__main__":
    unittest.main()
