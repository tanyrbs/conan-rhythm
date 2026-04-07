from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.Conan.rhythm.runtime_adapter import ConanRhythmAdapter


class PauseTopkScheduleTests(unittest.TestCase):
    @staticmethod
    def _build_adapter(**hparams):
        adapter = object.__new__(ConanRhythmAdapter)
        adapter.hparams = hparams
        return adapter

    def test_pause_topk_schedule_uses_absolute_step_by_default(self) -> None:
        adapter = self._build_adapter(
            rhythm_projector_pause_topk_ratio=0.42,
            rhythm_projector_pause_topk_ratio_train_start=0.40,
            rhythm_projector_pause_topk_ratio_train_end=0.42,
            rhythm_projector_pause_topk_ratio_anneal_steps=10000,
        )
        self.assertAlmostEqual(adapter.resolve_pause_topk_ratio(infer=False, global_steps=0), 0.40)
        self.assertAlmostEqual(adapter.resolve_pause_topk_ratio(infer=False, global_steps=5000), 0.41)
        self.assertAlmostEqual(adapter.resolve_pause_topk_ratio(infer=False, global_steps=10000), 0.42)

    def test_pause_topk_schedule_can_anchor_to_resume_step(self) -> None:
        adapter = self._build_adapter(
            rhythm_projector_pause_topk_ratio=0.42,
            rhythm_projector_pause_topk_ratio_train_start=0.40,
            rhythm_projector_pause_topk_ratio_train_end=0.42,
            rhythm_projector_pause_topk_ratio_anneal_steps=10000,
            rhythm_projector_pause_topk_anchor_step=105000,
        )
        self.assertAlmostEqual(adapter.resolve_pause_topk_ratio(infer=False, global_steps=105000), 0.40)
        self.assertAlmostEqual(adapter.resolve_pause_topk_ratio(infer=False, global_steps=110000), 0.41)
        self.assertAlmostEqual(adapter.resolve_pause_topk_ratio(infer=False, global_steps=115000), 0.42)

    def test_pause_topk_schedule_anchor_does_not_change_infer_ratio(self) -> None:
        adapter = self._build_adapter(
            rhythm_projector_pause_topk_ratio=0.42,
            rhythm_projector_pause_topk_ratio_train_start=0.40,
            rhythm_projector_pause_topk_ratio_train_end=0.42,
            rhythm_projector_pause_topk_ratio_anneal_steps=10000,
            rhythm_projector_pause_topk_anchor_step=105000,
        )
        self.assertAlmostEqual(adapter.resolve_pause_topk_ratio(infer=True, global_steps=105000), 0.42)


if __name__ == "__main__":
    unittest.main()
