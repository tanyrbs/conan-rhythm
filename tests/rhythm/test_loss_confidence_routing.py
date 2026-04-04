from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.Conan.rhythm.distill_confidence import (
    normalize_component_distill_confidence,
    normalize_distill_confidence,
)
from tasks.Conan.rhythm.losses import RhythmLossTargets, build_rhythm_loss_dict
from tasks.Conan.rhythm.targets import DistillConfidenceBundle, _normalize_distill_confidences


class RhythmLossConfidenceRoutingTests(unittest.TestCase):
    def test_shared_distill_confidence_keeps_floor_semantics_when_zero_and_preserve_zeros_disabled(self) -> None:
        normalized = normalize_distill_confidence(
            torch.tensor([[0.00], [0.01]], dtype=torch.float32),
            batch_size=2,
            device=torch.device("cpu"),
            floor=0.05,
            power=1.0,
            preserve_zeros=False,
        )

        self.assertTrue(torch.allclose(normalized[0:1], torch.tensor([[0.05]])))
        self.assertTrue(torch.allclose(normalized[1:2], torch.tensor([[0.05]])))

    def test_shape_distill_uses_dedicated_shape_confidence(self) -> None:
        execution = SimpleNamespace(
            speech_duration_exec=torch.tensor([[2.0, 0.0]], dtype=torch.float32),
            pause_after_exec=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
            planner=SimpleNamespace(
                speech_budget_win=torch.tensor([[2.0]], dtype=torch.float32),
                pause_budget_win=torch.tensor([[0.0]], dtype=torch.float32),
                raw_speech_budget_win=torch.tensor([[2.0]], dtype=torch.float32),
                raw_pause_budget_win=torch.tensor([[0.0]], dtype=torch.float32),
                boundary_score_unit=torch.zeros((1, 2), dtype=torch.float32),
                trace_context=torch.zeros((1, 2, 3), dtype=torch.float32),
                source_boundary_cue=torch.zeros((1, 2), dtype=torch.float32),
                feasible_total_budget_delta=torch.zeros((1, 1), dtype=torch.float32),
            ),
        )
        targets = RhythmLossTargets(
            speech_exec_tgt=torch.tensor([[2.0, 0.0]], dtype=torch.float32),
            pause_exec_tgt=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
            speech_budget_tgt=torch.tensor([[2.0]], dtype=torch.float32),
            pause_budget_tgt=torch.tensor([[0.0]], dtype=torch.float32),
            unit_mask=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
            dur_anchor_src=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
            distill_speech_tgt=torch.tensor([[0.0, 2.0]], dtype=torch.float32),
            distill_pause_tgt=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
            distill_exec_confidence=torch.tensor([[1.0]], dtype=torch.float32),
            distill_shape_confidence=torch.tensor([[0.0]], dtype=torch.float32),
            distill_speech_shape_weight=1.0,
            distill_pause_shape_weight=0.0,
            distill_allocation_weight=0.0,
            pause_boundary_weight=0.0,
        )

        losses = build_rhythm_loss_dict(execution, targets)

        self.assertGreater(float(losses["rhythm_distill_exec"].item()), 0.0)
        self.assertTrue(torch.allclose(losses["rhythm_distill_speech_shape"], torch.tensor(0.0)))
        self.assertTrue(torch.allclose(losses["rhythm_distill"], losses["rhythm_distill_exec"]))

    def test_component_distill_confidence_preserves_explicit_zero_gate(self) -> None:
        normalized = _normalize_distill_confidences(
            confidence_bundle=DistillConfidenceBundle(
                shared=torch.tensor([[0.80], [0.60]], dtype=torch.float32),
                exec=torch.tensor([[0.00], [0.01]], dtype=torch.float32),
                shape=torch.tensor([[0.00], [0.02]], dtype=torch.float32),
            ),
            batch_size=2,
            device=torch.device("cpu"),
            normalize_distill_confidence=lambda confidence, **kwargs: normalize_distill_confidence(
                confidence,
                floor=0.05,
                power=1.0,
                **kwargs,
            ),
            normalize_component_confidence=lambda confidence, **kwargs: normalize_component_distill_confidence(
                confidence,
                floor=0.05,
                power=1.0,
                preserve_zeros=True,
                **kwargs,
            ),
        )

        self.assertTrue(torch.allclose(normalized.exec[0:1], torch.tensor([[0.0]])))
        self.assertTrue(torch.allclose(normalized.shape[0:1], torch.tensor([[0.0]])))
        self.assertTrue(torch.allclose(normalized.exec[1:2], torch.tensor([[0.05]])))
        self.assertTrue(torch.allclose(normalized.shape[1:2], torch.tensor([[0.05]])))


if __name__ == "__main__":
    unittest.main()
