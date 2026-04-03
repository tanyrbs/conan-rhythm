from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.Conan.rhythm.targets import (
    DistillConfidenceBundle,
    RhythmTargetBuildConfig,
    build_rhythm_loss_targets_from_sample,
)


class RhythmTargetBuilderTests(unittest.TestCase):
    @staticmethod
    def _config() -> RhythmTargetBuildConfig:
        return RhythmTargetBuildConfig(
            primary_target_surface="guidance",
            distill_surface="cache",
            lambda_guidance=0.0,
            lambda_distill=0.5,
            distill_budget_weight=0.25,
            distill_allocation_weight=0.25,
            distill_prefix_weight=0.25,
            distill_speech_shape_weight=0.0,
            distill_pause_shape_weight=0.0,
            plan_local_weight=0.5,
            plan_cum_weight=1.0,
            pause_boundary_weight=0.35,
            budget_raw_weight=1.0,
            budget_exec_weight=0.25,
            feasible_debt_weight=0.05,
        )

    def test_primary_targets_survive_missing_distill_surface(self) -> None:
        unit_batch = SimpleNamespace(
            dur_anchor_src=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
            unit_mask=torch.ones((1, 2), dtype=torch.float32),
        )
        sample = {
            "rhythm_speech_exec_tgt": torch.tensor([[1.0, 1.0]], dtype=torch.float32),
            "rhythm_pause_exec_tgt": torch.tensor([[0.0, 0.0]], dtype=torch.float32),
            "rhythm_speech_budget_tgt": torch.tensor([[2.0]], dtype=torch.float32),
            "rhythm_pause_budget_tgt": torch.tensor([[0.0]], dtype=torch.float32),
            "rhythm_target_confidence": torch.tensor([[1.5]], dtype=torch.float32),
        }

        def _normalize_distill(confidence, *, batch_size: int, device: torch.device):
            return torch.ones((batch_size, 1), device=device)

        def _normalize_component(confidence, *, fallback_confidence: torch.Tensor, batch_size: int, device: torch.device):
            del confidence, batch_size, device
            return fallback_confidence

        targets = build_rhythm_loss_targets_from_sample(
            sample=sample,
            unit_batch=unit_batch,
            config=self._config(),
            runtime_teacher=None,
            algorithmic_teacher=None,
            offline_confidences=DistillConfidenceBundle(),
            normalize_distill_confidence=_normalize_distill,
            normalize_component_confidence=_normalize_component,
            build_prefix_carry_from_exec=lambda speech, pause, dur_anchor_src, unit_mask: (
                torch.zeros_like(speech),
                torch.zeros_like(speech),
            ),
            slice_rhythm_surface_to_student=lambda **kwargs: (
                kwargs["speech_exec"],
                kwargs["pause_exec"],
                kwargs.get("speech_budget"),
                kwargs.get("pause_budget"),
                kwargs.get("allocation"),
                kwargs.get("prefix_clock"),
                kwargs.get("prefix_backlog"),
            ),
        )
        self.assertIsNotNone(targets)
        self.assertIsNone(targets.distill_speech_tgt)
        self.assertIsNone(targets.distill_pause_tgt)
        self.assertTrue(torch.allclose(targets.sample_confidence, torch.ones((1, 1), dtype=torch.float32)))


if __name__ == "__main__":
    unittest.main()
