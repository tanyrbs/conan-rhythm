from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.Conan.rhythm.common.losses_impl import RhythmLossTargets, build_rhythm_loss_dict
from tasks.Conan.rhythm.common.targets_impl import (
    DistillConfidenceBundle,
    RhythmTargetBuildConfig,
    build_rhythm_loss_targets_from_sample,
)


class SRMDPLossIntegrationTests(unittest.TestCase):
    @staticmethod
    def _execution(
        speech: torch.Tensor,
        pause: torch.Tensor,
        *,
        dur_logratio_unit: torch.Tensor,
        role_memory_top_index_unit: torch.Tensor,
    ):
        speech_budget = speech.sum(dim=1, keepdim=True)
        pause_budget = pause.sum(dim=1, keepdim=True)
        planner = SimpleNamespace(
            raw_speech_budget_win=speech_budget,
            raw_pause_budget_win=pause_budget,
            speech_budget_win=speech_budget,
            pause_budget_win=pause_budget,
            feasible_total_budget_delta=torch.zeros_like(speech_budget),
            source_boundary_cue=torch.zeros_like(speech),
            dur_logratio_unit=dur_logratio_unit,
            role_memory_top_index_unit=role_memory_top_index_unit,
        )
        return SimpleNamespace(
            speech_duration_exec=speech,
            blank_duration_exec=pause,
            pause_after_exec=pause,
            planner=planner,
        )

    @staticmethod
    def _base_targets(
        *,
        speech_tgt: torch.Tensor,
        pause_tgt: torch.Tensor,
        srmdp_notimeline_weight: float = 0.0,
        srmdp_role_consistency_weight: float = 0.0,
        srmdp_memory_role_weight: float = 0.0,
        srmdp_role_id_src_tgt: torch.Tensor | None = None,
        srmdp_ref_memory_role_id_tgt: torch.Tensor | None = None,
    ) -> RhythmLossTargets:
        return RhythmLossTargets(
            speech_exec_tgt=speech_tgt,
            pause_exec_tgt=pause_tgt,
            speech_budget_tgt=speech_tgt.sum(dim=1, keepdim=True),
            pause_budget_tgt=pause_tgt.sum(dim=1, keepdim=True),
            unit_mask=torch.ones_like(speech_tgt),
            dur_anchor_src=torch.ones_like(speech_tgt),
            plan_local_weight=0.0,
            plan_cum_weight=0.0,
            unit_logratio_weight=0.0,
            srmdp_notimeline_weight=srmdp_notimeline_weight,
            srmdp_role_consistency_weight=srmdp_role_consistency_weight,
            srmdp_memory_role_weight=srmdp_memory_role_weight,
            srmdp_role_id_src_tgt=srmdp_role_id_src_tgt,
            srmdp_ref_memory_role_id_tgt=srmdp_ref_memory_role_id_tgt,
        )

    def test_long_stream_input_does_not_force_tail_retrieval(self) -> None:
        speech = torch.ones((1, 8), dtype=torch.float32)
        pause = torch.zeros_like(speech)
        targets = self._base_targets(
            speech_tgt=speech,
            pause_tgt=pause,
            srmdp_notimeline_weight=1.0,
        )
        good_execution = self._execution(
            speech,
            pause,
            dur_logratio_unit=torch.zeros_like(speech),
            role_memory_top_index_unit=torch.full_like(speech, 2.0),
        )
        bad_execution = self._execution(
            speech,
            pause,
            dur_logratio_unit=torch.zeros_like(speech),
            role_memory_top_index_unit=torch.arange(8, dtype=torch.float32).view(1, 8),
        )
        good_losses = build_rhythm_loss_dict(good_execution, targets)
        bad_losses = build_rhythm_loss_dict(bad_execution, targets)
        self.assertLess(
            float(good_losses["rhythm_srmdp_notimeline"].item()),
            float(bad_losses["rhythm_srmdp_notimeline"].item()),
        )
        self.assertLess(
            float(good_losses["rhythm_exec_stretch"].item()),
            float(bad_losses["rhythm_exec_stretch"].item()),
        )

    def test_same_local_role_prefers_similar_retrieval_and_stretch(self) -> None:
        speech = torch.ones((1, 6), dtype=torch.float32)
        pause = torch.zeros_like(speech)
        role_ids = torch.tensor([[0, 1, 0, 1, 0, 1]], dtype=torch.long)
        ref_memory_role_ids = torch.tensor([[0, 1, 0, 1, 0, 1]], dtype=torch.long)
        targets = self._base_targets(
            speech_tgt=speech,
            pause_tgt=pause,
            srmdp_role_consistency_weight=1.0,
            srmdp_memory_role_weight=1.0,
            srmdp_role_id_src_tgt=role_ids,
            srmdp_ref_memory_role_id_tgt=ref_memory_role_ids,
        )
        good_execution = self._execution(
            speech,
            pause,
            dur_logratio_unit=torch.tensor([[0.3, -0.2, 0.31, -0.19, 0.29, -0.18]], dtype=torch.float32),
            role_memory_top_index_unit=torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.float32),
        )
        bad_execution = self._execution(
            speech,
            pause,
            dur_logratio_unit=torch.tensor([[0.3, -0.2, -0.7, 0.8, 1.2, -1.1]], dtype=torch.float32),
            role_memory_top_index_unit=torch.tensor([[1, 0, 3, 2, 5, 4]], dtype=torch.float32),
        )
        good_losses = build_rhythm_loss_dict(good_execution, targets)
        bad_losses = build_rhythm_loss_dict(bad_execution, targets)
        self.assertLess(
            float(good_losses["rhythm_srmdp_role_consistency"].item()),
            float(bad_losses["rhythm_srmdp_role_consistency"].item()),
        )
        self.assertLess(
            float(good_losses["rhythm_srmdp_memory_role"].item()),
            float(bad_losses["rhythm_srmdp_memory_role"].item()),
        )
        self.assertLess(
            float(good_losses["rhythm_exec_stretch"].item()),
            float(bad_losses["rhythm_exec_stretch"].item()),
        )

    def test_target_builder_passes_srmdp_fields(self) -> None:
        unit_batch = SimpleNamespace(
            dur_anchor_src=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
            unit_mask=torch.ones((1, 3), dtype=torch.float32),
        )
        sample = {
            "rhythm_speech_exec_tgt": torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
            "rhythm_pause_exec_tgt": torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
            "rhythm_speech_budget_tgt": torch.tensor([[3.0]], dtype=torch.float32),
            "rhythm_pause_budget_tgt": torch.tensor([[0.0]], dtype=torch.float32),
            "rhythm_target_confidence": torch.tensor([[1.0]], dtype=torch.float32),
            "rhythm_srmdp_role_id_src_tgt": torch.tensor([[0, 1, 0]], dtype=torch.long),
            "rhythm_srmdp_ref_memory_role_id_tgt": torch.tensor([[0, 1, 0, 1]], dtype=torch.long),
            "rhythm_srmdp_ref_memory_mask_tgt": torch.tensor([[1.0, 1.0, 1.0, 0.0]], dtype=torch.float32),
        }
        config = RhythmTargetBuildConfig(
            primary_target_surface="guidance",
            distill_surface="cache",
            lambda_guidance=0.0,
            lambda_distill=0.0,
            distill_exec_weight=0.0,
            distill_budget_weight=0.0,
            distill_allocation_weight=0.0,
            distill_prefix_weight=0.0,
            distill_speech_shape_weight=0.0,
            distill_pause_shape_weight=0.0,
            plan_local_weight=0.0,
            plan_cum_weight=0.0,
            pause_boundary_weight=0.35,
            budget_raw_weight=1.0,
            budget_exec_weight=0.25,
            feasible_debt_weight=0.05,
            srmdp_role_consistency_weight=0.4,
            srmdp_notimeline_weight=0.2,
            srmdp_memory_role_weight=0.3,
        )

        targets = build_rhythm_loss_targets_from_sample(
            sample=sample,
            unit_batch=unit_batch,
            config=config,
            runtime_teacher=None,
            algorithmic_teacher=None,
            offline_confidences=DistillConfidenceBundle(),
            normalize_distill_confidence=lambda confidence, *, batch_size, device: torch.ones(
                (batch_size, 1), device=device
            ),
            normalize_component_confidence=lambda confidence, *, fallback_confidence, batch_size, device: (
                fallback_confidence
            ),
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
        assert targets is not None
        self.assertEqual(float(targets.srmdp_role_consistency_weight), 0.4)
        self.assertEqual(float(targets.srmdp_notimeline_weight), 0.2)
        self.assertEqual(float(targets.srmdp_memory_role_weight), 0.3)
        self.assertTrue(torch.equal(targets.srmdp_role_id_src_tgt, sample["rhythm_srmdp_role_id_src_tgt"]))
        self.assertTrue(
            torch.equal(targets.srmdp_ref_memory_role_id_tgt, sample["rhythm_srmdp_ref_memory_role_id_tgt"])
        )
        self.assertTrue(torch.equal(targets.srmdp_ref_memory_mask_tgt, sample["rhythm_srmdp_ref_memory_mask_tgt"]))


if __name__ == "__main__":
    unittest.main()
