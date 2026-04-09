from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.Conan.rhythm.losses import RhythmLossTargets, build_rhythm_loss_dict
from tasks.Conan.rhythm.loss_routing import update_public_loss_aliases
from tasks.Conan.rhythm.teacher_aux import build_runtime_teacher_aux_loss_dict


class RhythmLossComponentTests(unittest.TestCase):
    @staticmethod
    def _execution(
        speech: torch.Tensor,
        pause: torch.Tensor,
        *,
        source_boundary_cue: torch.Tensor | None = None,
        pause_support_prob: torch.Tensor | None = None,
        pause_allocation_weight: torch.Tensor | None = None,
        pause_support_logit: torch.Tensor | None = None,
        segment_shape_context: torch.Tensor | None = None,
        open_tail_mask: torch.Tensor | None = None,
        segment_roll_alpha: torch.Tensor | None = None,
    ):
        speech_budget = speech.sum(dim=1, keepdim=True)
        pause_budget = pause.sum(dim=1, keepdim=True)
        if source_boundary_cue is None:
            source_boundary_cue = torch.zeros_like(speech)
        if pause_allocation_weight is None:
            pause_allocation_weight = torch.softmax(pause + 1.0e-6, dim=1)
        planner = SimpleNamespace(
            raw_speech_budget_win=speech_budget,
            raw_pause_budget_win=pause_budget,
            speech_budget_win=speech_budget,
            pause_budget_win=pause_budget,
            pause_shape_unit=pause_allocation_weight,
            pause_support_prob_unit=pause_support_prob,
            pause_allocation_weight_unit=pause_allocation_weight,
            pause_support_logit_unit=pause_support_logit,
            source_boundary_cue=source_boundary_cue,
            segment_shape_context_unit=segment_shape_context,
            open_tail_mask_unit=open_tail_mask,
            segment_roll_alpha_unit=segment_roll_alpha,
            feasible_total_budget_delta=torch.zeros_like(speech_budget),
        )
        return SimpleNamespace(
            speech_duration_exec=speech,
            blank_duration_exec=pause,
            pause_after_exec=pause,
            planner=planner,
        )

    def test_shape_distill_uses_shape_confidence_path(self) -> None:
        speech = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        pause = torch.zeros_like(speech)
        execution = self._execution(speech, pause)
        targets = RhythmLossTargets(
            speech_exec_tgt=speech,
            pause_exec_tgt=pause,
            speech_budget_tgt=speech.sum(dim=1, keepdim=True),
            pause_budget_tgt=pause.sum(dim=1, keepdim=True),
            unit_mask=torch.ones_like(speech),
            dur_anchor_src=torch.ones_like(speech),
            plan_local_weight=0.0,
            plan_cum_weight=0.0,
            distill_speech_tgt=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
            distill_pause_tgt=pause,
            distill_exec_confidence=torch.tensor([[1.0], [0.0]], dtype=torch.float32),
            distill_shape_confidence=torch.tensor([[0.0], [1.0]], dtype=torch.float32),
            distill_speech_shape_weight=1.0,
        )
        losses = build_rhythm_loss_dict(execution, targets)
        self.assertTrue(torch.allclose(losses["rhythm_distill_exec"], torch.tensor(0.0)))
        self.assertGreater(float(losses["rhythm_distill_speech_shape"].item()), 0.1)

    def test_distill_exec_weight_zero_keeps_non_exec_distill_active(self) -> None:
        speech = torch.zeros((1, 2), dtype=torch.float32)
        pause = torch.zeros_like(speech)
        planner = SimpleNamespace(
            raw_speech_budget_win=torch.zeros((1, 1), dtype=torch.float32),
            raw_pause_budget_win=torch.zeros((1, 1), dtype=torch.float32),
            speech_budget_win=torch.zeros((1, 1), dtype=torch.float32),
            pause_budget_win=torch.zeros((1, 1), dtype=torch.float32),
            source_boundary_cue=torch.zeros_like(speech),
            feasible_total_budget_delta=torch.zeros((1, 1), dtype=torch.float32),
        )
        execution = SimpleNamespace(
            speech_duration_exec=speech,
            blank_duration_exec=pause,
            pause_after_exec=pause,
            planner=planner,
        )
        targets = RhythmLossTargets(
            speech_exec_tgt=speech,
            pause_exec_tgt=pause,
            speech_budget_tgt=torch.zeros((1, 1), dtype=torch.float32),
            pause_budget_tgt=torch.zeros((1, 1), dtype=torch.float32),
            unit_mask=torch.ones_like(speech),
            dur_anchor_src=torch.ones_like(speech),
            plan_local_weight=0.0,
            plan_cum_weight=0.0,
            distill_speech_tgt=torch.tensor([[2.0, 0.0]], dtype=torch.float32),
            distill_pause_tgt=pause,
            distill_speech_budget_tgt=torch.tensor([[2.0]], dtype=torch.float32),
            distill_pause_budget_tgt=torch.zeros((1, 1), dtype=torch.float32),
            distill_prefix_clock_tgt=torch.tensor([[0.5, 1.0]], dtype=torch.float32),
            distill_prefix_backlog_tgt=torch.tensor([[0.5, 0.0]], dtype=torch.float32),
            distill_confidence=torch.ones((1, 1), dtype=torch.float32),
            distill_exec_weight=0.0,
            distill_budget_weight=1.0,
            distill_prefix_weight=1.0,
            distill_speech_shape_weight=1.0,
        )
        losses = build_rhythm_loss_dict(execution, targets)
        self.assertTrue(torch.allclose(losses["rhythm_distill_exec"], torch.tensor(0.0)))
        self.assertGreater(float(losses["rhythm_distill_budget"].item()), 0.0)
        self.assertGreater(float(losses["rhythm_distill_prefix"].item()), 0.0)
        self.assertGreater(float(losses["rhythm_distill_speech_shape"].item()), 0.0)
        self.assertGreater(float(losses["rhythm_distill"].item()), 0.0)

    def test_plan_terms_are_zero_when_plan_weights_are_disabled(self) -> None:
        speech = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        pause = torch.zeros_like(speech)
        execution = self._execution(speech, pause)
        targets = RhythmLossTargets(
            speech_exec_tgt=speech,
            pause_exec_tgt=pause,
            speech_budget_tgt=speech.sum(dim=1, keepdim=True),
            pause_budget_tgt=pause.sum(dim=1, keepdim=True),
            unit_mask=torch.ones_like(speech),
            dur_anchor_src=torch.ones_like(speech),
            plan_local_weight=0.0,
            plan_cum_weight=0.0,
        )
        losses = build_rhythm_loss_dict(execution, targets)
        self.assertTrue(torch.allclose(losses["rhythm_plan_local"], torch.tensor(0.0)))
        self.assertTrue(torch.allclose(losses["rhythm_plan_cum"], torch.tensor(0.0)))
        self.assertTrue(torch.allclose(losses["rhythm_plan"], torch.tensor(0.0)))

    def test_plan_segment_shape_prefers_shape_aligned_planner(self) -> None:
        speech = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
        pause = torch.zeros_like(speech)
        open_tail = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
        shape_ctx = torch.tensor(
            [[[0.1, 0.0, -1.0], [1.8, 0.0, 1.2], [0.1, 0.0, -1.0]]],
            dtype=torch.float32,
        )
        good_execution = self._execution(
            speech,
            pause,
            segment_shape_context=shape_ctx,
            open_tail_mask=open_tail,
        )
        bad_execution = self._execution(
            speech,
            pause,
            segment_shape_context=shape_ctx,
            open_tail_mask=open_tail,
        )
        good_execution.planner.dur_logratio_unit = torch.tensor([[-0.9, 1.4, -0.9]], dtype=torch.float32)
        bad_execution.planner.dur_logratio_unit = torch.tensor([[1.4, -0.9, -0.9]], dtype=torch.float32)
        targets = RhythmLossTargets(
            speech_exec_tgt=speech,
            pause_exec_tgt=pause,
            speech_budget_tgt=speech.sum(dim=1, keepdim=True),
            pause_budget_tgt=pause.sum(dim=1, keepdim=True),
            unit_mask=torch.ones_like(speech),
            dur_anchor_src=torch.ones_like(speech),
            plan_local_weight=0.0,
            plan_cum_weight=0.0,
            plan_segment_shape_weight=1.0,
        )
        good_losses = build_rhythm_loss_dict(good_execution, targets)
        bad_losses = build_rhythm_loss_dict(bad_execution, targets)
        self.assertLess(
            float(good_losses["rhythm_plan_segment_shape"].item()),
            float(bad_losses["rhythm_plan_segment_shape"].item()),
        )

    def test_plan_pause_release_prefers_release_aligned_planner(self) -> None:
        speech = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
        pause = torch.zeros_like(speech)
        open_tail = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
        shape_ctx = torch.tensor(
            [[[0.1, 0.0, 0.0], [0.1, 2.0, 0.0], [0.1, 0.2, 0.0]]],
            dtype=torch.float32,
        )
        common_kwargs = dict(
            segment_shape_context=shape_ctx,
            open_tail_mask=open_tail,
            segment_roll_alpha=torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32),
            source_boundary_cue=torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32),
            pause_allocation_weight=torch.tensor([[0.1, 0.8, 0.1]], dtype=torch.float32),
        )
        good_execution = self._execution(
            speech,
            pause,
            pause_support_prob=torch.tensor([[0.1, 0.9, 0.1]], dtype=torch.float32),
            **common_kwargs,
        )
        bad_execution = self._execution(
            speech,
            pause,
            pause_support_prob=torch.tensor([[0.9, 0.1, 0.1]], dtype=torch.float32),
            **common_kwargs,
        )
        targets = RhythmLossTargets(
            speech_exec_tgt=speech,
            pause_exec_tgt=pause,
            speech_budget_tgt=speech.sum(dim=1, keepdim=True),
            pause_budget_tgt=pause.sum(dim=1, keepdim=True),
            unit_mask=torch.ones_like(speech),
            dur_anchor_src=torch.ones_like(speech),
            plan_local_weight=0.0,
            plan_cum_weight=0.0,
            plan_pause_release_weight=1.0,
        )
        good_losses = build_rhythm_loss_dict(good_execution, targets)
        bad_losses = build_rhythm_loss_dict(bad_execution, targets)
        self.assertLess(
            float(good_losses["rhythm_plan_pause_release"].item()),
            float(bad_losses["rhythm_plan_pause_release"].item()),
        )

    def test_pause_event_aux_penalizes_missed_pause_support(self) -> None:
        speech = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
        pause_pred = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        pause_tgt = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
        execution = self._execution(speech, pause_pred)
        common_kwargs = dict(
            speech_exec_tgt=speech,
            pause_exec_tgt=pause_tgt,
            speech_budget_tgt=speech.sum(dim=1, keepdim=True),
            pause_budget_tgt=pause_tgt.sum(dim=1, keepdim=True),
            unit_mask=torch.ones_like(speech),
            dur_anchor_src=torch.ones_like(speech),
            plan_local_weight=0.0,
            plan_cum_weight=0.0,
        )
        base_losses = build_rhythm_loss_dict(execution, RhythmLossTargets(**common_kwargs))
        event_losses = build_rhythm_loss_dict(
            execution,
            RhythmLossTargets(
                **common_kwargs,
                pause_event_weight=0.20,
                pause_event_threshold=0.5,
                pause_event_temperature=0.20,
                pause_event_pos_weight=2.5,
            ),
        )
        self.assertGreater(float(event_losses["rhythm_pause_event"].item()), 0.0)
        self.assertGreater(
            float(event_losses["rhythm_exec_pause"].item()),
            float(base_losses["rhythm_exec_pause"].item()),
        )
        self.assertTrue(
            torch.allclose(
                event_losses["rhythm_exec_pause_value"],
                base_losses["rhythm_exec_pause"],
            )
        )
        update_public_loss_aliases(event_losses, mel_loss_names=())
        self.assertTrue(torch.allclose(event_losses["L_exec_pause"], event_losses["rhythm_exec_pause"]))
        self.assertTrue(
            torch.allclose(
                event_losses["L_exec_pause_value"],
                event_losses["rhythm_exec_pause_value"],
            )
        )
        self.assertTrue(torch.allclose(event_losses["L_pause_event"], event_losses["rhythm_pause_event"]))

    def test_pause_event_aux_ignores_boundary_weighted_pause_mask(self) -> None:
        speech = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
        pause_pred = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        pause_tgt = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
        low_boundary_exec = self._execution(
            speech,
            pause_pred,
            source_boundary_cue=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        )
        target_boundary_exec = self._execution(
            speech,
            pause_pred,
            source_boundary_cue=torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32),
        )
        targets = RhythmLossTargets(
            speech_exec_tgt=speech,
            pause_exec_tgt=pause_tgt,
            speech_budget_tgt=speech.sum(dim=1, keepdim=True),
            pause_budget_tgt=pause_tgt.sum(dim=1, keepdim=True),
            unit_mask=torch.ones_like(speech),
            dur_anchor_src=torch.ones_like(speech),
            plan_local_weight=0.0,
            plan_cum_weight=0.0,
            pause_boundary_weight=0.5,
            pause_event_weight=0.20,
            pause_event_threshold=0.5,
            pause_event_temperature=0.20,
            pause_event_pos_weight=2.5,
        )
        low_boundary_losses = build_rhythm_loss_dict(low_boundary_exec, targets)
        target_boundary_losses = build_rhythm_loss_dict(target_boundary_exec, targets)
        self.assertTrue(
            torch.allclose(
                low_boundary_losses["rhythm_pause_event"],
                target_boundary_losses["rhythm_pause_event"],
                atol=1e-6,
            )
        )
        self.assertFalse(
            torch.allclose(
                low_boundary_losses["rhythm_exec_pause_value"],
                target_boundary_losses["rhythm_exec_pause_value"],
                atol=1e-6,
            )
        )

    def test_pause_support_aux_penalizes_misaligned_pause_distribution(self) -> None:
        speech = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
        pause_pred = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
        pause_tgt = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
        execution = self._execution(speech, pause_pred)
        common_kwargs = dict(
            speech_exec_tgt=speech,
            pause_exec_tgt=pause_tgt,
            speech_budget_tgt=speech.sum(dim=1, keepdim=True),
            pause_budget_tgt=pause_tgt.sum(dim=1, keepdim=True),
            unit_mask=torch.ones_like(speech),
            dur_anchor_src=torch.ones_like(speech),
            plan_local_weight=0.0,
            plan_cum_weight=0.0,
        )
        base_losses = build_rhythm_loss_dict(execution, RhythmLossTargets(**common_kwargs))
        support_losses = build_rhythm_loss_dict(
            execution,
            RhythmLossTargets(
                **common_kwargs,
                pause_support_weight=0.10,
            ),
        )
        self.assertGreater(float(support_losses["rhythm_pause_support"].item()), 0.0)
        self.assertGreater(
            float(support_losses["rhythm_exec_pause"].item()),
            float(base_losses["rhythm_exec_pause"].item()),
        )
        update_public_loss_aliases(support_losses, mel_loss_names=())
        self.assertTrue(torch.allclose(support_losses["L_pause_support"], support_losses["rhythm_pause_support"]))

    def test_pause_support_aux_uses_support_head_when_available(self) -> None:
        speech = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
        pause_pred = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        pause_tgt = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
        bad_execution = self._execution(
            speech,
            pause_pred,
            pause_support_prob=torch.tensor([[0.9, 0.1, 0.1]], dtype=torch.float32),
            pause_allocation_weight=torch.tensor([[0.1, 0.8, 0.1]], dtype=torch.float32),
            pause_support_logit=torch.tensor([[2.0, -2.0, -2.0]], dtype=torch.float32),
        )
        good_execution = self._execution(
            speech,
            pause_pred,
            pause_support_prob=torch.tensor([[0.1, 0.9, 0.1]], dtype=torch.float32),
            pause_allocation_weight=torch.tensor([[0.8, 0.1, 0.1]], dtype=torch.float32),
            pause_support_logit=torch.tensor([[-2.0, 2.0, -2.0]], dtype=torch.float32),
        )
        targets = RhythmLossTargets(
            speech_exec_tgt=speech,
            pause_exec_tgt=pause_tgt,
            speech_budget_tgt=speech.sum(dim=1, keepdim=True),
            pause_budget_tgt=pause_tgt.sum(dim=1, keepdim=True),
            unit_mask=torch.ones_like(speech),
            dur_anchor_src=torch.ones_like(speech),
            plan_local_weight=0.0,
            plan_cum_weight=0.0,
            pause_support_weight=0.20,
            pause_event_threshold=0.5,
            pause_event_pos_weight=2.0,
        )
        bad_losses = build_rhythm_loss_dict(bad_execution, targets)
        good_losses = build_rhythm_loss_dict(good_execution, targets)
        self.assertGreater(
            float(bad_losses["rhythm_pause_support"].item()),
            float(good_losses["rhythm_pause_support"].item()),
        )

    def test_pause_allocation_aux_uses_allocation_head_when_available(self) -> None:
        speech = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
        pause_pred = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        pause_tgt = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
        bad_execution = self._execution(
            speech,
            pause_pred,
            pause_support_prob=torch.tensor([[0.1, 0.9, 0.1]], dtype=torch.float32),
            pause_allocation_weight=torch.tensor([[0.8, 0.1, 0.1]], dtype=torch.float32),
            pause_support_logit=torch.tensor([[-2.0, 2.0, -2.0]], dtype=torch.float32),
        )
        good_execution = self._execution(
            speech,
            pause_pred,
            pause_support_prob=torch.tensor([[0.1, 0.9, 0.1]], dtype=torch.float32),
            pause_allocation_weight=torch.tensor([[0.1, 0.8, 0.1]], dtype=torch.float32),
            pause_support_logit=torch.tensor([[-2.0, 2.0, -2.0]], dtype=torch.float32),
        )
        targets = RhythmLossTargets(
            speech_exec_tgt=speech,
            pause_exec_tgt=pause_tgt,
            speech_budget_tgt=speech.sum(dim=1, keepdim=True),
            pause_budget_tgt=pause_tgt.sum(dim=1, keepdim=True),
            unit_mask=torch.ones_like(speech),
            dur_anchor_src=torch.ones_like(speech),
            plan_local_weight=0.0,
            plan_cum_weight=0.0,
            pause_allocation_weight=0.10,
        )
        bad_losses = build_rhythm_loss_dict(bad_execution, targets)
        good_losses = build_rhythm_loss_dict(good_execution, targets)
        self.assertGreater(
            float(bad_losses["rhythm_pause_allocation"].item()),
            float(good_losses["rhythm_pause_allocation"].item()),
        )
        update_public_loss_aliases(bad_losses, mel_loss_names=())
        self.assertTrue(torch.allclose(bad_losses["L_pause_allocation"], bad_losses["rhythm_pause_allocation"]))

    def test_feasible_debt_penalizes_budget_redistribution_repairs(self) -> None:
        speech = torch.tensor([[4.0, 4.0]], dtype=torch.float32)
        pause = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        planner = SimpleNamespace(
            speech_budget_win=torch.tensor([[4.0]], dtype=torch.float32),
            pause_budget_win=torch.tensor([[2.0]], dtype=torch.float32),
            raw_speech_budget_win=torch.tensor([[6.0]], dtype=torch.float32),
            raw_pause_budget_win=torch.tensor([[0.0]], dtype=torch.float32),
            source_boundary_cue=torch.zeros_like(speech),
            feasible_total_budget_delta=torch.tensor([[0.0]], dtype=torch.float32),
        )
        execution = SimpleNamespace(
            speech_duration_exec=speech,
            blank_duration_exec=pause,
            pause_after_exec=pause,
            planner=planner,
        )
        targets = RhythmLossTargets(
            speech_exec_tgt=speech,
            pause_exec_tgt=pause,
            speech_budget_tgt=planner.speech_budget_win,
            pause_budget_tgt=planner.pause_budget_win,
            unit_mask=torch.ones_like(speech),
            dur_anchor_src=speech,
        )
        losses = build_rhythm_loss_dict(execution, targets)
        self.assertGreater(float(losses["rhythm_feasible_debt"].item()), 0.0)

    def test_runtime_teacher_aux_is_separate_from_kd(self) -> None:
        teacher_losses = {
            "rhythm_exec_speech": torch.tensor(2.0, requires_grad=True),
            "rhythm_exec_pause": torch.tensor(1.0, requires_grad=True),
            "rhythm_budget": torch.tensor(4.0, requires_grad=True),
            "rhythm_prefix_state": torch.tensor(3.0, requires_grad=True),
        }
        aux = build_runtime_teacher_aux_loss_dict(
            teacher_losses=teacher_losses,
            hparams={
                "lambda_rhythm_exec_speech": 1.0,
                "lambda_rhythm_exec_pause": 0.5,
                "lambda_rhythm_budget": 0.25,
            },
            prefix_state_lambda=0.2,
            lambda_teacher_aux=0.5,
        )
        self.assertIn("rhythm_teacher_aux_loss", aux)
        self.assertAlmostEqual(float(aux["rhythm_teacher_aux_exec"].item()), 2.5, places=6)
        self.assertAlmostEqual(float(aux["rhythm_teacher_aux_state"].item()), 1.6, places=6)
        self.assertAlmostEqual(float(aux["rhythm_teacher_aux_loss"].item()), 2.05, places=6)
        self.assertAlmostEqual(float(aux["rhythm_teacher_aux"].item()), 2.05, places=6)
        losses = {
            "rhythm_distill": torch.tensor(1.25, requires_grad=True),
            "rhythm_pause_support": torch.tensor(0.1),
            **aux,
        }
        update_public_loss_aliases(losses, mel_loss_names=())
        self.assertAlmostEqual(float(losses["L_kd"].item()), 1.25, places=6)
        self.assertAlmostEqual(float(losses["L_teacher_aux"].item()), 2.05, places=6)
        self.assertAlmostEqual(float(losses["L_pause_support"].item()), 0.1, places=6)


if __name__ == "__main__":
    unittest.main()
