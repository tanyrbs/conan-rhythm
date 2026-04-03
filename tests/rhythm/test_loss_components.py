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
    def _execution(speech: torch.Tensor, pause: torch.Tensor):
        speech_budget = speech.sum(dim=1, keepdim=True)
        pause_budget = pause.sum(dim=1, keepdim=True)
        planner = SimpleNamespace(
            raw_speech_budget_win=speech_budget,
            raw_pause_budget_win=pause_budget,
            speech_budget_win=speech_budget,
            pause_budget_win=pause_budget,
            source_boundary_cue=torch.zeros_like(speech),
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
            **aux,
        }
        update_public_loss_aliases(losses, mel_loss_names=())
        self.assertAlmostEqual(float(losses["L_kd"].item()), 1.25, places=6)
        self.assertAlmostEqual(float(losses["L_teacher_aux"].item()), 2.05, places=6)


if __name__ == "__main__":
    unittest.main()
