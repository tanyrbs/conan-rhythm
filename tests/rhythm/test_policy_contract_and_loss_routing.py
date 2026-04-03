from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.Conan.rhythm.policy import resolve_pause_boundary_weight
from tasks.Conan.rhythm.config_contract_stage_rules import validate_stage_contract
from tasks.Conan.rhythm.loss_routing import route_conan_optimizer_losses, update_public_loss_aliases
from tasks.Conan.rhythm.targets import scale_rhythm_loss_terms


class PolicyContractAndLossRoutingTests(unittest.TestCase):
    def test_pause_boundary_weight_prefers_public_key_over_legacy_alias(self) -> None:
        value = resolve_pause_boundary_weight(
            {
                "rhythm_pause_exec_boundary_boost": 0.75,
                "rhythm_pause_boundary_weight": 0.35,
            }
        )
        self.assertAlmostEqual(value, 0.35)

    def test_plan_loss_keeps_grad_when_lambda_positive_without_aux_flag(self) -> None:
        losses = {"rhythm_plan": torch.tensor(1.0, requires_grad=True)}
        route_conan_optimizer_losses(
            losses,
            mel_loss_names=(),
            hparams={
                "rhythm_compact_joint_loss": False,
                "rhythm_enable_aux_optimizer_losses": False,
                "lambda_rhythm_plan": 0.2,
                "lambda_rhythm_guidance": 0.0,
                "lambda_rhythm_distill": 0.0,
                "lambda_rhythm_teacher_aux": 0.0,
            },
            schedule_only_stage=False,
        )
        self.assertTrue(losses["rhythm_plan"].requires_grad)

    def test_strict_mainline_rejects_plan_proxy_and_conflicting_pause_aliases(self) -> None:
        _, errors, warnings = validate_stage_contract(
            {
                "rhythm_enable_v2": True,
                "rhythm_stage": "student_kd",
                "rhythm_strict_mainline": True,
                "rhythm_cache_version": 5,
                "rhythm_dataset_target_mode": "cached_only",
                "rhythm_primary_target_surface": "teacher",
                "rhythm_teacher_target_source": "learned_offline",
                "rhythm_distill_surface": "cache",
                "rhythm_require_cached_teacher": True,
                "rhythm_enable_dual_mode_teacher": False,
                "rhythm_enable_learned_offline_teacher": False,
                "rhythm_runtime_enable_learned_offline_teacher": False,
                "rhythm_teacher_as_main": False,
                "rhythm_apply_train_override": False,
                "rhythm_apply_valid_override": False,
                "rhythm_require_retimed_cache": False,
                "rhythm_use_retimed_target_if_available": False,
                "rhythm_compact_joint_loss": True,
                "lambda_rhythm_guidance": 0.0,
                "lambda_rhythm_plan": 0.1,
                "lambda_rhythm_distill": 0.35,
                "lambda_rhythm_teacher_aux": 0.0,
                "rhythm_pause_exec_boundary_boost": 0.75,
                "rhythm_pause_boundary_weight": 0.35,
            }
        )
        self.assertIn("rhythm_strict_mainline requires lambda_rhythm_plan: 0.", errors)
        self.assertTrue(any("rhythm_pause_exec_boundary_boost and rhythm_pause_boundary_weight" in e for e in errors))
        self.assertTrue(any("lambda_rhythm_plan > 0 re-enables the optional planner proxy loss" in w for w in warnings))

    def test_scaled_terms_expose_plan_components_and_student_kd(self) -> None:
        scaled = scale_rhythm_loss_terms(
            {
                "rhythm_exec_speech": torch.tensor(1.0),
                "rhythm_exec_pause": torch.tensor(2.0),
                "rhythm_budget": torch.tensor(3.0),
                "rhythm_budget_raw_surface": torch.tensor(3.1),
                "rhythm_budget_exec_surface": torch.tensor(3.2),
                "rhythm_budget_total_surface": torch.tensor(3.3),
                "rhythm_budget_pause_share_surface": torch.tensor(3.4),
                "rhythm_feasible_debt": torch.tensor(0.5),
                "rhythm_prefix_state": torch.tensor(4.0),
                "rhythm_carry": torch.tensor(4.0),
                "rhythm_plan": torch.tensor(5.0),
                "rhythm_plan_local": torch.tensor(6.0),
                "rhythm_plan_cum": torch.tensor(7.0),
                "rhythm_guidance": torch.tensor(8.0),
                "rhythm_distill": torch.tensor(9.0),
                "rhythm_distill_exec": torch.tensor(1.5),
                "rhythm_distill_budget": torch.tensor(1.6),
                "rhythm_distill_budget_raw_surface": torch.tensor(1.7),
                "rhythm_distill_budget_exec_surface": torch.tensor(1.8),
                "rhythm_distill_budget_total_surface": torch.tensor(1.9),
                "rhythm_distill_budget_pause_share_surface": torch.tensor(2.0),
                "rhythm_distill_prefix": torch.tensor(2.1),
                "rhythm_distill_speech_shape": torch.tensor(2.2),
                "rhythm_distill_pause_shape": torch.tensor(2.3),
                "rhythm_distill_allocation": torch.tensor(2.4),
            },
            hparams={
                "lambda_rhythm_exec_speech": 1.0,
                "lambda_rhythm_exec_pause": 1.0,
                "lambda_rhythm_budget": 0.25,
                "lambda_rhythm_plan": 0.2,
                "rhythm_plan_local_weight": 0.5,
                "rhythm_plan_cum_weight": 1.0,
                "lambda_rhythm_guidance": 0.0,
                "lambda_rhythm_distill": 0.4,
                "rhythm_distill_budget_weight": 0.5,
                "rhythm_distill_prefix_weight": 0.25,
                "rhythm_distill_speech_shape_weight": 0.0,
                "rhythm_distill_pause_shape_weight": 0.0,
                "rhythm_distill_allocation_weight": 0.0,
                "rhythm_feasible_debt_weight": 0.05,
            },
            cumplan_lambda=0.15,
        )
        self.assertIn("rhythm_plan_local", scaled)
        self.assertIn("rhythm_plan_cum", scaled)
        self.assertIn("rhythm_distill_student", scaled)
        self.assertTrue(torch.allclose(scaled["rhythm_plan_local"], torch.tensor(0.6)))
        self.assertTrue(torch.allclose(scaled["rhythm_plan_cum"], torch.tensor(1.4)))
        self.assertTrue(torch.allclose(scaled["rhythm_distill_student"], torch.tensor(3.6)))

    def test_public_aliases_expose_plan_and_teacher_aux(self) -> None:
        losses = {
            "rhythm_plan": torch.tensor(1.2),
            "rhythm_plan_local": torch.tensor(0.4),
            "rhythm_plan_cum": torch.tensor(0.8),
            "rhythm_guidance": torch.tensor(0.3),
            "rhythm_distill": torch.tensor(0.7, requires_grad=True),
            "rhythm_distill_student": torch.tensor(0.5),
            "rhythm_teacher_aux": torch.tensor(0.2),
        }
        update_public_loss_aliases(losses, mel_loss_names=())
        self.assertTrue(torch.allclose(losses["L_plan"], torch.tensor(1.2)))
        self.assertTrue(torch.allclose(losses["L_plan_local"], torch.tensor(0.4)))
        self.assertTrue(torch.allclose(losses["L_plan_cum"], torch.tensor(0.8)))
        self.assertTrue(torch.allclose(losses["L_guidance"], torch.tensor(0.3)))
        self.assertTrue(torch.allclose(losses["L_kd_student"], torch.tensor(0.5)))
        self.assertTrue(torch.allclose(losses["L_teacher_aux"], torch.tensor(0.2)))


if __name__ == "__main__":
    unittest.main()
