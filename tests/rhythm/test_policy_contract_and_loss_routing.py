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
from tasks.Conan.rhythm.loss_routing import (
    compute_reporting_total_loss,
    route_conan_optimizer_losses,
    update_public_loss_aliases,
)
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
                "rhythm_binarize_teacher_targets": True,
                "rhythm_enable_dual_mode_teacher": False,
                "rhythm_enable_learned_offline_teacher": False,
                "rhythm_runtime_enable_learned_offline_teacher": False,
                "rhythm_teacher_as_main": False,
                "rhythm_optimize_module_only": True,
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

    def test_pause_recall_aux_warns_when_sparse_capacity_is_still_too_conservative(self) -> None:
        _, errors, warnings = validate_stage_contract(
            {
                "rhythm_enable_v2": True,
                "rhythm_stage": "teacher_offline",
                "rhythm_strict_mainline": False,
                "rhythm_cache_version": 5,
                "rhythm_dataset_target_mode": "cached_only",
                "rhythm_primary_target_surface": "guidance",
                "rhythm_teacher_target_source": "algorithmic",
                "rhythm_distill_surface": "none",
                "rhythm_require_cached_teacher": False,
                "rhythm_binarize_teacher_targets": False,
                "rhythm_enable_dual_mode_teacher": False,
                "rhythm_enable_learned_offline_teacher": True,
                "rhythm_runtime_enable_learned_offline_teacher": True,
                "rhythm_teacher_as_main": True,
                "rhythm_optimize_module_only": True,
                "rhythm_apply_train_override": False,
                "rhythm_apply_valid_override": False,
                "rhythm_require_retimed_cache": False,
                "rhythm_use_retimed_target_if_available": False,
                "rhythm_compact_joint_loss": False,
                "lambda_rhythm_guidance": 0.0,
                "lambda_rhythm_plan": 0.0,
                "lambda_rhythm_distill": 0.0,
                "lambda_rhythm_teacher_aux": 0.0,
                "rhythm_pause_event_weight": 0.20,
                "rhythm_pause_support_weight": 0.05,
                "rhythm_projector_pause_selection_mode": "simple",
                "rhythm_projector_pause_topk_ratio_train_end": 0.35,
                "rhythm_projector_pause_boundary_bias_weight": 0.15,
                "rhythm_pause_boundary_weight": 0.35,
                "rhythm_projector_pause_soft_temperature": 0.12,
                "rhythm_teacher_projector_force_full_commit": True,
                "rhythm_teacher_projector_soft_pause_selection": False,
            }
        )
        self.assertEqual(errors, [])
        self.assertTrue(any("pause-recall auxiliaries are enabled but rhythm_projector_pause_selection_mode is not 'sparse'" in w for w in warnings))
        self.assertTrue(any("pause-recall auxiliaries are enabled while rhythm_projector_pause_topk_ratio_train_end < 0.40" in w for w in warnings))
        self.assertTrue(any("rhythm_projector_pause_soft_temperature <= 0.12" in w for w in warnings))
        self.assertTrue(any("sparse support losers may receive little gradient" in w for w in warnings))

    def test_pause_recall_aux_warns_when_boundary_weighting_is_still_aggressive(self) -> None:
        _, errors, warnings = validate_stage_contract(
            {
                "rhythm_enable_v2": True,
                "rhythm_stage": "teacher_offline",
                "rhythm_strict_mainline": False,
                "rhythm_cache_version": 5,
                "rhythm_dataset_target_mode": "cached_only",
                "rhythm_primary_target_surface": "guidance",
                "rhythm_teacher_target_source": "algorithmic",
                "rhythm_distill_surface": "none",
                "rhythm_require_cached_teacher": False,
                "rhythm_binarize_teacher_targets": False,
                "rhythm_enable_dual_mode_teacher": False,
                "rhythm_enable_learned_offline_teacher": True,
                "rhythm_runtime_enable_learned_offline_teacher": True,
                "rhythm_teacher_as_main": True,
                "rhythm_optimize_module_only": True,
                "rhythm_apply_train_override": False,
                "rhythm_apply_valid_override": False,
                "rhythm_require_retimed_cache": False,
                "rhythm_use_retimed_target_if_available": False,
                "rhythm_compact_joint_loss": False,
                "lambda_rhythm_guidance": 0.0,
                "lambda_rhythm_plan": 0.0,
                "lambda_rhythm_distill": 0.0,
                "lambda_rhythm_teacher_aux": 0.0,
                "rhythm_pause_event_weight": 0.20,
                "rhythm_projector_pause_selection_mode": "sparse",
                "rhythm_projector_pause_topk_ratio_train_end": 0.45,
                "rhythm_projector_pause_soft_temperature": 0.18,
                "rhythm_teacher_projector_force_full_commit": True,
                "rhythm_teacher_projector_soft_pause_selection": True,
                "rhythm_projector_pause_boundary_bias_weight": 0.22,
                "rhythm_pause_boundary_weight": 0.50,
            }
        )
        self.assertEqual(errors, [])
        self.assertTrue(any("pre-vs-post-projector recall before increasing boundary weighting further" in w for w in warnings))

    def test_strict_mainline_rejects_cached_teacher_distill_without_dedupe(self) -> None:
        _, errors, _ = validate_stage_contract(
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
                "rhythm_binarize_teacher_targets": True,
                "rhythm_enable_dual_mode_teacher": False,
                "rhythm_enable_learned_offline_teacher": False,
                "rhythm_runtime_enable_learned_offline_teacher": False,
                "rhythm_teacher_as_main": False,
                "rhythm_optimize_module_only": True,
                "rhythm_apply_train_override": False,
                "rhythm_apply_valid_override": False,
                "rhythm_require_retimed_cache": False,
                "rhythm_use_retimed_target_if_available": False,
                "rhythm_compact_joint_loss": False,
                "rhythm_dedupe_teacher_primary_cache_distill": False,
                "lambda_rhythm_guidance": 0.0,
                "lambda_rhythm_plan": 0.0,
                "lambda_rhythm_distill": 0.35,
                "lambda_rhythm_teacher_aux": 0.0,
            }
        )
        self.assertTrue(any("rhythm_dedupe_teacher_primary_cache_distill" in e for e in errors))

    def test_strict_mainline_rejects_allocation_plus_shape_distill(self) -> None:
        _, errors, warnings = validate_stage_contract(
            {
                "rhythm_enable_v2": True,
                "rhythm_stage": "transitional",
                "rhythm_strict_mainline": True,
                "rhythm_cache_version": 5,
                "rhythm_dataset_target_mode": "cached_only",
                "rhythm_primary_target_surface": "teacher",
                "rhythm_teacher_target_source": "learned_offline",
                "rhythm_distill_surface": "cache",
                "rhythm_require_cached_teacher": True,
                "rhythm_binarize_teacher_targets": True,
                "rhythm_enable_dual_mode_teacher": False,
                "rhythm_enable_learned_offline_teacher": False,
                "rhythm_runtime_enable_learned_offline_teacher": False,
                "rhythm_teacher_as_main": False,
                "lambda_rhythm_guidance": 0.0,
                "lambda_rhythm_plan": 0.0,
                "lambda_rhythm_distill": 0.35,
                "lambda_rhythm_teacher_aux": 0.0,
                "rhythm_distill_allocation_weight": 0.2,
                "rhythm_distill_speech_shape_weight": 0.2,
                "rhythm_distill_pause_shape_weight": 0.0,
            }
        )
        self.assertTrue(
            any("double-constrains the same mass split" in e for e in errors),
            msg=f"expected strict-mainline error, got errors={errors!r}, warnings={warnings!r}",
        )

    def test_strict_mainline_accepts_new_duplicate_distill_alias(self) -> None:
        _, errors, _ = validate_stage_contract(
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
                "rhythm_binarize_teacher_targets": True,
                "rhythm_enable_dual_mode_teacher": False,
                "rhythm_enable_learned_offline_teacher": False,
                "rhythm_runtime_enable_learned_offline_teacher": False,
                "rhythm_teacher_as_main": False,
                "rhythm_optimize_module_only": True,
                "rhythm_apply_train_override": False,
                "rhythm_apply_valid_override": False,
                "rhythm_require_retimed_cache": False,
                "rhythm_use_retimed_target_if_available": False,
                "rhythm_compact_joint_loss": False,
                "rhythm_suppress_duplicate_primary_distill": True,
                "lambda_rhythm_guidance": 0.0,
                "lambda_rhythm_plan": 0.0,
                "lambda_rhythm_distill": 0.35,
                "lambda_rhythm_teacher_aux": 0.0,
                "rhythm_distill_exec_weight": 0.0,
                "rhythm_distill_budget_weight": 0.0,
                "rhythm_distill_prefix_weight": 0.0,
                "rhythm_distill_allocation_weight": 0.0,
                "rhythm_distill_speech_shape_weight": 0.25,
                "rhythm_distill_pause_shape_weight": 0.0,
            }
        )
        self.assertEqual(errors, [])

    def test_strict_mainline_requires_shape_distill_when_cache_teacher_distill_is_deduped(self) -> None:
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
                "rhythm_compact_joint_loss": False,
                "rhythm_dedupe_teacher_primary_cache_distill": True,
                "lambda_rhythm_guidance": 0.0,
                "lambda_rhythm_plan": 0.0,
                "lambda_rhythm_distill": 0.35,
                "lambda_rhythm_teacher_aux": 0.0,
                "rhythm_distill_exec_weight": 0.0,
                "rhythm_distill_budget_weight": 0.08,
                "rhythm_distill_prefix_weight": 0.50,
                "rhythm_distill_allocation_weight": 0.0,
                "rhythm_distill_speech_shape_weight": 0.0,
                "rhythm_distill_pause_shape_weight": 0.0,
            }
        )
        self.assertTrue(any("must keep rhythm_distill_speech_shape_weight > 0" in e for e in errors))

    def test_stage_contract_allows_zero_distill_confidence_floor(self) -> None:
        _, errors, _ = validate_stage_contract(
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
                "rhythm_binarize_teacher_targets": True,
                "rhythm_enable_dual_mode_teacher": False,
                "rhythm_enable_learned_offline_teacher": False,
                "rhythm_runtime_enable_learned_offline_teacher": False,
                "rhythm_teacher_as_main": False,
                "rhythm_optimize_module_only": True,
                "rhythm_apply_train_override": False,
                "rhythm_apply_valid_override": False,
                "rhythm_require_retimed_cache": False,
                "rhythm_use_retimed_target_if_available": False,
                "rhythm_compact_joint_loss": False,
                "rhythm_suppress_duplicate_primary_distill": True,
                "rhythm_distill_confidence_floor": 0.0,
                "lambda_rhythm_guidance": 0.0,
                "lambda_rhythm_plan": 0.0,
                "lambda_rhythm_distill": 0.35,
                "lambda_rhythm_teacher_aux": 0.0,
                "rhythm_distill_exec_weight": 0.0,
                "rhythm_distill_budget_weight": 0.0,
                "rhythm_distill_prefix_weight": 0.0,
                "rhythm_distill_allocation_weight": 0.0,
                "rhythm_distill_speech_shape_weight": 0.25,
                "rhythm_distill_pause_shape_weight": 0.0,
            }
        )
        self.assertEqual(errors, [])

    def test_external_reference_policy_errors_when_cached_targets_stay_self_conditioned(self) -> None:
        _, errors, warnings = validate_stage_contract(
            {
                "rhythm_enable_v2": True,
                "rhythm_stage": "transitional",
                "rhythm_strict_mainline": False,
                "rhythm_cache_version": 5,
                "rhythm_dataset_target_mode": "cached_only",
                "rhythm_cached_reference_policy": "sample_ref",
                "rhythm_primary_target_surface": "teacher",
                "rhythm_teacher_target_source": "learned_offline",
                "rhythm_distill_surface": "none",
                "rhythm_require_cached_teacher": True,
                "rhythm_binarize_teacher_targets": True,
                "rhythm_enable_dual_mode_teacher": False,
                "rhythm_enable_learned_offline_teacher": False,
                "rhythm_runtime_enable_learned_offline_teacher": False,
                "rhythm_teacher_as_main": False,
                "lambda_rhythm_guidance": 0.0,
                "lambda_rhythm_plan": 0.0,
                "lambda_rhythm_distill": 0.0,
                "lambda_rhythm_teacher_aux": 0.0,
            }
        )
        self.assertTrue(any("requires rhythm_dataset_target_mode: runtime_only" in error for error in errors))

    def test_runtime_only_external_teacher_requires_algorithmic_source(self) -> None:
        _, errors, _ = validate_stage_contract(
            {
                "rhythm_enable_v2": True,
                "rhythm_stage": "transitional",
                "rhythm_strict_mainline": False,
                "rhythm_cache_version": 5,
                "rhythm_dataset_target_mode": "runtime_only",
                "rhythm_cached_reference_policy": "sample_ref",
                "rhythm_primary_target_surface": "teacher",
                "rhythm_teacher_target_source": "learned_offline",
                "rhythm_distill_surface": "none",
                "rhythm_require_cached_teacher": False,
                "rhythm_binarize_teacher_targets": False,
                "rhythm_enable_dual_mode_teacher": False,
                "rhythm_enable_learned_offline_teacher": False,
                "rhythm_runtime_enable_learned_offline_teacher": False,
                "rhythm_teacher_as_main": False,
                "lambda_rhythm_guidance": 0.0,
                "lambda_rhythm_plan": 0.0,
                "lambda_rhythm_distill": 0.0,
                "lambda_rhythm_teacher_aux": 0.0,
            }
        )
        self.assertTrue(any("requires rhythm_teacher_target_source: algorithmic" in error for error in errors))

    def test_runtime_only_external_reference_warns_about_stale_cached_surface_binarization_flags(self) -> None:
        _, errors, warnings = validate_stage_contract(
            {
                "rhythm_enable_v2": True,
                "rhythm_stage": "transitional",
                "rhythm_strict_mainline": False,
                "rhythm_cache_version": 5,
                "rhythm_dataset_target_mode": "runtime_only",
                "rhythm_cached_reference_policy": "sample_ref",
                "rhythm_primary_target_surface": "teacher",
                "rhythm_teacher_target_source": "algorithmic",
                "rhythm_distill_surface": "none",
                "rhythm_require_cached_teacher": False,
                "rhythm_binarize_teacher_targets": True,
                "rhythm_binarize_retimed_mel_targets": True,
                "rhythm_enable_dual_mode_teacher": False,
                "rhythm_enable_learned_offline_teacher": False,
                "rhythm_runtime_enable_learned_offline_teacher": False,
                "rhythm_teacher_as_main": False,
                "lambda_rhythm_guidance": 0.0,
                "lambda_rhythm_plan": 0.0,
                "lambda_rhythm_distill": 0.0,
                "lambda_rhythm_teacher_aux": 0.0,
            }
        )
        self.assertEqual(errors, [])
        self.assertTrue(any("rhythm_binarize_teacher_targets" in warning for warning in warnings))
        self.assertTrue(any("rhythm_binarize_retimed_mel_targets" in warning for warning in warnings))

    def test_lambda_distill_requires_active_component_weight(self) -> None:
        _, errors, _ = validate_stage_contract(
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
                "rhythm_binarize_teacher_targets": True,
                "rhythm_enable_dual_mode_teacher": False,
                "rhythm_enable_learned_offline_teacher": False,
                "rhythm_runtime_enable_learned_offline_teacher": False,
                "rhythm_teacher_as_main": False,
                "rhythm_optimize_module_only": True,
                "rhythm_apply_train_override": False,
                "rhythm_apply_valid_override": False,
                "rhythm_require_retimed_cache": False,
                "rhythm_use_retimed_target_if_available": False,
                "rhythm_compact_joint_loss": False,
                "rhythm_dedupe_teacher_primary_cache_distill": True,
                "lambda_rhythm_guidance": 0.0,
                "lambda_rhythm_plan": 0.0,
                "lambda_rhythm_distill": 0.35,
                "lambda_rhythm_teacher_aux": 0.0,
                "rhythm_distill_exec_weight": 0.0,
                "rhythm_distill_budget_weight": 0.0,
                "rhythm_distill_prefix_weight": 0.0,
                "rhythm_distill_allocation_weight": 0.0,
                "rhythm_distill_speech_shape_weight": 0.0,
                "rhythm_distill_pause_shape_weight": 0.0,
            }
        )
        self.assertTrue(any("requires at least one active distillation component weight" in e for e in errors))

    def test_strict_mainline_rejects_duplicate_teacher_exec_distill(self) -> None:
        _, errors, _ = validate_stage_contract(
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
                "rhythm_binarize_teacher_targets": True,
                "rhythm_enable_dual_mode_teacher": False,
                "rhythm_enable_learned_offline_teacher": False,
                "rhythm_runtime_enable_learned_offline_teacher": False,
                "rhythm_teacher_as_main": False,
                "rhythm_optimize_module_only": True,
                "rhythm_apply_train_override": False,
                "rhythm_apply_valid_override": False,
                "rhythm_require_retimed_cache": False,
                "rhythm_use_retimed_target_if_available": False,
                "rhythm_compact_joint_loss": False,
                "rhythm_dedupe_teacher_primary_cache_distill": True,
                "lambda_rhythm_guidance": 0.0,
                "lambda_rhythm_plan": 0.0,
                "lambda_rhythm_distill": 0.35,
                "lambda_rhythm_teacher_aux": 0.0,
                "rhythm_distill_exec_weight": 0.2,
                "rhythm_distill_budget_weight": 0.0,
                "rhythm_distill_prefix_weight": 0.0,
                "rhythm_distill_allocation_weight": 0.0,
                "rhythm_distill_speech_shape_weight": 0.25,
                "rhythm_distill_pause_shape_weight": 0.0,
            }
        )
        self.assertTrue(any("rhythm_distill_exec_weight: 0" in e for e in errors))

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
                "rhythm_distill_same_source_exec": torch.tensor(1.0),
                "rhythm_distill_same_source_budget": torch.tensor(1.0),
                "rhythm_distill_same_source_prefix": torch.tensor(1.0),
                "rhythm_distill_same_source_any": torch.tensor(1.0),
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
        self.assertIn("rhythm_distill_same_source_any", scaled)
        self.assertTrue(torch.allclose(scaled["rhythm_plan_local"], torch.tensor(0.6)))
        self.assertTrue(torch.allclose(scaled["rhythm_plan_cum"], torch.tensor(1.4)))
        self.assertTrue(torch.allclose(scaled["rhythm_distill_student"], torch.tensor(3.6)))
        self.assertTrue(torch.allclose(scaled["rhythm_distill_same_source_any"], torch.tensor(1.0)))

    def test_public_aliases_expose_plan_and_teacher_aux(self) -> None:
        losses = {
            "rhythm_plan": torch.tensor(1.2),
            "rhythm_plan_local": torch.tensor(0.4),
            "rhythm_plan_cum": torch.tensor(0.8),
            "rhythm_guidance": torch.tensor(0.3),
            "rhythm_distill": torch.tensor(0.7, requires_grad=True),
            "rhythm_distill_student": torch.tensor(0.5),
            "rhythm_teacher_aux": torch.tensor(0.2),
            "rhythm_distill_same_source_any": torch.tensor(1.0),
            "rhythm_distill_same_source_exec": torch.tensor(1.0),
            "rhythm_distill_same_source_budget": torch.tensor(1.0),
            "rhythm_distill_same_source_prefix": torch.tensor(1.0),
        }
        update_public_loss_aliases(losses, mel_loss_names=())
        self.assertTrue(torch.allclose(losses["L_plan"], torch.tensor(1.2)))
        self.assertTrue(torch.allclose(losses["L_plan_local"], torch.tensor(0.4)))
        self.assertTrue(torch.allclose(losses["L_plan_cum"], torch.tensor(0.8)))
        self.assertTrue(torch.allclose(losses["L_guidance"], torch.tensor(0.3)))
        self.assertTrue(torch.allclose(losses["L_kd_student"], torch.tensor(0.5)))
        self.assertTrue(torch.allclose(losses["L_teacher_aux"], torch.tensor(0.2)))
        self.assertTrue(torch.allclose(losses["L_kd_same_source"], torch.tensor(1.0)))

    def test_reporting_total_loss_ignores_public_aliases_when_grad_is_enabled(self) -> None:
        losses = {
            "rhythm_exec_speech": torch.tensor(1.0, requires_grad=True),
            "rhythm_exec_pause": torch.tensor(2.0, requires_grad=True),
            "rhythm_budget": torch.tensor(3.0, requires_grad=True),
        }
        update_public_loss_aliases(losses, mel_loss_names=())
        total = compute_reporting_total_loss(losses)
        self.assertTrue(torch.allclose(total, torch.tensor(6.0)))

    def test_public_aliases_fall_back_to_non_compact_components(self) -> None:
        losses = {
            "rhythm_exec_speech": torch.tensor(1.0),
            "rhythm_exec_pause": torch.tensor(2.0),
            "rhythm_budget": torch.tensor(3.0),
            "rhythm_prefix_state": torch.tensor(4.0),
        }
        update_public_loss_aliases(losses, mel_loss_names=())
        self.assertTrue(torch.allclose(losses["L_rhythm_exec"], torch.tensor(3.0)))
        self.assertTrue(torch.allclose(losses["L_stream_state"], torch.tensor(7.0)))

    def test_reporting_total_loss_reconstructs_compact_objective_under_no_grad(self) -> None:
        losses = {
            "rhythm_exec_speech": torch.tensor(1.0),
            "rhythm_exec_pause": torch.tensor(2.0),
            "rhythm_exec": torch.tensor(3.0),
            "rhythm_budget": torch.tensor(4.0),
            "rhythm_prefix_state": torch.tensor(5.0),
            "rhythm_stream_state": torch.tensor(6.0),
            "rhythm_distill": torch.tensor(7.0),
            "rhythm_teacher_aux_loss": torch.tensor(8.0),
        }
        update_public_loss_aliases(losses, mel_loss_names=())
        total = compute_reporting_total_loss(
            losses,
            mel_loss_names=(),
            hparams={
                "rhythm_compact_joint_loss": True,
                "rhythm_enable_aux_optimizer_losses": False,
                "lambda_rhythm_plan": 0.0,
                "lambda_rhythm_guidance": 0.0,
                "lambda_rhythm_distill": 0.35,
                "lambda_rhythm_teacher_aux": 0.2,
            },
            schedule_only_stage=False,
        )
        self.assertTrue(torch.allclose(total, torch.tensor(24.0)))

    def test_student_kd_dedupe_warning_mentions_effective_remaining_shape_branch(self) -> None:
        _, _, warnings = validate_stage_contract(
            {
                "rhythm_enable_v2": True,
                "rhythm_stage": "student_kd",
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
                "rhythm_dedupe_teacher_primary_cache_distill": True,
                "lambda_rhythm_guidance": 0.0,
                "lambda_rhythm_plan": 0.0,
                "lambda_rhythm_distill": 0.35,
                "lambda_rhythm_teacher_aux": 0.0,
                "rhythm_distill_exec_weight": 0.0,
                "rhythm_distill_budget_weight": 0.1,
                "rhythm_distill_prefix_weight": 0.5,
                "rhythm_distill_speech_shape_weight": 0.25,
                "rhythm_distill_pause_shape_weight": 0.0,
                "rhythm_distill_allocation_weight": 0.0,
            }
        )
        self.assertTrue(any("remaining shape distill" in w for w in warnings))

    def test_student_kd_shape_only_dedupe_path_is_warning_free(self) -> None:
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
                "rhythm_binarize_teacher_targets": True,
                "rhythm_enable_dual_mode_teacher": False,
                "rhythm_enable_learned_offline_teacher": False,
                "rhythm_runtime_enable_learned_offline_teacher": False,
                "rhythm_teacher_as_main": False,
                "rhythm_optimize_module_only": True,
                "rhythm_apply_train_override": False,
                "rhythm_apply_valid_override": False,
                "rhythm_require_retimed_cache": False,
                "rhythm_use_retimed_target_if_available": False,
                "rhythm_compact_joint_loss": False,
                "rhythm_dedupe_teacher_primary_cache_distill": True,
                "lambda_rhythm_guidance": 0.0,
                "lambda_rhythm_plan": 0.0,
                "lambda_rhythm_distill": 0.35,
                "lambda_rhythm_teacher_aux": 0.0,
                "rhythm_distill_exec_weight": 0.0,
                "rhythm_distill_budget_weight": 0.0,
                "rhythm_distill_prefix_weight": 0.0,
                "rhythm_distill_allocation_weight": 0.0,
                "rhythm_distill_speech_shape_weight": 0.25,
                "rhythm_distill_pause_shape_weight": 0.25,
            }
        )
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])

    def test_inactive_kd_weights_emit_cleanup_warning(self) -> None:
        _, errors, warnings = validate_stage_contract(
            {
                "rhythm_enable_v2": True,
                "rhythm_stage": "student_retimed",
                "rhythm_cache_version": 5,
                "rhythm_dataset_target_mode": "cached_only",
                "rhythm_primary_target_surface": "teacher",
                "rhythm_teacher_target_source": "learned_offline",
                "rhythm_distill_surface": "none",
                "rhythm_require_cached_teacher": True,
                "rhythm_binarize_teacher_targets": True,
                "rhythm_enable_dual_mode_teacher": False,
                "rhythm_enable_learned_offline_teacher": False,
                "rhythm_runtime_enable_learned_offline_teacher": False,
                "rhythm_teacher_as_main": False,
                "rhythm_optimize_module_only": False,
                "rhythm_apply_train_override": True,
                "rhythm_apply_valid_override": True,
                "rhythm_require_retimed_cache": True,
                "rhythm_use_retimed_target_if_available": True,
                "rhythm_use_retimed_pitch_target": True,
                "rhythm_compact_joint_loss": True,
                "lambda_rhythm_guidance": 0.0,
                "lambda_rhythm_plan": 0.0,
                "lambda_rhythm_distill": 0.0,
                "lambda_rhythm_teacher_aux": 0.0,
                "rhythm_distill_prefix_weight": 0.25,
                "rhythm_distill_speech_shape_weight": 0.10,
            }
        )
        self.assertEqual(errors, [])
        self.assertTrue(any("inactive KD config clutter" in w for w in warnings))

    def test_route_compaction_materializes_weighted_stream_macro_under_no_grad(self) -> None:
        losses = {
            "rhythm_exec_speech": torch.tensor(1.0),
            "rhythm_exec_pause": torch.tensor(2.0),
            "rhythm_budget": torch.tensor(4.0),
            "rhythm_prefix_state": torch.tensor(5.0),
        }
        hparams = {
            "rhythm_compact_joint_loss": True,
            "rhythm_joint_budget_macro_weight": 0.35,
            "rhythm_joint_cumplan_macro_weight": 0.65,
            "rhythm_enable_aux_optimizer_losses": False,
            "lambda_rhythm_plan": 0.0,
            "lambda_rhythm_guidance": 0.0,
            "lambda_rhythm_distill": 0.0,
            "lambda_rhythm_teacher_aux": 0.0,
        }
        route_conan_optimizer_losses(
            losses,
            mel_loss_names=(),
            hparams=hparams,
            schedule_only_stage=False,
        )
        update_public_loss_aliases(losses, mel_loss_names=())
        self.assertTrue(torch.allclose(losses["rhythm_exec"], torch.tensor(3.0)))
        self.assertTrue(torch.allclose(losses["rhythm_stream_state"], torch.tensor(4.65)))
        self.assertTrue(torch.allclose(losses["L_stream_state"], torch.tensor(4.65)))
        total = compute_reporting_total_loss(
            losses,
            mel_loss_names=(),
            hparams=hparams,
            schedule_only_stage=False,
        )
        self.assertTrue(torch.allclose(total, torch.tensor(7.65)))


if __name__ == "__main__":
    unittest.main()
