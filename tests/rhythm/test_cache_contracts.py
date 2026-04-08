from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.Conan.rhythm.config_contract import build_contract_context
from tasks.Conan.rhythm.config_contract_cache_rules import validate_cache_field_contract
from tasks.Conan.rhythm.dataset_contracts import RhythmDatasetCacheContract
from tasks.Conan.rhythm.dataset_mixin import RhythmConanDatasetMixin


class _DummyDataset(RhythmConanDatasetMixin):
    def __init__(self, hparams: dict):
        self.hparams = hparams
        self.prefix = "train"


class RhythmCacheContractTests(unittest.TestCase):
    @staticmethod
    def _student_kd_shape_only_hparams() -> dict:
        return {
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
            "lambda_rhythm_guidance": 0.0,
            "lambda_rhythm_plan": 0.0,
            "lambda_rhythm_distill": 0.35,
            "lambda_rhythm_teacher_aux": 0.0,
            "rhythm_distill_exec_weight": 0.0,
            "rhythm_distill_budget_weight": 0.0,
            "rhythm_distill_allocation_weight": 0.0,
            "rhythm_distill_prefix_weight": 0.0,
            "rhythm_distill_speech_shape_weight": 0.25,
            "rhythm_distill_pause_shape_weight": 0.25,
        }

    @staticmethod
    def _legacy_kd_hparams() -> dict:
        return {
            "rhythm_enable_v2": True,
            "rhythm_stage": "legacy_dual_mode_kd",
            "rhythm_strict_mainline": False,
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
            "rhythm_distill_exec_weight": 1.0,
            "rhythm_distill_budget_weight": 0.08,
            "rhythm_distill_allocation_weight": 0.1,
            "rhythm_distill_prefix_weight": 0.5,
            "rhythm_distill_speech_shape_weight": 0.25,
            "rhythm_distill_pause_shape_weight": 0.25,
        }

    def test_shape_only_student_kd_uses_slim_cache_field_contract(self) -> None:
        hparams = self._student_kd_shape_only_hparams()
        context = build_contract_context(hparams, model_dry_run=True)
        report = validate_cache_field_contract(context)
        groups = set(report.required_field_groups)

        self.assertEqual(report.errors, ())
        self.assertIn(("rhythm_teacher_speech_exec_tgt",), groups)
        self.assertIn(("rhythm_teacher_speech_budget_tgt",), groups)
        self.assertIn(("rhythm_teacher_confidence",), groups)
        self.assertIn(("rhythm_teacher_confidence_shape",), groups)
        self.assertNotIn(("rhythm_teacher_allocation_tgt",), groups)
        self.assertNotIn(("rhythm_teacher_prefix_clock_tgt",), groups)
        self.assertNotIn(("rhythm_teacher_prefix_backlog_tgt",), groups)
        self.assertNotIn(("rhythm_teacher_confidence_exec",), groups)
        self.assertNotIn(("rhythm_teacher_confidence_prefix",), groups)
        self.assertNotIn(("rhythm_teacher_confidence_allocation",), groups)

    def test_shape_only_student_kd_dataset_cache_keys_are_slim(self) -> None:
        owner = SimpleNamespace(
            hparams=self._student_kd_shape_only_hparams(),
            _resolve_primary_target_surface=lambda: "teacher",
            _resolve_distill_surface=lambda: "cache",
        )
        keys = set(RhythmDatasetCacheContract(owner).required_cached_target_keys())

        self.assertIn("rhythm_teacher_speech_exec_tgt", keys)
        self.assertIn("rhythm_teacher_speech_budget_tgt", keys)
        self.assertIn("rhythm_teacher_confidence", keys)
        self.assertIn("rhythm_teacher_confidence_shape", keys)
        self.assertNotIn("rhythm_teacher_allocation_tgt", keys)
        self.assertNotIn("rhythm_teacher_prefix_clock_tgt", keys)
        self.assertNotIn("rhythm_teacher_prefix_backlog_tgt", keys)
        self.assertNotIn("rhythm_teacher_confidence_exec", keys)
        self.assertNotIn("rhythm_teacher_confidence_prefix", keys)
        self.assertNotIn("rhythm_teacher_confidence_allocation", keys)

    def test_shape_only_student_kd_runtime_export_skips_exec_confidence(self) -> None:
        keys = set(_DummyDataset(self._student_kd_shape_only_hparams())._resolve_runtime_target_export_keys())

        self.assertIn("rhythm_teacher_speech_exec_tgt", keys)
        self.assertIn("rhythm_teacher_speech_budget_tgt", keys)
        self.assertIn("rhythm_teacher_confidence_shape", keys)
        self.assertNotIn("rhythm_teacher_allocation_tgt", keys)
        self.assertNotIn("rhythm_teacher_prefix_clock_tgt", keys)
        self.assertNotIn("rhythm_teacher_prefix_backlog_tgt", keys)
        self.assertNotIn("rhythm_teacher_confidence_exec", keys)

    def test_legacy_kd_keeps_optional_teacher_sidecars(self) -> None:
        hparams = self._legacy_kd_hparams()
        context = build_contract_context(hparams, model_dry_run=True)
        groups = set(validate_cache_field_contract(context).required_field_groups)

        self.assertIn(("rhythm_teacher_allocation_tgt",), groups)
        self.assertIn(("rhythm_teacher_prefix_clock_tgt",), groups)
        self.assertIn(("rhythm_teacher_prefix_backlog_tgt",), groups)
        self.assertIn(("rhythm_teacher_confidence_exec",), groups)
        self.assertIn(("rhythm_teacher_confidence_budget",), groups)
        self.assertIn(("rhythm_teacher_confidence_prefix",), groups)
        self.assertIn(("rhythm_teacher_confidence_allocation",), groups)
        self.assertIn(("rhythm_teacher_confidence_shape",), groups)

    def test_runtime_only_teacher_stage_does_not_require_cached_teacher_fields(self) -> None:
        hparams = {
            "rhythm_enable_v2": True,
            "rhythm_stage": "transitional",
            "rhythm_strict_mainline": False,
            "rhythm_cache_version": 5,
            "rhythm_dataset_target_mode": "runtime_only",
            "rhythm_primary_target_surface": "teacher",
            "rhythm_teacher_target_source": "algorithmic",
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
        context = build_contract_context(hparams, model_dry_run=True)
        report = validate_cache_field_contract(context)
        groups = set(report.required_field_groups)

        self.assertNotIn(("rhythm_teacher_speech_exec_tgt",), groups)
        self.assertNotIn(("rhythm_teacher_speech_budget_tgt",), groups)
        self.assertNotIn(("rhythm_teacher_confidence",), groups)
        self.assertFalse(
            any("Primary surface is teacher but rhythm_binarize_teacher_targets is false." in warning for warning in report.warnings)
        )

    def test_export_debug_sidecars_require_phrase_bank_sidecar_groups(self) -> None:
        hparams = {
            "rhythm_enable_v2": True,
            "rhythm_stage": "transitional",
            "rhythm_strict_mainline": False,
            "rhythm_cache_version": 5,
            "rhythm_dataset_target_mode": "cached_only",
            "rhythm_primary_target_surface": "guidance",
            "rhythm_distill_surface": "none",
            "rhythm_enable_dual_mode_teacher": False,
            "rhythm_enable_learned_offline_teacher": False,
            "rhythm_runtime_enable_learned_offline_teacher": False,
            "rhythm_teacher_as_main": False,
            "rhythm_export_debug_sidecars": True,
            "lambda_rhythm_guidance": 0.0,
            "lambda_rhythm_plan": 0.0,
            "lambda_rhythm_distill": 0.0,
            "lambda_rhythm_teacher_aux": 0.0,
        }
        report = validate_cache_field_contract(build_contract_context(hparams, model_dry_run=True))
        groups = set(report.required_field_groups)

        self.assertIn(("planner_slow_rhythm_memory",), groups)
        self.assertIn(("ref_phrase_trace",), groups)
        self.assertIn(("planner_ref_phrase_trace",), groups)
        self.assertIn(("ref_phrase_boundary_strength",), groups)


if __name__ == "__main__":
    unittest.main()
