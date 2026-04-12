from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.Conan.rhythm.config_contract import build_contract_context
from tasks.Conan.rhythm.config_contract_cache_rules import validate_cache_field_contract
from tasks.Conan.rhythm.dataset_contracts import RhythmDatasetCacheContract
from tasks.Conan.rhythm.dataset_mixin import RhythmConanDatasetMixin
from tasks.Conan.rhythm.duration_v3.dataset_mixin import _align_target_runs_to_source_discrete


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
            "rhythm_cache_version": 6,
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
            "rhythm_cache_version": 6,
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
            "rhythm_cache_version": 6,
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
            "rhythm_cache_version": 6,
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

    def test_prompt_summary_training_source_cache_uses_pseudo_source_duration_context(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "pseudo_source_duration_perturbation": True,
                "rhythm_augmentation_deterministic": True,
                "seed": 7,
            }
        )
        source_cache = {
            "content_units": np.asarray([1, 2, 3], dtype=np.int64),
            "dur_anchor_src": np.asarray([3.0, 4.0, 5.0], dtype=np.float32),
            "sep_hint": np.asarray([0, 0, 1], dtype=np.int64),
            "sealed_mask": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        }
        perturbed = dataset._maybe_build_duration_v3_training_source_cache(source_cache, item_name="utt_a")
        self.assertEqual(perturbed["dur_anchor_src"].shape, source_cache["dur_anchor_src"].shape)
        self.assertFalse(np.allclose(perturbed["dur_anchor_src"], source_cache["dur_anchor_src"]))
        self.assertTrue(np.all(perturbed["dur_anchor_src"] > 0.0))
        self.assertTrue(np.allclose(source_cache["dur_anchor_src"], np.asarray([3.0, 4.0, 5.0], dtype=np.float32)))

    def test_prompt_summary_prompt_unit_conditioning_supports_truncation_and_dropout(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_prompt_truncation": 2.0,
                "rhythm_prompt_dropout": 0.5,
                "rhythm_augmentation_deterministic": True,
                "seed": 11,
            }
        )
        prompt_item = {
            "item_name": "prompt_a",
            "content_units": np.asarray([1, 2, 3, 4], dtype=np.int64),
            "dur_anchor_src": np.asarray([3.0, 4.0, 2.0, 5.0], dtype=np.float32),
            "source_silence_mask": np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "sep_hint": np.asarray([0, 0, 0, 0], dtype=np.int64),
            "source_boundary_cue": np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            "phrase_group_pos": np.asarray([0.0, 0.1, 0.2, 0.3], dtype=np.float32),
            "phrase_final_mask": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        }
        conditioning = dataset._build_reference_prompt_unit_conditioning(prompt_item, target_mode="runtime_only")
        mask = conditioning["prompt_unit_mask"]
        self.assertGreaterEqual(float(mask.sum()), 1.0)
        self.assertLessEqual(float(mask.sum()), 2.0)
        self.assertGreaterEqual(float(np.asarray(conditioning["prompt_speech_mask"]).sum()), 1.0)
        zeroed = mask <= 0.0
        self.assertTrue(np.all(conditioning["prompt_duration_obs"][zeroed] == 0.0))
        self.assertTrue(np.all(conditioning["prompt_source_boundary_cue"][zeroed] == 0.0))

    def test_prompt_summary_conditioning_runtime_prompt_no_longer_embeds_paired_target_sidecars(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_emit_silence_runs": True,
                "rhythm_prompt_truncation": 2.0,
                "rhythm_prompt_dropout": 0.5,
                "rhythm_augmentation_deterministic": True,
                "seed": 5,
            }
        )
        prompt_item = {
            "item_name": "prompt_b",
            "content_units": np.asarray([1, 57, 2], dtype=np.int64),
            "dur_anchor_src": np.asarray([3.0, 2.0, 4.0], dtype=np.float32),
            "source_silence_mask": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        }
        conditioning = dataset._build_reference_prompt_unit_conditioning(prompt_item, target_mode="runtime_only")
        self.assertNotIn("prompt_target_duration_obs", conditioning)
        self.assertNotIn("prompt_target_speech_mask", conditioning)
        self.assertGreaterEqual(float(np.asarray(conditioning["prompt_unit_mask"]).sum()), 1.0)

    def test_duration_v3_target_merge_projects_explicit_paired_target_runs_onto_source_lattice(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_emit_silence_runs": True,
                "rhythm_v3_allow_source_self_target_fallback": False,
            }
        )
        source_cache = {
            "content_units": np.asarray([1, 57, 2], dtype=np.int64),
            "dur_anchor_src": np.asarray([2.0, 1.0, 2.0], dtype=np.float32),
            "source_silence_mask": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        }
        paired_target_conditioning = {
            "paired_target_content_units": np.asarray([1, 57, 2], dtype=np.int64),
            "paired_target_duration_obs": np.asarray([3.0, 2.0, 1.0], dtype=np.float32),
            "paired_target_valid_mask": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            "paired_target_speech_mask": np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
            "paired_target_item_name": np.asarray(["paired_ref"], dtype=object),
        }
        merged = dataset._merge_duration_v3_rhythm_targets(
            item={"item_name": "src_a"},
            source_cache=source_cache,
            paired_target_conditioning=paired_target_conditioning,
            sample={},
        )
        self.assertTrue(np.allclose(merged["unit_duration_tgt"], np.asarray([3.0, 2.0, 1.0], dtype=np.float32)))
        self.assertEqual(merged["unit_confidence_tgt"].shape, (3,))
        self.assertEqual(merged["unit_confidence_local_tgt"].shape, (3,))
        self.assertEqual(merged["unit_confidence_coarse_tgt"].shape, (3,))
        self.assertEqual(merged["unit_alignment_coverage_tgt"].shape, (3,))
        self.assertEqual(merged["unit_alignment_match_tgt"].shape, (3,))
        self.assertEqual(merged["unit_alignment_cost_tgt"].shape, (3,))
        self.assertGreater(float(merged["unit_confidence_local_tgt"][0]), 0.5)
        self.assertGreater(float(merged["unit_confidence_local_tgt"][2]), 0.5)
        self.assertGreater(float(merged["unit_confidence_coarse_tgt"][0]), 0.5)
        self.assertGreater(float(merged["unit_confidence_coarse_tgt"][2]), 0.5)
        self.assertEqual(float(merged["unit_confidence_local_tgt"][1]), 0.0)
        self.assertGreater(float(merged["unit_confidence_coarse_tgt"][1]), 0.0)
        self.assertLessEqual(float(merged["unit_confidence_coarse_tgt"][1]), 0.35)
        self.assertTrue(np.all((merged["unit_alignment_coverage_tgt"] >= 0.0) & (merged["unit_alignment_coverage_tgt"] <= 1.0)))
        self.assertTrue(np.all((merged["unit_alignment_match_tgt"] >= 0.0) & (merged["unit_alignment_match_tgt"] <= 1.0)))
        self.assertTrue(np.all(np.isfinite(merged["unit_alignment_cost_tgt"])))
        self.assertTrue(np.all(merged["unit_alignment_cost_tgt"] >= 0.0))

    def test_duration_v3_target_merge_rejects_cross_text_paired_projection_when_same_text_is_required(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_minimal_v1_profile": True,
                "rhythm_v3_require_same_text_paired_target": True,
                "rhythm_v3_emit_silence_runs": True,
                "rhythm_v3_allow_source_self_target_fallback": False,
            }
        )
        source_cache = {
            "content_units": np.asarray([1, 57, 2], dtype=np.int64),
            "dur_anchor_src": np.asarray([2.0, 1.0, 2.0], dtype=np.float32),
            "source_silence_mask": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        }
        paired_target_conditioning = {
            "paired_target_content_units": np.asarray([1, 57, 2], dtype=np.int64),
            "paired_target_duration_obs": np.asarray([3.0, 2.0, 1.0], dtype=np.float32),
            "paired_target_valid_mask": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            "paired_target_speech_mask": np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
            "paired_target_text_signature": np.asarray([("txt", "target-a")], dtype=object),
            "source_text_signature": np.asarray([("txt", "source-b")], dtype=object),
        }
        with self.assertRaisesRegex(RuntimeError, "same-text paired target"):
            dataset._merge_duration_v3_rhythm_targets(
                item={"item_name": "src_a"},
                source_cache=source_cache,
                paired_target_conditioning=paired_target_conditioning,
                sample={},
            )

    def test_duration_v3_discrete_alignment_keeps_skipped_target_unassigned(self) -> None:
        assigned_source, assigned_cost = _align_target_runs_to_source_discrete(
            source_units=np.asarray([1, 2], dtype=np.int64),
            source_durations=np.asarray([2.0, 2.0], dtype=np.float32),
            source_silence=np.asarray([0.0, 0.0], dtype=np.float32),
            target_units=np.asarray([1, 99, 2], dtype=np.int64),
            target_durations=np.asarray([2.0, 1.0, 2.0], dtype=np.float32),
            target_silence=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        )
        self.assertEqual(assigned_source.tolist(), [0, -1, 1])
        self.assertGreaterEqual(float(assigned_cost[1]), 0.0)

    def test_duration_v3_target_merge_preserves_split_confidence_fields_from_sample(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
            }
        )
        sample = {
            "unit_duration_tgt": np.asarray([2.0, 3.0], dtype=np.float32),
            "unit_confidence_local_tgt": np.asarray([0.0, 0.8], dtype=np.float32),
            "unit_confidence_coarse_tgt": np.asarray([0.6, 0.9], dtype=np.float32),
            "unit_alignment_coverage_tgt": np.asarray([0.0, 1.0], dtype=np.float32),
            "unit_alignment_match_tgt": np.asarray([0.0, 1.0], dtype=np.float32),
            "unit_alignment_cost_tgt": np.asarray([1.0, 0.0], dtype=np.float32),
        }
        merged = dataset._merge_duration_v3_rhythm_targets(
            item={"item_name": "src_passthrough"},
            source_cache={},
            paired_target_conditioning={},
            sample=sample,
        )
        self.assertTrue(np.allclose(merged["unit_duration_tgt"], sample["unit_duration_tgt"]))
        self.assertTrue(np.allclose(merged["unit_confidence_local_tgt"], sample["unit_confidence_local_tgt"]))
        self.assertTrue(np.allclose(merged["unit_confidence_coarse_tgt"], sample["unit_confidence_coarse_tgt"]))
        self.assertTrue(np.allclose(merged["unit_confidence_tgt"], sample["unit_confidence_coarse_tgt"]))
        self.assertTrue(np.allclose(merged["unit_alignment_coverage_tgt"], sample["unit_alignment_coverage_tgt"]))
        self.assertTrue(np.allclose(merged["unit_alignment_match_tgt"], sample["unit_alignment_match_tgt"]))
        self.assertTrue(np.allclose(merged["unit_alignment_cost_tgt"], sample["unit_alignment_cost_tgt"]))

    def test_duration_v3_target_merge_rejects_self_paired_target_projection_without_explicit_fallback(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_allow_source_self_target_fallback": False,
            }
        )
        with self.assertRaises(RuntimeError):
            dataset._merge_duration_v3_rhythm_targets(
                item={"item_name": "src_self"},
                source_cache={
                    "content_units": np.asarray([1, 2], dtype=np.int64),
                    "dur_anchor_src": np.asarray([2.0, 3.0], dtype=np.float32),
                },
                paired_target_conditioning={
                    "paired_target_content_units": np.asarray([1, 2], dtype=np.int64),
                    "paired_target_duration_obs": np.asarray([2.0, 3.0], dtype=np.float32),
                    "paired_target_valid_mask": np.asarray([1.0, 1.0], dtype=np.float32),
                    "paired_target_speech_mask": np.asarray([1.0, 1.0], dtype=np.float32),
                    "paired_target_item_name": np.asarray(["src_self"], dtype=object),
                },
                sample={},
            )

    def test_duration_v3_target_merge_no_longer_auto_falls_back_to_source_self(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_stage": "bootstrap_pretrain",
                "rhythm_v3_allow_source_self_target_fallback": False,
            }
        )
        with self.assertRaises(RuntimeError):
            dataset._merge_duration_v3_rhythm_targets(
                item={"item_name": "src_b"},
                source_cache={
                    "content_units": np.asarray([1, 2], dtype=np.int64),
                    "dur_anchor_src": np.asarray([2.0, 3.0], dtype=np.float32),
                },
                paired_target_conditioning={},
                sample={},
            )


if __name__ == "__main__":
    unittest.main()
