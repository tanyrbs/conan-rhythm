from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.Conan.rhythm.config_contract import build_contract_context
from tasks.Conan.rhythm.config_contract_cache_rules import validate_cache_field_contract
from tasks.Conan.rhythm.dataset_contracts import RhythmDatasetCacheContract
from tasks.Conan.rhythm.dataset_errors import RhythmDatasetPrefilterDrop
from tasks.Conan.rhythm.dataset_mixin import RhythmConanDatasetMixin
from tasks.Conan.rhythm.duration_v3.dataset_mixin import _align_target_runs_to_source_discrete
from modules.Conan.rhythm_v3.source_cache import duration_v3_cache_meta_signature


class _DummyDataset(RhythmConanDatasetMixin):
    def __init__(self, hparams: dict):
        self.hparams = hparams
        self.prefix = "train"


class _RetryDatasetBase:
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index):
        item_name = "bad_prefilter" if int(index) == 0 else "good_prefilter"
        return {
            "id": int(index),
            "item_id": int(index),
            "_raw_item": {"item_name": item_name, "content_units": np.asarray([1], dtype=np.int64)},
        }


class _RetryAssembler:
    def assemble(self, *, sample, item, ref_item, item_name):
        del ref_item
        if item_name == "bad_prefilter":
            raise RhythmDatasetPrefilterDrop("paired target alignment prefilter dropped bad_prefilter")
        sample["assembled_item_name"] = item_name
        return sample


class _RetryDataset(RhythmConanDatasetMixin, _RetryDatasetBase):
    def __init__(self):
        self.hparams = {
            "rhythm_enable_v3": True,
            "rhythm_v3_alignment_prefilter_bad_samples": True,
            "rhythm_v3_alignment_prefilter_max_attempts": 2,
        }
        self.prefix = "train"
        self._assembler = _RetryAssembler()

    def _materialize_rhythm_cache_compat(self, raw_item, *, item_name: str):
        del item_name
        return raw_item

    def _resolve_rhythm_target_mode(self):
        return "cached_only"

    def _rhythm_sample_assembler(self):
        return self._assembler


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
        self.assertTrue(np.all(conditioning["prompt_closed_mask"][zeroed] == 0.0))
        self.assertTrue(np.all(conditioning["prompt_boundary_confidence"][zeroed] == 0.0))
        self.assertTrue(np.all(conditioning["prompt_source_boundary_cue"][zeroed] == 0.0))

    def test_adapt_source_cache_to_visible_prefix_restores_phrase_debug_sidecars(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_export_debug_sidecars": True,
            }
        )
        item = {
            "item_name": "prefix_debug_sidecars",
            "content_units": np.asarray([1, 2, 3], dtype=np.int64),
            "dur_anchor_src": np.asarray([3, 2, 4], dtype=np.int64),
            "open_run_mask": np.asarray([0, 0, 0], dtype=np.int64),
            "sealed_mask": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            "sep_hint": np.asarray([0, 1, 0], dtype=np.int64),
            "source_silence_mask": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
            "source_boundary_cue": np.asarray([0.1, 0.7, 0.2], dtype=np.float32),
            "phrase_group_index": np.asarray([0, 1, 1], dtype=np.int64),
            "phrase_group_pos": np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
            "phrase_final_mask": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
            "boundary_confidence": np.asarray([0.2, 0.8, 0.4], dtype=np.float32),
        }
        adapted = dataset._adapt_source_cache_to_visible_prefix(
            item=item,
            visible_tokens=np.arange(9, dtype=np.int64),
        )
        self.assertTrue(np.allclose(adapted["source_boundary_cue"], item["source_boundary_cue"]))
        self.assertTrue(np.array_equal(adapted["phrase_group_index"], item["phrase_group_index"]))
        self.assertTrue(np.allclose(adapted["phrase_group_pos"], item["phrase_group_pos"]))
        self.assertTrue(np.allclose(adapted["phrase_final_mask"], item["phrase_final_mask"]))

    def test_adapt_source_cache_to_visible_prefix_falls_back_when_debug_sidecar_shape_mismatches(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_export_debug_sidecars": True,
            }
        )
        item = {
            "item_name": "prefix_debug_sidecars_bad_shape",
            "content_units": np.asarray([1, 2, 3], dtype=np.int64),
            "dur_anchor_src": np.asarray([3, 2, 4], dtype=np.int64),
            "open_run_mask": np.asarray([0, 0, 0], dtype=np.int64),
            "sealed_mask": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            "sep_hint": np.asarray([0, 1, 0], dtype=np.int64),
            "source_silence_mask": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
            "source_boundary_cue": np.asarray([0.1, 0.7], dtype=np.float32),
            "phrase_group_index": np.asarray([0, 1], dtype=np.int64),
            "phrase_group_pos": np.asarray([0.0, 0.5], dtype=np.float32),
            "phrase_final_mask": np.asarray([0.0, 1.0], dtype=np.float32),
            "boundary_confidence": np.asarray([0.2, 0.8, 0.4], dtype=np.float32),
        }
        adapted = dataset._adapt_source_cache_to_visible_prefix(
            item=item,
            visible_tokens=np.arange(9, dtype=np.int64),
        )
        self.assertTrue(np.allclose(adapted["source_boundary_cue"], adapted["boundary_confidence"]))
        self.assertTrue(np.array_equal(adapted["phrase_group_index"], np.zeros((3,), dtype=np.int64)))
        self.assertTrue(np.allclose(adapted["phrase_group_pos"], np.zeros((3,), dtype=np.float32)))
        self.assertTrue(np.allclose(adapted["phrase_final_mask"], np.zeros((3,), dtype=np.float32)))

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

    def test_prompt_summary_conditioning_derives_prompt_global_weight_from_run_stability(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_emit_silence_runs": True,
                "rhythm_v3_g_variant": "weighted_median",
            }
        )
        prompt_item = {
            "item_name": "prompt_weighted",
            "content_units": np.asarray([1, 57, 2], dtype=np.int64),
            "dur_anchor_src": np.asarray([3.0, 2.0, 4.0], dtype=np.float32),
            "dur_sec": 5.25,
            "source_silence_mask": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
            "source_run_stability": np.asarray([1.0, 0.2, 0.6], dtype=np.float32),
        }
        conditioning = dataset._build_reference_prompt_unit_conditioning(prompt_item, target_mode="runtime_only")
        expected = np.asarray([1.0, 0.0, 0.6], dtype=np.float32)
        self.assertIn("prompt_global_weight", conditioning)
        self.assertIn("prompt_global_weight_present", conditioning)
        self.assertIn("prompt_speech_ratio_scalar", conditioning)
        self.assertIn("prompt_ref_len_sec", conditioning)
        self.assertIn("g_trim_ratio", conditioning)
        self.assertEqual(float(np.asarray(conditioning["prompt_global_weight_present"]).reshape(-1)[0]), 1.0)
        self.assertTrue(np.allclose(conditioning["prompt_global_weight"], expected))
        self.assertTrue(
            np.allclose(
                np.asarray(conditioning["prompt_speech_ratio_scalar"], dtype=np.float32),
                np.asarray([7.0 / 9.0], dtype=np.float32),
            )
        )
        self.assertTrue(
            np.allclose(
                np.asarray(conditioning["prompt_ref_len_sec"], dtype=np.float32),
                np.asarray([5.25], dtype=np.float32),
            )
        )

    def test_prompt_summary_conditioning_drops_edge_speech_runs_from_prompt_global_weight(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_emit_silence_runs": True,
                "rhythm_v3_g_variant": "weighted_median",
                "rhythm_v3_drop_edge_runs_for_g": 1,
            }
        )
        prompt_item = {
            "item_name": "prompt_edge_drop",
            "content_units": np.asarray([1, 2, 3, 4, 5], dtype=np.int64),
            "dur_anchor_src": np.asarray([3.0, 3.0, 3.0, 3.0, 3.0], dtype=np.float32),
            "source_silence_mask": np.zeros((5,), dtype=np.float32),
            "source_run_stability": np.ones((5,), dtype=np.float32),
        }
        conditioning = dataset._build_reference_prompt_unit_conditioning(prompt_item, target_mode="runtime_only")
        self.assertTrue(
            np.allclose(
                conditioning["prompt_global_weight"],
                np.asarray([0.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32),
            )
        )

    def test_prompt_summary_conditioning_exports_closed_and_boundary_sidecars(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_emit_silence_runs": True,
                "rhythm_v3_g_variant": "weighted_median",
                "rhythm_v3_min_boundary_confidence_for_g": 0.5,
            }
        )
        prompt_item = {
            "item_name": "prompt_clean_support",
            "content_units": np.asarray([1, 2, 3, 4], dtype=np.int64),
            "dur_anchor_src": np.asarray([3.0, 3.0, 3.0, 3.0], dtype=np.float32),
            "source_silence_mask": np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
            "source_run_stability": np.asarray([1.0, 0.8, 0.9, 0.6], dtype=np.float32),
            "sealed_mask": np.asarray([1.0, 0.0, 1.0, 1.0], dtype=np.float32),
            "boundary_confidence": np.asarray([0.95, 0.8, 0.3, 0.4], dtype=np.float32),
        }
        conditioning = dataset._build_reference_prompt_unit_conditioning(prompt_item, target_mode="runtime_only")

        self.assertIn("prompt_closed_mask", conditioning)
        self.assertIn("prompt_boundary_confidence", conditioning)
        self.assertTrue(
            np.allclose(
                conditioning["prompt_closed_mask"],
                np.asarray([1.0, 0.0, 1.0, 1.0], dtype=np.float32),
            )
        )
        self.assertTrue(
            np.allclose(
                conditioning["prompt_boundary_confidence"],
                np.asarray([0.95, 0.8, 0.3, 0.4], dtype=np.float32),
            )
        )
        weights = np.asarray(conditioning["prompt_global_weight"], dtype=np.float32)
        self.assertEqual(weights.shape, (4,))
        self.assertGreater(weights[0], 0.8)
        self.assertEqual(weights[1], 0.0)
        self.assertEqual(weights[2], 0.0)
        self.assertGreater(weights[3], 0.0)
        self.assertLess(weights[3], weights[0])

    def test_minimal_v1_prompt_summary_conditioning_requires_closed_boundary_sidecars(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_minimal_v1_profile": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_rate_mode": "simple_global",
                "rhythm_v3_simple_global_stats": True,
                "rhythm_v3_emit_silence_runs": True,
            }
        )
        prompt_item = {
            "item_name": "prompt_missing_clean_sidecars",
            "content_units": np.asarray([1, 2, 3], dtype=np.int64),
            "dur_anchor_src": np.asarray([3.0, 3.0, 3.0], dtype=np.float32),
            "dur_sec": 4.5,
            "source_silence_mask": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            "source_run_stability": np.asarray([1.0, 0.9, 0.8], dtype=np.float32),
        }
        with self.assertRaisesRegex(RuntimeError, "sealed_mask/prompt_closed_mask"):
            dataset._build_reference_prompt_unit_conditioning(prompt_item, target_mode="runtime_only")

    def test_minimal_v1_prompt_summary_conditioning_requires_prompt_ref_len_sidecar(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_minimal_v1_profile": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_rate_mode": "simple_global",
                "rhythm_v3_simple_global_stats": True,
                "rhythm_v3_emit_silence_runs": True,
                "hop_size": 0,
                "audio_sample_rate": 0,
            }
        )
        prompt_item = {
            "item_name": "prompt_missing_ref_len",
            "content_units": np.asarray([1, 2, 3], dtype=np.int64),
            "dur_anchor_src": np.asarray([3.0, 3.0, 3.0], dtype=np.float32),
            "source_silence_mask": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            "source_run_stability": np.asarray([1.0, 0.9, 0.8], dtype=np.float32),
            "sealed_mask": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            "boundary_confidence": np.asarray([0.9, 0.9, 0.9], dtype=np.float32),
        }
        with self.assertRaisesRegex(RuntimeError, "prompt_ref_len_sec/ref_len_sec/dur_sec"):
            dataset._build_reference_prompt_unit_conditioning(prompt_item, target_mode="runtime_only")

    def test_minimal_v1_prompt_summary_conditioning_prefilters_low_speech_ratio(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_minimal_v1_profile": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_rate_mode": "simple_global",
                "rhythm_v3_simple_global_stats": True,
                "rhythm_v3_emit_silence_runs": True,
                "hop_size": 0,
                "audio_sample_rate": 0,
            }
        )
        prompt_item = {
            "item_name": "prompt_low_speech_ratio",
            "content_units": np.asarray([1, 2], dtype=np.int64),
            "dur_anchor_src": np.asarray([3.0, 3.0], dtype=np.float32),
            "ref_len_sec": 4.2,
            "source_silence_mask": np.asarray([1.0, 0.0], dtype=np.float32),
            "source_run_stability": np.asarray([0.8, 0.9], dtype=np.float32),
            "sealed_mask": np.asarray([1.0, 1.0], dtype=np.float32),
            "boundary_confidence": np.asarray([0.9, 0.95], dtype=np.float32),
        }
        with self.assertRaisesRegex(RhythmDatasetPrefilterDrop, "speech_ratio"):
            dataset._build_reference_prompt_unit_conditioning(prompt_item, target_mode="runtime_only")

    def test_minimal_v1_prompt_summary_conditioning_prefilters_ref_len_out_of_range(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_minimal_v1_profile": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_rate_mode": "simple_global",
                "rhythm_v3_simple_global_stats": True,
                "rhythm_v3_emit_silence_runs": True,
                "hop_size": 0,
                "audio_sample_rate": 0,
            }
        )
        prompt_item = {
            "item_name": "prompt_ref_too_long",
            "content_units": np.asarray([1, 2, 3], dtype=np.int64),
            "dur_anchor_src": np.asarray([3.0, 3.0, 3.0], dtype=np.float32),
            "ref_len_sec": 9.2,
            "source_silence_mask": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            "source_run_stability": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            "sealed_mask": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            "boundary_confidence": np.asarray([0.9, 0.9, 0.9], dtype=np.float32),
        }
        with self.assertRaisesRegex(RhythmDatasetPrefilterDrop, "ref_len_sec"):
            dataset._build_reference_prompt_unit_conditioning(prompt_item, target_mode="runtime_only")

    def test_prompt_summary_conditioning_rejects_prompt_weight_shape_mismatch_by_default(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_emit_silence_runs": True,
            }
        )
        prompt_item = {
            "item_name": "prompt_bad_weight_shape",
            "content_units": np.asarray([1, 57, 2], dtype=np.int64),
            "dur_anchor_src": np.asarray([3.0, 2.0, 4.0], dtype=np.float32),
            "source_silence_mask": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
            "source_run_stability": np.asarray([1.0, 0.2], dtype=np.float32),
        }
        with self.assertRaisesRegex(RuntimeError, "prompt_run_stability shape mismatch"):
            dataset._build_reference_prompt_unit_conditioning(prompt_item, target_mode="runtime_only")

    def test_prompt_summary_conditioning_can_repair_prompt_weight_shape_when_explicitly_enabled(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_emit_silence_runs": True,
                "rhythm_v3_allow_prompt_weight_shape_repair": True,
            }
        )
        prompt_item = {
            "item_name": "prompt_repair_weight_shape",
            "content_units": np.asarray([1, 57, 2], dtype=np.int64),
            "dur_anchor_src": np.asarray([3.0, 2.0, 4.0], dtype=np.float32),
            "source_silence_mask": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
            "source_run_stability": np.asarray([1.0, 0.2], dtype=np.float32),
        }
        conditioning = dataset._build_reference_prompt_unit_conditioning(prompt_item, target_mode="runtime_only")
        expected = np.asarray([1.0, 0.0, 1.0], dtype=np.float32)
        self.assertTrue(np.allclose(conditioning["prompt_global_weight"], expected))

    def test_prompt_summary_conditioning_rejects_unit_norm_without_prompt_prior(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_emit_silence_runs": True,
                "rhythm_v3_g_variant": "unit_norm",
            }
        )
        prompt_item = {
            "item_name": "prompt_unit_norm",
            "content_units": np.asarray([1, 2, 3], dtype=np.int64),
            "dur_anchor_src": np.asarray([3.0, 2.0, 4.0], dtype=np.float32),
            "source_silence_mask": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        }
        with self.assertRaisesRegex(RuntimeError, "prompt_unit_log_prior"):
            dataset._build_reference_prompt_unit_conditioning(prompt_item, target_mode="runtime_only")

    def test_prompt_summary_conditioning_can_attach_unit_prior_from_config_path(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_emit_silence_runs": True,
                "rhythm_v3_g_variant": "unit_norm",
                "rhythm_v3_unit_prior_path": "mock_prior_bundle.npz",
            }
        )
        prompt_item = {
            "item_name": "prompt_unit_norm_cfg",
            "content_units": np.asarray([1, 2], dtype=np.int64),
            "dur_anchor_src": np.asarray([3.0, 4.0], dtype=np.float32),
            "source_silence_mask": np.asarray([0.0, 0.0], dtype=np.float32),
        }
        with mock.patch(
            "tasks.Conan.rhythm.duration_v3.dataset_mixin.maybe_attach_unit_log_prior_from_path",
            return_value={
                **prompt_item,
                "unit_log_prior": np.asarray([0.2, 0.4], dtype=np.float32),
            },
        ) as mocked_attach:
            conditioning = dataset._build_reference_prompt_unit_conditioning(prompt_item, target_mode="runtime_only")

        mocked_attach.assert_called_once()
        self.assertIn("prompt_unit_log_prior", conditioning)
        self.assertIn("prompt_unit_log_prior_present", conditioning)
        self.assertEqual(float(np.asarray(conditioning["prompt_unit_log_prior_present"]).reshape(-1)[0]), 1.0)
        self.assertTrue(np.allclose(conditioning["prompt_unit_log_prior"], np.asarray([0.2, 0.4], dtype=np.float32)))

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
        self.assertEqual(float(merged["unit_confidence_coarse_tgt"][1]), 0.0)
        self.assertTrue(np.all((merged["unit_alignment_coverage_tgt"] >= 0.0) & (merged["unit_alignment_coverage_tgt"] <= 1.0)))
        self.assertTrue(np.all((merged["unit_alignment_match_tgt"] >= 0.0) & (merged["unit_alignment_match_tgt"] <= 1.0)))
        self.assertTrue(np.all(np.isfinite(merged["unit_alignment_cost_tgt"])))
        self.assertTrue(np.all(merged["unit_alignment_cost_tgt"] >= 0.0))

    def test_duration_v3_target_merge_can_enable_silence_aux_confidence_floor(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_emit_silence_runs": True,
                "rhythm_v3_allow_silence_aux": True,
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
        }
        merged = dataset._merge_duration_v3_rhythm_targets(
            item={"item_name": "src_a"},
            source_cache=source_cache,
            paired_target_conditioning=paired_target_conditioning,
            sample={},
        )
        self.assertGreater(float(merged["unit_confidence_coarse_tgt"][1]), 0.0)
        self.assertLessEqual(float(merged["unit_confidence_coarse_tgt"][1]), 0.35)

    def test_duration_v3_target_merge_multiplies_alignment_confidence_by_annotation_weight(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_emit_silence_runs": True,
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
            "unit_annotation_weight_tgt": np.asarray([1.0, 0.5, 0.25], dtype=np.float32),
        }
        merged = dataset._merge_duration_v3_rhythm_targets(
            item={"item_name": "src_a"},
            source_cache=source_cache,
            paired_target_conditioning=paired_target_conditioning,
            sample={},
        )
        self.assertTrue(
            np.allclose(
                merged["unit_annotation_weight_tgt"],
                np.asarray([1.0, 0.5, 0.25], dtype=np.float32),
            )
        )
        self.assertGreater(float(merged["unit_confidence_local_tgt"][0]), float(merged["unit_confidence_local_tgt"][2]))
        self.assertGreater(float(merged["unit_confidence_coarse_tgt"][0]), float(merged["unit_confidence_coarse_tgt"][2]))

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

    def test_duration_v3_target_merge_accepts_precomputed_alignment_metadata(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_emit_silence_runs": True,
                "rhythm_v3_use_continuous_alignment": True,
            }
        )
        merged = dataset._merge_duration_v3_rhythm_targets(
            item={"item_name": "src_align"},
            source_cache={
                "content_units": np.asarray([1, 2], dtype=np.int64),
                "dur_anchor_src": np.asarray([2.0, 2.0], dtype=np.float32),
                "source_silence_mask": np.asarray([0.0, 0.0], dtype=np.float32),
            },
            paired_target_conditioning={
                "paired_target_content_units": np.asarray([1, 99, 2], dtype=np.int64),
                "paired_target_duration_obs": np.asarray([2.0, 1.0, 3.0], dtype=np.float32),
                "paired_target_valid_mask": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
                "paired_target_speech_mask": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
                "paired_target_alignment_kind": "continuous_precomputed",
                "paired_target_alignment_source": "hubert_state_dp_v1",
                "paired_target_alignment_version": "2026-04-12",
                "paired_target_alignment_assigned_source": np.asarray([0, -1, 1], dtype=np.int64),
                "paired_target_alignment_assigned_cost": np.asarray([0.0, 0.4, 0.1], dtype=np.float32),
            },
            sample={},
        )
        self.assertTrue(np.allclose(merged["unit_duration_tgt"], np.asarray([2.0, 3.0], dtype=np.float32)))
        self.assertTrue(np.allclose(merged["unit_duration_proj_raw_tgt"], merged["unit_duration_tgt"]))
        self.assertEqual(int(np.asarray(merged["unit_alignment_mode_id_tgt"]).reshape(-1)[0]), 1)
        self.assertEqual(str(np.asarray(merged["unit_alignment_kind_tgt"], dtype=object).reshape(-1)[0]), "continuous_precomputed")
        self.assertEqual(merged["alignment_source"], "hubert_state_dp_v1")
        self.assertEqual(merged["alignment_version"], "2026-04-12")
        self.assertTrue(np.all(merged["unit_alignment_cost_tgt"] >= 0.0))
        self.assertEqual(float(merged["unit_alignment_coverage_tgt"][0]), 1.0)
        self.assertEqual(float(merged["unit_alignment_coverage_tgt"][1]), 1.0)
        self.assertEqual(float(np.asarray(merged["unit_alignment_unmatched_speech_ratio_tgt"]).reshape(-1)[0]), 0.0)

    def test_duration_v3_target_merge_accepts_explicit_continuous_viterbi_provenance(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_emit_silence_runs": True,
                "rhythm_v3_use_continuous_alignment": True,
            }
        )
        merged = dataset._merge_duration_v3_rhythm_targets(
            item={"item_name": "src_align_viterbi_cached"},
            source_cache={
                "content_units": np.asarray([1, 2], dtype=np.int64),
                "dur_anchor_src": np.asarray([2.0, 2.0], dtype=np.float32),
                "source_silence_mask": np.asarray([0.0, 0.0], dtype=np.float32),
            },
            paired_target_conditioning={
                "paired_target_content_units": np.asarray([1, 99, 2], dtype=np.int64),
                "paired_target_duration_obs": np.asarray([2.0, 1.0, 3.0], dtype=np.float32),
                "paired_target_valid_mask": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
                "paired_target_speech_mask": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
                "paired_target_alignment_kind": "continuous_viterbi_v1",
                "paired_target_alignment_source": "run_state_viterbi",
                "paired_target_alignment_version": "2026-04-13",
                "paired_target_alignment_assigned_source": np.asarray([0, -1, 1], dtype=np.int64),
                "paired_target_alignment_assigned_cost": np.asarray([0.0, 0.4, 0.1], dtype=np.float32),
            },
            sample={},
        )
        self.assertEqual(int(np.asarray(merged["unit_alignment_mode_id_tgt"]).reshape(-1)[0]), 2)
        self.assertEqual(
            str(np.asarray(merged["unit_alignment_kind_tgt"], dtype=object).reshape(-1)[0]),
            "continuous_viterbi_v1",
        )
        self.assertEqual(merged["alignment_source"], "run_state_viterbi")
        self.assertEqual(merged["alignment_version"], "2026-04-13")

    def test_duration_v3_target_builds_continuous_viterbi_targets_from_frame_sidecars(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_emit_silence_runs": True,
                "rhythm_v3_use_continuous_alignment": True,
            }
        )
        built = dataset._build_paired_duration_v3_targets(
            item={"item_name": "src_align_viterbi"},
            source_cache={
                "content_units": np.asarray([11, 12, 13], dtype=np.int64),
                "dur_anchor_src": np.asarray([2.0, 2.0, 2.0], dtype=np.float32),
                "source_silence_mask": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            },
            paired_target_conditioning={
                "paired_target_content_units": np.asarray([11, 12, 13], dtype=np.int64),
                "paired_target_duration_obs": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
                "paired_target_valid_mask": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
                "paired_target_speech_mask": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
                "source_frame_states": np.asarray(
                    [
                        [1.0, 0.0, 0.0],
                        [0.9, 0.1, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.9, 0.1],
                        [0.0, 0.0, 1.0],
                        [0.1, 0.0, 0.9],
                    ],
                    dtype=np.float32,
                ),
                "source_frame_to_run": np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64),
                "paired_target_frame_states": np.asarray(
                    [
                        [1.0, 0.0, 0.0],
                        [0.9, 0.1, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.1, 0.9],
                        [0.0, 0.0, 1.0],
                        [0.1, 0.0, 0.9],
                    ],
                    dtype=np.float32,
                ),
                "paired_target_frame_speech_prob": np.ones((6,), dtype=np.float32),
                "paired_target_frame_valid": np.ones((6,), dtype=np.float32),
            },
        )
        self.assertTrue(np.allclose(built["unit_duration_tgt"], np.asarray([2.0, 1.0, 3.0], dtype=np.float32)))
        self.assertEqual(int(np.asarray(built["unit_alignment_mode_id_tgt"]).reshape(-1)[0]), 2)
        self.assertEqual(
            str(np.asarray(built["unit_alignment_kind_tgt"], dtype=object).reshape(-1)[0]),
            "continuous_viterbi_v1",
        )
        self.assertEqual(built["alignment_source"], "run_state_viterbi")
        self.assertEqual(built["alignment_version"], "2026-04-13")
        self.assertEqual(
            str(np.asarray(built["unit_alignment_source_tgt"], dtype=object).reshape(-1)[0]),
            "run_state_viterbi",
        )
        self.assertEqual(
            str(np.asarray(built["unit_alignment_version_tgt"], dtype=object).reshape(-1)[0]),
            "2026-04-13",
        )
        self.assertAlmostEqual(float(np.asarray(built["unit_alignment_is_continuous_tgt"]).reshape(-1)[0]), 1.0)
        self.assertTrue(
            np.allclose(
                np.asarray(built["unit_logstretch_proj_raw_tgt"], dtype=np.float32),
                np.log(np.asarray([2.0, 1.0, 3.0], dtype=np.float32) + 1.0e-6)
                - np.log(np.asarray([2.0, 2.0, 2.0], dtype=np.float32) + 1.0e-6),
                atol=1.0e-5,
            )
        )
        self.assertAlmostEqual(float(np.asarray(built["unit_alignment_band_ratio_tgt"]).reshape(-1)[0]), 0.08)
        self.assertAlmostEqual(float(np.asarray(built["unit_alignment_lambda_emb_tgt"]).reshape(-1)[0]), 1.0)
        self.assertAlmostEqual(float(np.asarray(built["unit_alignment_lambda_type_tgt"]).reshape(-1)[0]), 0.5)

    def test_duration_v3_target_build_rejects_noncontinuous_minimal_projection(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_minimal_v1_profile": True,
                "rhythm_v3_use_continuous_alignment": True,
            }
        )
        projection = {
            "projected": np.asarray([2.0, 3.0], dtype=np.float32),
            "confidence_local": np.asarray([0.9, 0.9], dtype=np.float32),
            "confidence_coarse": np.asarray([0.9, 0.9], dtype=np.float32),
            "coverage": np.asarray([1.0, 1.0], dtype=np.float32),
            "match_rate": np.asarray([1.0, 1.0], dtype=np.float32),
            "mean_cost": np.asarray([0.1, 0.1], dtype=np.float32),
            "alignment_kind": "discrete",
            "unmatched_speech_ratio": 0.0,
            "mean_local_confidence_speech": 0.9,
            "mean_coarse_confidence_speech": 0.9,
        }
        with mock.patch(
            "tasks.Conan.rhythm.duration_v3.dataset_mixin._project_target_runs_onto_source",
            return_value=projection,
        ):
            with self.assertRaisesRegex(RuntimeError, "continuous paired-target alignment"):
                dataset._build_paired_duration_v3_targets(
                    item={"item_name": "src_minimal_discrete"},
                    source_cache={
                        "content_units": np.asarray([1, 2], dtype=np.int64),
                        "dur_anchor_src": np.asarray([2.0, 3.0], dtype=np.float32),
                        "source_silence_mask": np.asarray([0.0, 0.0], dtype=np.float32),
                    },
                    paired_target_conditioning={
                        "paired_target_content_units": np.asarray([1, 2], dtype=np.int64),
                        "paired_target_duration_obs": np.asarray([2.0, 3.0], dtype=np.float32),
                        "paired_target_valid_mask": np.asarray([1.0, 1.0], dtype=np.float32),
                        "paired_target_speech_mask": np.asarray([1.0, 1.0], dtype=np.float32),
                    },
                )

    def test_duration_v3_target_build_rejects_low_quality_minimal_continuous_projection(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_minimal_v1_profile": True,
                "rhythm_v3_use_continuous_alignment": True,
                "rhythm_v3_alignment_unmatched_speech_ratio_max": 0.15,
                "rhythm_v3_alignment_mean_local_confidence_speech_min": 0.55,
                "rhythm_v3_alignment_mean_coarse_confidence_speech_min": 0.60,
            }
        )
        projection = {
            "projected": np.asarray([2.0, 3.0], dtype=np.float32),
            "confidence_local": np.asarray([0.4, 0.4], dtype=np.float32),
            "confidence_coarse": np.asarray([0.5, 0.5], dtype=np.float32),
            "coverage": np.asarray([1.0, 1.0], dtype=np.float32),
            "match_rate": np.asarray([1.0, 1.0], dtype=np.float32),
            "mean_cost": np.asarray([0.4, 0.5], dtype=np.float32),
            "alignment_kind": "continuous_viterbi_v1",
            "alignment_source": "run_state_viterbi",
            "alignment_version": "2026-04-13",
            "unmatched_speech_ratio": 0.2,
            "mean_local_confidence_speech": 0.4,
            "mean_coarse_confidence_speech": 0.5,
        }
        with mock.patch(
            "tasks.Conan.rhythm.duration_v3.dataset_mixin._project_target_runs_onto_source",
            return_value=projection,
        ):
            with self.assertRaisesRegex(RuntimeError, "unmatched_speech_ratio|mean_local_confidence_speech"):
                dataset._build_paired_duration_v3_targets(
                    item={"item_name": "src_minimal_low_quality"},
                    source_cache={
                        "content_units": np.asarray([1, 2], dtype=np.int64),
                        "dur_anchor_src": np.asarray([2.0, 3.0], dtype=np.float32),
                        "source_silence_mask": np.asarray([0.0, 0.0], dtype=np.float32),
                    },
                    paired_target_conditioning={
                        "paired_target_content_units": np.asarray([1, 2], dtype=np.int64),
                        "paired_target_duration_obs": np.asarray([2.0, 3.0], dtype=np.float32),
                        "paired_target_valid_mask": np.asarray([1.0, 1.0], dtype=np.float32),
                        "paired_target_speech_mask": np.asarray([1.0, 1.0], dtype=np.float32),
                    },
                )

    def test_duration_v3_target_build_strict_gate_rejects_low_quality_projection_outside_minimal(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_use_continuous_alignment": True,
                "rhythm_v3_gate_quality_strict": True,
                "rhythm_v3_alignment_unmatched_speech_ratio_max": 0.15,
                "rhythm_v3_alignment_mean_local_confidence_speech_min": 0.55,
                "rhythm_v3_alignment_mean_coarse_confidence_speech_min": 0.60,
            }
        )
        projection = {
            "projected": np.asarray([2.0, 3.0], dtype=np.float32),
            "confidence_local": np.asarray([0.4, 0.4], dtype=np.float32),
            "confidence_coarse": np.asarray([0.5, 0.5], dtype=np.float32),
            "coverage": np.asarray([1.0, 1.0], dtype=np.float32),
            "match_rate": np.asarray([1.0, 1.0], dtype=np.float32),
            "mean_cost": np.asarray([0.4, 0.5], dtype=np.float32),
            "alignment_kind": "continuous_viterbi_v1",
            "alignment_source": "run_state_viterbi",
            "alignment_version": "2026-04-13",
            "unmatched_speech_ratio": 0.2,
            "mean_local_confidence_speech": 0.4,
            "mean_coarse_confidence_speech": 0.5,
        }
        with mock.patch(
            "tasks.Conan.rhythm.duration_v3.dataset_mixin._project_target_runs_onto_source",
            return_value=projection,
        ):
            with self.assertRaisesRegex(RuntimeError, "paired target alignment rejected.*unmatched_speech_ratio"):
                dataset._build_paired_duration_v3_targets(
                    item={"item_name": "src_strict_low_quality"},
                    source_cache={
                        "content_units": np.asarray([1, 2], dtype=np.int64),
                        "dur_anchor_src": np.asarray([2.0, 3.0], dtype=np.float32),
                        "source_silence_mask": np.asarray([0.0, 0.0], dtype=np.float32),
                    },
                    paired_target_conditioning={
                        "paired_target_content_units": np.asarray([1, 2], dtype=np.int64),
                        "paired_target_duration_obs": np.asarray([2.0, 3.0], dtype=np.float32),
                        "paired_target_valid_mask": np.asarray([1.0, 1.0], dtype=np.float32),
                        "paired_target_speech_mask": np.asarray([1.0, 1.0], dtype=np.float32),
                    },
                )

    def test_duration_v3_target_build_falls_back_to_discrete_when_continuous_alignment_raises(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_use_continuous_alignment": True,
                "rhythm_v3_allow_discrete_fallback_from_continuous": True,
            }
        )
        discrete_projection = {
            "projected": np.asarray([2.0, 3.0], dtype=np.float32),
            "projected_weighted": np.asarray([2.0, 3.0], dtype=np.float32),
            "confidence_local": np.asarray([0.8, 0.6], dtype=np.float32),
            "confidence_coarse": np.asarray([0.9, 0.7], dtype=np.float32),
            "coverage": np.asarray([1.0, 1.0], dtype=np.float32),
            "match_rate": np.asarray([1.0, 1.0], dtype=np.float32),
            "mean_cost": np.asarray([0.1, 0.1], dtype=np.float32),
            "alignment_kind": "discrete",
            "unmatched_speech_ratio": 0.0,
            "mean_local_confidence_speech": 0.7,
            "mean_coarse_confidence_speech": 0.8,
        }
        with mock.patch(
            "tasks.Conan.rhythm.duration_v3.dataset_mixin._project_target_runs_onto_source",
            side_effect=[RuntimeError("band failure"), discrete_projection],
        ) as project_mock:
            built = dataset._build_paired_duration_v3_targets(
                item={"item_name": "src_fallback"},
                source_cache={
                    "content_units": np.asarray([1, 2], dtype=np.int64),
                    "dur_anchor_src": np.asarray([2.0, 3.0], dtype=np.float32),
                    "source_silence_mask": np.asarray([0.0, 0.0], dtype=np.float32),
                },
                paired_target_conditioning={
                    "paired_target_content_units": np.asarray([1, 2], dtype=np.int64),
                    "paired_target_duration_obs": np.asarray([2.0, 3.0], dtype=np.float32),
                    "paired_target_valid_mask": np.asarray([1.0, 1.0], dtype=np.float32),
                    "paired_target_speech_mask": np.asarray([1.0, 1.0], dtype=np.float32),
                    "source_frame_states": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                    "source_frame_to_run": np.asarray([0, 1], dtype=np.int64),
                    "paired_target_frame_states": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                },
            )
        self.assertEqual(project_mock.call_count, 2)
        self.assertTrue(bool(project_mock.call_args_list[0].kwargs["use_continuous_alignment"]))
        self.assertFalse(bool(project_mock.call_args_list[1].kwargs["use_continuous_alignment"]))
        self.assertEqual(
            str(np.asarray(built["unit_alignment_kind_tgt"], dtype=object).reshape(-1)[0]),
            "discrete",
        )
        self.assertAlmostEqual(float(np.asarray(built["unit_alignment_fallback_from_continuous_tgt"]).reshape(-1)[0]), 1.0)
        self.assertIn(
            "band failure",
            str(np.asarray(built["unit_alignment_fallback_reason_tgt"], dtype=object).reshape(-1)[0]),
        )
        self.assertTrue(
            np.allclose(
                built["unit_confidence_local_tgt"],
                np.asarray([0.4, 0.3], dtype=np.float32),
            )
        )
        self.assertTrue(
            np.allclose(
                built["unit_confidence_coarse_tgt"],
                np.asarray([0.63, 0.49], dtype=np.float32),
            )
        )

    def test_duration_v3_target_build_exports_projection_health_surface(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_use_continuous_alignment": True,
            }
        )
        projection = {
            "projected": np.asarray([2.0, 3.0], dtype=np.float32),
            "projected_weighted": np.asarray([2.0, 3.0], dtype=np.float32),
            "confidence_local": np.asarray([0.8, 0.6], dtype=np.float32),
            "confidence_coarse": np.asarray([0.9, 0.7], dtype=np.float32),
            "coverage": np.asarray([1.0, 1.0], dtype=np.float32),
            "match_rate": np.asarray([1.0, 1.0], dtype=np.float32),
            "mean_cost": np.asarray([0.1, 0.1], dtype=np.float32),
            "alignment_kind": "continuous_viterbi_v1",
            "alignment_source": "run_state_viterbi",
            "alignment_version": "2026-04-13",
            "unmatched_speech_ratio": 0.25,
            "mean_local_confidence_speech": 0.4,
            "mean_coarse_confidence_speech": 0.7,
            "projection_fallback_ratio": np.asarray([0.25], dtype=np.float32),
            "projection_hard_bad": np.asarray([1.0], dtype=np.float32),
        }
        with mock.patch(
            "tasks.Conan.rhythm.duration_v3.dataset_mixin._project_target_runs_onto_source",
            return_value=projection,
        ):
            built = dataset._build_paired_duration_v3_targets(
                item={"item_name": "src_projection_health"},
                source_cache={
                    "content_units": np.asarray([1, 2], dtype=np.int64),
                    "dur_anchor_src": np.asarray([2.0, 3.0], dtype=np.float32),
                    "source_silence_mask": np.asarray([0.0, 0.0], dtype=np.float32),
                },
                paired_target_conditioning={
                    "paired_target_content_units": np.asarray([1, 2], dtype=np.int64),
                    "paired_target_duration_obs": np.asarray([2.0, 3.0], dtype=np.float32),
                    "paired_target_valid_mask": np.asarray([1.0, 1.0], dtype=np.float32),
                    "paired_target_speech_mask": np.asarray([1.0, 1.0], dtype=np.float32),
                },
            )

        self.assertAlmostEqual(
            float(np.asarray(built["unit_alignment_projection_fallback_ratio_tgt"]).reshape(-1)[0]),
            0.25,
            places=6,
        )
        self.assertAlmostEqual(
            float(np.asarray(built["unit_alignment_projection_hard_bad_tgt"]).reshape(-1)[0]),
            1.0,
            places=6,
        )
        self.assertIn("unit_alignment_local_margin_p10_tgt", built)
        self.assertTrue(np.isnan(float(np.asarray(built["unit_alignment_local_margin_p10_tgt"]).reshape(-1)[0])))

    def test_duration_v3_target_build_prefilter_drops_hard_bad_projection_when_enabled(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_use_continuous_alignment": True,
                "rhythm_v3_alignment_prefilter_bad_samples": True,
            }
        )
        projection = {
            "projected": np.asarray([2.0, 3.0], dtype=np.float32),
            "projected_weighted": np.asarray([2.0, 3.0], dtype=np.float32),
            "confidence_local": np.asarray([0.8, 0.6], dtype=np.float32),
            "confidence_coarse": np.asarray([0.9, 0.7], dtype=np.float32),
            "coverage": np.asarray([1.0, 1.0], dtype=np.float32),
            "match_rate": np.asarray([1.0, 1.0], dtype=np.float32),
            "mean_cost": np.asarray([0.1, 0.1], dtype=np.float32),
            "alignment_kind": "continuous_viterbi_v1",
            "alignment_source": "run_state_viterbi",
            "alignment_version": "2026-04-13",
            "unmatched_speech_ratio": 0.25,
            "mean_local_confidence_speech": 0.4,
            "mean_coarse_confidence_speech": 0.7,
            "projection_fallback_ratio": np.asarray([0.25], dtype=np.float32),
            "projection_hard_bad": np.asarray([1.0], dtype=np.float32),
            "projection_gate_reason": "unmatched_speech_ratio=0.250 > 0.150",
            "projection_health_bucket": "hard_bad",
        }
        with mock.patch(
            "tasks.Conan.rhythm.duration_v3.dataset_mixin._project_target_runs_onto_source",
            return_value=projection,
        ):
            with self.assertRaisesRegex(RhythmDatasetPrefilterDrop, "prefilter dropped"):
                dataset._build_paired_duration_v3_targets(
                    item={"item_name": "src_prefilter_drop"},
                    source_cache={
                        "content_units": np.asarray([1, 2], dtype=np.int64),
                        "dur_anchor_src": np.asarray([2.0, 3.0], dtype=np.float32),
                        "source_silence_mask": np.asarray([0.0, 0.0], dtype=np.float32),
                    },
                    paired_target_conditioning={
                        "paired_target_content_units": np.asarray([1, 2], dtype=np.int64),
                        "paired_target_duration_obs": np.asarray([2.0, 3.0], dtype=np.float32),
                        "paired_target_valid_mask": np.asarray([1.0, 1.0], dtype=np.float32),
                        "paired_target_speech_mask": np.asarray([1.0, 1.0], dtype=np.float32),
                    },
                )

    def test_common_rhythm_dataset_getitem_retries_bad_alignment_prefilter(self) -> None:
        dataset = _RetryDataset()
        with mock.patch("numpy.random.randint", return_value=1):
            sample = dataset[0]

        self.assertEqual(sample["assembled_item_name"], "good_prefilter")

    def test_duration_v3_target_merge_preserves_cached_projection_health_surface(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_use_continuous_alignment": True,
            }
        )
        merged = dataset._merge_duration_v3_rhythm_targets(
            item={"item_name": "src_cached_projection_health"},
            source_cache={
                "content_units": np.asarray([1, 2], dtype=np.int64),
                "dur_anchor_src": np.asarray([2.0, 3.0], dtype=np.float32),
                "source_silence_mask": np.asarray([0.0, 0.0], dtype=np.float32),
            },
            paired_target_conditioning={},
            sample={
                "unit_duration_tgt": np.asarray([2.0, 3.0], dtype=np.float32),
                "unit_duration_proj_raw_tgt": np.asarray([2.0, 3.0], dtype=np.float32),
                "unit_alignment_mode_id_tgt": np.asarray([2], dtype=np.int64),
                "unit_alignment_kind_tgt": np.asarray(["continuous_viterbi_v1"], dtype=object),
                "unit_alignment_source_tgt": np.asarray(["run_state_viterbi"], dtype=object),
                "unit_alignment_version_tgt": np.asarray(["2026-04-13"], dtype=object),
                "unit_alignment_unmatched_speech_ratio_tgt": np.asarray([0.25], dtype=np.float32),
                "unit_alignment_mean_local_confidence_speech_tgt": np.asarray([0.4], dtype=np.float32),
                "unit_alignment_mean_coarse_confidence_speech_tgt": np.asarray([0.7], dtype=np.float32),
                "unit_alignment_projection_fallback_ratio_tgt": np.asarray([0.25], dtype=np.float32),
                "unit_alignment_projection_hard_bad_tgt": np.asarray([1.0], dtype=np.float32),
            },
        )

        self.assertAlmostEqual(
            float(np.asarray(merged["unit_alignment_projection_fallback_ratio_tgt"]).reshape(-1)[0]),
            0.25,
            places=6,
        )
        self.assertAlmostEqual(
            float(np.asarray(merged["unit_alignment_projection_hard_bad_tgt"]).reshape(-1)[0]),
            1.0,
            places=6,
        )

    def test_duration_v3_target_build_re_raises_when_discrete_fallback_disabled(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_use_continuous_alignment": True,
                "rhythm_v3_allow_discrete_fallback_from_continuous": False,
            }
        )
        with mock.patch(
            "tasks.Conan.rhythm.duration_v3.dataset_mixin._project_target_runs_onto_source",
            side_effect=RuntimeError("band failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "band failure"):
                dataset._build_paired_duration_v3_targets(
                    item={"item_name": "src_no_fallback"},
                    source_cache={
                        "content_units": np.asarray([1, 2], dtype=np.int64),
                        "dur_anchor_src": np.asarray([2.0, 3.0], dtype=np.float32),
                        "source_silence_mask": np.asarray([0.0, 0.0], dtype=np.float32),
                    },
                    paired_target_conditioning={
                        "paired_target_content_units": np.asarray([1, 2], dtype=np.int64),
                        "paired_target_duration_obs": np.asarray([2.0, 3.0], dtype=np.float32),
                        "paired_target_valid_mask": np.asarray([1.0, 1.0], dtype=np.float32),
                        "paired_target_speech_mask": np.asarray([1.0, 1.0], dtype=np.float32),
                        "source_frame_states": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                        "source_frame_to_run": np.asarray([0, 1], dtype=np.int64),
                        "paired_target_frame_states": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                    },
                )

    def test_duration_v3_target_build_can_select_weighted_occupancy_surface(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_use_continuous_alignment": True,
                "rhythm_v3_continuous_projection_surface": "weighted_occupancy",
            }
        )
        projection = {
            "projected": np.asarray([2.0, 1.0], dtype=np.float32),
            "projected_weighted": np.asarray([3.0, 4.0], dtype=np.float32),
            "confidence_local": np.asarray([0.8, 0.7], dtype=np.float32),
            "confidence_coarse": np.asarray([0.9, 0.85], dtype=np.float32),
            "coverage": np.asarray([1.0, 1.0], dtype=np.float32),
            "match_rate": np.asarray([1.0, 1.0], dtype=np.float32),
            "mean_cost": np.asarray([0.1, 0.2], dtype=np.float32),
            "alignment_kind": "continuous_viterbi_v1",
            "alignment_source": "run_state_viterbi",
            "alignment_version": "2026-04-13",
            "unmatched_speech_ratio": 0.0,
            "mean_local_confidence_speech": 0.75,
            "mean_coarse_confidence_speech": 0.85,
        }
        with mock.patch(
            "tasks.Conan.rhythm.duration_v3.dataset_mixin._project_target_runs_onto_source",
            return_value=projection,
        ):
            built = dataset._build_paired_duration_v3_targets(
                item={"item_name": "src_weighted_surface"},
                source_cache={
                    "content_units": np.asarray([1, 2], dtype=np.int64),
                    "dur_anchor_src": np.asarray([2.0, 2.0], dtype=np.float32),
                    "source_silence_mask": np.asarray([0.0, 0.0], dtype=np.float32),
                },
                paired_target_conditioning={
                    "paired_target_content_units": np.asarray([1, 2], dtype=np.int64),
                    "paired_target_duration_obs": np.asarray([2.0, 3.0], dtype=np.float32),
                    "paired_target_valid_mask": np.asarray([1.0, 1.0], dtype=np.float32),
                    "paired_target_speech_mask": np.asarray([1.0, 1.0], dtype=np.float32),
                },
            )
        self.assertTrue(np.allclose(built["unit_duration_tgt"], np.asarray([3.0, 4.0], dtype=np.float32)))
        self.assertTrue(np.allclose(built["unit_duration_proj_raw_tgt"], np.asarray([2.0, 1.0], dtype=np.float32)))
        self.assertTrue(np.allclose(built["unit_duration_proj_weighted_tgt"], np.asarray([3.0, 4.0], dtype=np.float32)))
        self.assertEqual(
            str(np.asarray(built["unit_duration_projection_surface_tgt"], dtype=object).reshape(-1)[0]),
            "weighted_occupancy",
        )

    def test_duration_v3_target_build_rejects_unknown_continuous_projection_surface(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_use_continuous_alignment": True,
                "rhythm_v3_continuous_projection_surface": "mystery_surface",
            }
        )
        projection = {
            "projected": np.asarray([2.0, 1.0], dtype=np.float32),
            "projected_weighted": np.asarray([3.0, 4.0], dtype=np.float32),
            "confidence_local": np.asarray([0.8, 0.7], dtype=np.float32),
            "confidence_coarse": np.asarray([0.9, 0.85], dtype=np.float32),
            "coverage": np.asarray([1.0, 1.0], dtype=np.float32),
            "match_rate": np.asarray([1.0, 1.0], dtype=np.float32),
            "mean_cost": np.asarray([0.1, 0.2], dtype=np.float32),
            "alignment_kind": "continuous_viterbi_v1",
            "alignment_source": "run_state_viterbi",
            "alignment_version": "2026-04-13",
            "unmatched_speech_ratio": 0.0,
            "mean_local_confidence_speech": 0.75,
            "mean_coarse_confidence_speech": 0.85,
        }
        with mock.patch(
            "tasks.Conan.rhythm.duration_v3.dataset_mixin._project_target_runs_onto_source",
            return_value=projection,
        ):
            with self.assertRaisesRegex(RuntimeError, "rhythm_v3_continuous_projection_surface"):
                dataset._build_paired_duration_v3_targets(
                    item={"item_name": "src_bad_surface"},
                    source_cache={
                        "content_units": np.asarray([1, 2], dtype=np.int64),
                        "dur_anchor_src": np.asarray([2.0, 2.0], dtype=np.float32),
                        "source_silence_mask": np.asarray([0.0, 0.0], dtype=np.float32),
                    },
                    paired_target_conditioning={
                        "paired_target_content_units": np.asarray([1, 2], dtype=np.int64),
                        "paired_target_duration_obs": np.asarray([2.0, 3.0], dtype=np.float32),
                        "paired_target_valid_mask": np.asarray([1.0, 1.0], dtype=np.float32),
                        "paired_target_speech_mask": np.asarray([1.0, 1.0], dtype=np.float32),
                    },
                )

    def test_duration_v3_target_build_passes_continuous_aligner_hparams_and_exports_alignment_diagnostics(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_use_continuous_alignment": True,
                "rhythm_v3_alignment_mode": "continuous_viterbi_v1",
                "rhythm_v3_alignment_lambda_emb": 1.4,
                "rhythm_v3_alignment_lambda_type": 0.7,
                "rhythm_v3_alignment_lambda_band": 0.3,
                "rhythm_v3_alignment_lambda_unit": 0.2,
                "rhythm_v3_alignment_band_ratio": 0.12,
                "rhythm_v3_alignment_bad_cost_threshold": 1.6,
                "rhythm_v3_alignment_min_dp_weight": 0.2,
                "rhythm_v3_alignment_allow_source_skip": True,
                "rhythm_v3_alignment_skip_penalty": 0.45,
            }
        )
        projection = {
            "projected": np.asarray([2.0, 0.0, 3.0], dtype=np.float32),
            "confidence_local": np.asarray([0.9, 0.0, 0.8], dtype=np.float32),
            "confidence_coarse": np.asarray([0.8, 0.0, 0.7], dtype=np.float32),
            "coverage": np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
            "coverage_binary": np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
            "coverage_fraction": np.asarray([1.0, 0.0, 0.75], dtype=np.float32),
            "expected_frame_support": np.asarray([2.0, 1.0, 4.0], dtype=np.float32),
            "match_rate": np.asarray([1.0, 0.0, 0.8], dtype=np.float32),
            "mean_cost": np.asarray([0.1, 2.0, 0.3], dtype=np.float32),
            "confidence_cost_term": np.asarray([0.9, 0.0, 0.7], dtype=np.float32),
            "confidence_margin_term": np.asarray([0.8, 0.0, 0.6], dtype=np.float32),
            "confidence_type_term": np.asarray([1.0, 0.0, 0.9], dtype=np.float32),
            "confidence_match_term": np.asarray([1.0, 0.0, 0.8], dtype=np.float32),
            "run_skip_flag": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
            "alignment_kind": "continuous_viterbi_v1",
            "alignment_source": "run_state_viterbi",
            "alignment_version": "2026-04-13",
            "unmatched_speech_ratio": 0.1,
            "projection_fallback_ratio": 0.1,
            "projection_hard_bad": 0.0,
            "mean_local_confidence_speech": 0.85,
            "mean_coarse_confidence_speech": 0.75,
            "alignment_global_path_cost": 1.5,
            "alignment_global_path_mean_cost": 0.5,
            "alignment_lambda_emb": np.asarray([1.4], dtype=np.float32),
            "alignment_lambda_type": np.asarray([0.7], dtype=np.float32),
            "alignment_band_ratio": np.asarray([0.12], dtype=np.float32),
            "alignment_min_dp_weight": np.asarray([0.2], dtype=np.float32),
            "source_valid_run_index": np.asarray([0, 1, 2], dtype=np.int64),
            "target_valid_run_index": np.asarray([0, 1, 2], dtype=np.int64),
            "target_valid_observation_index": np.asarray([0, 2, 3], dtype=np.int64),
            "target_valid_observation_index_pre_dp_filter": np.asarray([0, 1, 2, 3], dtype=np.int64),
            "target_dp_weight_pruned_index": np.asarray([1], dtype=np.int64),
        }
        with mock.patch(
            "tasks.Conan.rhythm.duration_v3.dataset_mixin._project_target_runs_onto_source",
            return_value=projection,
        ) as mocked_project:
            built = dataset._build_paired_duration_v3_targets(
                item={"item_name": "src_align_kwargs"},
                source_cache={
                    "content_units": np.asarray([1, 2, 3], dtype=np.int64),
                    "dur_anchor_src": np.asarray([2.0, 1.0, 4.0], dtype=np.float32),
                    "source_silence_mask": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
                },
                paired_target_conditioning={
                    "paired_target_content_units": np.asarray([1, 2, 3], dtype=np.int64),
                    "paired_target_duration_obs": np.asarray([2.0, 1.0, 3.0], dtype=np.float32),
                    "paired_target_valid_mask": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
                    "paired_target_speech_mask": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
                },
            )

        kwargs = mocked_project.call_args.kwargs
        self.assertEqual(kwargs["continuous_alignment_mode"], "continuous_viterbi_v1")
        self.assertEqual(
            kwargs["continuous_aligner_kwargs"],
            {
                "lambda_emb": 1.4,
                "lambda_type": 0.7,
                "lambda_band": 0.3,
                "lambda_unit": 0.2,
                "band_ratio": 0.12,
                "bad_cost_threshold": 1.6,
                "min_dp_weight": 0.2,
                "allow_source_skip": True,
                "skip_penalty": 0.45,
            },
        )
        self.assertTrue(
            np.allclose(
                built["unit_alignment_coverage_binary_tgt"],
                np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
            )
        )
        self.assertTrue(
            np.allclose(
                built["unit_alignment_coverage_fraction_tgt"],
                np.asarray([1.0, 0.0, 0.75], dtype=np.float32),
            )
        )
        self.assertTrue(
            np.allclose(
                built["unit_alignment_expected_frame_support_tgt"],
                np.asarray([2.0, 1.0, 4.0], dtype=np.float32),
            )
        )
        self.assertTrue(
            np.allclose(
                built["unit_alignment_confidence_cost_term_tgt"],
                np.asarray([0.9, 0.0, 0.7], dtype=np.float32),
            )
        )
        self.assertTrue(
            np.allclose(
                built["unit_alignment_confidence_margin_term_tgt"],
                np.asarray([0.8, 0.0, 0.6], dtype=np.float32),
            )
        )
        self.assertTrue(
            np.allclose(
                built["unit_alignment_confidence_type_term_tgt"],
                np.asarray([1.0, 0.0, 0.9], dtype=np.float32),
            )
        )
        self.assertTrue(
            np.allclose(
                built["unit_alignment_confidence_match_term_tgt"],
                np.asarray([1.0, 0.0, 0.8], dtype=np.float32),
            )
        )
        self.assertAlmostEqual(float(np.asarray(built["unit_alignment_min_dp_weight_tgt"]).reshape(-1)[0]), 0.2)
        self.assertTrue(
            np.allclose(
                built["unit_alignment_run_skip_flag_tgt"],
                np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
            )
        )
        self.assertAlmostEqual(
            float(np.asarray(built["unit_alignment_projection_hard_bad_tgt"]).reshape(-1)[0]),
            0.0,
        )
        self.assertAlmostEqual(
            float(np.asarray(built["unit_alignment_projection_hard_bad_source_tgt"]).reshape(-1)[0]),
            0.0,
        )
        self.assertAlmostEqual(
            float(np.asarray(built["unit_alignment_projection_fallback_ratio_tgt"]).reshape(-1)[0]),
            0.1,
        )
        self.assertAlmostEqual(
            float(np.asarray(built["unit_alignment_global_path_cost_tgt"]).reshape(-1)[0]),
            1.5,
        )
        self.assertAlmostEqual(
            float(np.asarray(built["unit_alignment_global_path_mean_cost_tgt"]).reshape(-1)[0]),
            0.5,
        )
        self.assertEqual(
            str(np.asarray(built["unit_alignment_projection_health_bucket_tgt"], dtype=object).reshape(-1)[0]),
            "degraded",
        )
        self.assertEqual(
            str(np.asarray(built["unit_alignment_projection_gate_reason_tgt"], dtype=object).reshape(-1)[0]),
            "",
        )
        self.assertEqual(
            np.asarray(built["unit_alignment_source_valid_run_index_tgt"], dtype=object).tolist(),
            [0, 1, 2],
        )
        self.assertEqual(
            np.asarray(built["unit_alignment_target_valid_run_index_tgt"], dtype=object).tolist(),
            [0, 1, 2],
        )
        self.assertEqual(
            np.asarray(built["unit_alignment_target_valid_observation_index_tgt"], dtype=object).tolist(),
            [0, 2, 3],
        )
        self.assertEqual(
            np.asarray(
                built["unit_alignment_target_valid_observation_index_pre_dp_filter_tgt"],
                dtype=object,
            ).tolist(),
            [0, 1, 2, 3],
        )
        self.assertEqual(
            np.asarray(built["unit_alignment_target_dp_weight_pruned_index_tgt"], dtype=object).tolist(),
            [1],
        )

    def test_duration_v3_target_build_rejects_blank_live_continuous_provenance_for_minimal_profile(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_minimal_v1_profile": True,
                "rhythm_v3_use_continuous_alignment": True,
                "rhythm_v3_alignment_mode": "continuous_viterbi_v1",
            }
        )
        projection = {
            "projected": np.asarray([2.0, 1.0], dtype=np.float32),
            "confidence_local": np.asarray([0.9, 0.8], dtype=np.float32),
            "confidence_coarse": np.asarray([0.9, 0.8], dtype=np.float32),
            "coverage": np.asarray([1.0, 1.0], dtype=np.float32),
            "match_rate": np.asarray([1.0, 1.0], dtype=np.float32),
            "mean_cost": np.asarray([0.1, 0.2], dtype=np.float32),
            "alignment_kind": "continuous_viterbi_v1",
            "alignment_source": "",
            "alignment_version": "",
            "unmatched_speech_ratio": 0.0,
            "mean_local_confidence_speech": 0.9,
            "mean_coarse_confidence_speech": 0.9,
        }
        with mock.patch(
            "tasks.Conan.rhythm.duration_v3.dataset_mixin._project_target_runs_onto_source",
            return_value=projection,
        ):
            with self.assertRaisesRegex(RuntimeError, "non-empty unit_alignment_source_tgt"):
                dataset._build_paired_duration_v3_targets(
                    item={"item_name": "src_missing_live_provenance"},
                    source_cache={
                        "content_units": np.asarray([1, 2], dtype=np.int64),
                        "dur_anchor_src": np.asarray([2.0, 1.0], dtype=np.float32),
                        "source_silence_mask": np.asarray([0.0, 0.0], dtype=np.float32),
                    },
                    paired_target_conditioning={
                        "paired_target_content_units": np.asarray([1, 2], dtype=np.int64),
                        "paired_target_duration_obs": np.asarray([2.0, 1.0], dtype=np.float32),
                        "paired_target_valid_mask": np.asarray([1.0, 1.0], dtype=np.float32),
                        "paired_target_speech_mask": np.asarray([1.0, 1.0], dtype=np.float32),
                    },
                )

    def test_duration_v3_target_merge_rejects_blank_cached_minimal_continuous_provenance(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_minimal_v1_profile": True,
                "rhythm_v3_use_continuous_alignment": True,
            }
        )
        source_cache = {
            "rhythm_v3_cache_meta": {
                "cache_version": 4,
                "silent_token": 57,
                "separator_aware": True,
                "tail_open_units": 1,
                "emit_silence_runs": True,
                "debounce_min_run_frames": 1,
                "phrase_boundary_threshold": 0.5,
            }
        }
        source_sig = duration_v3_cache_meta_signature(source_cache)
        with self.assertRaisesRegex(RuntimeError, "non-empty unit_alignment_source_tgt"):
            dataset._merge_duration_v3_rhythm_targets(
                item={"item_name": "src_cached_blank_provenance"},
                source_cache=source_cache,
                paired_target_conditioning={
                    "source_cache_meta_signature": np.asarray([source_sig], dtype=object),
                    "paired_target_cache_meta_signature": np.asarray(["target-sig"], dtype=object),
                    "paired_target_alignment_sidecar_signature": np.asarray(["sidecar-sig"], dtype=object),
                },
                sample={
                    "unit_duration_tgt": np.asarray([2.0, 3.0], dtype=np.float32),
                    "unit_duration_proj_raw_tgt": np.asarray([2.0, 3.0], dtype=np.float32),
                    "unit_alignment_mode_id_tgt": np.asarray([2], dtype=np.int64),
                    "unit_alignment_kind_tgt": np.asarray(["continuous_viterbi_v1"], dtype=object),
                    "unit_alignment_source_tgt": np.asarray(["   "], dtype=object),
                    "unit_alignment_version_tgt": np.asarray(["2026-04-13"], dtype=object),
                    "unit_alignment_unmatched_speech_ratio_tgt": np.asarray([0.0], dtype=np.float32),
                    "unit_alignment_mean_local_confidence_speech_tgt": np.asarray([0.9], dtype=np.float32),
                    "unit_alignment_mean_coarse_confidence_speech_tgt": np.asarray([0.9], dtype=np.float32),
                    "unit_alignment_source_cache_signature_tgt": np.asarray([source_sig], dtype=object),
                    "unit_alignment_target_cache_signature_tgt": np.asarray(["target-sig"], dtype=object),
                    "unit_alignment_sidecar_signature_tgt": np.asarray(["sidecar-sig"], dtype=object),
                },
            )

    def test_duration_v3_target_merge_rejects_unparseable_cached_minimal_continuous_quality(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_minimal_v1_profile": True,
                "rhythm_v3_use_continuous_alignment": True,
            }
        )
        source_cache = {
            "rhythm_v3_cache_meta": {
                "cache_version": 4,
                "silent_token": 57,
                "separator_aware": True,
                "tail_open_units": 1,
                "emit_silence_runs": True,
                "debounce_min_run_frames": 1,
                "phrase_boundary_threshold": 0.5,
            }
        }
        source_sig = duration_v3_cache_meta_signature(source_cache)
        with self.assertRaisesRegex(RuntimeError, "parseable unit_alignment_mean_local_confidence_speech_tgt"):
            dataset._merge_duration_v3_rhythm_targets(
                item={"item_name": "src_cached_bad_quality"},
                source_cache=source_cache,
                paired_target_conditioning={
                    "source_cache_meta_signature": np.asarray([source_sig], dtype=object),
                    "paired_target_cache_meta_signature": np.asarray(["target-sig"], dtype=object),
                    "paired_target_alignment_sidecar_signature": np.asarray(["sidecar-sig"], dtype=object),
                },
                sample={
                    "unit_duration_tgt": np.asarray([2.0, 3.0], dtype=np.float32),
                    "unit_duration_proj_raw_tgt": np.asarray([2.0, 3.0], dtype=np.float32),
                    "unit_alignment_mode_id_tgt": np.asarray([2], dtype=np.int64),
                    "unit_alignment_kind_tgt": np.asarray(["continuous_viterbi_v1"], dtype=object),
                    "unit_alignment_source_tgt": np.asarray(["run_state_viterbi"], dtype=object),
                    "unit_alignment_version_tgt": np.asarray(["2026-04-13"], dtype=object),
                    "unit_alignment_unmatched_speech_ratio_tgt": np.asarray([0.0], dtype=np.float32),
                    "unit_alignment_mean_local_confidence_speech_tgt": np.asarray(["bad"], dtype=object),
                    "unit_alignment_mean_coarse_confidence_speech_tgt": np.asarray([0.9], dtype=np.float32),
                    "unit_alignment_source_cache_signature_tgt": np.asarray([source_sig], dtype=object),
                    "unit_alignment_target_cache_signature_tgt": np.asarray(["target-sig"], dtype=object),
                    "unit_alignment_sidecar_signature_tgt": np.asarray(["sidecar-sig"], dtype=object),
                },
            )

    def test_alignment_sidecar_signature_changes_when_sidecar_contents_change(self) -> None:
        dataset = _DummyDataset({"rhythm_enable_v3": True})
        sig_a = dataset._build_alignment_sidecar_signature(
            source_cache_meta_signature="src-sig",
            target_cache_meta_signature="tgt-sig",
            source_frame_states=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            source_frame_to_run=np.asarray([0, 1], dtype=np.int64),
            target_frame_states=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            target_frame_speech_prob=np.asarray([1.0, 1.0], dtype=np.float32),
            target_frame_valid=np.asarray([1.0, 1.0], dtype=np.float32),
        )
        sig_b = dataset._build_alignment_sidecar_signature(
            source_cache_meta_signature="src-sig",
            target_cache_meta_signature="tgt-sig",
            source_frame_states=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
            source_frame_to_run=np.asarray([0, 1], dtype=np.int64),
            target_frame_states=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            target_frame_speech_prob=np.asarray([1.0, 1.0], dtype=np.float32),
            target_frame_valid=np.asarray([1.0, 1.0], dtype=np.float32),
        )
        self.assertNotEqual(sig_a, sig_b)

    def test_duration_v3_target_merge_rejects_alignment_metadata_without_continuous_provenance(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_emit_silence_runs": True,
                "rhythm_v3_use_continuous_alignment": True,
            }
        )
        with self.assertRaisesRegex(RuntimeError, "requires explicit source_frame_states/source_frame_to_run/target_frame_states sidecars"):
            dataset._merge_duration_v3_rhythm_targets(
                item={"item_name": "src_align_missing_kind"},
                source_cache={
                    "content_units": np.asarray([1, 2], dtype=np.int64),
                    "dur_anchor_src": np.asarray([2.0, 2.0], dtype=np.float32),
                    "source_silence_mask": np.asarray([0.0, 0.0], dtype=np.float32),
                },
                paired_target_conditioning={
                    "paired_target_content_units": np.asarray([1, 99, 2], dtype=np.int64),
                    "paired_target_duration_obs": np.asarray([2.0, 1.0, 3.0], dtype=np.float32),
                    "paired_target_valid_mask": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
                    "paired_target_speech_mask": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
                    "paired_target_alignment_kind": "continuous_precomputed",
                    "paired_target_alignment_assigned_source": np.asarray([0, -1, 1], dtype=np.int64),
                    "paired_target_alignment_assigned_cost": np.asarray([0.0, 0.4, 0.1], dtype=np.float32),
                },
                sample={},
            )

    def test_duration_v3_target_merge_rejects_continuous_alignment_without_precomputed_metadata(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_emit_silence_runs": True,
                "rhythm_v3_use_continuous_alignment": True,
            }
        )
        with self.assertRaisesRegex(RuntimeError, "requires explicit source_frame_states/source_frame_to_run/target_frame_states sidecars"):
            dataset._merge_duration_v3_rhythm_targets(
                item={"item_name": "src_missing_align"},
                source_cache={
                    "content_units": np.asarray([1, 2], dtype=np.int64),
                    "dur_anchor_src": np.asarray([2.0, 2.0], dtype=np.float32),
                    "source_silence_mask": np.asarray([0.0, 0.0], dtype=np.float32),
                },
                paired_target_conditioning={
                    "paired_target_content_units": np.asarray([1, 99, 2], dtype=np.int64),
                    "paired_target_duration_obs": np.asarray([2.0, 1.0, 3.0], dtype=np.float32),
                    "paired_target_valid_mask": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
                    "paired_target_speech_mask": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
                },
                sample={},
            )

    def test_duration_v3_optional_sample_keys_include_alignment_provenance_sidecars(self) -> None:
        dataset = _DummyDataset({"rhythm_enable_v3": True})
        keys = set(dataset._resolve_optional_sample_keys())
        collate_spec = dataset._build_optional_collate_spec()
        self.assertIn("unit_alignment_kind_tgt", keys)
        self.assertIn("unit_alignment_source_tgt", keys)
        self.assertIn("unit_alignment_version_tgt", keys)
        self.assertIn("unit_alignment_band_ratio_tgt", keys)
        self.assertIn("unit_alignment_lambda_emb_tgt", keys)
        self.assertIn("unit_alignment_lambda_type_tgt", keys)
        self.assertIn("unit_alignment_min_dp_weight_tgt", keys)
        self.assertIn("unit_alignment_allow_source_skip_tgt", keys)
        self.assertIn("unit_alignment_skip_penalty_tgt", keys)
        self.assertIn("unit_alignment_source_cache_signature_tgt", keys)
        self.assertIn("unit_alignment_target_cache_signature_tgt", keys)
        self.assertIn("unit_alignment_sidecar_signature_tgt", keys)
        self.assertIn("unit_alignment_source_valid_run_index_tgt", keys)
        self.assertIn("unit_alignment_target_valid_run_index_tgt", keys)
        self.assertIn("unit_alignment_target_valid_observation_index_tgt", keys)
        self.assertIn("unit_alignment_target_valid_observation_index_pre_dp_filter_tgt", keys)
        self.assertIn("unit_alignment_target_dp_weight_pruned_index_tgt", keys)
        self.assertIn("unit_alignment_run_skip_flag_tgt", keys)
        self.assertIn("unit_alignment_unmatched_target_speech_ratio_tgt", keys)
        self.assertIn("unit_alignment_skipped_source_speech_ratio_tgt", keys)
        self.assertIn("alignment_source", keys)
        self.assertIn("alignment_version", keys)
        self.assertEqual(collate_spec["unit_alignment_kind_tgt"][0], "object")
        self.assertEqual(collate_spec["unit_alignment_source_tgt"][0], "object")
        self.assertEqual(collate_spec["unit_alignment_version_tgt"][0], "object")
        self.assertEqual(collate_spec["unit_alignment_band_ratio_tgt"][0], "float")
        self.assertEqual(collate_spec["unit_alignment_lambda_emb_tgt"][0], "float")
        self.assertEqual(collate_spec["unit_alignment_lambda_type_tgt"][0], "float")
        self.assertEqual(collate_spec["unit_alignment_min_dp_weight_tgt"][0], "float")
        self.assertEqual(collate_spec["unit_alignment_allow_source_skip_tgt"][0], "float")
        self.assertEqual(collate_spec["unit_alignment_skip_penalty_tgt"][0], "float")
        self.assertEqual(collate_spec["unit_alignment_source_cache_signature_tgt"][0], "object")
        self.assertEqual(collate_spec["unit_alignment_target_cache_signature_tgt"][0], "object")
        self.assertEqual(collate_spec["unit_alignment_sidecar_signature_tgt"][0], "object")
        self.assertEqual(collate_spec["unit_alignment_source_valid_run_index_tgt"][0], "object")
        self.assertEqual(collate_spec["unit_alignment_target_valid_run_index_tgt"][0], "object")
        self.assertEqual(collate_spec["unit_alignment_target_valid_observation_index_tgt"][0], "object")
        self.assertEqual(collate_spec["unit_alignment_target_valid_observation_index_pre_dp_filter_tgt"][0], "object")
        self.assertEqual(collate_spec["unit_alignment_target_dp_weight_pruned_index_tgt"][0], "object")
        self.assertEqual(collate_spec["unit_alignment_run_skip_flag_tgt"][0], "float")
        self.assertEqual(collate_spec["unit_alignment_unmatched_target_speech_ratio_tgt"][0], "float")
        self.assertEqual(collate_spec["unit_alignment_skipped_source_speech_ratio_tgt"][0], "float")
        self.assertEqual(collate_spec["alignment_source"][0], "object")
        self.assertEqual(collate_spec["alignment_version"][0], "object")

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
            "unit_duration_proj_raw_tgt": np.asarray([2.0, 3.0], dtype=np.float32),
            "unit_alignment_mode_id_tgt": np.asarray([0], dtype=np.int64),
            "unit_alignment_kind_tgt": np.asarray(["discrete"], dtype=object),
            "unit_confidence_local_tgt": np.asarray([0.0, 0.8], dtype=np.float32),
            "unit_confidence_coarse_tgt": np.asarray([0.6, 0.9], dtype=np.float32),
            "unit_alignment_coverage_tgt": np.asarray([0.0, 1.0], dtype=np.float32),
            "unit_alignment_match_tgt": np.asarray([0.0, 1.0], dtype=np.float32),
            "unit_alignment_cost_tgt": np.asarray([1.0, 0.0], dtype=np.float32),
            "unit_alignment_unmatched_speech_ratio_tgt": np.asarray([0.5], dtype=np.float32),
            "unit_alignment_unmatched_target_speech_ratio_tgt": np.asarray([0.25], dtype=np.float32),
            "unit_alignment_skipped_source_speech_ratio_tgt": np.asarray([0.5], dtype=np.float32),
            "unit_alignment_mean_local_confidence_speech_tgt": np.asarray([0.4], dtype=np.float32),
            "unit_alignment_mean_coarse_confidence_speech_tgt": np.asarray([0.7], dtype=np.float32),
            "unit_alignment_projection_hard_bad_tgt": np.asarray([1.0], dtype=np.float32),
            "unit_alignment_projection_hard_bad_source_tgt": np.asarray([1.0], dtype=np.float32),
            "unit_alignment_projection_violation_count_tgt": np.asarray([1.0], dtype=np.float32),
            "unit_alignment_projection_fallback_ratio_tgt": np.asarray([0.5], dtype=np.float32),
            "unit_alignment_global_path_cost_tgt": np.asarray([3.0], dtype=np.float32),
            "unit_alignment_global_path_mean_cost_tgt": np.asarray([1.5], dtype=np.float32),
            "unit_alignment_projection_gate_reason_tgt": np.asarray(
                ["unmatched_speech_ratio=0.500 > 0.150"],
                dtype=object,
            ),
            "unit_alignment_projection_health_bucket_tgt": np.asarray(["hard_bad"], dtype=object),
        }
        merged = dataset._merge_duration_v3_rhythm_targets(
            item={"item_name": "src_passthrough"},
            source_cache={},
            paired_target_conditioning={},
            sample=sample,
        )
        self.assertTrue(np.allclose(merged["unit_duration_tgt"], sample["unit_duration_tgt"]))
        self.assertTrue(np.allclose(merged["unit_duration_proj_raw_tgt"], sample["unit_duration_proj_raw_tgt"]))
        self.assertEqual(int(np.asarray(merged["unit_alignment_mode_id_tgt"]).reshape(-1)[0]), 0)
        self.assertEqual(str(np.asarray(merged["unit_alignment_kind_tgt"], dtype=object).reshape(-1)[0]), "discrete")
        self.assertTrue(np.allclose(merged["unit_confidence_local_tgt"], sample["unit_confidence_local_tgt"]))
        self.assertTrue(np.allclose(merged["unit_confidence_coarse_tgt"], sample["unit_confidence_coarse_tgt"]))
        self.assertTrue(np.allclose(merged["unit_confidence_tgt"], sample["unit_confidence_coarse_tgt"]))
        self.assertTrue(np.allclose(merged["unit_alignment_coverage_tgt"], sample["unit_alignment_coverage_tgt"]))
        self.assertTrue(np.allclose(merged["unit_alignment_match_tgt"], sample["unit_alignment_match_tgt"]))
        self.assertTrue(np.allclose(merged["unit_alignment_cost_tgt"], sample["unit_alignment_cost_tgt"]))
        self.assertTrue(
            np.allclose(
                merged["unit_alignment_unmatched_speech_ratio_tgt"],
                sample["unit_alignment_unmatched_speech_ratio_tgt"],
            )
        )
        self.assertTrue(
            np.allclose(
                merged["unit_alignment_mean_local_confidence_speech_tgt"],
                sample["unit_alignment_mean_local_confidence_speech_tgt"],
            )
        )
        self.assertTrue(
            np.allclose(
                merged["unit_alignment_mean_coarse_confidence_speech_tgt"],
                sample["unit_alignment_mean_coarse_confidence_speech_tgt"],
            )
        )
        self.assertTrue(
            np.allclose(
                merged["unit_alignment_unmatched_target_speech_ratio_tgt"],
                sample["unit_alignment_unmatched_target_speech_ratio_tgt"],
            )
        )
        self.assertTrue(
            np.allclose(
                merged["unit_alignment_skipped_source_speech_ratio_tgt"],
                sample["unit_alignment_skipped_source_speech_ratio_tgt"],
            )
        )
        self.assertTrue(
            np.allclose(
                merged["unit_alignment_projection_hard_bad_tgt"],
                sample["unit_alignment_projection_hard_bad_tgt"],
            )
        )
        self.assertTrue(
            np.allclose(
                merged["unit_alignment_projection_hard_bad_source_tgt"],
                sample["unit_alignment_projection_hard_bad_source_tgt"],
            )
        )
        self.assertTrue(
            np.allclose(
                merged["unit_alignment_projection_fallback_ratio_tgt"],
                sample["unit_alignment_projection_fallback_ratio_tgt"],
            )
        )
        self.assertTrue(
            np.allclose(
                merged["unit_alignment_global_path_cost_tgt"],
                sample["unit_alignment_global_path_cost_tgt"],
            )
        )
        self.assertEqual(
            str(np.asarray(merged["unit_alignment_projection_gate_reason_tgt"], dtype=object).reshape(-1)[0]),
            "unmatched_speech_ratio=0.500 > 0.150",
        )
        self.assertEqual(
            str(np.asarray(merged["unit_alignment_projection_health_bucket_tgt"], dtype=object).reshape(-1)[0]),
            "hard_bad",
        )

    def test_duration_v3_target_merge_strict_gate_rejects_low_quality_cached_alignment(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_gate_quality_strict": True,
                "rhythm_v3_alignment_unmatched_speech_ratio_max": 0.15,
                "rhythm_v3_alignment_mean_local_confidence_speech_min": 0.55,
                "rhythm_v3_alignment_mean_coarse_confidence_speech_min": 0.60,
            }
        )
        sample = {
            "unit_duration_tgt": np.asarray([2.0, 3.0], dtype=np.float32),
            "unit_duration_proj_raw_tgt": np.asarray([2.0, 3.0], dtype=np.float32),
            "unit_alignment_mode_id_tgt": np.asarray([2], dtype=np.int64),
            "unit_alignment_kind_tgt": np.asarray(["continuous_viterbi_v1"], dtype=object),
            "unit_confidence_local_tgt": np.asarray([0.4, 0.4], dtype=np.float32),
            "unit_confidence_coarse_tgt": np.asarray([0.5, 0.5], dtype=np.float32),
            "unit_alignment_coverage_tgt": np.asarray([1.0, 1.0], dtype=np.float32),
            "unit_alignment_match_tgt": np.asarray([1.0, 1.0], dtype=np.float32),
            "unit_alignment_cost_tgt": np.asarray([0.4, 0.5], dtype=np.float32),
            "unit_alignment_unmatched_speech_ratio_tgt": np.asarray([0.2], dtype=np.float32),
            "unit_alignment_mean_local_confidence_speech_tgt": np.asarray([0.4], dtype=np.float32),
            "unit_alignment_mean_coarse_confidence_speech_tgt": np.asarray([0.5], dtype=np.float32),
        }
        with self.assertRaisesRegex(RuntimeError, "cached unit_duration_tgt:src_cached_strict"):
            dataset._merge_duration_v3_rhythm_targets(
                item={"item_name": "src_cached_strict"},
                source_cache={},
                paired_target_conditioning={},
                sample=sample,
            )

    def test_duration_v3_target_merge_rejects_cached_alignment_with_projection_fallback_ratio(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_minimal_v1_profile": True,
                "rhythm_v3_use_continuous_alignment": True,
            }
        )
        sample = {
            "unit_duration_tgt": np.asarray([2.0], dtype=np.float32),
            "unit_duration_proj_raw_tgt": np.asarray([2.0], dtype=np.float32),
            "unit_alignment_mode_id_tgt": np.asarray([2], dtype=np.int64),
            "unit_alignment_kind_tgt": np.asarray(["continuous_viterbi_v1"], dtype=object),
            "unit_alignment_source_tgt": np.asarray(["cached"], dtype=object),
            "unit_alignment_version_tgt": np.asarray(["2026-04-13"], dtype=object),
            "unit_alignment_source_cache_signature_tgt": np.asarray(["src_sign"], dtype=object),
            "unit_alignment_target_cache_signature_tgt": np.asarray(["tgt_sign"], dtype=object),
            "unit_alignment_sidecar_signature_tgt": np.asarray(["sidecar_sign"], dtype=object),
            "unit_alignment_unmatched_speech_ratio_tgt": np.asarray([0.1], dtype=np.float32),
            "unit_alignment_mean_local_confidence_speech_tgt": np.asarray([0.7], dtype=np.float32),
            "unit_alignment_mean_coarse_confidence_speech_tgt": np.asarray([0.8], dtype=np.float32),
            "unit_alignment_projection_fallback_ratio_tgt": np.asarray([0.25], dtype=np.float32),
            "unit_alignment_projection_health_bucket_tgt": np.asarray(["clean"], dtype=object),
            "unit_alignment_projection_gate_reason_tgt": np.asarray(["fallback_ratio"], dtype=object),
        }
        with self.assertRaisesRegex(RuntimeError, "projection_fallback_ratio_tgt > 0"):
            dataset._merge_duration_v3_rhythm_targets(
                item={"item_name": "cached_fallback_ratio"},
                source_cache={},
                paired_target_conditioning={},
                sample=sample,
            )

    def test_duration_v3_target_merge_rejects_cached_alignment_with_degraded_health_bucket(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_minimal_v1_profile": True,
                "rhythm_v3_use_continuous_alignment": True,
            }
        )
        sample = {
            "unit_duration_tgt": np.asarray([2.0], dtype=np.float32),
            "unit_duration_proj_raw_tgt": np.asarray([2.0], dtype=np.float32),
            "unit_alignment_mode_id_tgt": np.asarray([2], dtype=np.int64),
            "unit_alignment_kind_tgt": np.asarray(["continuous_viterbi_v1"], dtype=object),
            "unit_alignment_source_tgt": np.asarray(["cached"], dtype=object),
            "unit_alignment_version_tgt": np.asarray(["2026-04-14"], dtype=object),
            "unit_alignment_source_cache_signature_tgt": np.asarray(["src_sign"], dtype=object),
            "unit_alignment_target_cache_signature_tgt": np.asarray(["tgt_sign"], dtype=object),
            "unit_alignment_sidecar_signature_tgt": np.asarray(["sidecar_sign"], dtype=object),
            "unit_alignment_unmatched_speech_ratio_tgt": np.asarray([0.1], dtype=np.float32),
            "unit_alignment_mean_local_confidence_speech_tgt": np.asarray([0.7], dtype=np.float32),
            "unit_alignment_mean_coarse_confidence_speech_tgt": np.asarray([0.8], dtype=np.float32),
            "unit_alignment_projection_fallback_ratio_tgt": np.asarray([0.0], dtype=np.float32),
            "unit_alignment_projection_health_bucket_tgt": np.asarray(["degraded"], dtype=object),
        }
        with self.assertRaisesRegex(RuntimeError, "requires clean health bucket"):
            dataset._merge_duration_v3_rhythm_targets(
                item={"item_name": "cached_degraded_bucket"},
                source_cache={},
                paired_target_conditioning={},
                sample=sample,
            )

    def test_duration_v3_target_merge_rejects_cached_alignment_with_fallback_reason(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_minimal_v1_profile": True,
                "rhythm_v3_use_continuous_alignment": True,
            }
        )
        sample = {
            "unit_duration_tgt": np.asarray([2.0], dtype=np.float32),
            "unit_duration_proj_raw_tgt": np.asarray([2.0], dtype=np.float32),
            "unit_alignment_mode_id_tgt": np.asarray([2], dtype=np.int64),
            "unit_alignment_kind_tgt": np.asarray(["continuous_viterbi_v1"], dtype=object),
            "unit_alignment_source_tgt": np.asarray(["cached"], dtype=object),
            "unit_alignment_version_tgt": np.asarray(["2026-04-14"], dtype=object),
            "unit_alignment_source_cache_signature_tgt": np.asarray(["src_sign"], dtype=object),
            "unit_alignment_target_cache_signature_tgt": np.asarray(["tgt_sign"], dtype=object),
            "unit_alignment_sidecar_signature_tgt": np.asarray(["sidecar_sign"], dtype=object),
            "unit_alignment_unmatched_speech_ratio_tgt": np.asarray([0.1], dtype=np.float32),
            "unit_alignment_mean_local_confidence_speech_tgt": np.asarray([0.7], dtype=np.float32),
            "unit_alignment_mean_coarse_confidence_speech_tgt": np.asarray([0.8], dtype=np.float32),
            "unit_alignment_projection_fallback_ratio_tgt": np.asarray([0.0], dtype=np.float32),
            "unit_alignment_projection_health_bucket_tgt": np.asarray(["clean"], dtype=object),
            "unit_alignment_fallback_reason_tgt": np.asarray(["projection_retry"], dtype=object),
        }
        with self.assertRaisesRegex(RuntimeError, "forbids non-empty fallback_reason"):
            dataset._merge_duration_v3_rhythm_targets(
                item={"item_name": "cached_fallback_reason"},
                source_cache={},
                paired_target_conditioning={},
                sample=sample,
            )

    def test_duration_v3_projection_conditioning_copies_item_level_alignment_provenance_aliases(self) -> None:
        dataset = _DummyDataset({"rhythm_enable_v3": True})
        conditioning = dataset._build_paired_target_projection_conditioning(
            {
                "item_name": "paired_item",
                "content_units": np.asarray([1, 2], dtype=np.int64),
                "dur_anchor_src": np.asarray([2.0, 3.0], dtype=np.float32),
                "source_silence_mask": np.asarray([0.0, 0.0], dtype=np.float32),
                "unit_alignment_kind_tgt": np.asarray(["continuous_viterbi_v1"], dtype=object),
                "unit_alignment_mode_id_tgt": np.asarray([2], dtype=np.int64),
                "unit_alignment_source_tgt": np.asarray(["run_state_viterbi"], dtype=object),
                "unit_alignment_version_tgt": np.asarray(["2026-04-13"], dtype=object),
                "unit_alignment_band_ratio_tgt": np.asarray([0.08], dtype=np.float32),
                "unit_alignment_lambda_emb_tgt": np.asarray([1.0], dtype=np.float32),
                "unit_alignment_lambda_type_tgt": np.asarray([0.5], dtype=np.float32),
                "unit_alignment_assigned_source_debug": np.asarray([0, 1], dtype=np.int64),
                "unit_alignment_assigned_cost_debug": np.asarray([0.0, 0.1], dtype=np.float32),
            },
            target_mode="cached_only",
            source_item=None,
        )
        self.assertEqual(conditioning["paired_target_alignment_kind"], "continuous_viterbi_v1")
        self.assertEqual(int(np.asarray(conditioning["paired_target_alignment_mode_id"]).reshape(-1)[0]), 2)
        self.assertEqual(conditioning["paired_target_alignment_source"], "run_state_viterbi")
        self.assertEqual(conditioning["paired_target_alignment_version"], "2026-04-13")
        self.assertAlmostEqual(float(np.asarray(conditioning["paired_target_alignment_band_ratio"]).reshape(-1)[0]), 0.08)
        self.assertAlmostEqual(float(np.asarray(conditioning["paired_target_alignment_lambda_emb"]).reshape(-1)[0]), 1.0)
        self.assertAlmostEqual(float(np.asarray(conditioning["paired_target_alignment_lambda_type"]).reshape(-1)[0]), 0.5)
        self.assertTrue(
            np.array_equal(
                np.asarray(conditioning["paired_target_alignment_assigned_source"]),
                np.asarray([0, 1], dtype=np.int64),
            )
        )
        self.assertTrue(
            np.allclose(
                np.asarray(conditioning["paired_target_alignment_assigned_cost"], dtype=np.float32),
                np.asarray([0.0, 0.1], dtype=np.float32),
            )
        )

    def test_duration_v3_target_merge_requires_cached_provenance_for_minimal_continuous_passthrough(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_minimal_v1_profile": True,
                "rhythm_v3_use_continuous_alignment": True,
            }
        )
        with self.assertRaisesRegex(RuntimeError, "unit_duration_proj_raw_tgt"):
            dataset._merge_duration_v3_rhythm_targets(
                item={"item_name": "src_cached_missing_raw"},
                source_cache={},
                paired_target_conditioning={},
                sample={
                    "unit_duration_tgt": np.asarray([2.0, 3.0], dtype=np.float32),
                    "unit_alignment_mode_id_tgt": np.asarray([1], dtype=np.int64),
                },
            )
        with self.assertRaisesRegex(RuntimeError, "unit_alignment_mode_id_tgt"):
            dataset._merge_duration_v3_rhythm_targets(
                item={"item_name": "src_cached_missing_mode"},
                source_cache={},
                paired_target_conditioning={},
                sample={
                    "unit_duration_tgt": np.asarray([2.0, 3.0], dtype=np.float32),
                    "unit_duration_proj_raw_tgt": np.asarray([2.0, 3.0], dtype=np.float32),
                    "unit_alignment_kind_tgt": np.asarray(["continuous_precomputed"], dtype=object),
                    "unit_alignment_source_tgt": np.asarray(["hubert_state_dp_v1"], dtype=object),
                    "unit_alignment_version_tgt": np.asarray(["2026-04-12"], dtype=object),
                },
            )

    def test_duration_v3_target_merge_requires_cached_source_and_version_for_minimal_continuous_passthrough(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_minimal_v1_profile": True,
                "rhythm_v3_use_continuous_alignment": True,
            }
        )
        with self.assertRaisesRegex(RuntimeError, "unit_alignment_source_tgt|unit_alignment_version_tgt"):
            dataset._merge_duration_v3_rhythm_targets(
                item={"item_name": "src_cached_missing_provenance"},
                source_cache={},
                paired_target_conditioning={},
                sample={
                    "unit_duration_tgt": np.asarray([2.0, 3.0], dtype=np.float32),
                    "unit_duration_proj_raw_tgt": np.asarray([2.0, 3.0], dtype=np.float32),
                    "unit_alignment_mode_id_tgt": np.asarray([2], dtype=np.int64),
                    "unit_alignment_kind_tgt": np.asarray(["continuous_viterbi_v1"], dtype=object),
                },
            )

    def test_duration_v3_target_merge_preserves_explicit_cached_alignment_kind(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_minimal_v1_profile": True,
                "rhythm_v3_use_continuous_alignment": True,
            }
        )
        merged = dataset._merge_duration_v3_rhythm_targets(
            item={"item_name": "src_cached_kind"},
            source_cache={},
            paired_target_conditioning={},
            sample={
                "unit_duration_tgt": np.asarray([2.0, 3.0], dtype=np.float32),
                "unit_duration_proj_raw_tgt": np.asarray([2.0, 3.0], dtype=np.float32),
                "unit_alignment_mode_id_tgt": np.asarray([1], dtype=np.int64),
                "unit_alignment_kind_tgt": np.asarray(["continuous_precomputed"], dtype=object),
                "unit_alignment_source_tgt": np.asarray(["hubert_state_dp_v1"], dtype=object),
                "unit_alignment_version_tgt": np.asarray(["2026-04-12"], dtype=object),
                "unit_alignment_unmatched_speech_ratio_tgt": np.asarray([0.0], dtype=np.float32),
                "unit_alignment_mean_local_confidence_speech_tgt": np.asarray([0.9], dtype=np.float32),
                "unit_alignment_mean_coarse_confidence_speech_tgt": np.asarray([0.9], dtype=np.float32),
                "unit_alignment_source_cache_signature_tgt": np.asarray(["source-sig"], dtype=object),
                "unit_alignment_target_cache_signature_tgt": np.asarray(["target-sig"], dtype=object),
                "unit_alignment_sidecar_signature_tgt": np.asarray(["sidecar-sig"], dtype=object),
            },
        )
        self.assertEqual(int(np.asarray(merged["unit_alignment_mode_id_tgt"]).reshape(-1)[0]), 1)
        self.assertEqual(
            str(np.asarray(merged["unit_alignment_kind_tgt"], dtype=object).reshape(-1)[0]),
            "continuous_precomputed",
        )
        self.assertEqual(merged["alignment_source"], "hubert_state_dp_v1")
        self.assertEqual(merged["alignment_version"], "2026-04-12")

    def test_duration_v3_target_merge_rejects_cached_continuous_target_without_explicit_kind_string(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_minimal_v1_profile": True,
                "rhythm_v3_use_continuous_alignment": True,
            }
        )
        with self.assertRaisesRegex(RuntimeError, "unit_alignment_kind_tgt"):
            dataset._merge_duration_v3_rhythm_targets(
                item={"item_name": "src_cached_viterbi_kind"},
                source_cache={},
                paired_target_conditioning={},
                sample={
                    "unit_duration_tgt": np.asarray([2.0, 3.0], dtype=np.float32),
                    "unit_duration_proj_raw_tgt": np.asarray([2.0, 3.0], dtype=np.float32),
                    "unit_alignment_mode_id_tgt": np.asarray([2], dtype=np.int64),
                    "unit_alignment_source_tgt": np.asarray(["run_state_viterbi"], dtype=object),
                    "unit_alignment_version_tgt": np.asarray(["2026-04-13"], dtype=object),
                    "unit_alignment_unmatched_speech_ratio_tgt": np.asarray([0.0], dtype=np.float32),
                    "unit_alignment_mean_local_confidence_speech_tgt": np.asarray([0.9], dtype=np.float32),
                    "unit_alignment_mean_coarse_confidence_speech_tgt": np.asarray([0.9], dtype=np.float32),
                    "unit_alignment_source_cache_signature_tgt": np.asarray(["source-sig"], dtype=object),
                    "unit_alignment_target_cache_signature_tgt": np.asarray(["target-sig"], dtype=object),
                    "unit_alignment_sidecar_signature_tgt": np.asarray(["sidecar-sig"], dtype=object),
                },
            )

    def test_duration_v3_target_merge_rejects_cached_minimal_continuous_target_with_unknown_kind(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_minimal_v1_profile": True,
                "rhythm_v3_use_continuous_alignment": True,
            }
        )
        with self.assertRaisesRegex(RuntimeError, "supported continuous provenance|supported continuous paired-target alignment"):
            dataset._merge_duration_v3_rhythm_targets(
                item={"item_name": "src_cached_unknown_kind"},
                source_cache={},
                paired_target_conditioning={},
                sample={
                    "unit_duration_tgt": np.asarray([2.0, 3.0], dtype=np.float32),
                    "unit_duration_proj_raw_tgt": np.asarray([2.0, 3.0], dtype=np.float32),
                    "unit_alignment_mode_id_tgt": np.asarray([0], dtype=np.int64),
                    "unit_alignment_kind_tgt": np.asarray(["continuous_experimental"], dtype=object),
                    "unit_alignment_source_tgt": np.asarray(["run_state_viterbi"], dtype=object),
                    "unit_alignment_version_tgt": np.asarray(["2026-04-13"], dtype=object),
                    "unit_alignment_unmatched_speech_ratio_tgt": np.asarray([0.0], dtype=np.float32),
                    "unit_alignment_mean_local_confidence_speech_tgt": np.asarray([0.9], dtype=np.float32),
                    "unit_alignment_mean_coarse_confidence_speech_tgt": np.asarray([0.9], dtype=np.float32),
                    "unit_alignment_source_cache_signature_tgt": np.asarray(["source-sig"], dtype=object),
                    "unit_alignment_target_cache_signature_tgt": np.asarray(["target-sig"], dtype=object),
                    "unit_alignment_sidecar_signature_tgt": np.asarray(["sidecar-sig"], dtype=object),
                },
            )

    def test_duration_v3_target_merge_rejects_cached_minimal_continuous_top_level_only_provenance(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_minimal_v1_profile": True,
                "rhythm_v3_use_continuous_alignment": True,
            }
        )
        with self.assertRaisesRegex(RuntimeError, "unit_alignment_source_tgt|unit_alignment_version_tgt"):
            dataset._merge_duration_v3_rhythm_targets(
                item={"item_name": "src_cached_top_level_only_provenance"},
                source_cache={},
                paired_target_conditioning={},
                sample={
                    "unit_duration_tgt": np.asarray([2.0, 3.0], dtype=np.float32),
                    "unit_duration_proj_raw_tgt": np.asarray([2.0, 3.0], dtype=np.float32),
                    "unit_alignment_mode_id_tgt": np.asarray([2], dtype=np.int64),
                    "unit_alignment_kind_tgt": np.asarray(["continuous_viterbi_v1"], dtype=object),
                    "alignment_source": np.asarray(["run_state_viterbi"], dtype=object),
                    "alignment_version": np.asarray(["2026-04-13"], dtype=object),
                    "unit_alignment_unmatched_speech_ratio_tgt": np.asarray([0.0], dtype=np.float32),
                    "unit_alignment_mean_local_confidence_speech_tgt": np.asarray([0.9], dtype=np.float32),
                    "unit_alignment_mean_coarse_confidence_speech_tgt": np.asarray([0.9], dtype=np.float32),
                    "unit_alignment_source_cache_signature_tgt": np.asarray(["source-sig"], dtype=object),
                    "unit_alignment_target_cache_signature_tgt": np.asarray(["target-sig"], dtype=object),
                    "unit_alignment_sidecar_signature_tgt": np.asarray(["sidecar-sig"], dtype=object),
                },
            )

    def test_duration_v3_target_merge_requires_cached_quality_and_signature_fields_for_minimal_continuous_passthrough(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_minimal_v1_profile": True,
                "rhythm_v3_use_continuous_alignment": True,
            }
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "unit_alignment_mean_local_confidence_speech_tgt|unit_alignment_source_cache_signature_tgt",
        ):
            dataset._merge_duration_v3_rhythm_targets(
                item={"item_name": "src_cached_missing_quality_signature"},
                source_cache={},
                paired_target_conditioning={},
                sample={
                    "unit_duration_tgt": np.asarray([2.0, 3.0], dtype=np.float32),
                    "unit_duration_proj_raw_tgt": np.asarray([2.0, 3.0], dtype=np.float32),
                    "unit_alignment_mode_id_tgt": np.asarray([2], dtype=np.int64),
                    "unit_alignment_kind_tgt": np.asarray(["continuous_viterbi_v1"], dtype=object),
                    "unit_alignment_source_tgt": np.asarray(["run_state_viterbi"], dtype=object),
                    "unit_alignment_version_tgt": np.asarray(["2026-04-13"], dtype=object),
                    "unit_alignment_unmatched_speech_ratio_tgt": np.asarray([0.0], dtype=np.float32),
                    "unit_alignment_mean_coarse_confidence_speech_tgt": np.asarray([0.9], dtype=np.float32),
                    "unit_alignment_target_cache_signature_tgt": np.asarray(["target-sig"], dtype=object),
                    "unit_alignment_sidecar_signature_tgt": np.asarray(["sidecar-sig"], dtype=object),
                },
            )

    def test_duration_v3_target_merge_rejects_cached_minimal_continuous_targets_below_quality_thresholds(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_minimal_v1_profile": True,
                "rhythm_v3_use_continuous_alignment": True,
            }
        )
        source_cache = {
            "rhythm_v3_cache_meta": {
                "cache_version": 4,
                "silent_token": 57,
                "separator_aware": True,
                "tail_open_units": 1,
                "emit_silence_runs": True,
                "debounce_min_run_frames": 1,
                "phrase_boundary_threshold": 0.5,
            }
        }
        source_sig = duration_v3_cache_meta_signature(source_cache)
        with self.assertRaisesRegex(RuntimeError, "mean_local_confidence_speech"):
            dataset._merge_duration_v3_rhythm_targets(
                item={"item_name": "src_cached_low_quality"},
                source_cache=source_cache,
                paired_target_conditioning={
                    "source_cache_meta_signature": np.asarray([source_sig], dtype=object),
                    "paired_target_cache_meta_signature": np.asarray(["target-sig"], dtype=object),
                    "paired_target_alignment_sidecar_signature": np.asarray(["sidecar-sig"], dtype=object),
                },
                sample={
                    "unit_duration_tgt": np.asarray([2.0, 3.0], dtype=np.float32),
                    "unit_duration_proj_raw_tgt": np.asarray([2.0, 3.0], dtype=np.float32),
                    "unit_alignment_mode_id_tgt": np.asarray([2], dtype=np.int64),
                    "unit_alignment_kind_tgt": np.asarray(["continuous_viterbi_v1"], dtype=object),
                    "unit_alignment_source_tgt": np.asarray(["run_state_viterbi"], dtype=object),
                    "unit_alignment_version_tgt": np.asarray(["2026-04-13"], dtype=object),
                    "unit_alignment_unmatched_speech_ratio_tgt": np.asarray([0.0], dtype=np.float32),
                    "unit_alignment_mean_local_confidence_speech_tgt": np.asarray([0.1], dtype=np.float32),
                    "unit_alignment_mean_coarse_confidence_speech_tgt": np.asarray([0.9], dtype=np.float32),
                    "unit_alignment_source_cache_signature_tgt": np.asarray([source_sig], dtype=object),
                    "unit_alignment_target_cache_signature_tgt": np.asarray(["target-sig"], dtype=object),
                    "unit_alignment_sidecar_signature_tgt": np.asarray(["sidecar-sig"], dtype=object),
                },
            )

    def test_duration_v3_target_merge_rejects_cached_minimal_continuous_signature_mismatch(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_minimal_v1_profile": True,
                "rhythm_v3_use_continuous_alignment": True,
            }
        )
        source_cache = {
            "rhythm_v3_cache_meta": {
                "cache_version": 4,
                "silent_token": 57,
                "separator_aware": True,
                "tail_open_units": 1,
                "emit_silence_runs": True,
                "debounce_min_run_frames": 1,
                "phrase_boundary_threshold": 0.5,
            }
        }
        source_sig = duration_v3_cache_meta_signature(source_cache)
        with self.assertRaisesRegex(RuntimeError, "target signature mismatch"):
            dataset._merge_duration_v3_rhythm_targets(
                item={"item_name": "src_cached_sig_mismatch"},
                source_cache=source_cache,
                paired_target_conditioning={
                    "source_cache_meta_signature": np.asarray([source_sig], dtype=object),
                    "paired_target_cache_meta_signature": np.asarray(["target-sig"], dtype=object),
                    "paired_target_alignment_sidecar_signature": np.asarray(["sidecar-sig"], dtype=object),
                },
                sample={
                    "unit_duration_tgt": np.asarray([2.0, 3.0], dtype=np.float32),
                    "unit_duration_proj_raw_tgt": np.asarray([2.0, 3.0], dtype=np.float32),
                    "unit_alignment_mode_id_tgt": np.asarray([2], dtype=np.int64),
                    "unit_alignment_kind_tgt": np.asarray(["continuous_viterbi_v1"], dtype=object),
                    "unit_alignment_source_tgt": np.asarray(["run_state_viterbi"], dtype=object),
                    "unit_alignment_version_tgt": np.asarray(["2026-04-13"], dtype=object),
                    "unit_alignment_unmatched_speech_ratio_tgt": np.asarray([0.0], dtype=np.float32),
                    "unit_alignment_mean_local_confidence_speech_tgt": np.asarray([0.9], dtype=np.float32),
                    "unit_alignment_mean_coarse_confidence_speech_tgt": np.asarray([0.9], dtype=np.float32),
                    "unit_alignment_source_cache_signature_tgt": np.asarray([source_sig], dtype=object),
                    "unit_alignment_target_cache_signature_tgt": np.asarray(["stale-target-sig"], dtype=object),
                    "unit_alignment_sidecar_signature_tgt": np.asarray(["sidecar-sig"], dtype=object),
                },
            )

    def test_duration_v3_target_merge_preserves_cached_minimal_continuous_signatures(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_minimal_v1_profile": True,
                "rhythm_v3_use_continuous_alignment": True,
            }
        )
        source_cache = {
            "rhythm_v3_cache_meta": {
                "cache_version": 4,
                "silent_token": 57,
                "separator_aware": True,
                "tail_open_units": 1,
                "emit_silence_runs": True,
                "debounce_min_run_frames": 1,
                "phrase_boundary_threshold": 0.5,
            }
        }
        source_sig = duration_v3_cache_meta_signature(source_cache)
        merged = dataset._merge_duration_v3_rhythm_targets(
            item={"item_name": "src_cached_sig_ok"},
            source_cache=source_cache,
            paired_target_conditioning={
                "source_cache_meta_signature": np.asarray([source_sig], dtype=object),
                "paired_target_cache_meta_signature": np.asarray(["target-sig"], dtype=object),
                "paired_target_alignment_sidecar_signature": np.asarray(["sidecar-sig"], dtype=object),
            },
            sample={
                "unit_duration_tgt": np.asarray([2.0, 3.0], dtype=np.float32),
                "unit_duration_proj_raw_tgt": np.asarray([2.0, 3.0], dtype=np.float32),
                "unit_alignment_mode_id_tgt": np.asarray([2], dtype=np.int64),
                "unit_alignment_kind_tgt": np.asarray(["continuous_viterbi_v1"], dtype=object),
                "unit_alignment_source_tgt": np.asarray(["run_state_viterbi"], dtype=object),
                "unit_alignment_version_tgt": np.asarray(["2026-04-13"], dtype=object),
                "unit_alignment_unmatched_speech_ratio_tgt": np.asarray([0.0], dtype=np.float32),
                "unit_alignment_mean_local_confidence_speech_tgt": np.asarray([0.9], dtype=np.float32),
                "unit_alignment_mean_coarse_confidence_speech_tgt": np.asarray([0.9], dtype=np.float32),
                "unit_alignment_source_cache_signature_tgt": np.asarray([source_sig], dtype=object),
                "unit_alignment_target_cache_signature_tgt": np.asarray(["target-sig"], dtype=object),
                "unit_alignment_sidecar_signature_tgt": np.asarray(["sidecar-sig"], dtype=object),
            },
        )
        self.assertEqual(
            str(np.asarray(merged["unit_alignment_source_cache_signature_tgt"], dtype=object).reshape(-1)[0]),
            source_sig,
        )
        self.assertEqual(
            str(np.asarray(merged["unit_alignment_target_cache_signature_tgt"], dtype=object).reshape(-1)[0]),
            "target-sig",
        )
        self.assertEqual(
            str(np.asarray(merged["unit_alignment_sidecar_signature_tgt"], dtype=object).reshape(-1)[0]),
            "sidecar-sig",
        )

    def test_duration_v3_target_build_accepts_rhythm_v3_align_alias_hparams(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_minimal_v1_profile": True,
                "rhythm_v3_use_continuous_alignment": True,
                "rhythm_v3_alignment_mode": "continuous_viterbi_v1",
                "rhythm_v3_align_lambda_emb": 1.7,
                "rhythm_v3_align_lambda_type": 0.9,
                "rhythm_v3_align_lambda_band": 0.4,
                "rhythm_v3_align_lambda_unit": 0.3,
                "rhythm_v3_align_band_ratio": 0.11,
                "rhythm_v3_align_bad_cost_threshold": 1.7,
                "rhythm_v3_align_min_dp_weight": 0.25,
                "rhythm_v3_align_allow_source_skip": True,
                "rhythm_v3_align_skip_penalty": 0.25,
                "rhythm_v3_alignment_mean_local_confidence_speech_min": 0.0,
                "rhythm_v3_alignment_mean_coarse_confidence_speech_min": 0.0,
                "rhythm_v3_alignment_unmatched_speech_ratio_max": 1.0,
            }
        )
        built = dataset._build_paired_duration_v3_targets(
            item={"item_name": "src_align_alias"},
            source_cache={
                "content_units": np.asarray([1, 2, 3], dtype=np.int64),
                "dur_anchor_src": np.asarray([2.0, 2.0, 2.0], dtype=np.float32),
                "source_silence_mask": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
                "rhythm_v3_cache_meta": {
                    "cache_version": 4,
                    "silent_token": 57,
                    "separator_aware": True,
                    "tail_open_units": 1,
                    "emit_silence_runs": True,
                    "debounce_min_run_frames": 2,
                    "phrase_boundary_threshold": 0.55,
                },
            },
            paired_target_conditioning={
                "paired_target_content_units": np.asarray([1, 2, 3], dtype=np.int64),
                "paired_target_duration_obs": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
                "paired_target_valid_mask": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
                "paired_target_speech_mask": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
                "source_frame_states": np.asarray(
                    [
                        [1.0, 0.0, 0.0],
                        [0.9, 0.1, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.9, 0.1],
                        [0.0, 0.0, 1.0],
                        [0.1, 0.0, 0.9],
                    ],
                    dtype=np.float32,
                ),
                "source_frame_to_run": np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64),
                "paired_target_frame_states": np.asarray(
                    [
                        [1.0, 0.0, 0.0],
                        [0.9, 0.1, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
                "paired_target_frame_speech_prob": np.ones((4,), dtype=np.float32),
                "paired_target_frame_valid": np.ones((4,), dtype=np.float32),
                "source_cache_meta_signature": np.asarray(
                    [
                        "{\"cache_version\":4,\"debounce_min_run_frames\":2,\"emit_silence_runs\":true,\"phrase_boundary_threshold\":0.55,\"separator_aware\":true,\"silent_token\":57,\"tail_open_units\":1}"
                    ],
                    dtype=object,
                ),
                "paired_target_cache_meta_signature": np.asarray(
                    [
                        "{\"cache_version\":4,\"debounce_min_run_frames\":2,\"emit_silence_runs\":true,\"phrase_boundary_threshold\":0.55,\"separator_aware\":true,\"silent_token\":57,\"tail_open_units\":1}"
                    ],
                    dtype=object,
                ),
                "paired_target_alignment_sidecar_signature": np.asarray(["precomputed-sidecar-sig"], dtype=object),
            },
        )
        self.assertAlmostEqual(float(np.asarray(built["unit_alignment_lambda_emb_tgt"]).reshape(-1)[0]), 1.7)
        self.assertAlmostEqual(float(np.asarray(built["unit_alignment_lambda_type_tgt"]).reshape(-1)[0]), 0.9)
        self.assertAlmostEqual(float(np.asarray(built["unit_alignment_lambda_band_tgt"]).reshape(-1)[0]), 0.4)
        self.assertAlmostEqual(float(np.asarray(built["unit_alignment_lambda_unit_tgt"]).reshape(-1)[0]), 0.3)
        self.assertAlmostEqual(float(np.asarray(built["unit_alignment_band_ratio_tgt"]).reshape(-1)[0]), 0.11)
        self.assertAlmostEqual(float(np.asarray(built["unit_alignment_bad_cost_threshold_tgt"]).reshape(-1)[0]), 1.7)
        self.assertAlmostEqual(float(np.asarray(built["unit_alignment_min_dp_weight_tgt"]).reshape(-1)[0]), 0.25)
        self.assertAlmostEqual(float(np.asarray(built["unit_alignment_allow_source_skip_tgt"]).reshape(-1)[0]), 1.0)
        self.assertAlmostEqual(float(np.asarray(built["unit_alignment_skip_penalty_tgt"]).reshape(-1)[0]), 0.25)
        self.assertEqual(
            str(np.asarray(built["unit_alignment_sidecar_signature_tgt"], dtype=object).reshape(-1)[0]),
            "precomputed-sidecar-sig",
        )

    def test_duration_v3_target_build_rejects_continuous_viterbi_signature_mismatch(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_use_continuous_alignment": True,
                "rhythm_v3_alignment_mode": "continuous_viterbi_v1",
            }
        )
        with self.assertRaisesRegex(RuntimeError, "frontend/cache signature mismatch"):
            dataset._build_paired_duration_v3_targets(
                item={"item_name": "src_signature_mismatch"},
                source_cache={
                    "content_units": np.asarray([1, 2, 3], dtype=np.int64),
                    "dur_anchor_src": np.asarray([2.0, 2.0, 2.0], dtype=np.float32),
                    "source_silence_mask": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
                    "rhythm_v3_cache_meta": {
                        "cache_version": 4,
                        "silent_token": 57,
                        "separator_aware": True,
                        "tail_open_units": 1,
                        "emit_silence_runs": True,
                        "debounce_min_run_frames": 2,
                        "phrase_boundary_threshold": 0.55,
                    },
                },
                paired_target_conditioning={
                    "paired_target_content_units": np.asarray([1, 2, 3], dtype=np.int64),
                    "paired_target_duration_obs": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
                    "paired_target_valid_mask": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
                    "paired_target_speech_mask": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
                    "source_frame_states": np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
                    "source_frame_to_run": np.asarray([0, 1, 2], dtype=np.int64),
                    "paired_target_frame_states": np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
                    "paired_target_frame_speech_prob": np.ones((3,), dtype=np.float32),
                    "paired_target_frame_valid": np.ones((3,), dtype=np.float32),
                    "source_cache_meta_signature": np.asarray(["other-source-signature"], dtype=object),
                },
            )

    def test_duration_v3_target_merge_prefers_alignment_kind_string_over_mode_id(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_minimal_v1_profile": True,
                "rhythm_v3_use_continuous_alignment": True,
            }
        )
        merged = dataset._merge_duration_v3_rhythm_targets(
            item={"item_name": "src_cached_kind_string"},
            source_cache={},
            paired_target_conditioning={},
            sample={
                "unit_duration_tgt": np.asarray([2.0, 3.0], dtype=np.float32),
                "unit_duration_proj_raw_tgt": np.asarray([2.0, 3.0], dtype=np.float32),
                "unit_alignment_mode_id_tgt": np.asarray([1], dtype=np.int64),
                "unit_alignment_kind_tgt": np.asarray(["continuous_viterbi_v1"], dtype=object),
                "unit_alignment_source_tgt": np.asarray(["run_state_viterbi"], dtype=object),
                "unit_alignment_version_tgt": np.asarray(["2026-04-13"], dtype=object),
                "unit_alignment_unmatched_speech_ratio_tgt": np.asarray([0.0], dtype=np.float32),
                "unit_alignment_mean_local_confidence_speech_tgt": np.asarray([0.9], dtype=np.float32),
                "unit_alignment_mean_coarse_confidence_speech_tgt": np.asarray([0.9], dtype=np.float32),
                "unit_alignment_source_cache_signature_tgt": np.asarray(["source-sig"], dtype=object),
                "unit_alignment_target_cache_signature_tgt": np.asarray(["target-sig"], dtype=object),
                "unit_alignment_sidecar_signature_tgt": np.asarray(["sidecar-sig"], dtype=object),
            },
        )
        self.assertEqual(
            str(np.asarray(merged["unit_alignment_kind_tgt"], dtype=object).reshape(-1)[0]),
            "continuous_viterbi_v1",
        )

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

    def test_duration_v3_minimal_profile_runtime_rejects_source_self_target_fallback_even_if_flag_is_enabled(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_minimal_v1_profile": True,
                "rhythm_v3_allow_source_self_target_fallback": True,
            }
        )
        with self.assertRaisesRegex(RuntimeError, "source-self target fallback"):
            dataset._merge_duration_v3_rhythm_targets(
                item={"item_name": "src_minimal"},
                source_cache={
                    "content_units": np.asarray([1, 2], dtype=np.int64),
                    "dur_anchor_src": np.asarray([2.0, 3.0], dtype=np.float32),
                },
                paired_target_conditioning={},
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

    def test_duration_v3_reference_conditioning_requires_text_signatures_when_same_text_reference_is_disabled(self) -> None:
        dataset = _DummyDataset(
            {
                "rhythm_enable_v3": True,
                "rhythm_v3_backbone": "prompt_summary",
                "rhythm_v3_anchor_mode": "source_observed",
                "rhythm_v3_disallow_same_text_reference": True,
            }
        )
        item = {
            "item_name": "src_ref_guard",
            "content_units": np.asarray([1, 57, 2], dtype=np.int64),
            "dur_anchor_src": np.asarray([2.0, 1.0, 2.0], dtype=np.float32),
            "source_silence_mask": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        }
        ref_item = {
            "item_name": "ref_ref_guard",
            "content_units": np.asarray([3, 57, 4], dtype=np.int64),
            "dur_anchor_src": np.asarray([3.0, 1.0, 3.0], dtype=np.float32),
            "source_silence_mask": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        }
        with self.assertRaisesRegex(RuntimeError, "text signatures"):
            dataset._get_reference_rhythm_conditioning(
                ref_item,
                {},
                target_mode="runtime_only",
                item=item,
            )


if __name__ == "__main__":
    unittest.main()
