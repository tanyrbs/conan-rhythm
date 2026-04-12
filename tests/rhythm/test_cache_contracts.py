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
            "source_silence_mask": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
            "source_run_stability": np.asarray([1.0, 0.2, 0.6], dtype=np.float32),
        }
        conditioning = dataset._build_reference_prompt_unit_conditioning(prompt_item, target_mode="runtime_only")
        expected = np.asarray([1.0, 0.0, 0.7], dtype=np.float32)
        self.assertIn("prompt_global_weight", conditioning)
        self.assertIn("prompt_global_weight_present", conditioning)
        self.assertIn("g_trim_ratio", conditioning)
        self.assertEqual(float(np.asarray(conditioning["prompt_global_weight_present"]).reshape(-1)[0]), 1.0)
        self.assertTrue(np.allclose(conditioning["prompt_global_weight"], expected))

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
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            prior_path = Path(tmpdir) / "unit_prior.npz"
            np.savez(
                prior_path,
                unit_log_prior=np.asarray([0.0, 0.2, 0.4], dtype=np.float32),
                unit_prior_source=np.asarray(["demo"], dtype=object),
                unit_prior_version=np.asarray(["v1"], dtype=object),
                unit_prior_vocab_size=np.asarray([3], dtype=np.int64),
            )
            dataset = _DummyDataset(
                {
                    "rhythm_enable_v3": True,
                    "rhythm_v3_backbone": "prompt_summary",
                    "rhythm_v3_anchor_mode": "source_observed",
                    "rhythm_v3_emit_silence_runs": True,
                    "rhythm_v3_g_variant": "unit_norm",
                    "rhythm_v3_unit_prior_path": str(prior_path),
                }
            )
            prompt_item = {
                "item_name": "prompt_unit_norm_cfg",
                "content_units": np.asarray([1, 2], dtype=np.int64),
                "dur_anchor_src": np.asarray([3.0, 4.0], dtype=np.float32),
                "source_silence_mask": np.asarray([0.0, 0.0], dtype=np.float32),
            }
            conditioning = dataset._build_reference_prompt_unit_conditioning(prompt_item, target_mode="runtime_only")
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
        self.assertAlmostEqual(float(np.asarray(built["unit_alignment_band_ratio_tgt"]).reshape(-1)[0]), 0.08)
        self.assertAlmostEqual(float(np.asarray(built["unit_alignment_lambda_emb_tgt"]).reshape(-1)[0]), 1.0)
        self.assertAlmostEqual(float(np.asarray(built["unit_alignment_lambda_type_tgt"]).reshape(-1)[0]), 0.5)

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
        self.assertIn("alignment_source", keys)
        self.assertIn("alignment_version", keys)
        self.assertEqual(collate_spec["unit_alignment_kind_tgt"][0], "object")
        self.assertEqual(collate_spec["unit_alignment_source_tgt"][0], "object")
        self.assertEqual(collate_spec["unit_alignment_version_tgt"][0], "object")
        self.assertEqual(collate_spec["unit_alignment_band_ratio_tgt"][0], "float")
        self.assertEqual(collate_spec["unit_alignment_lambda_emb_tgt"][0], "float")
        self.assertEqual(collate_spec["unit_alignment_lambda_type_tgt"][0], "float")
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
            "unit_alignment_mean_local_confidence_speech_tgt": np.asarray([0.4], dtype=np.float32),
            "unit_alignment_mean_coarse_confidence_speech_tgt": np.asarray([0.7], dtype=np.float32),
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
                },
            )

    def test_duration_v3_target_merge_derives_human_readable_alignment_kind_from_mode_id(self) -> None:
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
                "alignment_source": np.asarray(["hubert_state_dp_v1"], dtype=object),
                "alignment_version": np.asarray(["2026-04-12"], dtype=object),
            },
        )
        self.assertEqual(int(np.asarray(merged["unit_alignment_mode_id_tgt"]).reshape(-1)[0]), 1)
        self.assertEqual(
            str(np.asarray(merged["unit_alignment_kind_tgt"], dtype=object).reshape(-1)[0]),
            "continuous_precomputed",
        )
        self.assertEqual(merged["alignment_source"], "hubert_state_dp_v1")
        self.assertEqual(merged["alignment_version"], "2026-04-12")

    def test_duration_v3_target_merge_supports_continuous_viterbi_mode_id_without_kind_string(self) -> None:
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
            item={"item_name": "src_cached_viterbi_kind"},
            source_cache={},
            paired_target_conditioning={},
            sample={
                "unit_duration_tgt": np.asarray([2.0, 3.0], dtype=np.float32),
                "unit_duration_proj_raw_tgt": np.asarray([2.0, 3.0], dtype=np.float32),
                "unit_alignment_mode_id_tgt": np.asarray([2], dtype=np.int64),
                "alignment_source": np.asarray(["run_state_viterbi"], dtype=object),
                "alignment_version": np.asarray(["2026-04-13"], dtype=object),
            },
        )
        self.assertEqual(
            str(np.asarray(merged["unit_alignment_kind_tgt"], dtype=object).reshape(-1)[0]),
            "continuous_viterbi_v1",
        )
        self.assertAlmostEqual(float(np.asarray(merged["unit_alignment_band_ratio_tgt"]).reshape(-1)[0]), 0.08)

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
                "alignment_source": np.asarray(["run_state_viterbi"], dtype=object),
                "alignment_version": np.asarray(["2026-04-13"], dtype=object),
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
