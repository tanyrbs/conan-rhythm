from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_ALIGNMENT_PROJECTION_PATH = ROOT / "tasks" / "Conan" / "rhythm" / "duration_v3" / "alignment_projection.py"
_SPEC = importlib.util.spec_from_file_location("rhythm_test_alignment_projection", _ALIGNMENT_PROJECTION_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Failed to load alignment_projection from {_ALIGNMENT_PROJECTION_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
ContinuousRunAligner = _MODULE.ContinuousRunAligner
project_target_runs_onto_source = _MODULE.project_target_runs_onto_source


class ContinuousAlignmentProjectionTests(unittest.TestCase):
    @staticmethod
    def _base_source() -> dict[str, np.ndarray]:
        return {
            "source_units": np.asarray([11, 12, 13], dtype=np.int64),
            "source_durations": np.asarray([3.0, 4.0, 5.0], dtype=np.float32),
            "source_silence_mask": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        }

    def test_continuous_precomputed_alignment_projects_monotone_path_and_occupancy(self) -> None:
        result = project_target_runs_onto_source(
            **self._base_source(),
            target_units=np.asarray([21, 22, 23, 24], dtype=np.int64),
            target_durations=np.asarray([2.0, 1.0, 4.0, 3.0], dtype=np.float32),
            target_valid_mask=np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            target_speech_mask=np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            use_continuous_alignment=True,
            precomputed_alignment={
                "assigned_source": np.asarray([0, 0, 1, 2], dtype=np.int64),
                "assigned_cost": np.asarray([0.05, 0.10, 0.15, 0.20], dtype=np.float32),
            },
        )

        self.assertEqual(result["alignment_kind"], "continuous_precomputed")
        np.testing.assert_array_equal(
            result["assigned_source"],
            np.asarray([0, 0, 1, 2], dtype=np.int64),
        )
        self.assertTrue(np.all(np.diff(result["assigned_source"]) >= 0))
        np.testing.assert_allclose(
            result["projected"],
            np.asarray([3.0, 4.0, 3.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            result["assigned_cost"],
            np.asarray([0.05, 0.10, 0.15, 0.20], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            result["source_valid_run_index"],
            np.asarray([0, 1, 2], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            result["target_valid_run_index"],
            np.asarray([0, 1, 2, 3], dtype=np.int64),
        )
        self.assertAlmostEqual(float(result["unmatched_speech_ratio"]), 0.0, places=6)
        self.assertAlmostEqual(float(result["projection_fallback_ratio"]), 0.0, places=6)
        self.assertAlmostEqual(
            float(result["mean_local_conf_speech"]),
            float(result["mean_local_confidence_speech"]),
            places=6,
        )
        self.assertAlmostEqual(float(result["projection_hard_bad"]), 0.0, places=6)
        self.assertTrue(np.isnan(float(result["alignment_global_path_cost"])))
        self.assertTrue(np.isnan(float(result["alignment_global_path_mean_cost"])))

    def test_continuous_alignment_request_without_metadata_fails_fast(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "requires explicit source_frame_states/source_frame_to_run/target_frame_states sidecars",
        ):
            project_target_runs_onto_source(
                **self._base_source(),
                target_units=np.asarray([21, 22], dtype=np.int64),
                target_durations=np.asarray([2.0, 1.0], dtype=np.float32),
                target_valid_mask=np.asarray([1.0, 1.0], dtype=np.float32),
                target_speech_mask=np.asarray([1.0, 1.0], dtype=np.float32),
                use_continuous_alignment=True,
                precomputed_alignment=None,
            )

    def test_target_valid_run_index_tracks_filtered_target_runs(self) -> None:
        result = project_target_runs_onto_source(
            **self._base_source(),
            target_units=np.asarray([21, 22, 23], dtype=np.int64),
            target_durations=np.asarray([2.0, 0.0, 5.0], dtype=np.float32),
            target_valid_mask=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            target_speech_mask=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            use_continuous_alignment=True,
            precomputed_alignment={
                "assigned_source": np.asarray([0, 2], dtype=np.int64),
                "assigned_cost": np.asarray([0.10, 0.20], dtype=np.float32),
            },
        )

        # Regression guard: the reported target indices should match the runs
        # that actually survived duration>0 and valid_mask filtering.
        np.testing.assert_array_equal(
            result["target_valid_run_index"],
            np.asarray([0, 2], dtype=np.int64),
        )

    def test_continuous_viterbi_alignment_projects_frame_occupancy(self) -> None:
        result = project_target_runs_onto_source(
            source_units=np.asarray([11, 12, 13], dtype=np.int64),
            source_durations=np.asarray([2.0, 2.0, 2.0], dtype=np.float32),
            source_silence_mask=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            target_units=np.asarray([11, 12, 13], dtype=np.int64),
            target_durations=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            target_valid_mask=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            target_speech_mask=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            use_continuous_alignment=True,
            source_frame_states=np.asarray(
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
            source_frame_to_run=np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64),
            target_frame_states=np.asarray(
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
            target_frame_speech_prob=np.ones((6,), dtype=np.float32),
            target_frame_valid=np.ones((6,), dtype=np.float32),
            target_frame_weight=np.asarray([1.0, 2.0, 1.0, 3.0, 1.0, 4.0], dtype=np.float32),
        )

        self.assertEqual(result["alignment_kind"], "continuous_viterbi_v1")
        self.assertEqual(result["alignment_mode"], "continuous_viterbi_v1")
        self.assertEqual(result["alignment_source"], "run_state_viterbi")
        self.assertEqual(result["alignment_version"], "2026-04-13")
        self.assertEqual(result["posterior_kind"], "viterbi_margin_only")
        self.assertEqual(result["confidence_kind"], "heuristic_v1")
        self.assertEqual(result["confidence_formula_version"], "heuristic_margin_cost_type_v1")
        self.assertEqual(result["source_progress_kind"], "anchor_duration_cdf")
        self.assertEqual(result["target_progress_kind"], "weighted_cdf")
        self.assertEqual(result["source_run_proto_kind"], "stability_boundary_weighted_mean_v1")
        np.testing.assert_array_equal(
            result["run_occ_expected_is_hard_proxy"],
            np.asarray([1], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            result["alignment_allow_source_skip"],
            np.asarray([0], dtype=np.int64),
        )
        self.assertEqual(result["run_occ_expected_semantics"], "hard_viterbi_proxy")
        np.testing.assert_array_equal(
            result["assigned_source"],
            np.asarray([0, 0, 1, 2, 2, 2], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            result["target_valid_run_index"],
            np.asarray([0, 1, 2], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            result["target_valid_observation_index"],
            np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            result["target_valid_observation_index_pre_dp_filter"],
            np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            result["target_dp_weight_pruned_index"],
            np.asarray([], dtype=np.int64),
        )
        np.testing.assert_allclose(
            result["projected"],
            np.asarray([2.0, 1.0, 3.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            result["run_occ_expected"],
            np.asarray([2.0, 1.0, 3.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            result["run_occ_hard"],
            np.asarray([2.0, 1.0, 3.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            result["run_occ_weighted"],
            np.asarray([3.0, 1.0, 8.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            result["coverage"],
            np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            result["coverage_binary"],
            np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            result["coverage_fraction"],
            np.asarray([1.0, 0.5, 1.5], dtype=np.float32),
        )
        np.testing.assert_allclose(
            result["expected_frame_support"],
            np.asarray([2.0, 2.0, 2.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            result["alignment_min_dp_weight"],
            np.asarray([0.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            result["run_skip_flag"],
            np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        )
        self.assertEqual(tuple(result["confidence_cost_term"].shape), (3,))
        self.assertEqual(tuple(result["confidence_margin_term"].shape), (3,))
        self.assertEqual(tuple(result["confidence_type_term"].shape), (3,))
        self.assertEqual(tuple(result["confidence_match_term"].shape), (3,))
        self.assertEqual(tuple(result["run_margin"].shape), (3,))
        self.assertIn("alignment_local_margin_p10", result)
        self.assertTrue(np.isfinite(float(result["alignment_local_margin_p10"])))
        self.assertTrue(np.all(result["confidence_cost_term"] > 0.0))
        self.assertTrue(np.all(result["confidence_margin_term"] > 0.0))
        self.assertTrue(np.all(result["confidence_type_term"] > 0.0))
        self.assertTrue(np.all(result["confidence_match_term"] > 0.0))
        self.assertEqual(tuple(result["source_run_proto"].shape), (3, 3))
        self.assertEqual(tuple(result["source_run_proto_var"].shape), (3, 3))
        np.testing.assert_allclose(
            result["source_frame_count"],
            np.asarray([2.0, 2.0, 2.0], dtype=np.float32),
        )
        self.assertTrue(np.all(result["source_frame_weight_sum"] > 0.0))

    def test_continuous_viterbi_alignment_prunes_low_weight_frames_before_dp(self) -> None:
        result = project_target_runs_onto_source(
            source_units=np.asarray([11, 12], dtype=np.int64),
            source_durations=np.asarray([2.0, 2.0], dtype=np.float32),
            source_silence_mask=np.asarray([0.0, 0.0], dtype=np.float32),
            target_units=np.asarray([11, 12], dtype=np.int64),
            target_durations=np.asarray([1.0, 1.0], dtype=np.float32),
            target_valid_mask=np.asarray([1.0, 1.0], dtype=np.float32),
            target_speech_mask=np.asarray([1.0, 1.0], dtype=np.float32),
            use_continuous_alignment=True,
            source_frame_states=np.asarray(
                [
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            source_frame_to_run=np.asarray([0, 0, 1, 1], dtype=np.int64),
            target_frame_states=np.asarray(
                [
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            target_frame_speech_prob=np.ones((4,), dtype=np.float32),
            target_frame_valid=np.ones((4,), dtype=np.float32),
            target_frame_weight=np.asarray([1.0, 0.1, 1.0, 1.0], dtype=np.float32),
            continuous_aligner_kwargs={
                "min_dp_weight": 0.5,
            },
        )

        np.testing.assert_allclose(
            result["alignment_min_dp_weight"],
            np.asarray([0.5], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            result["target_valid_observation_index_pre_dp_filter"],
            np.asarray([0, 1, 2, 3], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            result["target_dp_weight_pruned_index"],
            np.asarray([1], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            result["target_valid_observation_index"],
            np.asarray([0, 2, 3], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            result["assigned_source"],
            np.asarray([0, 1, 1], dtype=np.int64),
        )
        np.testing.assert_allclose(
            result["projected"],
            np.asarray([1.0, 2.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            result["run_skip_flag"],
            np.asarray([0.0, 0.0], dtype=np.float32),
        )

    def test_continuous_viterbi_skip_source_audit_mode_exports_zero_coverage_for_skipped_runs(self) -> None:
        result = project_target_runs_onto_source(
            source_units=np.asarray([11, 12, 13, 14], dtype=np.int64),
            source_durations=np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            source_silence_mask=np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            target_units=np.asarray([11, 13, 14], dtype=np.int64),
            target_durations=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            target_valid_mask=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            target_speech_mask=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            use_continuous_alignment=True,
            source_frame_states=np.asarray(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            source_frame_to_run=np.asarray([0, 1, 2, 3], dtype=np.int64),
            target_frame_states=np.asarray(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            target_frame_speech_prob=np.ones((3,), dtype=np.float32),
            target_frame_valid=np.ones((3,), dtype=np.float32),
            continuous_aligner_kwargs={
                "allow_source_skip": True,
                "skip_penalty": 0.05,
                "band_width": 2,
            },
        )

        np.testing.assert_array_equal(
            result["alignment_allow_source_skip"],
            np.asarray([1], dtype=np.int64),
        )
        np.testing.assert_allclose(
            result["alignment_skip_penalty"],
            np.asarray([0.05], dtype=np.float32),
        )
        self.assertEqual(int(np.asarray(result["assigned_source"], dtype=np.int64)[1]), 2)
        self.assertAlmostEqual(float(np.asarray(result["coverage_fraction"], dtype=np.float32)[1]), 0.0, places=6)
        self.assertAlmostEqual(float(np.asarray(result["confidence_local"], dtype=np.float32)[1]), 0.0, places=6)
        self.assertAlmostEqual(float(np.asarray(result["confidence_coarse"], dtype=np.float32)[1]), 0.0, places=6)
        self.assertAlmostEqual(float(np.asarray(result["skipped_source_speech_ratio"], dtype=np.float32)), 0.25, places=6)
        self.assertAlmostEqual(float(np.asarray(result["unmatched_speech_ratio"], dtype=np.float32)), 0.25, places=6)

    def test_anchor_aware_band_prior_prefers_long_anchor_run_later_than_uniform_index_prior(self) -> None:
        aligner = ContinuousRunAligner(
            lambda_emb=0.0,
            lambda_type=0.0,
            lambda_band=1.0,
            band_width=1,
            band_ratio=0.01,
        )
        source_run_proto = np.ones((5, 1), dtype=np.float32)
        source_run_types = np.zeros((5,), dtype=np.float32)
        target_frame_state = np.ones((12, 1), dtype=np.float32)
        target_frame_speech_prob = np.ones((12,), dtype=np.float32)

        cost_uniform = aligner.build_local_cost(
            source_run_proto=source_run_proto,
            source_run_types=source_run_types,
            target_frame_state=target_frame_state,
            target_frame_speech_prob=target_frame_speech_prob,
            source_durations=None,
        )
        cost_anchor = aligner.build_local_cost(
            source_run_proto=source_run_proto,
            source_run_types=source_run_types,
            target_frame_state=target_frame_state,
            target_frame_speech_prob=target_frame_speech_prob,
            source_durations=np.asarray([1.0, 1.0, 8.0, 1.0, 1.0], dtype=np.float32),
        )

        late_mid_frame = 9
        self.assertEqual(int(np.argmin(cost_uniform[late_mid_frame])), 3)
        self.assertEqual(int(np.argmin(cost_anchor[late_mid_frame])), 2)
        self.assertLess(
            float(cost_anchor[late_mid_frame, 2]),
            float(cost_uniform[late_mid_frame, 2]),
        )

    def test_bad_cost_frames_raise_unmatched_speech_ratio(self) -> None:
        result = project_target_runs_onto_source(
            source_units=np.asarray([11, 12, 13], dtype=np.int64),
            source_durations=np.asarray([2.0, 2.0, 2.0], dtype=np.float32),
            source_silence_mask=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            target_units=np.asarray([11, 12, 13], dtype=np.int64),
            target_durations=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            target_valid_mask=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            target_speech_mask=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            use_continuous_alignment=True,
            source_frame_states=np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            source_frame_to_run=np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64),
            target_frame_states=np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            target_frame_speech_prob=np.ones((3,), dtype=np.float32),
            target_frame_valid=np.ones((3,), dtype=np.float32),
        )

        np.testing.assert_array_equal(
            result["assigned_source"],
            np.asarray([0, 1, 2], dtype=np.int64),
        )
        self.assertGreater(float(result["assigned_cost"][1]), 1.2)
        self.assertAlmostEqual(float(result["unmatched_speech_ratio"]), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(float(result["projection_fallback_ratio"]), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(float(result["projection_hard_bad"]), 1.0, places=6)
        self.assertIn("projection_gate_reason", result)
        self.assertTrue(str(result["projection_gate_reason"]).strip())
        self.assertTrue(np.isfinite(float(result["alignment_global_path_cost"])))
        self.assertTrue(np.isfinite(float(result["alignment_global_path_mean_cost"])))

    def test_continuous_viterbi_skip_source_mode_allows_zero_occupancy_with_low_confidence(self) -> None:
        aligner = ContinuousRunAligner(
            lambda_emb=1.0,
            lambda_type=0.0,
            lambda_band=0.0,
            lambda_unit=0.0,
            allow_source_skip=True,
            skip_penalty=0.25,
        )
        source_run_units = np.asarray([11, 12, 13, 14], dtype=np.int64)
        source_run_types = np.zeros((4,), dtype=np.float32)
        source_frame_states = np.eye(4, dtype=np.float32)
        source_frame_to_run = np.asarray([0, 1, 2, 3], dtype=np.int64)
        target_frame_states = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        alignment = aligner.align(
            source_run_units=source_run_units,
            source_run_types=source_run_types,
            source_frame_states=source_frame_states,
            source_frame_to_run=source_frame_to_run,
            source_durations=np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            target_frame_states=target_frame_states,
            target_frame_speech_prob=np.ones((3,), dtype=np.float32),
            target_frame_valid=np.ones((3,), dtype=np.float32),
        )

        np.testing.assert_array_equal(
            alignment["assigned_source"],
            np.asarray([0, 1, 3], dtype=np.int64),
        )
        np.testing.assert_allclose(
            alignment["run_occ_viterbi"],
            np.asarray([1.0, 1.0, 0.0, 1.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            alignment["run_confidence_local_final"],
            np.asarray(alignment["run_confidence_local"], dtype=np.float32),
        )
        np.testing.assert_allclose(
            alignment["run_confidence_coarse_final"],
            np.asarray(alignment["run_confidence_coarse"], dtype=np.float32),
        )
        self.assertAlmostEqual(float(np.asarray(alignment["alignment_skip_penalty"]).reshape(-1)[0]), 0.25, places=6)
        self.assertEqual(int(np.asarray(alignment["alignment_allow_source_skip"]).reshape(-1)[0]), 1)
        self.assertEqual(float(alignment["run_confidence_local"][2]), 0.0)
        self.assertEqual(float(alignment["run_confidence_coarse"][2]), 0.0)

        projected = project_target_runs_onto_source(
            source_units=source_run_units,
            source_durations=np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            source_silence_mask=np.zeros((4,), dtype=np.float32),
            target_units=np.asarray([11, 12, 14], dtype=np.int64),
            target_durations=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            target_valid_mask=np.ones((3,), dtype=np.float32),
            target_speech_mask=np.ones((3,), dtype=np.float32),
            use_continuous_alignment=True,
            source_frame_states=source_frame_states,
            source_frame_to_run=source_frame_to_run,
            target_frame_states=target_frame_states,
            target_frame_speech_prob=np.ones((3,), dtype=np.float32),
            target_frame_valid=np.ones((3,), dtype=np.float32),
            continuous_aligner_kwargs={
                "lambda_emb": 1.0,
                "lambda_type": 0.0,
                "lambda_band": 0.0,
                "lambda_unit": 0.0,
                "allow_source_skip": True,
                "skip_penalty": 0.25,
            },
        )

        np.testing.assert_allclose(
            projected["projected"],
            np.asarray([1.0, 1.0, 0.0, 1.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            projected["coverage_binary"],
            np.asarray([1.0, 1.0, 0.0, 1.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            projected["coverage_fraction"],
            np.asarray([1.0, 1.0, 0.0, 1.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            projected["expected_frame_support"],
            np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        )
        self.assertEqual(float(projected["confidence_local"][2]), 0.0)
        self.assertEqual(float(projected["confidence_coarse"][2]), 0.0)

    def test_projection_health_exports_skipped_source_run_as_hard_bad_fallback(self) -> None:
        source_run_units = np.asarray([11, 12, 13, 14], dtype=np.int64)
        source_frame_states = np.eye(4, dtype=np.float32)
        target_frame_states = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        projected = project_target_runs_onto_source(
            source_units=source_run_units,
            source_durations=np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            source_silence_mask=np.zeros((4,), dtype=np.float32),
            target_units=np.asarray([11, 12, 14], dtype=np.int64),
            target_durations=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            target_valid_mask=np.ones((3,), dtype=np.float32),
            target_speech_mask=np.ones((3,), dtype=np.float32),
            use_continuous_alignment=True,
            source_frame_states=source_frame_states,
            source_frame_to_run=np.asarray([0, 1, 2, 3], dtype=np.int64),
            target_frame_states=target_frame_states,
            target_frame_speech_prob=np.ones((3,), dtype=np.float32),
            target_frame_valid=np.ones((3,), dtype=np.float32),
            alignment_soft_repair=True,
            continuous_aligner_kwargs={
                "lambda_emb": 1.0,
                "lambda_type": 0.0,
                "lambda_band": 0.0,
                "lambda_unit": 0.0,
                "allow_source_skip": True,
                "skip_penalty": 0.25,
            },
        )

        np.testing.assert_allclose(projected["projected"], np.asarray([1.0, 1.0, 0.0, 1.0], dtype=np.float32))
        np.testing.assert_allclose(
            projected["run_skip_flag"],
            np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        )
        self.assertAlmostEqual(float(projected["unmatched_speech_ratio"]), 0.25, places=6)
        self.assertIn("projection_fallback_ratio", projected)
        self.assertIn("projection_hard_bad", projected)
        self.assertAlmostEqual(
            float(np.asarray(projected["projection_fallback_ratio"], dtype=np.float32).reshape(-1)[0]),
            0.25,
            places=6,
        )
        self.assertAlmostEqual(
            float(np.asarray(projected["projection_hard_bad"], dtype=np.float32).reshape(-1)[0]),
            1.0,
            places=6,
        )

    def test_alignment_soft_repair_handles_missing_source_frame_support(self) -> None:
        projected = project_target_runs_onto_source(
            source_units=np.asarray([11, 12, 13], dtype=np.int64),
            source_durations=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            source_silence_mask=np.zeros((3,), dtype=np.float32),
            target_units=np.asarray([11, 12, 13], dtype=np.int64),
            target_durations=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            target_valid_mask=np.ones((3,), dtype=np.float32),
            target_speech_mask=np.ones((3,), dtype=np.float32),
            use_continuous_alignment=True,
            source_frame_states=np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            source_frame_to_run=np.asarray([0, 2], dtype=np.int64),
            target_frame_states=np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            target_frame_speech_prob=np.ones((3,), dtype=np.float32),
            target_frame_valid=np.ones((3,), dtype=np.float32),
            alignment_soft_repair=True,
        )

        self.assertEqual(projected["alignment_kind"], "continuous_viterbi_v1")
        np.testing.assert_allclose(
            projected["source_frame_count"],
            np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            projected["source_frame_weight_sum"],
            np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
        )
        self.assertEqual(tuple(projected["source_run_proto"].shape), (3, 3))
        self.assertTrue(np.all(np.isfinite(projected["source_run_proto"])))


if __name__ == "__main__":
    unittest.main()
