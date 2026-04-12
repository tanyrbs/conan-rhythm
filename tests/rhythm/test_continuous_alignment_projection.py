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

    def test_continuous_alignment_request_without_metadata_fails_fast(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "requires either continuous_precomputed metadata or explicit source_frame_states/source_frame_to_run/target_frame_states sidecars",
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
        )

        self.assertEqual(result["alignment_kind"], "continuous_viterbi_v1")
        self.assertEqual(result["alignment_source"], "run_state_viterbi")
        self.assertEqual(result["alignment_version"], "2026-04-13")
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
        np.testing.assert_allclose(
            result["projected"],
            np.asarray([2.0, 1.0, 3.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            result["run_occ_expected"],
            np.asarray([2.0, 1.0, 3.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            result["coverage"],
            np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
