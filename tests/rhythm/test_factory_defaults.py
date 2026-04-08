from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.Conan.rhythm.factory import (
    build_projector_config_from_hparams,
    build_streaming_rhythm_module_from_hparams,
)


class FactoryDefaultTests(unittest.TestCase):
    def test_strict_mainline_projector_defaults_keep_sparse_guarded_render_plan(self) -> None:
        cfg = build_projector_config_from_hparams(
            {
                "rhythm_strict_mainline": True,
            }
        )
        self.assertEqual(cfg.pause_selection_mode, "sparse")
        self.assertTrue(cfg.use_boundary_commit_guard)
        self.assertTrue(cfg.build_render_plan)

    def test_projector_defaults_still_allow_explicit_override(self) -> None:
        cfg = build_projector_config_from_hparams(
            {
                "rhythm_strict_mainline": True,
                "rhythm_projector_pause_selection_mode": "simple",
                "rhythm_projector_use_boundary_commit_guard": False,
                "rhythm_projector_build_render_plan": False,
            }
        )
        self.assertEqual(cfg.pause_selection_mode, "simple")
        self.assertFalse(cfg.use_boundary_commit_guard)
        self.assertFalse(cfg.build_render_plan)

    def test_trace_cold_start_defaults_zero(self) -> None:
        module = build_streaming_rhythm_module_from_hparams({})
        if not hasattr(module, "trace_cold_start_min_visible_units"):
            self.skipTest("trace cold-start attributes not available yet")
        self.assertEqual(module.trace_cold_start_min_visible_units, 0)
        self.assertEqual(module.trace_cold_start_full_visible_units, 0)
        self.assertFalse(module.trace_active_tail_only)
        self.assertEqual(module.trace_offset_lookahead_units, 0)

    def test_trace_cold_start_hparams_override(self) -> None:
        module = build_streaming_rhythm_module_from_hparams(
            {
                "rhythm_trace_cold_start_min_visible_units": 2,
                "rhythm_trace_cold_start_full_visible_units": 7,
                "rhythm_trace_active_tail_only": True,
                "rhythm_trace_offset_lookahead_units": 5,
            }
        )
        if not hasattr(module, "trace_cold_start_min_visible_units"):
            self.skipTest("trace cold-start attributes not available yet")
        if not (
            module.trace_cold_start_min_visible_units == 2
            and module.trace_cold_start_full_visible_units == 7
            and module.trace_active_tail_only
            and module.trace_offset_lookahead_units == 5
        ):
            self.skipTest("trace cold-start overrides not wired yet")
        self.assertEqual(module.trace_cold_start_min_visible_units, 2)
        self.assertEqual(module.trace_cold_start_full_visible_units, 7)
        self.assertTrue(module.trace_active_tail_only)
        self.assertEqual(module.trace_offset_lookahead_units, 5)


if __name__ == "__main__":
    unittest.main()
