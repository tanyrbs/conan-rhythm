from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.Conan.rhythm.factory import build_projector_config_from_hparams


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


if __name__ == "__main__":
    unittest.main()
