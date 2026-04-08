from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.Conan.rhythm.surface_metadata import materialize_rhythm_cache_compat_fields


class SurfaceMetadataCompatTests(unittest.TestCase):
    def test_materialize_rhythm_cache_compat_fields_adds_pause_blank_aliases_bidirectionally(self) -> None:
        blank_only = materialize_rhythm_cache_compat_fields(
            {
                "rhythm_blank_exec_tgt": np.asarray([1.0, 2.0], dtype=np.float32),
                "rhythm_blank_budget_tgt": np.asarray([3.0], dtype=np.float32),
            }
        )
        assert blank_only is not None
        self.assertIn("rhythm_pause_exec_tgt", blank_only)
        self.assertIn("rhythm_pause_budget_tgt", blank_only)
        self.assertTrue(np.allclose(blank_only["rhythm_pause_exec_tgt"], blank_only["rhythm_blank_exec_tgt"]))
        self.assertTrue(np.allclose(blank_only["rhythm_pause_budget_tgt"], blank_only["rhythm_blank_budget_tgt"]))

        pause_only = materialize_rhythm_cache_compat_fields(
            {
                "rhythm_pause_exec_tgt": np.asarray([2.0, 1.0], dtype=np.float32),
                "rhythm_pause_budget_tgt": np.asarray([4.0], dtype=np.float32),
            }
        )
        assert pause_only is not None
        self.assertIn("rhythm_blank_exec_tgt", pause_only)
        self.assertIn("rhythm_blank_budget_tgt", pause_only)
        self.assertTrue(np.allclose(pause_only["rhythm_blank_exec_tgt"], pause_only["rhythm_pause_exec_tgt"]))
        self.assertTrue(np.allclose(pause_only["rhythm_blank_budget_tgt"], pause_only["rhythm_pause_budget_tgt"]))


if __name__ == "__main__":
    unittest.main()
