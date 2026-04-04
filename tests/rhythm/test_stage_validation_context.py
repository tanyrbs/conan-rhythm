from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.Conan.rhythm.config_contract_rules.context import build_stage_validation_context


class StageValidationContextTests(unittest.TestCase):
    def test_context_defaults_match_runtime_for_optional_teacher_and_retimed_pitch_flags(self) -> None:
        ctx = build_stage_validation_context(
            {
                "rhythm_enable_v2": True,
                "rhythm_stage": "student_kd",
            }
        )
        self.assertFalse(ctx.knobs.enable_learned_offline_teacher)
        self.assertFalse(ctx.knobs.disable_pitch_when_retimed)


if __name__ == "__main__":
    unittest.main()
