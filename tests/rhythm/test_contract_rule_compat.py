from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.Conan.rhythm.config_contract_rules.compat import (
    resolve_duplicate_primary_distill_dedupe_flag,
    validate_pause_boundary_alias_consistency,
    validate_prefix_lambda_alias_consistency,
)


class ContractRuleCompatTests(unittest.TestCase):
    def test_duplicate_distill_alias_prefers_public_key(self) -> None:
        resolved = resolve_duplicate_primary_distill_dedupe_flag(
            {
                "rhythm_dedupe_teacher_primary_cache_distill": False,
                "rhythm_suppress_duplicate_primary_distill": True,
            }
        )
        self.assertFalse(resolved)

    def test_pause_boundary_alias_mismatch_is_an_error(self) -> None:
        errors: list[str] = []
        warnings: list[str] = []
        validate_pause_boundary_alias_consistency(
            {
                "rhythm_pause_exec_boundary_boost": 0.75,
                "rhythm_pause_boundary_weight": 0.35,
            },
            errors,
            warnings,
        )
        self.assertTrue(
            any("rhythm_pause_exec_boundary_boost and rhythm_pause_boundary_weight" in e for e in errors)
        )
        self.assertEqual(warnings, [])

    def test_cumplan_alias_match_only_warns(self) -> None:
        errors: list[str] = []
        warnings: list[str] = []
        validate_prefix_lambda_alias_consistency(
            {
                "lambda_rhythm_carry": 0.15,
                "lambda_rhythm_cumplan": 0.15,
            },
            errors,
            warnings,
        )
        self.assertEqual(errors, [])
        self.assertTrue(any("prefer lambda_rhythm_cumplan" in w for w in warnings))


if __name__ == "__main__":
    unittest.main()
