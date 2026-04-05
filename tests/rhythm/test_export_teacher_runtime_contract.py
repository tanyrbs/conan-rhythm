from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_rhythm_teacher_targets import _require_teacher_export_runtime


class TeacherExportRuntimeContractTests(unittest.TestCase):
    def test_accepts_teacher_as_main_runtime(self) -> None:
        execution = object()
        unit_batch = object()
        resolved_execution, resolved_unit_batch = _require_teacher_export_runtime(
            {
                "rhythm_execution": execution,
                "rhythm_unit_batch": unit_batch,
                "rhythm_teacher_as_main": torch.tensor(1.0),
            },
            np=np,
            torch=torch,
        )
        self.assertIs(resolved_execution, execution)
        self.assertIs(resolved_unit_batch, unit_batch)

    def test_accepts_teacher_only_runtime(self) -> None:
        execution = object()
        unit_batch = object()
        resolved_execution, resolved_unit_batch = _require_teacher_export_runtime(
            {
                "rhythm_execution": execution,
                "rhythm_unit_batch": unit_batch,
                "rhythm_teacher_only_stage": 1.0,
            },
            np=np,
            torch=torch,
        )
        self.assertIs(resolved_execution, execution)
        self.assertIs(resolved_unit_batch, unit_batch)

    def test_rejects_missing_teacher_runtime_semantics(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "teacher_offline runtime semantics"):
            _require_teacher_export_runtime(
                {
                    "rhythm_execution": object(),
                    "rhythm_unit_batch": object(),
                },
                np=np,
                torch=torch,
            )

    def test_rejects_shadow_offline_execution(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "rhythm_offline_execution"):
            _require_teacher_export_runtime(
                {
                    "rhythm_execution": object(),
                    "rhythm_unit_batch": object(),
                    "rhythm_teacher_as_main": 1.0,
                    "rhythm_offline_execution": object(),
                },
                np=np,
                torch=torch,
            )


if __name__ == "__main__":
    unittest.main()
