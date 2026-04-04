from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.run import resolve_task_cls


class RunTaskEntrypointTests(unittest.TestCase):
    def test_resolve_task_cls_rejects_empty_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty import path"):
            resolve_task_cls("")

    def test_resolve_task_cls_rejects_missing_module_separator(self) -> None:
        with self.assertRaisesRegex(ValueError, "pkg.module.ClassName"):
            resolve_task_cls("NotAPath")

    def test_resolve_task_cls_raises_import_error_for_missing_module(self) -> None:
        with self.assertRaisesRegex(ImportError, "Failed to import task module"):
            resolve_task_cls("missing.module.FakeTask")

    def test_resolve_task_cls_raises_import_error_for_missing_class(self) -> None:
        module_name = "tests.rhythm._tmp_run_task_module"
        module = types.ModuleType(module_name)
        sys.modules[module_name] = module
        try:
            with self.assertRaisesRegex(ImportError, "is not defined"):
                resolve_task_cls(f"{module_name}.FakeTask")
        finally:
            sys.modules.pop(module_name, None)

    def test_resolve_task_cls_returns_requested_class(self) -> None:
        module_name = "tests.rhythm._tmp_run_task_success"
        module = types.ModuleType(module_name)

        class FakeTask:
            pass

        module.FakeTask = FakeTask
        sys.modules[module_name] = module
        try:
            resolved = resolve_task_cls(f"{module_name}.FakeTask")
            self.assertIs(resolved, FakeTask)
        finally:
            sys.modules.pop(module_name, None)


if __name__ == "__main__":
    unittest.main()
