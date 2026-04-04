from __future__ import annotations

import importlib
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class SingleThreadEnvTests(unittest.TestCase):
    def test_import_has_no_side_effect_without_env_toggle(self) -> None:
        module_name = "utils.commons.single_thread_env"
        original = {
            key: os.environ.get(key)
            for key in ("CONAN_SINGLE_THREAD_ENV", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        }
        os.environ.pop("CONAN_SINGLE_THREAD_ENV", None)
        os.environ.pop("OMP_NUM_THREADS", None)
        os.environ.pop("MKL_NUM_THREADS", None)
        try:
            sys.modules.pop(module_name, None)
            importlib.import_module(module_name)
            self.assertIsNone(os.environ.get("OMP_NUM_THREADS"))
            self.assertIsNone(os.environ.get("MKL_NUM_THREADS"))
        finally:
            for key, value in original.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_env_toggle_applies_requested_thread_count(self) -> None:
        module_name = "utils.commons.single_thread_env"
        original = {key: os.environ.get(key) for key in ("CONAN_SINGLE_THREAD_ENV", "OMP_NUM_THREADS")}
        try:
            os.environ["CONAN_SINGLE_THREAD_ENV"] = "3"
            os.environ.pop("OMP_NUM_THREADS", None)
            sys.modules.pop(module_name, None)
            module = importlib.import_module(module_name)
            applied = module.maybe_apply_single_thread_env_from_env(force=True)
            self.assertEqual(applied, 3)
            self.assertEqual(os.environ.get("OMP_NUM_THREADS"), "3")
        finally:
            for key, value in original.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
