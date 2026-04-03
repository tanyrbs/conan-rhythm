from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.commons.ddp_config import (
    load_ddp_logging_data,
    parse_ddp_mode,
    resolve_ddp_runtime_config,
    save_ddp_logging_data,
)


class DdpConfigTests(unittest.TestCase):
    def test_parse_ddp_mode_supports_auto_and_bool_strings(self) -> None:
        self.assertIsNone(parse_ddp_mode("auto"))
        self.assertIsNone(parse_ddp_mode("detect"))
        self.assertTrue(parse_ddp_mode("true"))
        self.assertFalse(parse_ddp_mode("false"))
        self.assertTrue(parse_ddp_mode(1))
        self.assertFalse(parse_ddp_mode(0))

    def test_auto_static_graph_uses_prior_logging_hint(self) -> None:
        find_unused, static_graph = resolve_ddp_runtime_config(
            "auto",
            "auto",
            ddp_logging_data={"can_set_static_graph": True},
        )
        self.assertFalse(find_unused)
        self.assertTrue(static_graph)

    def test_auto_defaults_preserve_safe_first_run_behavior(self) -> None:
        find_unused, static_graph = resolve_ddp_runtime_config(
            "auto",
            "auto",
            ddp_logging_data={},
        )
        self.assertTrue(find_unused)
        self.assertFalse(static_graph)

    def test_save_and_load_ddp_logging_data_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "ddp_logging_data.json"
            save_ddp_logging_data(
                path,
                {"can_set_static_graph": True, "bucket_sizes": [1, 2, 3]},
                global_step=123,
                epoch=4,
            )
            data = load_ddp_logging_data(path)
        self.assertTrue(data["can_set_static_graph"])
        self.assertEqual(data["bucket_sizes"], [1, 2, 3])
        self.assertEqual(data["saved_global_step"], 123)
        self.assertEqual(data["saved_epoch"], 4)


if __name__ == "__main__":
    unittest.main()
