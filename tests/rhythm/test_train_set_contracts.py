from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.Conan.rhythm.preflight_support import _inspect_train_set_data_staging
from utils.commons.train_set_contracts import (
    collect_condition_map_issues,
    collect_shared_json_artifact_issues,
    normalize_train_set_dirs,
)


class TrainSetContractTests(unittest.TestCase):
    def _write_json(self, path: Path, payload) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def test_normalize_train_set_dirs_filters_empty_entries(self) -> None:
        self.assertEqual(
            normalize_train_set_dirs("foo|| bar |"),
            ["foo", "bar"],
        )

    def test_shared_json_artifact_check_flags_missing_base_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            base_dir = tmp_dir / "base"
            train_dir = tmp_dir / "train_a"
            base_dir.mkdir()
            train_dir.mkdir()
            self._write_json(train_dir / "phone_set.json", ["a", "b"])
            issues = collect_shared_json_artifact_issues(str(base_dir), [str(train_dir)])
        self.assertTrue(any("phone_set.json" in issue and "missing from binary_data_dir" in issue for issue in issues))

    def test_condition_map_check_accepts_equivalent_set_and_map_encodings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            base_dir = tmp_dir / "base"
            train_dir = tmp_dir / "train_a"
            base_dir.mkdir()
            train_dir.mkdir()
            self._write_json(base_dir / "style_set.json", ["neutral", "happy"])
            self._write_json(train_dir / "style_map.json", {"neutral": 0, "happy": 1})
            issues = collect_condition_map_issues(str(base_dir), [str(train_dir)])
        self.assertEqual(issues, [])

    def test_condition_map_check_flags_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            base_dir = tmp_dir / "base"
            train_dir = tmp_dir / "train_a"
            base_dir.mkdir()
            train_dir.mkdir()
            self._write_json(base_dir / "style_set.json", ["neutral", "happy"])
            self._write_json(train_dir / "style_map.json", {"neutral": 0, "sad": 1})
            issues = collect_condition_map_issues(str(base_dir), [str(train_dir)])
        self.assertTrue(any("style condition map" in issue for issue in issues))

    def test_preflight_train_set_checks_cover_train_sidecars_and_condition_maps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            base_dir = tmp_dir / "base"
            train_dir = tmp_dir / "train_a"
            base_dir.mkdir()
            train_dir.mkdir()
            self._write_json(base_dir / "style_set.json", ["neutral", "happy"])
            self._write_json(train_dir / "style_map.json", {"neutral": 0, "sad": 1})
            issues = _inspect_train_set_data_staging(
                str(base_dir),
                train_set_dirs=[str(train_dir)],
            )
        self.assertTrue(any("Missing indexed dataset for train_set" in issue for issue in issues))
        self.assertTrue(any("style condition map" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
