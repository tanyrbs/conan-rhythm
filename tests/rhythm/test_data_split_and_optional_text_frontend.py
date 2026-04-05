from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_gen.conan_binarizer import BinarizationError, ConanBinarizer
from scripts.build_libritts_local_processed_metadata import build_argparser, _normalize_split_arg_list


def _pop_modules(prefix: str) -> dict[str, object]:
    removed = {}
    for name in list(sys.modules):
        if name == prefix or name.startswith(f"{prefix}."):
            removed[name] = sys.modules.pop(name)
    return removed


def _restore_modules(prefix: str, removed: dict[str, object]) -> None:
    for name in list(sys.modules):
        if name == prefix or name.startswith(f"{prefix}."):
            sys.modules.pop(name, None)
    sys.modules.update(removed)


class OptionalTextFrontendGuardTests(unittest.TestCase):
    def test_tts_utils_import_does_not_require_g2p_frontend(self) -> None:
        removed_txt = _pop_modules("data_gen.tts.txt_processors")
        removed_utils = _pop_modules("tasks.tts.tts_utils")
        real_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "g2p_en" or name.startswith("g2p_en."):
                raise ModuleNotFoundError("No module named 'g2p_en'")
            return real_import(name, globals, locals, fromlist, level)

        try:
            with mock.patch("builtins.__import__", side_effect=fake_import):
                module = importlib.import_module("tasks.tts.tts_utils")
            self.assertTrue(hasattr(module, "load_data_binarizer"))
        finally:
            _restore_modules("tasks.tts.tts_utils", removed_utils)
            _restore_modules("data_gen.tts.txt_processors", removed_txt)


class ConanBinarizerSplitTests(unittest.TestCase):
    def _write_json(self, path: Path, payload) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def test_load_meta_data_prefers_explicit_split_labels_from_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            processed_dir = Path(tmp)
            self._write_json(
                processed_dir / "metadata.json",
                [
                    {"item_name": "123_trainclean100_1_0001", "split": "train-clean-100"},
                    {"item_name": "456_devclean_2_0001", "split": "dev-clean"},
                    {"item_name": "789_testclean_3_0001", "split": "test-clean"},
                ],
            )
            self._write_json(
                processed_dir / "build_summary.json",
                {
                    "split_tags": {
                        "train": "trainclean100",
                        "valid": "devclean",
                        "test": "testclean",
                    }
                },
            )
            with mock.patch.dict(
                "data_gen.conan_binarizer.hparams",
                {
                    "processed_data_dir": str(processed_dir),
                    "binarization_args": {"shuffle": False},
                    "valid_prefixes": ["p231", "p334"],
                    "test_prefixes": ["p231", "p334"],
                },
                clear=True,
            ):
                binarizer = ConanBinarizer(processed_data_dir=str(processed_dir))
                binarizer.load_meta_data()
            self.assertEqual(binarizer.train_item_names, ["123_trainclean100_1_0001"])
            self.assertEqual(binarizer.valid_item_names, ["456_devclean_2_0001"])
            self.assertEqual(binarizer.test_item_names, ["789_testclean_3_0001"])

    def test_load_meta_data_rejects_partial_explicit_split_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            processed_dir = Path(tmp)
            self._write_json(
                processed_dir / "metadata.json",
                [
                    {"item_name": "123_trainclean100_1_0001", "split": "train-clean-100"},
                    {"item_name": "456_devclean_2_0001"},
                ],
            )
            with mock.patch.dict(
                "data_gen.conan_binarizer.hparams",
                {
                    "processed_data_dir": str(processed_dir),
                    "binarization_args": {"shuffle": False},
                    "valid_prefixes": ["p231"],
                    "test_prefixes": ["p334"],
                },
                clear=True,
            ):
                binarizer = ConanBinarizer(processed_data_dir=str(processed_dir))
                with self.assertRaisesRegex(BinarizationError, "Only part of the metadata carries explicit split labels"):
                    binarizer.load_meta_data()

    def test_load_meta_data_rejects_overlapping_prefix_split_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            processed_dir = Path(tmp)
            self._write_json(
                processed_dir / "metadata.json",
                [
                    {"item_name": "p231_item_a"},
                    {"item_name": "p334_item_b"},
                    {"item_name": "p500_item_c"},
                ],
            )
            with mock.patch.dict(
                "data_gen.conan_binarizer.hparams",
                {
                    "processed_data_dir": str(processed_dir),
                    "binarization_args": {"shuffle": False},
                    "valid_prefixes": ["p231", "p334"],
                    "test_prefixes": ["p231"],
                },
                clear=True,
            ):
                binarizer = ConanBinarizer(processed_data_dir=str(processed_dir))
                with self.assertRaisesRegex(BinarizationError, "overlap"):
                    binarizer.load_meta_data()


class MetadataScriptDefaultsTests(unittest.TestCase):
    def test_build_libritts_metadata_defaults_to_full_split(self) -> None:
        args = build_argparser().parse_args(
            [
                "--raw_root",
                "dummy_raw",
                "--processed_data_dir",
                "dummy_processed",
            ]
        )
        self.assertEqual(args.train_limit, 0)
        self.assertEqual(args.valid_limit, 0)
        self.assertEqual(args.test_limit, 0)

    def test_build_libritts_metadata_accepts_multiple_train_splits(self) -> None:
        args = build_argparser().parse_args(
            [
                "--raw_root",
                "dummy_raw",
                "--processed_data_dir",
                "dummy_processed",
                "--train_split",
                "train-clean-100",
                "--train_split",
                "train-clean-360,train-other-500",
            ]
        )
        self.assertEqual(
            args.train_split,
            ["train-clean-100", "train-clean-360,train-other-500"],
        )

    def test_normalize_split_arg_list_dedupes_and_preserves_order(self) -> None:
        normalized = _normalize_split_arg_list(
            ["train-clean-100", "train-clean-360, train-clean-100", "train-other-500"],
            default="train-clean-100",
        )
        self.assertEqual(
            normalized,
            ["train-clean-100", "train-clean-360", "train-other-500"],
        )


if __name__ == "__main__":
    unittest.main()
