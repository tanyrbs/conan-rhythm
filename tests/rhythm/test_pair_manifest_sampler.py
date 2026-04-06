from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.tts.dataset_utils import BaseSpeechDataset
from utils.commons.hparams import hparams


class PairManifestSamplerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._hparams_backup = dict(hparams)

    def tearDown(self) -> None:
        hparams.clear()
        hparams.update(self._hparams_backup)

    @staticmethod
    def _build_items() -> list[dict]:
        def _item(name: str, *, spk_id: int) -> dict:
            return {
                "item_name": name,
                "mel": np.zeros((6, 80), dtype=np.float32),
                "spk_id": spk_id,
            }

        return [
            _item("A", spk_id=0),
            _item("B_fast", spk_id=0),
            _item("B_mid", spk_id=0),
            _item("B_slow", spk_id=0),
            _item("C", spk_id=1),
            _item("C_ref", spk_id=1),
        ]

    def _write_manifest(self, payload) -> str:
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(payload, handle)
        handle.flush()
        handle.close()
        return handle.name

    def test_pair_manifest_expands_fixed_pairs_and_marks_identity(self) -> None:
        manifest_path = self._write_manifest(
            {
                "train": [
                    {
                        "source": "A",
                        "refs": ["B_fast", "B_mid", "B_slow"],
                        "include_self": True,
                        "group_id": "group_A",
                    },
                    {
                        "source": "C",
                        "refs": ["C_ref"],
                        "include_self": True,
                        "group_id": "group_C",
                    },
                ]
            }
        )
        hparams.clear()
        hparams.update(
            {
                "binary_data_dir": ".",
                "test_ids": [],
                "min_frames": 0,
                "max_frames": 999,
                "frames_multiple": 1,
                "max_samples_per_spk": 16,
                "sort_by_len": True,
                "rhythm_pair_manifest_path": manifest_path,
                "rhythm_pair_manifest_prefixes": "train",
                "rhythm_pair_manifest_group_batches": True,
            }
        )
        dataset = BaseSpeechDataset("train", shuffle=False, items=self._build_items())
        self.assertEqual(len(dataset), 6)

        first = dataset[0]
        self.assertEqual(first["item_name"], "A")
        self.assertEqual(int(first["rhythm_pair_is_identity"].item()), 1)

        second = dataset[1]
        self.assertEqual(second["item_name"], "A")
        self.assertEqual(int(second["rhythm_pair_is_identity"].item()), 0)
        self.assertEqual(int(second["rhythm_pair_rank"].item()), 1)

        ordered = dataset.ordered_indices().tolist()
        self.assertEqual(ordered, [0, 1, 2, 3, 4, 5])

    def test_missing_manifest_item_raises_in_strict_mode(self) -> None:
        manifest_path = self._write_manifest({"train": [{"source": "A", "ref": "missing"}]})
        hparams.clear()
        hparams.update(
            {
                "binary_data_dir": ".",
                "test_ids": [],
                "min_frames": 0,
                "max_frames": 999,
                "frames_multiple": 1,
                "max_samples_per_spk": 16,
                "sort_by_len": True,
                "rhythm_pair_manifest_path": manifest_path,
                "rhythm_pair_manifest_prefixes": "train",
                "rhythm_pair_manifest_strict": True,
            }
        )
        with self.assertRaises(RuntimeError):
            BaseSpeechDataset("train", shuffle=False, items=self._build_items())


if __name__ == "__main__":
    unittest.main()
