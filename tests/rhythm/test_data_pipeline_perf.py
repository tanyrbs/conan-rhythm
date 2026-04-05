from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.Conan.rhythm.dataset_mixin import RhythmConanDatasetMixin
from tasks.tts.dataset_utils import FastSpeechDataset
from utils.commons.dataset_utils import BaseDataset, build_dataloader


class _TinyDataset(BaseDataset):
    def __init__(self, sizes, *, shuffle: bool):
        super().__init__(shuffle=shuffle)
        self.sizes = list(sizes)

    def __getitem__(self, index):
        return {"id": index}

    def collater(self, samples):
        return samples


class _CountingFastSpeechDataset(FastSpeechDataset):
    def __init__(self, items):
        self.get_item_calls = 0
        super().__init__(prefix="train", items=items, shuffle=False)

    def _get_item(self, local_idx):
        self.get_item_calls += 1
        return super()._get_item(local_idx)


class _DummyRhythmAssembler:
    def assemble(self, *, sample, item, ref_item, item_name: str):
        return {
            "item_name": item_name,
            "has_ref_item": ref_item is not None,
            "sample_keys": tuple(sorted(sample.keys())),
        }


class _CountingRhythmDataset(RhythmConanDatasetMixin, FastSpeechDataset):
    def __init__(self, items):
        self.get_item_calls = 0
        super().__init__(prefix="train", items=items, shuffle=False)

    def _get_item(self, local_idx):
        self.get_item_calls += 1
        return super()._get_item(local_idx)

    def _materialize_rhythm_cache_compat(self, item, *, item_name: str):
        return item

    def _resolve_rhythm_target_mode(self) -> str:
        return "runtime_only"

    def _rhythm_sample_assembler(self):
        return _DummyRhythmAssembler()


class DataPipelinePerfTests(unittest.TestCase):
    def test_endless_batch_sampler_cycles_without_materializing_huge_repeat_lists(self) -> None:
        with mock.patch.dict(
            "utils.commons.hparams.hparams",
            {
                "sort_by_len": False,
                "max_frames": 32,
                "ds_workers": 0,
                "seed": 1234,
                "dl_pin_memory": False,
                "dl_persistent_workers": False,
                "dl_prefetch_factor": 2,
            },
            clear=True,
        ):
            dataset = _TinyDataset([1, 1, 1, 1], shuffle=True)
            loader = build_dataloader(
                dataset,
                shuffle=True,
                max_sentences=1,
                endless=True,
                apply_batch_by_size=False,
            )
            sampler = loader.batch_sampler
            self.assertTrue(hasattr(sampler, "set_epoch"))
            cycle_len = len(sampler)
            iterator = iter(sampler)
            first_cycle = [tuple(next(iterator)) for _ in range(cycle_len)]
            second_cycle = [tuple(next(iterator)) for _ in range(cycle_len)]
            self.assertEqual(cycle_len, 4)
            self.assertNotEqual(first_cycle, second_cycle)

    def test_fastspeech_dataset_reuses_loaded_items(self) -> None:
        items = [
            {"item_name": "spk0_a", "spk_id": 0, "mel": torch.ones((4, 3)).numpy(), "f0": torch.tensor([1, 2, 3, 4]).numpy()},
            {"item_name": "spk0_b", "spk_id": 0, "mel": torch.ones((5, 3)).numpy(), "f0": torch.tensor([1, 2, 3, 4, 5]).numpy()},
        ]
        with mock.patch.dict(
            "utils.commons.hparams.hparams",
            {
                "sort_by_len": False,
                "max_frames": 16,
                "ds_workers": 0,
                "binary_data_dir": "",
                "frames_multiple": 1,
                "test_ids": [],
                "min_frames": 0,
                "use_spk_id": False,
                "use_spk_embed": False,
                "use_pitch_embed": True,
                "max_input_tokens": 16,
            },
            clear=True,
        ):
            dataset = _CountingFastSpeechDataset(items)
            dataset.spk2indices = {0: [0, 1]}
            dataset._spk_map_ready = True
            sample = dataset[0]
            self.assertEqual(dataset.get_item_calls, 2)
            self.assertIn("_raw_item", sample)
            self.assertIn("_raw_ref_item", sample)

    def test_rhythm_dataset_reuses_raw_items_from_basespeech(self) -> None:
        items = [
            {"item_name": "spk0_a", "spk_id": 0, "mel": torch.ones((4, 3)).numpy(), "f0": torch.tensor([1, 2, 3, 4]).numpy()},
            {"item_name": "spk0_b", "spk_id": 0, "mel": torch.ones((5, 3)).numpy(), "f0": torch.tensor([1, 2, 3, 4, 5]).numpy()},
        ]
        with mock.patch.dict(
            "utils.commons.hparams.hparams",
            {
                "sort_by_len": False,
                "max_frames": 16,
                "ds_workers": 0,
                "binary_data_dir": "",
                "frames_multiple": 1,
                "test_ids": [],
                "min_frames": 0,
                "use_spk_id": False,
                "use_spk_embed": False,
                "use_pitch_embed": True,
                "max_input_tokens": 16,
            },
            clear=True,
        ):
            dataset = _CountingRhythmDataset(items)
            dataset.spk2indices = {0: [0, 1]}
            dataset._spk_map_ready = True
            sample = dataset[0]
            self.assertEqual(dataset.get_item_calls, 2)
            self.assertTrue(sample["has_ref_item"])
            self.assertNotIn("_raw_item", sample["sample_keys"])
            self.assertNotIn("_raw_ref_item", sample["sample_keys"])

    def test_rhythm_dataset_collater_emits_explicit_content_lengths(self) -> None:
        with mock.patch.dict(
            "utils.commons.hparams.hparams",
            {
                "sort_by_len": False,
                "max_frames": 16,
                "ds_workers": 0,
                "binary_data_dir": "",
                "frames_multiple": 1,
                "test_ids": [],
                "min_frames": 0,
                "use_spk_id": False,
                "use_spk_embed": False,
                "use_pitch_embed": True,
                "max_input_tokens": 16,
            },
            clear=True,
        ):
            dataset = _CountingRhythmDataset([])
            sample_a = {
                "id": 0,
                "item_name": "spk0_a",
                "mel": torch.ones((4, 3), dtype=torch.float32),
                "ref_mel": torch.ones((4, 3), dtype=torch.float32),
                "content": torch.tensor([1, 2, 3, 4], dtype=torch.long),
                "f0": torch.ones((4,), dtype=torch.float32),
                "uv": torch.zeros((4,), dtype=torch.float32),
            }
            sample_b = {
                "id": 1,
                "item_name": "spk0_b",
                "mel": torch.ones((5, 3), dtype=torch.float32),
                "ref_mel": torch.ones((5, 3), dtype=torch.float32),
                "content": torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
                "f0": torch.ones((5,), dtype=torch.float32),
                "uv": torch.zeros((5,), dtype=torch.float32),
            }
            batch = dataset.collater([sample_a, sample_b])
        self.assertIn("content_lengths", batch)
        self.assertTrue(torch.equal(batch["content_lengths"], torch.tensor([4, 5], dtype=torch.long)))
        self.assertEqual(batch["content"].shape[:2], (2, 5))


if __name__ == "__main__":
    unittest.main()
