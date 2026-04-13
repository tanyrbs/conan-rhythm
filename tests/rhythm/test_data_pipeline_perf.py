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
from tasks.tts.dataset_utils import BaseSpeechDataset, FastSpeechDataset, FastSpeechWordDataset
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


class _CountingFastSpeechWordDataset(FastSpeechWordDataset):
    def __init__(self, items):
        self.get_item_calls = 0
        super().__init__(prefix="train", items=items, shuffle=False)

    def _get_item(self, local_idx):
        self.get_item_calls += 1
        return super()._get_item(local_idx)


class _CountingRawCacheFastSpeechDataset(FastSpeechDataset):
    def __init__(self, items):
        self.raw_cache_calls = 0
        super().__init__(prefix="train", items=items, shuffle=False)

    def _get_raw_item_cached(self, local_idx):
        self.raw_cache_calls += 1
        return super()._get_raw_item_cached(local_idx)


class _DummyRhythmAssembler:
    def __init__(self, owner=None):
        self.owner = owner

    def assemble(self, *, sample, item, ref_item, item_name: str):
        paired_target_item = None
        if self.owner is not None:
            paired_target_item = self.owner._resolve_paired_target_rhythm_item(
                sample=sample,
                item=item,
                target_mode=self.owner._resolve_rhythm_target_mode(),
            )
        return {
            "item_name": item_name,
            "ref_item_name": None if ref_item is None else ref_item.get("item_name"),
            "paired_target_item_name": None if paired_target_item is None else paired_target_item.get("item_name"),
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


class _LocalIdFallbackBase:
    def __init__(self, *, items, samples):
        self.items = list(items)
        self.samples = list(samples)
        self.get_item_calls = []
        self.prefix = "train"
        self.hparams = {
            "rhythm_cached_reference_policy": "sample_ref",
        }

    def __getitem__(self, index):
        return dict(self.samples[index])

    def _get_item(self, local_idx):
        self.get_item_calls.append(int(local_idx))
        return self.items[int(local_idx)]


class _LocalIdFallbackRhythmDataset(RhythmConanDatasetMixin, _LocalIdFallbackBase):
    def _materialize_rhythm_cache_compat(self, item, *, item_name: str):
        return item

    def _resolve_rhythm_target_mode(self) -> str:
        return "runtime_only"

    def _rhythm_sample_assembler(self):
        return _DummyRhythmAssembler(owner=self)


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
            self.assertEqual(sample["item_id"], 0)
            self.assertEqual(sample["ref_item_id"], 1)
            self.assertIn("_raw_item", sample)
            self.assertIn("_raw_ref_item", sample)

    def test_fastspeech_dataset_reuses_source_text_signature_without_extra_cache_lookup(self) -> None:
        items = [
            {
                "item_name": "spk0_a",
                "spk_id": 0,
                "mel": torch.ones((4, 3)).numpy(),
                "f0": torch.tensor([1, 2, 3, 4]).numpy(),
                "ph_token": torch.tensor([11, 12], dtype=torch.long).numpy(),
            },
            {
                "item_name": "spk0_b",
                "spk_id": 0,
                "mel": torch.ones((5, 3)).numpy(),
                "f0": torch.tensor([1, 2, 3, 4, 5]).numpy(),
                "ph_token": torch.tensor([21, 22], dtype=torch.long).numpy(),
            },
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
                "rhythm_v3_disallow_same_text_reference": True,
            },
            clear=True,
        ):
            dataset = _CountingRawCacheFastSpeechDataset(items)
            dataset.spk2indices = {0: [0, 1]}
            dataset._spk_map_ready = True
            sample = dataset[0]
            self.assertEqual(sample["ref_item_id"], 1)
            self.assertEqual(dataset.raw_cache_calls, 3)

    def test_fastspeech_dataset_fallback_uses_item_id_for_remapped_manifest_sample(self) -> None:
        items = [
            {"item_name": "spk0_a", "spk_id": 0, "mel": torch.ones((4, 3)).numpy(), "f0": torch.tensor([10, 11, 12, 13]).numpy()},
            {"item_name": "spk0_b", "spk_id": 0, "mel": torch.ones((5, 3)).numpy(), "f0": torch.tensor([20, 21, 22, 23, 24]).numpy()},
        ]

        original_getitem = BaseSpeechDataset.__getitem__

        def _getitem_without_raw_item(self, index):
            sample = original_getitem(self, index)
            sample.pop("_raw_item", None)
            return sample

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
            dataset._pair_entries = [
                {"src_local": 1, "ref_local": 0, "target_local": None, "group_id": 0, "pair_rank": 0, "is_identity": False}
            ]
            dataset.sizes = [5]
            dataset.spk2indices = {0: [0, 1]}
            dataset._spk_map_ready = True
            with mock.patch.object(BaseSpeechDataset, "__getitem__", _getitem_without_raw_item):
                sample = dataset[0]
            self.assertEqual(dataset.get_item_calls, 2)
            self.assertEqual(sample["item_id"], 1)
            self.assertEqual(sample["ref_item_id"], 0)
            self.assertEqual(sample["mel"].shape[0], 5)
            self.assertEqual(sample["f0"].shape[0], 5)
            self.assertEqual(sample["uv"].shape[0], 5)

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

    def test_fastspeech_word_dataset_fallback_uses_item_id_for_remapped_manifest_sample(self) -> None:
        items = [
            {
                "item_name": "spk0_a",
                "spk_id": 0,
                "mel": torch.ones((4, 3)).numpy(),
                "f0": torch.tensor([1, 2, 3, 4]).numpy(),
                "words": ["alpha"],
                "ph_words": ["AA"],
                "word_tokens": torch.tensor([101], dtype=torch.long).numpy(),
                "mel2word": torch.tensor([1, 1, 1, 1], dtype=torch.long).numpy(),
                "ph2word": torch.tensor([1], dtype=torch.long).numpy(),
            },
            {
                "item_name": "spk0_b",
                "spk_id": 0,
                "mel": torch.ones((5, 3)).numpy(),
                "f0": torch.tensor([5, 6, 7, 8, 9]).numpy(),
                "words": ["beta", "gamma"],
                "ph_words": ["BB", "CC"],
                "word_tokens": torch.tensor([201, 202], dtype=torch.long).numpy(),
                "mel2word": torch.tensor([1, 1, 2, 2, 2], dtype=torch.long).numpy(),
                "ph2word": torch.tensor([1, 2], dtype=torch.long).numpy(),
            },
        ]

        original_getitem = FastSpeechDataset.__getitem__

        def _getitem_without_raw_item(self, index):
            sample = original_getitem(self, index)
            sample.pop("_raw_item", None)
            return sample

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
                "use_word_input": False,
            },
            clear=True,
        ):
            dataset = _CountingFastSpeechWordDataset(items)
            dataset._pair_entries = [
                {"src_local": 1, "ref_local": 0, "target_local": None, "group_id": 0, "pair_rank": 0, "is_identity": False}
            ]
            dataset.sizes = [5]
            dataset.spk2indices = {0: [0, 1]}
            dataset._spk_map_ready = True
            with mock.patch.object(FastSpeechDataset, "__getitem__", _getitem_without_raw_item):
                sample = dataset[0]
            self.assertEqual(dataset.get_item_calls, 2)
            self.assertEqual(sample["item_id"], 1)
            self.assertEqual(sample["words"], ["beta", "gamma"])
            self.assertTrue(torch.equal(sample["word_tokens"], torch.tensor([201, 202], dtype=torch.long)))
            self.assertTrue(torch.equal(sample["mel2word"], torch.tensor([1, 1, 2, 2, 2], dtype=torch.long)))

    def test_rhythm_dataset_fallback_uses_explicit_source_local_id(self) -> None:
        items = [
            {"item_name": "wrong_index_0", "spk_id": 0},
            {"item_name": "wrong_index_1", "spk_id": 0},
            {"item_name": "source_local_2", "spk_id": 0},
        ]
        dataset = _LocalIdFallbackRhythmDataset(
            items=items,
            samples=[
                {
                    "id": 0,
                    "item_id": 2,
                    "ref_item_id": -1,
                }
            ],
        )

        sample = dataset[0]

        self.assertEqual(sample["item_name"], "source_local_2")
        self.assertEqual(dataset.get_item_calls, [2])

    def test_rhythm_dataset_fallback_requires_explicit_source_local_id(self) -> None:
        dataset = _LocalIdFallbackRhythmDataset(
            items=[{"item_name": "source", "spk_id": 0}],
            samples=[
                {
                    "id": 0,
                    "ref_item_id": -1,
                }
            ],
        )

        with self.assertRaisesRegex(RuntimeError, "no explicit item_id"):
            _ = dataset[0]

        self.assertEqual(dataset.get_item_calls, [])

    def test_rhythm_dataset_ref_and_paired_fallback_reuse_raw_item_cache(self) -> None:
        items = [
            {"item_name": "source_local_0", "spk_id": 0},
            {"item_name": "source_local_1", "spk_id": 0},
            {"item_name": "shared_ref_target_local_2", "spk_id": 0},
        ]
        dataset = _LocalIdFallbackRhythmDataset(
            items=items,
            samples=[
                {
                    "id": 0,
                    "item_id": 1,
                    "ref_item_id": 2,
                    "paired_target_item_id": 2,
                }
            ],
        )

        sample = dataset[0]

        self.assertEqual(sample["item_name"], "source_local_1")
        self.assertEqual(sample["ref_item_name"], "shared_ref_target_local_2")
        self.assertEqual(sample["paired_target_item_name"], "shared_ref_target_local_2")
        self.assertEqual(dataset.get_item_calls, [1, 2])

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
