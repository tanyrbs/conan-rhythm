import random
import numpy as np
import torch
from utils.commons.dataset_utils import BaseDataset, collate_1d_or_2d
from utils.commons.indexed_datasets import IndexedDataset
from utils.audio.pitch.utils import norm_interp_f0
from tasks.Conan.rhythm.pair_manifest import load_pair_manifest


class BaseSpeechDataset(BaseDataset):
    """Dataset that always draws a reference mel from the same speaker."""

    def __init__(self, prefix, shuffle=False, items=None, data_dir=None):
        super().__init__(shuffle)
        from utils.commons.hparams import hparams
        self.hparams = hparams
        self.data_dir = hparams['binary_data_dir'] if data_dir is None else data_dir
        self.prefix = prefix
        self.indexed_ds = None

        if items is not None:
            self.indexed_ds = items
            self.sizes = [1] * len(items)
            self.avail_idxs = list(range(len(self.sizes)))
        else:
            self.sizes = np.load(f"{self.data_dir}/{self.prefix}_lengths.npy")
            if prefix == 'test' and len(hparams['test_ids']) > 0:
                self.avail_idxs = hparams['test_ids']
            else:
                self.avail_idxs = list(range(len(self.sizes)))
            if prefix == 'train' and hparams['min_frames'] > 0:
                self.avail_idxs = [i for i in self.avail_idxs if self.sizes[i] >= hparams['min_frames']]
            self.sizes = [self.sizes[i] for i in self.avail_idxs]

        self.spk2indices = None
        self._spk_map_ready = False
        self._pair_entries = None
        self._pair_group_to_indices = None
        self._text_signature_cache = {}
        self._raw_item_cache = {}
        self._maybe_enable_pair_manifest()

    def _open_indexed_ds_if_needed(self):
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f"{self.data_dir}/{self.prefix}")

    @staticmethod
    def _normalize_manifest_prefixes(value) -> set[str]:
        if value is None:
            return {"train"}
        if isinstance(value, str):
            return {part.strip() for part in value.split(",") if part.strip()}
        if isinstance(value, (list, tuple, set)):
            return {str(part).strip() for part in value if str(part).strip()}
        return {str(value).strip()}

    def _should_use_pair_manifest(self) -> bool:
        path = self.hparams.get("rhythm_pair_manifest_path")
        if not path:
            return False
        enabled_prefixes = self._normalize_manifest_prefixes(
            self.hparams.get("rhythm_pair_manifest_prefixes", {"train"})
        )
        return self.prefix in enabled_prefixes

    def _item_name_to_local_index(self):
        self._open_indexed_ds_if_needed()
        mapping = {}
        for local_idx, global_idx in enumerate(self.avail_idxs):
            item = self.indexed_ds[global_idx]
            item_name = str(item["item_name"])
            if item_name in mapping:
                raise RuntimeError(
                    f"Pair manifest requires unique item_name values inside split '{self.prefix}', "
                    f"but found a duplicate: {item_name!r}."
                )
            mapping[item_name] = local_idx
        return mapping

    def _maybe_enable_pair_manifest(self):
        if not self._should_use_pair_manifest():
            return
        manifest_path = self.hparams.get("rhythm_pair_manifest_path")
        strict = bool(self.hparams.get("rhythm_pair_manifest_strict", True))
        item_name_to_local = self._item_name_to_local_index()
        raw_entries = load_pair_manifest(manifest_path, prefix=self.prefix)
        if not raw_entries:
            raise RuntimeError(
                f"Pair manifest '{manifest_path}' resolved no entries for split '{self.prefix}'."
            )
        base_sizes = list(self.sizes)
        pair_entries = []
        group_to_indices = {}
        group_to_numeric = {}
        next_group_id = 0
        for entry in raw_entries:
            source_name = entry["source_item_name"]
            ref_name = entry["ref_item_name"]
            target_name = entry.get("target_item_name")
            src_local = item_name_to_local.get(source_name)
            ref_local = item_name_to_local.get(ref_name)
            target_local = None if target_name in {None, ""} else item_name_to_local.get(str(target_name))
            if src_local is None or ref_local is None or (target_name not in {None, ""} and target_local is None):
                if strict:
                    raise RuntimeError(
                        f"Pair manifest entry ({source_name!r}, {ref_name!r}, target={target_name!r}) is missing from split '{self.prefix}' "
                        "after filtering. Rebuild the manifest or relax rhythm_pair_manifest_strict."
                    )
                continue
            group_label = str(entry.get("group_id", source_name))
            if group_label not in group_to_numeric:
                group_to_numeric[group_label] = next_group_id
                next_group_id += 1
            pair_index = len(pair_entries)
            numeric_group = group_to_numeric[group_label]
            pair_entries.append(
                {
                    "src_local": int(src_local),
                    "ref_local": int(ref_local),
                    "target_local": (None if target_local is None else int(target_local)),
                    "group_id": int(numeric_group),
                    "pair_rank": int(entry.get("pair_rank", 0)),
                    "is_identity": bool(src_local == ref_local),
                }
            )
            group_to_indices.setdefault(numeric_group, []).append(pair_index)
        if not pair_entries:
            raise RuntimeError(
                f"Pair manifest '{manifest_path}' yielded zero usable entries for split '{self.prefix}'."
            )
        self._pair_entries = pair_entries
        self._pair_group_to_indices = group_to_indices
        self.sizes = [base_sizes[entry["src_local"]] for entry in pair_entries]

    def _get_item(self, local_idx):
        global_idx = self.avail_idxs[local_idx] if self.avail_idxs is not None else local_idx
        self._open_indexed_ds_if_needed()
        return self.indexed_ds[global_idx]

    @staticmethod
    def _clone_cached_raw_item(item):
        return dict(item) if isinstance(item, dict) else item

    def _get_raw_item_cached(self, local_idx: int):
        local_idx = int(local_idx)
        cache = self._raw_item_cache
        item = cache.get(local_idx)
        if item is None:
            item = self._get_item(local_idx)
            cache[local_idx] = item
        return self._clone_cached_raw_item(item)

    @staticmethod
    def _coerce_optional_local_item_id(value) -> int | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() <= 0:
                return None
            value = value.reshape(-1)[0].item()
        elif isinstance(value, np.ndarray):
            if value.size <= 0:
                return None
            scalar = value.reshape(-1)[0]
            value = scalar.item() if hasattr(scalar, "item") else scalar
        elif isinstance(value, (list, tuple)):
            if not value:
                return None
            value = value[0]
        local_idx = int(value)
        return local_idx if local_idx >= 0 else None

    def _require_sample_local_item_id(self, sample) -> int:
        local_idx = self._coerce_optional_local_item_id(sample.get("item_id"))
        if local_idx is None:
            raise RuntimeError(
                "Dataset received no _raw_item from the upstream sample and no explicit item_id. "
                "Refusing to fall back to the dataset index because pair-manifest/filter/remap datasets "
                "can make the ordinal index differ from the source local item id."
            )
        return local_idx

    def _resolve_pair_entry(self, index):
        if self._pair_entries is None:
            return None
        return self._pair_entries[index]

    def _mel_to_tensor(self, mel, *, max_frames: int):
        spec = torch.as_tensor(mel, dtype=torch.float32)
        spec = spec[:max_frames]
        frames_multiple = int(self.hparams['frames_multiple'])
        max_frames = spec.shape[0] // frames_multiple * frames_multiple
        return spec[:max_frames]

    @staticmethod
    def _extract_text_signature(item):
        if not isinstance(item, dict):
            return None
        for key in ("ph_token", "txt_token", "txt_tokens", "word_token", "word_tokens"):
            value = item.get(key)
            if value is None:
                continue
            arr = np.asarray(value).reshape(-1)
            if arr.size > 0:
                return (key, tuple(arr.tolist()))
        for key in ("ph", "txt", "word", "words"):
            value = item.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return (key, text)
        return None

    def _get_text_signature(self, local_idx: int):
        local_idx = int(local_idx)
        if local_idx not in self._text_signature_cache:
            self._text_signature_cache[local_idx] = self._extract_text_signature(
                self._get_raw_item_cached(local_idx)
            )
        return self._text_signature_cache[local_idx]

    def _sample_reference_local_index(self, candidates, index: int, *, source_sig=None) -> int:
        if len(candidates) <= 1:
            return index
        strict_same_text_guard = False
        if "rhythm_v3_disallow_same_text_reference" in self.hparams:
            strict_same_text_guard = bool(self.hparams.get("rhythm_v3_disallow_same_text_reference", False))
        elif "rhythm_disallow_same_text_reference" in self.hparams:
            strict_same_text_guard = bool(self.hparams.get("rhythm_disallow_same_text_reference", False))
        if strict_same_text_guard:
            if source_sig is None:
                source_sig = self._get_text_signature(index)
            if source_sig is None:
                raise RuntimeError(
                    "rhythm_v3 strict reference sampling requires comparable text signatures for the source item."
                )
            filtered = []
            for cand in candidates:
                cand = int(cand)
                if cand == int(index):
                    continue
                cand_sig = self._get_text_signature(cand)
                if cand_sig is None:
                    raise RuntimeError(
                        "rhythm_v3 strict reference sampling requires comparable text signatures for reference candidates."
                    )
                if cand_sig != source_sig:
                    filtered.append(cand)
            if not filtered:
                raise RuntimeError(
                    "rhythm_v3 strict reference sampling could not find a different-text reference candidate."
                )
            candidates = filtered
        ref_local = random.choice(candidates)
        if ref_local != index:
            return ref_local
        alt_idx = random.randrange(len(candidates) - 1)
        if candidates[alt_idx] == index:
            alt_idx = len(candidates) - 1
        return int(candidates[alt_idx])

    def _build_speaker_map(self):
        if self._spk_map_ready:
            return

        import os
        from collections import defaultdict

        max_per_spk = int(self.hparams.get('max_samples_per_spk', 100))
        spk_ids_path = f"{self.data_dir}/{self.prefix}_spk_ids.npy"
        self.spk2indices = defaultdict(list)

        if os.path.exists(spk_ids_path):
            spk_ids = np.load(spk_ids_path, mmap_mode='r')
            local_spk_ids = spk_ids[self.avail_idxs]
            for local_idx in np.random.permutation(len(local_spk_ids)):
                sid = int(local_spk_ids[local_idx])
                bucket = self.spk2indices[sid]
                if len(bucket) < max_per_spk:
                    bucket.append(local_idx)
        else:
            for local_idx in np.random.permutation(len(self.avail_idxs)):
                sid = int(self._get_raw_item_cached(local_idx)['spk_id'])
                bucket = self.spk2indices[sid]
                if len(bucket) < max_per_spk:
                    bucket.append(local_idx)

        self._spk_map_ready = True

    def __getitem__(self, index):
        self._build_speaker_map()
        pair_entry = self._resolve_pair_entry(index)
        item_local = int(pair_entry["src_local"]) if pair_entry is not None else int(index)
        item = self._get_raw_item_cached(item_local)
        source_sig = self._extract_text_signature(item)
        self._text_signature_cache[int(item_local)] = source_sig
        hparams = self.hparams

        spec = self._mel_to_tensor(item['mel'], max_frames=hparams['max_frames'])

        spk_id = int(item['spk_id'])
        if pair_entry is not None:
            ref_local = int(pair_entry["ref_local"])
            paired_target_local = pair_entry.get("target_local")
        else:
            cand_locals = self.spk2indices[spk_id]
            ref_local = self._sample_reference_local_index(
                cand_locals,
                item_local,
                source_sig=source_sig,
            )
            paired_target_local = None
        ref_item = item if ref_local == item_local else self._get_raw_item_cached(ref_local)
        ref_spec = self._mel_to_tensor(ref_item['mel'], max_frames=hparams['max_frames'])
        paired_target_item = None
        if paired_target_local is not None:
            paired_target_local = int(paired_target_local)
            paired_target_item = (
                item if paired_target_local == item_local else self._get_raw_item_cached(paired_target_local)
            )

        sample = {
            'id': index,
            'item_name': item['item_name'],
            'item_id': item_local,
            'mel': spec,
            'mel_nonpadding': spec.abs().sum(-1) > 0,
            'ref_mel': ref_spec,
            'ref_item_id': ref_local,
            'paired_target_item_id': (-1 if paired_target_local is None else int(paired_target_local)),
            '_raw_item': item,
            '_raw_ref_item': ref_item,
        }
        if paired_target_item is not None:
            sample['_raw_paired_target_item'] = paired_target_item
        if pair_entry is not None:
            sample['rhythm_pair_group_id'] = torch.tensor([int(pair_entry["group_id"])], dtype=torch.long)
            sample['rhythm_pair_rank'] = torch.tensor([int(pair_entry["pair_rank"])], dtype=torch.long)
            sample['rhythm_pair_is_identity'] = torch.tensor(
                [1.0 if bool(pair_entry["is_identity"]) else 0.0],
                dtype=torch.float32,
            )
            sample['_rhythm_pair_manifest_entry'] = True

        if hparams.get('use_spk_embed', False):
            embed = item['spk_embed']
            if isinstance(embed, str):
                embed = torch.tensor([float(x) for x in embed.split()], dtype=torch.float32)
            else:
                embed = torch.as_tensor(embed, dtype=torch.float32)
            sample['spk_embed'] = embed

        if hparams.get('use_spk_id', False):
            sample['spk_id'] = spk_id

        return sample

    def collater(self, samples):
        if not samples:
            return {}
        hparams = self.hparams

        ids = torch.tensor([s['id'] for s in samples], dtype=torch.long)
        names = [s['item_name'] for s in samples]
        mels = collate_1d_or_2d([s['mel'] for s in samples], 0.0)
        ref_mels = collate_1d_or_2d([s['ref_mel'] for s in samples], 0.0)
        mel_lens = torch.tensor([s['mel'].shape[0] for s in samples], dtype=torch.long)
        ref_lens = torch.tensor([s['ref_mel'].shape[0] for s in samples], dtype=torch.long)

        batch = {
            'id': ids,
            'item_name': names,
            'nsamples': len(samples),
            'mels': mels,
            'mel_lengths': mel_lens,
            'ref_mels': ref_mels,
            'ref_mel_lengths': ref_lens,
        }

        if hparams.get('use_spk_embed', False):
            batch['spk_embed'] = torch.stack([s['spk_embed'] for s in samples])
        if hparams.get('use_spk_id', False):
            batch['spk_ids'] = torch.tensor([s['spk_id'] for s in samples], dtype=torch.long)

        return batch

    def ordered_indices(self):
        if self._pair_entries is None or not bool(self.hparams.get("rhythm_pair_manifest_group_batches", True)):
            return super().ordered_indices()
        group_to_indices = self._pair_group_to_indices or {}
        if not group_to_indices:
            return super().ordered_indices()
        group_ids = np.array(list(group_to_indices.keys()), dtype=np.int64)
        if self.shuffle:
            group_ids = np.random.permutation(group_ids)
            if self.sort_by_len:
                group_sizes = np.array([max(self._sizes[idx] for idx in group_to_indices[int(gid)]) for gid in group_ids])
                group_ids = group_ids[np.argsort(group_sizes, kind='mergesort')]
        indices = []
        for group_id in group_ids.tolist():
            indices.extend(group_to_indices[int(group_id)])
        return np.asarray(indices, dtype=np.int64)


class FastSpeechDataset(BaseSpeechDataset):
    """Dataset for FastSpeech-like models with ref mels & f0/uv."""

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        item = sample.get('_raw_item')
        if item is None:
            item = self._get_raw_item_cached(self._require_sample_local_item_id(sample))
        hparams = self.hparams

        if 'f0' in item:
            T = min(sample['mel'].shape[0], len(item['f0']))
        else:
            T = sample['mel'].shape[0]
        sample['mel'] = sample['mel'][:T]

        if hparams.get('use_pitch_embed', False):
            if 'f0' in item:
                f0, uv = norm_interp_f0(item['f0'][:T])
                sample['f0'] = torch.as_tensor(f0, dtype=torch.float32)
                sample['uv'] = torch.as_tensor(uv, dtype=torch.float32)
            else:
                sample['f0'], sample['uv'] = None, None
        else:
            sample['f0'], sample['uv'] = None, None

        return sample

    def collater(self, samples):
        batch = super().collater(samples)
        hparams = self.hparams
        if hparams.get('use_pitch_embed', False):
            batch['f0'] = collate_1d_or_2d([s['f0'] for s in samples], 0.0)
            batch['uv'] = collate_1d_or_2d([s['uv'] for s in samples])
        return batch


class FastSpeechWordDataset(FastSpeechDataset):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        item = sample.get('_raw_item')
        if item is None:
            item = self._get_raw_item_cached(self._require_sample_local_item_id(sample))
        max_frames = sample['mel'].shape[0]
        if 'word' in item:
            sample['words'] = item['word']
            sample['ph_words'] = item['ph_gb_word']
            sample['word_tokens'] = torch.as_tensor(item['word_token'], dtype=torch.long)
        else:
            sample['words'] = item['words']
            sample['ph_words'] = ' '.join(item['ph_words'])
            sample['word_tokens'] = torch.as_tensor(item['word_tokens'], dtype=torch.long)
        sample['mel2word'] = torch.as_tensor(item.get('mel2word'), dtype=torch.long)[:max_frames]
        sample['ph2word'] = torch.as_tensor(item['ph2word'][:self.hparams['max_input_tokens']], dtype=torch.long)
        return sample

    def collater(self, samples):
        batch = super().collater(samples)
        ph_words = [s['ph_words'] for s in samples]
        batch['ph_words'] = ph_words
        word_tokens = collate_1d_or_2d([s['word_tokens'] for s in samples], 0)
        batch['word_tokens'] = word_tokens
        mel2word = collate_1d_or_2d([s['mel2word'] for s in samples], 0)
        batch['mel2word'] = mel2word
        ph2word = collate_1d_or_2d([s['ph2word'] for s in samples], 0)
        batch['ph2word'] = ph2word
        batch['words'] = [s['words'] for s in samples]
        batch['word_lengths'] = torch.LongTensor([len(s['word_tokens']) for s in samples])
        if self.hparams['use_word_input']:
            batch['txt_tokens'] = batch['word_tokens']
            batch['txt_lengths'] = torch.LongTensor([s['word_tokens'].numel() for s in samples])
            batch['mel2ph'] = batch['mel2word']
        return batch
