import json
import os
import random
from collections import defaultdict

import numpy as np
import torch

from tasks.Conan.rhythm.pair_manifest import load_pair_manifest
from utils.audio.pitch.utils import norm_interp_f0
from utils.commons.dataset_utils import BaseDataset, collate_1d_or_2d
from utils.commons.indexed_datasets import IndexedDataset


class BaseSpeechDataset(BaseDataset):
    """Dataset that always draws a reference mel from the same speaker.

    When ``rhythm_pair_manifest`` is configured for the train split, the dataset
    switches from random same-speaker reference sampling to explicit pairwise
    entries ``(A, B)``. This is the minimal hook needed for pairwise bootstrap
    / same-A multi-B experiments without changing the rest of the data pipeline.
    """

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

        self._base_avail_idxs = list(self.avail_idxs) if self.avail_idxs is not None else None
        self._base_sizes = list(self.sizes) if self.sizes is not None else None
        self.spk2indices = None
        self._spk_map_ready = False
        self._local_spk_ids = None
        self._pair_entries = None
        self._pair_group_to_indices = None
        self._pair_group_order = None
        self._pair_manifest_ready = False
        self._maybe_enable_pair_manifest()

    def _open_indexed_ds_if_needed(self):
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f"{self.data_dir}/{self.prefix}")

    def _resolve_global_idx(self, local_idx, *, use_base=False):
        avail = self._base_avail_idxs if use_base else self.avail_idxs
        if avail is None:
            return int(local_idx)
        return int(avail[local_idx])

    def _get_item(self, local_idx):
        global_idx = self._resolve_global_idx(local_idx, use_base=False)
        self._open_indexed_ds_if_needed()
        return self.indexed_ds[global_idx]

    def _get_base_item(self, local_idx):
        global_idx = self._resolve_global_idx(local_idx, use_base=True)
        self._open_indexed_ds_if_needed()
        return self.indexed_ds[global_idx]

    def _mel_to_tensor(self, mel, *, max_frames: int):
        spec = torch.as_tensor(mel, dtype=torch.float32)
        spec = spec[:max_frames]
        frames_multiple = int(self.hparams['frames_multiple'])
        max_frames = spec.shape[0] // frames_multiple * frames_multiple
        return spec[:max_frames]

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
        path = str(
            self.hparams.get(
                'rhythm_pair_manifest',
                self.hparams.get(
                    'pair_manifest',
                    self.hparams.get('rhythm_pair_manifest_path', ''),
                ),
            )
            or ''
        ).strip()
        if not path:
            return False
        enabled_prefixes = self._normalize_manifest_prefixes(
            self.hparams.get("rhythm_pair_manifest_prefixes", {"train"})
        )
        return self.prefix in enabled_prefixes

    def _resolve_pair_manifest_path(self) -> str:
        path = str(
            self.hparams.get(
                'rhythm_pair_manifest',
                self.hparams.get(
                    'pair_manifest',
                    self.hparams.get('rhythm_pair_manifest_path', ''),
                ),
            )
            or ''
        ).strip()
        if not path or not self._should_use_pair_manifest():
            return ''
        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)
        return path

    @staticmethod
    def _load_legacy_pair_manifest_records(path: str):
        if path.endswith('.jsonl'):
            with open(path, 'r', encoding='utf-8') as f:
                return [json.loads(line) for line in f if line.strip()]
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            for key in ('pairs', 'entries', 'data'):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
            return []
        if isinstance(payload, list):
            return payload
        return []

    def _resolve_local_spk_ids(self):
        if self._local_spk_ids is not None:
            return self._local_spk_ids
        spk_ids_path = f"{self.data_dir}/{self.prefix}_spk_ids.npy"
        avail = self._base_avail_idxs if self._base_avail_idxs is not None else self.avail_idxs
        if os.path.exists(spk_ids_path):
            spk_ids = np.load(spk_ids_path, mmap_mode='r')
            self._local_spk_ids = np.asarray(spk_ids[avail], dtype=np.int64)
            return self._local_spk_ids
        base_sizes = self._base_sizes if self._base_sizes is not None else self.sizes
        self._local_spk_ids = np.asarray(
            [int(self._get_base_item(local_idx)['spk_id']) for local_idx in range(len(base_sizes))],
            dtype=np.int64,
        )
        return self._local_spk_ids

    def _build_item_name_to_local(self):
        mapping = {}
        base_sizes = self._base_sizes if self._base_sizes is not None else self.sizes
        for local_idx in range(len(base_sizes)):
            item = self._get_base_item(local_idx)
            item_name = str(item['item_name'])
            if item_name in mapping:
                raise RuntimeError(
                    f"Pair manifest requires unique item_name values inside split '{self.prefix}', "
                    f"but found duplicate {item_name!r} after filtering."
                )
            mapping[item_name] = int(local_idx)
        return mapping

    @staticmethod
    def _coerce_manifest_int(entry, keys):
        for key in keys:
            value = entry.get(key)
            if value in {None, ''}:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _coerce_manifest_name(entry, keys):
        for key in keys:
            value = entry.get(key)
            if value in {None, ''}:
                continue
            return str(value)
        return None

    def _maybe_enable_pair_manifest(self):
        manifest_path = self._resolve_pair_manifest_path()
        if not manifest_path:
            return
        strict = bool(self.hparams.get('rhythm_pair_manifest_strict', True))
        allow_legacy = bool(self.hparams.get('rhythm_pair_manifest_allow_legacy', True))
        try:
            records = load_pair_manifest(manifest_path, prefix=self.prefix)
        except Exception as exc:
            if not allow_legacy:
                raise
            records = self._load_legacy_pair_manifest_records(manifest_path)
            if len(records) <= 0:
                raise RuntimeError(
                    f"Pair manifest '{manifest_path}' could not be parsed for split '{self.prefix}'."
                ) from exc
        if len(records) <= 0:
            raise RuntimeError(f"Pair manifest is empty or unusable for split '{self.prefix}': {manifest_path}")

        item_name_to_local = None
        local_spk_ids = None
        require_same_speaker = bool(self.hparams.get('rhythm_pair_manifest_require_same_speaker', True))
        group_key_to_id = {}
        pair_entries = []
        base_sizes = self._base_sizes if self._base_sizes is not None else self.sizes
        base_len = len(base_sizes)

        for record_idx, raw_entry in enumerate(records):
            if not isinstance(raw_entry, dict):
                if strict:
                    raise RuntimeError(f"Pair manifest entry {record_idx} is not a dict.")
                continue
            src_local = self._coerce_manifest_int(
                raw_entry,
                ('src_local_id', 'source_local_id', 'src_index', 'source_index'),
            )
            ref_local = self._coerce_manifest_int(
                raw_entry,
                ('ref_local_id', 'reference_local_id', 'ref_index', 'reference_index'),
            )
            src_name = None
            ref_name = None
            if src_local is None or ref_local is None:
                if item_name_to_local is None:
                    item_name_to_local = self._build_item_name_to_local()
                if src_local is None:
                    src_name = self._coerce_manifest_name(
                        raw_entry,
                        ('src_item_name', 'source_item_name', 'item_name', 'src_name', 'source_name'),
                    )
                    src_local = item_name_to_local.get(src_name)
                if ref_local is None:
                    ref_name = self._coerce_manifest_name(
                        raw_entry,
                        ('ref_item_name', 'reference_item_name', 'ref_name', 'reference_name'),
                    )
                    ref_local = item_name_to_local.get(ref_name)
            if src_local is None or ref_local is None:
                if strict:
                    raise RuntimeError(
                        f"Pair manifest entry {record_idx} could not resolve source/reference item ids."
                    )
                continue
            src_local = int(src_local)
            ref_local = int(ref_local)
            if not (0 <= src_local < base_len and 0 <= ref_local < base_len):
                if strict:
                    raise RuntimeError(
                        f"Pair manifest entry {record_idx} out of range: src={src_local}, ref={ref_local}, size={base_len}."
                    )
                continue
            if require_same_speaker:
                if local_spk_ids is None:
                    local_spk_ids = self._resolve_local_spk_ids()
                if int(local_spk_ids[src_local]) != int(local_spk_ids[ref_local]):
                    raise RuntimeError(
                        f"Pair manifest entry {record_idx} is not same-speaker: src={src_local}, ref={ref_local}."
                    )
            group_key = raw_entry.get('group_id', None)
            if group_key in {None, ''}:
                group_key = src_name if src_name is not None else f"src:{src_local}"
            group_key = str(group_key)
            if group_key not in group_key_to_id:
                group_key_to_id[group_key] = len(group_key_to_id)
            group_slot = self._coerce_manifest_int(raw_entry, ('group_slot', 'slot', 'pair_slot', 'pair_rank'))
            if group_slot is None:
                group_slot = sum(1 for entry in pair_entries if entry['group_key'] == group_key)
            pair_weight = float(raw_entry.get('pair_weight', 1.0) or 1.0)
            pair_is_identity = float(bool(raw_entry.get('identity_anchor', False) or src_local == ref_local))
            pair_entries.append(
                {
                    'src_local': src_local,
                    'ref_local': ref_local,
                    'group_id': int(group_key_to_id[group_key]),
                    'group_key': group_key,
                    'group_slot': int(group_slot),
                    'pair_weight': pair_weight,
                    'pair_is_identity': pair_is_identity,
                    'pair_label': raw_entry.get('pair_label', None),
                }
            )

        if len(pair_entries) <= 0:
            raise RuntimeError(f"No usable pair manifest entries found in {manifest_path}.")

        pair_entries.sort(key=lambda entry: (entry['group_id'], entry['group_slot'], entry['src_local'], entry['ref_local']))
        self._pair_entries = pair_entries
        self._pair_group_to_indices = defaultdict(list)
        for pair_idx, entry in enumerate(pair_entries):
            self._pair_group_to_indices[int(entry['group_id'])].append(pair_idx)
        self._pair_group_order = self._pair_group_to_indices
        self.sizes = [int(base_sizes[entry['src_local']]) for entry in pair_entries]
        self._validate_pair_group_batch_constraints()
        self._pair_manifest_ready = True

    def _validate_pair_group_batch_constraints(self):
        if (
            not self._pair_group_to_indices
            or not bool(self.hparams.get("rhythm_pair_manifest_group_batches", True))
        ):
            return
        strict = bool(self.hparams.get("rhythm_pair_manifest_group_batches_strict", False))
        max_sentences = int(self.hparams.get("max_sentences", 0) or 0)
        max_tokens = int(self.hparams.get("max_tokens", 0) or 0)
        max_frames = int(self.hparams.get("max_frames", 0) or 0)
        violations = []
        for group_id, indices in self._pair_group_to_indices.items():
            group_size = len(indices)
            if group_size <= 0:
                continue
            group_frame_cap = max(
                min(int(self.sizes[idx]), max_frames) if max_frames > 0 else int(self.sizes[idx])
                for idx in indices
            )
            if max_sentences > 0 and group_size > max_sentences:
                violations.append(
                    f"group {group_id} has {group_size} pairs > max_sentences={max_sentences}"
                )
            if max_tokens > 0 and group_frame_cap * group_size > max_tokens:
                violations.append(
                    f"group {group_id} needs ~{group_frame_cap * group_size} tokens > max_tokens={max_tokens}"
                )
        if violations and strict:
            preview = "; ".join(violations[:4])
            raise RuntimeError(
                "Pair manifest grouped batches cannot stay intact under the current batch budget: "
                f"{preview}. Increase max_sentences/max_tokens, reduce group size, or disable "
                "rhythm_pair_manifest_group_batches_strict."
            )

    @staticmethod
    def _sample_reference_local_index(candidates, index: int) -> int:
        if len(candidates) <= 1:
            return index
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

        max_per_spk = int(self.hparams.get('max_samples_per_spk', 100))
        spk_ids_path = f"{self.data_dir}/{self.prefix}_spk_ids.npy"
        self.spk2indices = defaultdict(list)
        avail = self._base_avail_idxs if self._base_avail_idxs is not None else self.avail_idxs

        if os.path.exists(spk_ids_path):
            spk_ids = np.load(spk_ids_path, mmap_mode='r')
            local_spk_ids = spk_ids[avail]
            for local_idx in np.random.permutation(len(local_spk_ids)):
                sid = int(local_spk_ids[local_idx])
                bucket = self.spk2indices[sid]
                if len(bucket) < max_per_spk:
                    bucket.append(local_idx)
        else:
            for local_idx in np.random.permutation(len(avail)):
                sid = int(self._get_base_item(local_idx)['spk_id'])
                bucket = self.spk2indices[sid]
                if len(bucket) < max_per_spk:
                    bucket.append(local_idx)

        self._spk_map_ready = True

    def __getitem__(self, index):
        hparams = self.hparams
        if self._pair_manifest_ready and self._pair_entries is not None:
            pair_entry = self._pair_entries[index]
            src_local = int(pair_entry['src_local'])
            ref_local = int(pair_entry['ref_local'])
            item = self._get_base_item(src_local)
            ref_item = item if ref_local == src_local else self._get_base_item(ref_local)
            spec = self._mel_to_tensor(item['mel'], max_frames=hparams['max_frames'])
            ref_spec = self._mel_to_tensor(ref_item['mel'], max_frames=hparams['max_frames'])
            spk_id = int(item['spk_id'])
            sample = {
                'id': index,
                'item_name': item['item_name'],
                'mel': spec,
                'mel_nonpadding': spec.abs().sum(-1) > 0,
                'ref_mel': ref_spec,
                'ref_item_id': ref_local,
                '_raw_item': item,
                '_raw_ref_item': ref_item,
                'pair_group_id': int(pair_entry['group_id']),
                'pair_group_slot': int(pair_entry['group_slot']),
                'pair_is_identity': float(pair_entry['pair_is_identity']),
                'pair_weight': float(pair_entry['pair_weight']),
                'rhythm_pair_group_id': int(pair_entry['group_id']),
                'rhythm_pair_rank': int(pair_entry['group_slot']),
                'rhythm_pair_is_identity': float(pair_entry['pair_is_identity']),
                '_pair_manifest_entry': True,
                '_rhythm_pair_manifest_entry': True,
            }
        else:
            self._build_speaker_map()
            item = self._get_item(index)
            spec = self._mel_to_tensor(item['mel'], max_frames=hparams['max_frames'])

            spk_id = int(item['spk_id'])
            cand_locals = self.spk2indices[spk_id]
            ref_local = self._sample_reference_local_index(cand_locals, index)
            ref_item = item if ref_local == index else self._get_item(ref_local)
            ref_spec = self._mel_to_tensor(ref_item['mel'], max_frames=hparams['max_frames'])

            sample = {
                'id': index,
                'item_name': item['item_name'],
                'mel': spec,
                'mel_nonpadding': spec.abs().sum(-1) > 0,
                'ref_mel': ref_spec,
                'ref_item_id': ref_local,
                '_raw_item': item,
                '_raw_ref_item': ref_item,
            }

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
        if all('pair_group_id' in s for s in samples):
            batch['pair_group_id'] = torch.tensor([s['pair_group_id'] for s in samples], dtype=torch.long)
            batch['pair_group_slot'] = torch.tensor([s['pair_group_slot'] for s in samples], dtype=torch.long)
            batch['pair_is_identity'] = torch.tensor([s['pair_is_identity'] for s in samples], dtype=torch.float32)
            batch['pair_weight'] = torch.tensor([s['pair_weight'] for s in samples], dtype=torch.float32)
            batch['rhythm_pair_group_id'] = batch['pair_group_id']
            batch['rhythm_pair_rank'] = batch['pair_group_slot']
            batch['rhythm_pair_is_identity'] = batch['pair_is_identity']

        return batch

    def ordered_indices(self):
        if (
            not self._pair_manifest_ready
            or not self._pair_group_to_indices
            or not bool(self.hparams.get("rhythm_pair_manifest_group_batches", True))
        ):
            return super().ordered_indices()
        group_ids = list(self._pair_group_to_indices.keys())
        if self.shuffle:
            group_ids = list(np.random.permutation(group_ids))
        elif self.sort_by_len and self.sizes is not None:
            group_ids = sorted(
                group_ids,
                key=lambda gid: max(self.sizes[idx] for idx in self._pair_group_to_indices[gid]),
            )
        indices = []
        for group_id in group_ids:
            group_indices = sorted(
                self._pair_group_to_indices[group_id],
                key=lambda idx: self._pair_entries[idx]['group_slot'],
            )
            indices.extend(group_indices)
        return np.asarray(indices, dtype=np.int64)


class FastSpeechDataset(BaseSpeechDataset):
    """Dataset for FastSpeech-like models with ref mels & f0/uv."""

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        item = sample.get('_raw_item')
        if item is None:
            item = self._get_item(index)
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
            item = self._get_item(index)
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
