import random
import numpy as np
import torch
from utils.commons.dataset_utils import BaseDataset, collate_1d_or_2d
from utils.commons.indexed_datasets import IndexedDataset
from utils.audio.pitch.utils import norm_interp_f0


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

    def _open_indexed_ds_if_needed(self):
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f"{self.data_dir}/{self.prefix}")

    def _get_item(self, local_idx):
        global_idx = self.avail_idxs[local_idx] if self.avail_idxs is not None else local_idx
        self._open_indexed_ds_if_needed()
        return self.indexed_ds[global_idx]

    def _mel_to_tensor(self, mel, *, max_frames: int):
        spec = torch.as_tensor(mel, dtype=torch.float32)
        spec = spec[:max_frames]
        frames_multiple = int(self.hparams['frames_multiple'])
        max_frames = spec.shape[0] // frames_multiple * frames_multiple
        return spec[:max_frames]

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
                sid = int(self._get_item(local_idx)['spk_id'])
                bucket = self.spk2indices[sid]
                if len(bucket) < max_per_spk:
                    bucket.append(local_idx)

        self._spk_map_ready = True

    def __getitem__(self, index):
        self._build_speaker_map()
        item = self._get_item(index)
        hparams = self.hparams

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

        return batch


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
