import os
import random
import sys
import traceback
import types
from functools import wraps
from itertools import chain
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import ConcatDataset
from utils.commons.hparams import hparams
import torch.nn.functional as F

def collate_1d_or_2d(values, pad_idx=0, left_pad=False, shift_right=False, max_len=None, shift_id=1):
    if len(values[0].shape) == 1:
        return collate_1d(values, pad_idx, left_pad, shift_right, max_len, shift_id)
    else:
        return collate_2d(values, pad_idx, left_pad, shift_right, max_len)


def collate_1d(values, pad_idx=0, left_pad=False, shift_right=False, max_len=None, shift_id=1):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values) if max_len is None else max_len
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if shift_right:
            dst[1:] = src[:-1]
            dst[0] = shift_id
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


def collate_2d(values, pad_idx=0, left_pad=False, shift_right=False, max_len=None):
    """Convert a list of 2d tensors into a padded 3d tensor."""
    size = max(v.size(0) for v in values) if max_len is None else max_len
    res = values[0].new(len(values), size, values[0].shape[1]).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if shift_right:
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res
def pad_or_cut_1d(values: torch.tensor, tgt_len, pad_value=0):
    src_len = values.shape[0]
    if src_len < tgt_len:
        res = F.pad(values, [0, tgt_len - src_len], value=pad_value)
    else:
        res = values[:tgt_len]
    return res

def pad_or_cut_2d(values: torch.tensor, tgt_len, dim=-1, pad_value=0):
    if dim == 0 or dim == -2:
        src_len = values.shape[0]
        if src_len < tgt_len:
            res = F.pad(values, [0, 0, 0, tgt_len - src_len], value=pad_value)
        else:
            res = values[:tgt_len]
    elif dim == 1 or dim == -1:
        src_len = values.shape[1]
        if src_len < tgt_len:
            res = F.pad(values, [0, tgt_len - src_len], value=pad_value)
        else:
            res = values[:, :tgt_len]
    else:
        raise RuntimeError(f"Wrong dim number {dim} while the tensor only has {len(values.shape)} dimensions.")
    return res

def pad_or_cut_xd(values, tgt_len, dim=-1, pad_value=0):
    if len(values.shape) == 1:
        return pad_or_cut_1d(values, tgt_len, pad_value)
    else:
        return pad_or_cut_2d(values, tgt_len, dim, pad_value)

def _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
    if len(batch) == 0:
        return 0
    if len(batch) == max_sentences:
        return 1
    if num_tokens > max_tokens:
        return 1
    return 0


def batch_by_size(
        indices, num_tokens_fn, max_tokens=None, max_sentences=None,
        required_batch_size_multiple=1, distributed=False
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.

    Args:
        indices (List[int]): ordered list of dataset indices
        num_tokens_fn (callable): function that returns the number of tokens at
            a given index
        max_tokens (int, optional): max number of tokens in each batch
            (default: None).
        max_sentences (int, optional): max number of sentences in each
            batch (default: None).
        required_batch_size_multiple (int, optional): require batch size to
            be a multiple of N (default: 1).
    """
    max_tokens = max_tokens if max_tokens is not None else sys.maxsize
    max_sentences = max_sentences if max_sentences is not None else sys.maxsize
    bsz_mult = required_batch_size_multiple

    if isinstance(indices, types.GeneratorType):
        indices = np.fromiter(indices, dtype=np.int64, count=-1)

    sample_len = 0
    sample_lens = []
    batch = []
    batches = []
    for i in range(len(indices)):
        idx = indices[i]
        num_tokens = num_tokens_fn(idx)
        sample_lens.append(num_tokens)
        sample_len = max(sample_len, num_tokens)

        assert sample_len <= max_tokens, (
            "sentence at index {} of size {} exceeds max_tokens "
            "limit of {}!".format(idx, sample_len, max_tokens)
        )
        num_tokens = (len(batch) + 1) * sample_len

        if _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
            mod_len = max(
                bsz_mult * (len(batch) // bsz_mult),
                len(batch) % bsz_mult,
            )
            batches.append(batch[:mod_len])
            batch = batch[mod_len:]
            sample_lens = sample_lens[mod_len:]
            sample_len = max(sample_lens) if len(sample_lens) > 0 else 0
        batch.append(idx)
    if len(batch) > 0:
        batches.append(batch)
    return batches


def pad_batch_to_world_size(batch, num_replicas):
    batch = list(batch)
    if num_replicas <= 1 or len(batch) == 0:
        return batch
    pad_count = (-len(batch)) % num_replicas
    if pad_count > 0:
        batch = batch + [batch[-1]] * pad_count
    return batch


def shard_batches_for_ddp(batches, num_replicas, rank, pad_to_divisible=True):
    if num_replicas <= 1:
        return list(batches)
    sharded = []
    for batch in batches:
        batch = list(batch)
        if len(batch) <= 0:
            continue
        if pad_to_divisible:
            batch = pad_batch_to_world_size(batch, num_replicas)
        rank_batch = batch[rank::num_replicas]
        if len(rank_batch) > 0:
            sharded.append(rank_batch)
    return sharded


def _seed_dataloader_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _build_dataloader_generator(dataset, *, shuffle: bool, rank: int):
    base_seed = int(hparams.get('seed', 1234))
    prefix = getattr(dataset, 'prefix', None)
    if prefix is None and hasattr(dataset, 'datasets'):
        prefix = "|".join(str(getattr(ds, 'prefix', '')) for ds in getattr(dataset, 'datasets', []))
    prefix = str(prefix or '')
    prefix_offset = sum(ord(ch) for ch in prefix)
    generator = torch.Generator()
    generator.manual_seed(base_seed + prefix_offset + (17 if shuffle else 0) + rank * 100003)
    return generator


def unpack_dict_to_list(samples):
    samples_ = []
    bsz = samples.get('outputs').size(0)
    for i in range(bsz):
        res = {}
        for k, v in samples.items():
            try:
                res[k] = v[i]
            except:
                pass
        samples_.append(res)
    return samples_


def remove_padding(x, padding_idx=0):
    if x is None:
        return None
    assert len(x.shape) in [1, 2]
    if len(x.shape) == 2:  # [T, H]
        return x[np.abs(x).sum(-1) != padding_idx]
    elif len(x.shape) == 1:  # [T]
        return x[x != padding_idx]


def data_loader(fn):
    """
    Decorator to make any fx with this use the lazy property
    :param fn:
    :return:
    """

    wraps(fn)
    attr_name = '_lazy_' + fn.__name__

    def _get_data_loader(self):
        try:
            value = getattr(self, attr_name)
        except AttributeError:
            try:
                value = fn(self)  # Lazy evaluation, done only once.
            except AttributeError as e:
                # Guard against AttributeError suppression. (Issue #142)
                traceback.print_exc()
                error = f'{fn.__name__}: An AttributeError was encountered: ' + str(e)
                raise RuntimeError(error) from e
            setattr(self, attr_name, value)  # Memoize evaluation.
        return value

    return _get_data_loader


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, shuffle):
        super().__init__()
        self.hparams = hparams
        self.shuffle = shuffle
        self.sort_by_len = hparams['sort_by_len']
        self.sizes = None

    @property
    def _sizes(self):
        return self.sizes

    def __getitem__(self, index):
        raise NotImplementedError

    def collater(self, samples):
        raise NotImplementedError

    def __len__(self):
        return len(self._sizes)

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return min(self._sizes[index], hparams['max_frames'])

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
            if self.sort_by_len:
                indices = indices[np.argsort(np.array(self._sizes)[indices], kind='mergesort')]
        else:
            indices = np.arange(len(self))
        return indices

    @property
    def num_workers(self):
        return int(os.getenv('NUM_WORKERS', hparams['ds_workers']))


class BaseConcatDataset(ConcatDataset):
    def collater(self, samples):
        return self.datasets[0].collater(samples)

    @property
    def _sizes(self):
        if not hasattr(self, 'sizes'):
            self.sizes = list(chain.from_iterable([d._sizes for d in self.datasets]))
        return self.sizes

    def size(self, index):
        return min(self._sizes[index], hparams['max_frames'])

    def num_tokens(self, index):
        return self.size(index)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.datasets[0].shuffle:
            indices = np.random.permutation(len(self))
            if self.datasets[0].sort_by_len:
                indices = indices[np.argsort(np.array(self._sizes)[indices], kind='mergesort')]
        else:
            indices = np.arange(len(self))
        return indices

    @property
    def num_workers(self):
        return self.datasets[0].num_workers

def build_dataloader(dataset, shuffle, max_tokens=None, max_sentences=None,
                     required_batch_size_multiple=-1, endless=False, apply_batch_by_size=True, pin_memory=None, use_ddp=False):
    import torch.distributed as dist

    devices_cnt = torch.cuda.device_count()
    if devices_cnt == 0:
        devices_cnt = 1
    use_ddp = bool(use_ddp and dist.is_available() and dist.is_initialized())
    rank = 0
    if use_ddp:
        devices_cnt = dist.get_world_size()
        rank = dist.get_rank()
    if not use_ddp:
        devices_cnt = 1
    if required_batch_size_multiple == -1:
        required_batch_size_multiple = devices_cnt

    def shuffle_batches(batches):
        np.random.shuffle(batches)
        return batches

    if max_tokens is not None:
        max_tokens *= devices_cnt
    if max_sentences is not None:
        max_sentences *= devices_cnt
    indices = dataset.ordered_indices()
    if apply_batch_by_size:
        batch_sampler = batch_by_size(
            indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )
    else:
        batch_sampler = []
        for i in range(0, len(indices), max_sentences):
            batch_sampler.append(indices[i:i + max_sentences])

    if shuffle:
        batches = shuffle_batches(list(batch_sampler))
        if endless:
            batches = [b for _ in range(1000) for b in shuffle_batches(list(batch_sampler))]
    else:
        batches = batch_sampler
        if endless:
            batches = [b for _ in range(1000) for b in batches]
    num_workers = dataset.num_workers
    if use_ddp:
        num_replicas = dist.get_world_size()
        # Keep every sample reachable under DDP, including short tail batches.
        batches = shard_batches_for_ddp(
            (x for x in batches if len(x) > 0),
            num_replicas=num_replicas,
            rank=rank,
            pad_to_divisible=bool(shuffle),
        )
    resolved_pin_memory = hparams.get('dl_pin_memory', pin_memory)
    if resolved_pin_memory is None:
        resolved_pin_memory = torch.cuda.is_available()
    loader_kwargs = {
        'collate_fn': dataset.collater,
        'batch_sampler': batches,
        'num_workers': num_workers,
        'pin_memory': bool(resolved_pin_memory),
        'worker_init_fn': _seed_dataloader_worker,
        'generator': _build_dataloader_generator(dataset, shuffle=bool(shuffle), rank=rank),
    }
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = bool(hparams.get('dl_persistent_workers', True))
        prefetch_factor = int(hparams.get('dl_prefetch_factor', 2) or 2)
        if prefetch_factor > 0:
            loader_kwargs['prefetch_factor'] = prefetch_factor
    return torch.utils.data.DataLoader(dataset, **loader_kwargs)
