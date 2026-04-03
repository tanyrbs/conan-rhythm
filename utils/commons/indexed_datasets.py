import pickle
import sys
from copy import deepcopy

import numpy as np


def _ensure_numpy_pickle_compat():
    """Allow loading legacy/newer pickled numpy objects across numpy 1.x/2.x."""
    core_module = getattr(np, "core", None)
    if core_module is None:
        return
    sys.modules.setdefault("numpy._core", core_module)
    for name in ("multiarray", "_multiarray_umath", "numeric"):
        submodule = getattr(core_module, name, None)
        if submodule is not None:
            sys.modules.setdefault(f"numpy._core.{name}", submodule)


def _load_index_offsets(path):
    index_path = f"{path}.idx"
    try:
        return np.load(index_path, allow_pickle=True).item()['offsets']
    except ModuleNotFoundError as exc:
        if "numpy._core" not in str(exc):
            raise
        _ensure_numpy_pickle_compat()
        return np.load(index_path, allow_pickle=True).item()['offsets']


class IndexedDataset:
    def __init__(self, path, num_cache=1):
        super().__init__()
        self.path = path
        self.data_file = None
        self.data_offsets = _load_index_offsets(path)
        self.data_file = open(f"{path}.data", 'rb', buffering=-1)
        self.cache = []
        self.num_cache = num_cache

    def check_index(self, i):
        if i < 0 or i >= len(self.data_offsets) - 1:
            raise IndexError('index out of range')

    def __del__(self):
        if self.data_file:
            self.data_file.close()

    def __getitem__(self, i):
        self.check_index(i)
        if self.num_cache > 0:
            for c in self.cache:
                if c[0] == i:
                    return c[1]
        self.data_file.seek(self.data_offsets[i])
        b = self.data_file.read(self.data_offsets[i + 1] - self.data_offsets[i])
        item = pickle.loads(b)
        if self.num_cache > 0:
            self.cache = [(i, deepcopy(item))] + self.cache[:-1]
        return item

    def __len__(self):
        return len(self.data_offsets) - 1

class IndexedDatasetBuilder:
    def __init__(self, path):
        self.path = path
        self.out_file = open(f"{path}.data", 'wb')
        self.byte_offsets = [0]

    def add_item(self, item):
        s = pickle.dumps(item)
        bytes = self.out_file.write(s)
        self.byte_offsets.append(self.byte_offsets[-1] + bytes)

    def finalize(self):
        self.out_file.close()
        np.save(open(f"{self.path}.idx", 'wb'), {'offsets': self.byte_offsets})


if __name__ == "__main__":
    import random
    from tqdm import tqdm
    ds_path = '/tmp/indexed_ds_example'
    size = 100
    items = [{"a": np.random.normal(size=[10000, 10]),
              "b": np.random.normal(size=[10000, 10])} for i in range(size)]
    builder = IndexedDatasetBuilder(ds_path)
    for i in tqdm(range(size)):
        builder.add_item(items[i])
    builder.finalize()
    ds = IndexedDataset(ds_path)
    for i in tqdm(range(10000)):
        idx = random.randint(0, size - 1)
        assert (ds[idx]['a'] == items[idx]['a']).all()
