from __future__ import annotations

import sys
import unittest
from itertools import islice
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.commons.dataset_utils import _DynamicBatchSampler


class _ToyDataset:
    def __init__(self) -> None:
        self.ordered_calls = 0

    def ordered_indices(self):
        self.ordered_calls += 1
        return [0, 1, 2, 3, 4]

    def num_tokens(self, index):
        return 1


class DynamicBatchSamplerTests(unittest.TestCase):
    def test_endless_sampler_rebuilds_batches_without_large_preallocation(self) -> None:
        dataset = _ToyDataset()
        sampler = _DynamicBatchSampler(
            dataset,
            shuffle=False,
            max_tokens=None,
            max_sentences=2,
            required_batch_size_multiple=1,
            endless=True,
            apply_batch_by_size=False,
            use_ddp=False,
            num_replicas=1,
            rank=0,
            pad_to_divisible=False,
            seed=1234,
        )

        batches = list(islice(iter(sampler), 5))

        self.assertEqual(batches, [[0, 1], [2, 3], [4], [0, 1], [2, 3]])
        self.assertGreaterEqual(dataset.ordered_calls, 2)


if __name__ == "__main__":
    unittest.main()
