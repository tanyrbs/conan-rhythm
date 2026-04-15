from __future__ import annotations

from tasks.Conan.rhythm.common.dataset_mixin import CommonRhythmDatasetMixin
from tasks.Conan.rhythm.dataset_contract_constants import RhythmDatasetContractConstants
from tasks.Conan.rhythm.duration_v3.dataset_mixin import DurationV3DatasetMixin


class RhythmV1DatasetMixin(RhythmDatasetContractConstants, DurationV3DatasetMixin, CommonRhythmDatasetMixin):
    """Official rhythm_v3 V1 dataset surface."""


__all__ = ["RhythmV1DatasetMixin"]
