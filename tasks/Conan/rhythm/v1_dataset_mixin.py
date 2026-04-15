from __future__ import annotations

from tasks.Conan.rhythm.common.dataset_mixin import CommonRhythmDatasetMixin
from tasks.Conan.rhythm.dataset_mixin import RhythmConanDatasetMixin
from tasks.Conan.rhythm.duration_v3.dataset_mixin import DurationV3DatasetMixin


class RhythmV1DatasetMixin(DurationV3DatasetMixin, CommonRhythmDatasetMixin):
    """Official rhythm_v3 V1 dataset surface."""


for _name, _value in list(vars(RhythmConanDatasetMixin).items()):
    if _name.startswith("_RHYTHM_"):
        setattr(RhythmV1DatasetMixin, _name, _value)


__all__ = ["RhythmV1DatasetMixin"]
