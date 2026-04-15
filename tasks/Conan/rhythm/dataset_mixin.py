"""Compatibility shell for the split Conan rhythm dataset mixins."""

from __future__ import annotations

from tasks.Conan.rhythm.common.dataset_mixin import CommonRhythmDatasetMixin
from tasks.Conan.rhythm.dataset_contract_constants import RhythmDatasetContractConstants
from tasks.Conan.rhythm.duration_v3.dataset_mixin import DurationV3DatasetMixin
from tasks.Conan.rhythm.rhythm_v2.dataset_mixin import RhythmV2DatasetMixin


class RhythmConanDatasetMixin(RhythmDatasetContractConstants, DurationV3DatasetMixin, CommonRhythmDatasetMixin):
    """Maintained rhythm_v3 dataset shell."""


class LegacyRhythmConanDatasetMixin(RhythmDatasetContractConstants, RhythmV2DatasetMixin, CommonRhythmDatasetMixin):
    """Legacy rhythm_v2 dataset shell."""


__all__ = ["LegacyRhythmConanDatasetMixin", "RhythmConanDatasetMixin"]
