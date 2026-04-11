"""Compatibility shell for the split Conan rhythm task mixins."""

from __future__ import annotations

from tasks.Conan.rhythm.common.task_mixin import CommonRhythmTaskMixin
from tasks.Conan.rhythm.loss_routing import compute_reporting_total_loss
from tasks.Conan.rhythm.metrics import build_rhythm_metric_dict
from tasks.Conan.rhythm.duration_v3.task_mixin import DurationV3TaskMixin
from tasks.Conan.rhythm.rhythm_v2.task_mixin import RhythmV2TaskMixin
from utils.commons.hparams import hparams


class RhythmConanTaskMixin(DurationV3TaskMixin, RhythmV2TaskMixin, CommonRhythmTaskMixin):
    _LEGACY_RHYTHM_SOURCE_CACHE_REQUIRED_KEYS = (
        "content_units",
        "dur_anchor_src",
    )
    _LEGACY_RHYTHM_SOURCE_CACHE_OPTIONAL_KEYS = (
        "open_run_mask",
        "sealed_mask",
        "sep_hint",
        "boundary_confidence",
    )


__all__ = ["RhythmConanTaskMixin"]
