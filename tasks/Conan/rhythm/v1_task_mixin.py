from __future__ import annotations

from tasks.Conan.rhythm.common.task_mixin import CommonRhythmTaskMixin
from tasks.Conan.rhythm.duration_v3.task_mixin import DurationV3TaskMixin
from utils.commons.hparams import hparams


class RhythmV1TaskMixin(DurationV3TaskMixin, CommonRhythmTaskMixin):
    """Official rhythm_v3 V1 task surface."""


__all__ = ["RhythmV1TaskMixin", "hparams"]
