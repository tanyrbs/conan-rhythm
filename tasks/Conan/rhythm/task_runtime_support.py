from __future__ import annotations

from .duration_v3.task_runtime_support import DurationV3TaskRuntimeSupportMixin
from .rhythm_v2.task_runtime_support import RhythmV2TaskRuntimeSupport
from utils.commons.hparams import hparams


class RhythmTaskRuntimeSupport(DurationV3TaskRuntimeSupportMixin, RhythmV2TaskRuntimeSupport):
    """Compatibility facade combining shared, v2, and v3 runtime support."""


__all__ = ["RhythmTaskRuntimeSupport", "hparams"]
