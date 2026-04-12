from .contracts import DurationExecution, DurationRuntimeState, ReferenceDurationMemory, SourceUnitBatch
from .minimal_head import MinimalStreamingDurationHeadV1G
from .module import MixedEffectsDurationModule, StreamingDurationModule
from .summary_memory import (
    CausalUnitRunEncoder,
    PromptDurationMemoryEncoder,
    PromptGlobalConditionEncoderV1G,
    SharedSummaryCodebook,
    StreamingDurationHead,
)

__all__ = [
    "ConanDurationAdapter",
    "CausalUnitRunEncoder",
    "DurationExecution",
    "MixedEffectsDurationModule",
    "MinimalStreamingDurationHeadV1G",
    "PromptDurationMemoryEncoder",
    "PromptGlobalConditionEncoderV1G",
    "SharedSummaryCodebook",
    "DurationRuntimeState",
    "ReferenceDurationMemory",
    "SourceUnitBatch",
    "StreamingDurationModule",
    "StreamingDurationHead",
]


def __getattr__(name: str):
    if name == "ConanDurationAdapter":
        from .runtime_adapter import ConanDurationAdapter

        return ConanDurationAdapter
    raise AttributeError(name)
