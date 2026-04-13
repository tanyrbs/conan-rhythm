from .contracts import DurationExecution, DurationRuntimeState, ReferenceDurationMemory, SourceUnitBatch
from .global_condition import PromptGlobalConditionEncoderV1G
from .minimal_writer import MinimalStreamingDurationHeadV1G, MinimalStreamingDurationWriterV1G
from .module import MixedEffectsDurationModule, StreamingDurationModule
from .run_encoder import CausalUnitRunEncoder
from .summary_memory import (
    PromptDurationMemoryEncoder,
    SharedSummaryCodebook,
    StreamingDurationHead,
)

__all__ = [
    "ConanDurationAdapter",
    "CausalUnitRunEncoder",
    "DurationExecution",
    "MixedEffectsDurationModule",
    "MinimalStreamingDurationHeadV1G",
    "MinimalStreamingDurationWriterV1G",
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
