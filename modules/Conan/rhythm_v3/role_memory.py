from __future__ import annotations

"""Legacy compatibility shim for the old `role_memory` module name.

The maintained minimal V1-G surface now lives in `global_condition` and
`minimal_writer`; richer prompt-memory code remains in `summary_memory`.
This shim keeps the legacy import path alive without making it the canonical
surface.
"""

from .run_encoder import CausalUnitRunEncoder
from .summary_memory import (
    CausalRoleQueryEncoder,
    CausalStretchQueryEncoder,
    CausalSummaryQueryEncoder,
    PromptDurationMemoryEncoder,
    PromptSummaryDurationHead,
    PromptSummaryEncoder,
    SharedSummaryCodebook,
    StreamingDurationHead,
)
from .minimal_writer import MinimalStreamingDurationHeadV1G, MinimalStreamingDurationWriterV1G

SharedRoleCodebook = SharedSummaryCodebook
PromptSummaryProbeBank = SharedSummaryCodebook

__all__ = [
    "CausalUnitRunEncoder",
    "CausalRoleQueryEncoder",
    "CausalStretchQueryEncoder",
    "CausalSummaryQueryEncoder",
    "PromptDurationMemoryEncoder",
    "PromptSummaryDurationHead",
    "PromptSummaryEncoder",
    "PromptSummaryProbeBank",
    "SharedSummaryCodebook",
    "SharedRoleCodebook",
    "MinimalStreamingDurationHeadV1G",
    "MinimalStreamingDurationWriterV1G",
    "StreamingDurationHead",
]
