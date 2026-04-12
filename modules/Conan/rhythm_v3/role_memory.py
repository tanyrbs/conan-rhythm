from __future__ import annotations

"""Legacy compatibility shim for the old `role_memory` module name.

The maintained rhythm_v3 mainline now uses prompt-summary conditioning. New
code should import from the canonical `summary_memory` module, but this shim
keeps the legacy import path alive.
"""

from .summary_memory import (
    CausalUnitRunEncoder,
    CausalRoleQueryEncoder,
    CausalStretchQueryEncoder,
    CausalSummaryQueryEncoder,
    PromptDurationMemoryEncoder,
    PromptSummaryDurationHead,
    PromptSummaryEncoder,
    SharedSummaryCodebook,
    StreamingDurationHead,
)
from .minimal_head import MinimalStreamingDurationHeadV1G

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
    "StreamingDurationHead",
]
