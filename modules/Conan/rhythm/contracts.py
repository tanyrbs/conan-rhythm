from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class RhythmPublicInputs:
    content_units: torch.Tensor
    dur_anchor_src: torch.Tensor
    ref_rhythm_stats: torch.Tensor
    ref_rhythm_trace: torch.Tensor
    unit_mask: Optional[torch.Tensor] = None


@dataclass
class StreamingRhythmState:
    phase_ptr: torch.Tensor
    backlog: torch.Tensor
    clock_delta: torch.Tensor
    commit_frontier: torch.Tensor
    previous_speech_exec: Optional[torch.Tensor] = None
    previous_pause_exec: Optional[torch.Tensor] = None


@dataclass
class RhythmPlannerOutputs:
    speech_budget_win: torch.Tensor
    pause_budget_win: torch.Tensor
    dur_logratio_unit: torch.Tensor
    pause_weight_unit: torch.Tensor
    total_budget_win: torch.Tensor
    pause_share_win: torch.Tensor
    anchor_gate: torch.Tensor
    boundary_latent: torch.Tensor
    trace_context: torch.Tensor


@dataclass
class RhythmExecution:
    speech_duration_exec: torch.Tensor
    pause_after_exec: torch.Tensor
    effective_duration_exec: torch.Tensor
    commit_frontier: torch.Tensor
    planner: RhythmPlannerOutputs
    next_state: StreamingRhythmState
