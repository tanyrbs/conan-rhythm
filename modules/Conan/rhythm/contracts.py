from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .frame_plan import RhythmFramePlan


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
    phase_anchor_progress: Optional[torch.Tensor] = None
    phase_anchor_total: Optional[torch.Tensor] = None
    speech_budget_debt: Optional[torch.Tensor] = None
    pause_budget_debt: Optional[torch.Tensor] = None

    @property
    def previous_blank_exec(self) -> Optional[torch.Tensor]:
        return self.previous_pause_exec


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
    source_boundary_cue: Optional[torch.Tensor] = None

    @property
    def blank_budget_win(self) -> torch.Tensor:
        return self.pause_budget_win


@dataclass
class RhythmExecution:
    speech_duration_exec: torch.Tensor
    blank_duration_exec: torch.Tensor
    pause_after_exec: torch.Tensor
    effective_duration_exec: torch.Tensor
    commit_frontier: torch.Tensor
    slot_duration_exec: torch.Tensor
    slot_mask: torch.Tensor
    slot_is_blank: torch.Tensor
    slot_unit_index: torch.Tensor
    frame_plan: Optional["RhythmFramePlan"]
    planner: RhythmPlannerOutputs
    next_state: StreamingRhythmState

    @property
    def blank_slot_duration_exec(self) -> torch.Tensor:
        return self.slot_duration_exec

    @property
    def blank_slot_mask(self) -> torch.Tensor:
        return self.slot_mask

    @property
    def blank_slot_is_blank(self) -> torch.Tensor:
        return self.slot_is_blank

    @property
    def blank_slot_unit_index(self) -> torch.Tensor:
        return self.slot_unit_index


@dataclass
class RhythmTeacherTargets:
    speech_exec_tgt: torch.Tensor
    pause_exec_tgt: torch.Tensor
    speech_budget_tgt: torch.Tensor
    pause_budget_tgt: torch.Tensor
    allocation_tgt: torch.Tensor
    confidence: torch.Tensor
    trace_context: torch.Tensor
    prefix_clock_tgt: torch.Tensor
    prefix_backlog_tgt: torch.Tensor

    @property
    def blank_exec_tgt(self) -> torch.Tensor:
        return self.pause_exec_tgt

    @property
    def blank_budget_tgt(self) -> torch.Tensor:
        return self.pause_budget_tgt
