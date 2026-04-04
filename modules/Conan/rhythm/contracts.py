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
    clock_delta: torch.Tensor
    commit_frontier: torch.Tensor
    previous_speech_exec: Optional[torch.Tensor] = None
    previous_pause_exec: Optional[torch.Tensor] = None
    phase_anchor: Optional[torch.Tensor] = None

    @property
    def backlog(self) -> torch.Tensor:
        return self.clock_delta.clamp_min(0.0)

    @property
    def phase_anchor_progress(self) -> Optional[torch.Tensor]:
        if self.phase_anchor is None:
            return None
        return self.phase_anchor[..., 0]

    @property
    def phase_anchor_total(self) -> Optional[torch.Tensor]:
        if self.phase_anchor is None:
            return None
        return self.phase_anchor[..., 1]

    @property
    def phase_progress_ratio(self) -> Optional[torch.Tensor]:
        if self.phase_anchor is None:
            return None
        progress = self.phase_anchor[..., 0].float()
        total = self.phase_anchor[..., 1].float().clamp_min(1.0)
        return progress / total

    @property
    def phase_ptr_gap(self) -> Optional[torch.Tensor]:
        progress_ratio = self.phase_progress_ratio
        if progress_ratio is None:
            return None
        return self.phase_ptr.float() - progress_ratio

    @property
    def previous_blank_exec(self) -> Optional[torch.Tensor]:
        return self.previous_pause_exec


@dataclass
class RhythmPlannerOutputs:
    speech_budget_win: torch.Tensor
    pause_budget_win: torch.Tensor
    dur_logratio_unit: torch.Tensor
    pause_weight_unit: torch.Tensor
    boundary_score_unit: torch.Tensor
    trace_context: torch.Tensor
    source_boundary_cue: Optional[torch.Tensor] = None

    @property
    def blank_budget_win(self) -> torch.Tensor:
        return self.pause_budget_win

    @property
    def dur_shape_unit(self) -> torch.Tensor:
        """Planner-facing alias used by the weak-factorization mainline."""

        return self.dur_logratio_unit

    @property
    def pause_shape_unit(self) -> torch.Tensor:
        """Planner-facing alias used by the weak-factorization mainline."""

        return self.pause_weight_unit

    @property
    def total_budget_win(self) -> torch.Tensor:
        """Derived compatibility alias; do not persist a redundant field."""

        return self.speech_budget_win + self.pause_budget_win

    @property
    def pause_share_win(self) -> torch.Tensor:
        """Derived compatibility alias; only speech/pause budgets are primitive."""

        return self.pause_budget_win / self.total_budget_win.clamp_min(1e-6)

    @property
    def anchor_gate(self) -> torch.Tensor:
        """Compatibility alias for the retired multiplicative gate."""

        return torch.ones_like(self.speech_budget_win)

    @property
    def raw_speech_budget_win(self) -> torch.Tensor:
        return getattr(self, "_raw_speech_budget_win", self.speech_budget_win)

    @raw_speech_budget_win.setter
    def raw_speech_budget_win(self, value: torch.Tensor) -> None:
        self._raw_speech_budget_win = value

    @property
    def raw_pause_budget_win(self) -> torch.Tensor:
        return getattr(self, "_raw_pause_budget_win", self.pause_budget_win)

    @raw_pause_budget_win.setter
    def raw_pause_budget_win(self, value: torch.Tensor) -> None:
        self._raw_pause_budget_win = value

    @property
    def effective_speech_budget_win(self) -> torch.Tensor:
        return self.speech_budget_win

    @property
    def effective_pause_budget_win(self) -> torch.Tensor:
        return self.pause_budget_win

    @property
    def feasible_speech_budget_delta(self) -> torch.Tensor:
        return getattr(self, "_feasible_speech_budget_delta", torch.zeros_like(self.speech_budget_win))

    @feasible_speech_budget_delta.setter
    def feasible_speech_budget_delta(self, value: torch.Tensor) -> None:
        self._feasible_speech_budget_delta = value

    @property
    def feasible_pause_budget_delta(self) -> torch.Tensor:
        return getattr(self, "_feasible_pause_budget_delta", torch.zeros_like(self.pause_budget_win))

    @feasible_pause_budget_delta.setter
    def feasible_pause_budget_delta(self, value: torch.Tensor) -> None:
        self._feasible_pause_budget_delta = value

    @property
    def feasible_total_budget_delta(self) -> torch.Tensor:
        return getattr(self, "_feasible_total_budget_delta", torch.zeros_like(self.total_budget_win))

    @feasible_total_budget_delta.setter
    def feasible_total_budget_delta(self, value: torch.Tensor) -> None:
        self._feasible_total_budget_delta = value

    @property
    def boundary_latent(self) -> torch.Tensor:
        """Backward-compatible alias for legacy call sites/checkpoints."""

        return self.boundary_score_unit


@dataclass
class RhythmExecution:
    speech_duration_exec: torch.Tensor
    blank_duration_exec: torch.Tensor
    pause_after_exec: torch.Tensor
    effective_duration_exec: torch.Tensor
    commit_frontier: torch.Tensor
    slot_duration_exec: Optional[torch.Tensor]
    slot_mask: Optional[torch.Tensor]
    slot_is_blank: Optional[torch.Tensor]
    slot_unit_index: Optional[torch.Tensor]
    frame_plan: Optional["RhythmFramePlan"]
    planner: RhythmPlannerOutputs
    next_state: StreamingRhythmState

    @property
    def blank_slot_duration_exec(self) -> Optional[torch.Tensor]:
        return self.slot_duration_exec

    @property
    def blank_slot_mask(self) -> Optional[torch.Tensor]:
        return self.slot_mask

    @property
    def blank_slot_is_blank(self) -> Optional[torch.Tensor]:
        return self.slot_is_blank

    @property
    def blank_slot_unit_index(self) -> Optional[torch.Tensor]:
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
