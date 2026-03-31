from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .contracts import RhythmExecution, RhythmPlannerOutputs, StreamingRhythmState


@dataclass
class ProjectorConfig:
    min_speech_frames: float = 1.0
    max_speech_expand: float = 3.0
    tail_hold_units: int = 2
    boundary_commit_threshold: float = 0.45


class StreamingRhythmProjector(nn.Module):
    def __init__(self, config: ProjectorConfig | None = None) -> None:
        super().__init__()
        self.config = config or ProjectorConfig()

    def init_state(self, batch_size: int, device: torch.device) -> StreamingRhythmState:
        zeros = torch.zeros(batch_size, device=device)
        return StreamingRhythmState(
            phase_ptr=zeros.clone(),
            backlog=zeros.clone(),
            clock_delta=zeros.clone(),
            commit_frontier=torch.zeros(batch_size, dtype=torch.long, device=device),
            previous_speech_exec=None,
            previous_pause_exec=None,
        )

    @staticmethod
    def _renormalize_to_budget(values: torch.Tensor, mask: torch.Tensor, budget: torch.Tensor) -> torch.Tensor:
        total = (values * mask).sum(dim=1, keepdim=True).clamp_min(1e-6)
        return values * (budget / total)

    def _project_speech(
        self,
        *,
        dur_anchor_src: torch.Tensor,
        dur_logratio_unit: torch.Tensor,
        unit_mask: torch.Tensor,
        speech_budget_win: torch.Tensor,
    ) -> torch.Tensor:
        base = dur_anchor_src.float().clamp_min(self.config.min_speech_frames) * torch.exp(dur_logratio_unit.float())
        max_speech = dur_anchor_src.float().clamp_min(self.config.min_speech_frames) * self.config.max_speech_expand
        min_speech = torch.full_like(base, float(self.config.min_speech_frames))
        speech = torch.maximum(torch.minimum(base, max_speech), min_speech) * unit_mask
        speech = self._renormalize_to_budget(speech, unit_mask, speech_budget_win)
        return speech * unit_mask

    def _project_pause(
        self,
        *,
        pause_weight_unit: torch.Tensor,
        boundary_latent: torch.Tensor,
        unit_mask: torch.Tensor,
        pause_budget_win: torch.Tensor,
    ) -> torch.Tensor:
        scores = pause_weight_unit.float().clamp_min(0.0)
        scores = scores * (0.5 + boundary_latent.float()) * unit_mask
        score_total = scores.sum(dim=1, keepdim=True)
        fallback = unit_mask / unit_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        weights = torch.where(score_total > 0, scores / score_total.clamp_min(1e-6), fallback)
        return weights * pause_budget_win

    def _compute_commit_frontier(
        self,
        *,
        state: StreamingRhythmState,
        unit_mask: torch.Tensor,
        open_run_mask: torch.Tensor | None,
        boundary_latent: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, _ = unit_mask.shape
        active_len = unit_mask.long().sum(dim=1)
        if open_run_mask is None:
            open_run_mask = torch.zeros_like(unit_mask, dtype=torch.long)
        commit = state.commit_frontier.clone()
        for batch_idx in range(batch_size):
            visible = int(active_len[batch_idx].item())
            prev = int(state.commit_frontier[batch_idx].item())
            release_cap = min(max(0, visible - int(self.config.tail_hold_units)), visible)
            closed_prefix = 0
            for unit_idx in range(visible):
                if int(open_run_mask[batch_idx, unit_idx].item()) > 0:
                    break
                closed_prefix = unit_idx + 1
            candidate = min(release_cap, closed_prefix)
            if candidate > 0 and boundary_latent is not None and candidate < visible:
                boundary_value = float(boundary_latent[batch_idx, candidate - 1].item())
                if boundary_value < float(self.config.boundary_commit_threshold):
                    candidate = max(prev, candidate - 1)
            commit[batch_idx] = max(prev, candidate)
        return commit

    def _advance_state(
        self,
        *,
        state: StreamingRhythmState,
        dur_anchor_src: torch.Tensor,
        effective_duration_exec: torch.Tensor,
        commit_frontier: torch.Tensor,
        speech_duration_exec: torch.Tensor,
        pause_after_exec: torch.Tensor,
    ) -> StreamingRhythmState:
        batch_size = dur_anchor_src.size(0)
        next_phase = state.phase_ptr.clone()
        next_backlog = state.backlog.clone()
        next_clock = state.clock_delta.clone()
        for batch_idx in range(batch_size):
            prev = int(state.commit_frontier[batch_idx].item())
            curr = int(commit_frontier[batch_idx].item())
            if curr <= prev:
                continue
            exec_prefix = effective_duration_exec[batch_idx, prev:curr].sum()
            src_prefix = dur_anchor_src[batch_idx, prev:curr].float().sum()
            delta = exec_prefix - src_prefix
            next_clock[batch_idx] = next_clock[batch_idx] + delta
            next_backlog[batch_idx] = next_clock[batch_idx].clamp_min(0.0)
            visible_total = effective_duration_exec[batch_idx].sum().clamp_min(1.0)
            next_phase[batch_idx] = (next_phase[batch_idx] + exec_prefix / visible_total).clamp(0.0, 1.0)
        return StreamingRhythmState(
            phase_ptr=next_phase,
            backlog=next_backlog,
            clock_delta=next_clock,
            commit_frontier=commit_frontier.long(),
            previous_speech_exec=speech_duration_exec.detach(),
            previous_pause_exec=pause_after_exec.detach(),
        )

    def forward(
        self,
        *,
        dur_anchor_src: torch.Tensor,
        unit_mask: torch.Tensor,
        speech_budget_win: torch.Tensor,
        pause_budget_win: torch.Tensor,
        dur_logratio_unit: torch.Tensor,
        pause_weight_unit: torch.Tensor,
        boundary_latent: torch.Tensor,
        state: StreamingRhythmState,
        open_run_mask: torch.Tensor | None = None,
        planner: RhythmPlannerOutputs,
    ) -> RhythmExecution:
        unit_mask = unit_mask.float()
        speech_duration_exec = self._project_speech(
            dur_anchor_src=dur_anchor_src,
            dur_logratio_unit=dur_logratio_unit,
            unit_mask=unit_mask,
            speech_budget_win=speech_budget_win,
        )
        pause_after_exec = self._project_pause(
            pause_weight_unit=pause_weight_unit,
            boundary_latent=boundary_latent,
            unit_mask=unit_mask,
            pause_budget_win=pause_budget_win,
        )
        effective_duration_exec = (speech_duration_exec + pause_after_exec) * unit_mask
        commit_frontier = self._compute_commit_frontier(
            state=state,
            unit_mask=unit_mask,
            open_run_mask=open_run_mask,
            boundary_latent=boundary_latent,
        )
        next_state = self._advance_state(
            state=state,
            dur_anchor_src=dur_anchor_src,
            effective_duration_exec=effective_duration_exec,
            commit_frontier=commit_frontier,
            speech_duration_exec=speech_duration_exec,
            pause_after_exec=pause_after_exec,
        )
        return RhythmExecution(
            speech_duration_exec=speech_duration_exec,
            pause_after_exec=pause_after_exec,
            effective_duration_exec=effective_duration_exec,
            commit_frontier=commit_frontier,
            planner=planner,
            next_state=next_state,
        )
