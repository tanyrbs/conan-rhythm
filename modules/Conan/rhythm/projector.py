from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .contracts import RhythmExecution, RhythmPlannerOutputs, StreamingRhythmState
from .frame_plan import build_frame_plan_from_execution, build_interleaved_blank_slot_schedule


@dataclass
class ProjectorConfig:
    min_speech_frames: float = 1.0
    max_speech_expand: float = 3.0
    tail_hold_units: int = 2
    boundary_commit_threshold: float = 0.45
    pause_topk_ratio: float = 0.35
    pause_min_boundary_weight: float = 0.10
    pause_boundary_bias_weight: float = 0.15
    pause_train_soft: bool = True
    pause_soft_temperature: float = 0.12


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
            phase_anchor_progress=zeros.clone(),
            phase_anchor_total=zeros.clone(),
            speech_budget_debt=zeros.clone(),
            pause_budget_debt=zeros.clone(),
        )

    @staticmethod
    def _renormalize_to_budget(values: torch.Tensor, mask: torch.Tensor, budget: torch.Tensor) -> torch.Tensor:
        total = (values * mask).sum(dim=1, keepdim=True).clamp_min(1e-6)
        return values * (budget / total)

    def _lift_budgets_to_feasible_region(
        self,
        *,
        dur_anchor_src: torch.Tensor,
        unit_mask: torch.Tensor,
        speech_budget_win: torch.Tensor,
        pause_budget_win: torch.Tensor,
        state: StreamingRhythmState,
        reuse_prefix: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        speech_rows = []
        pause_rows = []
        previous_speech = state.previous_speech_exec
        previous_pause = state.previous_pause_exec
        for batch_idx in range(unit_mask.size(0)):
            speech_budget_row = speech_budget_win[batch_idx].float()
            pause_budget_row = pause_budget_win[batch_idx].float()
            frontier = int(state.commit_frontier[batch_idx].item()) if reuse_prefix else 0
            prefix_speech = speech_budget_row.new_tensor(0.0)
            prefix_pause = pause_budget_row.new_tensor(0.0)
            tail_mask = unit_mask[batch_idx].float().clone()
            if frontier > 0:
                if previous_speech is not None and batch_idx < previous_speech.size(0):
                    valid_frontier = min(frontier, int(previous_speech.size(1)), int(tail_mask.size(0)))
                    prefix_speech = (
                        previous_speech[batch_idx, :valid_frontier].float()
                        * tail_mask[:valid_frontier]
                    ).sum()
                    tail_mask[:valid_frontier] = 0.0
                if previous_pause is not None and batch_idx < previous_pause.size(0):
                    valid_frontier = min(frontier, int(previous_pause.size(1)), int(unit_mask.size(1)))
                    prefix_pause = (
                        previous_pause[batch_idx, :valid_frontier].float()
                        * unit_mask[batch_idx, :valid_frontier].float()
                    ).sum()
            min_tail_speech = tail_mask.sum() * float(self.config.min_speech_frames)
            speech_floor = prefix_speech + min_tail_speech
            pause_floor = prefix_pause
            speech_rows.append(torch.maximum(speech_budget_row, speech_floor))
            pause_rows.append(torch.maximum(pause_budget_row, pause_floor))
        return torch.stack(speech_rows, dim=0), torch.stack(pause_rows, dim=0)

    def _project_speech(
        self,
        *,
        dur_anchor_src: torch.Tensor,
        dur_logratio_unit: torch.Tensor,
        unit_mask: torch.Tensor,
        speech_budget_win: torch.Tensor,
        state: StreamingRhythmState,
        reuse_prefix: bool,
    ) -> torch.Tensor:
        base = dur_anchor_src.float().clamp_min(self.config.min_speech_frames) * torch.exp(dur_logratio_unit.float())
        max_speech = dur_anchor_src.float().clamp_min(self.config.min_speech_frames) * self.config.max_speech_expand
        min_speech = torch.full_like(base, float(self.config.min_speech_frames))
        candidate = torch.maximum(torch.minimum(base, max_speech), min_speech) * unit_mask
        speech = candidate.new_zeros(candidate.shape)
        previous = state.previous_speech_exec
        for batch_idx in range(candidate.size(0)):
            mask_row = unit_mask[batch_idx]
            budget_row = speech_budget_win[batch_idx : batch_idx + 1]
            frontier = int(state.commit_frontier[batch_idx].item())
            if reuse_prefix and previous is not None and batch_idx < previous.size(0) and frontier > 0:
                valid_frontier = min(frontier, int(previous.size(1)), int(candidate.size(1)))
                prefix = previous[batch_idx, :valid_frontier].float() * mask_row[:valid_frontier]
                speech[batch_idx, :valid_frontier] = prefix
            else:
                valid_frontier = 0
            tail_mask = mask_row.clone()
            if valid_frontier > 0:
                tail_mask[:valid_frontier] = 0.0
            remaining_budget = budget_row.squeeze(0) - speech[batch_idx].sum()
            if tail_mask.sum() > 0 and float(remaining_budget.item()) > 0:
                tail_values = self._renormalize_to_budget(
                    candidate[batch_idx : batch_idx + 1],
                    tail_mask.unsqueeze(0),
                    remaining_budget.view(1, 1),
                )[0]
                speech[batch_idx] = speech[batch_idx] + tail_values * tail_mask
        return speech * unit_mask

    def _project_pause(
        self,
        *,
        pause_weight_unit: torch.Tensor,
        boundary_latent: torch.Tensor,
        unit_mask: torch.Tensor,
        pause_budget_win: torch.Tensor,
        state: StreamingRhythmState,
        reuse_prefix: bool,
        soft_pause_selection: bool,
        pause_topk_ratio_override: float | None = None,
    ) -> torch.Tensor:
        scores = pause_weight_unit.float().clamp_min(0.0)
        boundary_bias = self.config.pause_boundary_bias_weight * (
            self.config.pause_min_boundary_weight + boundary_latent.float().clamp_min(0.0)
        )
        scores = (scores + boundary_bias) * unit_mask
        sparse_scores = scores.new_zeros(scores.shape)
        topk_ratio = self.config.pause_topk_ratio if pause_topk_ratio_override is None else pause_topk_ratio_override
        topk_ratio = float(max(0.0, min(1.0, topk_ratio)))
        temperature = float(max(1e-4, self.config.pause_soft_temperature))
        for batch_idx in range(scores.size(0)):
            visible = int(unit_mask[batch_idx].sum().item())
            if visible <= 0:
                continue
            topk = max(1, int(round(visible * topk_ratio)))
            row_scores = scores[batch_idx, :visible]
            keep_k = min(topk, visible)
            if keep_k >= visible:
                sparse_scores[batch_idx, :visible] = row_scores
                continue
            values, indices = torch.topk(row_scores, k=keep_k, dim=0)
            if soft_pause_selection:
                threshold = values[-1].detach()
                gate = torch.sigmoid((row_scores - threshold) / temperature)
                sparse_scores[batch_idx, :visible] = row_scores * gate
            else:
                sparse_scores[batch_idx, indices] = values

        pause = sparse_scores.new_zeros(sparse_scores.shape)
        previous = state.previous_pause_exec
        for batch_idx in range(sparse_scores.size(0)):
            mask_row = unit_mask[batch_idx]
            budget_row = pause_budget_win[batch_idx]
            frontier = int(state.commit_frontier[batch_idx].item())
            if reuse_prefix and previous is not None and batch_idx < previous.size(0) and frontier > 0:
                valid_frontier = min(frontier, int(previous.size(1)), int(sparse_scores.size(1)))
                prefix = previous[batch_idx, :valid_frontier].float() * mask_row[:valid_frontier]
                pause[batch_idx, :valid_frontier] = prefix
            else:
                valid_frontier = 0
            tail_mask = mask_row.clone()
            if valid_frontier > 0:
                tail_mask[:valid_frontier] = 0.0
            if float(tail_mask.sum().item()) <= 0:
                continue
            remaining_budget = (budget_row - pause[batch_idx].sum()).clamp_min(0.0)
            if float(remaining_budget.item()) <= 0:
                continue
            tail_mask_row = tail_mask.unsqueeze(0)
            fallback = tail_mask_row / tail_mask_row.sum(dim=1, keepdim=True).clamp_min(1.0)
            tail_candidate = sparse_scores[batch_idx : batch_idx + 1].clamp_min(0.0) * tail_mask_row
            # Keep pause allocation differentiable wrt pause scores / pause budget.
            tail_values = self._renormalize_to_budget(
                tail_candidate + fallback * 1e-6,
                tail_mask_row,
                remaining_budget.view(1, 1),
            )[0]
            pause[batch_idx] = pause[batch_idx] + tail_values * tail_mask
        return pause * unit_mask

    def _compute_commit_frontier(
        self,
        *,
        state: StreamingRhythmState,
        unit_mask: torch.Tensor,
        open_run_mask: torch.Tensor | None,
        boundary_latent: torch.Tensor | None,
        force_full_commit: bool,
    ) -> torch.Tensor:
        batch_size, _ = unit_mask.shape
        active_len = unit_mask.long().sum(dim=1)
        if force_full_commit:
            return active_len.long()
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
        speech_budget_debt: torch.Tensor | None = None,
        pause_budget_debt: torch.Tensor | None = None,
    ) -> StreamingRhythmState:
        batch_size = dur_anchor_src.size(0)
        next_phase = state.phase_ptr.clone()
        next_backlog = state.backlog.clone()
        next_clock = state.clock_delta.clone()
        next_phase_progress = (
            state.phase_anchor_progress.clone()
            if state.phase_anchor_progress is not None
            else torch.zeros_like(next_phase)
        )
        next_phase_total = (
            state.phase_anchor_total.clone()
            if state.phase_anchor_total is not None
            else torch.zeros_like(next_phase)
        )
        next_speech_budget_debt = (
            speech_budget_debt.detach().clone()
            if speech_budget_debt is not None
            else (
                state.speech_budget_debt.clone()
                if state.speech_budget_debt is not None
                else torch.zeros_like(next_phase)
            )
        )
        next_pause_budget_debt = (
            pause_budget_debt.detach().clone()
            if pause_budget_debt is not None
            else (
                state.pause_budget_debt.clone()
                if state.pause_budget_debt is not None
                else torch.zeros_like(next_phase)
            )
        )
        for batch_idx in range(batch_size):
            prev = int(state.commit_frontier[batch_idx].item())
            curr = int(commit_frontier[batch_idx].item())
            prev_phase = state.phase_ptr[batch_idx].float().clamp(0.0, 1.0)
            prev_phase_progress = next_phase_progress[batch_idx].float().clamp_min(0.0)
            prev_phase_total = next_phase_total[batch_idx].float().clamp_min(1.0)
            visible_anchor = dur_anchor_src[batch_idx].float().clamp_min(0.0)
            visible_anchor_total = visible_anchor.sum().clamp_min(1.0)
            committed_anchor = visible_anchor[:curr].sum()
            # phase_ptr tracks committed progress, not current visible-prefix ratio.
            # Keep both progress and denominator monotonic to avoid phase rollback/jitter.
            monotonic_phase_progress = torch.maximum(prev_phase_progress, committed_anchor)
            monotonic_phase_total = torch.maximum(prev_phase_total, visible_anchor_total)
            raw_phase = (monotonic_phase_progress / monotonic_phase_total).clamp(0.0, 1.0)
            next_phase_progress[batch_idx] = monotonic_phase_progress
            next_phase_total[batch_idx] = monotonic_phase_total
            next_phase[batch_idx] = torch.maximum(prev_phase, raw_phase)
            if curr <= prev:
                next_phase_progress[batch_idx] = prev_phase_progress
                next_phase_total[batch_idx] = prev_phase_total
                next_phase[batch_idx] = prev_phase
                continue
            exec_prefix = effective_duration_exec[batch_idx, prev:curr].sum()
            src_prefix = dur_anchor_src[batch_idx, prev:curr].float().sum()
            delta = exec_prefix - src_prefix
            next_clock[batch_idx] = next_clock[batch_idx] + delta
            next_backlog[batch_idx] = next_clock[batch_idx].clamp_min(0.0)
        return StreamingRhythmState(
            phase_ptr=next_phase,
            backlog=next_backlog,
            clock_delta=next_clock,
            commit_frontier=commit_frontier.long(),
            previous_speech_exec=speech_duration_exec.detach(),
            previous_pause_exec=pause_after_exec.detach(),
            phase_anchor_progress=next_phase_progress,
            phase_anchor_total=next_phase_total,
            speech_budget_debt=next_speech_budget_debt,
            pause_budget_debt=next_pause_budget_debt,
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
        reuse_prefix: bool = True,
        force_full_commit: bool = False,
        pause_topk_ratio_override: float | None = None,
    ) -> RhythmExecution:
        unit_mask = unit_mask.float()
        speech_budget_win, pause_budget_win = self._lift_budgets_to_feasible_region(
            dur_anchor_src=dur_anchor_src,
            unit_mask=unit_mask,
            speech_budget_win=speech_budget_win,
            pause_budget_win=pause_budget_win,
            state=state,
            reuse_prefix=reuse_prefix,
        )
        feasible_speech_budget_delta = (speech_budget_win - planner.speech_budget_win.float()).clamp_min(0.0)
        feasible_pause_budget_delta = (pause_budget_win - planner.pause_budget_win.float()).clamp_min(0.0)
        execution_planner = RhythmPlannerOutputs(
            speech_budget_win=speech_budget_win,
            pause_budget_win=pause_budget_win,
            dur_logratio_unit=planner.dur_logratio_unit,
            pause_weight_unit=planner.pause_weight_unit,
            total_budget_win=speech_budget_win + pause_budget_win,
            pause_share_win=pause_budget_win / (speech_budget_win + pause_budget_win).clamp_min(1e-6),
            anchor_gate=planner.anchor_gate,
            boundary_latent=planner.boundary_latent,
            trace_context=planner.trace_context,
            source_boundary_cue=planner.source_boundary_cue,
        )
        execution_planner.feasible_speech_budget_delta = feasible_speech_budget_delta.detach()
        execution_planner.feasible_pause_budget_delta = feasible_pause_budget_delta.detach()
        execution_planner.feasible_total_budget_delta = (
            feasible_speech_budget_delta + feasible_pause_budget_delta
        ).detach()
        speech_duration_exec = self._project_speech(
            dur_anchor_src=dur_anchor_src,
            dur_logratio_unit=dur_logratio_unit,
            unit_mask=unit_mask,
            speech_budget_win=speech_budget_win,
            state=state,
            reuse_prefix=reuse_prefix,
        )
        pause_after_exec = self._project_pause(
            pause_weight_unit=pause_weight_unit,
            boundary_latent=boundary_latent,
            unit_mask=unit_mask,
            pause_budget_win=pause_budget_win,
            state=state,
            reuse_prefix=reuse_prefix,
            soft_pause_selection=bool(
                self.training and torch.is_grad_enabled() and self.config.pause_train_soft and not force_full_commit
            ),
            pause_topk_ratio_override=pause_topk_ratio_override,
        )
        effective_duration_exec = (speech_duration_exec + pause_after_exec) * unit_mask
        slot_schedule = build_interleaved_blank_slot_schedule(
            speech_duration_exec=speech_duration_exec,
            blank_duration_exec=pause_after_exec,
            unit_mask=unit_mask,
        )
        frame_plan = build_frame_plan_from_execution(
            dur_anchor_src=dur_anchor_src,
            speech_exec=speech_duration_exec,
            pause_exec=pause_after_exec,
            unit_mask=unit_mask,
        )
        commit_frontier = self._compute_commit_frontier(
            state=state,
            unit_mask=unit_mask,
            open_run_mask=open_run_mask,
            boundary_latent=boundary_latent,
            force_full_commit=force_full_commit,
        )
        next_state = self._advance_state(
            state=state,
            dur_anchor_src=dur_anchor_src,
            effective_duration_exec=effective_duration_exec,
            commit_frontier=commit_frontier,
            speech_duration_exec=speech_duration_exec,
            pause_after_exec=pause_after_exec,
            speech_budget_debt=feasible_speech_budget_delta,
            pause_budget_debt=feasible_pause_budget_delta,
        )
        return RhythmExecution(
            speech_duration_exec=speech_duration_exec,
            blank_duration_exec=pause_after_exec,
            pause_after_exec=pause_after_exec,
            effective_duration_exec=effective_duration_exec,
            commit_frontier=commit_frontier,
            slot_duration_exec=slot_schedule.slot_duration_exec,
            slot_mask=slot_schedule.slot_mask,
            slot_is_blank=slot_schedule.slot_is_blank,
            slot_unit_index=slot_schedule.slot_unit_index,
            frame_plan=frame_plan,
            planner=execution_planner,
            next_state=next_state,
        )
