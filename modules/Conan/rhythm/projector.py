from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .contracts import RhythmExecution, RhythmPlannerOutputs, StreamingRhythmState
from .feasibility import FeasibleBudgetProjection, lift_projector_budgets_to_feasible_region
from .frame_plan import build_frame_plan_from_execution, build_interleaved_blank_slot_schedule
from .source_boundary import resolve_boundary_score_unit


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
    pause_selection_mode: str = "sparse"
    use_boundary_commit_guard: bool = True
    build_render_plan: bool = True


def _apply_pause_boundary_gain(
    scores: torch.Tensor,
    *,
    boundary_score_unit: torch.Tensor,
    unit_mask: torch.Tensor,
    pause_min_boundary_weight: float,
    pause_boundary_bias_weight: float,
) -> torch.Tensor:
    unit_mask_f = unit_mask.float()
    boundary_gain = 1.0 + float(pause_boundary_bias_weight) * (
        float(pause_min_boundary_weight) + boundary_score_unit.float().clamp_min(0.0)
    )
    return scores.float().clamp_min(0.0) * boundary_gain * unit_mask_f


def _pad_or_truncate_rows(values: torch.Tensor, *, target_size: int) -> torch.Tensor:
    if values.size(1) == target_size:
        return values
    if values.size(1) > target_size:
        return values[:, :target_size]
    pad = values.new_zeros((values.size(0), target_size - values.size(1)))
    return torch.cat([values, pad], dim=1)


def _build_prefix_reuse_mask(
    unit_mask: torch.Tensor,
    commit_frontier: torch.Tensor,
    *,
    reuse_prefix: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    active = unit_mask.float()
    if not reuse_prefix or active.size(1) <= 0:
        prefix_mask = active.new_zeros(active.shape)
        return prefix_mask, active
    steps = torch.arange(active.size(1), device=active.device)[None, :]
    frontier = commit_frontier.long().clamp(min=0, max=max(int(active.size(1)), 0))
    prefix_mask = (steps < frontier[:, None]).to(dtype=active.dtype) * active
    tail_mask = (active - prefix_mask).clamp_min(0.0)
    return prefix_mask, tail_mask


def _prefix_sum_from_cumsum(cumsum: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    if cumsum.size(1) <= 0:
        return cumsum.new_zeros((cumsum.size(0),))
    padded = torch.cat([cumsum.new_zeros((cumsum.size(0), 1)), cumsum], dim=1)
    safe_counts = counts.long().clamp(min=0, max=cumsum.size(1))
    return padded.gather(1, safe_counts.unsqueeze(1)).squeeze(1)


def _allocate_pause_budget(
    *,
    candidate_scores: torch.Tensor,
    unit_mask: torch.Tensor,
    pause_budget_win: torch.Tensor,
    previous_pause_exec: torch.Tensor | None,
    commit_frontier: torch.Tensor,
    reuse_prefix: bool,
) -> torch.Tensor:
    total_units = candidate_scores.size(1)
    active = unit_mask.float()
    prefix_mask, tail_mask = _build_prefix_reuse_mask(
        active,
        commit_frontier,
        reuse_prefix=bool(reuse_prefix and previous_pause_exec is not None),
    )
    prefix_row = candidate_scores.new_zeros((candidate_scores.size(0), total_units))
    if reuse_prefix and previous_pause_exec is not None:
        previous_pause_exec = previous_pause_exec.float()
        previous_pause_exec = _pad_or_truncate_rows(previous_pause_exec, target_size=total_units)
        if previous_pause_exec.size(0) != candidate_scores.size(0):
            padded = prefix_row.clone()
            rows = min(int(previous_pause_exec.size(0)), int(candidate_scores.size(0)))
            padded[:rows] = previous_pause_exec[:rows]
            previous_pause_exec = padded
        prefix_row = previous_pause_exec * prefix_mask
    remaining_budget = (pause_budget_win.float() - prefix_row.sum(dim=1, keepdim=True)).clamp_min(0.0)
    tail_candidate = candidate_scores.float().clamp_min(0.0) * tail_mask
    tail_total = tail_candidate.sum(dim=1, keepdim=True)
    tail_slots = tail_mask.sum(dim=1, keepdim=True)
    fallback = torch.where(
        tail_slots > 0.0,
        tail_mask / tail_slots.clamp_min(1.0),
        torch.zeros_like(tail_mask),
    )
    tail_distribution = torch.where(
        tail_total > 1e-6,
        tail_candidate / tail_total.clamp_min(1e-6),
        fallback,
    )
    tail_values = tail_distribution * remaining_budget
    return (prefix_row + tail_values) * active


def _project_pause_impl(
    *,
    pause_weight_unit: torch.Tensor,
    boundary_score_unit: torch.Tensor,
    unit_mask: torch.Tensor,
    pause_budget_win: torch.Tensor,
    previous_pause_exec: torch.Tensor | None,
    commit_frontier: torch.Tensor,
    reuse_prefix: bool,
    soft_pause_selection: bool,
    topk_ratio: float,
    pause_min_boundary_weight: float,
    pause_boundary_bias_weight: float,
    temperature: float,
    pause_support_prob_unit: torch.Tensor | None = None,
    pause_allocation_weight_unit: torch.Tensor | None = None,
) -> torch.Tensor:
    unit_mask_f = unit_mask.float()
    use_split = pause_support_prob_unit is not None or pause_allocation_weight_unit is not None
    if use_split:
        support_scores = (
            pause_support_prob_unit.float().clamp(0.0, 1.0)
            if pause_support_prob_unit is not None
            else pause_weight_unit.float().clamp_min(0.0)
        )
        allocation_scores = (
            pause_allocation_weight_unit.float().clamp_min(0.0)
            if pause_allocation_weight_unit is not None
            else pause_weight_unit.float().clamp_min(0.0)
        )
        ranking_scores = _apply_pause_boundary_gain(
            support_scores,
            boundary_score_unit=boundary_score_unit,
            unit_mask=unit_mask,
            pause_min_boundary_weight=pause_min_boundary_weight,
            pause_boundary_bias_weight=pause_boundary_bias_weight,
        )
        candidate_scores = _apply_pause_boundary_gain(
            allocation_scores * support_scores,
            boundary_score_unit=boundary_score_unit,
            unit_mask=unit_mask,
            pause_min_boundary_weight=pause_min_boundary_weight,
            pause_boundary_bias_weight=pause_boundary_bias_weight,
        )
    else:
        ranking_scores = _apply_pause_boundary_gain(
            pause_weight_unit,
            boundary_score_unit=boundary_score_unit,
            unit_mask=unit_mask,
            pause_min_boundary_weight=pause_min_boundary_weight,
            pause_boundary_bias_weight=pause_boundary_bias_weight,
        )
        candidate_scores = ranking_scores
    visible = unit_mask.float().sum(dim=1).long()
    topk = torch.round(visible.float() * float(topk_ratio)).long()
    topk = torch.where(visible > 0, topk.clamp(min=1), topk).clamp(max=visible)
    valid_mask = unit_mask.float() > 0.5
    sparse_scores = candidate_scores.new_zeros(candidate_scores.shape)
    if bool(torch.any(topk > 0).item()):
        k_max = max(1, int(topk.max().item()))
        masked_scores = ranking_scores.masked_fill(~valid_mask, float("-inf"))
        top_values, top_indices = torch.topk(masked_scores, k=k_max, dim=1)
        rank_mask = (
            torch.arange(k_max, device=ranking_scores.device)[None, :] < topk[:, None]
        ) & (top_values > float("-inf"))
        if soft_pause_selection:
            full_keep = topk >= visible
            threshold_index = (topk.clamp_min(1) - 1).unsqueeze(1)
            threshold = top_values.gather(1, threshold_index)
            gated = candidate_scores * torch.sigmoid((ranking_scores - threshold) / float(temperature))
            sparse_scores = torch.where(full_keep[:, None], candidate_scores, gated) * unit_mask_f
        else:
            selector = torch.zeros_like(candidate_scores)
            selector.scatter_(1, top_indices, rank_mask.to(dtype=candidate_scores.dtype))
            sparse_scores = candidate_scores * selector
    return _allocate_pause_budget(
        candidate_scores=sparse_scores,
        unit_mask=unit_mask,
        pause_budget_win=pause_budget_win,
        previous_pause_exec=previous_pause_exec,
        commit_frontier=commit_frontier,
        reuse_prefix=reuse_prefix,
    )


def _project_pause_simple_impl(
    *,
    pause_weight_unit: torch.Tensor,
    boundary_score_unit: torch.Tensor,
    unit_mask: torch.Tensor,
    pause_budget_win: torch.Tensor,
    previous_pause_exec: torch.Tensor | None,
    commit_frontier: torch.Tensor,
    reuse_prefix: bool,
    pause_min_boundary_weight: float,
    pause_boundary_bias_weight: float,
    pause_support_prob_unit: torch.Tensor | None = None,
    pause_allocation_weight_unit: torch.Tensor | None = None,
) -> torch.Tensor:
    use_split = pause_support_prob_unit is not None or pause_allocation_weight_unit is not None
    if use_split:
        support_scores = (
            pause_support_prob_unit.float().clamp(0.0, 1.0)
            if pause_support_prob_unit is not None
            else pause_weight_unit.float().clamp_min(0.0)
        )
        allocation_scores = (
            pause_allocation_weight_unit.float().clamp_min(0.0)
            if pause_allocation_weight_unit is not None
            else pause_weight_unit.float().clamp_min(0.0)
        )
        scores = _apply_pause_boundary_gain(
            allocation_scores * support_scores,
            boundary_score_unit=boundary_score_unit,
            unit_mask=unit_mask,
            pause_min_boundary_weight=pause_min_boundary_weight,
            pause_boundary_bias_weight=pause_boundary_bias_weight,
        )
    else:
        scores = _apply_pause_boundary_gain(
            pause_weight_unit,
            boundary_score_unit=boundary_score_unit,
            unit_mask=unit_mask,
            pause_min_boundary_weight=pause_min_boundary_weight,
            pause_boundary_bias_weight=pause_boundary_bias_weight,
        )
    return _allocate_pause_budget(
        candidate_scores=scores,
        unit_mask=unit_mask,
        pause_budget_win=pause_budget_win,
        previous_pause_exec=previous_pause_exec,
        commit_frontier=commit_frontier,
        reuse_prefix=reuse_prefix,
    )


class StreamingRhythmProjector(nn.Module):
    def __init__(self, config: ProjectorConfig | None = None) -> None:
        super().__init__()
        self.config = config or ProjectorConfig()

    def init_state(self, batch_size: int, device: torch.device) -> StreamingRhythmState:
        zeros = torch.zeros(batch_size, device=device)
        return StreamingRhythmState(
            phase_ptr=zeros.clone(),
            clock_delta=zeros.clone(),
            commit_frontier=torch.zeros(batch_size, dtype=torch.long, device=device),
            previous_speech_exec=None,
            previous_pause_exec=None,
            phase_anchor=torch.stack([zeros.clone(), zeros.clone()], dim=-1),
            trace_tail_reuse_count=torch.zeros(batch_size, dtype=torch.long, device=device),
        )

    @staticmethod
    def _renormalize_to_budget(values: torch.Tensor, mask: torch.Tensor, budget: torch.Tensor) -> torch.Tensor:
        total = (values * mask).sum(dim=1, keepdim=True).clamp_min(1e-6)
        return values * (budget / total)

    @staticmethod
    def _project_row_to_bounded_sum(
        *,
        desired: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Project a single active row onto a bounded simplex.

        This preserves per-unit lower/upper bounds while matching the requested
        total mass whenever the target is feasible. It is used on the tail only,
        so committed prefixes remain frozen.
        """

        desired = desired.float().reshape(-1)
        lower = lower.float().reshape(-1)
        upper = torch.maximum(upper.float().reshape(-1), lower)
        if desired.numel() <= 0:
            return desired

        target_value = target.float().reshape(-1)[:1]
        lower_total = lower.sum().reshape(1)
        upper_total = upper.sum().reshape(1)
        clipped_target = torch.minimum(torch.maximum(target_value, lower_total), upper_total)

        if float((upper - lower).sum().item()) <= 1e-6 or float((clipped_target - lower_total).item()) <= 1e-6:
            return lower

        residual = (desired - lower).clamp_min(0.0)
        capacity = (upper - lower).clamp_min(0.0)
        alloc = torch.zeros_like(capacity)
        remaining = (clipped_target - lower_total).clamp_min(0.0)
        active = capacity > 1e-6
        max_iters = int(capacity.numel()) + 2

        for _ in range(max_iters):
            if float(remaining.item()) <= 1e-6 or not bool(active.any().item()):
                break
            active_score = residual * active.float()
            if float(active_score.sum().item()) <= 1e-6:
                active_score = capacity * active.float()
            denom = active_score.sum().clamp_min(1e-6)
            proposal = remaining * (active_score / denom)
            new_alloc = torch.minimum(alloc + proposal, capacity)
            gained = new_alloc - alloc
            alloc = new_alloc
            remaining = (remaining - gained.sum()).clamp_min(0.0)
            active = (capacity - alloc) > 1e-6

        if float(remaining.item()) > 1e-6:
            slack = (capacity - alloc).clamp_min(0.0)
            if float(slack.sum().item()) > 1e-6:
                alloc = torch.minimum(
                    alloc + remaining * (slack / slack.sum().clamp_min(1e-6)),
                    capacity,
                )

        return lower + alloc

    def _lift_budgets_to_feasible_region(
        self,
        *,
        dur_anchor_src: torch.Tensor,
        unit_mask: torch.Tensor,
        speech_budget_win: torch.Tensor,
        pause_budget_win: torch.Tensor,
        state: StreamingRhythmState,
        reuse_prefix: bool,
    ) -> FeasibleBudgetProjection:
        return lift_projector_budgets_to_feasible_region(
            dur_anchor_src=dur_anchor_src,
            unit_mask=unit_mask,
            speech_budget_win=speech_budget_win,
            pause_budget_win=pause_budget_win,
            previous_speech_exec=state.previous_speech_exec,
            previous_pause_exec=state.previous_pause_exec,
            commit_frontier=state.commit_frontier,
            reuse_prefix=reuse_prefix,
            min_speech_frames=float(self.config.min_speech_frames),
            max_speech_expand=float(self.config.max_speech_expand),
        )

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
        scaled = dur_anchor_src.float() * torch.exp(dur_logratio_unit.float())
        scaled = scaled * unit_mask
        speech_rows = []
        for batch_idx in range(scaled.size(0)):
            mask_row = unit_mask[batch_idx : batch_idx + 1]
            budget_row = speech_budget_win[batch_idx].reshape(1, 1)
            frontier = int(state.commit_frontier[batch_idx].item())
            prefix_row = scaled.new_zeros((1, scaled.size(1)))
            if reuse_prefix and state.previous_speech_exec is not None and batch_idx < state.previous_speech_exec.size(0) and frontier > 0:
                valid_frontier = min(frontier, int(state.previous_speech_exec.size(1)), int(scaled.size(1)))
                prefix = state.previous_speech_exec[batch_idx : batch_idx + 1, :valid_frontier].float() * mask_row[:, :valid_frontier]
                if valid_frontier < scaled.size(1):
                    prefix_row = torch.cat([prefix, prefix.new_zeros((1, scaled.size(1) - valid_frontier))], dim=1)
                else:
                    prefix_row = prefix
            else:
                valid_frontier = 0
            tail_mask = mask_row.clone()
            if valid_frontier > 0:
                tail_mask = torch.cat(
                    [tail_mask.new_zeros((1, valid_frontier)), tail_mask[:, valid_frontier:]],
                    dim=1,
                )
            remaining_budget = (budget_row - prefix_row.sum(dim=1, keepdim=True)).clamp_min(0.0)
            if float(tail_mask.sum().item()) <= 0:
                speech_rows.append(prefix_row)
                continue

            tail_active = tail_mask.squeeze(0) > 0
            desired_tail = scaled[batch_idx].float().clamp_min(0.0)[tail_active]
            lower_tail = desired_tail.new_full(
                desired_tail.shape,
                float(self.config.min_speech_frames),
            )
            max_expand = float(self.config.max_speech_expand)
            if max_expand > 0.0:
                upper_tail = torch.maximum(
                    dur_anchor_src[batch_idx].float().clamp_min(0.0)[tail_active] * max_expand,
                    lower_tail,
                )
            else:
                upper_tail = lower_tail.new_full(
                    lower_tail.shape,
                    float(remaining_budget.item()) + float(self.config.min_speech_frames),
                )
            projected_tail = self._project_row_to_bounded_sum(
                desired=desired_tail,
                lower=lower_tail,
                upper=upper_tail,
                target=remaining_budget.squeeze(0),
            )
            tail_row = scaled.new_zeros((scaled.size(1),), dtype=torch.float32)
            tail_row[tail_active] = projected_tail
            speech_rows.append(prefix_row + tail_row.unsqueeze(0))
        speech = torch.cat(speech_rows, dim=0)
        return speech * unit_mask

    def _project_pause(
        self,
        *,
        pause_weight_unit: torch.Tensor,
        boundary_score_unit: torch.Tensor,
        unit_mask: torch.Tensor,
        pause_budget_win: torch.Tensor,
        state: StreamingRhythmState,
        reuse_prefix: bool,
        soft_pause_selection: bool,
        pause_topk_ratio_override: float | None = None,
        pause_support_prob_unit: torch.Tensor | None = None,
        pause_allocation_weight_unit: torch.Tensor | None = None,
    ) -> torch.Tensor:
        selection_mode = str(self.config.pause_selection_mode or "sparse").strip().lower()
        if selection_mode == "simple":
            return _project_pause_simple_impl(
                pause_weight_unit=pause_weight_unit,
                boundary_score_unit=boundary_score_unit,
                unit_mask=unit_mask,
                pause_budget_win=pause_budget_win,
                previous_pause_exec=state.previous_pause_exec,
                commit_frontier=state.commit_frontier,
                reuse_prefix=reuse_prefix,
                pause_min_boundary_weight=float(self.config.pause_min_boundary_weight),
                pause_boundary_bias_weight=float(self.config.pause_boundary_bias_weight),
                pause_support_prob_unit=pause_support_prob_unit,
                pause_allocation_weight_unit=pause_allocation_weight_unit,
            )
        topk_ratio = self.config.pause_topk_ratio if pause_topk_ratio_override is None else pause_topk_ratio_override
        topk_ratio = float(max(0.0, min(1.0, topk_ratio)))
        temperature = float(max(1e-4, self.config.pause_soft_temperature))
        return _project_pause_impl(
            pause_weight_unit=pause_weight_unit,
            boundary_score_unit=boundary_score_unit,
            unit_mask=unit_mask,
            pause_budget_win=pause_budget_win,
            previous_pause_exec=state.previous_pause_exec,
            commit_frontier=state.commit_frontier,
            reuse_prefix=reuse_prefix,
            soft_pause_selection=soft_pause_selection,
            topk_ratio=topk_ratio,
            pause_min_boundary_weight=float(self.config.pause_min_boundary_weight),
            pause_boundary_bias_weight=float(self.config.pause_boundary_bias_weight),
            temperature=temperature,
            pause_support_prob_unit=pause_support_prob_unit,
            pause_allocation_weight_unit=pause_allocation_weight_unit,
        )

    def _compute_commit_frontier(
        self,
        *,
        state: StreamingRhythmState,
        unit_mask: torch.Tensor,
        open_run_mask: torch.Tensor | None,
        boundary_score_unit: torch.Tensor | None,
        force_full_commit: bool,
    ) -> torch.Tensor:
        active_len = unit_mask.long().sum(dim=1)
        if force_full_commit:
            return active_len.long()
        if open_run_mask is None:
            open_run_mask = torch.zeros_like(unit_mask, dtype=torch.long)
        steps = torch.arange(unit_mask.size(1), device=unit_mask.device)[None, :]
        visible_mask = steps < active_len[:, None]
        open_visible = (open_run_mask.long() > 0) & visible_mask
        seen_open = torch.cumsum(open_visible.long(), dim=1)
        closed_prefix = (visible_mask & (seen_open == 0)).long().sum(dim=1)
        release_cap = (active_len - int(self.config.tail_hold_units)).clamp(min=0)
        candidate = torch.minimum(release_cap, closed_prefix)
        if (
            bool(self.config.use_boundary_commit_guard)
            and boundary_score_unit is not None
            and boundary_score_unit.size(1) > 0
        ):
            guard_mask = (candidate > 0) & (candidate < active_len)
            boundary_index = (candidate - 1).clamp_min(0)
            boundary_value = boundary_score_unit.float().gather(1, boundary_index.unsqueeze(1)).squeeze(1)
            candidate = torch.where(
                guard_mask & (boundary_value < float(self.config.boundary_commit_threshold)),
                (candidate - 1).clamp_min(0),
                candidate,
            )
        return torch.maximum(state.commit_frontier.long(), candidate.long())

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
        next_phase = state.phase_ptr.float().clone()
        next_clock = state.clock_delta.float().clone()
        next_phase_anchor = (
            state.phase_anchor.float().clone()
            if state.phase_anchor is not None
            else torch.zeros(next_phase.size(0), 2, device=next_phase.device, dtype=next_phase.dtype)
        )
        prev = state.commit_frontier.long().clamp(min=0, max=dur_anchor_src.size(1))
        curr = commit_frontier.long().clamp(min=0, max=dur_anchor_src.size(1))
        advance_mask = curr > prev
        prev_phase = state.phase_ptr.float().clamp(0.0, 1.0)
        prev_phase_progress = next_phase_anchor[:, 0].float().clamp_min(0.0)
        prev_phase_total = next_phase_anchor[:, 1].float().clamp_min(1.0)
        visible_anchor = dur_anchor_src.float().clamp_min(0.0)
        visible_anchor_total = visible_anchor.sum(dim=1).clamp_min(1.0)
        anchor_cumsum = torch.cumsum(visible_anchor, dim=1)
        exec_cumsum = torch.cumsum(effective_duration_exec.float(), dim=1)
        committed_anchor = _prefix_sum_from_cumsum(anchor_cumsum, curr)
        monotonic_phase_progress = torch.maximum(prev_phase_progress, committed_anchor)
        monotonic_phase_total = torch.maximum(prev_phase_total, visible_anchor_total)
        raw_phase = (monotonic_phase_progress / monotonic_phase_total.clamp_min(1.0)).clamp(0.0, 1.0)
        next_phase = torch.where(
            advance_mask,
            torch.maximum(prev_phase, raw_phase),
            prev_phase,
        )
        next_phase_anchor = torch.stack(
            [
                torch.where(advance_mask, monotonic_phase_progress, prev_phase_progress),
                torch.where(advance_mask, monotonic_phase_total, prev_phase_total),
            ],
            dim=-1,
        )
        exec_delta = _prefix_sum_from_cumsum(exec_cumsum, curr) - _prefix_sum_from_cumsum(exec_cumsum, prev)
        src_delta = _prefix_sum_from_cumsum(anchor_cumsum, curr) - _prefix_sum_from_cumsum(anchor_cumsum, prev)
        next_clock = next_clock + torch.where(advance_mask, exec_delta - src_delta, torch.zeros_like(next_clock))
        return StreamingRhythmState(
            phase_ptr=next_phase,
            clock_delta=next_clock,
            commit_frontier=commit_frontier.long(),
            previous_speech_exec=speech_duration_exec.detach(),
            previous_pause_exec=pause_after_exec.detach(),
            phase_anchor=next_phase_anchor,
            trace_tail_reuse_count=(
                state.trace_tail_reuse_count.long().detach().clone()
                if state.trace_tail_reuse_count is not None
                else torch.zeros_like(commit_frontier.long())
            ),
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
        boundary_score_unit: torch.Tensor,
        state: StreamingRhythmState,
        open_run_mask: torch.Tensor | None = None,
        planner: RhythmPlannerOutputs,
        reuse_prefix: bool = True,
        force_full_commit: bool = False,
        pause_topk_ratio_override: float | None = None,
        soft_pause_selection_override: bool | None = None,
    ) -> RhythmExecution:
        unit_mask = unit_mask.float()
        feasibility = self._lift_budgets_to_feasible_region(
            dur_anchor_src=dur_anchor_src,
            unit_mask=unit_mask,
            speech_budget_win=speech_budget_win,
            pause_budget_win=pause_budget_win,
            state=state,
            reuse_prefix=reuse_prefix,
        )
        speech_budget_win = feasibility.speech_budget_win
        pause_budget_win = feasibility.pause_budget_win
        feasible_speech_budget_delta = feasibility.speech_budget_delta
        feasible_pause_budget_delta = feasibility.pause_budget_delta
        planner_boundary_score = resolve_boundary_score_unit(planner, fallback=boundary_score_unit)
        execution_planner = RhythmPlannerOutputs(
            speech_budget_win=speech_budget_win,
            pause_budget_win=pause_budget_win,
            dur_logratio_unit=planner.dur_logratio_unit,
            pause_weight_unit=planner.pause_weight_unit,
            boundary_score_unit=planner_boundary_score,
            trace_context=planner.trace_context,
            pause_support_prob_unit=getattr(planner, "pause_support_prob_unit", None),
            pause_allocation_weight_unit=getattr(planner, "pause_allocation_weight_unit", None),
            pause_support_logit_unit=getattr(planner, "pause_support_logit_unit", None),
            pause_run_length_unit=getattr(planner, "pause_run_length_unit", None),
            pause_breath_debt_unit=getattr(planner, "pause_breath_debt_unit", None),
            source_boundary_cue=planner.source_boundary_cue,
            trace_reliability=getattr(planner, "trace_reliability", None),
            local_trace_path_weight=getattr(planner, "local_trace_path_weight", None),
            boundary_trace_path_weight=getattr(planner, "boundary_trace_path_weight", None),
            trace_phase_gap=getattr(planner, "trace_phase_gap", None),
            trace_phase_gap_runtime=getattr(planner, "trace_phase_gap_runtime", None),
            trace_phase_gap_anchor=getattr(planner, "trace_phase_gap_anchor", None),
            trace_coverage_alpha=getattr(planner, "trace_coverage_alpha", None),
            trace_blend=getattr(planner, "trace_blend", None),
            trace_tail_reuse_count=getattr(planner, "trace_tail_reuse_count", None),
            trace_tail_alpha=getattr(planner, "trace_tail_alpha", None),
            trace_gap_alpha=getattr(planner, "trace_gap_alpha", None),
            trace_reuse_alpha=getattr(planner, "trace_reuse_alpha", None),
        )
        execution_planner.raw_speech_budget_win = getattr(
            planner,
            "raw_speech_budget_win",
            planner.speech_budget_win,
        )
        execution_planner.raw_pause_budget_win = getattr(
            planner,
            "raw_pause_budget_win",
            planner.pause_budget_win,
        )
        execution_planner.feasible_speech_budget_delta = feasible_speech_budget_delta
        execution_planner.feasible_pause_budget_delta = feasible_pause_budget_delta
        execution_planner.feasible_total_budget_delta = feasibility.total_budget_delta
        selection_mode = str(self.config.pause_selection_mode or "sparse").strip().lower()
        resolved_pause_topk_ratio = (
            self.config.pause_topk_ratio if pause_topk_ratio_override is None else pause_topk_ratio_override
        )
        resolved_pause_topk_ratio = float(max(0.0, min(1.0, resolved_pause_topk_ratio)))
        soft_pause_selection_active = (
            bool(soft_pause_selection_override)
            if soft_pause_selection_override is not None
            else bool(
                self.training
                and torch.is_grad_enabled()
                and self.config.pause_train_soft
                and not force_full_commit
            )
        )
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
            boundary_score_unit=planner_boundary_score,
            unit_mask=unit_mask,
            pause_budget_win=pause_budget_win,
            state=state,
            reuse_prefix=reuse_prefix,
            soft_pause_selection=soft_pause_selection_active,
            pause_topk_ratio_override=pause_topk_ratio_override,
            pause_support_prob_unit=getattr(planner, "pause_support_prob_unit", None),
            pause_allocation_weight_unit=getattr(planner, "pause_allocation_weight_unit", None),
        )
        planner_batch = pause_budget_win.size(0)
        planner_device = pause_budget_win.device
        planner_dtype = pause_budget_win.dtype
        execution_planner.pause_topk_ratio = torch.full(
            (planner_batch, 1),
            resolved_pause_topk_ratio,
            dtype=planner_dtype,
            device=planner_device,
        )
        execution_planner.pause_soft_selection_active = torch.full(
            (planner_batch, 1),
            1.0 if soft_pause_selection_active else 0.0,
            dtype=planner_dtype,
            device=planner_device,
        )
        execution_planner.projector_force_full_commit = torch.full(
            (planner_batch, 1),
            1.0 if force_full_commit else 0.0,
            dtype=planner_dtype,
            device=planner_device,
        )
        execution_planner.pause_selection_mode_id = torch.full(
            (planner_batch, 1),
            1.0 if selection_mode == "sparse" else 0.0,
            dtype=planner_dtype,
            device=planner_device,
        )
        effective_duration_exec = (speech_duration_exec + pause_after_exec) * unit_mask
        slot_schedule = None
        frame_plan = None
        slot_duration_exec = None
        slot_mask = None
        slot_is_blank = None
        slot_unit_index = None
        if bool(self.config.build_render_plan):
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
            slot_duration_exec = slot_schedule.slot_duration_exec
            slot_mask = slot_schedule.slot_mask
            slot_is_blank = slot_schedule.slot_is_blank
            slot_unit_index = slot_schedule.slot_unit_index
        commit_frontier = self._compute_commit_frontier(
            state=state,
            unit_mask=unit_mask,
            open_run_mask=open_run_mask,
            boundary_score_unit=planner_boundary_score,
            force_full_commit=force_full_commit,
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
            blank_duration_exec=pause_after_exec,
            pause_after_exec=pause_after_exec,
            effective_duration_exec=effective_duration_exec,
            commit_frontier=commit_frontier,
            slot_duration_exec=slot_duration_exec,
            slot_mask=slot_mask,
            slot_is_blank=slot_is_blank,
            slot_unit_index=slot_unit_index,
            frame_plan=frame_plan,
            planner=execution_planner,
            next_state=next_state,
        )
