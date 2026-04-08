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
    debt_leak: float = 0.05
    debt_max_abs: float = 12.0
    debt_correction_horizon: float = 4.0


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


def _resolve_allocation_mask(
    unit_mask: torch.Tensor,
    allocation_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    active = unit_mask.float()
    if allocation_mask is None:
        return active
    return allocation_mask.float() * active


def _assert_prefix_contiguous(unit_mask: torch.Tensor) -> None:
    active = (unit_mask.float() > 0.5).float()
    visible_len = active.sum(dim=1).long()
    steps = torch.arange(active.size(1), device=active.device)[None, :]
    expected = (steps < visible_len[:, None]).float()
    if bool((active != expected).any().item()):
        raise AssertionError("unit_mask must represent a contiguous visible prefix for streaming projection")


def _sanitize_tail_allocation_mask(
    unit_mask: torch.Tensor,
    commit_frontier: torch.Tensor,
    allocation_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    active = _resolve_allocation_mask(unit_mask, allocation_mask)
    steps = torch.arange(active.size(1), device=active.device)[None, :]
    frontier = commit_frontier.long().to(device=active.device)
    if frontier.dim() == 0:
        frontier = frontier.unsqueeze(0)
    tail_mask = (steps >= frontier[:, None]).float()
    return active * tail_mask


def _build_allocation_masks(
    unit_mask: torch.Tensor,
    commit_frontier: torch.Tensor,
    *,
    reuse_prefix: bool,
    allocation_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    active = unit_mask.float()
    prefix_mask, _ = _build_prefix_reuse_mask(
        active,
        commit_frontier,
        reuse_prefix=reuse_prefix,
    )
    allocation_active = _resolve_allocation_mask(unit_mask, allocation_mask)
    tail_mask = allocation_active * (1.0 - prefix_mask)
    return active, prefix_mask, tail_mask


def _build_pause_selection_mask(
    unit_mask: torch.Tensor,
    commit_frontier: torch.Tensor,
    *,
    reuse_prefix: bool,
    previous_pause_exec: torch.Tensor | None,
    allocation_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    _, _, tail_mask = _build_allocation_masks(
        unit_mask,
        commit_frontier,
        reuse_prefix=bool(reuse_prefix and previous_pause_exec is not None),
        allocation_mask=allocation_mask,
    )
    return tail_mask


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
    allocation_mask: torch.Tensor | None,
    pause_budget_win: torch.Tensor,
    previous_pause_exec: torch.Tensor | None,
    commit_frontier: torch.Tensor,
    reuse_prefix: bool,
    boundary_score_unit: torch.Tensor | None = None,
) -> torch.Tensor:
    total_units = candidate_scores.size(1)
    active, prefix_mask, tail_mask = _build_allocation_masks(
        unit_mask,
        commit_frontier,
        reuse_prefix=bool(reuse_prefix and previous_pause_exec is not None),
        allocation_mask=allocation_mask,
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
    if boundary_score_unit is not None:
        boundary_prior = boundary_score_unit.float().clamp_min(0.0) * tail_mask
        boundary_total = boundary_prior.sum(dim=1, keepdim=True)
        tail_indices = torch.arange(total_units, device=tail_mask.device, dtype=torch.long)[None, :].expand_as(
            tail_mask.long()
        )
        last_tail_index = torch.where(
            tail_mask > 0.5,
            tail_indices,
            torch.full_like(tail_mask, -1, dtype=torch.long),
        ).max(dim=1).values
        last_slot_fallback = torch.zeros_like(tail_mask)
        valid_last = last_tail_index >= 0
        if bool(valid_last.any().item()):
            last_slot_fallback[valid_last] = last_slot_fallback[valid_last].scatter(
                1,
                last_tail_index[valid_last].unsqueeze(1),
                torch.ones((int(valid_last.sum().item()), 1), device=tail_mask.device, dtype=tail_mask.dtype),
            )
        fallback = torch.where(
            boundary_total > 1e-6,
            boundary_prior / boundary_total.clamp_min(1e-6),
            last_slot_fallback,
        )
    else:
        tail_indices = torch.arange(total_units, device=tail_mask.device, dtype=torch.long)[None, :].expand_as(
            tail_mask.long()
        )
        last_tail_index = torch.where(
            tail_mask > 0.5,
            tail_indices,
            torch.full_like(tail_mask, -1, dtype=torch.long),
        ).max(dim=1).values
        fallback = torch.zeros_like(tail_mask)
        valid_last = last_tail_index >= 0
        if bool(valid_last.any().item()):
            fallback[valid_last] = fallback[valid_last].scatter(
                1,
                last_tail_index[valid_last].unsqueeze(1),
                torch.ones((int(valid_last.sum().item()), 1), device=tail_mask.device, dtype=tail_mask.dtype),
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
    allocation_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    selection_mask = _build_pause_selection_mask(
        unit_mask,
        commit_frontier,
        reuse_prefix=reuse_prefix,
        previous_pause_exec=previous_pause_exec,
        allocation_mask=allocation_mask,
    )
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
            unit_mask=selection_mask,
            pause_min_boundary_weight=pause_min_boundary_weight,
            pause_boundary_bias_weight=pause_boundary_bias_weight,
        )
        candidate_scores = allocation_scores * support_scores * selection_mask
    else:
        ranking_scores = _apply_pause_boundary_gain(
            pause_weight_unit,
            boundary_score_unit=boundary_score_unit,
            unit_mask=selection_mask,
            pause_min_boundary_weight=pause_min_boundary_weight,
            pause_boundary_bias_weight=pause_boundary_bias_weight,
        )
        candidate_scores = ranking_scores
    visible = selection_mask.sum(dim=1).long()
    topk = torch.round(visible.float() * float(topk_ratio)).long()
    topk = torch.where(visible > 0, topk.clamp(min=1), topk).clamp(max=visible)
    valid_mask = selection_mask > 0.5
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
            sparse_scores = torch.where(full_keep[:, None], candidate_scores, gated) * selection_mask
        else:
            selector = torch.zeros_like(candidate_scores)
            selector.scatter_(1, top_indices, rank_mask.to(dtype=candidate_scores.dtype))
            sparse_scores = candidate_scores * selector * selection_mask
    leaked_scores = sparse_scores * (1.0 - selection_mask)
    if bool((leaked_scores.abs() > 1.0e-6).any().item()):
        raise AssertionError("pause sparse scores leaked outside tail allocation domain")
    return _allocate_pause_budget(
        candidate_scores=sparse_scores,
        unit_mask=unit_mask,
        allocation_mask=allocation_mask,
        pause_budget_win=pause_budget_win,
        previous_pause_exec=previous_pause_exec,
        commit_frontier=commit_frontier,
        reuse_prefix=reuse_prefix,
        boundary_score_unit=boundary_score_unit,
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
    allocation_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    allocation_mask_f = _resolve_allocation_mask(unit_mask, allocation_mask)
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
            unit_mask=allocation_mask_f,
            pause_min_boundary_weight=pause_min_boundary_weight,
            pause_boundary_bias_weight=pause_boundary_bias_weight,
        )
    else:
        scores = _apply_pause_boundary_gain(
            pause_weight_unit,
            boundary_score_unit=boundary_score_unit,
            unit_mask=allocation_mask_f,
            pause_min_boundary_weight=pause_min_boundary_weight,
            pause_boundary_bias_weight=pause_boundary_bias_weight,
        )
    return _allocate_pause_budget(
        candidate_scores=scores,
        unit_mask=unit_mask,
        allocation_mask=allocation_mask,
        pause_budget_win=pause_budget_win,
        previous_pause_exec=previous_pause_exec,
        commit_frontier=commit_frontier,
        reuse_prefix=reuse_prefix,
        boundary_score_unit=boundary_score_unit,
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
            phase_anchor_visible_total=zeros.clone(),
            trace_tail_reuse_count=torch.zeros(batch_size, dtype=torch.long, device=device),
        )

    @staticmethod
    def _renormalize_to_budget(values: torch.Tensor, mask: torch.Tensor, budget: torch.Tensor) -> torch.Tensor:
        total = (values * mask).sum(dim=1, keepdim=True).clamp_min(1e-6)
        return values * (budget / total)

    @staticmethod
    def _validate_prefix_reuse_consistency(
        *,
        state: StreamingRhythmState,
        unit_mask: torch.Tensor,
        reuse_prefix: bool,
    ) -> None:
        if not reuse_prefix:
            return
        frontier = state.commit_frontier.long().to(device=unit_mask.device)
        if frontier.numel() <= 0:
            return
        visible_len = unit_mask.float().sum(dim=1).long()
        if bool((visible_len < frontier).any().item()):
            raise ValueError(
                "reuse_prefix requires append-only visible prefixes: "
                "current chunk is shorter than committed frontier."
            )
        steps = torch.arange(unit_mask.size(1), device=unit_mask.device)[None, :]
        committed_mask = steps < frontier[:, None]
        committed_visible = unit_mask.float() > 0.5
        if bool((committed_mask & (~committed_visible)).any().item()):
            raise ValueError(
                "reuse_prefix requires committed prefix units to remain visible in the current chunk."
            )
        for name, previous_exec in (
            ("previous_speech_exec", state.previous_speech_exec),
            ("previous_pause_exec", state.previous_pause_exec),
        ):
            if previous_exec is None:
                continue
            if previous_exec.size(1) < int(frontier.max().item()):
                raise ValueError(
                    f"reuse_prefix requires {name} width >= committed frontier."
                )

    @staticmethod
    def _compute_local_rho(
        *,
        speech_exec: torch.Tensor,
        segment_mask: torch.Tensor | None,
        unit_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        if segment_mask is None:
            return None
        segment_mask_f = _resolve_allocation_mask(unit_mask, segment_mask)
        if float(segment_mask_f.sum().item()) <= 0.0:
            return torch.zeros_like(segment_mask_f)
        seg_exec = speech_exec.float() * segment_mask_f
        seg_total = seg_exec.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        seg_cum = torch.cumsum(seg_exec, dim=1)
        return (seg_cum / seg_total).clamp(0.0, 1.0) * segment_mask_f

    @staticmethod
    def _collapse_intra_phrase_alpha(
        *,
        local_rho_unit: torch.Tensor | None,
        segment_mask: torch.Tensor | None,
        unit_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        if local_rho_unit is None or segment_mask is None:
            return None
        segment_mask_f = _resolve_allocation_mask(unit_mask, segment_mask)
        valid = segment_mask_f > 0.5
        if not bool(valid.any().item()):
            return torch.zeros((unit_mask.size(0), 1), device=unit_mask.device, dtype=unit_mask.dtype)
        masked = torch.where(valid, local_rho_unit.float(), torch.full_like(local_rho_unit.float(), -1.0))
        alpha = masked.max(dim=1, keepdim=True).values.clamp_min(0.0)
        return alpha

    @staticmethod
    def _sample_phrase_trace_context(
        *,
        ref_phrase_trace: torch.Tensor | None,
        local_rho_unit: torch.Tensor | None,
        segment_mask: torch.Tensor | None,
        unit_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        if ref_phrase_trace is None or local_rho_unit is None or segment_mask is None:
            return None
        trace = ref_phrase_trace
        if trace.dim() == 4:
            if trace.size(1) != 1:
                return None
            trace = trace[:, 0]
        if trace.dim() != 3:
            return None
        batch_size, bins, trace_dim = trace.shape
        if local_rho_unit.size(0) != batch_size:
            return None
        segment_mask_f = _resolve_allocation_mask(unit_mask, segment_mask)
        if float(segment_mask_f.sum().item()) <= 0.0:
            return torch.zeros(
                (unit_mask.size(0), unit_mask.size(1), trace_dim),
                device=unit_mask.device,
                dtype=trace.dtype,
            )
        trace = trace.to(device=unit_mask.device, dtype=torch.float32)
        rho = local_rho_unit.float().to(device=unit_mask.device).clamp(0.0, 1.0)
        if bins <= 1:
            context = trace[:, :1, :].expand(-1, unit_mask.size(1), -1)
            return context * segment_mask_f.unsqueeze(-1)
        pos = rho * float(bins - 1)
        low = pos.floor().long().clamp(min=0, max=bins - 1)
        high = pos.ceil().long().clamp(min=0, max=bins - 1)
        batch_index = torch.arange(batch_size, device=unit_mask.device)[:, None]
        low_ctx = trace[batch_index, low]
        high_ctx = trace[batch_index, high]
        frac = (pos - low.float()).unsqueeze(-1)
        context = low_ctx * (1.0 - frac) + high_ctx * frac
        return context * segment_mask_f.unsqueeze(-1)

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
        segment_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scaled = dur_anchor_src.float() * torch.exp(dur_logratio_unit.float())
        active_mask = unit_mask.float()
        segment_mask_f = _resolve_allocation_mask(unit_mask, segment_mask)
        scaled = scaled * active_mask
        speech_rows = []
        for batch_idx in range(scaled.size(0)):
            mask_row = active_mask[batch_idx : batch_idx + 1]
            segment_row = segment_mask_f[batch_idx : batch_idx + 1]
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
            tail_mask = segment_row.clone()
            if valid_frontier > 0:
                tail_mask[:, :valid_frontier] = 0.0
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
        return speech * active_mask

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
        allocation_mask: torch.Tensor | None = None,
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
                allocation_mask=allocation_mask,
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
            allocation_mask=allocation_mask,
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
        _assert_prefix_contiguous(unit_mask)
        active_len = unit_mask.long().sum(dim=1)
        if force_full_commit:
            return active_len.long()
        if open_run_mask is None:
            open_run_mask = torch.zeros_like(unit_mask, dtype=torch.long)
        steps = torch.arange(unit_mask.size(1), device=unit_mask.device)[None, :]
        visible_mask = unit_mask.float() > 0.5
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

    @staticmethod
    def _resolve_planned_commit_frontier_override(
        *,
        planned_commit_frontier: torch.Tensor,
        state: StreamingRhythmState,
        unit_mask: torch.Tensor,
    ) -> torch.Tensor:
        commit_frontier = planned_commit_frontier.to(device=unit_mask.device).long()
        if commit_frontier.dim() == 0:
            commit_frontier = commit_frontier.unsqueeze(0)
        elif commit_frontier.dim() > 1:
            commit_frontier = commit_frontier.reshape(commit_frontier.size(0), -1)[:, 0]
        if commit_frontier.size(0) != unit_mask.size(0):
            raise ValueError(
                "planned_commit_frontier batch mismatch: "
                f"got {tuple(commit_frontier.shape)} for batch_size={unit_mask.size(0)}"
            )
        visible_len = unit_mask.float().sum(dim=1).long()
        commit_frontier = commit_frontier.clamp(min=0, max=unit_mask.size(1))
        commit_frontier = torch.minimum(commit_frontier, visible_len)
        return torch.maximum(state.commit_frontier.long().to(device=unit_mask.device), commit_frontier)

    def _advance_state(
        self,
        *,
        state: StreamingRhythmState,
        dur_anchor_src: torch.Tensor,
        unit_mask: torch.Tensor,
        effective_duration_exec: torch.Tensor,
        commit_frontier: torch.Tensor,
        speech_duration_exec: torch.Tensor,
        pause_after_exec: torch.Tensor,
        intra_phrase_alpha: torch.Tensor | None = None,
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
        visible_anchor = dur_anchor_src.float().clamp_min(0.0) * unit_mask.float()
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
        raw_clock_delta = torch.where(advance_mask, exec_delta - src_delta, torch.zeros_like(next_clock))
        correction_horizon = float(max(0.0, self.config.debt_correction_horizon))
        if correction_horizon > 0.0:
            raw_clock_delta = raw_clock_delta.clamp(-correction_horizon, correction_horizon)
        debt_leak = float(min(max(self.config.debt_leak, 0.0), 1.0))
        if debt_leak > 0.0:
            next_clock = next_clock * (1.0 - debt_leak)
        next_clock = next_clock + raw_clock_delta
        debt_max_abs = float(max(0.0, self.config.debt_max_abs))
        if debt_max_abs > 0.0:
            next_clock = next_clock.clamp(-debt_max_abs, debt_max_abs)
        return StreamingRhythmState(
            phase_ptr=next_phase,
            clock_delta=next_clock,
            commit_frontier=commit_frontier.long(),
            previous_speech_exec=speech_duration_exec.detach(),
            previous_pause_exec=pause_after_exec.detach(),
            phase_anchor=next_phase_anchor,
            phase_anchor_visible_total=visible_anchor_total,
            trace_tail_reuse_count=(
                state.trace_tail_reuse_count.long().detach().clone()
                if state.trace_tail_reuse_count is not None
                else torch.zeros_like(commit_frontier.long())
            ),
            intra_phrase_alpha=intra_phrase_alpha,
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
        self._validate_prefix_reuse_consistency(
            state=state,
            unit_mask=unit_mask,
            reuse_prefix=reuse_prefix,
        )
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
            trace_phrase_blend=getattr(planner, "trace_phrase_blend", None),
            trace_global_blend=getattr(planner, "trace_global_blend", None),
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
        execution_planner.chunk_summary = getattr(planner, "chunk_summary", None)
        execution_planner.chunk_structure_progress = getattr(planner, "chunk_structure_progress", None)
        execution_planner.chunk_commit_prob = getattr(planner, "chunk_commit_prob", None)
        execution_planner.phrase_open_prob = getattr(planner, "phrase_open_prob", None)
        execution_planner.phrase_close_prob = getattr(planner, "phrase_close_prob", None)
        execution_planner.phrase_role_prob = getattr(planner, "phrase_role_prob", None)
        execution_planner.phrase_prototype_summary = getattr(planner, "phrase_prototype_summary", None)
        execution_planner.phrase_prototype_stats = getattr(planner, "phrase_prototype_stats", None)
        execution_planner.prompt_reliability = getattr(planner, "prompt_reliability", None)
        execution_planner.boundary_style_residual_unit = getattr(planner, "boundary_style_residual_unit", None)
        execution_planner.intra_phrase_alpha = getattr(planner, "intra_phrase_alpha", None)
        execution_planner.commit_boundary_logit_unit = getattr(planner, "commit_boundary_logit_unit", None)
        execution_planner.commit_mask_unit = getattr(planner, "commit_mask_unit", None)
        execution_planner.planned_commit_frontier = getattr(planner, "planned_commit_frontier", None)
        execution_planner.commit_confidence = getattr(planner, "commit_confidence", None)
        execution_planner.segment_mask_unit = getattr(planner, "segment_mask_unit", None)
        execution_planner.pause_segment_mask_unit = getattr(planner, "pause_segment_mask_unit", None)
        execution_planner.phrase_speech_budget_win = getattr(planner, "phrase_speech_budget_win", None)
        execution_planner.phrase_pause_budget_win = getattr(planner, "phrase_pause_budget_win", None)
        execution_planner.ref_phrase_index = getattr(planner, "ref_phrase_index", None)
        execution_planner.ref_phrase_trace = getattr(planner, "ref_phrase_trace", None)
        execution_planner.ref_phrase_stats = getattr(planner, "ref_phrase_stats", None)
        execution_planner.active_phrase_start = getattr(planner, "active_phrase_start", None)
        execution_planner.active_phrase_end = getattr(planner, "active_phrase_end", None)
        execution_planner.local_trace_ctx_unit = getattr(planner, "local_trace_ctx_unit", None)
        execution_planner.local_rho_unit = getattr(planner, "local_rho_unit", None)
        sanitized_segment_mask = _sanitize_tail_allocation_mask(
            unit_mask=unit_mask,
            commit_frontier=state.commit_frontier,
            allocation_mask=getattr(planner, "segment_mask_unit", None),
        )
        sanitized_pause_segment_mask = _sanitize_tail_allocation_mask(
            unit_mask=unit_mask,
            commit_frontier=state.commit_frontier,
            allocation_mask=getattr(planner, "pause_segment_mask_unit", None),
        ) * sanitized_segment_mask
        execution_planner.segment_mask_unit = sanitized_segment_mask
        execution_planner.pause_segment_mask_unit = sanitized_pause_segment_mask
        speech_projection_budget = getattr(planner, "phrase_speech_budget_win", None)
        if speech_projection_budget is None:
            speech_projection_budget = speech_budget_win
        else:
            speech_projection_budget = torch.minimum(
                speech_projection_budget.float().to(device=speech_budget_win.device),
                speech_budget_win.float(),
            )
        pause_projection_budget = getattr(planner, "phrase_pause_budget_win", None)
        if pause_projection_budget is None:
            pause_projection_budget = pause_budget_win
        else:
            pause_projection_budget = torch.minimum(
                pause_projection_budget.float().to(device=pause_budget_win.device),
                pause_budget_win.float(),
            )
        execution_planner.phrase_speech_budget_win = speech_projection_budget
        execution_planner.phrase_pause_budget_win = pause_projection_budget
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
            speech_budget_win=speech_projection_budget,
            state=state,
            reuse_prefix=reuse_prefix,
            segment_mask=sanitized_segment_mask,
        )
        pause_after_exec = self._project_pause(
            pause_weight_unit=pause_weight_unit,
            boundary_score_unit=planner_boundary_score,
            unit_mask=unit_mask,
            pause_budget_win=pause_projection_budget,
            state=state,
            reuse_prefix=reuse_prefix,
            soft_pause_selection=soft_pause_selection_active,
            pause_topk_ratio_override=pause_topk_ratio_override,
            pause_support_prob_unit=getattr(planner, "pause_support_prob_unit", None),
            pause_allocation_weight_unit=getattr(planner, "pause_allocation_weight_unit", None),
            allocation_mask=sanitized_pause_segment_mask,
        )
        computed_local_rho = self._compute_local_rho(
            speech_exec=speech_duration_exec,
            segment_mask=sanitized_segment_mask,
            unit_mask=unit_mask,
        )
        if computed_local_rho is not None:
            execution_planner.local_rho_unit = computed_local_rho
        sampled_local_trace_ctx = self._sample_phrase_trace_context(
            ref_phrase_trace=getattr(planner, "ref_phrase_trace", None),
            local_rho_unit=execution_planner.local_rho_unit,
            segment_mask=sanitized_segment_mask,
            unit_mask=unit_mask,
        )
        if sampled_local_trace_ctx is not None:
            execution_planner.local_trace_ctx_unit = sampled_local_trace_ctx
        intra_phrase_alpha = self._collapse_intra_phrase_alpha(
            local_rho_unit=execution_planner.local_rho_unit,
            segment_mask=sanitized_segment_mask,
            unit_mask=unit_mask,
        )
        execution_planner.intra_phrase_alpha = intra_phrase_alpha
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
        planned_commit_frontier = getattr(planner, "planned_commit_frontier", None)
        if planned_commit_frontier is not None:
            commit_frontier = self._resolve_planned_commit_frontier_override(
                planned_commit_frontier=planned_commit_frontier,
                state=state,
                unit_mask=unit_mask,
            )
        else:
            commit_frontier = self._compute_commit_frontier(
                state=state,
                unit_mask=unit_mask,
                open_run_mask=open_run_mask,
                boundary_score_unit=planner_boundary_score,
                force_full_commit=force_full_commit,
            )
        execution_planner.planned_commit_frontier = commit_frontier
        next_state = self._advance_state(
            state=state,
            dur_anchor_src=dur_anchor_src,
            unit_mask=unit_mask,
            effective_duration_exec=effective_duration_exec,
            commit_frontier=commit_frontier,
            speech_duration_exec=speech_duration_exec,
            pause_after_exec=pause_after_exec,
            intra_phrase_alpha=execution_planner.intra_phrase_alpha,
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
