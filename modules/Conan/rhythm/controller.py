from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from modules.Conan.diff.net import CausalConv1d
from .compat import resolve_phase_decoupled_flag
from .contracts import BoundaryCommitDecision, StreamingRhythmState
from .pause_features import build_pause_support_feature_bundle


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int = 1, keepdim: bool = False) -> torch.Tensor:
    mask = mask.float()
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    total = (x * mask).sum(dim=dim, keepdim=keepdim)
    denom = mask.sum(dim=dim, keepdim=keepdim).clamp_min(1.0)
    return total / denom


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    mask = mask.float()
    masked_logits = logits.masked_fill(mask <= 0, float('-inf'))
    probs = torch.softmax(masked_logits, dim=dim)
    probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
    denom = probs.sum(dim=dim, keepdim=True).clamp_min(1e-6)
    return probs / denom
def resolve_budget_views_from_total_and_pause_share(
    *,
    total_budget: torch.Tensor,
    pause_share: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Expose raw budget branches before projector feasibility repair."""

    total_budget = total_budget.float().clamp_min(0.0)
    pause_share = pause_share.float().clamp(0.0, 1.0)
    raw_pause_budget = (total_budget * pause_share).clamp_min(0.0)
    raw_speech_budget = (total_budget - raw_pause_budget).clamp_min(0.0)
    return {
        'raw_speech_budget_win': raw_speech_budget,
        'raw_pause_budget_win': raw_pause_budget,
        'speech_budget_win': raw_speech_budget,
        'pause_budget_win': raw_pause_budget,
    }


@dataclass
class CommitConfig:
    mode: str = "legacy_projector"
    threshold: float = 0.65
    require_sealed_boundary: bool = True
    min_phrase_units: int = 2
    max_lookahead_units: int = 3
    sep_hint_bonus: float = 0.20
    boundary_confidence_weight: float = 0.40
    source_boundary_weight: float = 0.35
    planner_boundary_weight: float = 0.25


@dataclass
class ChunkStateBundle:
    chunk_summary: torch.Tensor
    structure_progress: torch.Tensor
    commit_now_prob: torch.Tensor
    phrase_open_prob: torch.Tensor
    phrase_close_prob: torch.Tensor
    phrase_role_prob: torch.Tensor
    active_tail_mask: torch.Tensor


def _build_active_tail_mask(
    unit_mask: torch.Tensor,
    commit_frontier: torch.Tensor,
) -> torch.Tensor:
    steps = torch.arange(unit_mask.size(1), device=unit_mask.device)[None, :]
    return ((steps >= commit_frontier[:, None]) & (unit_mask > 0.5)).float()


def _masked_last_value(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    value = value.float()
    mask = mask.float()
    valid_counts = mask.sum(dim=1).long()
    safe_index = (valid_counts - 1).clamp_min(0)
    gathered = value.gather(1, safe_index.unsqueeze(1)).squeeze(1)
    return torch.where(valid_counts > 0, gathered, torch.zeros_like(gathered))


def _masked_mean_scalar(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked_mean(value.float().unsqueeze(-1), mask.float(), dim=1).squeeze(-1)


def _resolve_commit_frontier_tensor(
    state: StreamingRhythmState,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    frontier = state.commit_frontier.long().to(device=device)
    if frontier.dim() == 0:
        frontier = frontier.unsqueeze(0)
    if frontier.size(0) != batch_size:
        raise ValueError(
            f"commit_frontier batch mismatch: got {tuple(frontier.shape)} for batch_size={batch_size}"
        )
    return frontier


def _build_commit_eligibility_mask(
    *,
    unit_mask: torch.Tensor,
    commit_frontier: torch.Tensor,
    open_run_mask: torch.Tensor | None,
    sealed_mask: torch.Tensor | None,
    require_sealed_boundary: bool,
    min_phrase_units: int,
    max_lookahead_units: int,
) -> torch.Tensor:
    steps = torch.arange(unit_mask.size(1), device=unit_mask.device)[None, :]
    eligible = steps >= commit_frontier[:, None]
    eligible = eligible & (unit_mask > 0.5)
    if min_phrase_units > 1:
        eligible = eligible & (steps >= (commit_frontier[:, None] + int(min_phrase_units) - 1))
    if max_lookahead_units > 0:
        eligible = eligible & (steps < (commit_frontier[:, None] + int(max_lookahead_units)))
    if require_sealed_boundary and sealed_mask is not None:
        eligible = eligible & (sealed_mask > 0.5)
    if open_run_mask is not None:
        eligible = eligible & ~(open_run_mask > 0.5)
    return eligible


def _compose_boundary_commit_score(
    *,
    boundary_score_unit: torch.Tensor,
    source_boundary_cue: torch.Tensor,
    boundary_confidence: torch.Tensor | None,
    sep_hint: torch.Tensor | None,
    config: CommitConfig,
) -> torch.Tensor:
    score = (
        float(config.source_boundary_weight) * source_boundary_cue.float()
        + float(config.planner_boundary_weight) * boundary_score_unit.float()
    )
    if boundary_confidence is not None:
        score = score + float(config.boundary_confidence_weight) * boundary_confidence.float()
    if sep_hint is not None:
        score = score + float(config.sep_hint_bonus) * sep_hint.float()
    return score


def _resolve_commit_end(
    *,
    commit_prob: torch.Tensor,
    eligible_mask: torch.Tensor,
    threshold: float,
    commit_frontier: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    steps = torch.arange(commit_prob.size(1), device=commit_prob.device)[None, :]
    above_threshold = (commit_prob >= float(threshold)) & eligible_mask
    chosen_inclusive = torch.where(
        above_threshold,
        steps,
        steps.new_full(steps.shape, -1),
    ).max(dim=1).values
    commit_end = torch.where(
        chosen_inclusive >= 0,
        chosen_inclusive + 1,
        commit_frontier,
    )
    committed = commit_end > commit_frontier
    return commit_end.long(), committed


class BoundaryCommitController(nn.Module):
    def __init__(self, config: CommitConfig | None = None) -> None:
        super().__init__()
        self.config = config or CommitConfig()
        if self.config.mode not in {"legacy_projector", "boundary_phrase"}:
            raise ValueError(f"Unsupported commit mode: {self.config.mode}")

    def forward(
        self,
        *,
        boundary_score_unit: torch.Tensor,
        source_boundary_cue: torch.Tensor,
        unit_mask: torch.Tensor,
        state: StreamingRhythmState,
        boundary_confidence: torch.Tensor | None = None,
        sep_hint: torch.Tensor | None = None,
        open_run_mask: torch.Tensor | None = None,
        sealed_mask: torch.Tensor | None = None,
    ) -> BoundaryCommitDecision:
        batch_size = int(unit_mask.size(0))
        commit_frontier = _resolve_commit_frontier_tensor(
            state,
            batch_size=batch_size,
            device=unit_mask.device,
        )
        eligible_mask = _build_commit_eligibility_mask(
            unit_mask=unit_mask,
            commit_frontier=commit_frontier,
            open_run_mask=open_run_mask,
            sealed_mask=sealed_mask,
            require_sealed_boundary=bool(self.config.require_sealed_boundary),
            min_phrase_units=max(1, int(self.config.min_phrase_units)),
            max_lookahead_units=max(0, int(self.config.max_lookahead_units)),
        )
        commit_score = _compose_boundary_commit_score(
            boundary_score_unit=boundary_score_unit,
            source_boundary_cue=source_boundary_cue,
            boundary_confidence=boundary_confidence,
            sep_hint=sep_hint,
            config=self.config,
        )
        masked_commit_score = commit_score.masked_fill(~eligible_mask, -1.0e4)
        commit_prob = torch.sigmoid(masked_commit_score)
        commit_end, committed = _resolve_commit_end(
            commit_prob=commit_prob,
            eligible_mask=eligible_mask,
            threshold=float(self.config.threshold),
            commit_frontier=commit_frontier,
        )
        masked_confidence = torch.where(
            eligible_mask,
            commit_prob,
            torch.zeros_like(commit_prob),
        )
        commit_confidence = masked_confidence.max(dim=1, keepdim=True).values
        return BoundaryCommitDecision(
            commit_end=commit_end,
            committed=committed,
            commit_score_unit=masked_commit_score,
            eligible_mask_unit=eligible_mask.float(),
            commit_confidence=commit_confidence,
        )


class ChunkStateHead(nn.Module):
    """Deterministic chunk-state summarizer for discrete prefix control.

    This deliberately avoids introducing new trainable weights. The goal of the
    current upgrade is to move authority away from a drifting phase scalar and
    toward prefix-safe source structure signals that already exist in the repo.
    """

    def forward(
        self,
        *,
        unit_mask: torch.Tensor,
        state: StreamingRhythmState,
        source_boundary_cue: torch.Tensor | None,
        boundary_score_unit: torch.Tensor | None,
        sep_hint: torch.Tensor | None = None,
        open_run_mask: torch.Tensor | None = None,
        sealed_mask: torch.Tensor | None = None,
        boundary_confidence: torch.Tensor | None = None,
    ) -> ChunkStateBundle:
        batch_size = int(unit_mask.size(0))
        commit_frontier = _resolve_commit_frontier_tensor(
            state,
            batch_size=batch_size,
            device=unit_mask.device,
        )
        active_tail_mask = _build_active_tail_mask(unit_mask, commit_frontier)
        visible_mask = (unit_mask > 0.5).float()
        visible_len = visible_mask.sum(dim=1).clamp_min(1.0)
        tail_len = active_tail_mask.sum(dim=1)
        tail_len_ratio = (tail_len / visible_len).clamp(0.0, 1.0)
        frontier_ratio = (commit_frontier.float() / visible_len).clamp(0.0, 1.0)

        boundary_primary = (
            source_boundary_cue.float()
            if source_boundary_cue is not None
            else (
                boundary_score_unit.float()
                if boundary_score_unit is not None
                else visible_mask.new_zeros(visible_mask.shape)
            )
        )
        if sep_hint is not None:
            boundary_primary = torch.maximum(boundary_primary, sep_hint.float().clamp(0.0, 1.0))
        confidence = (
            boundary_confidence.float().clamp(0.0, 1.0)
            if boundary_confidence is not None
            else boundary_primary.clamp(0.0, 1.0)
        )
        sealed_value = (
            (sealed_mask > 0.5).float()
            if sealed_mask is not None
            else torch.ones_like(boundary_primary)
        )
        open_value = (
            (open_run_mask > 0.5).float()
            if open_run_mask is not None
            else torch.zeros_like(boundary_primary)
        )
        sep_value = (
            sep_hint.float().clamp(0.0, 1.0)
            if sep_hint is not None
            else torch.zeros_like(boundary_primary)
        )

        boundary_mean = _masked_mean_scalar(boundary_primary, active_tail_mask)
        confidence_mean = _masked_mean_scalar(confidence, active_tail_mask)
        sealed_ratio = _masked_mean_scalar(sealed_value, active_tail_mask)
        open_ratio = _masked_mean_scalar(open_value, active_tail_mask)
        sep_ratio = _masked_mean_scalar(sep_value, active_tail_mask)
        last_boundary = _masked_last_value(boundary_primary * active_tail_mask, active_tail_mask)

        commit_now_prob = (
            0.40 * last_boundary
            + 0.20 * boundary_mean
            + 0.15 * confidence_mean
            + 0.15 * sealed_ratio
            + 0.10 * sep_ratio
        ).clamp(0.0, 1.0)
        commit_now_prob = (commit_now_prob * (1.0 - open_ratio).clamp(0.0, 1.0)).clamp(0.0, 1.0)
        phrase_close_prob = commit_now_prob
        phrase_open_prob = ((1.0 - tail_len_ratio) * (1.0 - phrase_close_prob)).clamp(0.0, 1.0)
        phrase_medial_prob = (1.0 - (phrase_open_prob + phrase_close_prob)).clamp_min(0.0)
        phrase_role_prob = torch.stack(
            [phrase_open_prob, phrase_medial_prob, phrase_close_prob],
            dim=-1,
        )
        phrase_role_prob = phrase_role_prob / phrase_role_prob.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        structure_progress = (
            0.55 * frontier_ratio
            + 0.20 * (1.0 - tail_len_ratio)
            + 0.25 * phrase_close_prob
        ).clamp(0.0, 1.0)
        chunk_summary = torch.stack(
            [
                frontier_ratio,
                tail_len_ratio,
                boundary_mean,
                confidence_mean,
                phrase_open_prob,
                phrase_close_prob,
            ],
            dim=-1,
        )
        return ChunkStateBundle(
            chunk_summary=chunk_summary,
            structure_progress=structure_progress,
            commit_now_prob=commit_now_prob,
            phrase_open_prob=phrase_open_prob,
            phrase_close_prob=phrase_close_prob,
            phrase_role_prob=phrase_role_prob,
            active_tail_mask=active_tail_mask,
        )


class ResidualTemporalBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 3,
        dilation: int = 1,
        *,
        causal: bool = True,
    ):
        super().__init__()
        if causal:
            self.conv = CausalConv1d(hidden_size, hidden_size, kernel_size=kernel_size, dilation=dilation)
        else:
            pad = dilation * (kernel_size - 1) // 2
            self.conv = nn.Conv1d(
                hidden_size,
                hidden_size,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=pad,
            )
        self.norm = nn.LayerNorm(hidden_size)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        y = self.norm(y)
        y = self.act(y)
        return x + y


class WindowBudgetController(nn.Module):
    """Budget head with compact conditioning."""

    def __init__(
        self,
        hidden_size: int,
        stats_dim: int,
        trace_dim: int,
        *,
        max_total_logratio: float = 0.8,
        pause_share_min: float = 0.0,
        pause_share_max: float = 0.45,
        pause_share_residual_max: float = 0.12,
        min_speech_frames: float = 1.0,
        boundary_feature_scale: float = 0.35,
        phase_feature_scale: float = 0.0,
        phase_decoupled_timing: bool | None = None,
        phase_free_timing: bool | None = None,
        debt_control_scale: float = 4.0,
        debt_pause_priority: float = 0.15,
        debt_speech_priority: float = 0.25,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.max_total_logratio = float(max_total_logratio)
        self.pause_share_min = float(pause_share_min)
        self.pause_share_max = float(max(pause_share_min, pause_share_max))
        self.pause_share_residual_max = float(max(0.0, pause_share_residual_max))
        self.min_speech_frames = float(min_speech_frames)
        self.boundary_feature_scale = float(boundary_feature_scale)
        self.phase_feature_scale = float(min(max(phase_feature_scale, 0.0), 1.0))
        self.phase_decoupled_timing = resolve_phase_decoupled_flag(
            default=False,
            phase_decoupled_timing=phase_decoupled_timing,
            phase_free_timing=phase_free_timing,
            where="WindowBudgetController.__init__",
        )
        self.phase_free_timing = self.phase_decoupled_timing
        self.debt_control_scale = float(max(debt_control_scale, 1.0e-3))
        self.debt_pause_priority = float(max(debt_pause_priority, 0.0))
        self.debt_speech_priority = float(max(debt_speech_priority, 0.0))

        self.anchor_proj = nn.Linear(1, hidden_size)
        self.boundary_proj = nn.Linear(1, hidden_size)
        self.trace_proj = nn.Linear(trace_dim, hidden_size)
        self.slow_proj = nn.Linear(trace_dim, hidden_size)
        self.stats_proj = nn.Linear(stats_dim, hidden_size)
        self.phase_proj = nn.Linear(1, hidden_size)
        self.backlog_proj = nn.Linear(2, hidden_size)
        self.in_proj = nn.Linear(hidden_size, hidden_size)
        self.blocks = nn.ModuleList([
            ResidualTemporalBlock(hidden_size, dilation=1, causal=causal),
            ResidualTemporalBlock(hidden_size, dilation=2, causal=causal),
            ResidualTemporalBlock(hidden_size, dilation=4, causal=causal),
        ])
        self.pool_mlp = nn.Sequential(
            nn.Linear(hidden_size + trace_dim + trace_dim + stats_dim + 5, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.total_budget_head = nn.Linear(hidden_size, 1)
        self.pause_share_residual_head = nn.Linear(hidden_size, 1)
        # Compatibility-only stub for old checkpoints; the maintained planner no
        # longer multiplies a separate anchor gate into the total log-ratio.
        self.anchor_gate_head = nn.Linear(hidden_size, 1)
        for param in self.anchor_gate_head.parameters():
            param.requires_grad = False

    def forward(
        self,
        *,
        unit_states: torch.Tensor,
        dur_anchor_src: torch.Tensor,
        unit_mask: torch.Tensor,
        planner_ref_stats: torch.Tensor,
        planner_trace_context: torch.Tensor,
        slow_rhythm_summary: torch.Tensor | None,
        boundary_score_unit: torch.Tensor | None,
        phase_ptr: torch.Tensor,
        clock_delta: torch.Tensor,
        commit_frontier: torch.Tensor | None = None,
        chunk_state: ChunkStateBundle | None = None,
        phrase_prototype_summary: torch.Tensor | None = None,
        phrase_prototype_stats: torch.Tensor | None = None,
        prompt_reliability: torch.Tensor | None = None,
        phase_decoupled_timing: bool | None = None,
        phase_free_timing: bool | None = None,
        debt_control_scale: float | None = None,
        debt_pause_priority: float | None = None,
        debt_speech_priority: float | None = None,
    ) -> dict[str, torch.Tensor]:
        unit_mask = unit_mask.float()
        src_log = torch.log1p(dur_anchor_src.float().clamp_min(0.0)).unsqueeze(-1)
        if boundary_score_unit is None:
            boundary_score_unit = unit_mask.new_zeros(unit_mask.shape)
        if slow_rhythm_summary is None:
            slow_rhythm_summary = masked_mean(planner_trace_context, unit_mask, dim=1)
        if prompt_reliability is not None:
            prompt_reliability = prompt_reliability.float().reshape(unit_states.size(0), 1).clamp(0.0, 1.0)
        else:
            prompt_reliability = unit_mask.new_ones((unit_states.size(0), 1))
        if phrase_prototype_summary is not None:
            slow_rhythm_summary = (
                slow_rhythm_summary.float() * (1.0 - prompt_reliability)
                + phrase_prototype_summary.float() * prompt_reliability
            )
        if phrase_prototype_stats is not None:
            planner_ref_stats = (
                planner_ref_stats.float() * (1.0 - prompt_reliability)
                + phrase_prototype_stats.float() * prompt_reliability
            )
        boundary_feat = (boundary_score_unit.float() * self.boundary_feature_scale).unsqueeze(-1)
        visible_len = unit_mask.float().sum(dim=1).clamp_min(1.0)
        if commit_frontier is not None:
            frontier_ratio = (
                commit_frontier.float().to(device=unit_mask.device).reshape(unit_mask.size(0), -1)[:, 0]
                / visible_len
            ).clamp(0.0, 1.0)
        elif chunk_state is not None:
            frontier_ratio = chunk_state.chunk_summary[:, 0].float().clamp(0.0, 1.0)
        else:
            frontier_ratio = torch.zeros((unit_mask.size(0),), device=unit_mask.device, dtype=unit_mask.dtype)
        structure_progress = frontier_ratio
        if chunk_state is not None:
            structure_progress = chunk_state.structure_progress.float()
        effective_phase_decoupled_timing = resolve_phase_decoupled_flag(
            default=self.phase_decoupled_timing,
            phase_decoupled_timing=phase_decoupled_timing,
            phase_free_timing=phase_free_timing,
            where="WindowBudgetController.forward",
        )
        effective_debt_control_scale = float(
            self.debt_control_scale if debt_control_scale is None else max(float(debt_control_scale), 1.0e-3)
        )
        effective_debt_pause_priority = float(
            self.debt_pause_priority if debt_pause_priority is None else max(float(debt_pause_priority), 0.0)
        )
        effective_debt_speech_priority = float(
            self.debt_speech_priority if debt_speech_priority is None else max(float(debt_speech_priority), 0.0)
        )
        if (not effective_phase_decoupled_timing) and self.phase_feature_scale > 0.0:
            if chunk_state is not None or commit_frontier is not None:
                structure_progress = (
                    structure_progress * (1.0 - self.phase_feature_scale)
                    + phase_ptr.float() * self.phase_feature_scale
                )
            else:
                structure_progress = phase_ptr.float()
        if effective_phase_decoupled_timing:
            phase_like = torch.zeros(
                (unit_states.size(0), unit_states.size(1), 1),
                device=unit_states.device,
                dtype=unit_states.dtype,
            )
        else:
            phase_like = structure_progress.view(-1, 1, 1).expand(-1, unit_states.size(1), -1)
        clock_delta = clock_delta.float()
        clock_ctrl = torch.tanh(clock_delta / effective_debt_control_scale)
        clock_pos = clock_ctrl.clamp_min(0.0)
        clock_neg = (-clock_ctrl).clamp_min(0.0)
        clock_pair = torch.stack([clock_pos, clock_neg], dim=-1)
        clock_pair = clock_pair.unsqueeze(1).expand(-1, unit_states.size(1), -1)

        x = (
            unit_states
            + self.anchor_proj(src_log)
            + self.boundary_proj(boundary_feat)
            + self.trace_proj(planner_trace_context)
            + self.slow_proj(slow_rhythm_summary).unsqueeze(1)
            + self.stats_proj(planner_ref_stats).unsqueeze(1)
            + self.phase_proj(phase_like)
            + self.backlog_proj(clock_pair)
        )
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)

        pooled_units = masked_mean(x, unit_mask, dim=1)
        pooled_trace = masked_mean(planner_trace_context, unit_mask, dim=1)
        pooled_anchor = masked_mean(src_log, unit_mask, dim=1).squeeze(-1)
        pooled_boundary = masked_mean(boundary_feat, unit_mask, dim=1).squeeze(-1)
        global_input = torch.cat(
            [
                pooled_units,
                pooled_trace,
                slow_rhythm_summary,
                planner_ref_stats,
                pooled_anchor.unsqueeze(-1),
                pooled_boundary.unsqueeze(-1),
                structure_progress.unsqueeze(-1),
                clock_pos.unsqueeze(-1),
                clock_neg.unsqueeze(-1),
            ],
            dim=-1,
        )
        global_hidden = self.pool_mlp(global_input)

        raw_total_logratio = torch.tanh(self.total_budget_head(global_hidden)) * self.max_total_logratio
        raw_total_logratio = (
            raw_total_logratio
            - clock_pos.unsqueeze(-1) * effective_debt_speech_priority
            + 0.25 * clock_neg.unsqueeze(-1) * effective_debt_speech_priority
        ).clamp(-self.max_total_logratio, self.max_total_logratio)
        pause_ratio_hint = planner_ref_stats[:, 1:2].float().clamp(self.pause_share_min, self.pause_share_max)
        pause_share_delta = (
            torch.tanh(self.pause_share_residual_head(global_hidden)) * self.pause_share_residual_max
        )
        pause_share = (
            pause_ratio_hint
            + pause_share_delta
            + clock_neg.unsqueeze(-1) * effective_debt_pause_priority
            - clock_pos.unsqueeze(-1) * (0.5 * effective_debt_pause_priority)
        ).clamp(self.pause_share_min, self.pause_share_max)

        src_total = (dur_anchor_src.float() * unit_mask).sum(dim=1, keepdim=True).clamp_min(1.0)
        total_budget = src_total * torch.exp(raw_total_logratio)
        return resolve_budget_views_from_total_and_pause_share(
            total_budget=total_budget,
            pause_share=pause_share,
        )


class UnitRedistributionHead(nn.Module):
    """Local redistribution head with no budget-hidden leakage."""

    def __init__(
        self,
        hidden_size: int,
        trace_dim: int,
        *,
        max_unit_logratio: float = 0.6,
        boundary_feature_scale: float = 0.35,
        pause_source_boundary_weight: float = 0.20,
        pause_support_split_enable: bool = False,
        pause_breath_features_enable: bool = False,
        pause_breath_reset_threshold: float = 0.55,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.max_unit_logratio = float(max_unit_logratio)
        self.boundary_feature_scale = float(boundary_feature_scale)
        self.pause_source_boundary_weight = float(pause_source_boundary_weight)
        self.pause_support_split_enable = bool(pause_support_split_enable)
        self.pause_breath_features_enable = bool(pause_breath_features_enable)
        self.pause_breath_reset_threshold = float(max(0.0, min(1.0, pause_breath_reset_threshold)))
        self.anchor_proj = nn.Linear(1, hidden_size)
        self.trace_proj = nn.Linear(trace_dim, hidden_size)
        self.slow_proj = nn.Linear(trace_dim, hidden_size)
        self.boundary_proj = nn.Linear(1, hidden_size)
        self.pause_feature_proj = nn.Linear(2, hidden_size) if self.pause_breath_features_enable else None
        self.in_proj = nn.Linear(hidden_size, hidden_size)
        self.blocks = nn.ModuleList([
            ResidualTemporalBlock(hidden_size, dilation=1, causal=causal),
            ResidualTemporalBlock(hidden_size, dilation=2, causal=causal),
        ])
        self.logratio_head = nn.Linear(hidden_size, 1)
        self.pause_head = nn.Linear(hidden_size, 1)
        self.pause_allocation_head = nn.Linear(hidden_size, 1) if self.pause_support_split_enable else None

    def forward(
        self,
        *,
        unit_states: torch.Tensor,
        dur_anchor_src: torch.Tensor,
        planner_trace_context: torch.Tensor,
        unit_mask: torch.Tensor,
        slow_rhythm_summary: torch.Tensor | None = None,
        boundary_score_unit: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        unit_mask = unit_mask.float()
        if boundary_score_unit is None:
            boundary_score_unit = unit_mask.new_zeros(unit_mask.shape)
        if slow_rhythm_summary is None:
            slow_rhythm_summary = masked_mean(planner_trace_context, unit_mask, dim=1)
        src_log = torch.log1p(dur_anchor_src.float().clamp_min(0.0)).unsqueeze(-1)
        boundary_feat = (boundary_score_unit.float() * self.boundary_feature_scale).unsqueeze(-1)
        pause_feature_bundle = build_pause_support_feature_bundle(
            dur_anchor_src=dur_anchor_src,
            unit_mask=unit_mask,
            boundary_score_unit=boundary_score_unit,
            source_boundary_cue=None,
            reset_threshold=self.pause_breath_reset_threshold,
        )
        x = (
            unit_states
            + self.anchor_proj(src_log)
            + self.trace_proj(planner_trace_context)
            + self.slow_proj(slow_rhythm_summary).unsqueeze(1)
            + self.boundary_proj(boundary_feat)
        )
        if self.pause_feature_proj is not None:
            x = x + self.pause_feature_proj(pause_feature_bundle.feature_tensor)
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)

        raw_logratio = torch.tanh(self.logratio_head(x).squeeze(-1)) * self.max_unit_logratio
        mean_logratio = masked_mean(raw_logratio.unsqueeze(-1), unit_mask, dim=1, keepdim=True).squeeze(-1)
        dur_logratio = (raw_logratio - mean_logratio) * unit_mask

        pause_logits = self.pause_head(x).squeeze(-1)
        pause_logits = pause_logits + self.pause_source_boundary_weight * boundary_score_unit.float()
        ret = {
            'dur_logratio_unit': dur_logratio,
            'pause_run_length_unit': pause_feature_bundle.run_length_unit,
            'pause_breath_debt_unit': pause_feature_bundle.breath_debt_unit,
        }
        if self.pause_allocation_head is None:
            ret['pause_weight_unit'] = masked_softmax(pause_logits, unit_mask, dim=1) * unit_mask
            return ret
        pause_support_prob = torch.sigmoid(pause_logits) * unit_mask
        allocation_logits = self.pause_allocation_head(x).squeeze(-1)
        pause_allocation_weight = masked_softmax(allocation_logits, unit_mask, dim=1) * unit_mask
        combined = pause_support_prob * pause_allocation_weight * unit_mask
        combined_total = combined.sum(dim=1, keepdim=True)
        pause_weight = torch.where(
            combined_total > 1e-6,
            combined / combined_total.clamp_min(1e-6),
            pause_allocation_weight,
        ) * unit_mask
        ret.update(
            {
                'pause_weight_unit': pause_weight,
                'pause_support_logit_unit': pause_logits * unit_mask,
                'pause_support_prob_unit': pause_support_prob,
                'pause_allocation_weight_unit': pause_allocation_weight,
            }
        )
        return ret
