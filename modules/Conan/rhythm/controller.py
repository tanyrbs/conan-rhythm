from __future__ import annotations

import torch
import torch.nn as nn

from modules.Conan.diff.net import CausalConv1d


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
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.max_total_logratio = float(max_total_logratio)
        self.pause_share_min = float(pause_share_min)
        self.pause_share_max = float(max(pause_share_min, pause_share_max))
        self.pause_share_residual_max = float(max(0.0, pause_share_residual_max))
        self.min_speech_frames = float(min_speech_frames)
        self.boundary_feature_scale = float(boundary_feature_scale)

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
    ) -> dict[str, torch.Tensor]:
        unit_mask = unit_mask.float()
        src_log = torch.log1p(dur_anchor_src.float().clamp_min(0.0)).unsqueeze(-1)
        if boundary_score_unit is None:
            boundary_score_unit = unit_mask.new_zeros(unit_mask.shape)
        if slow_rhythm_summary is None:
            slow_rhythm_summary = masked_mean(planner_trace_context, unit_mask, dim=1)
        boundary_feat = (boundary_score_unit.float() * self.boundary_feature_scale).unsqueeze(-1)
        phase = phase_ptr.view(-1, 1, 1).expand(-1, unit_states.size(1), -1)
        clock_delta = clock_delta.float()
        clock_pos = clock_delta.clamp_min(0.0)
        clock_neg = (-clock_delta).clamp_min(0.0)
        clock_pair = torch.stack([clock_pos, clock_neg], dim=-1)
        clock_pair = clock_pair.unsqueeze(1).expand(-1, unit_states.size(1), -1)

        x = (
            unit_states
            + self.anchor_proj(src_log)
            + self.boundary_proj(boundary_feat)
            + self.trace_proj(planner_trace_context)
            + self.slow_proj(slow_rhythm_summary).unsqueeze(1)
            + self.stats_proj(planner_ref_stats).unsqueeze(1)
            + self.phase_proj(phase)
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
                phase_ptr.unsqueeze(-1).float(),
                clock_pos.unsqueeze(-1),
                clock_neg.unsqueeze(-1),
            ],
            dim=-1,
        )
        global_hidden = self.pool_mlp(global_input)

        raw_total_logratio = torch.tanh(self.total_budget_head(global_hidden)) * self.max_total_logratio
        pause_ratio_hint = planner_ref_stats[:, 1:2].float().clamp(self.pause_share_min, self.pause_share_max)
        pause_share_delta = (
            torch.tanh(self.pause_share_residual_head(global_hidden)) * self.pause_share_residual_max
        )
        pause_share = (pause_ratio_hint + pause_share_delta).clamp(self.pause_share_min, self.pause_share_max)

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
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.max_unit_logratio = float(max_unit_logratio)
        self.boundary_feature_scale = float(boundary_feature_scale)
        self.pause_source_boundary_weight = float(pause_source_boundary_weight)
        self.anchor_proj = nn.Linear(1, hidden_size)
        self.trace_proj = nn.Linear(trace_dim, hidden_size)
        self.slow_proj = nn.Linear(trace_dim, hidden_size)
        self.boundary_proj = nn.Linear(1, hidden_size)
        self.in_proj = nn.Linear(hidden_size, hidden_size)
        self.blocks = nn.ModuleList([
            ResidualTemporalBlock(hidden_size, dilation=1, causal=causal),
            ResidualTemporalBlock(hidden_size, dilation=2, causal=causal),
        ])
        self.logratio_head = nn.Linear(hidden_size, 1)
        self.pause_head = nn.Linear(hidden_size, 1)

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
        x = (
            unit_states
            + self.anchor_proj(src_log)
            + self.trace_proj(planner_trace_context)
            + self.slow_proj(slow_rhythm_summary).unsqueeze(1)
            + self.boundary_proj(boundary_feat)
        )
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)

        raw_logratio = torch.tanh(self.logratio_head(x).squeeze(-1)) * self.max_unit_logratio
        mean_logratio = masked_mean(raw_logratio.unsqueeze(-1), unit_mask, dim=1, keepdim=True).squeeze(-1)
        dur_logratio = (raw_logratio - mean_logratio) * unit_mask

        pause_logits = self.pause_head(x).squeeze(-1)
        pause_logits = pause_logits + self.pause_source_boundary_weight * boundary_score_unit.float()
        pause_weight = masked_softmax(pause_logits, unit_mask, dim=1) * unit_mask

        return {
            'dur_logratio_unit': dur_logratio,
            'pause_weight_unit': pause_weight,
        }
