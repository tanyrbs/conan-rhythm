from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .contracts import BoundaryCommitDecision, StreamingRhythmState


@dataclass
class CommitConfig:
    mode: str = "boundary_phrase"
    threshold: float = 0.55
    require_sealed_boundary: bool = True
    min_phrase_units: int = 2
    # 0 means unbounded lookahead; this keeps offline/stage-1 training from
    # collapsing to tiny prefixes while still allowing streaming configs to cap
    # commit distance explicitly.
    max_lookahead_units: int = 0
    sep_hint_bonus: float = 0.15
    boundary_confidence_weight: float = 0.20
    source_boundary_weight: float = 0.45
    planner_boundary_weight: float = 0.35


def _broadcast_like(value: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor:
    if value is None:
        return ref.new_zeros(ref.shape)
    tensor = value.float()
    if tensor.shape == ref.shape:
        return tensor
    if tensor.dim() == 2 and tensor.size(1) == 1:
        return tensor.expand_as(ref)
    if tensor.dim() == 1 and tensor.size(0) == ref.size(0):
        return tensor[:, None].expand_as(ref)
    return tensor.reshape(ref.size(0), -1)[:, :1].expand_as(ref)


def build_segment_mask(
    *,
    unit_mask: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
) -> torch.Tensor:
    unit_mask = unit_mask.float()
    steps = torch.arange(unit_mask.size(1), device=unit_mask.device)[None, :]
    start = start.long().clamp(min=0, max=unit_mask.size(1))
    end = end.long().clamp(min=0, max=unit_mask.size(1))
    return ((steps >= start[:, None]) & (steps < end[:, None]) & (unit_mask > 0.5)).float()


class BoundaryCommitController(nn.Module):
    def __init__(self, config: CommitConfig | None = None) -> None:
        super().__init__()
        self.config = config or CommitConfig()

    def forward(
        self,
        *,
        boundary_score_unit: torch.Tensor,
        source_boundary_cue: torch.Tensor | None,
        boundary_confidence: torch.Tensor | None,
        sep_hint: torch.Tensor | None,
        unit_mask: torch.Tensor,
        open_run_mask: torch.Tensor | None,
        sealed_mask: torch.Tensor | None,
        state: StreamingRhythmState,
    ) -> BoundaryCommitDecision:
        unit_mask = unit_mask.float()
        B, T = unit_mask.shape
        idx = torch.arange(T, device=unit_mask.device)[None, :]
        frontier = state.commit_frontier.long().clamp(min=0, max=T)

        eligible = (idx >= frontier[:, None]) & (unit_mask > 0.5)
        min_phrase_units = max(1, int(self.config.min_phrase_units))
        eligible = eligible & (idx >= (frontier[:, None] + min_phrase_units - 1))

        if bool(self.config.require_sealed_boundary) and sealed_mask is not None:
            eligible = eligible & (sealed_mask.float() > 0.5)
        if open_run_mask is not None:
            eligible = eligible & ~(open_run_mask.float() > 0.5)
        max_lookahead_units = int(self.config.max_lookahead_units)
        if max_lookahead_units > 0:
            eligible = eligible & (idx < (frontier[:, None] + max_lookahead_units))

        planner_score = boundary_score_unit.float().clamp(0.0, 1.0) * unit_mask
        source_score = _broadcast_like(source_boundary_cue, planner_score).clamp(0.0, 1.0) * unit_mask
        confidence_score = _broadcast_like(boundary_confidence, planner_score).clamp(0.0, 1.0) * unit_mask
        sep_score = _broadcast_like(sep_hint, planner_score).clamp(0.0, 1.0) * unit_mask

        base_weight = (
            float(max(0.0, self.config.source_boundary_weight))
            + float(max(0.0, self.config.planner_boundary_weight))
            + float(max(0.0, self.config.boundary_confidence_weight))
        )
        if base_weight > 0.0:
            score = (
                float(max(0.0, self.config.source_boundary_weight)) * source_score
                + float(max(0.0, self.config.planner_boundary_weight)) * planner_score
                + float(max(0.0, self.config.boundary_confidence_weight)) * confidence_score
            ) / base_weight
        else:
            score = planner_score
        if float(self.config.sep_hint_bonus) > 0.0:
            score = (score + float(self.config.sep_hint_bonus) * sep_score).clamp(0.0, 1.0)
        score = score * unit_mask

        masked_score = score.masked_fill(~eligible, float("-inf"))
        enough = masked_score >= float(self.config.threshold)
        chosen = torch.where(enough, idx, idx.new_full(idx.shape, -1)).max(dim=1).values
        committed = chosen >= frontier
        commit_end = torch.where(committed, chosen, frontier - 1).clamp(min=-1)

        max_score = masked_score.max(dim=1, keepdim=True).values
        commit_confidence = torch.where(
            torch.isfinite(max_score),
            max_score.clamp(0.0, 1.0),
            max_score.new_zeros(max_score.shape),
        )
        return BoundaryCommitDecision(
            commit_end=commit_end.long(),
            committed=committed,
            commit_score_unit=torch.where(
                torch.isfinite(masked_score),
                masked_score,
                masked_score.new_zeros(masked_score.shape),
            ),
            eligible_mask_unit=eligible.float(),
            commit_confidence=commit_confidence,
        )
