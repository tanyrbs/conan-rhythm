from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_standardize(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    total = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (x * mask).sum(dim=1, keepdim=True) / total
    var = (((x - mean) ** 2) * mask).sum(dim=1, keepdim=True) / total
    return ((x - mean) / var.clamp_min(1e-6).sqrt()) * mask


def build_source_boundary_cue(
    *,
    dur_anchor_src: torch.Tensor,
    unit_mask: torch.Tensor,
    sep_hint: torch.Tensor | None = None,
    open_run_mask: torch.Tensor | None = None,
    sealed_mask: torch.Tensor | None = None,
    boundary_confidence: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build a cheap, prefix-safe source-side boundary prior."""

    unit_mask = unit_mask.float()
    log_anchor = torch.log1p(dur_anchor_src.float().clamp_min(0.0)) * unit_mask

    prev_anchor = F.pad(log_anchor[:, :-1], (1, 0))
    next_anchor = F.pad(log_anchor[:, 1:], (0, 1))
    local_peak = torch.relu(log_anchor - 0.5 * (prev_anchor + next_anchor))
    local_jump = 0.5 * (
        torch.abs(log_anchor - prev_anchor) + torch.abs(next_anchor - log_anchor)
    )

    peak_score = torch.sigmoid(_masked_standardize(local_peak, unit_mask))
    jump_score = torch.sigmoid(_masked_standardize(local_jump, unit_mask))
    cue = 0.25 * peak_score + 0.15 * jump_score

    if sep_hint is not None:
        cue = cue + 0.40 * sep_hint.float()
    if boundary_confidence is not None:
        cue = cue + 0.20 * boundary_confidence.float()
    if open_run_mask is not None:
        cue = cue * (1.0 - 0.25 * open_run_mask.float())
    if sealed_mask is not None:
        cue = cue * (0.80 + 0.20 * sealed_mask.float())

    return cue.clamp(0.0, 1.0) * unit_mask


def compose_boundary_score_unit(
    *,
    unit_mask: torch.Tensor,
    source_boundary_cue: torch.Tensor | None = None,
    boundary_trace: torch.Tensor | None = None,
    source_weight: float = 0.65,
) -> torch.Tensor:
    """Deterministic boundary evidence sidecar.

    The maintained planner no longer uses a learnable boundary latent.
    Boundary evidence is a deterministic blend of:
      - source-side prefix-safe structural cues
      - reference-side boundary trace
    """

    unit_mask = unit_mask.float()
    if source_boundary_cue is None and boundary_trace is None:
        return unit_mask.new_zeros(unit_mask.shape)
    if source_boundary_cue is None:
        source_boundary_cue = unit_mask.new_zeros(unit_mask.shape)
    else:
        source_boundary_cue = source_boundary_cue.float().clamp(0.0, 1.0)
    if boundary_trace is None:
        boundary_trace = unit_mask.new_zeros(unit_mask.shape)
    else:
        if boundary_trace.dim() == 3 and boundary_trace.size(-1) == 1:
            boundary_trace = boundary_trace.squeeze(-1)
        boundary_trace = boundary_trace.float().clamp(0.0, 1.0)
    source_weight = float(max(0.0, min(1.0, source_weight)))
    trace_weight = 1.0 - source_weight
    blended = source_weight * source_boundary_cue + trace_weight * boundary_trace
    agreement = torch.minimum(source_boundary_cue, boundary_trace)
    score = 0.85 * blended + 0.15 * agreement
    return score.clamp(0.0, 1.0) * unit_mask


def resolve_boundary_score_unit(planner, fallback: torch.Tensor | None = None) -> torch.Tensor | None:
    if planner is None:
        return fallback
    boundary_score = getattr(planner, "boundary_score_unit", None)
    if boundary_score is None:
        boundary_score = getattr(planner, "boundary_latent", None)
    if boundary_score is None:
        return fallback
    return boundary_score


def build_deterministic_boundary_score(
    *,
    source_boundary_cue: torch.Tensor | None,
    boundary_trace: torch.Tensor | None,
    unit_mask: torch.Tensor,
    source_weight: float = 0.35,
    trace_weight: float = 0.35,
) -> torch.Tensor:
    """Backward-compatible alias for the old deterministic boundary helper."""

    total = float(source_weight) + float(trace_weight)
    normalized_source = 0.5 if total <= 0.0 else float(source_weight) / total
    return compose_boundary_score_unit(
        unit_mask=unit_mask,
        source_boundary_cue=source_boundary_cue,
        boundary_trace=boundary_trace,
        source_weight=normalized_source,
    )
