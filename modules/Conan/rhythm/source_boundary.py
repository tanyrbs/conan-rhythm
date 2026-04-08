from __future__ import annotations

import torch
import torch.nn.functional as F


BOUNDARY_TYPE_JOIN = 0
BOUNDARY_TYPE_WEAK = 1
BOUNDARY_TYPE_PHRASE = 2


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


def classify_boundary_type_unit(
    *,
    unit_mask: torch.Tensor,
    source_boundary_cue: torch.Tensor | None = None,
    boundary_score_unit: torch.Tensor | None = None,
    sep_hint: torch.Tensor | None = None,
    open_run_mask: torch.Tensor | None = None,
    sealed_mask: torch.Tensor | None = None,
    boundary_confidence: torch.Tensor | None = None,
    weak_boundary_threshold: float = 0.40,
    phrase_boundary_threshold: float = 0.55,
) -> torch.Tensor:
    """Classify unit-level junctions into JOIN / WEAK / PHRASE.

    The returned code for unit i describes the boundary *after* unit i.
    The maintained runtime uses these types to:
      - keep phrase retrieval tied to committed PHRASE boundaries only
      - allow weak local timing variation without forcing phrase retrieval
      - keep JOIN slots pause-free in the projector
    """

    unit_mask_f = unit_mask.float()
    source = (
        source_boundary_cue.float().clamp(0.0, 1.0)
        if source_boundary_cue is not None
        else unit_mask_f.new_zeros(unit_mask_f.shape)
    )
    planner = (
        boundary_score_unit.float().clamp(0.0, 1.0)
        if boundary_score_unit is not None
        else unit_mask_f.new_zeros(unit_mask_f.shape)
    )
    sep = (
        sep_hint.float().clamp(0.0, 1.0)
        if sep_hint is not None
        else unit_mask_f.new_zeros(unit_mask_f.shape)
    )
    conf = (
        boundary_confidence.float().clamp(0.0, 1.0)
        if boundary_confidence is not None
        else unit_mask_f.new_zeros(unit_mask_f.shape)
    )
    sealed = (
        sealed_mask.float().clamp(0.0, 1.0)
        if sealed_mask is not None
        else unit_mask_f.new_ones(unit_mask_f.shape)
    )
    open_tail = (
        open_run_mask.float().clamp(0.0, 1.0)
        if open_run_mask is not None
        else unit_mask_f.new_zeros(unit_mask_f.shape)
    )

    weak_threshold = float(max(0.0, min(1.0, weak_boundary_threshold)))
    phrase_threshold = float(max(weak_threshold, min(1.0, phrase_boundary_threshold)))

    weak_strength = (
        0.40 * source
        + 0.30 * planner
        + 0.20 * sep
        + 0.10 * conf
    )
    phrase_strength = (
        0.45 * planner
        + 0.30 * source
        + 0.15 * conf
        + 0.10 * sep
    )
    valid = (unit_mask_f > 0.5) & (sealed > 0.5) & ~(open_tail > 0.5)
    boundary_type = torch.full_like(unit_mask_f, BOUNDARY_TYPE_JOIN, dtype=torch.long)
    weak_mask = valid & (weak_strength >= weak_threshold)
    phrase_mask = valid & (phrase_strength >= phrase_threshold)
    boundary_type = torch.where(weak_mask, torch.full_like(boundary_type, BOUNDARY_TYPE_WEAK), boundary_type)
    boundary_type = torch.where(phrase_mask, torch.full_like(boundary_type, BOUNDARY_TYPE_PHRASE), boundary_type)
    boundary_type = torch.where(unit_mask_f > 0.5, boundary_type, torch.zeros_like(boundary_type))
    return boundary_type


def build_boundary_type_masks(
    boundary_type_unit: torch.Tensor | None,
    *,
    unit_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    unit_mask_f = unit_mask.float()
    if boundary_type_unit is None:
        zeros = unit_mask_f.new_zeros(unit_mask_f.shape)
        return zeros, zeros, zeros
    boundary_type_unit = boundary_type_unit.long().to(device=unit_mask.device)
    join_mask = (boundary_type_unit == BOUNDARY_TYPE_JOIN).float() * unit_mask_f
    weak_mask = (boundary_type_unit == BOUNDARY_TYPE_WEAK).float() * unit_mask_f
    phrase_mask = (boundary_type_unit == BOUNDARY_TYPE_PHRASE).float() * unit_mask_f
    return join_mask, weak_mask, phrase_mask


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
