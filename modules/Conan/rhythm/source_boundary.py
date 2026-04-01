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
    """Build a cheap, prefix-safe boundary cue.

    This stays intentionally heuristic and non-learned:
    - separator-aware unit breaks are treated as strong evidence
    - local duration peaks / jumps are treated as weaker phrase-boundary evidence
    - open tail units are mildly suppressed because they are not yet commit-safe
    """

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
