from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PauseSupportFeatureBundle:
    run_length_unit: torch.Tensor
    breath_debt_unit: torch.Tensor
    reset_mask: torch.Tensor

    @property
    def feature_tensor(self) -> torch.Tensor:
        return torch.stack([self.run_length_unit, self.breath_debt_unit], dim=-1)


def build_pause_support_feature_bundle(
    *,
    dur_anchor_src: torch.Tensor,
    unit_mask: torch.Tensor,
    boundary_score_unit: torch.Tensor | None = None,
    source_boundary_cue: torch.Tensor | None = None,
    reset_threshold: float = 0.55,
) -> PauseSupportFeatureBundle:
    """Cheap planner-side pause support features.

    The intent is to expose a coarse "breath debt / run-length" prior:
      - run_length_unit: how many visible source units have elapsed since the
        last strong reset boundary
      - breath_debt_unit: how much anchor mass has accumulated since that reset

    Boundary resets are applied *after* the current unit so the current unit's
    feature still reflects the run that would be ended by a pause placed there.
    """

    unit_mask = unit_mask.float()
    anchor = dur_anchor_src.float().clamp_min(0.0) * unit_mask
    if boundary_score_unit is None and source_boundary_cue is None:
        reset_signal = unit_mask.new_zeros(unit_mask.shape)
    else:
        reset_parts = []
        if boundary_score_unit is not None:
            reset_parts.append(boundary_score_unit.float().clamp(0.0, 1.0))
        if source_boundary_cue is not None:
            reset_parts.append(source_boundary_cue.float().clamp(0.0, 1.0))
        reset_signal = torch.stack(reset_parts, dim=0).amax(dim=0)
    reset_signal = reset_signal * unit_mask
    reset_threshold = float(max(0.0, min(1.0, reset_threshold)))
    reset_mask = (reset_signal >= reset_threshold).float() * unit_mask

    run_length = anchor.new_zeros(anchor.shape)
    breath_debt = anchor.new_zeros(anchor.shape)
    visible = unit_mask.sum(dim=1).clamp_min(1.0)
    total_anchor = anchor.sum(dim=1).clamp_min(1.0)

    for batch_idx in range(anchor.size(0)):
        unit_count = 0.0
        anchor_count = 0.0
        visible_steps = int(unit_mask[batch_idx].sum().item())
        for step_idx in range(visible_steps):
            unit_count += 1.0
            anchor_count += float(anchor[batch_idx, step_idx].item())
            run_length[batch_idx, step_idx] = unit_count / float(max(visible_steps, 1))
            breath_debt[batch_idx, step_idx] = anchor_count
            if bool(reset_mask[batch_idx, step_idx].item()):
                unit_count = 0.0
                anchor_count = 0.0

    breath_debt = torch.log1p(breath_debt) / torch.log1p(total_anchor).unsqueeze(1).clamp_min(1.0e-6)
    breath_debt = breath_debt * unit_mask
    run_length = run_length * unit_mask
    return PauseSupportFeatureBundle(
        run_length_unit=run_length,
        breath_debt_unit=breath_debt,
        reset_mask=reset_mask,
    )
