from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .frame_plan import (
    BlankSlotSchedule,
    RhythmFramePlan,
    build_frame_plan,
    build_interleaved_blank_slot_schedule,
)


@dataclass
class RenderedRhythmSequence:
    frame_states: torch.Tensor
    frame_tokens: torch.Tensor
    speech_mask: torch.Tensor
    blank_mask: torch.Tensor
    total_mask: torch.Tensor
    frame_slot_index: torch.Tensor
    frame_unit_index: torch.Tensor
    frame_phase_features: torch.Tensor
    frame_plan: RhythmFramePlan


def render_rhythm_sequence(
    *,
    content_units: torch.Tensor,
    silent_token: int,
    speech_state_fn: Callable[[torch.Tensor], torch.Tensor],
    pause_state: torch.Tensor,
    frame_plan: RhythmFramePlan | None = None,
    dur_anchor_src: torch.Tensor | None = None,
    slot_duration_exec: torch.Tensor | None = None,
    slot_mask: torch.Tensor | None = None,
    slot_is_blank: torch.Tensor | None = None,
    slot_unit_index: torch.Tensor | None = None,
    frame_state_post_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> RenderedRhythmSequence:
    if frame_plan is None:
        if (
            dur_anchor_src is None
            or slot_duration_exec is None
            or slot_mask is None
            or slot_is_blank is None
            or slot_unit_index is None
        ):
            raise ValueError("render_rhythm_sequence requires either frame_plan or the full slot schedule + dur_anchor_src.")
        frame_plan = build_frame_plan(
            dur_anchor_src=dur_anchor_src,
            slot_duration_exec=slot_duration_exec,
            slot_mask=slot_mask,
            slot_is_blank=slot_is_blank,
            slot_unit_index=slot_unit_index,
        )

    unit_states = speech_state_fn(content_units.long())
    hidden_size = int(unit_states.size(-1))
    safe_unit_index = frame_plan.frame_unit_index.clamp_min(0)

    frame_states = unit_states.gather(
        1,
        safe_unit_index.unsqueeze(-1).expand(-1, -1, hidden_size),
    )
    frame_tokens = content_units.long().gather(1, safe_unit_index)

    blank_mask_bool = frame_plan.blank_mask > 0.5
    pause_state_seq = pause_state.view(1, 1, hidden_size).to(device=unit_states.device, dtype=unit_states.dtype)
    pause_state_seq = pause_state_seq.expand(frame_states.size(0), frame_states.size(1), -1)
    frame_states = torch.where(blank_mask_bool.unsqueeze(-1), pause_state_seq, frame_states)
    frame_tokens = frame_tokens.masked_fill(blank_mask_bool, int(silent_token))
    frame_tokens = frame_tokens.masked_fill(frame_plan.total_mask <= 0, int(silent_token))
    frame_states = frame_states * frame_plan.total_mask.unsqueeze(-1)

    if frame_state_post_fn is not None:
        frame_states = frame_state_post_fn(
            frame_states,
            frame_plan.frame_phase_features,
            frame_plan.blank_mask,
            frame_plan.total_mask,
        )
        frame_states = frame_states * frame_plan.total_mask.unsqueeze(-1)

    return RenderedRhythmSequence(
        frame_states=frame_states,
        frame_tokens=frame_tokens,
        speech_mask=frame_plan.speech_mask,
        blank_mask=frame_plan.blank_mask,
        total_mask=frame_plan.total_mask,
        frame_slot_index=frame_plan.frame_slot_index,
        frame_unit_index=frame_plan.frame_unit_index,
        frame_phase_features=frame_plan.frame_phase_features,
        frame_plan=frame_plan,
    )
