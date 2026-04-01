from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class BlankSlotSchedule:
    slot_duration_exec: torch.Tensor
    slot_mask: torch.Tensor
    slot_is_blank: torch.Tensor
    slot_unit_index: torch.Tensor


@dataclass
class RenderedRhythmSequence:
    frame_states: torch.Tensor
    frame_tokens: torch.Tensor
    speech_mask: torch.Tensor
    blank_mask: torch.Tensor
    total_mask: torch.Tensor
    frame_slot_index: torch.Tensor
    frame_unit_index: torch.Tensor


def _pad_sequences(
    sequences: list[torch.Tensor],
    *,
    pad_value: float = 0.0,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if len(sequences) <= 0:
        return torch.zeros((0, 0), dtype=dtype or torch.float32)
    max_len = max(int(seq.size(0)) for seq in sequences)
    tail_shape = tuple(sequences[0].shape[1:])
    out = sequences[0].new_full((len(sequences), max_len, *tail_shape), pad_value)
    if dtype is not None:
        out = out.to(dtype=dtype)
    for idx, seq in enumerate(sequences):
        if seq.numel() > 0:
            out[idx, : seq.size(0)] = seq.to(dtype=out.dtype)
    return out


def build_interleaved_blank_slot_schedule(
    *,
    speech_duration_exec: torch.Tensor,
    blank_duration_exec: torch.Tensor,
    unit_mask: torch.Tensor,
) -> BlankSlotSchedule:
    speech_duration_exec = speech_duration_exec.float()
    blank_duration_exec = blank_duration_exec.float()
    unit_mask = unit_mask.float()
    batch_size, num_units = speech_duration_exec.shape
    slot_count = num_units * 2
    slot_duration = speech_duration_exec.new_zeros((batch_size, slot_count))
    slot_mask = speech_duration_exec.new_zeros((batch_size, slot_count))
    slot_is_blank = torch.zeros((batch_size, slot_count), dtype=torch.long, device=speech_duration_exec.device)
    slot_unit_index = torch.zeros((batch_size, slot_count), dtype=torch.long, device=speech_duration_exec.device)
    slot_duration[:, 0::2] = speech_duration_exec
    slot_duration[:, 1::2] = blank_duration_exec
    slot_mask[:, 0::2] = unit_mask
    slot_mask[:, 1::2] = unit_mask
    slot_is_blank[:, 1::2] = 1
    slot_unit_index[:, 0::2] = torch.arange(num_units, device=speech_duration_exec.device)[None, :]
    slot_unit_index[:, 1::2] = torch.arange(num_units, device=speech_duration_exec.device)[None, :]
    return BlankSlotSchedule(
        slot_duration_exec=slot_duration,
        slot_mask=slot_mask,
        slot_is_blank=slot_is_blank,
        slot_unit_index=slot_unit_index,
    )


def render_rhythm_sequence(
    *,
    content_units: torch.Tensor,
    slot_duration_exec: torch.Tensor,
    slot_mask: torch.Tensor,
    slot_is_blank: torch.Tensor,
    slot_unit_index: torch.Tensor,
    silent_token: int,
    speech_state_fn: Callable[[torch.Tensor], torch.Tensor],
    pause_state: torch.Tensor,
) -> RenderedRhythmSequence:
    """Render unit-level rhythm decisions into frame-level decoder inputs.

    The renderer is deliberately kept simple:
    - speech frames repeat the unit speech state
    - pause frames repeat a learned pause state
    - pause token id is the configured silent token
    """

    slot_duration_exec = torch.round(slot_duration_exec.float()).long().clamp_min(0)
    slot_mask = slot_mask.float()
    slot_is_blank = slot_is_blank.long()
    slot_unit_index = slot_unit_index.long()
    device = content_units.device
    batch_size = int(content_units.size(0))
    hidden_size = int(pause_state.numel())

    unit_states = speech_state_fn(content_units.long())
    frame_state_list: list[torch.Tensor] = []
    frame_token_list: list[torch.Tensor] = []
    frame_blank_list: list[torch.Tensor] = []
    frame_slot_index_list: list[torch.Tensor] = []
    frame_unit_index_list: list[torch.Tensor] = []

    for batch_idx in range(batch_size):
        states = []
        tokens = []
        blank_mask = []
        slot_indices = []
        unit_indices = []
        num_slots = int(slot_mask[batch_idx].sum().item())
        for slot_idx in range(num_slots):
            duration = int(slot_duration_exec[batch_idx, slot_idx].item())
            if duration <= 0:
                continue
            unit_idx = int(slot_unit_index[batch_idx, slot_idx].item())
            is_blank = int(slot_is_blank[batch_idx, slot_idx].item()) > 0
            unit_id = int(content_units[batch_idx, unit_idx].item())
            if is_blank:
                pause_state_seq = pause_state.view(1, hidden_size).expand(duration, -1).to(device=device)
                states.append(pause_state_seq)
                tokens.append(torch.full((duration,), int(silent_token), dtype=torch.long, device=device))
                blank_mask.append(torch.ones((duration,), dtype=torch.float32, device=device))
            else:
                speech_state = unit_states[batch_idx, unit_idx].unsqueeze(0).expand(duration, -1)
                states.append(speech_state)
                tokens.append(torch.full((duration,), unit_id, dtype=torch.long, device=device))
                blank_mask.append(torch.zeros((duration,), dtype=torch.float32, device=device))
            slot_indices.append(torch.full((duration,), int(slot_idx), dtype=torch.long, device=device))
            unit_indices.append(torch.full((duration,), int(unit_idx), dtype=torch.long, device=device))
        if len(states) <= 0:
            states = [pause_state.view(1, hidden_size).to(device=device)]
            tokens = [torch.full((1,), int(silent_token), dtype=torch.long, device=device)]
            blank_mask = [torch.ones((1,), dtype=torch.float32, device=device)]
            slot_indices = [torch.zeros((1,), dtype=torch.long, device=device)]
            unit_indices = [torch.zeros((1,), dtype=torch.long, device=device)]
        frame_state_list.append(torch.cat(states, dim=0))
        frame_token_list.append(torch.cat(tokens, dim=0))
        frame_blank_list.append(torch.cat(blank_mask, dim=0))
        frame_slot_index_list.append(torch.cat(slot_indices, dim=0))
        frame_unit_index_list.append(torch.cat(unit_indices, dim=0))

    frame_states = _pad_sequences(frame_state_list, pad_value=0.0)
    frame_tokens = _pad_sequences(frame_token_list, pad_value=int(silent_token), dtype=torch.long)
    blank_mask = _pad_sequences(frame_blank_list, pad_value=0.0)
    frame_slot_index = _pad_sequences(frame_slot_index_list, pad_value=-1, dtype=torch.long)
    frame_unit_index = _pad_sequences(frame_unit_index_list, pad_value=-1, dtype=torch.long)
    speech_mask = (1.0 - blank_mask).clamp(0.0, 1.0)
    total_mask = torch.zeros_like(speech_mask)
    for batch_idx, seq in enumerate(frame_token_list):
        total_mask[batch_idx, : seq.size(0)] = 1.0
    blank_mask = blank_mask * total_mask
    speech_mask = speech_mask * total_mask
    return RenderedRhythmSequence(
        frame_states=frame_states,
        frame_tokens=frame_tokens,
        speech_mask=speech_mask,
        blank_mask=blank_mask,
        total_mask=total_mask,
        frame_slot_index=frame_slot_index,
        frame_unit_index=frame_unit_index,
    )
