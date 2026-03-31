from __future__ import annotations

from typing import Callable

import torch


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


def render_rhythm_sequence(
    *,
    content_units: torch.Tensor,
    speech_duration_exec: torch.Tensor,
    pause_after_exec: torch.Tensor,
    unit_mask: torch.Tensor,
    silent_token: int,
    speech_state_fn: Callable[[torch.Tensor], torch.Tensor],
    pause_state: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Render unit-level rhythm decisions into frame-level decoder inputs.

    The renderer is deliberately kept simple:
    - speech frames repeat the unit speech state
    - pause frames repeat a learned pause state
    - pause token id is the configured silent token
    """

    speech_duration_exec = torch.round(speech_duration_exec.float()).long().clamp_min(0)
    pause_after_exec = torch.round(pause_after_exec.float()).long().clamp_min(0)
    unit_mask = unit_mask.float()
    device = content_units.device
    batch_size = int(content_units.size(0))
    hidden_size = int(pause_state.numel())

    unit_states = speech_state_fn(content_units.long())
    frame_state_list: list[torch.Tensor] = []
    frame_token_list: list[torch.Tensor] = []

    for batch_idx in range(batch_size):
        states = []
        tokens = []
        num_units = int(unit_mask[batch_idx].sum().item())
        for unit_idx in range(num_units):
            unit_id = int(content_units[batch_idx, unit_idx].item())
            speech_len = int(speech_duration_exec[batch_idx, unit_idx].item())
            pause_len = int(pause_after_exec[batch_idx, unit_idx].item())
            if speech_len > 0:
                speech_state = unit_states[batch_idx, unit_idx].unsqueeze(0).expand(speech_len, -1)
                states.append(speech_state)
                tokens.append(torch.full((speech_len,), unit_id, dtype=torch.long, device=device))
            if pause_len > 0:
                pause_state_seq = pause_state.view(1, hidden_size).expand(pause_len, -1).to(device=device)
                states.append(pause_state_seq)
                tokens.append(torch.full((pause_len,), int(silent_token), dtype=torch.long, device=device))
        if len(states) <= 0:
            states = [pause_state.view(1, hidden_size).to(device=device)]
            tokens = [torch.full((1,), int(silent_token), dtype=torch.long, device=device)]
        frame_state_list.append(torch.cat(states, dim=0))
        frame_token_list.append(torch.cat(tokens, dim=0))

    frame_states = _pad_sequences(frame_state_list, pad_value=0.0)
    frame_tokens = _pad_sequences(frame_token_list, pad_value=int(silent_token), dtype=torch.long)
    frame_mask = frame_tokens.ne(int(silent_token)).float()
    total_mask = torch.zeros_like(frame_mask)
    for batch_idx, seq in enumerate(frame_token_list):
        total_mask[batch_idx, : seq.size(0)] = 1.0
    return {
        "frame_states": frame_states,
        "frame_tokens": frame_tokens,
        "frame_mask": frame_mask,
        "total_mask": total_mask,
    }
