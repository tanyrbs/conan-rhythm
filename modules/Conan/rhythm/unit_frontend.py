from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


@dataclass
class RhythmUnitBatch:
    content_units: torch.Tensor
    dur_anchor_src: torch.Tensor
    unit_mask: torch.Tensor
    open_run_mask: torch.Tensor
    sep_hint: torch.Tensor


def _compress_token_sequence(
    token_sequence: Sequence[int],
    *,
    silent_token: int | None = None,
    separator_aware: bool = False,
) -> tuple[list[int], list[int], list[int]]:
    units: list[int] = []
    durations: list[int] = []
    sep_hint: list[int] = []
    prev_token: int | None = None
    saw_separator = False
    for token in token_sequence:
        token = int(token)
        if token < 0:
            continue
        if silent_token is not None and token == int(silent_token):
            if separator_aware and len(units) > 0:
                saw_separator = True
            continue
        start_new = len(units) == 0 or token != prev_token or (separator_aware and saw_separator)
        if start_new:
            if saw_separator and separator_aware and len(sep_hint) > 0:
                sep_hint[-1] = 1
            units.append(token)
            durations.append(1)
            sep_hint.append(0)
        else:
            durations[-1] += 1
        prev_token = token
        saw_separator = False
    return units, durations, sep_hint


class RhythmUnitFrontend:
    """Prefix-safe token->unit frontend.

    Keep:
      - token dedup into units
      - dur_anchor_src
      - internal open_run_mask / sep_hint sidecars

    Do not keep as public front-door:
      - unit_type
      - boundary_hint
    """

    def __init__(
        self,
        *,
        silent_token: int | None = None,
        separator_aware: bool = True,
        tail_open_units: int = 1,
    ) -> None:
        self.silent_token = silent_token
        self.separator_aware = bool(separator_aware)
        self.tail_open_units = max(1, int(tail_open_units))

    @staticmethod
    def _pad(seqs: list[torch.Tensor], pad_value: int = 0, *, dtype=torch.long) -> torch.Tensor:
        max_len = max((seq.numel() for seq in seqs), default=0)
        out = torch.full((len(seqs), max_len), pad_value, dtype=dtype)
        for idx, seq in enumerate(seqs):
            if seq.numel() > 0:
                out[idx, : seq.numel()] = seq.to(dtype=dtype)
        return out

    def from_token_lists(
        self,
        batch_tokens: Iterable[Sequence[int]],
        *,
        mark_last_open: bool = True,
        device: torch.device | None = None,
    ) -> RhythmUnitBatch:
        unit_list = []
        dur_list = []
        open_list = []
        sep_list = []
        mask_list = []
        for token_sequence in batch_tokens:
            units, durations, sep_hint = _compress_token_sequence(
                token_sequence,
                silent_token=self.silent_token,
                separator_aware=self.separator_aware,
            )
            unit_tensor = torch.tensor(units, dtype=torch.long)
            dur_tensor = torch.tensor(durations, dtype=torch.long)
            open_tensor = torch.zeros_like(unit_tensor)
            if mark_last_open and unit_tensor.numel() > 0:
                open_tensor[max(0, unit_tensor.numel() - self.tail_open_units):] = 1
            sep_tensor = torch.tensor(sep_hint, dtype=torch.long)
            mask_tensor = torch.ones_like(unit_tensor, dtype=torch.float32)
            unit_list.append(unit_tensor)
            dur_list.append(dur_tensor)
            open_list.append(open_tensor)
            sep_list.append(sep_tensor)
            mask_list.append(mask_tensor)

        content_units = self._pad(unit_list, pad_value=0, dtype=torch.long)
        dur_anchor_src = self._pad(dur_list, pad_value=0, dtype=torch.long)
        open_run_mask = self._pad(open_list, pad_value=0, dtype=torch.long)
        sep_hint = self._pad(sep_list, pad_value=0, dtype=torch.long)
        unit_mask = self._pad([m.long() for m in mask_list], pad_value=0, dtype=torch.long).float()

        if device is not None:
            content_units = content_units.to(device)
            dur_anchor_src = dur_anchor_src.to(device)
            open_run_mask = open_run_mask.to(device)
            sep_hint = sep_hint.to(device)
            unit_mask = unit_mask.to(device)

        return RhythmUnitBatch(
            content_units=content_units,
            dur_anchor_src=dur_anchor_src,
            unit_mask=unit_mask,
            open_run_mask=open_run_mask,
            sep_hint=sep_hint,
        )

    def from_content_tensor(
        self,
        content: torch.Tensor,
        *,
        content_lengths: torch.Tensor | None = None,
        mark_last_open: bool = True,
    ) -> RhythmUnitBatch:
        if content.dim() != 2:
            raise ValueError(f"content must be rank-2 [B,T], got {tuple(content.shape)}")
        batch_tokens = []
        batch_size, total_steps = content.shape
        if content_lengths is None:
            content_lengths = torch.full(
                (batch_size,),
                int(total_steps),
                dtype=torch.long,
                device=content.device,
            )
        for batch_idx in range(batch_size):
            valid_len = int(content_lengths[batch_idx].item())
            valid_len = max(0, min(valid_len, int(total_steps)))
            tokens = content[batch_idx, :valid_len].detach().cpu().tolist()
            batch_tokens.append(tokens)
        return self.from_token_lists(
            batch_tokens,
            mark_last_open=mark_last_open,
            device=content.device,
        )

    def from_precomputed(
        self,
        *,
        content_units: torch.Tensor,
        dur_anchor_src: torch.Tensor,
        unit_mask: torch.Tensor | None = None,
        open_run_mask: torch.Tensor | None = None,
        sep_hint: torch.Tensor | None = None,
    ) -> RhythmUnitBatch:
        content_units = content_units.long()
        dur_anchor_src = dur_anchor_src.long()
        if unit_mask is None:
            unit_mask = dur_anchor_src.gt(0).float()
        else:
            unit_mask = unit_mask.float()
        if open_run_mask is None:
            open_run_mask = torch.zeros_like(content_units)
        if sep_hint is None:
            sep_hint = torch.zeros_like(content_units)
        return RhythmUnitBatch(
            content_units=content_units,
            dur_anchor_src=dur_anchor_src,
            unit_mask=unit_mask,
            open_run_mask=open_run_mask.long(),
            sep_hint=sep_hint.long(),
        )
