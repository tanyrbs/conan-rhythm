from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from .unitizer import (
    StreamingRunLengthUnitizer,
    StreamingUnitizerRowState,
    StreamingUnitizerState,
    build_compressed_sequence,
)


@dataclass
class RhythmUnitBatch:
    content_units: torch.Tensor
    dur_anchor_src: torch.Tensor
    unit_mask: torch.Tensor
    open_run_mask: torch.Tensor
    sealed_mask: torch.Tensor
    sep_hint: torch.Tensor
    boundary_confidence: torch.Tensor


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
        self.unitizer = StreamingRunLengthUnitizer(
            silent_token=silent_token,
            separator_aware=separator_aware,
            tail_open_units=tail_open_units,
        )

    @staticmethod
    def _pad(seqs: list[torch.Tensor], pad_value: int = 0, *, dtype=torch.long) -> torch.Tensor:
        max_len = max((seq.numel() for seq in seqs), default=0)
        out = torch.full((len(seqs), max_len), pad_value, dtype=dtype)
        for idx, seq in enumerate(seqs):
            if seq.numel() > 0:
                out[idx, : seq.numel()] = seq.to(dtype=dtype)
        return out

    @staticmethod
    def _move_row_state(row: StreamingUnitizerRowState, device: torch.device) -> StreamingUnitizerRowState:
        return StreamingUnitizerRowState(
            units=row.units.to(device=device, dtype=torch.long),
            durations=row.durations.to(device=device, dtype=torch.long),
            sep_hint=row.sep_hint.to(device=device, dtype=torch.long),
            last_token=int(row.last_token),
            pending_separator=bool(row.pending_separator),
        )

    def _batch_from_compressed(self, compressed_list: list, *, device: torch.device | None = None) -> RhythmUnitBatch:
        unit_list = [torch.tensor(item.units, dtype=torch.long) for item in compressed_list]
        dur_list = [torch.tensor(item.durations, dtype=torch.long) for item in compressed_list]
        open_list = [torch.tensor(item.open_run_mask, dtype=torch.long) for item in compressed_list]
        sealed_list = [torch.tensor(item.sealed_mask, dtype=torch.long) for item in compressed_list]
        sep_list = [torch.tensor(item.sep_hint, dtype=torch.long) for item in compressed_list]
        boundary_list = [torch.tensor(item.boundary_confidence, dtype=torch.float32) for item in compressed_list]
        return self._batch_from_tensors(
            unit_list=unit_list,
            dur_list=dur_list,
            open_list=open_list,
            sealed_list=sealed_list,
            sep_list=sep_list,
            boundary_list=boundary_list,
            device=device,
        )

    def _batch_from_row_states(
        self,
        row_states: list[StreamingUnitizerRowState],
        *,
        mark_last_open: bool = True,
        device: torch.device | None = None,
    ) -> RhythmUnitBatch:
        unit_list: list[torch.Tensor] = []
        dur_list: list[torch.Tensor] = []
        open_list: list[torch.Tensor] = []
        sealed_list: list[torch.Tensor] = []
        sep_list: list[torch.Tensor] = []
        boundary_list: list[torch.Tensor] = []
        for row in row_states:
            units, durations, open_run_mask, sealed_mask, sep_hint, boundary_confidence = self.unitizer._export_row_tensors(
                row,
                mark_last_open=mark_last_open,
            )
            unit_list.append(units)
            dur_list.append(durations)
            open_list.append(open_run_mask)
            sealed_list.append(sealed_mask)
            sep_list.append(sep_hint)
            boundary_list.append(boundary_confidence)
        return self._batch_from_tensors(
            unit_list=unit_list,
            dur_list=dur_list,
            open_list=open_list,
            sealed_list=sealed_list,
            sep_list=sep_list,
            boundary_list=boundary_list,
            device=device,
        )

    def _batch_from_tensors(
        self,
        *,
        unit_list: list[torch.Tensor],
        dur_list: list[torch.Tensor],
        open_list: list[torch.Tensor],
        sealed_list: list[torch.Tensor],
        sep_list: list[torch.Tensor],
        boundary_list: list[torch.Tensor],
        device: torch.device | None = None,
    ) -> RhythmUnitBatch:
        mask_list = [torch.ones_like(unit_tensor, dtype=torch.float32) for unit_tensor in unit_list]
        content_units = self._pad(unit_list, pad_value=0, dtype=torch.long)
        dur_anchor_src = self._pad(dur_list, pad_value=0, dtype=torch.long)
        open_run_mask = self._pad(open_list, pad_value=0, dtype=torch.long)
        sealed_mask = self._pad(sealed_list, pad_value=0, dtype=torch.long)
        sep_hint = self._pad(sep_list, pad_value=0, dtype=torch.long)
        boundary_confidence = self._pad(boundary_list, pad_value=0, dtype=torch.float32)
        unit_mask = self._pad([m.long() for m in mask_list], pad_value=0, dtype=torch.long).float()

        if device is not None:
            content_units = content_units.to(device)
            dur_anchor_src = dur_anchor_src.to(device)
            open_run_mask = open_run_mask.to(device)
            sealed_mask = sealed_mask.to(device)
            sep_hint = sep_hint.to(device)
            boundary_confidence = boundary_confidence.to(device)
            unit_mask = unit_mask.to(device)

        return RhythmUnitBatch(
            content_units=content_units,
            dur_anchor_src=dur_anchor_src,
            unit_mask=unit_mask,
            open_run_mask=open_run_mask,
            sealed_mask=sealed_mask.float(),
            sep_hint=sep_hint,
            boundary_confidence=boundary_confidence,
        )

    def from_token_lists(
        self,
        batch_tokens: Iterable[Sequence[int]],
        *,
        mark_last_open: bool = True,
        device: torch.device | None = None,
    ) -> RhythmUnitBatch:
        compressed_list = []
        for token_sequence in batch_tokens:
            compressed_list.append(
                build_compressed_sequence(
                    token_sequence,
                    silent_token=self.silent_token,
                    separator_aware=self.separator_aware,
                    tail_open_units=self.tail_open_units,
                    mark_last_open=mark_last_open,
                )
            )
        return self._batch_from_compressed(compressed_list, device=device)

    def from_content_tensor(
        self,
        content: torch.Tensor,
        *,
        content_lengths: torch.Tensor | None = None,
        mark_last_open: bool = True,
    ) -> RhythmUnitBatch:
        if content.dim() != 2:
            raise ValueError(f"content must be rank-2 [B,T], got {tuple(content.shape)}")
        batch_size, total_steps = content.shape
        if content_lengths is None:
            content_lengths = torch.full(
                (batch_size,),
                int(total_steps),
                dtype=torch.long,
                device=content.device,
            )
        state = self.init_stream_state(
            batch_size,
            device=content.device,
        )
        batch, _ = self.step_content_tensor(
            content,
            state,
            content_lengths=content_lengths,
            mark_last_open=mark_last_open,
        )
        return batch

    def from_precomputed(
        self,
        *,
        content_units: torch.Tensor,
        dur_anchor_src: torch.Tensor,
        unit_mask: torch.Tensor | None = None,
        open_run_mask: torch.Tensor | None = None,
        sealed_mask: torch.Tensor | None = None,
        sep_hint: torch.Tensor | None = None,
        boundary_confidence: torch.Tensor | None = None,
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
        if sealed_mask is None:
            sealed_mask = (1 - open_run_mask.long()).clamp_min(0).float() * unit_mask.float()
        else:
            sealed_mask = sealed_mask.float() * unit_mask.float()
        if boundary_confidence is None:
            boundary_confidence = sep_hint.float() * unit_mask.float()
        else:
            boundary_confidence = boundary_confidence.float() * unit_mask.float()
        return RhythmUnitBatch(
            content_units=content_units,
            dur_anchor_src=dur_anchor_src,
            unit_mask=unit_mask,
            open_run_mask=open_run_mask.long(),
            sealed_mask=sealed_mask,
            sep_hint=sep_hint.long(),
            boundary_confidence=boundary_confidence,
        )

    def init_stream_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
    ) -> StreamingUnitizerState:
        return self.unitizer.init_state(batch_size, device=device)

    def step_content_tensor(
        self,
        content: torch.Tensor,
        state: StreamingUnitizerState,
        *,
        content_lengths: torch.Tensor | None = None,
        mark_last_open: bool = True,
    ) -> tuple[RhythmUnitBatch, StreamingUnitizerState]:
        if content.dim() != 2:
            raise ValueError(f"content must be rank-2 [B,T], got {tuple(content.shape)}")
        batch_size, total_steps = content.shape
        if content_lengths is None:
            content_lengths = torch.full(
                (batch_size,),
                int(total_steps),
                dtype=torch.long,
                device=content.device,
            )
        next_rows: list[StreamingUnitizerRowState] = []
        for batch_idx in range(batch_size):
            valid_len = int(content_lengths[batch_idx].item())
            valid_len = max(0, min(valid_len, int(total_steps)))
            row_state = state.rows[batch_idx] if batch_idx < len(state.rows) else self.unitizer._empty_row_state(content.device)
            row_state = self._move_row_state(row_state, content.device)
            chunk = content[batch_idx, :valid_len].long()
            next_rows.append(self.unitizer.step_chunk_tensor(chunk, row_state))
        next_state = StreamingUnitizerState(rows=next_rows)
        batch = self._batch_from_row_states(next_rows, mark_last_open=mark_last_open, device=content.device)
        return batch, next_state

    def step_token_lists(
        self,
        batch_tokens: Iterable[Sequence[int]],
        state: StreamingUnitizerState,
        *,
        mark_last_open: bool = True,
        device: torch.device | None = None,
    ) -> tuple[RhythmUnitBatch, StreamingUnitizerState]:
        compressed_list, next_state = self.unitizer.step_token_lists(
            batch_tokens,
            state,
            mark_last_open=mark_last_open,
        )
        batch = self._batch_from_compressed(compressed_list, device=device)
        return batch, next_state
