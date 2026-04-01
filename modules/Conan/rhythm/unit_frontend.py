from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from .unitizer import (
    StreamingRunLengthUnitizer,
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
        sealed_list = []
        sep_list = []
        boundary_list = []
        mask_list = []
        for token_sequence in batch_tokens:
            compressed = build_compressed_sequence(
                token_sequence,
                silent_token=self.silent_token,
                separator_aware=self.separator_aware,
                tail_open_units=self.tail_open_units,
                mark_last_open=mark_last_open,
            )
            unit_tensor = torch.tensor(compressed.units, dtype=torch.long)
            dur_tensor = torch.tensor(compressed.durations, dtype=torch.long)
            open_tensor = torch.tensor(compressed.open_run_mask, dtype=torch.long)
            sealed_tensor = torch.tensor(compressed.sealed_mask, dtype=torch.long)
            sep_tensor = torch.tensor(compressed.sep_hint, dtype=torch.long)
            boundary_tensor = torch.tensor(compressed.boundary_confidence, dtype=torch.float32)
            mask_tensor = torch.ones_like(unit_tensor, dtype=torch.float32)
            unit_list.append(unit_tensor)
            dur_list.append(dur_tensor)
            open_list.append(open_tensor)
            sealed_list.append(sealed_tensor)
            sep_list.append(sep_tensor)
            boundary_list.append(boundary_tensor)
            mask_list.append(mask_tensor)

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

    def init_stream_state(self, batch_size: int) -> StreamingUnitizerState:
        return self.unitizer.init_state(batch_size)

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
        unit_list = [torch.tensor(item.units, dtype=torch.long) for item in compressed_list]
        dur_list = [torch.tensor(item.durations, dtype=torch.long) for item in compressed_list]
        open_list = [torch.tensor(item.open_run_mask, dtype=torch.long) for item in compressed_list]
        sealed_list = [torch.tensor(item.sealed_mask, dtype=torch.long) for item in compressed_list]
        sep_list = [torch.tensor(item.sep_hint, dtype=torch.long) for item in compressed_list]
        boundary_list = [torch.tensor(item.boundary_confidence, dtype=torch.float32) for item in compressed_list]
        mask_list = [torch.ones_like(unit_tensor, dtype=torch.float32) for unit_tensor in unit_list]
        batch = RhythmUnitBatch(
            content_units=self._pad(unit_list, pad_value=0, dtype=torch.long),
            dur_anchor_src=self._pad(dur_list, pad_value=0, dtype=torch.long),
            unit_mask=self._pad([m.long() for m in mask_list], pad_value=0, dtype=torch.long).float(),
            open_run_mask=self._pad(open_list, pad_value=0, dtype=torch.long),
            sealed_mask=self._pad(sealed_list, pad_value=0, dtype=torch.long).float(),
            sep_hint=self._pad(sep_list, pad_value=0, dtype=torch.long),
            boundary_confidence=self._pad(boundary_list, pad_value=0, dtype=torch.float32),
        )
        if device is not None:
            batch = RhythmUnitBatch(
                content_units=batch.content_units.to(device),
                dur_anchor_src=batch.dur_anchor_src.to(device),
                unit_mask=batch.unit_mask.to(device),
                open_run_mask=batch.open_run_mask.to(device),
                sealed_mask=batch.sealed_mask.to(device),
                sep_hint=batch.sep_hint.to(device),
                boundary_confidence=batch.boundary_confidence.to(device),
            )
        return batch, next_state
