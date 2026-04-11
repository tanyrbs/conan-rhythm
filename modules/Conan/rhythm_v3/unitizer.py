from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F


@dataclass
class CompressedUnitSequence:
    units: list[int]
    durations: list[int]
    silence_mask: list[int]
    sep_hint: list[int]
    open_run_mask: list[int]
    sealed_mask: list[int]
    boundary_confidence: list[float]
    tail_buffer: list[int]


@dataclass
class StreamingUnitizerRowState:
    units: torch.Tensor
    durations: torch.Tensor
    silence_mask: torch.Tensor
    sep_hint: torch.Tensor
    last_token: int = -1
    pending_separator: bool = False


@dataclass
class StreamingUnitizerState:
    rows: list[StreamingUnitizerRowState]


def compress_token_sequence(
    token_sequence: Sequence[int],
    *,
    silent_token: int | None = None,
    separator_aware: bool = False,
    emit_silence_runs: bool = False,
) -> tuple[list[int], list[int], list[int], list[int]]:
    units: list[int] = []
    durations: list[int] = []
    silence_mask: list[int] = []
    sep_hint: list[int] = []
    prev_token: int | None = None
    saw_separator = False
    for token in token_sequence:
        token = int(token)
        if token < 0:
            continue
        is_silence = silent_token is not None and token == int(silent_token)
        if is_silence and not emit_silence_runs:
            if separator_aware and len(units) > 0:
                saw_separator = True
            continue
        start_new = len(units) == 0 or token != prev_token or (separator_aware and saw_separator)
        if start_new:
            if saw_separator and separator_aware and len(sep_hint) > 0:
                sep_hint[-1] = 1
            units.append(token)
            durations.append(1)
            silence_mask.append(1 if is_silence else 0)
            sep_hint.append(0)
        else:
            durations[-1] += 1
        prev_token = token
        saw_separator = False
    return units, durations, silence_mask, sep_hint


def _standardize_1d(values: torch.Tensor) -> torch.Tensor:
    if values.numel() <= 0:
        return values
    mean = values.mean()
    var = ((values - mean) ** 2).mean()
    return (values - mean) / var.clamp_min(1e-6).sqrt()


def _to_1d_tensor(
    values: Sequence[int] | Sequence[float] | torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        out = values.detach()
        if device is None:
            device = out.device
        return out.to(device=device, dtype=dtype).reshape(-1)
    return torch.tensor(list(values), dtype=dtype, device=device)


def _estimate_boundary_confidence_tensor(
    durations: Sequence[int] | torch.Tensor,
    sep_hint: Sequence[int] | torch.Tensor,
    open_run_mask: Sequence[int] | torch.Tensor,
) -> torch.Tensor:
    device = None
    for candidate in (durations, sep_hint, open_run_mask):
        if isinstance(candidate, torch.Tensor):
            device = candidate.device
            break
    dur = _to_1d_tensor(durations, dtype=torch.float32, device=device)
    if dur.numel() <= 0:
        return dur
    sep = _to_1d_tensor(sep_hint, dtype=torch.float32, device=dur.device)
    open_mask = _to_1d_tensor(open_run_mask, dtype=torch.float32, device=dur.device)
    log_anchor = torch.log1p(dur.clamp_min(0.0))
    prev_anchor = F.pad(log_anchor[:-1], (1, 0))
    next_anchor = F.pad(log_anchor[1:], (0, 1))
    local_peak = torch.relu(log_anchor - 0.5 * (prev_anchor + next_anchor))
    local_jump = 0.5 * (torch.abs(log_anchor - prev_anchor) + torch.abs(next_anchor - log_anchor))
    cue = 0.30 * torch.sigmoid(_standardize_1d(local_peak))
    cue = cue + 0.20 * torch.sigmoid(_standardize_1d(local_jump))
    cue = cue + 0.55 * sep
    cue = cue * (1.0 - 0.25 * open_mask)
    return cue.clamp(0.0, 1.0)


def estimate_boundary_confidence(
    durations: Sequence[int],
    sep_hint: Sequence[int],
    open_run_mask: Sequence[int],
) -> list[float]:
    return _estimate_boundary_confidence_tensor(durations, sep_hint, open_run_mask).tolist()


def build_compressed_sequence(
    token_sequence: Sequence[int],
    *,
    silent_token: int | None = None,
    separator_aware: bool = False,
    tail_open_units: int = 1,
    mark_last_open: bool = True,
    emit_silence_runs: bool = False,
) -> CompressedUnitSequence:
    units, durations, silence_mask, sep_hint = compress_token_sequence(
        token_sequence,
        silent_token=silent_token,
        separator_aware=separator_aware,
        emit_silence_runs=emit_silence_runs,
    )
    open_run_mask = [0 for _ in units]
    if mark_last_open and len(open_run_mask) > 0:
        keep_open_from = max(0, len(open_run_mask) - max(1, int(tail_open_units)))
        for idx in range(keep_open_from, len(open_run_mask)):
            open_run_mask[idx] = 1
    sealed_mask = [1 - x for x in open_run_mask]
    boundary_confidence = estimate_boundary_confidence(durations, sep_hint, open_run_mask)
    tail_buffer = units[max(0, len(units) - max(1, int(tail_open_units))):] if mark_last_open else []
    return CompressedUnitSequence(
        units=units,
        durations=durations,
        silence_mask=silence_mask,
        sep_hint=sep_hint,
        open_run_mask=open_run_mask,
        sealed_mask=sealed_mask,
        boundary_confidence=boundary_confidence,
        tail_buffer=tail_buffer,
    )


class StreamingRunLengthUnitizer:
    def __init__(
        self,
        *,
        silent_token: int | None = None,
        separator_aware: bool = True,
        tail_open_units: int = 1,
        emit_silence_runs: bool = False,
    ) -> None:
        self.silent_token = silent_token
        self.separator_aware = bool(separator_aware)
        self.tail_open_units = max(1, int(tail_open_units))
        self.emit_silence_runs = bool(emit_silence_runs)

    @staticmethod
    def _empty_row_state(device: torch.device | None = None) -> StreamingUnitizerRowState:
        return StreamingUnitizerRowState(
            units=torch.empty(0, dtype=torch.long, device=device),
            durations=torch.empty(0, dtype=torch.long, device=device),
            silence_mask=torch.empty(0, dtype=torch.long, device=device),
            sep_hint=torch.empty(0, dtype=torch.long, device=device),
            last_token=-1,
            pending_separator=False,
        )

    @staticmethod
    def _clone_row_state(row: StreamingUnitizerRowState) -> StreamingUnitizerRowState:
        return StreamingUnitizerRowState(
            units=row.units.clone(),
            durations=row.durations.clone(),
            silence_mask=row.silence_mask.clone(),
            sep_hint=row.sep_hint.clone(),
            last_token=int(row.last_token),
            pending_separator=bool(row.pending_separator),
        )

    def init_state(self, batch_size: int, device: torch.device | None = None) -> StreamingUnitizerState:
        return StreamingUnitizerState(rows=[self._empty_row_state(device) for _ in range(int(batch_size))])

    def compress(self, token_sequence: Sequence[int], *, mark_last_open: bool = True) -> CompressedUnitSequence:
        return build_compressed_sequence(
            token_sequence,
            silent_token=self.silent_token,
            separator_aware=self.separator_aware,
            tail_open_units=self.tail_open_units,
            mark_last_open=mark_last_open,
            emit_silence_runs=self.emit_silence_runs,
        )

    def _export_row_tensors(
        self,
        row_state: StreamingUnitizerRowState,
        *,
        mark_last_open: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        units = row_state.units
        durations = row_state.durations
        silence_mask = row_state.silence_mask
        sep_hint = row_state.sep_hint
        device = units.device if units.numel() > 0 else durations.device
        open_run_mask = torch.zeros_like(units, dtype=torch.long, device=device)
        if mark_last_open and open_run_mask.numel() > 0:
            keep_open_from = max(0, open_run_mask.numel() - self.tail_open_units)
            open_run_mask[keep_open_from:] = 1
        sealed_mask = (1 - open_run_mask).clamp_min(0)
        boundary_confidence = _estimate_boundary_confidence_tensor(durations, sep_hint, open_run_mask)
        return units, durations, silence_mask, open_run_mask, sealed_mask, sep_hint, boundary_confidence

    def _export_row(
        self,
        row_state: StreamingUnitizerRowState,
        *,
        mark_last_open: bool = True,
    ) -> CompressedUnitSequence:
        units, durations, silence_mask, open_run_mask, sealed_mask, sep_hint, boundary_confidence = self._export_row_tensors(
            row_state,
            mark_last_open=mark_last_open,
        )
        tail_buffer = units[max(0, units.numel() - self.tail_open_units):].tolist() if mark_last_open else []
        return CompressedUnitSequence(
            units=units.tolist(),
            durations=durations.tolist(),
            silence_mask=silence_mask.tolist(),
            sep_hint=sep_hint.tolist(),
            open_run_mask=open_run_mask.tolist(),
            sealed_mask=sealed_mask.tolist(),
            boundary_confidence=boundary_confidence.tolist(),
            tail_buffer=tail_buffer,
        )

    def step_chunk_tensor(
        self,
        token_chunk: torch.Tensor,
        state_row: StreamingUnitizerRowState,
    ) -> StreamingUnitizerRowState:
        if token_chunk.dim() != 1:
            token_chunk = token_chunk.reshape(-1)
        row = self._clone_row_state(state_row)
        units = row.units
        durations = row.durations
        silence_mask = row.silence_mask
        sep_hint = row.sep_hint
        last_token = int(row.last_token)
        pending_separator = bool(row.pending_separator)
        appended_units: list[int] = []
        appended_durations: list[int] = []
        appended_silence: list[int] = []
        appended_sep: list[int] = []
        sep_hint_cloned = False
        durations_cloned = False

        def _has_previous_unit() -> bool:
            return bool(units.numel() > 0 or len(appended_units) > 0)

        def _mark_previous_separator() -> None:
            nonlocal sep_hint, sep_hint_cloned
            if appended_sep:
                appended_sep[-1] = 1
                return
            if sep_hint.numel() <= 0:
                return
            if not sep_hint_cloned:
                sep_hint = sep_hint.clone()
                sep_hint_cloned = True
            sep_hint[-1] = 1

        def _extend_previous_duration() -> None:
            nonlocal durations, durations_cloned
            if appended_durations:
                appended_durations[-1] += 1
                return
            if durations.numel() <= 0:
                appended_units.append(token_id)
                appended_durations.append(1)
                appended_silence.append(1 if (self.silent_token is not None and token_id == int(self.silent_token)) else 0)
                appended_sep.append(0)
                return
            if not durations_cloned:
                durations = durations.clone()
                durations_cloned = True
            durations[-1] += 1

        for token in token_chunk:
            token_id = int(token.item())
            if token_id < 0:
                continue
            is_silence = self.silent_token is not None and token_id == int(self.silent_token)
            if is_silence and not self.emit_silence_runs:
                if self.separator_aware and _has_previous_unit():
                    pending_separator = True
                continue
            start_new = (not _has_previous_unit()) or token_id != last_token or (self.separator_aware and pending_separator)
            if start_new:
                if self.separator_aware and pending_separator and _has_previous_unit():
                    _mark_previous_separator()
                appended_units.append(token_id)
                appended_durations.append(1)
                appended_silence.append(1 if is_silence else 0)
                appended_sep.append(0)
            else:
                _extend_previous_duration()
            last_token = token_id
            pending_separator = False

        if appended_units:
            device = units.device if units.numel() > 0 else token_chunk.device
            new_units = torch.tensor(appended_units, dtype=torch.long, device=device)
            new_durations = torch.tensor(appended_durations, dtype=torch.long, device=device)
            new_silence = torch.tensor(appended_silence, dtype=torch.long, device=device)
            new_sep_hint = torch.tensor(appended_sep, dtype=torch.long, device=device)
            units = torch.cat([units, new_units], dim=0) if units.numel() > 0 else new_units
            durations = torch.cat([durations, new_durations], dim=0) if durations.numel() > 0 else new_durations
            silence_mask = torch.cat([silence_mask, new_silence], dim=0) if silence_mask.numel() > 0 else new_silence
            sep_hint = torch.cat([sep_hint, new_sep_hint], dim=0) if sep_hint.numel() > 0 else new_sep_hint

        row.units = units
        row.durations = durations
        row.silence_mask = silence_mask
        row.sep_hint = sep_hint
        row.last_token = last_token if row.units.numel() > 0 else -1
        row.pending_separator = pending_separator
        return row

    def step_token_lists(
        self,
        batch_tokens: Iterable[Sequence[int]],
        state: StreamingUnitizerState,
        *,
        mark_last_open: bool = True,
    ) -> tuple[list[CompressedUnitSequence], StreamingUnitizerState]:
        results: list[CompressedUnitSequence] = []
        next_rows: list[StreamingUnitizerRowState] = []
        for idx, token_chunk in enumerate(batch_tokens):
            row_state = state.rows[idx] if idx < len(state.rows) else self._empty_row_state()
            chunk_tensor = torch.tensor(list(token_chunk), dtype=torch.long, device=row_state.units.device)
            next_row = self.step_chunk_tensor(chunk_tensor, row_state)
            next_rows.append(next_row)
            results.append(self._export_row(next_row, mark_last_open=mark_last_open))
        return results, StreamingUnitizerState(rows=next_rows)
