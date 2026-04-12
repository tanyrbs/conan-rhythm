from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
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
    run_stability: list[float]
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


def _merge_short_flicker_runs(
    units: list[int],
    durations: list[int],
    silence_mask: list[int],
    sep_hint: list[int],
    *,
    min_speech_frames: int = 2,
    min_silence_frames: int = 2,
) -> tuple[list[int], list[int], list[int], list[int]]:
    min_speech_frames = int(max(1, min_speech_frames))
    min_silence_frames = int(max(1, min_silence_frames))
    if len(units) < 3 or (min_speech_frames <= 1 and min_silence_frames <= 1):
        return units, durations, silence_mask, sep_hint
    changed = True
    while changed and len(units) >= 3:
        changed = False
        idx = 1
        while idx < len(units) - 1:
            min_run_frames = min_silence_frames if int(silence_mask[idx]) > 0 else min_speech_frames
            bridgeable = (
                int(units[idx - 1]) == int(units[idx + 1])
                and int(silence_mask[idx - 1]) == int(silence_mask[idx + 1])
                and int(sep_hint[idx - 1]) == 0
                and int(sep_hint[idx]) == 0
            )
            if int(durations[idx]) <= min_run_frames and bridgeable:
                durations[idx - 1] += int(durations[idx]) + int(durations[idx + 1])
                sep_hint[idx - 1] = int(bool(sep_hint[idx - 1] or sep_hint[idx + 1]))
                del units[idx : idx + 2]
                del durations[idx : idx + 2]
                del silence_mask[idx : idx + 2]
                del sep_hint[idx : idx + 2]
                changed = True
                idx = max(1, idx - 1)
                continue
            idx += 1
    return units, durations, silence_mask, sep_hint


def _suppress_micro_silence_runs(
    units: list[int],
    durations: list[int],
    silence_mask: list[int],
    sep_hint: list[int],
    *,
    max_silence_frames: int = 1,
) -> tuple[list[int], list[int], list[int], list[int]]:
    max_silence_frames = int(max(0, max_silence_frames))
    if len(units) < 3 or max_silence_frames <= 0:
        return units, durations, silence_mask, sep_hint
    idx = 1
    while idx < len(units) - 1:
        is_micro_silence = (
            int(silence_mask[idx]) == 1
            and int(durations[idx]) <= max_silence_frames
            and int(sep_hint[idx - 1]) == 0
            and int(sep_hint[idx]) == 0
            and int(silence_mask[idx - 1]) == 0
            and int(silence_mask[idx + 1]) == 0
        )
        if is_micro_silence:
            if int(durations[idx - 1]) <= int(durations[idx + 1]):
                durations[idx - 1] += int(durations[idx])
            else:
                durations[idx + 1] += int(durations[idx])
            del units[idx]
            del durations[idx]
            del silence_mask[idx]
            del sep_hint[idx]
            continue
        idx += 1
    return units, durations, silence_mask, sep_hint


def _stabilize_run_lists(
    units: list[int],
    durations: list[int],
    silence_mask: list[int],
    sep_hint: list[int],
    *,
    min_speech_frames: int = 2,
    min_silence_frames: int = 2,
    max_micro_silence_frames: int = 1,
) -> tuple[list[int], list[int], list[int], list[int]]:
    units, durations, silence_mask, sep_hint = _merge_short_flicker_runs(
        units,
        durations,
        silence_mask,
        sep_hint,
        min_speech_frames=min_speech_frames,
        min_silence_frames=min_silence_frames,
    )
    units, durations, silence_mask, sep_hint = _suppress_micro_silence_runs(
        units,
        durations,
        silence_mask,
        sep_hint,
        max_silence_frames=max_micro_silence_frames,
    )
    return units, durations, silence_mask, sep_hint


def _merge_short_jitter_runs(
    units: list[int],
    durations: list[int],
    silence_mask: list[int],
    sep_hint: list[int],
    *,
    min_run_frames: int = 2,
) -> tuple[list[int], list[int], list[int], list[int]]:
    return _merge_short_flicker_runs(
        units,
        durations,
        silence_mask,
        sep_hint,
        min_speech_frames=min_run_frames,
        min_silence_frames=min_run_frames,
    )


def _cpu_int64_list(values: torch.Tensor | Sequence[int] | Sequence[float]) -> list[int]:
    if isinstance(values, torch.Tensor):
        return values.detach().to(device="cpu", dtype=torch.long).reshape(-1).numpy().tolist()
    return np.asarray(values, dtype=np.int64).reshape(-1).tolist()


def _debounce_tail_tensors(
    units: torch.Tensor,
    durations: torch.Tensor,
    silence_mask: torch.Tensor,
    sep_hint: torch.Tensor,
    *,
    mutable_start: int = 0,
    min_speech_frames: int = 2,
    min_silence_frames: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if units.numel() < 3:
        return units, durations, silence_mask, sep_hint
    prefix_len = max(0, min(int(mutable_start), int(units.numel())))
    if int(units.numel()) - prefix_len < 3:
        return units, durations, silence_mask, sep_hint
    tail_units = _cpu_int64_list(units[prefix_len:])
    tail_durations = _cpu_int64_list(durations[prefix_len:])
    tail_silence = _cpu_int64_list(silence_mask[prefix_len:])
    tail_sep = _cpu_int64_list(sep_hint[prefix_len:])
    tail_units, tail_durations, tail_silence, tail_sep = _stabilize_run_lists(
        tail_units,
        tail_durations,
        tail_silence,
        tail_sep,
        min_speech_frames=min_speech_frames,
        min_silence_frames=min_silence_frames,
        max_micro_silence_frames=1,
    )
    device = units.device
    tail_units_tensor = torch.tensor(tail_units, dtype=units.dtype, device=device)
    tail_durations_tensor = torch.tensor(tail_durations, dtype=durations.dtype, device=device)
    tail_silence_tensor = torch.tensor(tail_silence, dtype=silence_mask.dtype, device=device)
    tail_sep_tensor = torch.tensor(tail_sep, dtype=sep_hint.dtype, device=device)
    if prefix_len <= 0:
        return tail_units_tensor, tail_durations_tensor, tail_silence_tensor, tail_sep_tensor
    return (
        torch.cat([units[:prefix_len], tail_units_tensor], dim=0),
        torch.cat([durations[:prefix_len], tail_durations_tensor], dim=0),
        torch.cat([silence_mask[:prefix_len], tail_silence_tensor], dim=0),
        torch.cat([sep_hint[:prefix_len], tail_sep_tensor], dim=0),
    )


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


def _estimate_run_stability_tensor(
    durations: Sequence[int] | Sequence[float] | torch.Tensor,
    silence_mask: Sequence[int] | Sequence[float] | torch.Tensor,
    open_run_mask: Sequence[int] | Sequence[float] | torch.Tensor,
    *,
    sep_hint: Sequence[int] | Sequence[float] | torch.Tensor | None = None,
    boundary_confidence: Sequence[float] | torch.Tensor | None = None,
    min_speech_frames: int = 2,
    min_silence_frames: int = 2,
) -> torch.Tensor:
    device = None
    for candidate in (durations, silence_mask, open_run_mask, sep_hint, boundary_confidence):
        if isinstance(candidate, torch.Tensor):
            device = candidate.device
            break
    dur = _to_1d_tensor(durations, dtype=torch.float32, device=device)
    if dur.numel() <= 0:
        return dur
    silence = _to_1d_tensor(silence_mask, dtype=torch.float32, device=dur.device)
    open_mask = _to_1d_tensor(open_run_mask, dtype=torch.float32, device=dur.device)
    sep = (
        _to_1d_tensor(sep_hint, dtype=torch.float32, device=dur.device)
        if sep_hint is not None
        else torch.zeros_like(dur)
    )
    boundary = (
        _to_1d_tensor(boundary_confidence, dtype=torch.float32, device=dur.device)
        if boundary_confidence is not None
        else _estimate_boundary_confidence_tensor(dur, sep, open_mask)
    ).clamp(0.0, 1.0)
    min_speech = float(max(1, int(min_speech_frames)))
    min_silence = float(max(1, int(min_silence_frames)))
    min_frames = torch.where(
        silence > 0.5,
        torch.full_like(dur, min_silence),
        torch.full_like(dur, min_speech),
    )
    length_term = torch.clamp(dur / min_frames.clamp_min(1.0), min=0.0, max=1.0)
    boundary_term = torch.maximum(boundary, sep.clamp(0.0, 1.0))
    stability = 0.20 + 0.60 * length_term + 0.20 * boundary_term
    stability = stability * (1.0 - 0.35 * open_mask.clamp(0.0, 1.0))
    return stability.clamp(0.05, 1.0)


def estimate_run_stability(
    durations: Sequence[int] | Sequence[float],
    silence_mask: Sequence[int] | Sequence[float],
    open_run_mask: Sequence[int] | Sequence[float],
    *,
    sep_hint: Sequence[int] | Sequence[float] | None = None,
    boundary_confidence: Sequence[float] | None = None,
    min_speech_frames: int = 2,
    min_silence_frames: int = 2,
) -> list[float]:
    return _estimate_run_stability_tensor(
        durations,
        silence_mask,
        open_run_mask,
        sep_hint=sep_hint,
        boundary_confidence=boundary_confidence,
        min_speech_frames=min_speech_frames,
        min_silence_frames=min_silence_frames,
    ).tolist()


def build_compressed_sequence(
    token_sequence: Sequence[int],
    *,
    silent_token: int | None = None,
    separator_aware: bool = False,
    tail_open_units: int = 1,
    mark_last_open: bool = True,
    emit_silence_runs: bool = False,
    debounce_min_run_frames: int = 1,
) -> CompressedUnitSequence:
    units, durations, silence_mask, sep_hint = compress_token_sequence(
        token_sequence,
        silent_token=silent_token,
        separator_aware=separator_aware,
        emit_silence_runs=emit_silence_runs,
    )
    open_tail = min(len(units), max(0, int(tail_open_units)) if mark_last_open else 0)
    sealed_limit = max(0, len(units) - open_tail)
    if sealed_limit > 0:
        prefix_units, prefix_durations, prefix_silence, prefix_sep = _stabilize_run_lists(
            units[:sealed_limit],
            durations[:sealed_limit],
            silence_mask[:sealed_limit],
            sep_hint[:sealed_limit],
            min_speech_frames=debounce_min_run_frames,
            min_silence_frames=debounce_min_run_frames,
            max_micro_silence_frames=1,
        )
        units = prefix_units + units[sealed_limit:]
        durations = prefix_durations + durations[sealed_limit:]
        silence_mask = prefix_silence + silence_mask[sealed_limit:]
        sep_hint = prefix_sep + sep_hint[sealed_limit:]
    elif not mark_last_open:
        units, durations, silence_mask, sep_hint = _stabilize_run_lists(
            units,
            durations,
            silence_mask,
            sep_hint,
            min_speech_frames=debounce_min_run_frames,
            min_silence_frames=debounce_min_run_frames,
            max_micro_silence_frames=1,
        )
    open_run_mask = [0 for _ in units]
    if mark_last_open and len(open_run_mask) > 0:
        keep_open_from = max(0, len(open_run_mask) - open_tail)
        for idx in range(keep_open_from, len(open_run_mask)):
            open_run_mask[idx] = 1
    sealed_mask = [1 - x for x in open_run_mask]
    boundary_confidence = estimate_boundary_confidence(durations, sep_hint, open_run_mask)
    run_stability = estimate_run_stability(
        durations,
        silence_mask,
        open_run_mask,
        sep_hint=sep_hint,
        boundary_confidence=boundary_confidence,
        min_speech_frames=debounce_min_run_frames,
        min_silence_frames=debounce_min_run_frames,
    )
    tail_buffer = units[max(0, len(units) - open_tail):] if (mark_last_open and open_tail > 0) else []
    return CompressedUnitSequence(
        units=units,
        durations=durations,
        silence_mask=silence_mask,
        sep_hint=sep_hint,
        open_run_mask=open_run_mask,
        sealed_mask=sealed_mask,
        boundary_confidence=boundary_confidence,
        run_stability=run_stability,
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
        debounce_min_run_frames: int = 1,
    ) -> None:
        self.silent_token = silent_token
        self.separator_aware = bool(separator_aware)
        self.tail_open_units = max(1, int(tail_open_units))
        self.emit_silence_runs = bool(emit_silence_runs)
        self.debounce_min_run_frames = int(max(1, debounce_min_run_frames))

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
            debounce_min_run_frames=self.debounce_min_run_frames,
        )

    def _export_row_tensors(
        self,
        row_state: StreamingUnitizerRowState,
        *,
        mark_last_open: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        units_list = _cpu_int64_list(row_state.units)
        durations_list = _cpu_int64_list(row_state.durations)
        silence_list = _cpu_int64_list(row_state.silence_mask)
        sep_list = _cpu_int64_list(row_state.sep_hint)
        open_tail = min(len(units_list), self.tail_open_units if mark_last_open else 0)
        sealed_limit = max(0, len(units_list) - open_tail)
        if sealed_limit > 0:
            prefix_units = units_list[:sealed_limit]
            prefix_durations = durations_list[:sealed_limit]
            prefix_silence = silence_list[:sealed_limit]
            prefix_sep = sep_list[:sealed_limit]
            prefix_units, prefix_durations, prefix_silence, prefix_sep = _stabilize_run_lists(
                prefix_units,
                prefix_durations,
                prefix_silence,
                prefix_sep,
                min_speech_frames=self.debounce_min_run_frames,
                min_silence_frames=self.debounce_min_run_frames,
                max_micro_silence_frames=1,
            )
            units_list = prefix_units + units_list[sealed_limit:]
            durations_list = prefix_durations + durations_list[sealed_limit:]
            silence_list = prefix_silence + silence_list[sealed_limit:]
            sep_list = prefix_sep + sep_list[sealed_limit:]
        elif not mark_last_open:
            units_list, durations_list, silence_list, sep_list = _stabilize_run_lists(
                units_list,
                durations_list,
                silence_list,
                sep_list,
                min_speech_frames=self.debounce_min_run_frames,
                min_silence_frames=self.debounce_min_run_frames,
                max_micro_silence_frames=1,
            )
        device = row_state.units.device if row_state.units.numel() > 0 else row_state.durations.device
        units = torch.tensor(units_list, dtype=torch.long, device=device)
        durations = torch.tensor(durations_list, dtype=torch.long, device=device)
        silence_mask = torch.tensor(silence_list, dtype=torch.long, device=device)
        sep_hint = torch.tensor(sep_list, dtype=torch.long, device=device)
        open_run_mask = torch.zeros_like(units, dtype=torch.long, device=device)
        if mark_last_open and open_run_mask.numel() > 0:
            keep_open_from = max(0, open_run_mask.numel() - self.tail_open_units)
            open_run_mask[keep_open_from:] = 1
        sealed_mask = (1 - open_run_mask).clamp_min(0)
        boundary_confidence = _estimate_boundary_confidence_tensor(durations, sep_hint, open_run_mask)
        run_stability = _estimate_run_stability_tensor(
            durations,
            silence_mask,
            open_run_mask,
            sep_hint=sep_hint,
            boundary_confidence=boundary_confidence,
            min_speech_frames=self.debounce_min_run_frames,
            min_silence_frames=self.debounce_min_run_frames,
        )
        return units, durations, silence_mask, open_run_mask, sealed_mask, sep_hint, boundary_confidence, run_stability

    def _export_row(
        self,
        row_state: StreamingUnitizerRowState,
        *,
        mark_last_open: bool = True,
    ) -> CompressedUnitSequence:
        units, durations, silence_mask, open_run_mask, sealed_mask, sep_hint, boundary_confidence, run_stability = self._export_row_tensors(
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
            run_stability=run_stability.tolist(),
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
        prev_units_len = int(units.numel())
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

        units, durations, silence_mask, sep_hint = _debounce_tail_tensors(
            units,
            durations,
            silence_mask,
            sep_hint,
            mutable_start=max(0, prev_units_len - self.tail_open_units),
            min_speech_frames=self.debounce_min_run_frames,
            min_silence_frames=self.debounce_min_run_frames,
        )

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
