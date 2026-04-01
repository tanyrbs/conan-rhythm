from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F


@dataclass
class CompressedUnitSequence:
    units: list[int]
    durations: list[int]
    sep_hint: list[int]
    open_run_mask: list[int]
    sealed_mask: list[int]
    boundary_confidence: list[float]
    tail_buffer: list[int]


@dataclass
class StreamingUnitizerState:
    raw_tokens: list[list[int]]


def compress_token_sequence(
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


def _standardize_1d(values: torch.Tensor) -> torch.Tensor:
    if values.numel() <= 0:
        return values
    mean = values.mean()
    var = ((values - mean) ** 2).mean()
    return (values - mean) / var.clamp_min(1e-6).sqrt()


def estimate_boundary_confidence(
    durations: Sequence[int],
    sep_hint: Sequence[int],
    open_run_mask: Sequence[int],
) -> list[float]:
    if len(durations) <= 0:
        return []
    dur = torch.tensor(list(durations), dtype=torch.float32)
    sep = torch.tensor(list(sep_hint), dtype=torch.float32)
    open_mask = torch.tensor(list(open_run_mask), dtype=torch.float32)
    log_anchor = torch.log1p(dur.clamp_min(0.0))
    prev_anchor = F.pad(log_anchor[:-1], (1, 0))
    next_anchor = F.pad(log_anchor[1:], (0, 1))
    local_peak = torch.relu(log_anchor - 0.5 * (prev_anchor + next_anchor))
    local_jump = 0.5 * (torch.abs(log_anchor - prev_anchor) + torch.abs(next_anchor - log_anchor))
    cue = 0.30 * torch.sigmoid(_standardize_1d(local_peak))
    cue = cue + 0.20 * torch.sigmoid(_standardize_1d(local_jump))
    cue = cue + 0.55 * sep
    cue = cue * (1.0 - 0.25 * open_mask)
    return cue.clamp(0.0, 1.0).tolist()


def build_compressed_sequence(
    token_sequence: Sequence[int],
    *,
    silent_token: int | None = None,
    separator_aware: bool = False,
    tail_open_units: int = 1,
    mark_last_open: bool = True,
) -> CompressedUnitSequence:
    units, durations, sep_hint = compress_token_sequence(
        token_sequence,
        silent_token=silent_token,
        separator_aware=separator_aware,
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
    ) -> None:
        self.silent_token = silent_token
        self.separator_aware = bool(separator_aware)
        self.tail_open_units = max(1, int(tail_open_units))

    def init_state(self, batch_size: int) -> StreamingUnitizerState:
        return StreamingUnitizerState(raw_tokens=[[] for _ in range(int(batch_size))])

    def compress(self, token_sequence: Sequence[int], *, mark_last_open: bool = True) -> CompressedUnitSequence:
        return build_compressed_sequence(
            token_sequence,
            silent_token=self.silent_token,
            separator_aware=self.separator_aware,
            tail_open_units=self.tail_open_units,
            mark_last_open=mark_last_open,
        )

    def step_token_lists(
        self,
        batch_tokens: Iterable[Sequence[int]],
        state: StreamingUnitizerState,
        *,
        mark_last_open: bool = True,
    ) -> tuple[list[CompressedUnitSequence], StreamingUnitizerState]:
        results: list[CompressedUnitSequence] = []
        new_state = StreamingUnitizerState(raw_tokens=[list(tokens) for tokens in state.raw_tokens])
        for idx, token_chunk in enumerate(batch_tokens):
            history = list(new_state.raw_tokens[idx])
            history.extend(int(token) for token in token_chunk)
            new_state.raw_tokens[idx] = history
            results.append(self.compress(history, mark_last_open=mark_last_open))
        return results, new_state
