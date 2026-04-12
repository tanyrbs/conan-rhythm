from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from modules.Conan.rhythm.source_boundary import build_source_boundary_cue

from .unit_frontend import RhythmUnitFrontend
from .unitizer import estimate_boundary_confidence, estimate_run_stability


def _as_token_list(content_tokens) -> list[int]:
    if isinstance(content_tokens, str):
        return [int(float(x)) for x in content_tokens.split() if str(x).strip() != ""]
    if isinstance(content_tokens, np.ndarray):
        return [int(x) for x in content_tokens.tolist()]
    if torch.is_tensor(content_tokens):
        return [int(x) for x in content_tokens.detach().cpu().reshape(-1).tolist()]
    return [int(x) for x in content_tokens]


@lru_cache(maxsize=8)
def _cached_frontend(
    silent_token: int | None,
    separator_aware: bool,
    tail_open_units: int,
    emit_silence_runs: bool,
    debounce_min_run_frames: int,
) -> RhythmUnitFrontend:
    return RhythmUnitFrontend(
        silent_token=silent_token,
        separator_aware=separator_aware,
        tail_open_units=tail_open_units,
        emit_silence_runs=emit_silence_runs,
        debounce_min_run_frames=debounce_min_run_frames,
    )


def build_source_phrase_cache(
    *,
    dur_anchor_src,
    sep_hint,
    open_run_mask,
    sealed_mask,
    boundary_confidence,
    phrase_boundary_threshold: float = 0.55,
) -> dict[str, np.ndarray]:
    dur_anchor_src = torch.tensor(np.asarray(dur_anchor_src), dtype=torch.float32).unsqueeze(0)
    sep_hint = torch.tensor(np.asarray(sep_hint), dtype=torch.long).unsqueeze(0)
    open_run_mask = torch.tensor(np.asarray(open_run_mask), dtype=torch.long).unsqueeze(0)
    sealed_mask = torch.tensor(np.asarray(sealed_mask), dtype=torch.float32).unsqueeze(0)
    boundary_confidence = torch.tensor(np.asarray(boundary_confidence), dtype=torch.float32).unsqueeze(0)
    unit_mask = dur_anchor_src.gt(0).float()
    source_boundary_cue = build_source_boundary_cue(
        dur_anchor_src=dur_anchor_src,
        unit_mask=unit_mask,
        sep_hint=sep_hint,
        open_run_mask=open_run_mask,
        sealed_mask=sealed_mask,
        boundary_confidence=boundary_confidence,
    )[0]
    visible = int(unit_mask[0].sum().item())
    phrase_group_index = torch.zeros_like(source_boundary_cue, dtype=torch.long)
    phrase_group_pos = torch.zeros_like(source_boundary_cue)
    phrase_final_mask = torch.zeros_like(source_boundary_cue)
    if visible > 0:
        break_mask = (source_boundary_cue[:visible] >= float(phrase_boundary_threshold)).float()
        if sep_hint.size(1) >= visible:
            break_mask = torch.maximum(break_mask, sep_hint[0, :visible].float())
        phrase_starts = [0]
        for idx in range(max(visible - 1, 0)):
            if float(break_mask[idx].item()) > 0:
                phrase_starts.append(idx + 1)
                phrase_final_mask[idx] = 1.0
        phrase_final_mask[visible - 1] = 1.0
        phrase_starts = sorted(set(int(x) for x in phrase_starts if 0 <= int(x) < visible))
        for group_id, start in enumerate(phrase_starts):
            end = phrase_starts[group_id + 1] if group_id + 1 < len(phrase_starts) else visible
            length = max(1, end - start)
            phrase_group_index[start:end] = group_id
            if length == 1:
                phrase_group_pos[start] = 1.0
            else:
                phrase_group_pos[start:end] = torch.linspace(0.0, 1.0, steps=length)
    return {
        "source_boundary_cue": source_boundary_cue.cpu().numpy().astype(np.float32),
        "phrase_group_index": phrase_group_index.cpu().numpy().astype(np.int64),
        "phrase_group_pos": phrase_group_pos.cpu().numpy().astype(np.float32),
        "phrase_final_mask": phrase_final_mask.cpu().numpy().astype(np.float32),
    }


def build_source_rhythm_cache_v3(
    content_tokens,
    *,
    silent_token: int | None = None,
    separator_aware: bool = True,
    tail_open_units: int = 1,
    emit_silence_runs: bool = False,
    debounce_min_run_frames: int = 1,
    phrase_boundary_threshold: float = 0.55,
) -> dict[str, np.ndarray]:
    frontend = _cached_frontend(
        silent_token=silent_token,
        separator_aware=separator_aware,
        tail_open_units=tail_open_units,
        emit_silence_runs=emit_silence_runs,
        debounce_min_run_frames=debounce_min_run_frames,
    )
    token_tensor = torch.tensor(
        [_as_token_list(content_tokens)],
        dtype=torch.long,
    )
    batch = frontend.from_content_tensor(
        token_tensor,
        mark_last_open=False,
    )
    source_cache = {
        "content_units": batch.content_units[0].cpu().numpy().astype(np.int64),
        "dur_anchor_src": batch.dur_anchor_src[0].cpu().numpy().astype(np.int64),
        "source_silence_mask": batch.silence_mask[0].cpu().numpy().astype(np.float32),
        "open_run_mask": batch.open_run_mask[0].cpu().numpy().astype(np.int64),
        "sealed_mask": batch.sealed_mask[0].cpu().numpy().astype(np.int64),
        "sep_hint": batch.sep_hint[0].cpu().numpy().astype(np.int64),
        "boundary_confidence": batch.boundary_confidence[0].cpu().numpy().astype(np.float32),
        "source_run_stability": batch.run_stability[0].cpu().numpy().astype(np.float32),
    }
    source_cache.update(
        build_source_phrase_cache(
            dur_anchor_src=source_cache["dur_anchor_src"],
            sep_hint=source_cache["sep_hint"],
            open_run_mask=source_cache["open_run_mask"],
            sealed_mask=source_cache["sealed_mask"],
            boundary_confidence=source_cache["boundary_confidence"],
            phrase_boundary_threshold=phrase_boundary_threshold,
        )
    )
    return source_cache


build_source_rhythm_cache = build_source_rhythm_cache_v3


__all__ = [
    "build_source_phrase_cache",
    "build_source_rhythm_cache",
    "build_source_rhythm_cache_v3",
    "estimate_boundary_confidence",
    "estimate_run_stability",
]
