from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from tasks.Conan.rhythm.duration_v3.alignment_projection import (
    as_float32_1d,
    as_int64_1d,
    project_target_runs_onto_source,
    resolve_run_silence_mask,
)

from .core import RhythmV3DebugRecord


def build_projection_debug_payload(
    *,
    source_units,
    source_durations,
    source_silence_mask=None,
    target_units=None,
    target_durations=None,
    target_valid_mask=None,
    target_speech_mask=None,
    use_continuous_alignment: bool = False,
) -> dict[str, np.ndarray]:
    source_units = as_int64_1d(source_units)
    source_durations = as_float32_1d(source_durations)
    source_silence_mask = resolve_run_silence_mask(
        size=source_durations.shape[0],
        silence_mask=source_silence_mask,
    )
    target_units = as_int64_1d(target_units)
    target_durations = as_float32_1d(target_durations)
    target_valid_mask = as_float32_1d(target_valid_mask)
    target_speech_mask = as_float32_1d(target_speech_mask)

    projection = project_target_runs_onto_source(
        source_units=source_units,
        source_durations=source_durations,
        source_silence_mask=source_silence_mask,
        target_units=target_units,
        target_durations=target_durations,
        target_valid_mask=target_valid_mask,
        target_speech_mask=target_speech_mask,
        use_continuous_alignment=use_continuous_alignment,
    )
    return {
        "unit_duration_tgt": np.asarray(projection["projected"], dtype=np.float32),
        "unit_confidence_local_tgt": np.asarray(projection["confidence_local"], dtype=np.float32),
        "unit_confidence_coarse_tgt": np.asarray(projection["confidence_coarse"], dtype=np.float32),
        "unit_confidence_tgt": np.asarray(projection["confidence_coarse"], dtype=np.float32),
        "unit_alignment_coverage_tgt": np.asarray(projection["coverage"], dtype=np.float32),
        "unit_alignment_match_tgt": np.asarray(projection["match_rate"], dtype=np.float32),
        "unit_alignment_cost_tgt": np.asarray(projection["mean_cost"], dtype=np.float32),
        "paired_target_content_units_debug": np.asarray(target_units, dtype=np.int64),
        "paired_target_duration_obs_debug": np.asarray(target_durations, dtype=np.float32),
        "paired_target_valid_mask_debug": np.asarray(target_valid_mask, dtype=np.float32),
        "paired_target_speech_mask_debug": np.asarray(target_speech_mask, dtype=np.float32),
        "unit_alignment_assigned_source_debug": np.asarray(projection["assigned_source"], dtype=np.int64),
        "unit_alignment_assigned_cost_debug": np.asarray(projection["assigned_cost"], dtype=np.float32),
        "unit_alignment_source_valid_run_index_debug": np.asarray(
            projection["source_valid_run_index"],
            dtype=np.int64,
        ),
    }


def attach_projection_debug(
    record: RhythmV3DebugRecord,
    *,
    target_units,
    target_durations,
    target_valid_mask,
    target_speech_mask,
    use_continuous_alignment: bool = False,
) -> RhythmV3DebugRecord:
    if record.source_content_units is None or record.source_duration_obs is None:
        raise ValueError("attach_projection_debug requires source_content_units and source_duration_obs in the record.")
    payload = build_projection_debug_payload(
        source_units=record.source_content_units,
        source_durations=record.source_duration_obs,
        source_silence_mask=record.source_silence_mask,
        target_units=target_units,
        target_durations=target_durations,
        target_valid_mask=target_valid_mask,
        target_speech_mask=target_speech_mask,
        use_continuous_alignment=use_continuous_alignment,
    )
    return replace(record, **payload)


__all__ = [
    "attach_projection_debug",
    "build_projection_debug_payload",
]
