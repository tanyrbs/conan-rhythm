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
    alignment_kind = str(projection.get("alignment_kind", "discrete")).strip() or "discrete"
    alignment_source = str(projection.get("alignment_source", "")).strip()
    alignment_version = str(projection.get("alignment_version", "")).strip()
    payload = {
        "unit_alignment_kind_tgt": np.asarray([alignment_kind], dtype=object),
        "unit_alignment_source_tgt": np.asarray([alignment_source], dtype=object),
        "unit_alignment_version_tgt": np.asarray([alignment_version], dtype=object),
    }
    optional_vector_sidecars = {
        "run_margin": "unit_alignment_run_margin_tgt",
        "run_mean_cost": "unit_alignment_run_mean_cost_tgt",
        "run_type_agree": "unit_alignment_run_type_agree_tgt",
        "run_occ_weighted": "unit_alignment_run_occ_weighted_tgt",
        "run_occ_expected": "unit_alignment_run_occ_expected_tgt",
        "run_entropy": "unit_alignment_run_entropy_tgt",
        "run_posterior_mass_on_path": "unit_alignment_run_posterior_mass_on_path_tgt",
        "posterior_band_left": "unit_alignment_posterior_band_left_debug",
        "posterior_band_right": "unit_alignment_posterior_band_right_debug",
        "posterior_values": "unit_alignment_posterior_values_debug",
    }
    for projection_key, payload_key in optional_vector_sidecars.items():
        value = projection.get(projection_key)
        if value is None:
            continue
        dtype = np.int64 if "band_" in projection_key else np.float32
        payload[payload_key] = np.asarray(value, dtype=dtype)
    payload.update({
        "unit_duration_tgt": np.asarray(projection["projected"], dtype=np.float32),
        "unit_duration_proj_raw_tgt": np.asarray(projection["projected"], dtype=np.float32),
        "unit_confidence_local_tgt": np.asarray(projection["confidence_local"], dtype=np.float32),
        "unit_confidence_coarse_tgt": np.asarray(projection["confidence_coarse"], dtype=np.float32),
        "unit_confidence_tgt": np.asarray(projection["confidence_coarse"], dtype=np.float32),
        "unit_alignment_coverage_tgt": np.asarray(projection["coverage"], dtype=np.float32),
        "unit_alignment_match_tgt": np.asarray(projection["match_rate"], dtype=np.float32),
        "unit_alignment_cost_tgt": np.asarray(projection["mean_cost"], dtype=np.float32),
        "unit_alignment_unmatched_speech_ratio_tgt": np.asarray(
            [projection["unmatched_speech_ratio"]],
            dtype=np.float32,
        ),
        "unit_alignment_mean_local_confidence_speech_tgt": np.asarray(
            [projection["mean_local_confidence_speech"]],
            dtype=np.float32,
        ),
        "unit_alignment_mean_coarse_confidence_speech_tgt": np.asarray(
            [projection["mean_coarse_confidence_speech"]],
            dtype=np.float32,
        ),
        "unit_alignment_mode_id_tgt": np.asarray(
            [1 if alignment_kind.lower().startswith("continuous") else 0],
            dtype=np.int64,
        ),
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
    })
    return payload


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
    metadata = dict(record.metadata or {})
    alignment_kind = payload.get("unit_alignment_kind_tgt")
    alignment_source = payload.get("unit_alignment_source_tgt")
    alignment_version = payload.get("unit_alignment_version_tgt")
    if alignment_kind is not None:
        metadata.setdefault("alignment_kind", str(np.asarray(alignment_kind, dtype=object).reshape(-1)[0]))
    if alignment_source is not None:
        metadata.setdefault("alignment_source", str(np.asarray(alignment_source, dtype=object).reshape(-1)[0]))
    if alignment_version is not None:
        metadata.setdefault("alignment_version", str(np.asarray(alignment_version, dtype=object).reshape(-1)[0]))
    return replace(record, metadata=metadata, **payload)


__all__ = [
    "attach_projection_debug",
    "build_projection_debug_payload",
]
