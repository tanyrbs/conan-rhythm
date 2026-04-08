from __future__ import annotations

import numpy as np

RHYTHM_CACHE_VERSION = 5
RHYTHM_UNIT_HOP_MS = 20
RHYTHM_TRACE_HOP_MS = 80
RHYTHM_REFERENCE_MODE_STATIC_REF_FULL = 0
RHYTHM_GUIDANCE_SURFACE_NAME = "ref_guidance_v2"
RHYTHM_TEACHER_SURFACE_ALGORITHMIC_NAME = "offline_teacher_surface_v1"
RHYTHM_TEACHER_SURFACE_LEARNED_OFFLINE_NAME = "offline_teacher_surface_learned_offline_v1"
RHYTHM_TEACHER_SURFACE_NAME = RHYTHM_TEACHER_SURFACE_ALGORITHMIC_NAME
RHYTHM_TEACHER_TARGET_SOURCE_ALGORITHMIC = 0
RHYTHM_TEACHER_TARGET_SOURCE_LEARNED_OFFLINE = 1
RHYTHM_RETIMED_SOURCE_GUIDANCE = 0
RHYTHM_RETIMED_SOURCE_TEACHER = 1
RHYTHM_CACHE_COMPATIBLE_VERSIONS = {
    RHYTHM_CACHE_VERSION: (4,),
}

_BLANK_ALIAS_KEYS = {
    "rhythm_pause_exec_tgt": "rhythm_blank_exec_tgt",
    "rhythm_pause_budget_tgt": "rhythm_blank_budget_tgt",
    "rhythm_guidance_pause_tgt": "rhythm_guidance_blank_tgt",
    "rhythm_teacher_pause_exec_tgt": "rhythm_teacher_blank_exec_tgt",
    "rhythm_teacher_pause_budget_tgt": "rhythm_teacher_blank_budget_tgt",
}


def normalize_teacher_target_source(value) -> str:
    source = str(value or "algorithmic").strip().lower()
    aliases = {
        "algo": "algorithmic",
        "heuristic": "algorithmic",
        "legacy": "algorithmic",
        "rule": "algorithmic",
        "rules": "algorithmic",
        "teacher": "learned_offline",
        "offline": "learned_offline",
        "offline_teacher": "learned_offline",
        "learned": "learned_offline",
        "learned-offline": "learned_offline",
        "cache": "learned_offline",
        "cached": "learned_offline",
        "cached_teacher": "learned_offline",
    }
    normalized = aliases.get(source, source)
    if normalized not in {"algorithmic", "learned_offline"}:
        raise ValueError(f"Unsupported rhythm_teacher_target_source: {value}")
    return normalized


def resolve_teacher_surface_name(teacher_target_source) -> str:
    normalized = normalize_teacher_target_source(teacher_target_source)
    if normalized == "learned_offline":
        return RHYTHM_TEACHER_SURFACE_LEARNED_OFFLINE_NAME
    return RHYTHM_TEACHER_SURFACE_ALGORITHMIC_NAME


def resolve_teacher_target_source_id(teacher_target_source) -> int:
    normalized = normalize_teacher_target_source(teacher_target_source)
    if normalized == "learned_offline":
        return RHYTHM_TEACHER_TARGET_SOURCE_LEARNED_OFFLINE
    return RHYTHM_TEACHER_TARGET_SOURCE_ALGORITHMIC


def resolve_teacher_target_source_from_id(source_id) -> str:
    value = int(source_id)
    if value == RHYTHM_TEACHER_TARGET_SOURCE_LEARNED_OFFLINE:
        return "learned_offline"
    if value == RHYTHM_TEACHER_TARGET_SOURCE_ALGORITHMIC:
        return "algorithmic"
    raise ValueError(f"Unsupported rhythm teacher target source id: {source_id}")


def infer_teacher_target_source_from_surface_name(surface_name) -> str | None:
    if surface_name in {None, ""}:
        return None
    surface = str(surface_name).strip()
    if surface == RHYTHM_TEACHER_SURFACE_LEARNED_OFFLINE_NAME:
        return "learned_offline"
    if surface == RHYTHM_TEACHER_SURFACE_ALGORITHMIC_NAME:
        return "algorithmic"
    return None


def infer_teacher_target_source_id_from_surface_name(surface_name) -> int | None:
    source = infer_teacher_target_source_from_surface_name(surface_name)
    if source is None:
        return None
    return resolve_teacher_target_source_id(source)


def resolve_retimed_target_surface_name(
    source_id,
    *,
    teacher_surface_name=None,
    teacher_target_source=None,
) -> str:
    source_value = int(source_id)
    if source_value == RHYTHM_RETIMED_SOURCE_GUIDANCE:
        return RHYTHM_GUIDANCE_SURFACE_NAME
    if source_value == RHYTHM_RETIMED_SOURCE_TEACHER:
        if teacher_surface_name not in {None, ""}:
            return str(teacher_surface_name)
        if teacher_target_source is None:
            teacher_target_source = "algorithmic"
        return resolve_teacher_surface_name(teacher_target_source)
    raise ValueError(f"Unsupported rhythm retimed target source id: {source_id}")


def infer_retimed_target_source_id(
    surface_name,
    *,
    teacher_surface_name=None,
) -> int | None:
    if surface_name in {None, ""}:
        return None
    surface = str(surface_name).strip()
    if surface == RHYTHM_GUIDANCE_SURFACE_NAME:
        return RHYTHM_RETIMED_SOURCE_GUIDANCE
    if surface in {
        RHYTHM_TEACHER_SURFACE_ALGORITHMIC_NAME,
        RHYTHM_TEACHER_SURFACE_LEARNED_OFFLINE_NAME,
    }:
        return RHYTHM_RETIMED_SOURCE_TEACHER
    if teacher_surface_name not in {None, ""} and surface == str(teacher_surface_name).strip():
        return RHYTHM_RETIMED_SOURCE_TEACHER
    return None


def compatible_rhythm_cache_versions(expected_version) -> tuple[int, ...]:
    expected = int(expected_version)
    compatible = [expected]
    for version in RHYTHM_CACHE_COMPATIBLE_VERSIONS.get(expected, ()):
        version = int(version)
        if version not in compatible:
            compatible.append(version)
    return tuple(compatible)


def is_rhythm_cache_version_compatible(found_version, expected_version) -> bool:
    found = int(found_version)
    return found in compatible_rhythm_cache_versions(expected_version)


def materialize_rhythm_cache_compat_fields(item: dict | None) -> dict | None:
    if item is None:
        return None
    adapted = dict(item)
    teacher_surface_name = None
    if "rhythm_teacher_surface_name" in adapted:
        teacher_surface_name = str(np.asarray(adapted["rhythm_teacher_surface_name"]).reshape(-1)[0])
    if "rhythm_teacher_target_source_id" in adapted:
        teacher_source_id = int(np.asarray(adapted["rhythm_teacher_target_source_id"]).reshape(-1)[0])
    else:
        teacher_source_id = infer_teacher_target_source_id_from_surface_name(teacher_surface_name)
        if teacher_source_id is not None:
            adapted["rhythm_teacher_target_source_id"] = np.asarray([teacher_source_id], dtype=np.int64)
    if "rhythm_teacher_surface_name" not in adapted and teacher_source_id is not None:
        adapted["rhythm_teacher_surface_name"] = np.asarray(
            [resolve_teacher_surface_name(resolve_teacher_target_source_from_id(teacher_source_id))],
            dtype=np.str_,
        )
        teacher_surface_name = str(np.asarray(adapted["rhythm_teacher_surface_name"]).reshape(-1)[0])
    if "rhythm_retimed_target_source_id" in adapted:
        retimed_source_id = int(np.asarray(adapted["rhythm_retimed_target_source_id"]).reshape(-1)[0])
    else:
        retimed_surface_name = None
        if "rhythm_retimed_target_surface_name" in adapted:
            retimed_surface_name = str(np.asarray(adapted["rhythm_retimed_target_surface_name"]).reshape(-1)[0])
        retimed_source_id = infer_retimed_target_source_id(
            retimed_surface_name,
            teacher_surface_name=teacher_surface_name,
        )
        if retimed_source_id is not None:
            adapted["rhythm_retimed_target_source_id"] = np.asarray([retimed_source_id], dtype=np.int64)
    if "rhythm_retimed_target_surface_name" not in adapted and retimed_source_id is not None:
        teacher_target_source = None
        if teacher_source_id is not None:
            teacher_target_source = resolve_teacher_target_source_from_id(teacher_source_id)
        adapted["rhythm_retimed_target_surface_name"] = np.asarray(
            [
                resolve_retimed_target_surface_name(
                    retimed_source_id,
                    teacher_surface_name=teacher_surface_name,
                    teacher_target_source=teacher_target_source,
                )
            ],
            dtype=np.str_,
        )
    return with_blank_aliases(adapted)


def with_blank_aliases(bundle: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = dict(bundle)
    for pause_key, blank_key in _BLANK_ALIAS_KEYS.items():
        if pause_key in out and blank_key not in out:
            out[blank_key] = np.asarray(out[pause_key]).copy()
        if blank_key in out and pause_key not in out:
            out[pause_key] = np.asarray(out[blank_key]).copy()
    return out


def build_cache_metadata(
    *,
    trace_bins: int,
    trace_horizon: float,
    slow_topk: int,
    selector_cell_size: int,
    source_phrase_threshold: float,
    target_confidence: float,
    guidance_confidence: float,
    retimed_target_source: str | None = None,
    retimed_target_confidence: float | None = None,
    teacher_confidence: float | None = None,
    teacher_target_source: str | None = None,
    teacher_surface_name: str | None = None,
) -> dict[str, np.ndarray]:
    meta = {
        "rhythm_cache_version": np.asarray([RHYTHM_CACHE_VERSION], dtype=np.int64),
        "rhythm_unit_hop_ms": np.asarray([RHYTHM_UNIT_HOP_MS], dtype=np.int64),
        "rhythm_trace_hop_ms": np.asarray([RHYTHM_TRACE_HOP_MS], dtype=np.int64),
        "rhythm_trace_bins": np.asarray([int(trace_bins)], dtype=np.int64),
        "rhythm_trace_horizon": np.asarray([float(trace_horizon)], dtype=np.float32),
        "rhythm_slow_topk": np.asarray([int(slow_topk)], dtype=np.int64),
        "rhythm_selector_cell_size": np.asarray([int(selector_cell_size)], dtype=np.int64),
        "rhythm_source_phrase_threshold": np.asarray([float(source_phrase_threshold)], dtype=np.float32),
        "rhythm_reference_mode_id": np.asarray([RHYTHM_REFERENCE_MODE_STATIC_REF_FULL], dtype=np.int64),
        "rhythm_target_confidence": np.asarray([float(target_confidence)], dtype=np.float32),
        "rhythm_guidance_confidence": np.asarray([float(guidance_confidence)], dtype=np.float32),
        "rhythm_guidance_surface_name": np.asarray([RHYTHM_GUIDANCE_SURFACE_NAME], dtype=np.str_),
    }
    if teacher_target_source is not None or teacher_surface_name is not None or teacher_confidence is not None:
        teacher_source = normalize_teacher_target_source(teacher_target_source or "algorithmic")
        teacher_surface = str(teacher_surface_name or resolve_teacher_surface_name(teacher_source))
        meta["rhythm_teacher_target_source_id"] = np.asarray(
            [resolve_teacher_target_source_id(teacher_source)], dtype=np.int64
        )
        meta["rhythm_teacher_surface_name"] = np.asarray([teacher_surface], dtype=np.str_)
    if teacher_confidence is not None:
        meta["rhythm_teacher_confidence"] = np.asarray([float(teacher_confidence)], dtype=np.float32)
    if retimed_target_source is not None:
        target_source = str(retimed_target_source).strip().lower()
        source_id = RHYTHM_RETIMED_SOURCE_TEACHER if target_source == "teacher" else RHYTHM_RETIMED_SOURCE_GUIDANCE
        source_name = (
            str(teacher_surface_name or resolve_teacher_surface_name(teacher_target_source or "algorithmic"))
            if source_id == RHYTHM_RETIMED_SOURCE_TEACHER
            else RHYTHM_GUIDANCE_SURFACE_NAME
        )
        meta["rhythm_retimed_target_source_id"] = np.asarray([source_id], dtype=np.int64)
        meta["rhythm_retimed_target_surface_name"] = np.asarray([source_name], dtype=np.str_)
        if retimed_target_confidence is not None:
            meta["rhythm_retimed_target_confidence"] = np.asarray([float(retimed_target_confidence)], dtype=np.float32)
    return meta


__all__ = [
    "RHYTHM_CACHE_VERSION",
    "RHYTHM_GUIDANCE_SURFACE_NAME",
    "RHYTHM_REFERENCE_MODE_STATIC_REF_FULL",
    "RHYTHM_RETIMED_SOURCE_GUIDANCE",
    "RHYTHM_RETIMED_SOURCE_TEACHER",
    "RHYTHM_TEACHER_SURFACE_ALGORITHMIC_NAME",
    "RHYTHM_TEACHER_SURFACE_LEARNED_OFFLINE_NAME",
    "RHYTHM_TEACHER_SURFACE_NAME",
    "RHYTHM_TEACHER_TARGET_SOURCE_ALGORITHMIC",
    "RHYTHM_TEACHER_TARGET_SOURCE_LEARNED_OFFLINE",
    "RHYTHM_TRACE_HOP_MS",
    "RHYTHM_UNIT_HOP_MS",
    "build_cache_metadata",
    "compatible_rhythm_cache_versions",
    "infer_retimed_target_source_id",
    "infer_teacher_target_source_from_surface_name",
    "infer_teacher_target_source_id_from_surface_name",
    "is_rhythm_cache_version_compatible",
    "materialize_rhythm_cache_compat_fields",
    "normalize_teacher_target_source",
    "resolve_retimed_target_surface_name",
    "resolve_teacher_surface_name",
    "resolve_teacher_target_source_from_id",
    "resolve_teacher_target_source_id",
    "with_blank_aliases",
]
