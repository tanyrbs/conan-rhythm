from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .unit_frontend import RhythmUnitFrontend
from .unitizer import estimate_boundary_confidence, estimate_run_stability


DURATION_V3_SOURCE_CACHE_VERSION = 4
DURATION_V3_CACHE_META_KEY = "rhythm_v3_cache_meta"
UNIT_LOG_PRIOR_META_KEY = "rhythm_v3_unit_prior_meta"


def build_duration_v3_cache_meta(
    *,
    silent_token: int | None,
    separator_aware: bool,
    tail_open_units: int,
    emit_silence_runs: bool,
    debounce_min_run_frames: int,
    phrase_boundary_threshold: float,
) -> dict[str, int | float | bool | None]:
    return {
        "cache_version": int(DURATION_V3_SOURCE_CACHE_VERSION),
        "silent_token": (None if silent_token is None else int(silent_token)),
        "separator_aware": bool(separator_aware),
        "tail_open_units": int(tail_open_units),
        "emit_silence_runs": bool(emit_silence_runs),
        "debounce_min_run_frames": int(debounce_min_run_frames),
        "phrase_boundary_threshold": float(phrase_boundary_threshold),
    }


def _normalize_duration_v3_cache_meta(
    cache_meta: Mapping[str, object] | None,
) -> dict[str, int | float | bool | None] | None:
    if not isinstance(cache_meta, Mapping):
        return None
    if "cache_version" not in cache_meta:
        return None
    silent_token = cache_meta.get("silent_token")
    return {
        "cache_version": int(cache_meta.get("cache_version", 0)),
        "silent_token": (None if silent_token is None else int(silent_token)),
        "separator_aware": bool(cache_meta.get("separator_aware", False)),
        "tail_open_units": int(cache_meta.get("tail_open_units", 0)),
        "emit_silence_runs": bool(cache_meta.get("emit_silence_runs", False)),
        "debounce_min_run_frames": int(cache_meta.get("debounce_min_run_frames", 0)),
        "phrase_boundary_threshold": float(cache_meta.get("phrase_boundary_threshold", 0.0)),
    }


def resolve_duration_v3_cache_meta(source) -> dict[str, int | float | bool | None] | None:
    if isinstance(source, Mapping):
        direct = _normalize_duration_v3_cache_meta(source.get(DURATION_V3_CACHE_META_KEY))
        if direct is not None:
            return direct
        return _normalize_duration_v3_cache_meta(source)
    return _normalize_duration_v3_cache_meta(getattr(source, DURATION_V3_CACHE_META_KEY, None))


def duration_v3_cache_meta_signature(cache_meta: Mapping[str, object] | None) -> str:
    normalized = resolve_duration_v3_cache_meta(cache_meta)
    if normalized is None:
        return "missing"
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def build_duration_v3_frontend_signature(
    cache_meta: Mapping[str, object] | None,
    *,
    g_variant: str,
    drop_edge_runs_for_g: int,
    unit_prior_path: str | None,
    summary_pool_speech_only: bool,
    emit_silence_runs: bool | None = None,
) -> str:
    normalized = resolve_duration_v3_cache_meta(cache_meta)
    unit_prior_meta = None
    if unit_prior_path:
        resolved_path = Path(unit_prior_path).resolve()
        bundle = load_unit_log_prior_bundle(str(resolved_path))
        stat = resolved_path.stat()
        unit_prior_meta = {
            "path": str(resolved_path),
            "mtime_ns": int(stat.st_mtime_ns),
            "size": int(stat.st_size),
            "source": bundle.get("unit_prior_source"),
            "version": bundle.get("unit_prior_version"),
            "frontend_meta_signature": bundle.get("unit_prior_frontend_signature"),
            "min_count": bundle.get("unit_prior_min_count"),
            "default_policy": bundle.get("unit_prior_default_policy"),
            "global_backoff": bundle.get("unit_prior_global_backoff"),
            "sha1": bundle.get("unit_prior_sha1"),
            "emit_silence_runs": bundle.get("unit_prior_emit_silence_runs"),
            "debounce_min_run_frames": bundle.get("unit_prior_debounce_min_run_frames"),
            "silent_token": bundle.get("unit_prior_silent_token"),
            "separator_aware": bundle.get("unit_prior_separator_aware"),
            "tail_open_units": bundle.get("unit_prior_tail_open_units"),
            "phrase_boundary_threshold": bundle.get("unit_prior_phrase_boundary_threshold"),
            "filter_exclude_open_runs": bundle.get("unit_prior_filter_exclude_open_runs"),
            "filter_only_sealed_runs": bundle.get("unit_prior_filter_only_sealed_runs"),
            "filter_drop_edge_runs": bundle.get("unit_prior_filter_drop_edge_runs"),
            "filter_min_run_stability": bundle.get("unit_prior_filter_min_run_stability"),
            "special_token_policy": bundle.get("unit_prior_special_token_policy"),
        }
    signature = {
        "cache_meta": normalized,
        "g_variant": str(g_variant or "").strip().lower(),
        "drop_edge_runs_for_g": int(drop_edge_runs_for_g),
        "unit_prior_path": (
            str(Path(unit_prior_path).resolve()) if unit_prior_path else None
        ),
        "unit_prior_bundle": unit_prior_meta,
        "summary_pool_speech_only": bool(summary_pool_speech_only),
        "emit_silence_runs": (
            bool(emit_silence_runs)
            if emit_silence_runs is not None
            else (None if normalized is None else bool(normalized.get("emit_silence_runs", False)))
        ),
    }
    return json.dumps(signature, sort_keys=True, separators=(",", ":"))


def attach_duration_v3_cache_meta(
    cache: dict,
    *,
    silent_token: int | None,
    separator_aware: bool,
    tail_open_units: int,
    emit_silence_runs: bool,
    debounce_min_run_frames: int,
    phrase_boundary_threshold: float,
) -> dict:
    cache[DURATION_V3_CACHE_META_KEY] = build_duration_v3_cache_meta(
        silent_token=silent_token,
        separator_aware=separator_aware,
        tail_open_units=tail_open_units,
        emit_silence_runs=emit_silence_runs,
        debounce_min_run_frames=debounce_min_run_frames,
        phrase_boundary_threshold=phrase_boundary_threshold,
    )
    return cache


def assert_duration_v3_cache_meta_compatible(
    cache_meta,
    *,
    silent_token: int | None,
    separator_aware: bool,
    tail_open_units: int,
    emit_silence_runs: bool,
    debounce_min_run_frames: int,
    phrase_boundary_threshold: float,
) -> dict[str, int | float | bool | None]:
    actual = resolve_duration_v3_cache_meta(cache_meta)
    if actual is None:
        raise ValueError("Duration-v3 source cache is missing rhythm_v3_cache_meta.")
    expected = build_duration_v3_cache_meta(
        silent_token=silent_token,
        separator_aware=separator_aware,
        tail_open_units=tail_open_units,
        emit_silence_runs=emit_silence_runs,
        debounce_min_run_frames=debounce_min_run_frames,
        phrase_boundary_threshold=phrase_boundary_threshold,
    )
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if isinstance(expected_value, float):
            if abs(float(actual_value) - float(expected_value)) > 1.0e-6:
                mismatches.append(f"{key}: got {actual_value!r}, expected {expected_value!r}")
        elif actual_value != expected_value:
            mismatches.append(f"{key}: got {actual_value!r}, expected {expected_value!r}")
    if mismatches:
        raise ValueError(
            "Duration-v3 source cache meta mismatch: " + "; ".join(mismatches)
        )
    return actual


def _as_token_list(content_tokens) -> list[int]:
    if isinstance(content_tokens, str):
        return [int(float(x)) for x in content_tokens.split() if str(x).strip() != ""]
    if isinstance(content_tokens, np.ndarray):
        return [int(x) for x in content_tokens.tolist()]
    if torch.is_tensor(content_tokens):
        return [int(x) for x in content_tokens.detach().cpu().reshape(-1).tolist()]
    return [int(x) for x in content_tokens]


def _finite_max_or_nan(values) -> float:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size <= 0:
        return float("nan")
    return float(finite.max())


def _coerce_single_mel_tensor(mel) -> torch.Tensor | None:
    if mel is None:
        return None
    if torch.is_tensor(mel):
        mel_tensor = mel.detach().to(device="cpu", dtype=torch.float32)
    else:
        mel_tensor = torch.as_tensor(np.asarray(mel), dtype=torch.float32)
    if mel_tensor.dim() == 3:
        if int(mel_tensor.size(0)) != 1:
            raise ValueError(f"mel must be rank-2 or batch-size-1 rank-3, got {tuple(mel_tensor.shape)}")
        mel_tensor = mel_tensor[0]
    if mel_tensor.dim() != 2:
        raise ValueError(f"mel must be rank-2 [T,C] or [C,T], got {tuple(mel_tensor.shape)}")
    if mel_tensor.size(-1) == 80:
        return mel_tensor.contiguous()
    if mel_tensor.size(0) == 80:
        return mel_tensor.transpose(0, 1).contiguous()
    if mel_tensor.size(0) < mel_tensor.size(1):
        return mel_tensor.transpose(0, 1).contiguous()
    return mel_tensor.contiguous()


def _resize_binary_mask(mask: torch.Tensor, *, target_length: int) -> torch.Tensor:
    target_length = int(max(0, target_length))
    if target_length <= 0:
        return mask.new_zeros((0,), dtype=torch.float32)
    mask = mask.reshape(-1).float()
    if int(mask.numel()) == target_length:
        return mask
    if int(mask.numel()) <= 0:
        return mask.new_zeros((target_length,), dtype=torch.float32)
    resized = F.interpolate(
        mask.view(1, 1, -1),
        size=target_length,
        mode="nearest",
    ).view(-1)
    if int(resized.numel()) > target_length:
        resized = resized[:target_length]
    return resized.float()


def infer_frame_silence_mask_from_mel(
    mel,
    *,
    target_length: int | None = None,
    pause_energy_threshold_std: float = -0.5,
    pause_delta_quantile: float = 0.35,
) -> torch.Tensor:
    mel_tensor = _coerce_single_mel_tensor(mel)
    if mel_tensor is None or int(mel_tensor.size(0)) <= 0:
        target = 0 if target_length is None else int(max(0, target_length))
        return torch.zeros((target,), dtype=torch.float32)
    energy = mel_tensor.mean(dim=-1)
    energy_mean = energy.mean()
    energy_std = energy.std(unbiased=False).clamp_min(1.0e-6)
    energy_z = (energy - energy_mean) / energy_std
    delta = torch.zeros_like(energy)
    delta[1:] = (energy[1:] - energy[:-1]).abs()
    delta_threshold = torch.quantile(
        delta,
        float(max(0.01, min(0.95, pause_delta_quantile))),
    )
    pause_mask = (
        (energy_z <= float(pause_energy_threshold_std))
        & (delta <= delta_threshold)
    ).float()
    if int(pause_mask.numel()) > 1:
        kernel = min(3, int(pause_mask.numel()))
        if kernel % 2 == 0:
            kernel = max(1, kernel - 1)
        smoothed = F.max_pool1d(
            pause_mask.view(1, 1, -1),
            kernel_size=kernel,
            stride=1,
            padding=kernel // 2,
        ).view(-1)
        if int(smoothed.numel()) > int(pause_mask.numel()):
            smoothed = smoothed[: int(pause_mask.numel())]
        pause_mask = (smoothed >= 0.5).float()
    if target_length is not None:
        pause_mask = _resize_binary_mask(pause_mask, target_length=int(target_length))
    return pause_mask.float()


def _resolve_frame_silence_mask(
    content_tokens,
    *,
    silent_token: int | None,
    frame_silence_mask=None,
    mel=None,
) -> torch.Tensor | None:
    token_list = _as_token_list(content_tokens)
    target_length = int(len(token_list))
    token_mask = None
    if silent_token is not None and target_length > 0:
        token_mask = torch.tensor(
            [1.0 if int(token) == int(silent_token) else 0.0 for token in token_list],
            dtype=torch.float32,
        )
    resolved_mask = None
    if frame_silence_mask is not None:
        if torch.is_tensor(frame_silence_mask):
            resolved_mask = frame_silence_mask.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        else:
            resolved_mask = torch.as_tensor(np.asarray(frame_silence_mask), dtype=torch.float32).reshape(-1)
        resolved_mask = _resize_binary_mask(resolved_mask, target_length=target_length)
    elif mel is not None:
        resolved_mask = infer_frame_silence_mask_from_mel(mel, target_length=target_length)
    if resolved_mask is None:
        return token_mask
    resolved_mask = resolved_mask.clamp(0.0, 1.0)
    if token_mask is not None:
        resolved_mask = torch.maximum(resolved_mask, token_mask)
    return resolved_mask


def collect_duration_v3_frontend_diagnostics(
    source,
    *,
    silent_token: int | None = None,
) -> dict[str, float | int]:
    source_map = source if isinstance(source, Mapping) else {}
    dur_anchor_src = np.asarray(source_map.get("dur_anchor_src", []), dtype=np.int64).reshape(-1)
    content_units = np.asarray(source_map.get("content_units", []), dtype=np.int64).reshape(-1)
    source_silence_mask = np.asarray(
        source_map.get("source_silence_mask", np.zeros_like(dur_anchor_src, dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1)
    sep_hint = np.asarray(
        source_map.get("sep_hint", np.zeros_like(dur_anchor_src, dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1)
    boundary_confidence = np.asarray(
        source_map.get("boundary_confidence", np.zeros_like(dur_anchor_src, dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1)
    source_boundary_cue = np.asarray(
        source_map.get("source_boundary_cue", np.zeros_like(dur_anchor_src, dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1)
    raw_tokens = source_map.get("hubert", source_map.get("content_tokens"))
    raw_silent_token_count = 0
    if silent_token is not None:
        silent_token = int(silent_token)
        if raw_tokens is not None:
            raw_silent_token_count = sum(1 for token in _as_token_list(raw_tokens) if int(token) == silent_token)
        elif dur_anchor_src.size > 0 and content_units.size == dur_anchor_src.size:
            raw_silent_token_count = int(dur_anchor_src[content_units == silent_token].sum())
    return {
        "unit_count": int(dur_anchor_src.size),
        "raw_silent_token_count": int(raw_silent_token_count),
        "sep_nonzero_count": int((sep_hint > 0.5).sum()),
        "source_silence_run_count": int((source_silence_mask > 0.5).sum()),
        "boundary_confidence_max": _finite_max_or_nan(boundary_confidence),
        "source_boundary_cue_max": _finite_max_or_nan(source_boundary_cue),
    }


def _coerce_unit_prior_array(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        arr = value
    elif torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.size <= 0:
        raise ValueError("unit_log_prior must contain at least one value.")
    return arr


def _extract_scalar_meta(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return str(value.item())
        flat = value.reshape(-1)
        if flat.size <= 0:
            return None
        return str(flat[0].item() if hasattr(flat[0], "item") else flat[0])
    if torch.is_tensor(value):
        flat = value.detach().cpu().reshape(-1)
        if flat.numel() <= 0:
            return None
        return str(flat[0].item())
    return str(value)


def _extract_float_meta(value) -> float | None:
    scalar = _extract_scalar_meta(value)
    if scalar is None:
        return None
    try:
        return float(scalar)
    except (TypeError, ValueError):
        return None


def _extract_int_meta(value) -> int | None:
    scalar = _extract_scalar_meta(value)
    if scalar is None:
        return None
    try:
        return int(float(scalar))
    except (TypeError, ValueError):
        return None


def _extract_bool_meta(value) -> bool | None:
    scalar = _extract_scalar_meta(value)
    if scalar is None:
        return None
    normalized = str(scalar).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def _coerce_bool_prior_mask(value, *, expected_size: int) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        arr = value
    elif torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    arr = np.asarray(arr).reshape(-1)
    if arr.size != int(expected_size):
        raise ValueError(
            "unit_log_prior_is_default size mismatch: "
            f"got {int(arr.size)}, expected {int(expected_size)}"
        )
    return arr.astype(bool, copy=False)


def _coerce_int_prior_array(value, *, expected_size: int) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        arr = value
    elif torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    arr = np.asarray(arr, dtype=np.int64).reshape(-1)
    if arr.size != int(expected_size):
        raise ValueError(
            "unit_count size mismatch: "
            f"got {int(arr.size)}, expected {int(expected_size)}"
        )
    return arr


def _resolve_unit_prior_default_mask(bundle: Mapping[str, object], *, expected_size: int) -> np.ndarray | None:
    default_mask = _coerce_bool_prior_mask(
        bundle.get("unit_log_prior_is_default", bundle.get("unit_prior_is_default")),
        expected_size=int(expected_size),
    )
    if default_mask is not None:
        return default_mask
    min_count = _extract_int_meta(bundle.get("unit_prior_min_count"))
    if min_count is None:
        return None
    counts = _coerce_int_prior_array(
        bundle.get("unit_count", bundle.get("unit_counts", bundle.get("unit_prior_count"))),
        expected_size=int(expected_size),
    )
    if counts is None:
        return None
    return counts < int(min_count)


def _resolve_unit_prior_default_value(bundle: Mapping[str, object], prior: np.ndarray) -> float:
    explicit = _extract_float_meta(bundle.get("global_speech_log_prior"))
    if explicit is None:
        explicit = _extract_float_meta(bundle.get("unit_prior_default_value"))
    if explicit is not None and np.isfinite(explicit):
        return float(explicit)
    finite = np.asarray(prior, dtype=np.float32).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size > 0:
        return float(np.median(finite))
    return 0.0


def _normalize_unit_prior_bundle(
    payload,
    *,
    default_source: str | None = None,
) -> dict[str, object]:
    if isinstance(payload, Mapping):
        prior_value = payload.get("unit_log_prior", payload.get("unit_prior"))
        if prior_value is None:
            raise ValueError("unit prior bundle is missing unit_log_prior.")
        prior = _coerce_unit_prior_array(prior_value)
        source = _extract_scalar_meta(payload.get("unit_prior_source", default_source))
        version = _extract_scalar_meta(payload.get("unit_prior_version"))
        min_count = _extract_int_meta(payload.get("unit_prior_min_count"))
        default_policy = _extract_scalar_meta(payload.get("unit_prior_default_policy", "legacy_prior_median"))
        global_backoff = _extract_scalar_meta(payload.get("unit_prior_global_backoff"))
        default_mask = _resolve_unit_prior_default_mask(
            payload,
            expected_size=int(prior.shape[0]),
        )
        count_array = _coerce_int_prior_array(
            payload.get("unit_count", payload.get("unit_counts", payload.get("unit_prior_count"))),
            expected_size=int(prior.shape[0]),
        )
        backoff_weight_value = payload.get("unit_prior_backoff_weight")
        backoff_weight = (
            None if backoff_weight_value is None else _coerce_unit_prior_array(backoff_weight_value)
        )
        if backoff_weight is not None and int(backoff_weight.shape[0]) != int(prior.shape[0]):
            raise ValueError(
                "unit_prior_backoff_weight size mismatch: "
                f"got {int(backoff_weight.shape[0])}, expected {int(prior.shape[0])}"
            )
        frontend_signature = _extract_scalar_meta(payload.get("unit_prior_frontend_signature"))
        silent_token = _extract_scalar_meta(payload.get("unit_prior_silent_token"))
        separator_aware = _extract_bool_meta(payload.get("unit_prior_separator_aware"))
        tail_open_units = _extract_int_meta(payload.get("unit_prior_tail_open_units"))
        emit_silence_runs = _extract_bool_meta(payload.get("unit_prior_emit_silence_runs"))
        debounce_min_run_frames = _extract_int_meta(payload.get("unit_prior_debounce_min_run_frames"))
        phrase_boundary_threshold = _extract_float_meta(payload.get("unit_prior_phrase_boundary_threshold"))
        filter_exclude_open_runs = _extract_bool_meta(payload.get("unit_prior_filter_exclude_open_runs"))
        filter_only_sealed_runs = _extract_bool_meta(payload.get("unit_prior_filter_only_sealed_runs"))
        filter_drop_edge_runs = _extract_int_meta(payload.get("unit_prior_filter_drop_edge_runs"))
        filter_min_run_stability = _extract_float_meta(payload.get("unit_prior_filter_min_run_stability"))
        prior_sha1 = _extract_scalar_meta(payload.get("unit_prior_sha1"))
        prior_build_cmd = _extract_scalar_meta(payload.get("unit_prior_build_cmd"))
        prior_input_manifest_sha1 = _extract_scalar_meta(payload.get("unit_prior_input_manifest_sha1"))
        prior_input_count = _extract_int_meta(payload.get("unit_prior_input_count"))
        special_token_policy = _extract_scalar_meta(payload.get("unit_prior_special_token_policy"))
        accepted_run_count = _extract_int_meta(payload.get("unit_prior_accepted_run_count"))
    else:
        prior = _coerce_unit_prior_array(payload)
        source = _extract_scalar_meta(default_source)
        version = None
        min_count = None
        default_policy = "legacy_prior_median"
        global_backoff = None
        default_mask = None
        count_array = None
        backoff_weight = None
        frontend_signature = None
        silent_token = None
        separator_aware = None
        tail_open_units = None
        emit_silence_runs = None
        debounce_min_run_frames = None
        phrase_boundary_threshold = None
        filter_exclude_open_runs = None
        filter_only_sealed_runs = None
        filter_drop_edge_runs = None
        filter_min_run_stability = None
        prior_sha1 = None
        prior_build_cmd = None
        prior_input_manifest_sha1 = None
        prior_input_count = None
        special_token_policy = None
        accepted_run_count = None
    default_value = _resolve_unit_prior_default_value(
        payload if isinstance(payload, Mapping) else {},
        prior,
    )
    default_count = (
        None
        if default_mask is None
        else int(np.asarray(default_mask, dtype=np.int64).sum())
    )
    observed_count = (
        None
        if count_array is None
        else int((np.asarray(count_array, dtype=np.int64) > 0).sum())
    )
    low_count_count = (
        None
        if count_array is None or min_count is None
        else int(
            (
                (np.asarray(count_array, dtype=np.int64) > 0)
                & (np.asarray(count_array, dtype=np.int64) < int(min_count))
            ).sum()
        )
    )
    return {
        "unit_log_prior": prior.astype(np.float32, copy=False),
        "unit_count": count_array,
        "unit_prior_backoff_weight": backoff_weight,
        "unit_prior_source": source,
        "unit_prior_version": version,
        "unit_prior_vocab_size": int(prior.shape[0]),
        "global_speech_log_prior": float(default_value),
        "unit_prior_min_count": min_count,
        "unit_prior_default_policy": default_policy,
        "unit_prior_global_backoff": global_backoff,
        "unit_prior_default_value": float(default_value),
        "unit_log_prior_is_default": default_mask,
        "unit_prior_default_count": default_count,
        "unit_prior_observed_count": observed_count,
        "unit_prior_low_count_count": low_count_count,
        "unit_prior_frontend_signature": frontend_signature,
        "unit_prior_silent_token": silent_token,
        "unit_prior_separator_aware": separator_aware,
        "unit_prior_tail_open_units": tail_open_units,
        "unit_prior_emit_silence_runs": emit_silence_runs,
        "unit_prior_debounce_min_run_frames": debounce_min_run_frames,
        "unit_prior_phrase_boundary_threshold": phrase_boundary_threshold,
        "unit_prior_filter_exclude_open_runs": filter_exclude_open_runs,
        "unit_prior_filter_only_sealed_runs": filter_only_sealed_runs,
        "unit_prior_filter_drop_edge_runs": filter_drop_edge_runs,
        "unit_prior_filter_min_run_stability": filter_min_run_stability,
        "unit_prior_sha1": prior_sha1,
        "unit_prior_build_cmd": prior_build_cmd,
        "unit_prior_input_manifest_sha1": prior_input_manifest_sha1,
        "unit_prior_input_count": prior_input_count,
        "unit_prior_special_token_policy": special_token_policy,
        "unit_prior_accepted_run_count": accepted_run_count,
    }


@lru_cache(maxsize=4)
def _load_unit_log_prior_bundle_cached(
    path: str,
    mtime_ns: int,
    size: int,
) -> dict[str, object]:
    prior_path = Path(path)
    suffix = prior_path.suffix.lower()
    payload = None
    if suffix == ".npy":
        payload = np.load(prior_path, allow_pickle=True)
    elif suffix == ".npz":
        with np.load(prior_path, allow_pickle=True) as data:
            payload = {key: data[key] for key in data.files}
    elif suffix in {".pt", ".pth"}:
        payload = torch.load(prior_path, map_location="cpu", weights_only=False)
    else:
        raise ValueError(
            f"Unsupported unit prior file format: {prior_path}. Expected one of: .npy, .npz, .pt, .pth"
        )
    bundle = _normalize_unit_prior_bundle(payload, default_source=str(prior_path))
    bundle["unit_prior_path"] = str(prior_path)
    bundle["unit_prior_path_mtime_ns"] = int(mtime_ns)
    bundle["unit_prior_path_size"] = int(size)
    return bundle


def load_unit_log_prior_bundle(path: str) -> dict[str, object]:
    prior_path = Path(path)
    if not prior_path.exists():
        raise FileNotFoundError(f"unit prior path does not exist: {prior_path}")
    stat = prior_path.stat()
    return _load_unit_log_prior_bundle_cached(
        str(prior_path),
        int(stat.st_mtime_ns),
        int(stat.st_size),
    )


def attach_unit_log_prior_to_source_cache(
    cache: dict,
    *,
    unit_prior_bundle: Mapping[str, object] | None = None,
    unit_log_prior=None,
    unit_prior_source: str | None = None,
    unit_prior_version: str | None = None,
    overwrite: bool = False,
) -> dict:
    if "content_units" not in cache:
        raise ValueError("source cache must contain content_units before attaching unit_log_prior.")
    if not overwrite and cache.get("unit_log_prior") is not None:
        return cache
    bundle = (
        dict(unit_prior_bundle)
        if isinstance(unit_prior_bundle, Mapping)
        else _normalize_unit_prior_bundle(
            {
                "unit_log_prior": unit_log_prior,
                "unit_prior_source": unit_prior_source,
                "unit_prior_version": unit_prior_version,
            }
        )
    )
    prior = _coerce_unit_prior_array(bundle["unit_log_prior"])
    default_value = _resolve_unit_prior_default_value(bundle, prior)
    default_mask = _resolve_unit_prior_default_mask(
        bundle,
        expected_size=int(prior.shape[0]),
    )
    count_array = _coerce_int_prior_array(
        bundle.get("unit_count", bundle.get("unit_counts", bundle.get("unit_prior_count"))),
        expected_size=int(prior.shape[0]),
    )
    cache_signature = duration_v3_cache_meta_signature(cache)
    prior_signature = _extract_scalar_meta(bundle.get("unit_prior_frontend_signature"))
    if prior_signature not in {None, "missing", "mixed"} and cache_signature != prior_signature:
        raise ValueError(
            "unit prior frontend signature mismatch: "
            f"bundle={prior_signature!r} cache={cache_signature!r}"
        )
    content_units = np.asarray(cache["content_units"], dtype=np.int64).reshape(-1)
    if content_units.size <= 0:
        cache["unit_log_prior"] = np.zeros((0,), dtype=np.float32)
        cache["unit_log_prior_is_default"] = np.zeros((0,), dtype=np.int64)
        cache["unit_log_prior_count"] = np.zeros((0,), dtype=np.int64)
        out_of_vocab_count = 0
    else:
        in_vocab = (content_units >= 0) & (content_units < int(prior.shape[0]))
        mapped = np.full((content_units.shape[0],), float(default_value), dtype=np.float32)
        default_flags = np.ones((content_units.shape[0],), dtype=np.int64)
        mapped_count = np.zeros((content_units.shape[0],), dtype=np.int64)
        if bool(in_vocab.any()):
            mapped[in_vocab] = prior[content_units[in_vocab]].astype(np.float32, copy=False)
            if default_mask is None:
                default_flags[in_vocab] = 0
            else:
                default_flags[in_vocab] = default_mask[content_units[in_vocab]].astype(np.int64, copy=False)
            if count_array is not None:
                mapped_count[in_vocab] = count_array[content_units[in_vocab]].astype(np.int64, copy=False)
        cache["unit_log_prior"] = mapped
        cache["unit_log_prior_is_default"] = default_flags
        cache["unit_log_prior_count"] = mapped_count
        out_of_vocab_count = int((~in_vocab).sum())
    default_count = int(np.asarray(cache["unit_log_prior_is_default"], dtype=np.int64).sum())
    min_count = _extract_int_meta(bundle.get("unit_prior_min_count"))
    observed_count = (
        None
        if count_array is None
        else int((np.asarray(count_array, dtype=np.int64) > 0).sum())
    )
    low_count_count = (
        None
        if count_array is None or min_count is None
        else int(
            (
                (np.asarray(count_array, dtype=np.int64) > 0)
                & (np.asarray(count_array, dtype=np.int64) < int(min_count))
            ).sum()
        )
    )
    cache_meta = resolve_duration_v3_cache_meta(cache)
    cache[UNIT_LOG_PRIOR_META_KEY] = {
        "unit_prior_source": _extract_scalar_meta(bundle.get("unit_prior_source")),
        "unit_prior_version": _extract_scalar_meta(bundle.get("unit_prior_version")),
        "unit_prior_vocab_size": int(prior.shape[0]),
        "unit_prior_path": _extract_scalar_meta(bundle.get("unit_prior_path")),
        "unit_prior_min_count": (None if min_count is None else int(min_count)),
        "unit_prior_default_value": float(default_value),
        "unit_prior_default_policy": _extract_scalar_meta(bundle.get("unit_prior_default_policy", "legacy_prior_median")),
        "unit_prior_global_backoff": _extract_scalar_meta(bundle.get("unit_prior_global_backoff")),
        "unit_prior_default_count": int(default_count),
        "unit_prior_observed_count": observed_count,
        "unit_prior_low_count_count": low_count_count,
        "unit_prior_unseen_count": int(default_count),
        "unit_prior_out_of_vocab_count": int(out_of_vocab_count),
        "unit_prior_frontend_signature": prior_signature,
        "unit_prior_silent_token": _extract_scalar_meta(bundle.get("unit_prior_silent_token")),
        "unit_prior_separator_aware": _extract_bool_meta(bundle.get("unit_prior_separator_aware")),
        "unit_prior_tail_open_units": _extract_int_meta(bundle.get("unit_prior_tail_open_units")),
        "unit_prior_emit_silence_runs": _extract_bool_meta(bundle.get("unit_prior_emit_silence_runs")),
        "unit_prior_debounce_min_run_frames": _extract_int_meta(bundle.get("unit_prior_debounce_min_run_frames")),
        "unit_prior_phrase_boundary_threshold": _extract_float_meta(bundle.get("unit_prior_phrase_boundary_threshold")),
        "unit_prior_filter_exclude_open_runs": _extract_bool_meta(bundle.get("unit_prior_filter_exclude_open_runs")),
        "unit_prior_filter_only_sealed_runs": _extract_bool_meta(bundle.get("unit_prior_filter_only_sealed_runs")),
        "unit_prior_filter_drop_edge_runs": _extract_int_meta(bundle.get("unit_prior_filter_drop_edge_runs")),
        "unit_prior_filter_min_run_stability": _extract_float_meta(bundle.get("unit_prior_filter_min_run_stability")),
        "unit_prior_path_mtime_ns": _extract_int_meta(bundle.get("unit_prior_path_mtime_ns")),
        "unit_prior_path_size": _extract_int_meta(bundle.get("unit_prior_path_size")),
        "unit_prior_sha1": _extract_scalar_meta(bundle.get("unit_prior_sha1")),
        "unit_prior_build_cmd": _extract_scalar_meta(bundle.get("unit_prior_build_cmd")),
        "unit_prior_input_manifest_sha1": _extract_scalar_meta(bundle.get("unit_prior_input_manifest_sha1")),
        "unit_prior_input_count": _extract_int_meta(bundle.get("unit_prior_input_count")),
        "unit_prior_special_token_policy": _extract_scalar_meta(bundle.get("unit_prior_special_token_policy")),
        "unit_prior_accepted_run_count": _extract_int_meta(bundle.get("unit_prior_accepted_run_count")),
        "frontend_meta_signature": cache_signature,
        "silent_token": (None if cache_meta is None else cache_meta.get("silent_token")),
        "separator_aware": (None if cache_meta is None else bool(cache_meta.get("separator_aware", False))),
        "tail_open_units": (None if cache_meta is None else int(cache_meta.get("tail_open_units", 0))),
        "emit_silence_runs": (None if cache_meta is None else bool(cache_meta.get("emit_silence_runs", False))),
        "debounce_min_run_frames": (
            None if cache_meta is None else int(cache_meta.get("debounce_min_run_frames", 0))
        ),
        "phrase_boundary_threshold": (
            None if cache_meta is None else float(cache_meta.get("phrase_boundary_threshold", 0.0))
        ),
    }
    return cache


def maybe_attach_unit_log_prior_from_path(
    cache: dict,
    *,
    unit_prior_path: str | None,
    overwrite: bool = False,
) -> dict:
    if not unit_prior_path:
        return cache
    bundle = load_unit_log_prior_bundle(str(unit_prior_path))
    return attach_unit_log_prior_to_source_cache(
        cache,
        unit_prior_bundle=bundle,
        overwrite=overwrite,
    )


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


def _build_source_boundary_cue_v3(
    *,
    dur_anchor_src: torch.Tensor,
    unit_mask: torch.Tensor,
    sep_hint: torch.Tensor,
    open_run_mask: torch.Tensor,
    sealed_mask: torch.Tensor,
    boundary_confidence: torch.Tensor,
) -> torch.Tensor:
    from modules.Conan.rhythm.source_boundary import build_source_boundary_cue as _legacy_build_source_boundary_cue

    return _legacy_build_source_boundary_cue(
        dur_anchor_src=dur_anchor_src,
        unit_mask=unit_mask,
        sep_hint=sep_hint,
        open_run_mask=open_run_mask,
        sealed_mask=sealed_mask,
        boundary_confidence=boundary_confidence,
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
    dur_anchor_src = torch.as_tensor(dur_anchor_src, dtype=torch.float32).reshape(1, -1)
    sep_hint = torch.as_tensor(sep_hint, dtype=torch.long).reshape(1, -1)
    open_run_mask = torch.as_tensor(open_run_mask, dtype=torch.long).reshape(1, -1)
    sealed_mask = torch.as_tensor(sealed_mask, dtype=torch.float32).reshape(1, -1)
    boundary_confidence = torch.as_tensor(boundary_confidence, dtype=torch.float32).reshape(1, -1)
    unit_mask = dur_anchor_src.gt(0).float()
    source_boundary_cue = _build_source_boundary_cue_v3(
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
    emit_silence_runs: bool = True,
    debounce_min_run_frames: int = 2,
    phrase_boundary_threshold: float = 0.55,
    unit_prior_path: str | None = None,
    mel=None,
    frame_silence_mask=None,
) -> dict[str, np.ndarray]:
    token_list = _as_token_list(content_tokens)
    frontend = _cached_frontend(
        silent_token=silent_token,
        separator_aware=separator_aware,
        tail_open_units=tail_open_units,
        emit_silence_runs=emit_silence_runs,
        debounce_min_run_frames=debounce_min_run_frames,
    )
    token_tensor = torch.tensor(
        [token_list],
        dtype=torch.long,
    )
    resolved_frame_silence_mask = _resolve_frame_silence_mask(
        token_list,
        silent_token=silent_token,
        frame_silence_mask=frame_silence_mask,
        mel=mel,
    )
    frame_silence_tensor = None
    if resolved_frame_silence_mask is not None:
        frame_silence_tensor = resolved_frame_silence_mask.reshape(1, -1).to(dtype=torch.float32)
    batch = frontend.from_content_tensor(
        token_tensor,
        mark_last_open=False,
        frame_silence_mask=frame_silence_tensor,
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
    if resolved_frame_silence_mask is not None and int(source_cache["dur_anchor_src"].shape[0]) > 0:
        sep_hint = np.asarray(source_cache["sep_hint"], dtype=np.int64).copy()
        silence_runs = np.asarray(source_cache["source_silence_mask"], dtype=np.float32) > 0.5
        silence_indices = np.flatnonzero(silence_runs)
        for silence_idx in silence_indices.tolist():
            if silence_idx > 0:
                sep_hint[silence_idx - 1] = 1
            if silence_idx < int(sep_hint.shape[0] - 1):
                sep_hint[silence_idx] = 1
        source_cache["sep_hint"] = sep_hint.astype(np.int64, copy=False)
        source_cache["boundary_confidence"] = np.asarray(
            estimate_boundary_confidence(
                durations=source_cache["dur_anchor_src"].tolist(),
                sep_hint=source_cache["sep_hint"].tolist(),
                open_run_mask=source_cache["open_run_mask"].tolist(),
            ),
            dtype=np.float32,
        )
        source_cache["source_run_stability"] = np.asarray(
            estimate_run_stability(
                durations=source_cache["dur_anchor_src"].tolist(),
                silence_mask=source_cache["source_silence_mask"].tolist(),
                open_run_mask=source_cache["open_run_mask"].tolist(),
                sep_hint=source_cache["sep_hint"].tolist(),
                boundary_confidence=source_cache["boundary_confidence"].tolist(),
                min_speech_frames=debounce_min_run_frames,
                min_silence_frames=debounce_min_run_frames,
            ),
            dtype=np.float32,
        )
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
    if resolved_frame_silence_mask is not None:
        source_cache["source_silence_frame_count"] = np.asarray(
            [int((resolved_frame_silence_mask > 0.5).sum().item())],
            dtype=np.int32,
        )
    frontend_diag = collect_duration_v3_frontend_diagnostics(
        {
            **source_cache,
            "hubert": np.asarray(token_list, dtype=np.int64),
        },
        silent_token=silent_token,
    )
    source_cache.update(
        {
            "raw_silent_token_count": np.asarray([frontend_diag["raw_silent_token_count"]], dtype=np.int32),
            "sep_nonzero_count": np.asarray([frontend_diag["sep_nonzero_count"]], dtype=np.int32),
            "source_silence_run_count": np.asarray([frontend_diag["source_silence_run_count"]], dtype=np.int32),
            "boundary_confidence_max": np.asarray([frontend_diag["boundary_confidence_max"]], dtype=np.float32),
            "source_boundary_cue_max": np.asarray([frontend_diag["source_boundary_cue_max"]], dtype=np.float32),
        }
    )
    source_cache = attach_duration_v3_cache_meta(
        source_cache,
        silent_token=silent_token,
        separator_aware=separator_aware,
        tail_open_units=tail_open_units,
        emit_silence_runs=emit_silence_runs,
        debounce_min_run_frames=debounce_min_run_frames,
        phrase_boundary_threshold=phrase_boundary_threshold,
    )
    return maybe_attach_unit_log_prior_from_path(
        source_cache,
        unit_prior_path=unit_prior_path,
    )


build_source_rhythm_cache = build_source_rhythm_cache_v3


__all__ = [
    "DURATION_V3_CACHE_META_KEY",
    "DURATION_V3_SOURCE_CACHE_VERSION",
    "UNIT_LOG_PRIOR_META_KEY",
    "assert_duration_v3_cache_meta_compatible",
    "attach_unit_log_prior_to_source_cache",
    "attach_duration_v3_cache_meta",
    "build_duration_v3_cache_meta",
    "build_duration_v3_frontend_signature",
    "collect_duration_v3_frontend_diagnostics",
    "build_source_phrase_cache",
    "build_source_rhythm_cache",
    "build_source_rhythm_cache_v3",
    "duration_v3_cache_meta_signature",
    "estimate_boundary_confidence",
    "estimate_run_stability",
    "infer_frame_silence_mask_from_mel",
    "load_unit_log_prior_bundle",
    "maybe_attach_unit_log_prior_from_path",
    "resolve_duration_v3_cache_meta",
]
