from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
import torch

from modules.Conan.rhythm_v3.g_stats import (
    build_softclean_weights,
    compute_global_rate,
    is_softclean_global_rate_variant,
    summarize_global_rate_support,
)
from tasks.Conan.rhythm.duration_v3.metrics import (
    budget_hit_rate,
    cumulative_drift,
    prefix_discrepancy,
    same_text_gap,
    silence_leakage,
    tempo_explainability,
    tempo_monotonicity,
    tempo_tie_rate,
    transfer_slope,
)

from .core import RhythmV3DebugRecord, derive_record, load_debug_records, record_summary, weighted_median_np


DEFAULT_REVIEW_SILENCE_TAU = 0.35
DEFAULT_REVIEW_BOUNDARY_THRESHOLD = 0.55
DEFAULT_REVIEW_UNIT_STEP_MS = 20.0
DEFAULT_GATE_MIN_SPEECH_RATIO = 0.6
_REAL_REF_CONDITIONS = {"", "nan", "real", "real_reference"}
_NEGATIVE_CONTROL_REF_CONDITIONS = {"source_only", "random_ref", "shuffled_ref"}


def weighted_median(values: Any, weight: Any | None = None) -> float:
    return weighted_median_np(
        np.asarray(values, dtype=np.float32),
        None if weight is None else np.asarray(weight, dtype=np.float32),
    )


def bootstrap_ci(
    values: Any,
    *,
    reducer: Callable[[np.ndarray], float] | None = None,
    n_boot: int = 512,
    ci: float = 95.0,
    seed: int = 0,
) -> tuple[float, float, float]:
    vals = np.asarray(values, dtype=np.float32).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size <= 0:
        return float("nan"), float("nan"), float("nan")
    reducer = reducer or (lambda x: float(np.nanmean(x)))
    point = float(reducer(vals))
    if vals.size == 1:
        return point, point, point
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(max(1, int(n_boot))):
        sample = vals[rng.integers(0, vals.size, size=vals.size)]
        boots.append(float(reducer(sample)))
    boots_arr = np.asarray(boots, dtype=np.float32)
    alpha = max(0.0, min(49.0, (100.0 - float(ci)) * 0.5))
    low = float(np.nanpercentile(boots_arr, alpha))
    high = float(np.nanpercentile(boots_arr, 100.0 - alpha))
    return low, point, high


def ensure_debug_records(
    items: str | Path | RhythmV3DebugRecord | Mapping[str, Any] | Sequence[Any],
) -> list[RhythmV3DebugRecord]:
    if isinstance(items, RhythmV3DebugRecord):
        return [items]
    if isinstance(items, Mapping):
        return [RhythmV3DebugRecord.from_mapping(dict(items))]
    if isinstance(items, (str, Path)):
        return load_debug_records(items)
    records: list[RhythmV3DebugRecord] = []
    for item in items:
        records.extend(ensure_debug_records(item))
    return records


def _meta(record: RhythmV3DebugRecord, *keys: str, default: Any = None) -> Any:
    meta = dict(record.metadata or {})
    for key in keys:
        value = meta.get(key)
        if value is not None:
            return value
    return default


def _as_float(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _as_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _record_prompt_domain_invalid(record: RhythmV3DebugRecord) -> bool:
    domain_valid = _as_float(_meta(record, "g_domain_valid", "g_valid", default=float("nan")))
    return bool(np.isfinite(domain_valid) and domain_valid <= 0.5)


def _normalize_ref_condition(value: Any) -> str:
    return _as_str(value, default="").strip().lower()


def _stable_slope_strength(
    x: Any,
    y: Any,
    *,
    metric_fn: Callable[[Any, Any], Mapping[str, float]],
) -> float:
    arr_x, arr_y = _align_pair(x, y, dtype=np.float32)
    if arr_x is None or arr_y is None:
        return float("nan")
    valid = np.isfinite(arr_x) & np.isfinite(arr_y)
    if not bool(np.any(valid)):
        return float("nan")
    arr_x = arr_x[valid]
    arr_y = arr_y[valid]
    if arr_x.size <= 0:
        return float("nan")
    if float(np.nanmax(arr_x) - np.nanmin(arr_x)) <= 1.0e-6:
        return 0.0
    metric = metric_fn(arr_x, arr_y)
    slope = float(metric.get("robust_slope", float("nan")))
    return 0.0 if not np.isfinite(slope) else slope


def _compute_negative_control_gap(
    frame: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    metric_fn: Callable[[Any, Any], Mapping[str, float]],
) -> tuple[float, int, int]:
    if frame.empty or "ref_condition" not in frame or x_col not in frame or y_col not in frame:
        return float("nan"), 0, 0
    ref_condition = frame["ref_condition"].map(_normalize_ref_condition)
    real = frame[ref_condition.isin(_REAL_REF_CONDITIONS)]
    negative = frame[ref_condition.isin(_NEGATIVE_CONTROL_REF_CONDITIONS)]
    real_count = int(real.shape[0])
    negative_count = int(negative.shape[0])
    if real.empty or negative.empty:
        return float("nan"), real_count, negative_count
    real_strength = _stable_slope_strength(real[x_col], real[y_col], metric_fn=metric_fn)
    negative_strength = _stable_slope_strength(negative[x_col], negative[y_col], metric_fn=metric_fn)
    if not np.isfinite(real_strength) or not np.isfinite(negative_strength):
        return float("nan"), real_count, negative_count
    return float(real_strength - negative_strength), real_count, negative_count


def _infer_speaker_id(value: Any, default: str = "") -> str:
    name = _as_str(value, default=default).strip()
    if not name:
        return default
    return name.split("_", 1)[0]


def _safe_array(value: Any, *, dtype: np.dtype | type = np.float32) -> np.ndarray | None:
    if value is None:
        return None
    try:
        return np.asarray(value, dtype=dtype).reshape(-1)
    except Exception:
        return None


def _safe_bool_array(value: Any) -> np.ndarray | None:
    arr = _safe_array(value, dtype=np.float32)
    if arr is None:
        return None
    return arr > 0.5


def _align_pair(
    a: Any,
    b: Any,
    *,
    dtype: np.dtype | type = np.float32,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    arr_a = _safe_array(a, dtype=dtype)
    arr_b = _safe_array(b, dtype=dtype)
    if arr_a is None or arr_b is None:
        return None, None
    width = min(int(arr_a.shape[0]), int(arr_b.shape[0]))
    if width <= 0:
        return None, None
    return arr_a[:width], arr_b[:width]


def _speech_tempo(duration: Any, speech_mask: Any) -> float:
    return compute_speech_tempo_for_analysis(
        source_duration_obs=duration,
        source_speech_mask=speech_mask,
    )


def _warn_status(name: str, status: str) -> None:
    if status == "ok":
        return
    warnings.warn(f"{name} failed: {status}", RuntimeWarning, stacklevel=3)


def _resolve_analysis_softclean_weight(
    *,
    variant: str,
    speech_tensor: torch.Tensor,
    valid_tensor: torch.Tensor | None,
    explicit_weight: torch.Tensor | None,
    closed_mask: Any | None = None,
    boundary_confidence: Any | None = None,
) -> torch.Tensor | None:
    if explicit_weight is not None:
        return explicit_weight
    if not is_softclean_global_rate_variant(variant):
        return None
    closed_tensor = None
    boundary_tensor = None
    if closed_mask is not None:
        closed_tensor = torch.as_tensor(
            np.asarray(closed_mask, dtype=np.float32).reshape(1, -1),
            dtype=torch.float32,
        )
    if boundary_confidence is not None:
        boundary_tensor = torch.as_tensor(
            np.asarray(boundary_confidence, dtype=np.float32).reshape(1, -1),
            dtype=torch.float32,
        )
    return build_softclean_weights(
        speech_mask=speech_tensor,
        valid_mask=valid_tensor,
        closed_mask=closed_tensor,
        boundary_confidence=boundary_tensor,
    )


def _maybe_with_status(
    value: float,
    status: str,
    *,
    name: str,
    return_status: bool,
) -> float | tuple[float, str]:
    if return_status:
        return float(value), str(status)
    _warn_status(name, str(status))
    return float(value)


def _status_is_ok(status: Any) -> bool:
    normalized = _as_str(status, default="").strip().lower()
    return normalized == "ok"


def _resolve_prompt_domain_stats(
    *,
    prompt_duration: np.ndarray | None,
    prompt_speech: np.ndarray | None,
    speech_ratio_hint: Any = None,
    support_count_hint: Any = None,
    valid_count_hint: Any = None,
    g_valid_hint: Any = None,
    g_domain_valid_hint: Any = None,
    min_speech_ratio_hint: Any = None,
    default_min_speech_ratio: float = DEFAULT_GATE_MIN_SPEECH_RATIO,
) -> dict[str, float]:
    support_count = _as_float(support_count_hint, default=float("nan"))
    valid_count = _as_float(valid_count_hint, default=float("nan"))
    speech_ratio = _as_float(speech_ratio_hint, default=float("nan"))
    min_speech_ratio = _as_float(min_speech_ratio_hint, default=float("nan"))
    if not np.isfinite(min_speech_ratio):
        min_speech_ratio = float(default_min_speech_ratio)

    if prompt_duration is not None:
        duration = np.asarray(prompt_duration, dtype=np.float32).reshape(-1)
        if not np.isfinite(valid_count):
            valid_count = float(duration.shape[0])
        if prompt_speech is not None:
            speech = np.asarray(prompt_speech, dtype=np.float32).reshape(-1)
            width = min(int(duration.shape[0]), int(speech.shape[0]))
            duration = duration[:width]
            speech = speech[:width]
            if not np.isfinite(speech_ratio):
                speech_ratio = float(
                    np.sum(duration * speech) / max(float(np.sum(duration)), 1.0e-6)
                )
            if not np.isfinite(support_count):
                support_count = float(np.sum((speech > 0.5).astype(np.float32)))

    g_valid_support = _as_float(g_valid_hint, default=float("nan"))
    if not np.isfinite(g_valid_support):
        g_valid_support = (
            1.0 if np.isfinite(support_count) and float(support_count) >= 1.0 else 0.0
        )
    g_domain_valid = _as_float(g_domain_valid_hint, default=float("nan"))
    if not np.isfinite(g_domain_valid):
        g_domain_valid = (
            1.0
            if bool(g_valid_support > 0.5)
            and np.isfinite(speech_ratio)
            and float(speech_ratio) >= (float(min_speech_ratio) - 1.0e-6)
            else 0.0
        )
    return {
        "g_valid_support": float(g_valid_support),
        "g_domain_valid": float(g_domain_valid),
        "prompt_speech_ratio": float(speech_ratio),
        "g_min_speech_ratio": float(min_speech_ratio),
        "g_support_count": float(support_count),
        "g_valid_count": float(valid_count),
    }


def compute_g(
    duration_obs: Any,
    *,
    speech_mask: Any,
    valid_mask: Any | None = None,
    variant: str = "raw_median",
    trim_ratio: float = 0.2,
    drop_edge_runs: int = 0,
    weight: Any | None = None,
    unit_ids: Any | None = None,
    unit_log_prior: Any | None = None,
    closed_mask: Any | None = None,
    boundary_confidence: Any | None = None,
    min_boundary_confidence: float | None = None,
    support_mask: Any | None = None,
    return_status: bool = False,
) -> float | tuple[float, str]:
    duration = _safe_array(duration_obs, dtype=np.float32)
    speech = _safe_array(speech_mask, dtype=np.float32)
    valid = None if valid_mask is None else _safe_array(valid_mask, dtype=np.float32)
    if duration is None or speech is None:
        missing = "duration_obs" if duration is None else "speech_mask"
        return _maybe_with_status(
            float("nan"),
            f"missing:{missing}",
            name="compute_g",
            return_status=return_status,
        )
    try:
        duration_tensor = torch.as_tensor(duration.reshape(1, -1), dtype=torch.float32)
        speech_tensor = torch.as_tensor(speech.reshape(1, -1), dtype=torch.float32)
        valid_tensor = (
            None
            if valid is None
            else torch.as_tensor(valid.reshape(1, -1), dtype=torch.float32)
        )
        resolved_support = None
        resolved_weight = (
            None
            if weight is None
            else torch.as_tensor(np.asarray(weight, dtype=np.float32).reshape(1, -1), dtype=torch.float32)
        )
        if support_mask is not None:
            resolved_support = torch.as_tensor(
                np.asarray(support_mask, dtype=np.float32).reshape(1, -1),
                dtype=torch.float32,
            )
        elif closed_mask is not None or boundary_confidence is not None:
            support_stats = summarize_global_rate_support(
                speech_mask=speech_tensor,
                valid_mask=valid_tensor,
                duration_obs=duration_tensor,
                drop_edge_runs=int(drop_edge_runs),
                closed_mask=(
                    None
                    if closed_mask is None
                    else torch.as_tensor(
                        np.asarray(closed_mask, dtype=np.float32).reshape(1, -1),
                        dtype=torch.float32,
                    )
                ),
                boundary_confidence=(
                    None
                    if boundary_confidence is None
                    else torch.as_tensor(
                        np.asarray(boundary_confidence, dtype=np.float32).reshape(1, -1),
                        dtype=torch.float32,
                    )
                ),
                min_boundary_confidence=min_boundary_confidence,
            )
            if is_softclean_global_rate_variant(variant):
                resolved_support = (speech_tensor > 0.5) & (
                    valid_tensor > 0.5 if valid_tensor is not None else torch.ones_like(speech_tensor, dtype=torch.bool)
                )
                resolved_weight = _resolve_analysis_softclean_weight(
                    variant=variant,
                    speech_tensor=speech_tensor,
                    valid_tensor=valid_tensor,
                    explicit_weight=resolved_weight,
                    closed_mask=closed_mask,
                    boundary_confidence=boundary_confidence,
                )
            else:
                resolved_support = support_stats.support_mask.float()
        value = compute_global_rate(
            log_dur=torch.log(duration_tensor.clamp_min(1.0e-4)),
            speech_mask=speech_tensor,
            valid_mask=valid_tensor,
            variant=variant,
            trim_ratio=float(trim_ratio),
            drop_edge_runs=int(drop_edge_runs),
            weight=resolved_weight,
            unit_ids=None
            if unit_ids is None
            else torch.as_tensor(np.asarray(unit_ids, dtype=np.int64).reshape(1, -1), dtype=torch.long),
            unit_prior=None
            if unit_log_prior is None
            else torch.as_tensor(np.asarray(unit_log_prior, dtype=np.float32), dtype=torch.float32),
            support_mask=resolved_support,
        )
        return _maybe_with_status(
            float(value.reshape(-1)[0].item()),
            "ok",
            name="compute_g",
            return_status=return_status,
        )
    except Exception as exc:
        return _maybe_with_status(
            float("nan"),
            f"error:{type(exc).__name__}",
            name="compute_g",
            return_status=return_status,
        )


def compute_source_global_rate_for_analysis(
    source_log_dur: Any | None = None,
    *,
    source_duration_obs: Any | None = None,
    source_speech_mask: Any,
    source_valid_mask: Any | None = None,
    g_variant: str = "raw_median",
    source_weight: Any | None = None,
    source_unit_ids: Any | None = None,
    source_unit_prior: Any | None = None,
    g_trim_ratio: float = 0.2,
    drop_edge_runs: int = 0,
    source_closed_mask: Any | None = None,
    source_boundary_confidence: Any | None = None,
    min_boundary_confidence: float | None = None,
    support_mask: Any | None = None,
    require_explicit_speech_mask: bool = False,
    return_status: bool = False,
) -> float | tuple[float, str]:
    log_dur = _safe_array(source_log_dur, dtype=np.float32)
    if log_dur is None:
        duration = _safe_array(source_duration_obs, dtype=np.float32)
        if duration is None:
            return _maybe_with_status(
                float("nan"),
                "missing:source_duration_obs",
                name="compute_source_global_rate_for_analysis",
                return_status=return_status,
            )
        log_dur = np.log(np.clip(duration, 1.0e-4, None))
    speech = _safe_array(source_speech_mask, dtype=np.float32)
    valid = None if source_valid_mask is None else _safe_array(source_valid_mask, dtype=np.float32)
    if speech is None:
        return _maybe_with_status(
            float("nan"),
            "missing:source_speech_mask",
            name="compute_source_global_rate_for_analysis",
            return_status=return_status,
        )
    try:
        log_dur_tensor = torch.as_tensor(log_dur.reshape(1, -1), dtype=torch.float32)
        speech_tensor = torch.as_tensor(speech.reshape(1, -1), dtype=torch.float32)
        valid_tensor = (
            None
            if valid is None
            else torch.as_tensor(valid.reshape(1, -1), dtype=torch.float32)
        )
        resolved_support = None
        resolved_weight = (
            None
            if source_weight is None
            else torch.as_tensor(np.asarray(source_weight, dtype=np.float32).reshape(1, -1), dtype=torch.float32)
        )
        if support_mask is not None:
            resolved_support = torch.as_tensor(
                np.asarray(support_mask, dtype=np.float32).reshape(1, -1),
                dtype=torch.float32,
            )
        elif source_closed_mask is not None or source_boundary_confidence is not None:
            duration_for_support = _safe_array(source_duration_obs, dtype=np.float32)
            if duration_for_support is None:
                duration_for_support = np.exp(log_dur).astype(np.float32, copy=False)
            support_stats = summarize_global_rate_support(
                speech_mask=speech_tensor,
                valid_mask=valid_tensor,
                duration_obs=torch.as_tensor(duration_for_support.reshape(1, -1), dtype=torch.float32),
                drop_edge_runs=int(drop_edge_runs),
                closed_mask=(
                    None
                    if source_closed_mask is None
                    else torch.as_tensor(
                        np.asarray(source_closed_mask, dtype=np.float32).reshape(1, -1),
                        dtype=torch.float32,
                    )
                ),
                boundary_confidence=(
                    None
                    if source_boundary_confidence is None
                    else torch.as_tensor(
                        np.asarray(source_boundary_confidence, dtype=np.float32).reshape(1, -1),
                        dtype=torch.float32,
                    )
                ),
                min_boundary_confidence=min_boundary_confidence,
            )
            if is_softclean_global_rate_variant(g_variant):
                resolved_support = (speech_tensor > 0.5) & (
                    valid_tensor > 0.5 if valid_tensor is not None else torch.ones_like(speech_tensor, dtype=torch.bool)
                )
                resolved_weight = _resolve_analysis_softclean_weight(
                    variant=g_variant,
                    speech_tensor=speech_tensor,
                    valid_tensor=valid_tensor,
                    explicit_weight=resolved_weight,
                    closed_mask=source_closed_mask,
                    boundary_confidence=source_boundary_confidence,
                )
            else:
                resolved_support = support_stats.support_mask.float()
        value = compute_global_rate(
            log_dur=log_dur_tensor,
            speech_mask=speech_tensor,
            valid_mask=valid_tensor,
            variant=g_variant,
            trim_ratio=float(g_trim_ratio),
            drop_edge_runs=int(drop_edge_runs),
            weight=resolved_weight,
            unit_ids=None
            if source_unit_ids is None
            else torch.as_tensor(np.asarray(source_unit_ids, dtype=np.int64).reshape(1, -1), dtype=torch.long),
            unit_prior=None
            if source_unit_prior is None
            else torch.as_tensor(np.asarray(source_unit_prior, dtype=np.float32), dtype=torch.float32),
            support_mask=resolved_support,
        )
        return _maybe_with_status(
            float(value.reshape(-1)[0].item()),
            "ok",
            name="compute_source_global_rate_for_analysis",
            return_status=return_status,
        )
    except Exception as exc:
        return _maybe_with_status(
            float("nan"),
            f"error:{type(exc).__name__}",
            name="compute_source_global_rate_for_analysis",
            return_status=return_status,
        )


def compute_speech_tempo_for_analysis(
    *,
    source_log_dur: Any | None = None,
    source_duration_obs: Any | None = None,
    source_speech_mask: Any,
    source_valid_mask: Any | None = None,
    source_closed_mask: Any | None = None,
    source_boundary_confidence: Any | None = None,
    min_boundary_confidence: float | None = None,
    g_variant: str = "raw_median",
    source_weight: Any | None = None,
    source_unit_ids: Any | None = None,
    source_unit_prior: Any | None = None,
    g_trim_ratio: float = 0.2,
    drop_edge_runs: int = 0,
) -> float:
    global_rate = compute_source_global_rate_for_analysis(
        source_log_dur,
        source_duration_obs=source_duration_obs,
        source_speech_mask=source_speech_mask,
        source_valid_mask=source_valid_mask,
        source_closed_mask=source_closed_mask,
        source_boundary_confidence=source_boundary_confidence,
        min_boundary_confidence=min_boundary_confidence,
        g_variant=g_variant,
        source_weight=source_weight,
        source_unit_ids=source_unit_ids,
        source_unit_prior=source_unit_prior,
        g_trim_ratio=g_trim_ratio,
        drop_edge_runs=drop_edge_runs,
    )
    if not np.isfinite(global_rate):
        return float("nan")
    return float(np.exp(-float(global_rate)))


def compute_prefix_tempo(record: RhythmV3DebugRecord) -> np.ndarray | None:
    derived = derive_record(record)
    return None if derived.source_rate_seq is None else derived.source_rate_seq.copy()


def compute_a_i(global_rate: float, prefix_tempo: Any, *, tau_g: float | None = None) -> np.ndarray | None:
    prefix = _safe_array(prefix_tempo, dtype=np.float32)
    if prefix is None or not np.isfinite(global_rate):
        return None
    analytic = float(global_rate) - prefix
    if tau_g is not None:
        analytic = np.clip(analytic, -float(tau_g), float(tau_g))
    return analytic.astype(np.float32, copy=False)


def compute_b_star(
    z_star: Any,
    a_i: Any,
    *,
    speech_mask: Any,
    weight: Any | None = None,
) -> float:
    z = _safe_array(z_star, dtype=np.float32)
    a = _safe_array(a_i, dtype=np.float32)
    speech = _safe_bool_array(speech_mask)
    if z is None or a is None or speech is None:
        return float("nan")
    width = min(int(z.shape[0]), int(a.shape[0]), int(speech.shape[0]))
    if width <= 0:
        return float("nan")
    valid = speech[:width]
    if not bool(np.any(valid)):
        return float("nan")
    weights = None if weight is None else _safe_array(weight, dtype=np.float32)[:width]
    return weighted_median(z[:width][valid] - a[:width][valid], None if weights is None else weights[valid])


def _resolve_gate0_drop_reason(
    *,
    g_ref: float,
    g_src: float,
    c_star: float,
    g_compute_status: str,
    g_src_compute_status: str,
    g_domain_valid: float,
    speech_ratio: float,
    min_speech_ratio: float,
    has_explicit_prompt_speech_mask: bool = True,
    require_explicit_prompt_speech_mask: bool = False,
) -> str:
    if bool(require_explicit_prompt_speech_mask) and not bool(has_explicit_prompt_speech_mask):
        return "g_ref:missing:source_speech_mask"
    if not _status_is_ok(g_compute_status):
        return f"g_ref:{g_compute_status}"
    if not _status_is_ok(g_src_compute_status):
        return f"g_src:{g_src_compute_status}"
    if not np.isfinite(c_star):
        return "missing:c_star"
    if not np.isfinite(g_ref):
        return "g_ref:nonfinite"
    if not np.isfinite(g_src):
        return "g_src:nonfinite"
    if not np.isfinite(g_domain_valid) or float(g_domain_valid) <= 0.5:
        return "g_domain:invalid"
    if not np.isfinite(speech_ratio) or float(speech_ratio) < float(min_speech_ratio):
        return "ref:speech_ratio"
    delta_g = float(g_ref - g_src)
    if not np.isfinite(delta_g):
        return "delta_g:nonfinite"
    return "ok"


def _resolve_record_ids(record: RhythmV3DebugRecord, index: int) -> dict[str, Any]:
    item_name = record.item_name or f"item_{index:06d}"
    pair_id = _meta(record, "pair_id", "rhythm_pair_group_id", default=item_name)
    return {
        "item_name": item_name,
        "sample_id": _as_str(_meta(record, "sample_id", default=item_name)),
        "utt_id": _as_str(_meta(record, "utt_id", "src_id", "sample_id", default=item_name)),
        "pair_id": _as_str(pair_id, default=item_name),
        "split": _as_str(record.split),
        "chunk_scheme": _as_str(_meta(record, "chunk_scheme", "replay_scheme", default="default")),
        "commit_lag": int(_as_float(_meta(record, "commit_lag", default=0.0), default=0.0)),
        "eval_mode": _as_str(_meta(record, "eval_mode", "rhythm_v3_eval_mode", default="")),
    }


def _resolve_prefix_ratio(record: RhythmV3DebugRecord, *, full_record: RhythmV3DebugRecord | None = None) -> float:
    explicit = _meta(record, "prefix_ratio")
    if explicit is not None:
        return float(np.clip(_as_float(explicit, default=0.0), 0.0, 1.0))
    commit = _safe_array(record.commit_mask, dtype=np.float32)
    if commit is not None and commit.size > 0:
        return float(np.clip(commit.mean(), 0.0, 1.0))
    if full_record is not None:
        full_len = 0 if full_record.source_duration_obs is None else int(np.asarray(full_record.source_duration_obs).reshape(-1).shape[0])
        curr_len = 0 if record.source_duration_obs is None else int(np.asarray(record.source_duration_obs).reshape(-1).shape[0])
        if full_len > 0:
            return float(np.clip(float(curr_len) / float(full_len), 0.0, 1.0))
    return 0.0


def _resolve_boundary_type(
    record: RhythmV3DebugRecord,
    run_idx: int,
    *,
    boundary_threshold: float = DEFAULT_REVIEW_BOUNDARY_THRESHOLD,
) -> str:
    total = 0 if record.source_duration_obs is None else int(np.asarray(record.source_duration_obs).reshape(-1).shape[0])
    if total > 0 and run_idx >= (total - 1):
        return "sentence_final"
    sep = _safe_array(record.sep_mask, dtype=np.float32)
    cue = _safe_array(record.source_boundary_cue, dtype=np.float32)
    sep_flag = False if sep is None or run_idx >= sep.shape[0] else bool(sep[run_idx] > 0.5)
    cue_flag = False if cue is None or run_idx >= cue.shape[0] else bool(cue[run_idx] >= float(boundary_threshold))
    if sep_flag or cue_flag:
        return "phrase_boundary"
    return "phrase_internal"


def build_run_table(
    items: str | Path | RhythmV3DebugRecord | Mapping[str, Any] | Sequence[Any],
    *,
    silence_tau: float = DEFAULT_REVIEW_SILENCE_TAU,
    boundary_threshold: float = DEFAULT_REVIEW_BOUNDARY_THRESHOLD,
) -> pd.DataFrame:
    records = ensure_debug_records(items)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        ids = _resolve_record_ids(record, index)
        if record.source_duration_obs is None:
            continue
        derived = derive_record(record, silence_tau=float(silence_tau))
        n_src = _safe_array(record.source_duration_obs, dtype=np.float32)
        n_star = _safe_array(
            record.unit_duration_proj_raw_tgt
            if record.unit_duration_proj_raw_tgt is not None
            else record.unit_duration_tgt,
            dtype=np.float32,
        )
        omega = _safe_array(record.unit_confidence_tgt, dtype=np.float32)
        if omega is None:
            omega = _safe_array(record.unit_confidence_local_tgt, dtype=np.float32)
        if omega is None:
            omega = _safe_array(record.unit_confidence_coarse_tgt, dtype=np.float32)
        if omega is None:
            omega = np.ones_like(n_src, dtype=np.float32)
        p_i = None if derived.source_rate_seq is None else derived.source_rate_seq
        a_i = None if derived.analytic_shift is None else derived.analytic_shift
        r_star = None if derived.oracle_local is None else derived.oracle_local
        r_pred = None if derived.prediction_local is None else derived.prediction_local
        z_star = None if derived.target_logstretch is None else derived.target_logstretch
        n_pred_cont = _safe_array(record.unit_duration_raw, dtype=np.float32)
        k_pred_disc = _safe_array(record.unit_duration_exec, dtype=np.float32)
        source_units = _safe_array(record.source_content_units, dtype=np.int64)
        speech_mask = derived.speech_mask
        silence_mask = derived.silence_mask
        sealed_mask = _safe_bool_array(record.sealed_mask)
        commit_mask = _safe_bool_array(record.commit_mask)
        run_stability = _safe_array(record.source_run_stability, dtype=np.float32)
        for run_idx in range(int(n_src.shape[0])):
            rows.append(
                {
                    **ids,
                    "run_idx": int(run_idx),
                    "unit_id": -1 if source_units is None or run_idx >= source_units.shape[0] else int(source_units[run_idx]),
                    "run_type": "sil" if bool(silence_mask[run_idx] > 0.5) else "sp",
                    "boundary_type": _resolve_boundary_type(record, run_idx, boundary_threshold=boundary_threshold),
                    "n_src": float(n_src[run_idx]),
                    "n_star": np.nan if n_star is None or run_idx >= n_star.shape[0] else float(n_star[run_idx]),
                    "omega": np.nan if omega is None or run_idx >= omega.shape[0] else float(omega[run_idx]),
                    "p_i": np.nan if p_i is None or run_idx >= p_i.shape[0] else float(p_i[run_idx]),
                    "g": np.nan if derived.global_rate is None else float(derived.global_rate),
                    "a_i": np.nan if a_i is None or run_idx >= a_i.shape[0] else float(a_i[run_idx]),
                    "z_star": np.nan if z_star is None or run_idx >= z_star.shape[0] else float(z_star[run_idx]),
                    "z_star_raw": np.nan if z_star is None or run_idx >= z_star.shape[0] else float(z_star[run_idx]),
                    "b_star": np.nan if derived.oracle_bias is None else float(derived.oracle_bias),
                    "b_pred": np.nan if derived.prediction_bias is None else float(derived.prediction_bias),
                    "r_star": np.nan if r_star is None or run_idx >= r_star.shape[0] else float(r_star[run_idx]),
                    "r_pred": np.nan if r_pred is None or run_idx >= r_pred.shape[0] else float(r_pred[run_idx]),
                    "n_pred_cont": np.nan if n_pred_cont is None or run_idx >= n_pred_cont.shape[0] else float(n_pred_cont[run_idx]),
                    "k_pred_disc": np.nan if k_pred_disc is None or run_idx >= k_pred_disc.shape[0] else float(k_pred_disc[run_idx]),
                    "is_speech": float(speech_mask[run_idx] > 0.5),
                    "is_silence": float(silence_mask[run_idx] > 0.5),
                    "is_closed": 0.0 if sealed_mask is None or run_idx >= sealed_mask.shape[0] else float(sealed_mask[run_idx]),
                    "is_committed": 0.0 if commit_mask is None or run_idx >= commit_mask.shape[0] else float(commit_mask[run_idx]),
                    "run_stability": np.nan if run_stability is None or run_idx >= run_stability.shape[0] else float(run_stability[run_idx]),
                }
            )
    return pd.DataFrame(rows)


def build_ref_crop_table(
    items: str | Path | RhythmV3DebugRecord | Mapping[str, Any] | Sequence[Any],
    *,
    g_variant: str = "raw_median",
    g_trim_ratio: float = 0.2,
    drop_edge_runs: int = 0,
    unit_step_ms: float = DEFAULT_REVIEW_UNIT_STEP_MS,
    require_explicit_speech_mask: bool = False,
    min_speech_ratio: float = DEFAULT_GATE_MIN_SPEECH_RATIO,
    min_boundary_confidence: float | None = None,
) -> pd.DataFrame:
    records = ensure_debug_records(items)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        ids = _resolve_record_ids(record, index)
        derived = derive_record(record)
        prompt_duration = _safe_array(record.prompt_duration_obs, dtype=np.float32)
        prompt_valid = _safe_array(record.prompt_valid_mask, dtype=np.float32)
        prompt_speech_explicit = _safe_array(record.prompt_speech_mask, dtype=np.float32)
        prompt_speech = prompt_speech_explicit
        source_duration = _safe_array(record.source_duration_obs, dtype=np.float32)
        source_silence = _safe_array(record.source_silence_mask, dtype=np.float32)
        source_valid = _safe_array(record.unit_mask, dtype=np.float32)
        source_units = _safe_array(record.source_content_units, dtype=np.int64)
        prompt_units = _safe_array(record.prompt_content_units, dtype=np.int64)
        if (
            not bool(require_explicit_speech_mask)
            and prompt_speech is None
            and prompt_duration is not None
            and derived.prompt_speech_mask is not None
        ):
            prompt_speech = derived.prompt_speech_mask.astype(np.float32, copy=False)
        if record.global_rate is not None and not _record_prompt_domain_invalid(record):
            g_ref = float(record.global_rate)
            g_compute_status = "ok" if np.isfinite(g_ref) else "invalid:record.global_rate"
        else:
            if _record_prompt_domain_invalid(record):
                g_ref = float("nan")
                g_compute_status = "invalid:prompt_domain"
            else:
                g_ref, g_compute_status = compute_source_global_rate_for_analysis(
                    source_duration_obs=prompt_duration,
                    source_speech_mask=prompt_speech,
                    source_valid_mask=prompt_valid,
                    source_weight=(
                        _safe_array(getattr(record, "prompt_global_weight", None), dtype=np.float32)
                        if g_variant in {"weighted_median", "softclean_wmed", "softclean_wtmean"}
                        else None
                    ),
                    source_closed_mask=getattr(record, "prompt_closed_mask", None),
                    source_boundary_confidence=getattr(record, "prompt_boundary_confidence", None),
                    min_boundary_confidence=min_boundary_confidence,
                    g_variant=g_variant,
                    g_trim_ratio=g_trim_ratio,
                    drop_edge_runs=drop_edge_runs,
                    source_unit_ids=prompt_units,
                    require_explicit_speech_mask=require_explicit_speech_mask,
                    return_status=True,
                )
        source_speech = (
            np.ones_like(source_duration, dtype=np.float32)
            if source_duration is not None and source_silence is None
            else (1.0 - np.clip(source_silence, 0.0, 1.0) if source_silence is not None else None)
        )
        g_src, g_src_compute_status = compute_source_global_rate_for_analysis(
            source_duration_obs=source_duration,
            source_speech_mask=source_speech,
            source_valid_mask=np.ones_like(source_duration, dtype=np.float32) if source_duration is not None and source_valid is None else source_valid,
            source_weight=(
                _safe_array(record.source_run_stability, dtype=np.float32)
                if g_variant in {"weighted_median", "softclean_wmed", "softclean_wtmean"}
                else None
            ),
            source_closed_mask=getattr(record, "sealed_mask", None),
            source_boundary_confidence=getattr(record, "source_boundary_cue", None),
            min_boundary_confidence=min_boundary_confidence,
            g_variant=g_variant,
            g_trim_ratio=g_trim_ratio,
            drop_edge_runs=drop_edge_runs,
            source_unit_ids=source_units,
            return_status=True,
        )
        g_src_prefix_mean = float("nan")
        g_src_prefix_final = float("nan")
        if derived.source_rate_seq is not None and derived.speech_mask is not None:
            speech_valid = derived.speech_mask > 0.5
            if bool(np.any(speech_valid)):
                g_src_prefix_mean = float(np.nanmean(derived.source_rate_seq[speech_valid]))
                speech_idx = np.flatnonzero(speech_valid)
                if speech_idx.size > 0:
                    g_src_prefix_final = float(derived.source_rate_seq[int(speech_idx[-1])])
        ref_len_sec = _as_float(
            _meta(record, "ref_len_sec", default=None),
            default=(
                float("nan")
                if prompt_duration is None
                else float(prompt_duration.sum()) * float(unit_step_ms) / 1000.0
            ),
        )
        domain_stats = _resolve_prompt_domain_stats(
            prompt_duration=prompt_duration,
            prompt_speech=prompt_speech_explicit if bool(require_explicit_speech_mask) else prompt_speech,
            speech_ratio_hint=_meta(record, "prompt_speech_ratio", "speech_ratio", default=None),
            support_count_hint=_meta(record, "g_support_count", default=None),
            valid_count_hint=_meta(record, "g_valid_count", default=None),
            g_valid_hint=_meta(record, "g_valid_support", "g_valid", default=None),
            g_domain_valid_hint=_meta(record, "g_domain_valid", default=None),
            min_speech_ratio_hint=_meta(record, "g_min_speech_ratio", default=None),
            default_min_speech_ratio=float(min_speech_ratio),
        )
        speech_ratio = _as_float(
            _meta(record, "speech_ratio", default=None),
            default=domain_stats["prompt_speech_ratio"],
        )
        src_prompt_id = _as_str(_meta(record, "src_prompt_id", "prompt_id", default=ids["item_name"]))
        ref_prompt_id = _as_str(_meta(record, "ref_prompt_id", "reference_prompt_id", "ref_item_name", default=""))
        tgt_prompt_id = _as_str(_meta(record, "tgt_prompt_id", "target_prompt_id", "paired_target_item_name", default=""))
        same_text_reference = _meta(record, "same_text_reference", "same_text", default=None)
        if same_text_reference is None:
            ref_sig = _meta(record, "reference_text_signature", default=None)
            src_sig = _meta(record, "source_text_signature", default=None)
            if ref_sig is not None and src_sig is not None:
                same_text_reference = int(ref_sig == src_sig)
            elif src_prompt_id and ref_prompt_id:
                same_text_reference = int(src_prompt_id == ref_prompt_id)
        same_text_target = _meta(record, "same_text_target", default=None)
        if same_text_target is None:
            tgt_sig = _meta(record, "paired_target_text_signature", default=None)
            src_sig = _meta(record, "source_text_signature", default=None)
            if tgt_sig is not None and src_sig is not None:
                same_text_target = int(tgt_sig == src_sig)
            elif src_prompt_id and tgt_prompt_id:
                same_text_target = int(src_prompt_id == tgt_prompt_id)
        speech_target_stat = float("nan")
        analytic_target_stat = float("nan")
        if derived.target_logstretch is not None and derived.speech_mask is not None:
            valid = derived.speech_mask > 0.5
            if bool(np.any(valid)):
                weight = _safe_array(record.unit_confidence_tgt, dtype=np.float32)
                if weight is None:
                    weight = _safe_array(record.unit_confidence_coarse_tgt, dtype=np.float32)
                if weight is None:
                    weight = np.ones_like(derived.target_logstretch, dtype=np.float32)
                speech_target_stat = weighted_median(derived.target_logstretch[valid], weight[valid])
                if derived.analytic_shift is not None:
                    analytic_target_stat = weighted_median(derived.analytic_shift[valid], weight[valid])
        src_spk = _as_str(_meta(record, "src_spk", "source_speaker", default=""))
        if not src_spk:
            src_spk = _infer_speaker_id(record.item_name)
        ref_spk = _as_str(_meta(record, "ref_spk", "reference_speaker", "reference_item_speaker", default=""))
        if not ref_spk:
            ref_spk = _infer_speaker_id(_meta(record, "ref_item_name", "ref_prompt_id", default=""))
        tgt_spk = _as_str(_meta(record, "tgt_spk", "target_speaker", default=""))
        if not tgt_spk:
            tgt_spk = _infer_speaker_id(_meta(record, "paired_target_item_name", "tgt_prompt_id", default=""))
        same_speaker_reference = _meta(record, "same_speaker_reference", "same_speaker", default=None)
        if same_speaker_reference is None and src_spk and ref_spk:
            same_speaker_reference = int(src_spk == ref_spk)
        same_speaker_target = _meta(record, "same_speaker_target", default=None)
        if same_speaker_target is None and src_spk and tgt_spk:
            same_speaker_target = int(src_spk == tgt_spk)
        c_star = np.nan if derived.oracle_bias is None else float(derived.oracle_bias)
        delta_g = float(g_ref - g_src) if np.isfinite(g_ref) and np.isfinite(g_src) else float("nan")
        delta_g_ref_minus_src_prefix = (
            float(g_ref - g_src_prefix_mean)
            if np.isfinite(g_ref) and np.isfinite(g_src_prefix_mean)
            else float("nan")
        )
        delta_g_ref_minus_src_prefix_final = (
            float(g_ref - g_src_prefix_final)
            if np.isfinite(g_ref) and np.isfinite(g_src_prefix_final)
            else float("nan")
        )
        gate0_drop_reason = _resolve_gate0_drop_reason(
            g_ref=g_ref,
            g_src=g_src,
            c_star=c_star,
            g_compute_status=g_compute_status,
            g_src_compute_status=g_src_compute_status,
            g_domain_valid=domain_stats["g_domain_valid"],
            speech_ratio=speech_ratio,
            min_speech_ratio=domain_stats["g_min_speech_ratio"],
            has_explicit_prompt_speech_mask=prompt_speech_explicit is not None,
            require_explicit_prompt_speech_mask=require_explicit_speech_mask,
        )
        rows.append(
            {
                **ids,
                "crop_id": _as_str(_meta(record, "crop_id", default=f"crop_{index:06d}")),
                "g_variant": g_variant,
                "src_spk": src_spk,
                "tgt_spk": tgt_spk,
                "ref_spk": ref_spk,
                "src_prompt_id": src_prompt_id,
                "tgt_prompt_id": tgt_prompt_id,
                "ref_prompt_id": ref_prompt_id,
                "ref_len_sec": ref_len_sec,
                "speech_ratio": speech_ratio,
                "same_text": np.nan if same_text_reference is None else float(same_text_reference),
                "same_text_reference": np.nan if same_text_reference is None else float(same_text_reference),
                "same_text_target": np.nan if same_text_target is None else float(same_text_target),
                "same_speaker": np.nan if same_speaker_reference is None else float(same_speaker_reference),
                "same_speaker_reference": np.nan if same_speaker_reference is None else float(same_speaker_reference),
                "same_speaker_target": np.nan if same_speaker_target is None else float(same_speaker_target),
                "ref_condition": _as_str(_meta(record, "ref_condition", default="")),
                "lexical_mismatch": _meta(record, "lexical_mismatch", default=np.nan),
                "g_crop": g_ref,
                "g_compute_status": g_compute_status,
                "g_src": g_src,
                "g_src_compute_status": g_src_compute_status,
                "g_src_utt": g_src,
                "g_src_prefix_mean": g_src_prefix_mean,
                "g_src_prefix_final": g_src_prefix_final,
                "delta_g": delta_g,
                "delta_g_ref_minus_src_utt": delta_g,
                "delta_g_ref_minus_src_prefix": delta_g_ref_minus_src_prefix,
                "delta_g_ref_minus_src_prefix_final": delta_g_ref_minus_src_prefix_final,
                "delta_g_ref_minus_src_utt_neg": (-delta_g if np.isfinite(delta_g) else float("nan")),
                "delta_g_ref_minus_src_prefix_neg": (
                    -delta_g_ref_minus_src_prefix
                    if np.isfinite(delta_g_ref_minus_src_prefix)
                    else float("nan")
                ),
                "delta_g_ref_minus_src_prefix_final_neg": (
                    -delta_g_ref_minus_src_prefix_final
                    if np.isfinite(delta_g_ref_minus_src_prefix_final)
                    else float("nan")
                ),
                "c_star": c_star,
                "abar_sp_star": analytic_target_stat,
                "zbar_sp_star": speech_target_stat,
                "g_valid_support": domain_stats["g_valid_support"],
                "g_domain_valid": domain_stats["g_domain_valid"],
                "prompt_speech_ratio": domain_stats["prompt_speech_ratio"],
                "g_min_speech_ratio": domain_stats["g_min_speech_ratio"],
                "prompt_g_invalid_no_speech": _as_float(_meta(record, "prompt_g_invalid_no_speech", default=np.nan)),
                "prompt_g_invalid_low_speech_ratio": _as_float(
                    _meta(record, "prompt_g_invalid_low_speech_ratio", default=np.nan)
                ),
                "prompt_g_invalid_ref_len": _as_float(_meta(record, "prompt_g_invalid_ref_len", default=np.nan)),
                "prompt_g_invalid_support": _as_float(_meta(record, "prompt_g_invalid_support", default=np.nan)),
                "prompt_g_invalid_clean": _as_float(_meta(record, "prompt_g_invalid_clean", default=np.nan)),
                "prompt_g_invalid_missing_closed": _as_float(
                    _meta(record, "prompt_g_invalid_missing_closed", default=np.nan)
                ),
                "prompt_g_invalid_missing_boundary": _as_float(
                    _meta(record, "prompt_g_invalid_missing_boundary", default=np.nan)
                ),
                "prompt_speech_mask_explicit": 1.0 if prompt_speech_explicit is not None else 0.0,
                "gate0_row_dropped": 0.0 if gate0_drop_reason == "ok" else 1.0,
                "gate0_drop_reason": gate0_drop_reason,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    group_size = frame.groupby("pair_id")["crop_id"].transform("count")
    longest = frame.groupby("pair_id")["ref_len_sec"].transform(lambda col: np.nanmax(np.asarray(col, dtype=np.float32)))
    frame["g_full"] = np.where(frame["ref_len_sec"].to_numpy() >= longest.to_numpy() - 1.0e-8, frame["g_crop"], np.nan)
    frame["g_full"] = frame.groupby("pair_id")["g_full"].transform(lambda col: col.ffill().bfill())
    frame["has_crop_comparison"] = group_size > 1
    frame["g_crop_abs_err"] = np.where(
        frame["has_crop_comparison"].to_numpy(),
        np.abs(frame["g_crop"].to_numpy(dtype=np.float32) - frame["g_full"].to_numpy(dtype=np.float32)),
        np.nan,
    )
    return frame


def build_prefix_replay_table(
    items: str | Path | RhythmV3DebugRecord | Mapping[str, Any] | Sequence[Any],
) -> pd.DataFrame:
    records = ensure_debug_records(items)
    grouped: dict[tuple[str, str, int], list[tuple[int, RhythmV3DebugRecord]]] = {}
    for index, record in enumerate(records):
        ids = _resolve_record_ids(record, index)
        grouped.setdefault((ids["utt_id"], ids["chunk_scheme"], ids["commit_lag"]), []).append((index, record))
    rows: list[dict[str, Any]] = []
    for (_, _, _), group in grouped.items():
        _, full_record = max(group, key=lambda item: _resolve_prefix_ratio(item[1]))
        full_source = _safe_array(full_record.source_duration_obs, dtype=np.float32)
        full_exec = _safe_array(full_record.unit_duration_exec, dtype=np.float32)
        full_cont = _safe_array(full_record.unit_duration_raw, dtype=np.float32)
        full_len = 0 if full_source is None else int(full_source.shape[0])
        for index, record in group:
            ids = _resolve_record_ids(record, index)
            prefix_ratio = _resolve_prefix_ratio(record, full_record=full_record)
            source = _safe_array(record.source_duration_obs, dtype=np.float32)
            if source is None:
                continue
            exec_disc = _safe_array(record.unit_duration_exec, dtype=np.float32)
            exec_cont = _safe_array(record.unit_duration_raw, dtype=np.float32)
            sealed = _safe_bool_array(record.sealed_mask)
            commit = _safe_bool_array(record.commit_mask)
            drift = _safe_array(record.prefix_unit_offset, dtype=np.float32)
            hit_pos = _safe_array(record.projector_budget_hit_pos, dtype=np.float32)
            hit_neg = _safe_array(record.projector_budget_hit_neg, dtype=np.float32)
            for run_idx in range(int(source.shape[0])):
                row_hit = 0.0
                if hit_pos is not None and run_idx < hit_pos.shape[0] and hit_pos[run_idx] > 0.5:
                    row_hit = 1.0
                if hit_neg is not None and run_idx < hit_neg.shape[0] and hit_neg[run_idx] > 0.5:
                    row_hit = 1.0
                rows.append(
                    {
                        **ids,
                        "prefix_ratio": float(prefix_ratio),
                        "run_idx": int(run_idx),
                        "is_observed": 1.0,
                        "is_closed": 0.0 if sealed is None or run_idx >= sealed.shape[0] else float(sealed[run_idx]),
                        "is_committed": 0.0 if commit is None or run_idx >= commit.shape[0] else float(commit[run_idx]),
                        "n_prefix": float(source[run_idx]),
                        "n_full": np.nan if full_source is None or run_idx >= full_source.shape[0] else float(full_source[run_idx]),
                        "n_prefix_runs": int(source.shape[0]),
                        "n_full_runs": int(full_len),
                        "n_pred_cont": np.nan if exec_cont is None or run_idx >= exec_cont.shape[0] else float(exec_cont[run_idx]),
                        "n_full_cont": np.nan if full_cont is None or run_idx >= full_cont.shape[0] else float(full_cont[run_idx]),
                        "k_pred_disc": np.nan if exec_disc is None or run_idx >= exec_disc.shape[0] else float(exec_disc[run_idx]),
                        "k_full_disc": np.nan if full_exec is None or run_idx >= full_exec.shape[0] else float(full_exec[run_idx]),
                        "budget_drift": np.nan if drift is None or run_idx >= drift.shape[0] else float(drift[run_idx]),
                        "budget_hit": row_hit,
                    }
                )
    return pd.DataFrame(rows)


def compute_run_stability(prefix_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if prefix_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    closed = prefix_df[(prefix_df["is_closed"] > 0.5) & prefix_df["n_full"].notna()].copy()
    if closed.empty:
        panel_a = pd.DataFrame(columns=["chunk_scheme", "prefix_ratio", "rewrite_rate", "exact_match_ratio", "mult_mae", "n_closed"])
    else:
        closed["rewrite"] = (closed["n_prefix"] != closed["n_full"]).astype(float)
        closed["exact_match"] = 1.0 - closed["rewrite"]
        closed["abs_err"] = (closed["n_prefix"] - closed["n_full"]).abs()
        panel_a = (
            closed.groupby(["chunk_scheme", "prefix_ratio"], as_index=False)
            .agg(
                rewrite_rate=("rewrite", "mean"),
                exact_match_ratio=("exact_match", "mean"),
                mult_mae=("abs_err", "mean"),
                n_closed=("run_idx", "count"),
            )
            .sort_values(["chunk_scheme", "prefix_ratio"])
        )
    panel_b = (
        prefix_df.groupby(["utt_id", "chunk_scheme", "prefix_ratio"], as_index=False)
        .agg(n_prefix_runs=("n_prefix_runs", "max"), n_full_runs=("n_full_runs", "max"))
    )
    if not panel_b.empty:
        panel_b["count_drift"] = (
            (panel_b["n_prefix_runs"] - panel_b["n_full_runs"]).abs()
            / panel_b["n_full_runs"].clip(lower=1.0)
        )
        panel_b = (
            panel_b.groupby(["chunk_scheme", "prefix_ratio"], as_index=False)
            .agg(
                count_drift=("count_drift", "median"),
                count_drift_p25=("count_drift", lambda s: float(np.nanpercentile(np.asarray(s, dtype=np.float32), 25.0))),
                count_drift_p75=("count_drift", lambda s: float(np.nanpercentile(np.asarray(s, dtype=np.float32), 75.0))),
            )
            .sort_values(["chunk_scheme", "prefix_ratio"])
        )
    return panel_a, panel_b


def summarize_global_cue_review(ref_crop_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if ref_crop_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    target_col = "zbar_sp_star" if "zbar_sp_star" in ref_crop_df.columns else "c_star"
    sliced_rows: list[dict[str, Any]] = []
    definitions = [
        (
            "same_text_reference",
            ref_crop_df["same_text_reference"].map(lambda x: "same_text" if float(x) > 0.5 else "cross_text")
            if "same_text_reference" in ref_crop_df
            else (
                ref_crop_df["same_text"].map(lambda x: "same_text" if float(x) > 0.5 else "cross_text")
                if "same_text" in ref_crop_df
                else None
            ),
        ),
        ("lexical_mismatch", pd.cut(pd.to_numeric(ref_crop_df["lexical_mismatch"], errors="coerce"), bins=[-np.inf, 0.1, 0.3, 0.6, np.inf], labels=["<=0.1", "0.1-0.3", "0.3-0.6", ">0.6"]) if "lexical_mismatch" in ref_crop_df else None),
        ("ref_len_sec", pd.cut(ref_crop_df["ref_len_sec"], bins=[0.0, 3.0, 5.0, 8.0, np.inf], labels=["<3s", "3-5s", "5-8s", ">=8s"])),
        ("speech_ratio", pd.cut(ref_crop_df["speech_ratio"], bins=[0.0, 0.4, 0.6, 0.8, 1.01], labels=["<0.4", "0.4-0.6", "0.6-0.8", ">=0.8"])),
    ]
    for slice_name, slice_values in definitions:
        if slice_values is None:
            continue
        work = ref_crop_df.copy()
        work["slice_value"] = slice_values.astype("object")
        for slice_value, group in work.groupby("slice_value", dropna=True):
            metric = tempo_explainability(group["delta_g"], group[target_col])
            decomp = tempo_explainability(group["delta_g"], group["c_star"])
            sliced_rows.append(
                {
                    "slice_name": slice_name,
                    "slice_value": slice_value,
                    "count": int(group[["delta_g", target_col]].dropna().shape[0]),
                    "spearman": float(metric["spearman"]),
                    "robust_slope": float(metric["robust_slope"]),
                    "r2_like": float(metric["r2_like"]),
                    "coarse_residual_spearman": float(decomp["spearman"]),
                    "coarse_residual_slope": float(decomp["robust_slope"]),
                }
            )
    crop_only = ref_crop_df[ref_crop_df["has_crop_comparison"] > 0].copy() if "has_crop_comparison" in ref_crop_df else ref_crop_df.iloc[0:0].copy()
    return crop_only, pd.DataFrame(sliced_rows)


def _piecewise_oracle_bias(sp_df: pd.DataFrame, segments: int) -> pd.Series:
    if sp_df.empty:
        return pd.Series(dtype=np.float32)
    seg = pd.qcut(sp_df["run_idx"].rank(method="first"), q=min(max(1, int(segments)), int(sp_df.shape[0])), labels=False, duplicates="drop")
    out = pd.Series(index=sp_df.index, dtype=np.float32)
    for _, group in sp_df.groupby(seg):
        bias = weighted_median(group["z_star"] - group["a_i"], group["omega"])
        out.loc[group.index] = float(bias)
    return out.astype(np.float32)


def _lf_ratio(values: Any, cutoff_ratio: float = 0.2) -> float:
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return float("nan")
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x)) ** 2
    if spec.size <= 0:
        return float("nan")
    k = max(1, int(np.ceil(spec.size * float(cutoff_ratio))))
    return float(spec[:k].sum() / max(float(spec.sum()), 1.0e-8))


def summarize_oracle_decomposition(
    run_df: pd.DataFrame,
    *,
    piecewise_segments: Sequence[int] = (2, 4),
    low_freq_cutoff_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if run_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    speech = run_df[(run_df["run_type"] == "sp") & run_df["z_star"].notna() & run_df["a_i"].notna()].copy()
    suff_rows: list[dict[str, Any]] = []
    lf_rows: list[dict[str, Any]] = []
    calib_rows: list[dict[str, Any]] = []
    for utt_id, group in speech.groupby("utt_id"):
        z = group["z_star"].to_numpy(dtype=np.float32)
        a = group["a_i"].to_numpy(dtype=np.float32)
        omega = np.clip(group["omega"].to_numpy(dtype=np.float32), 1.0e-6, None)
        b_star = weighted_median(z - a, omega)
        approximations: list[tuple[str, np.ndarray]] = [("analytic", a), ("scalar_coarse", a + float(b_star))]
        for segments in piecewise_segments:
            piecewise = _piecewise_oracle_bias(group, int(segments))
            approximations.append((f"piecewise_k{int(segments)}", a + piecewise.to_numpy(dtype=np.float32)))
        for name, pred in approximations:
            abs_err = np.abs(pred - z)
            suff_rows.append(
                {
                    "utt_id": utt_id,
                    "approximation": name,
                    "mae": float(abs_err.mean()),
                    "weighted_mae": float(np.sum(abs_err * omega) / max(float(omega.sum()), 1.0e-6)),
                }
            )
        residual = z - (a + float(b_star))
        lf_rows.append({"utt_id": utt_id, "lf_ratio": _lf_ratio(residual, cutoff_ratio=low_freq_cutoff_ratio)})
        b_pred = group["b_pred"].dropna()
        calib_rows.append(
            {
                "utt_id": utt_id,
                "b_star": float(b_star),
                "b_pred": float(b_pred.iloc[0]) if not b_pred.empty else float("nan"),
            }
        )
    return pd.DataFrame(suff_rows), pd.DataFrame(lf_rows), pd.DataFrame(calib_rows)


def build_silence_audit_table(
    run_df: pd.DataFrame,
    *,
    silence_tau: float = DEFAULT_REVIEW_SILENCE_TAU,
) -> pd.DataFrame:
    if run_df.empty:
        return pd.DataFrame()
    silence = run_df[(run_df["run_type"] == "sil") & run_df["z_star"].notna() & run_df["a_i"].notna()].copy()
    if silence.empty:
        return silence
    silence["z_pseudo"] = np.clip(
        silence["a_i"].to_numpy(dtype=np.float32) + silence["b_star"].fillna(0.0).to_numpy(dtype=np.float32),
        -float(silence_tau),
        float(silence_tau),
    )
    silence["err_to_pseudo"] = silence["z_star"].to_numpy(dtype=np.float32) - silence["z_pseudo"].to_numpy(dtype=np.float32)
    silence["abs_err_to_pseudo"] = np.abs(silence["err_to_pseudo"].to_numpy(dtype=np.float32))
    return silence


def summarize_silence_audit(silence_df: pd.DataFrame) -> pd.DataFrame:
    if silence_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for boundary_type, group in silence_df.groupby("boundary_type"):
        corr = group["z_star"].corr(group["z_pseudo"], method="spearman")
        rows.append(
            {
                "boundary_type": boundary_type,
                "count": int(group.shape[0]),
                "spearman": float(corr) if corr is not None else float("nan"),
                "mae": float(group["abs_err_to_pseudo"].mean()),
                "cond_var": float(np.var(group["z_star"].to_numpy(dtype=np.float32) - group["z_pseudo"].to_numpy(dtype=np.float32))),
            }
        )
    return pd.DataFrame(rows)


def compute_commit_metrics(prefix_df: pd.DataFrame) -> pd.DataFrame:
    if prefix_df.empty:
        return pd.DataFrame()
    committed = prefix_df[(prefix_df["is_committed"] > 0.5) & prefix_df["k_full_disc"].notna()].copy()
    if committed.empty:
        return pd.DataFrame()
    committed["rewrite"] = (committed["k_pred_disc"] != committed["k_full_disc"]).astype(float)
    committed["abs_gap"] = (committed["k_pred_disc"] - committed["k_full_disc"]).abs()
    committed["rounding_err"] = (committed["k_pred_disc"] - committed["n_pred_cont"]).abs()
    return (
        committed.groupby(["commit_lag", "prefix_ratio"], as_index=False)
        .agg(
            rewrite_rate=("rewrite", "mean"),
            committed_gap=("abs_gap", "mean"),
            rounding_err=("rounding_err", "mean"),
            budget_drift=("budget_drift", lambda s: float(np.nanmedian(np.abs(np.asarray(s, dtype=np.float32))))),
            hit_rate=("budget_hit", "mean"),
        )
        .sort_values(["commit_lag", "prefix_ratio"])
    )


def build_prefix_silence_review_table(
    items: str | Path | RhythmV3DebugRecord | Mapping[str, Any] | Sequence[Any],
) -> pd.DataFrame:
    records = ensure_debug_records(items)
    grouped: dict[tuple[str, str], list[tuple[int, RhythmV3DebugRecord]]] = {}
    for index, record in enumerate(records):
        sample_id = _as_str(_meta(record, "sample_id", default=record.item_name or f"sample_{index:06d}"))
        eval_mode = _as_str(_meta(record, "eval_mode", "rhythm_v3_eval_mode", default=""))
        grouped.setdefault((sample_id, eval_mode), []).append((index, record))
    rows: list[dict[str, Any]] = []
    for (sample_id, eval_mode), group in grouped.items():
        sorted_group = sorted(group, key=lambda item: _resolve_prefix_ratio(item[1]))
        _, short_record = sorted_group[0]
        _, long_record = sorted_group[-1]
        short_ratio = _resolve_prefix_ratio(short_record, full_record=long_record)
        long_ratio = _resolve_prefix_ratio(long_record, full_record=long_record)
        short_logstretch, long_logstretch = _align_pair(short_record.unit_logstretch, long_record.unit_logstretch, dtype=np.float32)
        short_commit, long_commit = _align_pair(short_record.commit_mask, long_record.commit_mask, dtype=np.float32)
        commit_mask = None if short_commit is None or long_commit is None else np.minimum(short_commit, long_commit)
        discrepancy = float("nan")
        leakage = float("nan")
        if short_logstretch is not None and long_logstretch is not None and commit_mask is not None:
            discrepancy = float(prefix_discrepancy(short_logstretch, long_logstretch, commit_mask).item())
            short_derived = derive_record(short_record)
            long_derived = derive_record(long_record)
            short_speech, long_speech = _align_pair(short_derived.speech_mask, long_derived.speech_mask, dtype=np.float32)
            short_silence, long_silence = _align_pair(short_derived.silence_mask, long_derived.silence_mask, dtype=np.float32)
            if short_speech is not None and long_speech is not None and short_silence is not None and long_silence is not None:
                delta = long_logstretch - short_logstretch
                leakage = float(
                    silence_leakage(
                        delta,
                        np.minimum(short_speech, long_speech),
                        np.minimum(short_silence, long_silence),
                    ).item()
                )
        budget_hit = float(
            budget_hit_rate(
                long_record.projector_budget_hit_pos,
                long_record.projector_budget_hit_neg,
            ).item()
        )
        hit_pos = _safe_array(long_record.projector_budget_hit_pos, dtype=np.float32)
        hit_neg = _safe_array(long_record.projector_budget_hit_neg, dtype=np.float32)
        hit_pos_rate = float(np.mean(hit_pos > 0.5)) if hit_pos is not None and hit_pos.size > 0 else float("nan")
        hit_neg_rate = float(np.mean(hit_neg > 0.5)) if hit_neg is not None and hit_neg.size > 0 else float("nan")
        drift = float(cumulative_drift(long_record.prefix_unit_offset).item())
        rows.append(
            {
                "sample_id": sample_id,
                "eval_mode": eval_mode,
                "prefix_ratio": long_ratio,
                "short_prefix_ratio": short_ratio,
                "prefix_discrepancy": discrepancy,
                "budget_hit": budget_hit,
                "budget_hit_rate": budget_hit,
                "budget_hit_pos_rate": hit_pos_rate,
                "budget_hit_neg_rate": hit_neg_rate,
                "cumulative_drift": drift,
                "silence_leakage": leakage,
            }
        )
    return pd.DataFrame(rows)


def build_monotonicity_table(
    items: str | Path | RhythmV3DebugRecord | Mapping[str, Any] | Sequence[Any],
    *,
    g_variant: str = "raw_median",
    g_trim_ratio: float = 0.2,
    drop_edge_runs: int = 0,
    require_explicit_speech_mask: bool = False,
    min_boundary_confidence: float | None = None,
) -> pd.DataFrame:
    records = ensure_debug_records(items)
    grouped: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = {}
    for index, record in enumerate(records):
        meta = dict(record.metadata or {})
        src_id = _as_str(meta.get("src_id", record.item_name or f"src_{index:06d}"))
        sample_id = _as_str(meta.get("sample_id", record.item_name or src_id))
        pair_id = _as_str(meta.get("pair_id", meta.get("rhythm_pair_group_id", sample_id)))
        ref_condition = _normalize_ref_condition(meta.get("ref_condition", ""))
        ref_bin = _as_str(meta.get("ref_bin", meta.get("tempo_bin", ""))).strip().lower()
        if ref_bin not in {"slow", "mid", "fast"}:
            continue
        derived = derive_record(record)
        eval_mode = _as_str(meta.get("eval_mode", meta.get("rhythm_v3_eval_mode", "")))
        source_weight = (
            _safe_array(record.source_run_stability, dtype=np.float32)
            if g_variant in {"weighted_median", "softclean_wmed", "softclean_wtmean"}
            else None
        )
        prompt_weight = (
            _safe_array(getattr(record, "prompt_global_weight", None), dtype=np.float32)
            if g_variant in {"weighted_median", "softclean_wmed", "softclean_wtmean"}
            else None
        )
        tempo_src = compute_speech_tempo_for_analysis(
            source_duration_obs=record.source_duration_obs,
            source_speech_mask=derived.speech_mask,
            source_valid_mask=record.unit_mask,
            source_weight=source_weight,
            source_closed_mask=record.sealed_mask,
            source_boundary_confidence=record.source_boundary_cue,
            min_boundary_confidence=min_boundary_confidence,
            g_variant=g_variant,
            g_trim_ratio=g_trim_ratio,
            drop_edge_runs=drop_edge_runs,
            source_unit_ids=record.source_content_units,
        )
        tempo_out = compute_speech_tempo_for_analysis(
            source_duration_obs=record.unit_duration_exec,
            source_speech_mask=derived.speech_mask,
            source_valid_mask=record.unit_mask,
            source_weight=source_weight,
            source_closed_mask=record.sealed_mask,
            source_boundary_confidence=record.source_boundary_cue,
            min_boundary_confidence=min_boundary_confidence,
            g_variant=g_variant,
            g_trim_ratio=g_trim_ratio,
            drop_edge_runs=drop_edge_runs,
            source_unit_ids=record.source_content_units,
        )
        tempo_ref = compute_speech_tempo_for_analysis(
            source_duration_obs=record.prompt_duration_obs,
            source_speech_mask=(
                record.prompt_speech_mask
                if record.prompt_speech_mask is not None
                else (None if bool(require_explicit_speech_mask) else derived.prompt_speech_mask)
            ),
            source_valid_mask=record.prompt_valid_mask,
            source_weight=prompt_weight,
            source_closed_mask=record.prompt_closed_mask,
            source_boundary_confidence=record.prompt_boundary_confidence,
            min_boundary_confidence=min_boundary_confidence,
            g_variant=g_variant,
            g_trim_ratio=g_trim_ratio,
            drop_edge_runs=drop_edge_runs,
            source_unit_ids=record.prompt_content_units,
        )
        g_ref = float("nan") if _record_prompt_domain_invalid(record) else float(derived.global_rate) if derived.global_rate is not None else float("nan")
        g_src_prefix_mean = float("nan")
        g_src_prefix_final = float("nan")
        if derived.source_rate_seq is not None and derived.speech_mask is not None:
            speech_valid = derived.speech_mask > 0.5
            if bool(np.any(speech_valid)):
                g_src_prefix_mean = float(np.nanmean(derived.source_rate_seq[speech_valid]))
                speech_idx = np.flatnonzero(speech_valid)
                if speech_idx.size > 0:
                    g_src_prefix_final = float(derived.source_rate_seq[int(speech_idx[-1])])
        g_src_utt = compute_source_global_rate_for_analysis(
            source_duration_obs=record.source_duration_obs,
            source_speech_mask=derived.speech_mask,
            source_valid_mask=record.unit_mask,
            source_weight=(
                _safe_array(record.source_run_stability, dtype=np.float32)
                if g_variant in {"weighted_median", "softclean_wmed", "softclean_wtmean"}
                else None
            ),
            source_closed_mask=record.sealed_mask,
            source_boundary_confidence=record.source_boundary_cue,
            min_boundary_confidence=min_boundary_confidence,
            g_variant=g_variant,
            g_trim_ratio=g_trim_ratio,
            drop_edge_runs=drop_edge_runs,
            source_unit_ids=record.source_content_units,
        )
        grouped.setdefault((src_id, eval_mode, ref_condition, pair_id), {})[ref_bin] = {
            "sample_id": sample_id,
            "pair_id": pair_id,
            "tempo_out": tempo_out,
            "tempo_src": tempo_src,
            "tempo_ref": tempo_ref,
            "g_ref": g_ref,
            "g_src_utt": g_src_utt,
            "g_src_prefix_mean": g_src_prefix_mean,
            "g_src_prefix_final": g_src_prefix_final,
            "g_domain_valid": _as_float(
                meta.get("g_domain_valid", 1.0 if np.isfinite(g_ref) else float("nan"))
            ),
            "item_name": record.item_name or src_id,
            "ref_condition": ref_condition,
            "same_text_reference": _as_float(meta.get("same_text_reference", meta.get("same_text", float("nan")))),
            "same_speaker_reference": _as_float(
                meta.get(
                    "same_speaker_reference",
                    meta.get("same_speaker", float("nan")),
                )
            ),
            "prompt_speech_mask_explicit": 1.0 if record.prompt_speech_mask is not None else 0.0,
        }
    rows: list[dict[str, Any]] = []
    for (src_id, eval_mode, ref_condition, pair_id), bins in grouped.items():
        mono = float("nan")
        anti_mono = float("nan")
        tie_rate = float("nan")
        if all(name in bins for name in ("slow", "mid", "fast")):
            triplet_valid = all(
                np.isfinite(float(bins[name]["g_domain_valid"])) and float(bins[name]["g_domain_valid"]) > 0.5
                for name in ("slow", "mid", "fast")
            )
            if triplet_valid:
                mono = float(
                    tempo_monotonicity(
                        [bins["slow"]["tempo_out"]],
                        [bins["mid"]["tempo_out"]],
                        [bins["fast"]["tempo_out"]],
                    ).item()
                )
                anti_mono = float(
                    tempo_monotonicity(
                        [bins["slow"]["tempo_out"]],
                        [bins["mid"]["tempo_out"]],
                        [bins["fast"]["tempo_out"]],
                        increasing=False,
                    ).item()
                )
                tie_rate = float(
                    tempo_tie_rate(
                        [bins["slow"]["tempo_out"]],
                        [bins["mid"]["tempo_out"]],
                        [bins["fast"]["tempo_out"]],
                    ).item()
                )
        for ref_bin, payload in sorted(bins.items()):
            delta_g = (
                float(payload["g_ref"] - payload["g_src_utt"])
                if np.isfinite(payload["g_ref"]) and np.isfinite(payload["g_src_utt"])
                else float("nan")
            )
            delta_g_prefix = (
                float(payload["g_ref"] - payload["g_src_prefix_mean"])
                if np.isfinite(payload["g_ref"]) and np.isfinite(payload["g_src_prefix_mean"])
                else float("nan")
            )
            delta_g_prefix_final = (
                float(payload["g_ref"] - payload["g_src_prefix_final"])
                if np.isfinite(payload["g_ref"]) and np.isfinite(payload["g_src_prefix_final"])
                else float("nan")
            )
            rows.append(
                {
                    "src_id": src_id,
                    "sample_id": payload["sample_id"],
                    "pair_id": payload["pair_id"],
                    "ref_bin": ref_bin,
                    "tempo_out": payload["tempo_out"],
                    "tempo_src": payload["tempo_src"],
                    "tempo_delta": (
                        float(payload["tempo_out"] - payload["tempo_src"])
                        if np.isfinite(payload["tempo_out"]) and np.isfinite(payload["tempo_src"])
                        else float("nan")
                    ),
                    "tempo_ref": payload["tempo_ref"],
                    "g_ref": payload["g_ref"],
                    "g_src_utt": payload["g_src_utt"],
                    "g_src_prefix_mean": payload["g_src_prefix_mean"],
                    "g_src_prefix_final": payload["g_src_prefix_final"],
                    "g_domain_valid": payload["g_domain_valid"],
                    "delta_g": delta_g,
                    "delta_g_ref_minus_src_utt": delta_g,
                    "delta_g_ref_minus_src_prefix": delta_g_prefix,
                    "delta_g_ref_minus_src_prefix_final": delta_g_prefix_final,
                    "delta_g_ref_minus_src_utt_neg": (-delta_g if np.isfinite(delta_g) else float("nan")),
                    "delta_g_ref_minus_src_prefix_neg": (-delta_g_prefix if np.isfinite(delta_g_prefix) else float("nan")),
                    "delta_g_ref_minus_src_prefix_final_neg": (
                        -delta_g_prefix_final if np.isfinite(delta_g_prefix_final) else float("nan")
                    ),
                    "eval_mode": eval_mode,
                    "triplet_id": f"{src_id}|{eval_mode}|{ref_condition}|{pair_id}",
                    "ref_condition": ref_condition,
                    "same_text_reference": payload["same_text_reference"],
                    "same_speaker_reference": payload["same_speaker_reference"],
                    "mono_triplet_ok": mono,
                    "anti_mono_triplet_ok": anti_mono,
                    "tempo_tie_triplet": tie_rate,
                    "item_name": payload["item_name"],
                    "prompt_speech_mask_explicit": payload["prompt_speech_mask_explicit"],
                }
            )
    return pd.DataFrame(rows)


def summarize_falsification_ladder(
    ref_crop_df: pd.DataFrame,
    monotonicity_df: pd.DataFrame,
    prefix_silence_df: pd.DataFrame,
) -> pd.DataFrame:
    eval_modes: set[str] = set()
    for frame in (ref_crop_df, monotonicity_df, prefix_silence_df):
        if not frame.empty and "eval_mode" in frame:
            eval_modes.update(_as_str(mode) for mode in frame["eval_mode"].dropna().tolist())
    if not eval_modes:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for eval_mode in sorted(eval_modes):
        crop_mode = ref_crop_df[ref_crop_df["eval_mode"] == eval_mode].copy() if not ref_crop_df.empty and "eval_mode" in ref_crop_df else pd.DataFrame()
        mono_mode = monotonicity_df[monotonicity_df["eval_mode"] == eval_mode].copy() if not monotonicity_df.empty and "eval_mode" in monotonicity_df else pd.DataFrame()
        prefix_mode = prefix_silence_df[prefix_silence_df["eval_mode"] == eval_mode].copy() if not prefix_silence_df.empty and "eval_mode" in prefix_silence_df else pd.DataFrame()

        signal_explain = tempo_explainability(crop_mode.get("delta_g", []), crop_mode.get("zbar_sp_star", [])) if not crop_mode.empty else {
            "spearman": float("nan"),
            "robust_slope": float("nan"),
            "r2_like": float("nan"),
        }
        analytic_explain = tempo_explainability(crop_mode.get("delta_g", []), crop_mode.get("abar_sp_star", [])) if not crop_mode.empty else {
            "spearman": float("nan"),
            "robust_slope": float("nan"),
            "r2_like": float("nan"),
        }
        decomp_explain = tempo_explainability(crop_mode.get("delta_g", []), crop_mode.get("c_star", [])) if not crop_mode.empty else {
            "spearman": float("nan"),
            "robust_slope": float("nan"),
            "r2_like": float("nan"),
        }
        prefix_signal_explain = tempo_explainability(
            crop_mode.get("delta_g_ref_minus_src_prefix", []),
            crop_mode.get("zbar_sp_star", []),
        ) if not crop_mode.empty else {
            "spearman": float("nan"),
            "robust_slope": float("nan"),
            "r2_like": float("nan"),
        }
        gap = float("nan")
        if not crop_mode.empty and "same_text_reference" in crop_mode:
            same = crop_mode[crop_mode["same_text_reference"] > 0.5]
            cross = crop_mode[crop_mode["same_text_reference"] <= 0.5]
            if not same.empty and not cross.empty:
                same_metric = tempo_explainability(same["delta_g"], same["zbar_sp_star"])
                cross_metric = tempo_explainability(cross["delta_g"], cross["zbar_sp_star"])
                gap = float(same_text_gap(same_metric["robust_slope"], cross_metric["robust_slope"]).item())

        mono_rate = float("nan")
        anti_mono_rate = float("nan")
        tie_rate = float("nan")
        tempo_transfer = {
            "spearman": float("nan"),
            "robust_slope": float("nan"),
            "r2_like": float("nan"),
        }
        negative_control_gap = float("nan")
        real_reference_count = 0
        negative_control_count = 0
        invalid_g_rate = float("nan")
        if not mono_mode.empty:
            triplet_keys = ["triplet_id"] if "triplet_id" in mono_mode.columns else ["src_id", "eval_mode", "pair_id", "ref_condition"]
            unique_triplets = mono_mode.drop_duplicates(subset=[key for key in triplet_keys if key in mono_mode.columns])
            mono_rate = float(np.nanmean(unique_triplets["mono_triplet_ok"].to_numpy(dtype=np.float32)))
            if "anti_mono_triplet_ok" in unique_triplets:
                anti_mono_rate = float(np.nanmean(unique_triplets["anti_mono_triplet_ok"].to_numpy(dtype=np.float32)))
            if "tempo_tie_triplet" in unique_triplets:
                tie_rate = float(np.nanmean(unique_triplets["tempo_tie_triplet"].to_numpy(dtype=np.float32)))
            tempo_transfer = transfer_slope(mono_mode.get("delta_g", []), mono_mode.get("tempo_delta", []))
            negative_control_gap, real_reference_count, negative_control_count = _compute_negative_control_gap(
                mono_mode,
                x_col="delta_g",
                y_col="tempo_delta",
                metric_fn=transfer_slope,
            )
            if "g_domain_valid" in mono_mode.columns:
                domain_values = pd.to_numeric(mono_mode["g_domain_valid"], errors="coerce").to_numpy(dtype=np.float32)
                domain_values = domain_values[np.isfinite(domain_values)]
                if domain_values.size > 0:
                    invalid_g_rate = float(np.mean(domain_values < 0.5))
        elif not crop_mode.empty:
            crop_target_col = "zbar_sp_star" if "zbar_sp_star" in crop_mode.columns else "c_star"
            negative_control_gap, real_reference_count, negative_control_count = _compute_negative_control_gap(
                crop_mode,
                x_col="delta_g",
                y_col=crop_target_col,
                metric_fn=tempo_explainability,
            )
            if "g_domain_valid" in crop_mode.columns:
                domain_values = pd.to_numeric(crop_mode["g_domain_valid"], errors="coerce").to_numpy(dtype=np.float32)
                domain_values = domain_values[np.isfinite(domain_values)]
                if domain_values.size > 0:
                    invalid_g_rate = float(np.mean(domain_values < 0.5))

        rows.append(
            {
                "eval_mode": eval_mode,
                "signal_explainability_spearman": float(signal_explain["spearman"]),
                "signal_explainability_slope": float(signal_explain["robust_slope"]),
                "signal_explainability_r2_like": float(signal_explain["r2_like"]),
                "analytic_signal_spearman": float(analytic_explain["spearman"]),
                "analytic_signal_slope": float(analytic_explain["robust_slope"]),
                "analytic_signal_r2_like": float(analytic_explain["r2_like"]),
                "coarse_residual_spearman": float(decomp_explain["spearman"]),
                "coarse_residual_slope": float(decomp_explain["robust_slope"]),
                "coarse_residual_r2_like": float(decomp_explain["r2_like"]),
                "prefix_signal_explainability_spearman": float(prefix_signal_explain["spearman"]),
                "prefix_signal_explainability_slope": float(prefix_signal_explain["robust_slope"]),
                "prefix_signal_explainability_r2_like": float(prefix_signal_explain["r2_like"]),
                "explainability_spearman": float(decomp_explain["spearman"]),
                "explainability_slope": float(decomp_explain["robust_slope"]),
                "explainability_r2_like": float(decomp_explain["r2_like"]),
                "monotonicity_rate": mono_rate,
                "anti_monotonicity_rate": anti_mono_rate,
                "tempo_tie_rate": tie_rate,
                "tempo_transfer_slope": float(tempo_transfer["robust_slope"]),
                "tempo_transfer_spearman": float(tempo_transfer["spearman"]),
                "invalid_g_rate": invalid_g_rate,
                "negative_control_gap": negative_control_gap,
                "same_text_gap": gap,
                "silence_leakage": float(np.nanmean(prefix_mode["silence_leakage"].to_numpy(dtype=np.float32))) if not prefix_mode.empty else float("nan"),
                "prefix_discrepancy": float(np.nanmean(prefix_mode["prefix_discrepancy"].to_numpy(dtype=np.float32))) if not prefix_mode.empty else float("nan"),
                "budget_hit_rate": float(np.nanmean(prefix_mode["budget_hit_rate"].to_numpy(dtype=np.float32))) if not prefix_mode.empty and "budget_hit_rate" in prefix_mode else float("nan"),
                "cumulative_drift": float(np.nanmean(prefix_mode["cumulative_drift"].to_numpy(dtype=np.float32))) if not prefix_mode.empty else float("nan"),
                "n_ref_crops": int(crop_mode.shape[0]),
                "n_triplets": (
                    0
                    if mono_mode.empty
                    else int(
                        mono_mode["triplet_id"].nunique()
                        if "triplet_id" in mono_mode.columns
                        else mono_mode.drop_duplicates(
                            subset=[key for key in ("src_id", "eval_mode", "pair_id", "ref_condition") if key in mono_mode.columns]
                        ).shape[0]
                    )
                ),
                "n_prefix_samples": int(prefix_mode.shape[0]),
                "n_real_references": int(real_reference_count),
                "n_negative_controls": int(negative_control_count),
            }
        )
    return pd.DataFrame(rows)


def _build_record_summary_table(
    records: Sequence[RhythmV3DebugRecord],
    *,
    unit_step_ms: float = DEFAULT_REVIEW_UNIT_STEP_MS,
    local_rate_decay: float = 0.95,
    silence_tau: float = DEFAULT_REVIEW_SILENCE_TAU,
    g_variant: str = "raw_median",
    g_trim_ratio: float = 0.2,
    drop_edge_runs: int = 0,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            record_summary(
                record,
                unit_step_ms=unit_step_ms,
                local_rate_decay=local_rate_decay,
                silence_tau=silence_tau,
                g_variant=g_variant,
                g_trim_ratio=g_trim_ratio,
                drop_edge_runs=drop_edge_runs,
            )
        )
    return pd.DataFrame(rows)


def build_local_residual_review_table(
    items: str | Path | RhythmV3DebugRecord | Mapping[str, Any] | Sequence[Any],
    *,
    unit_step_ms: float = DEFAULT_REVIEW_UNIT_STEP_MS,
    local_rate_decay: float = 0.95,
    silence_tau: float = DEFAULT_REVIEW_SILENCE_TAU,
    g_variant: str = "raw_median",
    g_trim_ratio: float = 0.2,
    drop_edge_runs: int = 0,
) -> pd.DataFrame:
    records = ensure_debug_records(items)
    summary_df = _build_record_summary_table(
        records,
        unit_step_ms=unit_step_ms,
        local_rate_decay=local_rate_decay,
        silence_tau=silence_tau,
        g_variant=g_variant,
        g_trim_ratio=g_trim_ratio,
        drop_edge_runs=drop_edge_runs,
    )
    if summary_df.empty:
        return pd.DataFrame()
    keep = [
        "sample_id",
        "utt_id",
        "pair_id",
        "item_name",
        "eval_mode",
        "ref_bin",
        "ref_condition",
        "split",
        "chunk_scheme",
        "commit_lag",
        "global_rate",
        "g_domain_valid",
        "coarse_scalar_raw",
        "coarse_target_abs_err",
        "local_residual_abs_mean",
        "residual_gate_mean",
        "residual_target_corr",
        "residual_target_slope",
        "residual_target_count",
        "residual_bias_share",
        "local_silence_delta_share",
        "speech_mae",
        "speech_weighted_mae",
        "silence_mae",
        "silence_leakage",
        "prefix_discrepancy",
        "budget_hit_pos_rate",
        "budget_hit_neg_rate",
        "cumulative_drift",
    ]
    available = [column for column in keep if column in summary_df.columns]
    return summary_df[available].copy()


def build_gate3_local_table(
    items: str | Path | RhythmV3DebugRecord | Mapping[str, Any] | Sequence[Any],
    *,
    silence_tau: float = DEFAULT_REVIEW_SILENCE_TAU,
    boundary_threshold: float = DEFAULT_REVIEW_BOUNDARY_THRESHOLD,
    unit_step_ms: float = DEFAULT_REVIEW_UNIT_STEP_MS,
    local_rate_decay: float = 0.95,
    g_variant: str = "raw_median",
    g_trim_ratio: float = 0.2,
    drop_edge_runs: int = 0,
) -> pd.DataFrame:
    records = ensure_debug_records(items)
    run_df = build_run_table(
        records,
        silence_tau=silence_tau,
        boundary_threshold=boundary_threshold,
    )
    if run_df.empty:
        return pd.DataFrame()
    summary_df = _build_record_summary_table(
        records,
        unit_step_ms=unit_step_ms,
        local_rate_decay=local_rate_decay,
        silence_tau=silence_tau,
        g_variant=g_variant,
        g_trim_ratio=g_trim_ratio,
        drop_edge_runs=drop_edge_runs,
    )
    work = run_df.copy()
    work = work[
        (work["run_type"] == "sp")
        & (pd.to_numeric(work["is_committed"], errors="coerce") > 0.5)
        & work["r_star"].notna()
        & work["r_pred"].notna()
        & work["omega"].notna()
    ].copy()
    if work.empty:
        return pd.DataFrame()
    group_cols = [
        column
        for column in ("sample_id", "utt_id", "pair_id", "item_name", "eval_mode")
        if column in work.columns
    ]
    summary_key_cols = [column for column in group_cols if column in summary_df.columns]
    summary_lookup: dict[tuple[Any, ...], dict[str, Any]] = {}
    if summary_key_cols:
        for row in summary_df.to_dict(orient="records"):
            key = tuple(row.get(column) for column in summary_key_cols)
            summary_lookup.setdefault(key, row)
    rows: list[dict[str, Any]] = []
    optional_cols = ["split", "chunk_scheme", "commit_lag"]
    extra_summary_cols = [
        "ref_bin",
        "ref_condition",
        "global_rate",
        "g_domain_valid",
        "coarse_scalar_raw",
        "coarse_target_abs_err",
        "residual_gate_mean",
        "residual_target_corr",
        "residual_target_slope",
        "residual_target_count",
        "residual_bias_share",
        "local_silence_delta_share",
        "speech_mae",
        "speech_weighted_mae",
        "silence_mae",
        "silence_leakage",
        "prefix_discrepancy",
        "budget_hit_pos_rate",
        "budget_hit_neg_rate",
        "cumulative_drift",
    ]
    for keys, group in work.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = {column: value for column, value in zip(group_cols, keys)}
        omega = np.clip(group["omega"].to_numpy(dtype=np.float32), 1.0e-6, None)
        r_star = group["r_star"].to_numpy(dtype=np.float32)
        r_pred = group["r_pred"].to_numpy(dtype=np.float32)
        err = r_pred - r_star
        abs_err = np.abs(err)
        huber = np.where(abs_err < 0.25, 0.5 * (err ** 2) / 0.25, abs_err - 0.125)
        weight_sum = max(float(np.sum(omega)), 1.0e-6)
        valid = np.isfinite(r_star) & np.isfinite(r_pred)
        corr = float("nan")
        if int(np.sum(valid)) >= 3:
            corr = float(pd.Series(r_star[valid]).corr(pd.Series(r_pred[valid]), method="spearman"))
        row = {
            **key_map,
            "local_residual_mae": float(np.sum(abs_err * omega) / weight_sum),
            "local_residual_huber": float(np.sum(huber * omega) / weight_sum),
            "local_residual_corr": corr,
            "local_residual_mean": float(np.sum(r_pred * omega) / weight_sum),
            "local_residual_abs_mean": float(np.sum(np.abs(r_pred) * omega) / weight_sum),
            "committed_speech_runs": int(group.shape[0]),
            "speech_weight_sum": weight_sum,
            "b_star": float(group["b_star"].dropna().iloc[0]) if group["b_star"].notna().any() else float("nan"),
            "b_pred": float(group["b_pred"].dropna().iloc[0]) if group["b_pred"].notna().any() else float("nan"),
        }
        for column in optional_cols:
            row[column] = (
                group[column].dropna().iloc[0]
                if column in group.columns and group[column].notna().any()
                else (0 if column == "commit_lag" else "")
            )
        if summary_key_cols:
            summary_key = tuple(key_map.get(column) for column in summary_key_cols)
            summary_row = summary_lookup.get(summary_key)
            if summary_row is not None:
                for column in extra_summary_cols:
                    if column in summary_row:
                        row[column] = summary_row.get(column)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_monotonicity_intervention(monotonicity_df: pd.DataFrame):
    import matplotlib.pyplot as plt

    if monotonicity_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4.8), constrained_layout=True)
        ax.text(0.5, 0.5, "No monotonicity intervention data", ha="center", va="center")
        ax.set_axis_off()
        fig.suptitle("Gate Figure: Monotonicity Intervention", fontsize=14)
        return fig

    modes = sorted(str(mode) for mode in monotonicity_df["eval_mode"].dropna().unique().tolist()) or ["unknown"]
    fig, axes = plt.subplots(1, len(modes), figsize=(5.4 * len(modes), 4.8), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    order = ["slow", "mid", "fast"]
    for ax, eval_mode in zip(axes, modes):
        work = monotonicity_df[monotonicity_df["eval_mode"] == eval_mode].copy()
        if work.empty:
            ax.text(0.5, 0.5, f"No {eval_mode} data", ha="center", va="center")
            ax.set_axis_off()
            continue
        series = [work.loc[work["ref_bin"] == ref_bin, "tempo_out"].dropna().to_numpy(dtype=np.float32) for ref_bin in order]
        if any(arr.size > 0 for arr in series):
            ax.boxplot(series, labels=order, showmeans=True)
        paired = work.pivot_table(
            index="triplet_id" if "triplet_id" in work.columns else "src_id",
            columns="ref_bin",
            values="tempo_out",
            aggfunc="first",
        )
        for _, row in paired.iterrows():
            if all(name in row.index and np.isfinite(row[name]) for name in order):
                ax.plot(np.arange(1, 4), [row["slow"], row["mid"], row["fast"]], color="#4C78A8", alpha=0.18, linewidth=1.0)
        triplet_keys = ["triplet_id"] if "triplet_id" in work.columns else ["src_id", "eval_mode", "pair_id", "ref_condition"]
        unique_triplets = work.drop_duplicates(subset=[key for key in triplet_keys if key in work.columns])
        mono_rate = float(np.nanmean(unique_triplets["mono_triplet_ok"].to_numpy(dtype=np.float32)))
        ax.set_title(f"{eval_mode}\nmono={mono_rate:.3f}" if np.isfinite(mono_rate) else eval_mode)
        ax.set_ylabel("tempo_out")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Gate Figure: Monotonicity Intervention", fontsize=14)
    return fig


def plot_prefix_silence_stability(prefix_silence_df: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), constrained_layout=True)
    metrics = [
        ("silence_leakage", "silence_leakage"),
        ("prefix_discrepancy", "prefix_discrepancy"),
        ("budget_hit_rate", "budget_hit_rate"),
        ("cumulative_drift", "cumulative_drift"),
    ]
    if prefix_silence_df.empty:
        for ax, (_, title) in zip(axes.reshape(-1), metrics):
            ax.text(0.5, 0.5, "No prefix/silence stability data", ha="center", va="center")
            ax.set_title(title)
            ax.set_axis_off()
        fig.suptitle("Gate Figure: Prefix + Silence Stability", fontsize=14)
        return fig

    modes = sorted(str(mode) for mode in prefix_silence_df["eval_mode"].dropna().unique().tolist()) or ["unknown"]
    for ax, (column, title) in zip(axes.reshape(-1), metrics):
        series = [
            prefix_silence_df.loc[prefix_silence_df["eval_mode"] == eval_mode, column].dropna().to_numpy(dtype=np.float32)
            for eval_mode in modes
        ]
        if any(arr.size > 0 for arr in series):
            ax.boxplot(series, labels=modes, showmeans=True)
        else:
            ax.text(0.5, 0.5, f"No {column} data", ha="center", va="center")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Gate Figure: Prefix + Silence Stability", fontsize=14)
    return fig


def plot_falsification_ladder(ladder_df: pd.DataFrame):
    import matplotlib.pyplot as plt

    metrics = [
        ("explainability_slope", "slope"),
        ("monotonicity_rate", "mono"),
        ("negative_control_gap", "neg-ctrl gap"),
        ("same_text_gap", "same-text gap"),
        ("silence_leakage", "silence leak"),
        ("prefix_discrepancy", "prefix disc"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(21, 4.8), constrained_layout=True)
    if ladder_df.empty:
        for ax, (_, title) in zip(np.asarray(axes).reshape(-1), metrics):
            ax.text(0.5, 0.5, "No ladder summary", ha="center", va="center")
            ax.set_title(title)
            ax.set_axis_off()
        fig.suptitle("Gate Figure: Analytic / Coarse / Learned Ladder", fontsize=14)
        return fig

    x = np.arange(ladder_df.shape[0])
    labels = ladder_df["eval_mode"].astype(str).tolist()
    for ax, (column, title) in zip(np.asarray(axes).reshape(-1), metrics):
        ax.bar(x, ladder_df[column].to_numpy(dtype=np.float32), color="#4C78A8")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Gate Figure: Analytic / Coarse / Learned Ladder", fontsize=14)
    return fig


def plot_run_lattice_stability(prefix_df: pd.DataFrame):
    import matplotlib.pyplot as plt

    panel_a, panel_b = compute_run_stability(prefix_df)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    ax_a, ax_b, ax_c = axes
    if panel_a.empty:
        ax_a.text(0.5, 0.5, "No closed-prefix stability data", ha="center", va="center")
        ax_c.text(0.5, 0.5, "No closed-prefix multiplicity data", ha="center", va="center")
    else:
        for chunk_scheme, group in panel_a.groupby("chunk_scheme"):
            group = group.sort_values("prefix_ratio")
            ax_a.plot(group["prefix_ratio"], group["rewrite_rate"], marker="o", label=chunk_scheme)
            ax_c.plot(group["prefix_ratio"], group["exact_match_ratio"], marker="o", label=f"{chunk_scheme}: exact")
        ax_a.set_title("Panel A: Closed-Run Rewrite Rate")
        ax_a.set_xlabel("Prefix coverage")
        ax_a.set_ylabel("rewrite_rate")
        ax_a.set_ylim(0.0, 1.0)
        ax_a.grid(alpha=0.25)
        ax_a.legend(loc="best")
        ax_c2 = ax_c.twinx()
        for chunk_scheme, group in panel_a.groupby("chunk_scheme"):
            group = group.sort_values("prefix_ratio")
            ax_c2.plot(group["prefix_ratio"], group["mult_mae"], linestyle="--", marker="s", label=f"{chunk_scheme}: MAE")
        ax_c.set_title("Panel C: Exact Match / Multiplicity MAE")
        ax_c.set_xlabel("Prefix coverage")
        ax_c.set_ylabel("exact_match_ratio")
        ax_c.set_ylim(0.0, 1.0)
        ax_c2.set_ylabel("mult_mae")
        ax_c.grid(alpha=0.25)
    if panel_b.empty:
        ax_b.text(0.5, 0.5, "No run-count drift data", ha="center", va="center")
    else:
        for chunk_scheme, group in panel_b.groupby("chunk_scheme"):
            group = group.sort_values("prefix_ratio")
            ax_b.plot(group["prefix_ratio"], group["count_drift"], marker="o", label=chunk_scheme)
            ax_b.fill_between(group["prefix_ratio"], group["count_drift_p25"], group["count_drift_p75"], alpha=0.15)
        ax_b.set_title("Panel B: Run-Count Drift")
        ax_b.set_xlabel("Prefix coverage")
        ax_b.set_ylabel("count_drift")
        ax_b.grid(alpha=0.25)
        ax_b.legend(loc="best")
    fig.suptitle("Figure A: Run-Lattice Stability", fontsize=14)
    return fig


def plot_global_cue_survival(ref_crop_df: pd.DataFrame):
    import matplotlib.pyplot as plt

    crop_only, sliced = summarize_global_cue_review(ref_crop_df)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    ax_a, ax_b, ax_c = axes
    if crop_only.empty:
        ax_a.text(0.5, 0.5, "Need multi-crop groups to audit g stability", ha="center", va="center")
    else:
        work = crop_only.copy()
        work["len_bin"] = pd.cut(work["ref_len_sec"], bins=[0.0, 3.0, 5.0, 8.0, np.inf], labels=["<3s", "3-5s", "5-8s", ">=8s"])
        work["speech_bin"] = pd.cut(work["speech_ratio"], bins=[0.0, 0.4, 0.6, 0.8, 1.01], labels=["<0.4", "0.4-0.6", "0.6-0.8", ">=0.8"])
        pivot = work.pivot_table(index="speech_bin", columns="len_bin", values="g_crop_abs_err", aggfunc="median")
        img = ax_a.imshow(pivot.to_numpy(dtype=np.float32), aspect="auto", origin="lower", cmap="magma")
        ax_a.set_xticks(range(len(pivot.columns)))
        ax_a.set_xticklabels([str(x) for x in pivot.columns], rotation=25)
        ax_a.set_yticks(range(len(pivot.index)))
        ax_a.set_yticklabels([str(x) for x in pivot.index])
        ax_a.set_title("Panel A: |g_crop - g_full|")
        ax_a.set_xlabel("ref_len_sec")
        ax_a.set_ylabel("speech_ratio")
        fig.colorbar(img, ax=ax_a, fraction=0.046, pad=0.04)
    info = ref_crop_df[["delta_g", "c_star"]].dropna()
    if "zbar_sp_star" in ref_crop_df.columns:
        info = ref_crop_df[["delta_g", "zbar_sp_star"]].dropna().rename(columns={"zbar_sp_star": "target_signal"})
    else:
        info = ref_crop_df[["delta_g", "c_star"]].dropna().rename(columns={"c_star": "target_signal"})
    if info.empty:
        ax_b.text(0.5, 0.5, "No delta_g / signal pairs", ha="center", va="center")
    else:
        hb = ax_b.hexbin(info["delta_g"], info["target_signal"], gridsize=24, cmap="viridis", mincnt=1)
        metric = tempo_explainability(info["delta_g"], info["target_signal"])
        slope = float(metric["robust_slope"])
        x = np.linspace(float(info["delta_g"].min()), float(info["delta_g"].max()), 128, dtype=np.float32)
        y = slope * (x - float(np.median(info["delta_g"]))) + float(np.median(info["target_signal"]))
        ax_b.plot(x, y, color="white", linewidth=2.0, label=f"slope={slope:.3f}")
        ax_b.set_title("Panel B: delta_g vs total speech target")
        ax_b.set_xlabel("delta_g")
        ax_b.set_ylabel("zbar_sp_star")
        ax_b.legend(loc="best")
        fig.colorbar(hb, ax=ax_b, fraction=0.046, pad=0.04)
    if sliced.empty:
        ax_c.text(0.5, 0.5, "No slice summary", ha="center", va="center")
    else:
        labels = [f"{row.slice_name}:{row.slice_value}" for row in sliced.itertuples(index=False)]
        ax_c.bar(np.arange(len(labels)), sliced["spearman"].to_numpy(dtype=np.float32), color="#4C78A8")
        ax_c.set_xticks(np.arange(len(labels)))
        ax_c.set_xticklabels(labels, rotation=45, ha="right")
        ax_c.set_title("Panel C: Contamination Slices")
        ax_c.set_ylabel("Spearman(delta_g, total speech target)")
        ax_c.grid(axis="y", alpha=0.25)
    fig.suptitle("Figure B: Global Cue Survival", fontsize=14)
    return fig


def plot_oracle_decomposition(run_df: pd.DataFrame):
    import matplotlib.pyplot as plt

    suff, lf, calib = summarize_oracle_decomposition(run_df)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    ax_a, ax_b, ax_c = axes
    if suff.empty:
        ax_a.text(0.5, 0.5, "No speech oracle decomposition data", ha="center", va="center")
    else:
        summary = suff.groupby("approximation", as_index=False)["weighted_mae"].mean()
        ax_a.bar(summary["approximation"], summary["weighted_mae"], color=["#72B7B2", "#F58518", "#E45756", "#54A24B"][: summary.shape[0]])
        ax_a.set_title("Panel A: Coarse Sufficiency")
        ax_a.set_ylabel("weighted_mae")
        ax_a.tick_params(axis="x", rotation=20)
    if lf.empty:
        ax_b.text(0.5, 0.5, "No residual spectrum data", ha="center", va="center")
    else:
        ax_b.violinplot(lf["lf_ratio"].dropna().to_numpy(dtype=np.float32), showmeans=True, showextrema=True)
        ax_b.set_title("Panel B: Residual Low-Frequency Ratio")
        ax_b.set_xticks([1])
        ax_b.set_xticklabels(["scalar coarse residual"])
        ax_b.set_ylabel("LF ratio")
    if calib.empty:
        ax_c.text(0.5, 0.5, "No coarse calibration data", ha="center", va="center")
    else:
        valid = calib[["b_star", "b_pred"]].dropna()
        if valid.empty:
            ax_c.text(0.5, 0.5, "No valid b* / b pairs", ha="center", va="center")
        else:
            hb = ax_c.hexbin(valid["b_star"], valid["b_pred"], gridsize=20, cmap="plasma", mincnt=1)
            low = float(min(valid["b_star"].min(), valid["b_pred"].min()))
            high = float(max(valid["b_star"].max(), valid["b_pred"].max()))
            ax_c.plot([low, high], [low, high], color="white", linestyle="--", linewidth=1.5)
            ax_c.set_title("Panel C: b_pred vs b*")
            ax_c.set_xlabel("b_star")
            ax_c.set_ylabel("b_pred")
            fig.colorbar(hb, ax=ax_c, fraction=0.046, pad=0.04)
    fig.suptitle("Figure C: Oracle Decomposition", fontsize=14)
    return fig


def plot_silence_theory_audit(run_df: pd.DataFrame, *, silence_tau: float = DEFAULT_REVIEW_SILENCE_TAU):
    import matplotlib.pyplot as plt

    silence_df = build_silence_audit_table(run_df, silence_tau=silence_tau)
    summary = summarize_silence_audit(silence_df)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    ax_a, ax_b, ax_c = axes
    if silence_df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No silence audit data", ha="center", va="center")
    else:
        grouped = [group["err_to_pseudo"].dropna().to_numpy(dtype=np.float32) for _, group in silence_df.groupby("boundary_type")]
        labels = [str(name) for name, _ in silence_df.groupby("boundary_type")]
        if grouped:
            ax_a.violinplot(grouped, showmeans=True, showextrema=True)
            ax_a.set_xticks(np.arange(1, len(labels) + 1))
            ax_a.set_xticklabels(labels, rotation=20)
        ax_a.set_title("Panel A: raw - pseudo by boundary")
        ax_a.set_ylabel("err_to_pseudo")
        hb = ax_b.hexbin(silence_df["z_pseudo"], silence_df["z_star"], gridsize=22, cmap="cividis", mincnt=1)
        ax_b.set_title("Panel B: raw vs pseudo")
        ax_b.set_xlabel("z_pseudo")
        ax_b.set_ylabel("z_star_raw")
        fig.colorbar(hb, ax=ax_b, fraction=0.046, pad=0.04)
        if summary.empty:
            ax_c.text(0.5, 0.5, "No silence summary", ha="center", va="center")
        else:
            x = np.arange(summary.shape[0])
            ax_c.bar(x, summary["spearman"], color="#4C78A8", label="spearman")
            ax_c2 = ax_c.twinx()
            ax_c2.plot(x, summary["mae"], color="#E45756", marker="o", label="mae")
            ax_c.set_xticks(x)
            ax_c.set_xticklabels(summary["boundary_type"], rotation=20)
            ax_c.set_title("Panel C: Boundary-conditioned summary")
            ax_c.set_ylabel("spearman")
            ax_c2.set_ylabel("mae")
            ax_c.grid(axis="y", alpha=0.25)
    fig.suptitle("Figure D: Silence Theory Audit", fontsize=14)
    return fig


def plot_online_commit_semantics(prefix_df: pd.DataFrame):
    import matplotlib.pyplot as plt

    summary = compute_commit_metrics(prefix_df)
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.reshape(-1)
    if summary.empty:
        for ax in (ax_a, ax_b, ax_c, ax_d):
            ax.text(0.5, 0.5, "No committed-prefix replay data", ha="center", va="center")
        fig.suptitle("Figure E: Online Commit Semantics", fontsize=14)
        return fig
    ax_d2 = ax_d.twinx()
    for commit_lag, group in summary.groupby("commit_lag"):
        group = group.sort_values("prefix_ratio")
        label = f"lag={int(commit_lag)}"
        ax_a.plot(group["prefix_ratio"], group["rewrite_rate"], marker="o", label=label)
        ax_b.plot(group["prefix_ratio"], group["committed_gap"], marker="o", label=label)
        ax_c.plot(group["prefix_ratio"], group["budget_drift"], marker="o", label=label)
        ax_d.plot(group["prefix_ratio"], group["hit_rate"], marker="o", label=label)
        ax_d2.plot(group["prefix_ratio"], group["rounding_err"], linestyle="--", marker="s", alpha=0.7)
    ax_a.set_title("Panel A: committed rewrite rate")
    ax_b.set_title("Panel B: short vs long committed gap")
    ax_c.set_title("Panel C: budget drift")
    ax_d.set_title("Panel D: budget hit / rounding err")
    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.set_xlabel("prefix_ratio")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    ax_a.set_ylabel("rewrite_rate")
    ax_b.set_ylabel("committed_gap")
    ax_c.set_ylabel("|budget_drift| median")
    ax_d.set_ylabel("budget_hit_rate")
    ax_d2.set_ylabel("rounding_err")
    fig.suptitle("Figure E: Online Commit Semantics", fontsize=14)
    return fig


def save_review_figure_bundle(
    items: str | Path | RhythmV3DebugRecord | Mapping[str, Any] | Sequence[Any],
    *,
    output_dir: str | Path,
    g_variant: str = "raw_median",
    g_trim_ratio: float = 0.2,
    drop_edge_runs: int = 0,
    silence_tau: float = DEFAULT_REVIEW_SILENCE_TAU,
    boundary_threshold: float = DEFAULT_REVIEW_BOUNDARY_THRESHOLD,
) -> dict[str, Path]:
    records = ensure_debug_records(items)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_df = build_run_table(records, silence_tau=silence_tau, boundary_threshold=boundary_threshold)
    ref_crop_df = build_ref_crop_table(records, g_variant=g_variant, g_trim_ratio=g_trim_ratio, drop_edge_runs=drop_edge_runs)
    prefix_df = build_prefix_replay_table(records)
    monotonicity_df = build_monotonicity_table(records, g_variant=g_variant, g_trim_ratio=g_trim_ratio, drop_edge_runs=drop_edge_runs)
    prefix_silence_df = build_prefix_silence_review_table(records)
    local_residual_df = build_local_residual_review_table(
        records,
        local_rate_decay=0.95,
        silence_tau=silence_tau,
        g_variant=g_variant,
        g_trim_ratio=g_trim_ratio,
        drop_edge_runs=drop_edge_runs,
    )
    gate3_local_df = build_gate3_local_table(
        records,
        silence_tau=silence_tau,
        boundary_threshold=boundary_threshold,
        unit_step_ms=DEFAULT_REVIEW_UNIT_STEP_MS,
        local_rate_decay=0.95,
        g_variant=g_variant,
        g_trim_ratio=g_trim_ratio,
        drop_edge_runs=drop_edge_runs,
    )
    paths = {
        "run_table": out_dir / "run_table.csv",
        "ref_crop_table": out_dir / "ref_crop_table.csv",
        "prefix_replay_table": out_dir / "prefix_replay_table.csv",
        "monotonicity_table": out_dir / "monotonicity_table.csv",
        "prefix_silence_review_table": out_dir / "prefix_silence_review_table.csv",
        "local_residual_review_table": out_dir / "local_residual_review_table.csv",
        "gate3_local_table": out_dir / "gate3_local_table.csv",
    }
    run_df.to_csv(paths["run_table"], index=False)
    ref_crop_df.to_csv(paths["ref_crop_table"], index=False)
    prefix_df.to_csv(paths["prefix_replay_table"], index=False)
    monotonicity_df.to_csv(paths["monotonicity_table"], index=False)
    prefix_silence_df.to_csv(paths["prefix_silence_review_table"], index=False)
    local_residual_df.to_csv(paths["local_residual_review_table"], index=False)
    gate3_local_df.to_csv(paths["gate3_local_table"], index=False)
    figures = {
        "fig_a_run_lattice_stability.png": plot_run_lattice_stability(prefix_df),
        "fig_b_global_cue_survival.png": plot_global_cue_survival(ref_crop_df),
        "fig_c_oracle_decomposition.png": plot_oracle_decomposition(run_df),
        "fig_d_silence_theory_audit.png": plot_silence_theory_audit(run_df, silence_tau=silence_tau),
        "fig_e_online_commit_semantics.png": plot_online_commit_semantics(prefix_df),
    }
    for name, fig in figures.items():
        path = out_dir / name
        fig.savefig(path, dpi=180)
        fig.clf()
        paths[name] = path
    return paths


def save_validation_gate_bundle(
    items: str | Path | RhythmV3DebugRecord | Mapping[str, Any] | Sequence[Any],
    *,
    output_dir: str | Path,
    g_variant: str = "raw_median",
    g_trim_ratio: float = 0.2,
    drop_edge_runs: int = 0,
) -> dict[str, Path]:
    records = ensure_debug_records(items)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_crop_df = build_ref_crop_table(
        records,
        g_variant=g_variant,
        g_trim_ratio=g_trim_ratio,
        drop_edge_runs=drop_edge_runs,
        require_explicit_speech_mask=True,
    )
    monotonicity_df = build_monotonicity_table(
        records,
        g_variant=g_variant,
        g_trim_ratio=g_trim_ratio,
        drop_edge_runs=drop_edge_runs,
        require_explicit_speech_mask=True,
    )
    prefix_silence_df = build_prefix_silence_review_table(records)
    local_residual_df = build_local_residual_review_table(
        records,
        g_variant=g_variant,
        g_trim_ratio=g_trim_ratio,
        drop_edge_runs=drop_edge_runs,
    )
    gate3_local_df = build_gate3_local_table(
        records,
        unit_step_ms=DEFAULT_REVIEW_UNIT_STEP_MS,
        local_rate_decay=0.95,
        g_variant=g_variant,
        g_trim_ratio=g_trim_ratio,
        drop_edge_runs=drop_edge_runs,
    )
    ladder_df = summarize_falsification_ladder(ref_crop_df, monotonicity_df, prefix_silence_df)
    paths = {
        "gate_ref_crop_table": out_dir / "gate_ref_crop_table.csv",
        "gate_monotonicity_table": out_dir / "gate_monotonicity_table.csv",
        "gate_prefix_silence_table": out_dir / "gate_prefix_silence_table.csv",
        "gate_mode_ladder_table": out_dir / "gate_mode_ladder_table.csv",
        "gate_local_residual_table": out_dir / "gate_local_residual_table.csv",
        "gate_gate3_local_table": out_dir / "gate_gate3_local_table.csv",
    }
    ref_crop_df.to_csv(paths["gate_ref_crop_table"], index=False)
    monotonicity_df.to_csv(paths["gate_monotonicity_table"], index=False)
    prefix_silence_df.to_csv(paths["gate_prefix_silence_table"], index=False)
    ladder_df.to_csv(paths["gate_mode_ladder_table"], index=False)
    local_residual_df.to_csv(paths["gate_local_residual_table"], index=False)
    gate3_local_df.to_csv(paths["gate_gate3_local_table"], index=False)
    figures = {
        "gate_fig_monotonicity_intervention.png": plot_monotonicity_intervention(monotonicity_df),
        "gate_fig_prefix_silence_stability.png": plot_prefix_silence_stability(prefix_silence_df),
        "gate_fig_mode_ladder.png": plot_falsification_ladder(ladder_df),
    }
    for name, fig in figures.items():
        path = out_dir / name
        fig.savefig(path, dpi=180)
        fig.clf()
        paths[name] = path
    return paths


__all__ = [
    "DEFAULT_REVIEW_BOUNDARY_THRESHOLD",
    "DEFAULT_REVIEW_SILENCE_TAU",
    "DEFAULT_REVIEW_UNIT_STEP_MS",
    "bootstrap_ci",
    "build_gate3_local_table",
    "build_local_residual_review_table",
    "build_monotonicity_table",
    "build_prefix_replay_table",
    "build_prefix_silence_review_table",
    "build_ref_crop_table",
    "build_run_table",
    "build_silence_audit_table",
    "compute_a_i",
    "compute_b_star",
    "compute_commit_metrics",
    "compute_g",
    "compute_prefix_tempo",
    "compute_source_global_rate_for_analysis",
    "compute_speech_tempo_for_analysis",
    "compute_run_stability",
    "ensure_debug_records",
    "plot_falsification_ladder",
    "plot_global_cue_survival",
    "plot_monotonicity_intervention",
    "plot_online_commit_semantics",
    "plot_oracle_decomposition",
    "plot_prefix_silence_stability",
    "plot_run_lattice_stability",
    "plot_silence_theory_audit",
    "save_review_figure_bundle",
    "save_validation_gate_bundle",
    "summarize_falsification_ladder",
    "summarize_global_cue_review",
    "summarize_oracle_decomposition",
    "summarize_silence_audit",
    "weighted_median",
]
