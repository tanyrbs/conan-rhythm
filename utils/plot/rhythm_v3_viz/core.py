from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch

from modules.Conan.rhythm_v3.math_utils import (
    build_causal_source_prefix_rate_seq,
    normalize_src_prefix_stat_mode,
    resolve_default_source_rate_init,
)


DEFAULT_UNIT_STEP_MS = 20.0


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _to_numpy(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    return value


def _as_object_scalar(value: Any) -> Any:
    value = _to_numpy(value)
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        flat = value.reshape(-1)
        if flat.size <= 0:
            return None
        item = flat[0]
        if hasattr(item, "item"):
            try:
                item = item.item()
            except Exception:
                pass
        if isinstance(item, (bytes, bytearray)):
            return item.decode("utf-8")
        return item
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return value


def _as_float(value: Any, *, default: float = np.nan) -> float:
    scalar = _as_object_scalar(value)
    if scalar is None:
        return float(default)
    try:
        return float(scalar)
    except Exception:
        return float(default)


def _as_optional_array(
    value: Any,
    *,
    dtype: np.dtype | type | None = np.float32,
) -> Optional[np.ndarray]:
    value = _to_numpy(value)
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        arr = value
    else:
        try:
            arr = np.asarray(value)
        except Exception:
            return None
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _as_optional_vector(
    value: Any,
    *,
    dtype: np.dtype | type = np.float32,
) -> Optional[np.ndarray]:
    arr = _as_optional_array(value, dtype=dtype)
    if arr is None:
        return None
    return np.asarray(arr).reshape(-1)


def _slice_batch_value(value: Any, batch_index: int) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.ndim <= 0:
            return value
        if value.size(0) <= batch_index:
            return None
        return value[batch_index]
    if isinstance(value, np.ndarray):
        if value.ndim <= 0:
            return value
        if value.shape[0] <= batch_index:
            return None
        return value[batch_index]
    if isinstance(value, (list, tuple)):
        if len(value) <= batch_index:
            return None
        return value[batch_index]
    return value


def _extract_mapping_value(mapping: Mapping[str, Any] | None, key: str, batch_index: int) -> Any:
    if not _is_mapping(mapping):
        return None
    if key not in mapping:
        return None
    return _slice_batch_value(mapping[key], batch_index)


def _infer_batch_size(*containers: Any) -> int:
    for container in containers:
        if isinstance(container, Mapping):
            for value in container.values():
                if isinstance(value, torch.Tensor) and value.ndim >= 1:
                    return int(value.size(0))
                if isinstance(value, np.ndarray) and value.ndim >= 1:
                    return int(value.shape[0])
                if isinstance(value, (list, tuple)) and len(value) > 0:
                    return int(len(value))
        elif hasattr(container, "__dict__"):
            for value in vars(container).values():
                if isinstance(value, torch.Tensor) and value.ndim >= 1:
                    return int(value.size(0))
    return 1


def _extract_attr_or_key(obj: Any, name: str, batch_index: int) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return _extract_mapping_value(obj, name, batch_index)
    value = getattr(obj, name, None)
    return _slice_batch_value(value, batch_index)


def _last_masked_value(arr: np.ndarray | None, mask: np.ndarray | None) -> float:
    if arr is None or mask is None:
        return float("nan")
    idx = np.flatnonzero(np.asarray(mask, dtype=np.float32).reshape(-1) > 0.5)
    if idx.size <= 0:
        return float("nan")
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    return float(flat[int(idx[-1])])


def _extract_prompt_domain_valid_scalar(ref_memory: Any, batch_index: int) -> float | None:
    value = _extract_attr_or_key(ref_memory, "prompt_g_domain_valid", batch_index)
    if value is None:
        return None
    arr = np.asarray(_to_numpy(value), dtype=np.float32).reshape(-1)
    if arr.size <= 0:
        return None
    return float(arr[0])


def _resolve_debug_global_rate(
    *,
    ref_memory: Any,
    execution: Any,
    batch_index: int,
) -> float | None:
    prompt_domain_valid = _extract_prompt_domain_valid_scalar(ref_memory, batch_index)
    if prompt_domain_valid is not None and np.isfinite(prompt_domain_valid) and prompt_domain_valid <= 0.5:
        return float("nan")
    for value in (
        _extract_attr_or_key(execution, "g_ref", batch_index),
        _extract_attr_or_key(ref_memory, "global_rate", batch_index),
    ):
        if value is None:
            continue
        arr = np.asarray(_to_numpy(value)).reshape(-1)
        if arr.size <= 0:
            continue
        return float(arr[0])
    return None


def _ensure_1d_mask(mask: Optional[np.ndarray], reference: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if reference is None:
        return _as_optional_vector(mask)
    if mask is None:
        return np.ones_like(reference, dtype=np.float32)
    out = _as_optional_vector(mask, dtype=np.float32)
    if out is None:
        return np.ones_like(reference, dtype=np.float32)
    if out.shape[0] == reference.shape[0]:
        return out
    resized = np.zeros_like(reference, dtype=np.float32)
    limit = min(out.shape[0], reference.shape[0])
    resized[:limit] = out[:limit]
    return resized


def _infer_speaker_id(value: Any) -> str:
    if value is None:
        return ""
    name = str(value).strip()
    if not name:
        return ""
    return name.split("_", 1)[0]


def _rhythm_text_signature(item) -> tuple | str | None:
    if not isinstance(item, Mapping):
        return None
    for key in ("ph_token", "txt_token", "txt_tokens", "word_token", "word_tokens"):
        value = item.get(key)
        if value is None:
            continue
        arr = np.asarray(value).reshape(-1)
        if arr.size > 0:
            return (key, tuple(arr.tolist()))
    for key in ("ph", "txt", "word", "words"):
        value = item.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return (key, text)
    return None


def _maybe_set_meta(meta: dict[str, Any], key: str, value: Any) -> None:
    if key in meta:
        return
    if value is None:
        return
    meta[key] = value


def _safe_log(x: np.ndarray, eps: float = 1.0e-4) -> np.ndarray:
    return np.log(np.clip(x.astype(np.float32), eps, None))


def weighted_median_np(values: np.ndarray, weight: np.ndarray | None = None) -> float:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size <= 0:
        return 0.0
    if weight is None:
        return float(np.median(values))
    weight = np.asarray(weight, dtype=np.float32).reshape(-1)
    if weight.size != values.size:
        raise ValueError("weight must have the same shape as values")
    weight = np.clip(weight, 1.0e-4, None)
    order = np.argsort(values)
    values = values[order]
    weight = weight[order]
    cdf = np.cumsum(weight)
    cutoff = 0.5 * float(weight.sum())
    idx = int(np.searchsorted(cdf, cutoff, side="left"))
    idx = max(0, min(idx, values.size - 1))
    return float(values[idx])


@dataclass
class RhythmV3DerivedRecord:
    speech_mask: np.ndarray
    silence_mask: np.ndarray
    prompt_speech_mask: Optional[np.ndarray]
    target_logstretch: Optional[np.ndarray]
    source_logdur: Optional[np.ndarray]
    prompt_logdur: Optional[np.ndarray]
    source_rate_seq: Optional[np.ndarray]
    global_rate: Optional[float]
    analytic_shift: Optional[np.ndarray]
    oracle_bias: Optional[float]
    oracle_local: Optional[np.ndarray]
    oracle_silence_pseudo: Optional[np.ndarray]
    prediction_logstretch: Optional[np.ndarray]
    prediction_bias: Optional[float]
    prediction_local: Optional[np.ndarray]


def _resolve_prompt_g_config(
    meta: Mapping[str, Any],
    *,
    fallback_variant: str = "raw_median",
    fallback_trim_ratio: float = 0.2,
    fallback_drop_edge_runs: int = 0,
    fallback_min_boundary_confidence: float | None = None,
) -> dict[str, Any]:
    min_boundary = meta.get(
        "prompt_min_boundary_confidence_for_g",
        meta.get(
            "rhythm_v3_prompt_min_boundary_confidence_for_g",
            fallback_min_boundary_confidence,
        ),
    )
    return {
        "variant": str(
            meta.get(
                "prompt_g_variant",
                meta.get(
                    "rhythm_v3_prompt_g_variant",
                    meta.get("g_variant", fallback_variant),
                ),
            )
            or fallback_variant
        ),
        "trim_ratio": _as_float(
            meta.get(
                "prompt_g_trim_ratio",
                meta.get(
                    "rhythm_v3_prompt_g_trim_ratio",
                    meta.get("g_trim_ratio", fallback_trim_ratio),
                ),
            ),
            default=float(fallback_trim_ratio),
        ),
        "drop_edge_runs": int(
            round(
                _as_float(
                    meta.get(
                        "prompt_g_drop_edge_runs",
                        meta.get(
                            "rhythm_v3_prompt_g_drop_edge_runs",
                            meta.get("g_drop_edge_runs", fallback_drop_edge_runs),
                        ),
                    ),
                    default=float(fallback_drop_edge_runs),
                )
            )
        ),
        "min_boundary_confidence": None if min_boundary is None else float(min_boundary),
    }


def _resolve_src_g_config(
    meta: Mapping[str, Any],
    *,
    fallback_variant: str = "raw_median",
    fallback_trim_ratio: float = 0.2,
    fallback_drop_edge_runs: int = 0,
    fallback_min_boundary_confidence: float | None = None,
) -> dict[str, Any]:
    min_boundary = meta.get(
        "src_min_boundary_confidence_for_g",
        meta.get(
            "rhythm_v3_src_min_boundary_confidence_for_g",
            meta.get(
                "min_boundary_confidence_for_g",
                meta.get("rhythm_v3_min_boundary_confidence_for_g", fallback_min_boundary_confidence),
            ),
        ),
    )
    return {
        "variant": str(
            meta.get(
                "src_g_variant",
                meta.get(
                    "rhythm_v3_src_g_variant",
                    meta.get("g_variant", fallback_variant),
                ),
            )
            or fallback_variant
        ),
        "trim_ratio": _as_float(
            meta.get(
                "src_g_trim_ratio",
                meta.get(
                    "rhythm_v3_src_g_trim_ratio",
                    meta.get("g_trim_ratio", fallback_trim_ratio),
                ),
            ),
            default=float(fallback_trim_ratio),
        ),
        "drop_edge_runs": int(
            round(
                _as_float(
                    meta.get(
                        "src_g_drop_edge_runs",
                        meta.get(
                            "rhythm_v3_src_g_drop_edge_runs",
                            meta.get(
                                "drop_edge_runs_for_g",
                                meta.get("rhythm_v3_drop_edge_runs_for_g", fallback_drop_edge_runs),
                            ),
                        ),
                    ),
                    default=float(fallback_drop_edge_runs),
                )
            )
        ),
        "min_boundary_confidence": None if min_boundary is None else float(min_boundary),
    }


@dataclass
class RhythmV3DebugRecord:
    item_name: Optional[str] = None
    split: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source_content_units: Optional[np.ndarray] = None
    source_duration_obs: Optional[np.ndarray] = None
    source_silence_mask: Optional[np.ndarray] = None
    source_boundary_cue: Optional[np.ndarray] = None
    source_run_stability: Optional[np.ndarray] = None
    unit_mask: Optional[np.ndarray] = None
    sealed_mask: Optional[np.ndarray] = None
    sep_mask: Optional[np.ndarray] = None
    commit_mask: Optional[np.ndarray] = None
    prompt_content_units: Optional[np.ndarray] = None
    prompt_duration_obs: Optional[np.ndarray] = None
    prompt_valid_mask: Optional[np.ndarray] = None
    prompt_speech_mask: Optional[np.ndarray] = None
    prompt_closed_mask: Optional[np.ndarray] = None
    prompt_boundary_confidence: Optional[np.ndarray] = None
    prompt_global_weight: Optional[np.ndarray] = None
    prompt_log_duration: Optional[np.ndarray] = None
    prompt_log_residual: Optional[np.ndarray] = None
    unit_duration_tgt: Optional[np.ndarray] = None
    unit_duration_proj_raw_tgt: Optional[np.ndarray] = None
    unit_alignment_mode_id_tgt: Optional[np.ndarray] = None
    unit_alignment_kind_tgt: Optional[np.ndarray] = None
    unit_alignment_source_tgt: Optional[np.ndarray] = None
    unit_alignment_version_tgt: Optional[np.ndarray] = None
    unit_confidence_tgt: Optional[np.ndarray] = None
    unit_confidence_local_tgt: Optional[np.ndarray] = None
    unit_confidence_coarse_tgt: Optional[np.ndarray] = None
    unit_alignment_coverage_tgt: Optional[np.ndarray] = None
    unit_alignment_match_tgt: Optional[np.ndarray] = None
    unit_alignment_cost_tgt: Optional[np.ndarray] = None
    unit_alignment_unmatched_speech_ratio_tgt: Optional[np.ndarray] = None
    unit_alignment_mean_local_confidence_speech_tgt: Optional[np.ndarray] = None
    unit_alignment_mean_coarse_confidence_speech_tgt: Optional[np.ndarray] = None
    paired_target_content_units_debug: Optional[np.ndarray] = None
    paired_target_duration_obs_debug: Optional[np.ndarray] = None
    paired_target_valid_mask_debug: Optional[np.ndarray] = None
    paired_target_speech_mask_debug: Optional[np.ndarray] = None
    unit_alignment_assigned_source_debug: Optional[np.ndarray] = None
    unit_alignment_assigned_cost_debug: Optional[np.ndarray] = None
    unit_alignment_source_valid_run_index_debug: Optional[np.ndarray] = None
    unit_alignment_run_margin_tgt: Optional[np.ndarray] = None
    unit_alignment_run_mean_cost_tgt: Optional[np.ndarray] = None
    unit_alignment_run_type_agree_tgt: Optional[np.ndarray] = None
    unit_alignment_run_occ_weighted_tgt: Optional[np.ndarray] = None
    unit_alignment_run_occ_expected_tgt: Optional[np.ndarray] = None
    unit_alignment_run_entropy_tgt: Optional[np.ndarray] = None
    unit_alignment_run_posterior_mass_on_path_tgt: Optional[np.ndarray] = None
    unit_alignment_posterior_band_left_debug: Optional[np.ndarray] = None
    unit_alignment_posterior_band_right_debug: Optional[np.ndarray] = None
    unit_alignment_posterior_values_debug: Optional[np.ndarray] = None
    global_rate: Optional[float] = None
    source_rate_seq: Optional[np.ndarray] = None
    g_src_prefix_final: Optional[np.ndarray] = None
    source_rate_init_value: Optional[np.ndarray] = None
    global_shift_analytic: Optional[np.ndarray] = None
    global_bias_scalar: Optional[float] = None
    coarse_logstretch: Optional[np.ndarray] = None
    coarse_correction: Optional[np.ndarray] = None
    coarse_correction_pred: Optional[np.ndarray] = None
    local_residual: Optional[np.ndarray] = None
    local_residual_pred: Optional[np.ndarray] = None
    speech_pred: Optional[np.ndarray] = None
    silence_pred: Optional[np.ndarray] = None
    unit_logstretch: Optional[np.ndarray] = None
    unit_logstretch_raw: Optional[np.ndarray] = None
    unit_duration_exec: Optional[np.ndarray] = None
    unit_duration_raw: Optional[np.ndarray] = None
    projector_preclamp_exec: Optional[np.ndarray] = None
    projector_preclamp_duration_exec: Optional[np.ndarray] = None
    projector_preclamp_prefix_cumsum: Optional[np.ndarray] = None
    projector_clamp_delta: Optional[np.ndarray] = None
    projector_clamp_mass: Optional[np.ndarray] = None
    prefix_unit_offset: Optional[np.ndarray] = None
    projector_prefix_drift: Optional[np.ndarray] = None
    projector_rounding_residual: Optional[np.ndarray] = None
    projector_rounding_regret: Optional[np.ndarray] = None
    projector_projection_regret: Optional[np.ndarray] = None
    projector_budget_hit_pos: Optional[np.ndarray] = None
    projector_budget_hit_neg: Optional[np.ndarray] = None
    projector_budget_hit_mask: Optional[np.ndarray] = None
    projector_boundary_hit: Optional[np.ndarray] = None
    projector_boundary_decay_applied: Optional[np.ndarray] = None
    analytic_gap_raw: Optional[np.ndarray] = None
    analytic_gap_clipped: Optional[np.ndarray] = None
    analytic_clip_hit: Optional[np.ndarray] = None
    analytic_clip_hit_rate: Optional[np.ndarray] = None
    coarse_scalar_raw: Optional[np.ndarray] = None
    global_term_before_local: Optional[np.ndarray] = None
    analytic_gap_clip_value: Optional[np.ndarray] = None
    residual_gate_mean: Optional[np.ndarray] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "RhythmV3DebugRecord":
        payload = dict(mapping)
        if "metadata" in payload and payload["metadata"] is None:
            payload["metadata"] = {}
        return cls(**payload)


def build_debug_record(
    *,
    batch_index: int = 0,
    sample: Mapping[str, Any] | None = None,
    model_output: Mapping[str, Any] | None = None,
    source_batch: Any = None,
    ref_memory: Any = None,
    execution: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> RhythmV3DebugRecord:
    if source_batch is None and _is_mapping(model_output):
        source_batch = model_output.get("rhythm_unit_batch")
    if ref_memory is None and _is_mapping(model_output):
        ref_memory = model_output.get("rhythm_ref_conditioning")
    if execution is None and _is_mapping(model_output):
        execution = model_output.get("rhythm_execution")

    item_name = _as_object_scalar(_extract_mapping_value(sample, "item_name", batch_index))
    split = _as_object_scalar(_extract_mapping_value(sample, "split", batch_index))
    meta = dict(metadata or {})
    if _is_mapping(sample):
        for key in (
            "src_wav",
            "ref_wav",
            "paired_target_item_name",
            "pair_id",
            "sample_id",
            "src_id",
            "src_prompt_id",
            "tgt_prompt_id",
            "ref_prompt_id",
            "ref_item_name",
            "same_text_reference",
            "same_text_target",
            "same_text",
            "lexical_mismatch",
            "ref_len_sec",
            "speech_ratio",
            "alignment_kind",
            "alignment_source",
            "alignment_version",
            "target_duration_surface",
            "ref_condition",
            "ref_bin",
            "tempo_bin",
            "g_support_count",
            "g_speech_count",
            "g_valid_count",
            "g_support_ratio_vs_speech",
            "g_support_ratio_vs_valid",
            "g_valid",
            "g_drop_edge_runs",
            "g_strict_speech_only",
            "g_trim_ratio",
            "prompt_g_variant",
            "prompt_g_trim_ratio",
            "prompt_g_drop_edge_runs",
            "prompt_min_boundary_confidence_for_g",
            "prompt_global_weight_present",
            "prompt_unit_log_prior_present",
            "prompt_unit_prior_vocab_size",
            "unit_alignment_kind_tgt",
            "unit_alignment_source_tgt",
            "unit_alignment_version_tgt",
        ):
            value = _extract_mapping_value(sample, key, batch_index)
            scalar = _as_object_scalar(value)
            if scalar is not None:
                meta.setdefault(key, scalar)
        for key in (
            "unit_alignment_unmatched_speech_ratio_tgt",
            "unit_alignment_mean_local_confidence_speech_tgt",
            "unit_alignment_mean_coarse_confidence_speech_tgt",
        ):
            scalar = _as_object_scalar(_extract_mapping_value(sample, key, batch_index))
            if scalar is not None:
                meta.setdefault(key, scalar)
        pair_group = _as_object_scalar(_extract_mapping_value(sample, "rhythm_pair_group_id", batch_index))
        pair_rank = _as_object_scalar(_extract_mapping_value(sample, "rhythm_pair_rank", batch_index))
        pair_identity = _as_object_scalar(_extract_mapping_value(sample, "rhythm_pair_is_identity", batch_index))
        if pair_group is not None:
            _maybe_set_meta(meta, "rhythm_pair_group_id", pair_group)
            _maybe_set_meta(meta, "pair_id", pair_group)
        if pair_rank is not None:
            _maybe_set_meta(meta, "rhythm_pair_rank", pair_rank)
        if pair_identity is not None:
            _maybe_set_meta(meta, "rhythm_pair_is_identity", pair_identity)

        raw_item = _extract_mapping_value(sample, "_raw_item", batch_index)
        raw_ref_item = _extract_mapping_value(sample, "_raw_ref_item", batch_index)
        raw_tgt_item = _extract_mapping_value(sample, "_raw_paired_target_item", batch_index)
        if isinstance(raw_item, Mapping):
            raw_item_name = _as_object_scalar(raw_item.get("item_name"))
            _maybe_set_meta(meta, "src_item_name", raw_item_name)
            _maybe_set_meta(meta, "src_prompt_id", raw_item_name)
            _maybe_set_meta(meta, "src_spk", _infer_speaker_id(raw_item_name))
            _maybe_set_meta(meta, "source_text_signature", _rhythm_text_signature(raw_item))
            _maybe_set_meta(meta, "src_wav", raw_item.get("wav_fn"))
        if isinstance(raw_ref_item, Mapping):
            raw_ref_name = _as_object_scalar(raw_ref_item.get("item_name"))
            _maybe_set_meta(meta, "ref_item_name", raw_ref_name)
            _maybe_set_meta(meta, "ref_prompt_id", raw_ref_name)
            _maybe_set_meta(meta, "ref_spk", _infer_speaker_id(raw_ref_name))
            _maybe_set_meta(meta, "reference_text_signature", _rhythm_text_signature(raw_ref_item))
            _maybe_set_meta(meta, "ref_wav", raw_ref_item.get("wav_fn"))
        if isinstance(raw_tgt_item, Mapping):
            raw_tgt_name = _as_object_scalar(raw_tgt_item.get("item_name"))
            _maybe_set_meta(meta, "paired_target_item_name", raw_tgt_name)
            _maybe_set_meta(meta, "tgt_prompt_id", raw_tgt_name)
            _maybe_set_meta(meta, "tgt_spk", _infer_speaker_id(raw_tgt_name))
            _maybe_set_meta(meta, "paired_target_text_signature", _rhythm_text_signature(raw_tgt_item))

        source_sig = meta.get("source_text_signature")
        ref_sig = meta.get("reference_text_signature")
        tgt_sig = meta.get("paired_target_text_signature")
        if source_sig is not None and ref_sig is not None:
            _maybe_set_meta(meta, "same_text_reference", int(source_sig == ref_sig))
        if source_sig is not None and tgt_sig is not None:
            _maybe_set_meta(meta, "same_text_target", int(source_sig == tgt_sig))
        if item_name is not None:
            _maybe_set_meta(meta, "src_spk", _infer_speaker_id(item_name))
            _maybe_set_meta(meta, "src_prompt_id", item_name)
        alignment_kind_value = _as_object_scalar(_extract_mapping_value(sample, "unit_alignment_kind_tgt", batch_index))
        if alignment_kind_value is not None:
            _maybe_set_meta(meta, "alignment_kind", str(alignment_kind_value))
        alignment_source_value = _as_object_scalar(_extract_mapping_value(sample, "unit_alignment_source_tgt", batch_index))
        if alignment_source_value is not None:
            _maybe_set_meta(meta, "alignment_source", str(alignment_source_value))
        alignment_version_value = _as_object_scalar(_extract_mapping_value(sample, "unit_alignment_version_tgt", batch_index))
        if alignment_version_value is not None:
            _maybe_set_meta(meta, "alignment_version", str(alignment_version_value))
        alignment_mode_id = _as_object_scalar(_extract_mapping_value(sample, "unit_alignment_mode_id_tgt", batch_index))
        if alignment_mode_id is not None:
            try:
                alignment_mode_id = int(alignment_mode_id)
            except Exception:
                alignment_mode_id = 0
            if "alignment_kind" not in meta:
                _maybe_set_meta(
                    meta,
                    "alignment_kind",
                    "continuous" if alignment_mode_id in {1, 2} else "discrete",
                )
        if _extract_mapping_value(sample, "unit_duration_proj_raw_tgt", batch_index) is not None:
            _maybe_set_meta(meta, "target_duration_surface", "projection_raw")
        ref_name = meta.get("ref_item_name") or meta.get("ref_prompt_id")
        if ref_name is not None:
            _maybe_set_meta(meta, "ref_spk", _infer_speaker_id(ref_name))
        tgt_name = meta.get("paired_target_item_name") or meta.get("tgt_prompt_id")
        if tgt_name is not None:
            _maybe_set_meta(meta, "tgt_spk", _infer_speaker_id(tgt_name))
        src_spk = meta.get("src_spk")
        ref_spk = meta.get("ref_spk")
        tgt_spk = meta.get("tgt_spk")
        if src_spk is not None and ref_spk is not None:
            _maybe_set_meta(meta, "same_speaker_reference", int(str(src_spk) == str(ref_spk)))
            _maybe_set_meta(meta, "same_speaker", int(str(src_spk) == str(ref_spk)))
        if src_spk is not None and tgt_spk is not None:
            _maybe_set_meta(meta, "same_speaker_target", int(str(src_spk) == str(tgt_spk)))
    eval_mode = _as_object_scalar(_extract_attr_or_key(execution, "eval_mode", batch_index))
    if eval_mode is not None:
        _maybe_set_meta(meta, "eval_mode", eval_mode)
    if _is_mapping(model_output):
        g_variant = model_output.get("rhythm_v3_g_variant")
        if g_variant is not None:
            _maybe_set_meta(meta, "g_variant", _as_object_scalar(g_variant))
            _maybe_set_meta(meta, "src_g_variant", _as_object_scalar(g_variant))
        eval_mode_value = model_output.get("rhythm_v3_eval_mode")
        if eval_mode_value is not None:
            _maybe_set_meta(meta, "rhythm_v3_eval_mode", _as_object_scalar(eval_mode_value))
        for key, meta_key in (
            ("rhythm_debug_g_support_count", "g_support_count"),
            ("rhythm_debug_g_speech_count", "g_speech_count"),
            ("rhythm_debug_g_valid_count", "g_valid_count"),
            ("rhythm_debug_g_support_ratio_vs_speech", "g_support_ratio_vs_speech"),
            ("rhythm_debug_g_support_ratio_vs_valid", "g_support_ratio_vs_valid"),
            ("rhythm_debug_g_valid", "g_valid"),
            ("rhythm_debug_g_valid_support", "g_valid_support"),
            ("rhythm_debug_g_domain_valid", "g_domain_valid"),
            ("rhythm_debug_g_min_speech_ratio", "g_min_speech_ratio"),
            ("rhythm_debug_prompt_speech_ratio", "prompt_speech_ratio"),
            ("rhythm_debug_g_drop_edge_runs", "g_drop_edge_runs"),
            ("rhythm_debug_g_strict_speech_only", "g_strict_speech_only"),
            ("rhythm_debug_prompt_g_speech_ratio_weighted", "prompt_g_speech_ratio_weighted"),
            ("rhythm_debug_prompt_g_speech_ratio_count", "prompt_g_speech_ratio_count"),
            ("rhythm_debug_prompt_g_invalid_no_speech", "prompt_g_invalid_no_speech"),
            ("rhythm_debug_prompt_g_invalid_low_speech_ratio", "prompt_g_invalid_low_speech_ratio"),
            ("rhythm_debug_prompt_g_invalid_ref_len", "prompt_g_invalid_ref_len"),
            ("rhythm_debug_prompt_g_invalid_support", "prompt_g_invalid_support"),
            ("rhythm_debug_prompt_g_invalid_clean", "prompt_g_invalid_clean"),
            ("rhythm_debug_prompt_g_invalid_missing_closed", "prompt_g_invalid_missing_closed"),
            ("rhythm_debug_prompt_g_invalid_missing_boundary", "prompt_g_invalid_missing_boundary"),
            ("rhythm_debug_prompt_global_weight_present", "prompt_global_weight_present"),
            ("rhythm_debug_prompt_unit_log_prior_present", "prompt_unit_log_prior_present"),
            ("rhythm_debug_prompt_unit_prior_vocab_size", "prompt_unit_prior_vocab_size"),
            ("rhythm_debug_detach_global_term_in_local_head", "detach_global_term_in_local_head"),
        ):
            value = model_output.get(key)
            scalar = _as_object_scalar(_to_numpy(value) if value is not None else None)
            if scalar is not None:
                _maybe_set_meta(meta, meta_key, scalar)
        if model_output.get("rhythm_v3_g_trim_ratio") is not None:
            _maybe_set_meta(meta, "g_trim_ratio", _as_object_scalar(model_output.get("rhythm_v3_g_trim_ratio")))
            _maybe_set_meta(meta, "src_g_trim_ratio", _as_object_scalar(model_output.get("rhythm_v3_g_trim_ratio")))
        for key in (
            "rhythm_v3_prompt_g_variant",
            "rhythm_v3_prompt_g_trim_ratio",
            "rhythm_v3_prompt_g_drop_edge_runs",
            "rhythm_v3_prompt_min_boundary_confidence_for_g",
            "rhythm_v3_src_g_variant",
            "rhythm_v3_src_g_trim_ratio",
            "rhythm_v3_src_g_drop_edge_runs",
            "rhythm_v3_src_min_boundary_confidence_for_g",
        ):
            value = model_output.get(key)
            if value is not None:
                _maybe_set_meta(meta, key.replace("rhythm_v3_", ""), _as_object_scalar(value))
        for key in (
            "rhythm_v3_min_boundary_confidence_for_g",
            "rhythm_v3_src_prefix_stat_mode",
            "rhythm_v3_src_prefix_min_support",
            "rhythm_v3_minimal_v1_profile",
            "rhythm_v3_strict_minimal_claim_profile",
            "rhythm_v3_use_continuous_alignment",
            "rhythm_v3_alignment_mode",
        ):
            if model_output.get(key) is not None:
                _maybe_set_meta(meta, key, _as_object_scalar(model_output.get(key)))
        if model_output.get("rhythm_v3_src_rate_init_mode") is not None:
            _maybe_set_meta(
                meta,
                "src_rate_init_mode",
                _as_object_scalar(model_output.get("rhythm_v3_src_rate_init_mode")),
            )
        if model_output.get("rhythm_v3_src_prefix_contract_scope") is not None:
            _maybe_set_meta(
                meta,
                "src_prefix_contract_scope",
                _as_object_scalar(model_output.get("rhythm_v3_src_prefix_contract_scope")),
            )
        if model_output.get("rhythm_v3_src_prefix_requires_full_history") is not None:
            _maybe_set_meta(
                meta,
                "src_prefix_requires_full_history",
                _as_object_scalar(model_output.get("rhythm_v3_src_prefix_requires_full_history")),
            )

    return RhythmV3DebugRecord(
        item_name=None if item_name is None else str(item_name),
        split=None if split is None else str(split),
        metadata=meta,
        source_content_units=_as_optional_vector(
            _extract_mapping_value(sample, "content_units", batch_index)
            if _is_mapping(sample)
            else _extract_attr_or_key(source_batch, "content_units", batch_index),
            dtype=np.int64,
        ),
        source_duration_obs=_as_optional_vector(
            _extract_mapping_value(sample, "dur_anchor_src", batch_index)
            if _is_mapping(sample) and "dur_anchor_src" in sample
            else _extract_attr_or_key(source_batch, "source_duration_obs", batch_index),
        ),
        source_silence_mask=_as_optional_vector(
            _extract_mapping_value(sample, "source_silence_mask", batch_index)
            if _is_mapping(sample) and "source_silence_mask" in sample
            else _extract_attr_or_key(source_batch, "source_silence_mask", batch_index),
        ),
        source_boundary_cue=_as_optional_vector(
            _extract_mapping_value(sample, "source_boundary_cue", batch_index)
            if _is_mapping(sample) and "source_boundary_cue" in sample
            else _extract_attr_or_key(source_batch, "source_boundary_cue", batch_index),
        ),
        source_run_stability=_as_optional_vector(
            _extract_mapping_value(sample, "source_run_stability", batch_index)
            if _is_mapping(sample) and "source_run_stability" in sample
            else _extract_attr_or_key(source_batch, "source_run_stability", batch_index),
        ),
        unit_mask=_as_optional_vector(
            _extract_mapping_value(sample, "unit_mask", batch_index)
            if _is_mapping(sample) and "unit_mask" in sample
            else _extract_attr_or_key(source_batch, "unit_mask", batch_index),
        ),
        sealed_mask=_as_optional_vector(
            _extract_mapping_value(sample, "sealed_mask", batch_index)
            if _is_mapping(sample) and "sealed_mask" in sample
            else _extract_attr_or_key(source_batch, "sealed_mask", batch_index),
        ),
        sep_mask=_as_optional_vector(
            _extract_mapping_value(sample, "sep_mask", batch_index)
            if _is_mapping(sample) and "sep_mask" in sample
            else _extract_attr_or_key(source_batch, "sep_mask", batch_index),
        ),
        commit_mask=_as_optional_vector(_extract_attr_or_key(execution, "commit_mask", batch_index)),
        prompt_content_units=_as_optional_vector(_extract_mapping_value(sample, "prompt_content_units", batch_index), dtype=np.int64),
        prompt_duration_obs=_as_optional_vector(_extract_mapping_value(sample, "prompt_duration_obs", batch_index)),
        prompt_valid_mask=_as_optional_vector(
            _extract_mapping_value(sample, "prompt_valid_mask", batch_index)
            if _is_mapping(sample) and "prompt_valid_mask" in sample
            else _extract_mapping_value(sample, "prompt_unit_mask", batch_index),
        ),
        prompt_speech_mask=_as_optional_vector(_extract_mapping_value(sample, "prompt_speech_mask", batch_index)),
        prompt_closed_mask=_as_optional_vector(_extract_mapping_value(sample, "prompt_closed_mask", batch_index)),
        prompt_boundary_confidence=_as_optional_vector(
            _extract_mapping_value(sample, "prompt_boundary_confidence", batch_index)
        ),
        prompt_global_weight=_as_optional_vector(_extract_mapping_value(sample, "prompt_global_weight", batch_index)),
        prompt_log_duration=_as_optional_vector(_extract_attr_or_key(ref_memory, "prompt_log_duration", batch_index)),
        prompt_log_residual=_as_optional_vector(_extract_attr_or_key(ref_memory, "prompt_log_residual", batch_index)),
        unit_duration_tgt=_as_optional_vector(_extract_mapping_value(sample, "unit_duration_tgt", batch_index)),
        unit_duration_proj_raw_tgt=_as_optional_vector(_extract_mapping_value(sample, "unit_duration_proj_raw_tgt", batch_index)),
        unit_alignment_mode_id_tgt=_as_optional_vector(
            _extract_mapping_value(sample, "unit_alignment_mode_id_tgt", batch_index),
            dtype=np.int64,
        ),
        unit_alignment_kind_tgt=_as_optional_array(
            _extract_mapping_value(sample, "unit_alignment_kind_tgt", batch_index),
            dtype=None,
        ),
        unit_alignment_source_tgt=_as_optional_array(
            _extract_mapping_value(sample, "unit_alignment_source_tgt", batch_index),
            dtype=None,
        ),
        unit_alignment_version_tgt=_as_optional_array(
            _extract_mapping_value(sample, "unit_alignment_version_tgt", batch_index),
            dtype=None,
        ),
        unit_confidence_tgt=_as_optional_vector(_extract_mapping_value(sample, "unit_confidence_tgt", batch_index)),
        unit_confidence_local_tgt=_as_optional_vector(_extract_mapping_value(sample, "unit_confidence_local_tgt", batch_index)),
        unit_confidence_coarse_tgt=_as_optional_vector(_extract_mapping_value(sample, "unit_confidence_coarse_tgt", batch_index)),
        unit_alignment_coverage_tgt=_as_optional_vector(_extract_mapping_value(sample, "unit_alignment_coverage_tgt", batch_index)),
        unit_alignment_match_tgt=_as_optional_vector(_extract_mapping_value(sample, "unit_alignment_match_tgt", batch_index)),
        unit_alignment_cost_tgt=_as_optional_vector(_extract_mapping_value(sample, "unit_alignment_cost_tgt", batch_index)),
        unit_alignment_unmatched_speech_ratio_tgt=_as_optional_vector(
            _extract_mapping_value(sample, "unit_alignment_unmatched_speech_ratio_tgt", batch_index)
        ),
        unit_alignment_mean_local_confidence_speech_tgt=_as_optional_vector(
            _extract_mapping_value(sample, "unit_alignment_mean_local_confidence_speech_tgt", batch_index)
        ),
        unit_alignment_mean_coarse_confidence_speech_tgt=_as_optional_vector(
            _extract_mapping_value(sample, "unit_alignment_mean_coarse_confidence_speech_tgt", batch_index)
        ),
        paired_target_content_units_debug=_as_optional_vector(
            _extract_mapping_value(sample, "paired_target_content_units_debug", batch_index),
            dtype=np.int64,
        ),
        paired_target_duration_obs_debug=_as_optional_vector(_extract_mapping_value(sample, "paired_target_duration_obs_debug", batch_index)),
        paired_target_valid_mask_debug=_as_optional_vector(_extract_mapping_value(sample, "paired_target_valid_mask_debug", batch_index)),
        paired_target_speech_mask_debug=_as_optional_vector(_extract_mapping_value(sample, "paired_target_speech_mask_debug", batch_index)),
        unit_alignment_assigned_source_debug=_as_optional_vector(
            _extract_mapping_value(sample, "unit_alignment_assigned_source_debug", batch_index),
            dtype=np.int64,
        ),
        unit_alignment_assigned_cost_debug=_as_optional_vector(_extract_mapping_value(sample, "unit_alignment_assigned_cost_debug", batch_index)),
        unit_alignment_source_valid_run_index_debug=_as_optional_vector(
            _extract_mapping_value(sample, "unit_alignment_source_valid_run_index_debug", batch_index),
            dtype=np.int64,
        ),
        unit_alignment_run_margin_tgt=_as_optional_vector(_extract_mapping_value(sample, "unit_alignment_run_margin_tgt", batch_index)),
        unit_alignment_run_mean_cost_tgt=_as_optional_vector(_extract_mapping_value(sample, "unit_alignment_run_mean_cost_tgt", batch_index)),
        unit_alignment_run_type_agree_tgt=_as_optional_vector(_extract_mapping_value(sample, "unit_alignment_run_type_agree_tgt", batch_index)),
        unit_alignment_run_occ_weighted_tgt=_as_optional_vector(_extract_mapping_value(sample, "unit_alignment_run_occ_weighted_tgt", batch_index)),
        unit_alignment_run_occ_expected_tgt=_as_optional_vector(_extract_mapping_value(sample, "unit_alignment_run_occ_expected_tgt", batch_index)),
        unit_alignment_run_entropy_tgt=_as_optional_vector(_extract_mapping_value(sample, "unit_alignment_run_entropy_tgt", batch_index)),
        unit_alignment_run_posterior_mass_on_path_tgt=_as_optional_vector(
            _extract_mapping_value(sample, "unit_alignment_run_posterior_mass_on_path_tgt", batch_index)
        ),
        unit_alignment_posterior_band_left_debug=_as_optional_vector(
            _extract_mapping_value(sample, "unit_alignment_posterior_band_left_debug", batch_index),
            dtype=np.int64,
        ),
        unit_alignment_posterior_band_right_debug=_as_optional_vector(
            _extract_mapping_value(sample, "unit_alignment_posterior_band_right_debug", batch_index),
            dtype=np.int64,
        ),
        unit_alignment_posterior_values_debug=_as_optional_vector(
            _extract_mapping_value(sample, "unit_alignment_posterior_values_debug", batch_index)
        ),
        global_rate=_resolve_debug_global_rate(
            ref_memory=ref_memory,
            execution=execution,
            batch_index=batch_index,
        ),
        source_rate_seq=_as_optional_vector(
            _extract_attr_or_key(execution, "source_rate_seq", batch_index)
            if _extract_attr_or_key(execution, "source_rate_seq", batch_index) is not None
            else _extract_attr_or_key(execution, "g_src_prefix", batch_index)
        ),
        g_src_prefix_final=_as_optional_vector(_extract_attr_or_key(execution, "g_src_prefix_final", batch_index)),
        source_rate_init_value=_as_optional_vector(
            _extract_attr_or_key(model_output, "rhythm_v3_source_rate_init", batch_index)
        ),
        global_shift_analytic=_as_optional_vector(_extract_attr_or_key(execution, "global_shift_analytic", batch_index)),
        global_bias_scalar=(
            None
            if _extract_attr_or_key(execution, "global_bias_scalar", batch_index) is None
            else float(np.asarray(_to_numpy(_extract_attr_or_key(execution, "global_bias_scalar", batch_index))).reshape(-1)[0])
        ),
        coarse_logstretch=_as_optional_vector(_extract_attr_or_key(execution, "coarse_logstretch", batch_index)),
        coarse_correction=_as_optional_vector(_extract_attr_or_key(execution, "coarse_correction", batch_index)),
        coarse_correction_pred=_as_optional_vector(_extract_attr_or_key(execution, "coarse_correction_pred", batch_index)),
        local_residual=_as_optional_vector(_extract_attr_or_key(execution, "local_residual", batch_index)),
        local_residual_pred=_as_optional_vector(_extract_attr_or_key(execution, "local_residual_pred", batch_index)),
        speech_pred=_as_optional_vector(_extract_attr_or_key(execution, "speech_pred", batch_index)),
        silence_pred=_as_optional_vector(_extract_attr_or_key(execution, "silence_pred", batch_index)),
        unit_logstretch=_as_optional_vector(_extract_attr_or_key(execution, "unit_logstretch", batch_index)),
        unit_logstretch_raw=_as_optional_vector(_extract_attr_or_key(execution, "unit_logstretch_raw", batch_index)),
        unit_duration_exec=_as_optional_vector(_extract_attr_or_key(execution, "unit_duration_exec", batch_index)),
        unit_duration_raw=_as_optional_vector(_extract_attr_or_key(execution, "unit_duration_raw", batch_index)),
        projector_preclamp_exec=_as_optional_vector(
            _extract_attr_or_key(execution, "projector_preclamp_exec", batch_index)
        ),
        projector_preclamp_duration_exec=_as_optional_vector(
            _extract_attr_or_key(execution, "projector_preclamp_duration_exec", batch_index)
        ),
        projector_preclamp_prefix_cumsum=_as_optional_vector(
            _extract_attr_or_key(execution, "projector_preclamp_prefix_cumsum", batch_index)
        ),
        projector_clamp_delta=_as_optional_vector(
            _extract_attr_or_key(execution, "projector_clamp_delta", batch_index)
        ),
        projector_clamp_mass=_as_optional_vector(
            _extract_attr_or_key(execution, "projector_clamp_mass", batch_index)
        ),
        prefix_unit_offset=_as_optional_vector(_extract_attr_or_key(execution, "prefix_unit_offset", batch_index)),
        projector_prefix_drift=_as_optional_vector(
            _extract_attr_or_key(execution, "projector_prefix_drift", batch_index)
            if _extract_attr_or_key(execution, "projector_prefix_drift", batch_index) is not None
            else _extract_attr_or_key(execution, "prefix_unit_offset", batch_index)
        ),
        projector_rounding_residual=_as_optional_vector(
            _extract_attr_or_key(execution, "projector_rounding_residual", batch_index)
            if _extract_attr_or_key(execution, "projector_rounding_residual", batch_index) is not None
            else _extract_attr_or_key(getattr(execution, "next_state", None), "rounding_residual", batch_index)
        ),
        projector_rounding_regret=_as_optional_vector(
            _extract_attr_or_key(execution, "projector_rounding_regret", batch_index)
        ),
        projector_projection_regret=_as_optional_vector(
            _extract_attr_or_key(execution, "projector_projection_regret", batch_index)
        ),
        projector_budget_hit_pos=_as_optional_vector(_extract_attr_or_key(execution, "projector_budget_hit_pos", batch_index)),
        projector_budget_hit_neg=_as_optional_vector(_extract_attr_or_key(execution, "projector_budget_hit_neg", batch_index)),
        projector_budget_hit_mask=_as_optional_vector(
            _extract_attr_or_key(execution, "projector_budget_hit_mask", batch_index)
            if _extract_attr_or_key(execution, "projector_budget_hit_mask", batch_index) is not None
            else (
                np.logical_or(
                    _as_optional_vector(_extract_attr_or_key(execution, "projector_budget_hit_pos", batch_index), dtype=np.float32)
                    > 0.5,
                    _as_optional_vector(_extract_attr_or_key(execution, "projector_budget_hit_neg", batch_index), dtype=np.float32)
                    > 0.5,
                ).astype(np.float32)
                if _extract_attr_or_key(execution, "projector_budget_hit_pos", batch_index) is not None
                and _extract_attr_or_key(execution, "projector_budget_hit_neg", batch_index) is not None
                else None
            )
        ),
        projector_boundary_hit=_as_optional_vector(_extract_attr_or_key(execution, "projector_boundary_hit", batch_index)),
        projector_boundary_decay_applied=_as_optional_vector(
            _extract_attr_or_key(execution, "projector_boundary_decay_applied", batch_index)
        ),
        analytic_gap_raw=_as_optional_vector(_extract_attr_or_key(execution, "analytic_gap_raw", batch_index)),
        analytic_gap_clipped=_as_optional_vector(_extract_attr_or_key(execution, "analytic_gap_clipped", batch_index)),
        analytic_clip_hit=_as_optional_vector(_extract_attr_or_key(execution, "analytic_clip_hit", batch_index)),
        analytic_clip_hit_rate=_as_optional_vector(_extract_attr_or_key(execution, "analytic_clip_hit_rate", batch_index)),
        coarse_scalar_raw=_as_optional_vector(_extract_attr_or_key(execution, "coarse_scalar_raw", batch_index)),
        global_term_before_local=_as_optional_vector(
            _extract_attr_or_key(execution, "global_term_before_local", batch_index)
            if _extract_attr_or_key(execution, "global_term_before_local", batch_index) is not None
            else _extract_attr_or_key(execution, "unit_global_term_before_local", batch_index)
        ),
        analytic_gap_clip_value=_as_optional_vector(
            _extract_attr_or_key(execution, "analytic_gap_clip_value", batch_index)
        ),
        residual_gate_mean=_as_optional_vector(_extract_attr_or_key(execution, "residual_gate_mean", batch_index)),
    )


def build_debug_records_from_batch(
    *,
    sample: Mapping[str, Any] | None = None,
    model_output: Mapping[str, Any] | None = None,
    source_batch: Any = None,
    ref_memory: Any = None,
    execution: Any = None,
    metadata: Mapping[str, Any] | None = None,
    batch_indices: Sequence[int] | None = None,
) -> list[RhythmV3DebugRecord]:
    batch_size = _infer_batch_size(sample, model_output or {}, vars(source_batch) if source_batch is not None else {})
    indices = list(batch_indices) if batch_indices is not None else list(range(batch_size))
    return [
        build_debug_record(
            batch_index=batch_index,
            sample=sample,
            model_output=model_output,
            source_batch=source_batch,
            ref_memory=ref_memory,
            execution=execution,
            metadata=metadata,
        )
        for batch_index in indices
    ]


def derive_record(
    record: RhythmV3DebugRecord,
    *,
    local_rate_decay: float = 0.95,
    silence_tau: float = 0.35,
) -> RhythmV3DerivedRecord:
    source_duration = record.source_duration_obs
    speech_mask = None
    silence_mask = None
    if source_duration is not None:
        unit_mask = _ensure_1d_mask(record.unit_mask, source_duration)
        silence_mask = _ensure_1d_mask(record.source_silence_mask, source_duration)
        silence_mask = np.clip(silence_mask, 0.0, 1.0) * np.clip(unit_mask, 0.0, 1.0)
        speech_mask = np.clip(unit_mask - silence_mask, 0.0, 1.0)
    else:
        speech_mask = np.zeros((0,), dtype=np.float32)
        silence_mask = np.zeros((0,), dtype=np.float32)

    source_logdur = None if source_duration is None else _safe_log(source_duration)
    if record.prompt_log_duration is not None:
        prompt_logdur = record.prompt_log_duration.astype(np.float32)
    elif record.prompt_duration_obs is not None:
        prompt_logdur = _safe_log(record.prompt_duration_obs)
    else:
        prompt_logdur = None

    prompt_speech_mask = None
    if record.prompt_duration_obs is not None:
        prompt_valid_mask = _ensure_1d_mask(record.prompt_valid_mask, record.prompt_duration_obs)
        prompt_speech_mask = _ensure_1d_mask(record.prompt_speech_mask, record.prompt_duration_obs)
        prompt_speech_mask = np.clip(prompt_speech_mask, 0.0, 1.0) * np.clip(prompt_valid_mask, 0.0, 1.0)

    global_rate = record.global_rate
    meta = dict(record.metadata or {})
    prompt_domain_valid = _as_float(
        meta.get("g_domain_valid", meta.get("g_valid", float("nan"))),
        default=float("nan"),
    )
    if np.isfinite(prompt_domain_valid) and prompt_domain_valid <= 0.5:
        global_rate = float("nan")
    if global_rate is None and prompt_logdur is not None and prompt_speech_mask is not None:
        valid = prompt_speech_mask > 0.5
        if np.any(valid):
            global_rate = float(np.median(prompt_logdur[valid]))

    source_rate_seq = record.source_rate_seq
    if source_rate_seq is None and source_logdur is not None:
        obs = torch.from_numpy(source_logdur.reshape(1, -1))
        speech = torch.from_numpy(speech_mask.reshape(1, -1))
        unit_mask_np = _ensure_1d_mask(record.unit_mask, source_logdur)
        sealed_mask_np = _ensure_1d_mask(record.sealed_mask, source_logdur)
        boundary_np = (
            None
            if record.source_boundary_cue is None
            else np.asarray(record.source_boundary_cue, dtype=np.float32).reshape(1, -1)
        )
        stability_np = (
            None
            if record.source_run_stability is None
            else np.asarray(record.source_run_stability, dtype=np.float32).reshape(1, -1)
        )
        src_g_cfg = _resolve_src_g_config(meta)
        src_prefix_min_support = int(
            round(
                _as_float(
                    meta.get("src_prefix_min_support", meta.get("rhythm_v3_src_prefix_min_support")),
                    default=3.0,
                )
            )
        )
        src_prefix_stat_mode = normalize_src_prefix_stat_mode(
            meta.get("src_prefix_stat_mode", meta.get("rhythm_v3_src_prefix_stat_mode", "ema"))
        )
        src_rate_init_mode = str(
            meta.get("src_rate_init_mode", meta.get("rhythm_v3_src_rate_init_mode", "first_speech"))
            or "first_speech"
        ).strip().lower()
        drop_edge_runs = int(src_g_cfg["drop_edge_runs"])
        prefix_weight_t = None
        if stability_np is not None and src_g_cfg["variant"] in {"weighted_median", "softclean_wmed", "softclean_wtmean"}:
            prefix_weight_t = torch.from_numpy(stability_np) * speech
        learned_init_rate = None
        if record.source_rate_init_value is not None:
            init_np = np.asarray(record.source_rate_init_value, dtype=np.float32).reshape(-1)
            if init_np.size > 0 and np.isfinite(float(init_np[0])):
                learned_init_rate = torch.as_tensor(
                    init_np[:1].reshape(1, 1),
                    dtype=torch.float32,
                )
        default_init_rate = resolve_default_source_rate_init(
            observed_log=obs.float(),
            speech_mask=speech.float(),
            src_rate_init_mode=src_rate_init_mode,
            learned_init_rate=learned_init_rate,
            auto_fallback="first_speech",
        )
        source_rate_seq_t, _ = build_causal_source_prefix_rate_seq(
            observed_log=obs,
            speech_mask=speech,
            init_rate=None,
            default_init_rate=default_init_rate,
            stat_mode=src_prefix_stat_mode,
            decay=float(local_rate_decay),
            variant=str(src_g_cfg["variant"]),
            trim_ratio=float(src_g_cfg["trim_ratio"]),
            min_support=max(1, src_prefix_min_support),
            weight=prefix_weight_t,
            valid_mask=torch.from_numpy(unit_mask_np.reshape(1, -1)),
            closed_mask=torch.from_numpy(sealed_mask_np.reshape(1, -1)),
            boundary_confidence=(
                None if boundary_np is None else torch.from_numpy(boundary_np)
            ),
            min_boundary_confidence=src_g_cfg["min_boundary_confidence"],
            drop_edge_runs=max(0, drop_edge_runs),
            min_speech_ratio=0.0,
            unit_ids=(
                None
                if record.source_content_units is None
                else torch.from_numpy(np.asarray(record.source_content_units, dtype=np.int64).reshape(1, -1))
            ),
        )
        source_rate_seq = source_rate_seq_t[0].detach().cpu().numpy().astype(np.float32)

    analytic_shift = record.global_shift_analytic
    if analytic_shift is None and global_rate is not None and source_rate_seq is not None:
        analytic_shift = (float(global_rate) - source_rate_seq).astype(np.float32)

    target_logstretch = None
    raw_target_duration = record.unit_duration_proj_raw_tgt if record.unit_duration_proj_raw_tgt is not None else record.unit_duration_tgt
    if raw_target_duration is not None and source_duration is not None:
        target_logstretch = _safe_log(raw_target_duration) - _safe_log(source_duration)

    oracle_bias = None
    oracle_local = None
    oracle_silence_pseudo = None
    if target_logstretch is not None and analytic_shift is not None:
        coarse_weight = record.unit_confidence_coarse_tgt
        speech_valid = speech_mask > 0.5
        if np.any(speech_valid):
            oracle_bias = weighted_median_np(
                target_logstretch[speech_valid] - analytic_shift[speech_valid],
                None if coarse_weight is None else coarse_weight[speech_valid],
            )
            oracle_local = (target_logstretch - analytic_shift - float(oracle_bias)).astype(np.float32)
            oracle_silence_pseudo = np.clip(
                analytic_shift + float(oracle_bias),
                -float(silence_tau),
                float(silence_tau),
            ).astype(np.float32)

    prediction_logstretch = record.unit_logstretch
    prediction_bias = record.global_bias_scalar
    prediction_local = record.local_residual_pred
    if prediction_local is None:
        prediction_local = record.local_residual
    return RhythmV3DerivedRecord(
        speech_mask=speech_mask.astype(np.float32),
        silence_mask=silence_mask.astype(np.float32),
        prompt_speech_mask=None if prompt_speech_mask is None else prompt_speech_mask.astype(np.float32),
        target_logstretch=None if target_logstretch is None else target_logstretch.astype(np.float32),
        source_logdur=None if source_logdur is None else source_logdur.astype(np.float32),
        prompt_logdur=None if prompt_logdur is None else prompt_logdur.astype(np.float32),
        source_rate_seq=None if source_rate_seq is None else source_rate_seq.astype(np.float32),
        global_rate=None if global_rate is None else float(global_rate),
        analytic_shift=None if analytic_shift is None else analytic_shift.astype(np.float32),
        oracle_bias=None if oracle_bias is None else float(oracle_bias),
        oracle_local=None if oracle_local is None else oracle_local.astype(np.float32),
        oracle_silence_pseudo=None if oracle_silence_pseudo is None else oracle_silence_pseudo.astype(np.float32),
        prediction_logstretch=None if prediction_logstretch is None else prediction_logstretch.astype(np.float32),
        prediction_bias=None if prediction_bias is None else float(prediction_bias),
        prediction_local=None if prediction_local is None else prediction_local.astype(np.float32),
    )


def record_summary(
    record: RhythmV3DebugRecord,
    *,
    unit_step_ms: float = DEFAULT_UNIT_STEP_MS,
    local_rate_decay: float = 0.95,
    silence_tau: float = 0.35,
    g_variant: str = "raw_median",
    g_trim_ratio: float = 0.2,
    drop_edge_runs: int = 0,
) -> dict[str, Any]:
    from .review import (
        DEFAULT_GATE_MIN_SPEECH_RATIO,
        _resolve_prompt_domain_stats,
        compute_source_global_rate_for_analysis,
        compute_speech_tempo_for_analysis,
        _resolve_gate0_drop_reason,
        weighted_median,
    )
    from tasks.Conan.rhythm.duration_v3.metrics import (
        cumulative_drift,
        local_silence_delta_share,
        residual_bias_share,
        residual_target_stats,
        silence_leakage,
        speech_weighted_mae,
    )

    derived = derive_record(record, local_rate_decay=local_rate_decay, silence_tau=silence_tau)
    meta = dict(record.metadata or {})
    prompt_g_cfg = _resolve_prompt_g_config(
        meta,
        fallback_variant=g_variant,
        fallback_trim_ratio=g_trim_ratio,
        fallback_drop_edge_runs=drop_edge_runs,
        fallback_min_boundary_confidence=None,
    )
    src_g_cfg = _resolve_src_g_config(
        meta,
        fallback_variant=g_variant,
        fallback_trim_ratio=g_trim_ratio,
        fallback_drop_edge_runs=drop_edge_runs,
        fallback_min_boundary_confidence=meta.get(
            "min_boundary_confidence_for_g",
            meta.get("rhythm_v3_min_boundary_confidence_for_g"),
        ),
    )
    prompt_total = 0.0 if record.prompt_duration_obs is None else float(np.sum(record.prompt_duration_obs))
    prompt_speech = (
        0.0
        if record.prompt_duration_obs is None or derived.prompt_speech_mask is None
        else float(np.sum(record.prompt_duration_obs * derived.prompt_speech_mask))
    )
    source_total = 0.0 if record.source_duration_obs is None else float(np.sum(record.source_duration_obs))
    raw_target_duration = record.unit_duration_proj_raw_tgt if record.unit_duration_proj_raw_tgt is not None else record.unit_duration_tgt
    target_total = 0.0 if raw_target_duration is None else float(np.sum(raw_target_duration))
    exec_total = 0.0 if record.unit_duration_exec is None else float(np.sum(record.unit_duration_exec))
    speech_valid = derived.speech_mask > 0.5
    silence_valid = derived.silence_mask > 0.5
    source_valid_mask = _ensure_1d_mask(record.unit_mask, record.source_duration_obs)
    analysis_source_weight = (
        np.asarray(record.source_run_stability, dtype=np.float32).reshape(-1)
        if src_g_cfg["variant"] in {"weighted_median", "softclean_wmed", "softclean_wtmean"}
        and record.source_run_stability is not None
        else None
    )
    target_weight = record.unit_confidence_tgt
    if target_weight is None:
        target_weight = record.unit_confidence_coarse_tgt
    if target_weight is None:
        target_weight = record.unit_confidence_local_tgt
    if target_weight is None and record.source_duration_obs is not None:
        target_weight = np.ones_like(record.source_duration_obs, dtype=np.float32)
    speech_target = (
        None
        if derived.target_logstretch is None
        else derived.target_logstretch[speech_valid]
    )
    speech_analytic = (
        None
        if derived.analytic_shift is None
        else derived.analytic_shift[speech_valid]
    )
    if record.global_rate is None:
        prompt_speech_for_g = record.prompt_speech_mask
        g_ref, g_compute_status = compute_source_global_rate_for_analysis(
            source_duration_obs=record.prompt_duration_obs,
            source_speech_mask=prompt_speech_for_g,
            source_valid_mask=record.prompt_valid_mask,
            source_weight=record.prompt_global_weight,
            source_closed_mask=record.prompt_closed_mask,
            source_boundary_confidence=record.prompt_boundary_confidence,
            min_boundary_confidence=prompt_g_cfg["min_boundary_confidence"],
            g_variant=str(prompt_g_cfg["variant"]),
            g_trim_ratio=float(prompt_g_cfg["trim_ratio"]),
            drop_edge_runs=int(prompt_g_cfg["drop_edge_runs"]),
            source_unit_ids=record.prompt_content_units,
            require_explicit_speech_mask=True,
            return_status=True,
        )
    else:
        g_ref = float(record.global_rate)
        g_compute_status = "ok" if np.isfinite(g_ref) else "invalid:record.global_rate"
    g_src_utt, g_src_compute_status = compute_source_global_rate_for_analysis(
        source_duration_obs=record.source_duration_obs,
        source_speech_mask=derived.speech_mask,
        source_valid_mask=source_valid_mask,
        source_weight=analysis_source_weight,
        source_closed_mask=record.sealed_mask,
        source_boundary_confidence=record.source_boundary_cue,
        min_boundary_confidence=src_g_cfg["min_boundary_confidence"],
        g_variant=str(src_g_cfg["variant"]),
        g_trim_ratio=float(src_g_cfg["trim_ratio"]),
        drop_edge_runs=int(src_g_cfg["drop_edge_runs"]),
        source_unit_ids=record.source_content_units,
        require_explicit_speech_mask=False,
        return_status=True,
    )
    g_src_prefix_mean = float("nan")
    if derived.source_rate_seq is not None and bool(np.any(speech_valid)):
        g_src_prefix_mean = float(np.nanmean(derived.source_rate_seq[speech_valid]))
    g_src_prefix_final = _as_float(
        record.g_src_prefix_final,
        default=_last_masked_value(derived.source_rate_seq, derived.speech_mask),
    )
    delta_g = float(g_ref - g_src_utt) if np.isfinite(g_ref) and np.isfinite(g_src_utt) else float("nan")
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
    c_star = np.nan if derived.oracle_bias is None else float(derived.oracle_bias)
    zbar_sp_star = float("nan")
    if derived.target_logstretch is not None and bool(np.any(speech_valid)):
        weight = np.asarray(target_weight, dtype=np.float32).reshape(-1)
        zbar_sp_star = weighted_median(derived.target_logstretch[speech_valid], weight[speech_valid])
    tempo_src = compute_speech_tempo_for_analysis(
        source_duration_obs=record.source_duration_obs,
        source_speech_mask=derived.speech_mask,
        source_valid_mask=source_valid_mask,
        source_weight=analysis_source_weight,
        source_closed_mask=record.sealed_mask,
        source_boundary_confidence=record.source_boundary_cue,
        min_boundary_confidence=src_g_cfg["min_boundary_confidence"],
        g_variant=str(src_g_cfg["variant"]),
        g_trim_ratio=float(src_g_cfg["trim_ratio"]),
        drop_edge_runs=int(src_g_cfg["drop_edge_runs"]),
        source_unit_ids=record.source_content_units,
    )
    tempo_ref_runtime = float(np.exp(-float(g_ref))) if np.isfinite(g_ref) else float("nan")
    tempo_out_raw = compute_speech_tempo_for_analysis(
        source_duration_obs=record.unit_duration_raw,
        source_speech_mask=derived.speech_mask,
        source_valid_mask=source_valid_mask,
        source_weight=analysis_source_weight,
        source_closed_mask=record.sealed_mask,
        source_boundary_confidence=record.source_boundary_cue,
        min_boundary_confidence=src_g_cfg["min_boundary_confidence"],
        g_variant=str(src_g_cfg["variant"]),
        g_trim_ratio=float(src_g_cfg["trim_ratio"]),
        drop_edge_runs=int(src_g_cfg["drop_edge_runs"]),
        source_unit_ids=record.source_content_units,
    ) if record.unit_duration_raw is not None else float("nan")
    continuous_duration = _as_optional_vector(
        record.projector_preclamp_duration_exec
        if record.projector_preclamp_duration_exec is not None
        else (
            record.projector_preclamp_exec
            if record.projector_preclamp_exec is not None
            else record.unit_duration_raw
        ),
        dtype=np.float32,
    )
    tempo_out_continuous = compute_speech_tempo_for_analysis(
        source_duration_obs=continuous_duration,
        source_speech_mask=derived.speech_mask,
        source_valid_mask=source_valid_mask,
        source_weight=analysis_source_weight,
        source_closed_mask=record.sealed_mask,
        source_boundary_confidence=record.source_boundary_cue,
        min_boundary_confidence=src_g_cfg["min_boundary_confidence"],
        g_variant=str(src_g_cfg["variant"]),
        g_trim_ratio=float(src_g_cfg["trim_ratio"]),
        drop_edge_runs=int(src_g_cfg["drop_edge_runs"]),
        source_unit_ids=record.source_content_units,
    )
    tempo_out_preproj = tempo_out_continuous
    tempo_out_exec = compute_speech_tempo_for_analysis(
        source_duration_obs=record.unit_duration_exec,
        source_speech_mask=derived.speech_mask,
        source_valid_mask=source_valid_mask,
        source_weight=analysis_source_weight,
        source_closed_mask=record.sealed_mask,
        source_boundary_confidence=record.source_boundary_cue,
        min_boundary_confidence=src_g_cfg["min_boundary_confidence"],
        g_variant=str(src_g_cfg["variant"]),
        g_trim_ratio=float(src_g_cfg["trim_ratio"]),
        drop_edge_runs=int(src_g_cfg["drop_edge_runs"]),
        source_unit_ids=record.source_content_units,
    )
    tempo_out = tempo_out_exec
    source_duration_arr = _as_optional_vector(record.source_duration_obs, dtype=np.float32)

    def _speech_exec_surface(duration_arr: np.ndarray | None) -> tuple[float, float]:
        if duration_arr is None or source_duration_arr is None or not bool(np.any(speech_valid)):
            return float("nan"), float("nan")
        width = min(
            int(duration_arr.shape[0]),
            int(source_duration_arr.shape[0]),
            int(speech_valid.shape[0]),
        )
        if width <= 0 or not bool(np.any(speech_valid[:width])):
            return float("nan"), float("nan")
        src = source_duration_arr[:width].astype(np.float32)
        dst = duration_arr[:width].astype(np.float32)
        speech_row = speech_valid[:width]
        src_speech = src[speech_row]
        dst_speech = dst[speech_row]
        total_src = float(np.sum(src_speech))
        total_dst = float(np.sum(dst_speech))
        ratio = float("nan") if total_src <= 1.0e-3 else float(total_dst / total_src)
        weights = np.clip(src_speech, 1.0e-3, None)
        logstretch = np.log(np.clip(dst_speech, 1.0e-3, None)) - np.log(weights)
        mean_logstretch = float(np.sum(logstretch * weights) / np.sum(weights))
        return ratio, mean_logstretch

    speech_exec_ratio_continuous, mean_speech_logstretch_continuous = _speech_exec_surface(
        continuous_duration
    )
    speech_exec_ratio_projected, mean_speech_logstretch_projected = _speech_exec_surface(
        _as_optional_vector(record.unit_duration_exec, dtype=np.float32)
    )
    analytic_gap_clip_value = _as_float(
        record.analytic_gap_clip_value,
        default=_as_float(meta.get("analytic_gap_clip"), default=0.35),
    )
    analytic_gap_preclip_abs_mean = np.nan
    analytic_saturation_rate = np.nan
    analytic_gap_raw_abs_mean = np.nan
    tempo_out_preclip = float("nan")
    if derived.source_rate_seq is not None and np.isfinite(g_ref) and bool(np.any(speech_valid)):
        analytic_gap_preclip = (float(g_ref) - derived.source_rate_seq).astype(np.float32)
        analytic_gap_preclip_abs_mean = float(np.mean(np.abs(analytic_gap_preclip[speech_valid])))
        if np.isfinite(analytic_gap_clip_value) and analytic_gap_clip_value > 0.0:
            analytic_saturation_rate = float(
                np.mean(np.abs(analytic_gap_preclip[speech_valid]) >= (analytic_gap_clip_value - 1.0e-6))
            )
        else:
            analytic_saturation_rate = 0.0
        if source_duration_arr is not None:
            width = min(
                int(source_duration_arr.shape[0]),
                int(analytic_gap_preclip.shape[0]),
                int(derived.speech_mask.shape[0]),
            )
            preclip_duration = source_duration_arr[:width].astype(np.float32, copy=True)
            speech_prefix = (derived.speech_mask[:width] > 0.5)
            preclip_duration[speech_prefix] = (
                preclip_duration[speech_prefix] * np.exp(analytic_gap_preclip[:width][speech_prefix])
            ).astype(np.float32)
            valid_prefix_mask = None if source_valid_mask is None else np.asarray(source_valid_mask, dtype=np.float32).reshape(-1)[:width]
            tempo_out_preclip = compute_speech_tempo_for_analysis(
                source_duration_obs=preclip_duration,
                source_speech_mask=derived.speech_mask[:width],
                source_valid_mask=valid_prefix_mask,
                source_weight=(
                    None if analysis_source_weight is None else analysis_source_weight[:width]
                ),
                source_closed_mask=(
                    None
                    if record.sealed_mask is None
                    else np.asarray(record.sealed_mask, dtype=np.float32).reshape(-1)[:width]
                ),
                source_boundary_confidence=(
                    None
                    if record.source_boundary_cue is None
                    else np.asarray(record.source_boundary_cue, dtype=np.float32).reshape(-1)[:width]
                ),
                min_boundary_confidence=src_g_cfg["min_boundary_confidence"],
                g_variant=str(src_g_cfg["variant"]),
                g_trim_ratio=float(src_g_cfg["trim_ratio"]),
                drop_edge_runs=int(src_g_cfg["drop_edge_runs"]),
                source_unit_ids=(
                    None
                    if record.source_content_units is None
                    else np.asarray(record.source_content_units).reshape(-1)[:width]
                ),
            )
    if record.analytic_gap_raw is not None and bool(np.any(speech_valid)):
        analytic_gap_raw_arr = np.asarray(record.analytic_gap_raw, dtype=np.float32).reshape(-1)
        width = min(int(analytic_gap_raw_arr.shape[0]), int(speech_valid.shape[0]))
        if width > 0 and bool(np.any(speech_valid[:width])):
            analytic_gap_raw_abs_mean = float(np.mean(np.abs(analytic_gap_raw_arr[:width][speech_valid[:width]])))
    elif np.isfinite(analytic_gap_preclip_abs_mean):
        analytic_gap_raw_abs_mean = float(analytic_gap_preclip_abs_mean)
    analytic_clip_hit_rate = _as_float(
        record.analytic_clip_hit_rate,
        default=(
            float(np.mean(np.asarray(record.analytic_clip_hit, dtype=np.float32).reshape(-1)[speech_valid] > 0.5))
            if record.analytic_clip_hit is not None and bool(np.any(speech_valid))
            else analytic_saturation_rate
        ),
    )
    analytic_gap_abs_mean = (
        float(np.mean(np.abs(derived.analytic_shift[speech_valid])))
        if derived.analytic_shift is not None and bool(np.any(speech_valid))
        else np.nan
    )
    projector_bucket_count = np.nan
    projected_duration_arr = _as_optional_vector(record.unit_duration_exec, dtype=np.float32)
    if projected_duration_arr is not None and bool(np.any(speech_valid)):
        width = min(int(projected_duration_arr.shape[0]), int(speech_valid.shape[0]))
        if width > 0 and bool(np.any(speech_valid[:width])):
            projector_bucket_count = float(
                np.unique(np.round(projected_duration_arr[:width][speech_valid[:width]]).astype(np.int32)).size
            )
    coarse_bias_abs_mean = np.nan
    if record.coarse_correction is not None and bool(np.any(speech_valid)):
        coarse_arr = np.asarray(record.coarse_correction, dtype=np.float32).reshape(-1)
        coarse_bias_abs_mean = float(np.mean(np.abs(coarse_arr[speech_valid])))
    elif record.global_bias_scalar is not None:
        coarse_bias_abs_mean = float(abs(record.global_bias_scalar))
    prediction_local = (
        np.asarray(record.local_residual_pred, dtype=np.float32).reshape(-1)
        if record.local_residual_pred is not None
        else (
            np.asarray(record.local_residual, dtype=np.float32).reshape(-1)
            if record.local_residual is not None
            else None
        )
    )
    local_residual_abs_mean = (
        float(np.mean(np.abs(prediction_local[speech_valid])))
        if prediction_local is not None and bool(np.any(speech_valid))
        else np.nan
    )
    coarse_scalar_raw = _as_float(
        record.coarse_scalar_raw,
        default=(np.nan if derived.prediction_bias is None else float(derived.prediction_bias)),
    )
    residual_gate_mean = _as_float(record.residual_gate_mean, default=np.nan)
    global_term_before_local = _as_optional_vector(record.global_term_before_local, dtype=np.float32)
    if global_term_before_local is None and derived.analytic_shift is not None:
        global_term_before_local = derived.analytic_shift.astype(np.float32, copy=True)
        if derived.prediction_bias is not None:
            global_term_before_local = global_term_before_local + float(derived.prediction_bias)
    residual_stats = {
        "spearman": float("nan"),
        "robust_slope": float("nan"),
        "r2_like": float("nan"),
        "count": 0.0,
    }
    if prediction_local is not None and derived.oracle_local is not None:
        residual_stats = residual_target_stats(
            prediction_local,
            derived.oracle_local,
            derived.speech_mask,
        )
    residual_bias_ratio = float("nan")
    if prediction_local is not None and np.isfinite(coarse_scalar_raw):
        residual_bias_ratio = float(
            residual_bias_share(prediction_local, derived.speech_mask, [coarse_scalar_raw]).item()
        )
    local_delta_share = float("nan")
    if prediction_local is not None:
        if global_term_before_local is not None and derived.prediction_logstretch is not None:
            local_delta_share = float(
                local_silence_delta_share(
                    derived.prediction_logstretch,
                    global_term_before_local,
                    derived.speech_mask,
                    derived.silence_mask,
                ).item()
            )
        else:
            local_delta_share = float(
                silence_leakage(prediction_local, derived.speech_mask, derived.silence_mask).item()
            )
    correction_delta = None
    if derived.prediction_logstretch is not None and derived.analytic_shift is not None:
        correction_delta = derived.prediction_logstretch - derived.analytic_shift
    leakage = (
        float(silence_leakage(correction_delta, derived.speech_mask, derived.silence_mask).item())
        if correction_delta is not None
        else np.nan
    )
    cumulative_prefix_drift = float(cumulative_drift(record.prefix_unit_offset).item())
    budget_hit_pos_rate = (
        float(np.mean(record.projector_budget_hit_pos.reshape(-1) > 0.5))
        if record.projector_budget_hit_pos is not None and record.projector_budget_hit_pos.size > 0
        else np.nan
    )
    budget_hit_neg_rate = (
        float(np.mean(record.projector_budget_hit_neg.reshape(-1) > 0.5))
        if record.projector_budget_hit_neg is not None and record.projector_budget_hit_neg.size > 0
        else np.nan
    )
    budget_hit_any_rate = float("nan")
    if np.isfinite(budget_hit_pos_rate) or np.isfinite(budget_hit_neg_rate):
        budget_hit_any_rate = float(
            np.nanmax(np.asarray([budget_hit_pos_rate, budget_hit_neg_rate], dtype=np.float32))
        )
    sample_id = str(meta.get("sample_id", record.item_name or ""))
    pair_id = str(meta.get("pair_id", meta.get("rhythm_pair_group_id", sample_id)))
    src_prompt_id = str(meta.get("src_prompt_id", meta.get("source_item_name", record.item_name or "")))
    tgt_prompt_id = str(meta.get("tgt_prompt_id", meta.get("paired_target_item_name", "")))
    ref_prompt_id = str(meta.get("ref_prompt_id", meta.get("ref_item_name", "")))
    src_spk = str(meta.get("src_spk", _infer_speaker_id(src_prompt_id)))
    tgt_spk = str(meta.get("tgt_spk", _infer_speaker_id(tgt_prompt_id)))
    ref_spk = str(meta.get("ref_spk", _infer_speaker_id(ref_prompt_id)))
    same_text_reference = meta.get("same_text_reference", meta.get("same_text", None))
    if same_text_reference is None:
        source_sig = meta.get("source_text_signature")
        ref_sig = meta.get("reference_text_signature")
        if source_sig is not None and ref_sig is not None:
            same_text_reference = int(source_sig == ref_sig)
        elif src_prompt_id and ref_prompt_id:
            same_text_reference = int(src_prompt_id == ref_prompt_id)
    same_text_target = meta.get("same_text_target", None)
    if same_text_target is None:
        source_sig = meta.get("source_text_signature")
        tgt_sig = meta.get("paired_target_text_signature")
        if source_sig is not None and tgt_sig is not None:
            same_text_target = int(source_sig == tgt_sig)
        elif src_prompt_id and tgt_prompt_id:
            same_text_target = int(src_prompt_id == tgt_prompt_id)
    same_speaker_reference = meta.get("same_speaker_reference", meta.get("same_speaker", None))
    if same_speaker_reference is None and src_spk and ref_spk:
        same_speaker_reference = int(src_spk == ref_spk)
    same_speaker_target = meta.get("same_speaker_target", None)
    if same_speaker_target is None and src_spk and tgt_spk:
        same_speaker_target = int(src_spk == tgt_spk)
    lexical_mismatch = np.nan
    if meta.get("lexical_mismatch") is not None:
        try:
            lexical_mismatch = float(meta.get("lexical_mismatch"))
        except Exception:
            lexical_mismatch = np.nan
    final_prefix_offset = (
        np.nan
        if record.prefix_unit_offset is None or record.prefix_unit_offset.size <= 0
        else float(record.prefix_unit_offset.reshape(-1)[-1])
    )
    prompt_speech_explicit = _as_optional_vector(record.prompt_speech_mask, dtype=np.float32)
    min_prompt_speech_ratio = _as_float(
        meta.get("g_min_speech_ratio", meta.get("min_prompt_speech_ratio", DEFAULT_GATE_MIN_SPEECH_RATIO)),
        default=DEFAULT_GATE_MIN_SPEECH_RATIO,
    )
    prompt_domain_stats = _resolve_prompt_domain_stats(
        prompt_duration=_as_optional_vector(record.prompt_duration_obs, dtype=np.float32),
        prompt_speech=prompt_speech_explicit,
        speech_ratio_hint=meta.get("prompt_speech_ratio", meta.get("speech_ratio")),
        support_count_hint=meta.get("g_support_count"),
        valid_count_hint=meta.get("g_valid_count"),
        g_valid_hint=meta.get("g_valid_support", meta.get("g_valid")),
        g_domain_valid_hint=meta.get("g_domain_valid"),
        min_speech_ratio_hint=meta.get("g_min_speech_ratio", meta.get("min_prompt_speech_ratio")),
        default_min_speech_ratio=min_prompt_speech_ratio,
    )
    gate0_drop_reason = _resolve_gate0_drop_reason(
        g_ref=g_ref,
        g_src=g_src_utt,
        c_star=c_star,
        g_compute_status=g_compute_status,
        g_src_compute_status=g_src_compute_status,
        g_domain_valid=prompt_domain_stats["g_domain_valid"],
        speech_ratio=prompt_domain_stats["prompt_speech_ratio"],
        min_speech_ratio=prompt_domain_stats["g_min_speech_ratio"],
        has_explicit_prompt_speech_mask=prompt_speech_explicit is not None,
        require_explicit_prompt_speech_mask=True,
    )
    summary = {
        "sample_id": sample_id,
        "src_id": str(meta.get("src_id", sample_id)),
        "utt_id": str(meta.get("utt_id", meta.get("src_id", sample_id))),
        "pair_id": pair_id,
        "src_spk": src_spk,
        "tgt_spk": tgt_spk,
        "ref_spk": ref_spk,
        "src_prompt_id": src_prompt_id,
        "tgt_prompt_id": tgt_prompt_id,
        "ref_prompt_id": ref_prompt_id,
        "eval_mode": str(meta.get("eval_mode", meta.get("rhythm_v3_eval_mode", ""))),
        "ref_bin": str(meta.get("ref_bin", meta.get("tempo_bin", ""))),
        "g_variant": str(meta.get("g_variant", g_variant)),
        "src_prefix_stat_mode": str(
            meta.get("src_prefix_stat_mode", meta.get("rhythm_v3_src_prefix_stat_mode", ""))
        ),
        "g_ref": g_ref,
        "g_compute_status": g_compute_status,
        "g_src_utt": g_src_utt,
        "g_src_compute_status": g_src_compute_status,
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
        "gate0_row_dropped": 0.0 if gate0_drop_reason == "ok" else 1.0,
        "gate0_drop_reason": gate0_drop_reason,
        "g_support_count": _as_float(meta.get("g_support_count"), default=prompt_domain_stats["g_support_count"]),
        "g_speech_count": _as_float(meta.get("g_speech_count"), default=np.nan),
        "g_valid_count": _as_float(meta.get("g_valid_count"), default=prompt_domain_stats["g_valid_count"]),
        "g_support_ratio_vs_speech": _as_float(meta.get("g_support_ratio_vs_speech"), default=np.nan),
        "g_support_ratio_vs_valid": _as_float(meta.get("g_support_ratio_vs_valid"), default=np.nan),
        "g_valid": _as_float(meta.get("g_valid"), default=np.nan),
        "g_valid_support": prompt_domain_stats["g_valid_support"],
        "g_domain_valid": prompt_domain_stats["g_domain_valid"],
        "g_drop_edge_runs": _as_float(meta.get("g_drop_edge_runs"), default=np.nan),
        "g_strict_speech_only": _as_float(meta.get("g_strict_speech_only"), default=np.nan),
        "g_trim_ratio": _as_float(meta.get("g_trim_ratio"), default=np.nan),
        "g_min_speech_ratio": prompt_domain_stats["g_min_speech_ratio"],
        "prompt_speech_ratio": prompt_domain_stats["prompt_speech_ratio"],
        "prompt_g_speech_ratio_weighted": _as_float(
            meta.get("prompt_g_speech_ratio_weighted"),
            default=prompt_domain_stats["prompt_speech_ratio"],
        ),
        "prompt_g_speech_ratio_count": _as_float(
            meta.get("prompt_g_speech_ratio_count"),
            default=np.nan,
        ),
        "prompt_g_invalid_no_speech": _as_float(meta.get("prompt_g_invalid_no_speech"), default=np.nan),
        "prompt_g_invalid_low_speech_ratio": _as_float(
            meta.get("prompt_g_invalid_low_speech_ratio"),
            default=np.nan,
        ),
        "prompt_g_invalid_ref_len": _as_float(meta.get("prompt_g_invalid_ref_len"), default=np.nan),
        "prompt_g_invalid_support": _as_float(meta.get("prompt_g_invalid_support"), default=np.nan),
        "prompt_g_invalid_clean": _as_float(meta.get("prompt_g_invalid_clean"), default=np.nan),
        "prompt_g_invalid_missing_closed": _as_float(
            meta.get("prompt_g_invalid_missing_closed"),
            default=np.nan,
        ),
        "prompt_g_invalid_missing_boundary": _as_float(
            meta.get("prompt_g_invalid_missing_boundary"),
            default=np.nan,
        ),
        "prompt_speech_mask_explicit": 1.0 if prompt_speech_explicit is not None else 0.0,
        "prompt_global_weight_present": _as_float(meta.get("prompt_global_weight_present"), default=np.nan),
        "prompt_unit_log_prior_present": _as_float(meta.get("prompt_unit_log_prior_present"), default=np.nan),
        "prompt_unit_prior_vocab_size": _as_float(meta.get("prompt_unit_prior_vocab_size"), default=np.nan),
        "c_star": c_star,
        "zbar_sp_star": zbar_sp_star,
        "tempo_src": tempo_src,
        "tempo_ref_runtime": tempo_ref_runtime,
        "tempo_out_raw": tempo_out_raw,
        "tempo_out_preproj": tempo_out_preproj,
        "tempo_out_exec": tempo_out_exec,
        "tempo_out_preclip": tempo_out_preclip,
        "tempo_out_continuous": tempo_out_continuous,
        "tempo_out_preprojector": tempo_out_continuous,
        "tempo_out_projected": tempo_out_exec,
        "tempo_out": tempo_out_exec,
        "speech_exec_ratio_continuous": speech_exec_ratio_continuous,
        "speech_exec_ratio_projected": speech_exec_ratio_projected,
        "speech_exec_ratio": speech_exec_ratio_projected,
        "mean_speech_logstretch_continuous": mean_speech_logstretch_continuous,
        "mean_speech_logstretch_projected": mean_speech_logstretch_projected,
        "mean_speech_logstretch_exec": mean_speech_logstretch_projected,
        "tempo_delta": (
            float(tempo_out - tempo_src)
            if np.isfinite(tempo_out) and np.isfinite(tempo_src)
            else float("nan")
        ),
        "ref_len_sec": float(prompt_total * float(unit_step_ms) / 1000.0),
        "speech_ratio": prompt_domain_stats["prompt_speech_ratio"],
        "same_text": np.nan if same_text_reference is None else float(same_text_reference),
        "same_text_reference": np.nan if same_text_reference is None else float(same_text_reference),
        "same_text_target": np.nan if same_text_target is None else float(same_text_target),
        "same_speaker": np.nan if same_speaker_reference is None else float(same_speaker_reference),
        "same_speaker_reference": np.nan if same_speaker_reference is None else float(same_speaker_reference),
        "same_speaker_target": np.nan if same_speaker_target is None else float(same_speaker_target),
        "alignment_kind": str(meta.get("alignment_kind", "")),
        "alignment_source": str(meta.get("alignment_source", "")),
        "alignment_version": str(meta.get("alignment_version", "")),
        "target_duration_surface": str(meta.get("target_duration_surface", "")),
        "ref_condition": str(meta.get("ref_condition", "")),
        "lexical_mismatch": lexical_mismatch,
        "analytic_gap_clip_value": analytic_gap_clip_value,
        "analytic_gap_preclip_abs_mean": analytic_gap_preclip_abs_mean,
        "analytic_gap_raw_abs_mean": analytic_gap_raw_abs_mean,
        "analytic_gap_runtime_abs_mean": analytic_gap_abs_mean,
        "analytic_gap_abs_mean": analytic_gap_abs_mean,
        "analytic_saturation_rate": analytic_saturation_rate,
        "analytic_gap_clip_hit_rate": analytic_clip_hit_rate,
        "analytic_clip_hit_rate": analytic_clip_hit_rate,
        "coarse_bias_abs_mean": coarse_bias_abs_mean,
        "coarse_scalar_raw": coarse_scalar_raw,
        "coarse_target_abs_err": (
            float(abs(float(derived.prediction_bias) - float(c_star)))
            if derived.prediction_bias is not None and np.isfinite(c_star)
            else np.nan
        ),
        "local_residual_abs_mean": local_residual_abs_mean,
        "residual_gate_mean": residual_gate_mean,
        "detach_global_term_in_local_head": _as_float(
            meta.get("detach_global_term_in_local_head"),
            default=np.nan,
        ),
        "residual_target_corr": residual_stats["spearman"],
        "residual_target_slope": residual_stats["robust_slope"],
        "residual_target_count": residual_stats["count"],
        "residual_bias_share": residual_bias_ratio,
        "local_silence_delta_share": local_delta_share,
        "silence_leakage": leakage,
        "prefix_discrepancy": float(meta.get("prefix_discrepancy")) if meta.get("prefix_discrepancy") is not None else np.nan,
        "budget_hit_pos_rate": budget_hit_pos_rate,
        "budget_hit_neg_rate": budget_hit_neg_rate,
        "budget_hit_any_rate": budget_hit_any_rate,
        "projector_boundary_hit_rate": (
            float(np.mean(record.projector_boundary_hit.reshape(-1) > 0.5))
            if record.projector_boundary_hit is not None and record.projector_boundary_hit.size > 0
            else np.nan
        ),
        "projector_boundary_decay_rate": (
            float(np.mean(record.projector_boundary_decay_applied.reshape(-1) > 0.5))
            if getattr(record, "projector_boundary_decay_applied", None) is not None
            and record.projector_boundary_decay_applied.size > 0
            else np.nan
        ),
        "projector_bucket_count": projector_bucket_count,
        "cumulative_drift": cumulative_prefix_drift,
        "item_name": record.item_name or "",
        "split": record.split or "",
        "reference_seconds": float(prompt_total * float(unit_step_ms) / 1000.0),
        "reference_speech_ratio": prompt_domain_stats["prompt_speech_ratio"],
        "source_total_units": float(source_total),
        "target_total_units": float(target_total),
        "exec_total_units": float(exec_total),
        "projection_conservation_error": float(target_total - exec_total),
        "source_run_count": 0 if record.source_duration_obs is None else int(record.source_duration_obs.shape[0]),
        "commit_ratio": (
            0.0
            if record.commit_mask is None
            else float(np.mean(np.clip(record.commit_mask, 0.0, 1.0)))
        ),
        "alignment_coverage_mean": (
            np.nan
            if record.unit_alignment_coverage_tgt is None
            else float(np.mean(record.unit_alignment_coverage_tgt))
        ),
        "alignment_match_mean": (
            np.nan
            if record.unit_alignment_match_tgt is None
            else float(np.mean(record.unit_alignment_match_tgt))
        ),
        "alignment_cost_mean": (
            np.nan
            if record.unit_alignment_cost_tgt is None
            else float(np.mean(record.unit_alignment_cost_tgt))
        ),
        "alignment_unmatched_speech_ratio": (
            np.nan
            if record.unit_alignment_unmatched_speech_ratio_tgt is None
            else float(np.asarray(record.unit_alignment_unmatched_speech_ratio_tgt).reshape(-1)[0])
        ),
        "alignment_mean_local_confidence_speech": (
            np.nan
            if record.unit_alignment_mean_local_confidence_speech_tgt is None
            else float(np.asarray(record.unit_alignment_mean_local_confidence_speech_tgt).reshape(-1)[0])
        ),
        "alignment_mean_coarse_confidence_speech": (
            np.nan
            if record.unit_alignment_mean_coarse_confidence_speech_tgt is None
            else float(np.asarray(record.unit_alignment_mean_coarse_confidence_speech_tgt).reshape(-1)[0])
        ),
        "global_rate": g_ref,
        "oracle_bias": c_star,
        "predicted_bias": np.nan if derived.prediction_bias is None else float(derived.prediction_bias),
        "speech_target_mean": np.nan if speech_target is None or speech_target.size <= 0 else float(np.mean(speech_target)),
        "speech_analytic_mean": np.nan if speech_analytic is None or speech_analytic.size <= 0 else float(np.mean(speech_analytic)),
        "final_prefix_offset": final_prefix_offset,
        "silence_fraction": float(np.mean(silence_valid.astype(np.float32))) if silence_valid.size > 0 else 0.0,
    }
    if derived.target_logstretch is not None and derived.prediction_logstretch is not None:
        diff = derived.prediction_logstretch - derived.target_logstretch
        summary["speech_mae"] = float(np.mean(np.abs(diff[speech_valid]))) if np.any(speech_valid) else np.nan
        summary["speech_weighted_mae"] = float(
            speech_weighted_mae(
                derived.prediction_logstretch,
                derived.target_logstretch,
                derived.speech_mask,
                target_weight,
            ).item()
        )
        summary["silence_mae"] = float(np.mean(np.abs(diff[silence_valid]))) if np.any(silence_valid) else np.nan
    else:
        summary["speech_mae"] = np.nan
        summary["speech_weighted_mae"] = np.nan
        summary["silence_mae"] = np.nan
    return summary


def save_debug_records(
    records: Sequence[RhythmV3DebugRecord | Mapping[str, Any]],
    path: str | Path,
) -> None:
    payload = [
        record.to_dict() if isinstance(record, RhythmV3DebugRecord) else dict(record)
        for record in records
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_debug_records(path: str | Path) -> list[RhythmV3DebugRecord]:
    path = Path(path)
    if path.is_dir():
        records: list[RhythmV3DebugRecord] = []
        for child in sorted(path.glob("*.pt")):
            records.extend(load_debug_records(child))
        return records
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            payload = [dict(data.items())]
    else:
        raise ValueError(f"Unsupported debug record format: {path}")
    if isinstance(payload, Mapping):
        payload = [payload]
    return [RhythmV3DebugRecord.from_mapping(dict(item)) for item in payload]


__all__ = [
    "DEFAULT_UNIT_STEP_MS",
    "RhythmV3DebugRecord",
    "RhythmV3DerivedRecord",
    "build_debug_record",
    "build_debug_records_from_batch",
    "derive_record",
    "load_debug_records",
    "record_summary",
    "save_debug_records",
    "weighted_median_np",
]
