from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.commons.single_thread_env import apply_single_thread_env

apply_single_thread_env()

import numpy as np
import torch

from modules.Conan.rhythm_v3.g_stats import summarize_global_rate_support
from modules.Conan.rhythm_v3.math_utils import (
    build_causal_source_prefix_rate_seq,
    normalize_src_prefix_stat_mode,
    resolve_default_source_rate_init,
)
from modules.Conan.rhythm_v3.source_cache import build_source_rhythm_cache_v3
from tasks.Conan.dataset import ConanDataset
from tasks.Conan.rhythm.dataset_errors import RhythmDatasetPrefilterDrop
from tasks.Conan.rhythm.duration_v3.metrics import tempo_explainability
from utils.commons.hparams import set_hparams
from utils.plot.rhythm_v3_viz.core import build_debug_records_from_batch, derive_record, weighted_median_np
from utils.plot.rhythm_v3_viz.review import compute_source_global_rate_for_analysis


DEFAULT_CONFIG = "egs/local_arctic_rhythm_v3_quick.yaml"
DEFAULT_SPLITS = "train,valid,test"
DEFAULT_CANDIDATES = "57,71,72,63"
DEFAULT_DROP_EDGES = "1,3"
DEFAULT_REFERENCE_MODES = "target_as_ref"
DEFAULT_OUTPUT_CSV = "tmp/gate1_boundary_audit/counterfactual_static_gate0_rows.csv"
DEFAULT_OUTPUT_JSON = "tmp/gate1_boundary_audit/counterfactual_static_gate0_report.json"
DEFAULT_RELAXED_HPARAMS = (
    "use_pitch_embed=False,"
    "rhythm_v3_gate_quality_strict=False,"
    "rhythm_v3_strict_eval_invalid_g=True,"
    "rhythm_v3_alignment_prefilter_bad_samples=False,"
    "rhythm_v3_alignment_unmatched_speech_ratio_max=1.0,"
    "rhythm_v3_alignment_local_margin_p10_min=0.0,"
    "rhythm_v3_alignment_mean_local_confidence_speech_min=0.0,"
    "rhythm_v3_alignment_mean_coarse_confidence_speech_min=0.0"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pair-level static Gate 0 audit for counterfactual prompt-side silent-token candidates."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Config YAML.")
    parser.add_argument("--splits", default=DEFAULT_SPLITS, help="Comma-separated dataset splits.")
    parser.add_argument("--candidate_tokens", default=DEFAULT_CANDIDATES, help="Comma-separated candidate token ids.")
    parser.add_argument("--drop_edge_runs", default=DEFAULT_DROP_EDGES, help="Comma-separated drop_edge_runs_for_g values.")
    parser.add_argument(
        "--reference_modes",
        default=DEFAULT_REFERENCE_MODES,
        help="Comma-separated prompt reference modes: manifest,target_as_ref.",
    )
    parser.add_argument("--binary_data_dir", default="", help="Optional binary_data_dir override.")
    parser.add_argument("--processed_data_dir", default="", help="Optional processed_data_dir override.")
    parser.add_argument("--hparams", default="", help="Extra hparam overrides appended after relaxed defaults.")
    parser.add_argument("--output_csv", default=DEFAULT_OUTPUT_CSV, help="Where to write flattened pair rows.")
    parser.add_argument("--output_json", default=DEFAULT_OUTPUT_JSON, help="Where to write the structured report.")
    return parser.parse_args()


def _compose_hparams_override(args: argparse.Namespace) -> str:
    parts = [DEFAULT_RELAXED_HPARAMS]
    if args.binary_data_dir:
        parts.append(f"binary_data_dir='{args.binary_data_dir}'")
    if args.processed_data_dir:
        parts.append(f"processed_data_dir='{args.processed_data_dir}'")
    if args.hparams:
        parts.append(args.hparams)
    return ",".join(part for part in parts if part)


def _parse_splits(text: str) -> list[str]:
    splits = [part.strip() for part in str(text).split(",") if part.strip()]
    if not splits:
        raise ValueError("At least one split is required.")
    return splits


def _parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    if not values:
        raise ValueError("At least one integer value is required.")
    return values


def _parse_reference_modes(text: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    valid = {"manifest", "target_as_ref"}
    for part in str(text).split(","):
        value = str(part).strip().lower()
        if not value or value in seen:
            continue
        if value not in valid:
            raise ValueError(f"Unsupported reference_mode={value!r}; expected one of {sorted(valid)}.")
        seen.add(value)
        values.append(value)
    if not values:
        raise ValueError("At least one reference mode is required.")
    return values


def _prompt_tensor(key: str, value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if key == "prompt_content_units" or key.endswith("_vocab_size"):
            return value.long()
        return value.float()
    if key == "prompt_content_units" or key.endswith("_vocab_size"):
        return torch.as_tensor(value, dtype=torch.long)
    return torch.as_tensor(value, dtype=torch.float32)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _weighted_speech_stat(
    values: np.ndarray | None,
    *,
    speech_valid: np.ndarray,
    confidence: np.ndarray | None,
) -> float:
    if values is None:
        return float("nan")
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    width = min(int(values.shape[0]), int(speech_valid.shape[0]))
    if width <= 0:
        return float("nan")
    keep = speech_valid[:width]
    if not bool(np.any(keep)):
        return float("nan")
    row_values = values[:width]
    row_weight = None
    if confidence is not None:
        row_weight = np.asarray(confidence, dtype=np.float32).reshape(-1)[:width]
    return float(weighted_median_np(row_values[keep], None if row_weight is None else row_weight[keep]))


def _weighted_speech_mean_stat(
    values: np.ndarray | None,
    *,
    speech_valid: np.ndarray,
    confidence: np.ndarray | None,
) -> float:
    if values is None:
        return float("nan")
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    width = min(int(values.shape[0]), int(speech_valid.shape[0]))
    if width <= 0:
        return float("nan")
    keep = speech_valid[:width]
    if not bool(np.any(keep)):
        return float("nan")
    row_values = values[:width][keep]
    if confidence is None:
        return float(np.mean(row_values, dtype=np.float64))
    row_weight = np.asarray(confidence, dtype=np.float32).reshape(-1)[:width][keep]
    row_weight = np.clip(row_weight, 0.0, None)
    weight_sum = float(np.sum(row_weight, dtype=np.float64))
    if weight_sum <= 1.0e-6:
        return float(np.mean(row_values, dtype=np.float64))
    return float(np.sum(row_values * row_weight, dtype=np.float64) / weight_sum)


def _speech_total_logratio(
    *,
    source_duration_obs: np.ndarray | None,
    target_duration_obs: np.ndarray | None,
    speech_valid: np.ndarray,
) -> tuple[float, float]:
    if source_duration_obs is None or target_duration_obs is None:
        return float("nan"), float("nan")
    source_duration = np.asarray(source_duration_obs, dtype=np.float32).reshape(-1)
    target_duration = np.asarray(target_duration_obs, dtype=np.float32).reshape(-1)
    width = min(int(source_duration.shape[0]), int(target_duration.shape[0]), int(speech_valid.shape[0]))
    if width <= 0:
        return float("nan"), float("nan")
    keep = np.asarray(speech_valid, dtype=bool)[:width]
    if not bool(np.any(keep)):
        return float("nan"), float("nan")
    src_mass = float(np.sum(source_duration[:width][keep], dtype=np.float64))
    tgt_mass = float(np.sum(target_duration[:width][keep], dtype=np.float64))
    if not np.isfinite(src_mass) or not np.isfinite(tgt_mass) or src_mass <= 1.0e-6 or tgt_mass <= 0.0:
        return float("nan"), float("nan")
    ratio = float(tgt_mass / src_mass)
    return ratio, float(np.log(max(ratio, 1.0e-6)))


def _fit_affine_on_speech_np(
    x: np.ndarray | None,
    y: np.ndarray | None,
    *,
    speech_valid: np.ndarray,
    weight: np.ndarray | None,
    beta1_min: float = 0.0,
    beta1_max: float = 1.5,
    min_count: int = 6,
    min_var: float = 1.0e-4,
) -> tuple[float, float]:
    if x is None or y is None:
        return float("nan"), float("nan")
    x_arr = np.asarray(x, dtype=np.float32).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float32).reshape(-1)
    width = min(int(x_arr.shape[0]), int(y_arr.shape[0]), int(speech_valid.shape[0]))
    if width <= 0:
        return float("nan"), float("nan")
    keep = np.asarray(speech_valid, dtype=bool)[:width]
    if int(np.sum(keep, dtype=np.int64)) < int(min_count):
        return float("nan"), float("nan")
    xv = x_arr[:width][keep].astype(np.float64, copy=False)
    yv = y_arr[:width][keep].astype(np.float64, copy=False)
    if weight is None:
        wv = np.ones_like(xv, dtype=np.float64)
    else:
        wv = np.clip(np.asarray(weight, dtype=np.float32).reshape(-1)[:width][keep], 0.0, None).astype(np.float64, copy=False)
    w_sum = float(np.sum(wv, dtype=np.float64))
    if w_sum <= 1.0e-8:
        wv = np.ones_like(xv, dtype=np.float64)
        w_sum = float(np.sum(wv, dtype=np.float64))
    wv = wv / max(w_sum, 1.0e-8)
    mx = float(np.sum(wv * xv, dtype=np.float64))
    my = float(np.sum(wv * yv, dtype=np.float64))
    x0 = xv - mx
    y0 = yv - my
    var_x = float(np.sum(wv * x0 * x0, dtype=np.float64))
    if not np.isfinite(var_x) or var_x < float(min_var):
        return float(my), 1.0
    cov_xy = float(np.sum(wv * x0 * y0, dtype=np.float64))
    beta1 = float(np.clip(cov_xy / max(var_x, min_var), beta1_min, beta1_max))
    beta0 = float(my - (beta1 * mx))
    return beta0, beta1


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _is_valid_ref_len(value: float, *, min_ref_len_sec: float, max_ref_len_sec: float) -> bool:
    return bool(np.isfinite(value) and value >= min_ref_len_sec and value <= max_ref_len_sec)


def _build_counterfactual_conditioning(
    *,
    ds: ConanDataset,
    raw_ref_item: dict[str, Any],
    candidate_token: int,
) -> dict[str, Any]:
    hubert = np.asarray(raw_ref_item["hubert"])
    cache = build_source_rhythm_cache_v3(
        hubert,
        silent_token=int(candidate_token),
        separator_aware=bool(ds.hparams.get("rhythm_separator_aware", True)),
        tail_open_units=int(ds.hparams.get("rhythm_tail_open_units", 1)),
        emit_silence_runs=bool(ds.hparams.get("rhythm_v3_emit_silence_runs", True)),
        debounce_min_run_frames=int(ds.hparams.get("rhythm_v3_debounce_min_run_frames", 2)),
        phrase_boundary_threshold=float(ds.hparams.get("rhythm_source_phrase_threshold", 0.55)),
        unit_prior_path=None,
        mel=raw_ref_item.get("mel"),
    )
    prompt_item = dict(raw_ref_item)
    prompt_item.update(cache)
    return ds._build_reference_prompt_unit_conditioning(
        prompt_item,
        target_mode=ds._resolve_rhythm_target_mode(),
    )


def _compute_pair_row(
    *,
    ds: ConanDataset,
    split: str,
    fetch_index: int,
    candidate_token: int,
    drop_edge_runs: int,
    reference_mode: str,
) -> dict[str, Any]:
    pair_entry = getattr(ds, "_pair_entries", [])[fetch_index]
    src_local = int(pair_entry["src_local"])
    ref_local = int(pair_entry["ref_local"])
    tgt_local = int(pair_entry["target_local"])

    raw_source_item = ds._get_raw_item_cached(src_local)
    raw_target_item = ds._get_raw_item_cached(tgt_local)
    if str(reference_mode) == "target_as_ref":
        raw_ref_item = raw_target_item
        ref_local_resolved = tgt_local
    else:
        raw_ref_item = ds._get_raw_item_cached(ref_local)
        ref_local_resolved = ref_local

    same_text_reference = int(ds._same_rhythm_text(raw_source_item, raw_ref_item))
    same_text_target = int(ds._same_rhythm_text(raw_source_item, raw_target_item))
    same_speaker_reference = int(
        str(raw_source_item.get("speaker", "")).strip() == str(raw_ref_item.get("speaker", "")).strip()
    )
    same_speaker_target = int(
        str(raw_source_item.get("speaker", "")).strip() == str(raw_target_item.get("speaker", "")).strip()
    )
    try:
        sample = copy.deepcopy(ds[fetch_index])
    except RuntimeError as exc:
        protocol_misaligned = int(
            same_text_reference == 0
            and same_text_target == 1
            and same_speaker_reference == 1
            and same_speaker_target == 0
        )
        return {
            "split": str(split),
            "sample_id": f"{split}:{fetch_index}",
            "pair_id": int(pair_entry.get("group_id", fetch_index)),
            "src_id": str(raw_source_item.get("item_name", "")),
            "src_item_name": str(raw_source_item.get("item_name", "")),
            "ref_item_name": str(raw_ref_item.get("item_name", "")),
            "tgt_item_name": str(raw_target_item.get("item_name", "")),
            "src_spk": str(raw_source_item.get("speaker", "")),
            "ref_spk": str(raw_ref_item.get("speaker", "")),
            "tgt_spk": str(raw_target_item.get("speaker", "")),
            "candidate_token": int(candidate_token),
            "drop_edge_runs_for_g": int(drop_edge_runs),
            "reference_mode": str(reference_mode),
            "same_text_reference": same_text_reference,
            "same_text_target": same_text_target,
            "same_speaker_reference": same_speaker_reference,
            "same_speaker_target": same_speaker_target,
            "protocol_misaligned": int(protocol_misaligned),
            "protocol_slice": (
                "cross_text_prompt_vs_cross_speaker_target"
                if protocol_misaligned > 0
                else "clean_total_claim"
            ),
            "src_prefix_stat_mode": str(ds.hparams.get("rhythm_v3_src_prefix_stat_mode", "ema")),
            "g_ref": float("nan"),
            "g_ref_status": "sample_fetch_error",
            "g_src_utt": float("nan"),
            "g_src_prefix_mean": float("nan"),
            "g_src_prefix_final": float("nan"),
            "g_src_status": "sample_fetch_error",
            "delta_g": float("nan"),
            "delta_g_ref_minus_src_utt": float("nan"),
            "delta_g_ref_minus_src_prefix": float("nan"),
            "delta_g_ref_minus_src_prefix_final": float("nan"),
            "abar_sp_star": float("nan"),
            "abar_sp_star_runtime": float("nan"),
            "amean_sp_star": float("nan"),
            "amean_sp_star_runtime": float("nan"),
            "c_star": float("nan"),
            "c_star_runtime": float("nan"),
            "c_star_runtime_affine": float("nan"),
            "cmean_sp_star": float("nan"),
            "cmean_sp_star_runtime": float("nan"),
            "cmean_sp_star_runtime_affine": float("nan"),
            "zbar_sp_star": float("nan"),
            "zmean_sp_star": float("nan"),
            "speech_total_ratio_star": float("nan"),
            "speech_total_logratio_star": float("nan"),
            "beta0_runtime_fit": float("nan"),
            "beta1_runtime_fit": float("nan"),
            "analytic_gap_clip": float("nan"),
            "analytic_saturation_rate": float("nan"),
            "support_seed_count": 0.0,
            "support_count": 0.0,
            "clean_count": 0.0,
            "support_fraction": 0.0,
            "edge_runs_dropped": 0.0,
            "support_domain_valid": 0,
            "g_domain_valid": 0,
            "prompt_ref_len_sec": float("nan"),
            "prompt_speech_ratio": float("nan"),
            "speech_ratio_valid": 0,
            "ref_len_valid": 0,
            "error": str(exc),
        }
    sample["_raw_item"] = raw_source_item
    sample["_raw_ref_item"] = raw_ref_item
    sample["_raw_paired_target_item"] = raw_target_item
    sample["ref_item_id"] = int(ref_local_resolved)
    try:
        conditioning = _build_counterfactual_conditioning(
            ds=ds,
            raw_ref_item=raw_ref_item,
            candidate_token=int(candidate_token),
        )
    except (RhythmDatasetPrefilterDrop, Exception) as exc:
        protocol_misaligned = int(
            same_text_reference == 0
            and same_text_target == 1
            and same_speaker_reference == 1
            and same_speaker_target == 0
        )
        return {
            "split": str(split),
            "sample_id": f"{split}:{fetch_index}",
            "pair_id": int(pair_entry.get("group_id", fetch_index)),
            "src_id": str(raw_source_item.get("item_name", "")),
            "src_item_name": str(raw_source_item.get("item_name", "")),
            "ref_item_name": str(raw_ref_item.get("item_name", "")),
            "tgt_item_name": str(raw_target_item.get("item_name", "")),
            "src_spk": str(raw_source_item.get("speaker", "")),
            "ref_spk": str(raw_ref_item.get("speaker", "")),
            "tgt_spk": str(raw_target_item.get("speaker", "")),
            "candidate_token": int(candidate_token),
            "drop_edge_runs_for_g": int(drop_edge_runs),
            "reference_mode": str(reference_mode),
            "same_text_reference": same_text_reference,
            "same_text_target": same_text_target,
            "same_speaker_reference": same_speaker_reference,
            "same_speaker_target": same_speaker_target,
            "protocol_misaligned": int(protocol_misaligned),
            "protocol_slice": (
                "cross_text_prompt_vs_cross_speaker_target"
                if protocol_misaligned > 0
                else "clean_total_claim"
            ),
            "src_prefix_stat_mode": str(ds.hparams.get("rhythm_v3_src_prefix_stat_mode", "ema")),
            "g_ref": float("nan"),
            "g_ref_status": f"conditioning_error:{type(exc).__name__}",
            "g_src_utt": float("nan"),
            "g_src_prefix_mean": float("nan"),
            "g_src_prefix_final": float("nan"),
            "g_src_status": "conditioning_error",
            "delta_g": float("nan"),
            "delta_g_ref_minus_src_utt": float("nan"),
            "delta_g_ref_minus_src_prefix": float("nan"),
            "delta_g_ref_minus_src_prefix_final": float("nan"),
            "abar_sp_star": float("nan"),
            "abar_sp_star_runtime": float("nan"),
            "amean_sp_star": float("nan"),
            "amean_sp_star_runtime": float("nan"),
            "c_star": float("nan"),
            "c_star_runtime": float("nan"),
            "c_star_runtime_affine": float("nan"),
            "cmean_sp_star": float("nan"),
            "cmean_sp_star_runtime": float("nan"),
            "cmean_sp_star_runtime_affine": float("nan"),
            "zbar_sp_star": float("nan"),
            "zmean_sp_star": float("nan"),
            "speech_total_ratio_star": float("nan"),
            "speech_total_logratio_star": float("nan"),
            "beta0_runtime_fit": float("nan"),
            "beta1_runtime_fit": float("nan"),
            "analytic_gap_clip": float("nan"),
            "analytic_saturation_rate": float("nan"),
            "support_seed_count": 0.0,
            "support_count": 0.0,
            "clean_count": 0.0,
            "support_fraction": 0.0,
            "edge_runs_dropped": 0.0,
            "support_domain_valid": 0,
            "g_domain_valid": 0,
            "prompt_ref_len_sec": float("nan"),
            "prompt_speech_ratio": float("nan"),
            "speech_ratio_valid": 0,
            "ref_len_valid": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }
    for key in list(sample.keys()):
        if key.startswith("prompt_"):
            sample.pop(key)
    for key, value in conditioning.items():
        sample[key] = _prompt_tensor(key, value)

    batch = ds.collater([sample])
    src_rate_init_mode = str(
        ds.hparams.get("rhythm_v3_src_rate_init_mode", "first_speech") or "first_speech"
    ).strip().lower()
    metadata = {
        "sample_id": f"{split}:{fetch_index}",
        "pair_id": int(pair_entry.get("group_id", fetch_index)),
        "src_id": str(raw_source_item.get("item_name", "")),
        "src_prompt_id": str(raw_source_item.get("item_name", "")),
        "src_spk": str(raw_source_item.get("item_name", "")).split("_", 1)[0],
        "ref_item_name": str(raw_ref_item.get("item_name", "")),
        "ref_prompt_id": str(raw_ref_item.get("item_name", "")),
        "ref_spk": str(raw_ref_item.get("item_name", "")).split("_", 1)[0],
        "tgt_prompt_id": str(raw_target_item.get("item_name", "")),
        "tgt_spk": str(raw_target_item.get("item_name", "")).split("_", 1)[0],
        "same_text_reference": same_text_reference,
        "same_text_target": same_text_target,
        "same_speaker_reference": same_speaker_reference,
        "same_speaker_target": same_speaker_target,
        "src_rate_init_mode": str(src_rate_init_mode),
    }
    record = build_debug_records_from_batch(sample=batch, metadata=metadata)[0]
    derived = derive_record(record)

    speech_mask = np.asarray(derived.speech_mask, dtype=np.float32).reshape(-1)
    speech_valid = speech_mask > 0.5
    source_valid_mask = (
        np.asarray(record.unit_mask, dtype=np.float32).reshape(-1)
        if record.unit_mask is not None
        else np.ones_like(speech_mask, dtype=np.float32)
    )
    target_logstretch = None if derived.target_logstretch is None else np.asarray(derived.target_logstretch, dtype=np.float32).reshape(-1)
    source_rate_seq = None if derived.source_rate_seq is None else np.asarray(derived.source_rate_seq, dtype=np.float32).reshape(-1)

    g_variant = str(ds.hparams.get("rhythm_v3_g_variant", "raw_median"))
    g_trim_ratio = float(ds.hparams.get("rhythm_v3_g_trim_ratio", 0.2) or 0.2)
    src_prefix_stat_mode = normalize_src_prefix_stat_mode(
        ds.hparams.get("rhythm_v3_src_prefix_stat_mode", "ema")
    )
    src_prefix_min_support = int(ds.hparams.get("rhythm_v3_src_prefix_min_support", 3) or 3)
    local_rate_decay = float(ds.hparams.get("rhythm_v3_local_rate_decay", 0.95) or 0.95)
    min_boundary_confidence = ds.hparams.get("rhythm_v3_min_boundary_confidence_for_g", None)
    if min_boundary_confidence is not None:
        min_boundary_confidence = float(min_boundary_confidence)
    min_support_log_iqr = float(ds.hparams.get("rhythm_v3_min_support_log_iqr_for_g", 0.0) or 0.0)
    min_support_log_span = float(ds.hparams.get("rhythm_v3_min_support_log_span_for_g", 0.0) or 0.0)
    min_support_unique = int(ds.hparams.get("rhythm_v3_min_support_unique_for_g", 1) or 1)

    prompt_duration_obs = np.asarray(conditioning["prompt_duration_obs"], dtype=np.float32).reshape(-1)
    prompt_valid_mask = np.asarray(
        conditioning.get("prompt_valid_mask", conditioning.get("prompt_unit_mask")),
        dtype=np.float32,
    ).reshape(-1)
    prompt_speech_mask = np.asarray(conditioning["prompt_speech_mask"], dtype=np.float32).reshape(-1)
    prompt_closed_mask = np.asarray(conditioning["prompt_closed_mask"], dtype=np.float32).reshape(-1)
    prompt_boundary_confidence = np.asarray(conditioning["prompt_boundary_confidence"], dtype=np.float32).reshape(-1)
    prompt_ref_len_sec = _safe_float(np.asarray(conditioning.get("prompt_ref_len_sec")).reshape(-1)[0] if conditioning.get("prompt_ref_len_sec") is not None else float("nan"))
    prompt_speech_ratio = _safe_float(
        np.asarray(conditioning.get("prompt_speech_ratio_scalar")).reshape(-1)[0]
        if conditioning.get("prompt_speech_ratio_scalar") is not None
        else float("nan")
    )

    support_stats = summarize_global_rate_support(
        speech_mask=torch.as_tensor(prompt_speech_mask, dtype=torch.float32),
        valid_mask=torch.as_tensor(prompt_valid_mask, dtype=torch.float32),
        duration_obs=torch.as_tensor(prompt_duration_obs, dtype=torch.float32),
        drop_edge_runs=int(drop_edge_runs),
        min_speech_ratio=float(ds.hparams.get("rhythm_v3_min_prompt_speech_ratio", 0.6) or 0.6),
        closed_mask=torch.as_tensor(prompt_closed_mask, dtype=torch.float32),
        boundary_confidence=torch.as_tensor(prompt_boundary_confidence, dtype=torch.float32),
        min_boundary_confidence=min_boundary_confidence,
        min_support_log_iqr=min_support_log_iqr,
        min_support_log_span=min_support_log_span,
        min_support_unique_count=min_support_unique,
    )
    g_ref, g_ref_status = compute_source_global_rate_for_analysis(
        source_duration_obs=prompt_duration_obs,
        source_speech_mask=prompt_speech_mask,
        source_valid_mask=prompt_valid_mask,
        source_weight=conditioning.get("prompt_global_weight"),
        g_variant=g_variant,
        source_unit_ids=np.asarray(conditioning["prompt_content_units"], dtype=np.int64).reshape(-1),
        g_trim_ratio=g_trim_ratio,
        drop_edge_runs=int(drop_edge_runs),
        source_closed_mask=prompt_closed_mask,
        source_boundary_confidence=prompt_boundary_confidence,
        min_boundary_confidence=min_boundary_confidence,
        return_status=True,
    )
    source_support_stats = summarize_global_rate_support(
        speech_mask=torch.as_tensor(speech_mask, dtype=torch.float32),
        valid_mask=torch.as_tensor(source_valid_mask, dtype=torch.float32),
        duration_obs=torch.as_tensor(record.source_duration_obs, dtype=torch.float32),
        drop_edge_runs=int(drop_edge_runs),
        min_speech_ratio=0.0,
        min_speech_runs=max(1, int(src_prefix_min_support)),
        closed_mask=(
            torch.as_tensor(record.sealed_mask, dtype=torch.float32)
            if record.sealed_mask is not None
            else None
        ),
        boundary_confidence=(
            torch.as_tensor(record.source_boundary_cue, dtype=torch.float32)
            if record.source_boundary_cue is not None
            else None
        ),
        min_boundary_confidence=min_boundary_confidence,
        min_support_log_iqr=min_support_log_iqr,
        min_support_log_span=min_support_log_span,
        min_support_unique_count=min_support_unique,
    )
    source_support_mask = source_support_stats.support_mask.detach().cpu().numpy().reshape(-1).astype(bool)
    source_support_count = float(source_support_stats.support_count.reshape(-1)[0].item())
    source_clean_count = float(source_support_stats.clean_count.reshape(-1)[0].item())
    source_support_domain_valid = float(
        (
            source_support_stats.control_valid
            if isinstance(source_support_stats.control_valid, torch.Tensor)
            else source_support_stats.domain_valid
        ).reshape(-1)[0].item()
    )
    g_src_utt, g_src_status = compute_source_global_rate_for_analysis(
        source_duration_obs=record.source_duration_obs,
        source_speech_mask=speech_mask,
        source_valid_mask=source_valid_mask,
        source_weight=(
            np.asarray(record.source_run_stability, dtype=np.float32).reshape(-1)
            if g_variant in {"weighted_median", "softclean_wmed", "softclean_wtmean"}
            and record.source_run_stability is not None
            else None
        ),
        g_variant=g_variant,
        source_unit_ids=record.source_content_units,
        g_trim_ratio=g_trim_ratio,
        drop_edge_runs=int(drop_edge_runs),
        source_closed_mask=record.sealed_mask,
        source_boundary_confidence=record.source_boundary_cue,
        min_boundary_confidence=min_boundary_confidence,
        return_status=True,
    )
    if record.source_rate_seq is None:
        observed_log = torch.log(
            torch.as_tensor(record.source_duration_obs, dtype=torch.float32).reshape(1, -1).clamp_min(1.0e-4)
        ) * torch.as_tensor(source_valid_mask, dtype=torch.float32).reshape(1, -1)
        speech_tensor = torch.as_tensor(speech_mask, dtype=torch.float32).reshape(1, -1)
        valid_tensor = torch.as_tensor(source_valid_mask, dtype=torch.float32).reshape(1, -1)
        default_init_rate = resolve_default_source_rate_init(
            observed_log=observed_log,
            speech_mask=speech_tensor * valid_tensor,
            src_rate_init_mode=src_rate_init_mode,
            learned_init_rate=None,
            auto_fallback="first_speech",
        )
        prefix_weight = None
        if (
            g_variant in {"weighted_median", "softclean_wmed", "softclean_wtmean"}
            and record.source_run_stability is not None
        ):
            prefix_weight = (
                torch.as_tensor(record.source_run_stability, dtype=torch.float32).reshape(1, -1).clamp(0.0, 1.0)
                * speech_tensor
                * valid_tensor
            )
        source_rate_seq_t, _ = build_causal_source_prefix_rate_seq(
            observed_log=observed_log,
            speech_mask=speech_tensor * valid_tensor,
            init_rate=None,
            default_init_rate=default_init_rate,
            stat_mode=src_prefix_stat_mode,
            decay=local_rate_decay,
            variant=g_variant,
            trim_ratio=g_trim_ratio,
            min_support=src_prefix_min_support,
            weight=prefix_weight,
            valid_mask=valid_tensor,
            closed_mask=(
                torch.as_tensor(record.sealed_mask, dtype=torch.float32).reshape(1, -1)
                if record.sealed_mask is not None
                else None
            ),
            boundary_confidence=(
                torch.as_tensor(record.source_boundary_cue, dtype=torch.float32).reshape(1, -1) * valid_tensor
                if record.source_boundary_cue is not None
                else None
            ),
            min_boundary_confidence=min_boundary_confidence,
            drop_edge_runs=int(drop_edge_runs),
            min_speech_ratio=0.0,
            min_support_log_iqr=min_support_log_iqr,
            min_support_log_span=min_support_log_span,
            min_support_unique_count=min_support_unique,
            unit_ids=(
                None
                if record.source_content_units is None
                else torch.as_tensor(record.source_content_units, dtype=torch.long).reshape(1, -1)
            ),
        )
        source_rate_seq = source_rate_seq_t[0].detach().cpu().numpy().astype(np.float32)
    delta_g = float(g_ref - g_src_utt) if np.isfinite(g_ref) and np.isfinite(g_src_utt) else float("nan")
    g_src_prefix_mean = float("nan")
    g_src_prefix_final = float("nan")
    if source_rate_seq is not None and bool(np.any(speech_valid)):
        prefix_keep = speech_valid[: source_rate_seq.shape[0]]
        if source_support_mask.shape[0] > 0:
            prefix_keep = prefix_keep & source_support_mask[: prefix_keep.shape[0]]
        if bool(np.any(prefix_keep)):
            g_src_prefix_mean = float(np.nanmean(source_rate_seq[: prefix_keep.shape[0]][prefix_keep]))
        speech_idx = np.flatnonzero(
            prefix_keep if bool(np.any(prefix_keep)) else speech_valid[: source_rate_seq.shape[0]]
        )
        if speech_idx.size > 0:
            g_src_prefix_final = float(source_rate_seq[int(speech_idx[-1])])
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

    abar_sp_star = float("nan")
    abar_sp_star_runtime = float("nan")
    amean_sp_star = float("nan")
    amean_sp_star_runtime = float("nan")
    c_star = float("nan")
    c_star_runtime = float("nan")
    cmean_sp_star = float("nan")
    cmean_sp_star_runtime = float("nan")
    c_star_runtime_affine = float("nan")
    cmean_sp_star_runtime_affine = float("nan")
    zbar_sp_star = float("nan")
    zmean_sp_star = float("nan")
    speech_total_ratio_star = float("nan")
    speech_total_logratio_star = float("nan")
    beta0_runtime_fit = float("nan")
    beta1_runtime_fit = float("nan")
    analytic_saturation_rate = float("nan")
    analytic_gap_clip = float(ds.hparams.get("rhythm_v3_analytic_gap_clip", 0.35) or 0.0)
    if (
        target_logstretch is not None
        and source_rate_seq is not None
        and np.isfinite(g_ref)
        and bool(np.any(speech_valid))
    ):
        width = min(
            int(target_logstretch.shape[0]),
            int(source_rate_seq.shape[0]),
            int(speech_mask.shape[0]),
        )
        target_duration_obs = None
        if record.unit_duration_proj_raw_tgt is not None:
            target_duration_obs = np.asarray(record.unit_duration_proj_raw_tgt, dtype=np.float32).reshape(-1)
        elif record.unit_duration_tgt is not None:
            target_duration_obs = np.asarray(record.unit_duration_tgt, dtype=np.float32).reshape(-1)
        source_duration_obs = (
            np.asarray(record.source_duration_obs, dtype=np.float32).reshape(-1)
            if record.source_duration_obs is not None
            else None
        )
        target_logstretch = target_logstretch[:width]
        source_rate_seq = source_rate_seq[:width]
        speech_valid = speech_valid[:width]
        stat_valid = speech_valid.copy()
        if source_support_mask.shape[0] > 0:
            stat_valid = stat_valid & source_support_mask[:width]
        confidence = None
        if record.unit_confidence_coarse_tgt is not None:
            confidence = np.asarray(record.unit_confidence_coarse_tgt, dtype=np.float32).reshape(-1)[:width]
        elif record.unit_confidence_tgt is not None:
            confidence = np.asarray(record.unit_confidence_tgt, dtype=np.float32).reshape(-1)[:width]
        analytic_shift_preclip = (float(g_ref) - source_rate_seq).astype(np.float32)
        if analytic_gap_clip > 0.0:
            analytic_shift_runtime = np.clip(
                analytic_shift_preclip,
                -analytic_gap_clip,
                analytic_gap_clip,
            ).astype(np.float32)
            analytic_saturation_rate = float(
                np.mean(np.abs(analytic_shift_preclip[stat_valid]) >= (analytic_gap_clip - 1.0e-6))
            ) if bool(np.any(stat_valid)) else float("nan")
        else:
            analytic_shift_runtime = analytic_shift_preclip
            analytic_saturation_rate = 0.0
        residual = target_logstretch - analytic_shift_preclip
        residual_runtime = target_logstretch - analytic_shift_runtime
        speech_total_ratio_star, speech_total_logratio_star = _speech_total_logratio(
            source_duration_obs=None if source_duration_obs is None else source_duration_obs[:width],
            target_duration_obs=None if target_duration_obs is None else target_duration_obs[:width],
            speech_valid=stat_valid,
        )
        beta0_runtime_fit, beta1_runtime_fit = _fit_affine_on_speech_np(
            analytic_shift_runtime,
            target_logstretch,
            speech_valid=stat_valid,
            weight=confidence,
        )
        if np.isfinite(beta0_runtime_fit) and np.isfinite(beta1_runtime_fit):
            residual_runtime_affine = target_logstretch - (
                float(beta0_runtime_fit) + (float(beta1_runtime_fit) * analytic_shift_runtime)
            )
            c_star_runtime_affine = _weighted_speech_stat(
                residual_runtime_affine,
                speech_valid=stat_valid,
                confidence=confidence,
            )
            cmean_sp_star_runtime_affine = _weighted_speech_mean_stat(
                residual_runtime_affine,
                speech_valid=stat_valid,
                confidence=confidence,
            )
        abar_sp_star = _weighted_speech_stat(
            analytic_shift_preclip,
            speech_valid=stat_valid,
            confidence=confidence,
        )
        abar_sp_star_runtime = _weighted_speech_stat(
            analytic_shift_runtime,
            speech_valid=stat_valid,
            confidence=confidence,
        )
        amean_sp_star = _weighted_speech_mean_stat(
            analytic_shift_preclip,
            speech_valid=stat_valid,
            confidence=confidence,
        )
        amean_sp_star_runtime = _weighted_speech_mean_stat(
            analytic_shift_runtime,
            speech_valid=stat_valid,
            confidence=confidence,
        )
        c_star = _weighted_speech_stat(
            residual,
            speech_valid=stat_valid,
            confidence=confidence,
        )
        c_star_runtime = _weighted_speech_stat(
            residual_runtime,
            speech_valid=stat_valid,
            confidence=confidence,
        )
        cmean_sp_star = _weighted_speech_mean_stat(
            residual,
            speech_valid=stat_valid,
            confidence=confidence,
        )
        cmean_sp_star_runtime = _weighted_speech_mean_stat(
            residual_runtime,
            speech_valid=stat_valid,
            confidence=confidence,
        )
        zbar_sp_star = _weighted_speech_stat(
            target_logstretch,
            speech_valid=stat_valid,
            confidence=confidence,
        )
        zmean_sp_star = _weighted_speech_mean_stat(
            target_logstretch,
            speech_valid=stat_valid,
            confidence=confidence,
        )

    min_prompt_speech_ratio = float(ds.hparams.get("rhythm_v3_min_prompt_speech_ratio", 0.6) or 0.6)
    min_prompt_ref_len_sec = float(ds.hparams.get("rhythm_v3_min_prompt_ref_len_sec", 3.0) or 3.0)
    max_prompt_ref_len_sec = float(ds.hparams.get("rhythm_v3_max_prompt_ref_len_sec", 8.0) or 8.0)
    speech_ratio_valid = bool(np.isfinite(prompt_speech_ratio) and prompt_speech_ratio >= (min_prompt_speech_ratio - 1.0e-6))
    ref_len_valid = _is_valid_ref_len(
        prompt_ref_len_sec,
        min_ref_len_sec=min_prompt_ref_len_sec,
        max_ref_len_sec=max_prompt_ref_len_sec,
    )
    support_count = float(support_stats.support_count.reshape(-1)[0].item())
    support_seed_count = float(support_stats.support_seed_count.reshape(-1)[0].item())
    clean_count = float(support_stats.clean_count.reshape(-1)[0].item())
    support_domain_valid = float(
        (
            support_stats.control_valid
            if isinstance(support_stats.control_valid, torch.Tensor)
            else support_stats.domain_valid
        ).reshape(-1)[0].item()
    )
    source_g_domain_valid = int(
        np.isfinite(g_src_utt)
        and source_clean_count > 0.5
        and source_support_domain_valid > 0.5
    )
    g_domain_valid = int(
        support_count > 0.5
        and clean_count > 0.5
        and support_domain_valid > 0.5
        and source_g_domain_valid > 0
        and speech_ratio_valid
        and ref_len_valid
    )
    protocol_misaligned = int(
        int(metadata["same_text_reference"]) == 0
        and int(metadata["same_text_target"]) == 1
        and int(metadata["same_speaker_reference"]) == 1
        and int(metadata["same_speaker_target"]) == 0
    )
    protocol_slice = (
        "cross_text_prompt_vs_cross_speaker_target"
        if protocol_misaligned > 0
        else "clean_total_claim"
    )

    return {
        "split": split,
        "fetch_index": int(fetch_index),
        "pair_id": int(pair_entry.get("group_id", fetch_index)),
        "src_item_name": str(raw_source_item.get("item_name", "")),
        "ref_item_name": str(raw_ref_item.get("item_name", "")),
        "tgt_item_name": str(raw_target_item.get("item_name", "")),
        "src_spk": str(raw_source_item.get("speaker", "")),
        "ref_spk": str(raw_ref_item.get("speaker", "")),
        "tgt_spk": str(raw_target_item.get("speaker", "")),
        "candidate_token": int(candidate_token),
        "drop_edge_runs_for_g": int(drop_edge_runs),
        "reference_mode": str(reference_mode),
        "same_text_reference": int(metadata["same_text_reference"]),
        "same_text_target": int(metadata["same_text_target"]),
        "same_speaker_reference": int(metadata["same_speaker_reference"]),
        "same_speaker_target": int(metadata["same_speaker_target"]),
        "protocol_misaligned": int(protocol_misaligned),
        "protocol_slice": str(protocol_slice),
        "src_prefix_stat_mode": str(src_prefix_stat_mode),
        "g_ref": float(g_ref),
        "g_ref_status": str(g_ref_status),
        "g_src_utt": float(g_src_utt),
        "g_src_prefix_mean": float(g_src_prefix_mean),
        "g_src_prefix_final": float(g_src_prefix_final),
        "g_src_status": str(g_src_status),
        "delta_g": float(delta_g),
        "delta_g_ref_minus_src_utt": float(delta_g),
        "delta_g_ref_minus_src_prefix": float(delta_g_ref_minus_src_prefix),
        "delta_g_ref_minus_src_prefix_final": float(delta_g_ref_minus_src_prefix_final),
        "abar_sp_star": float(abar_sp_star),
        "abar_sp_star_runtime": float(abar_sp_star_runtime),
        "amean_sp_star": float(amean_sp_star),
        "amean_sp_star_runtime": float(amean_sp_star_runtime),
        "c_star": float(c_star),
        "c_star_runtime": float(c_star_runtime),
        "c_star_runtime_affine": float(c_star_runtime_affine),
        "cmean_sp_star": float(cmean_sp_star),
        "cmean_sp_star_runtime": float(cmean_sp_star_runtime),
        "cmean_sp_star_runtime_affine": float(cmean_sp_star_runtime_affine),
        "zbar_sp_star": float(zbar_sp_star),
        "zmean_sp_star": float(zmean_sp_star),
        "speech_total_ratio_star": float(speech_total_ratio_star),
        "speech_total_logratio_star": float(speech_total_logratio_star),
        "beta0_runtime_fit": float(beta0_runtime_fit),
        "beta1_runtime_fit": float(beta1_runtime_fit),
        "analytic_gap_clip": float(analytic_gap_clip),
        "analytic_saturation_rate": float(analytic_saturation_rate),
        "support_seed_count": support_seed_count,
        "support_count": support_count,
        "source_support_count": float(source_support_count),
        "clean_count": clean_count,
        "source_clean_count": float(source_clean_count),
        "support_fraction": float(support_stats.support_fraction.reshape(-1)[0].item()),
        "edge_runs_dropped": float(support_stats.edge_runs_dropped.reshape(-1)[0].item()),
        "support_log_iqr": float(
            support_stats.support_log_iqr.reshape(-1)[0].item()
            if isinstance(support_stats.support_log_iqr, torch.Tensor)
            else float("nan")
        ),
        "support_log_span": float(
            support_stats.support_log_span.reshape(-1)[0].item()
            if isinstance(support_stats.support_log_span, torch.Tensor)
            else float("nan")
        ),
        "support_unique_count": float(
            support_stats.support_unique_count.reshape(-1)[0].item()
            if isinstance(support_stats.support_unique_count, torch.Tensor)
            else float("nan")
        ),
        "support_domain_valid": support_domain_valid,
        "source_support_domain_valid": float(source_support_domain_valid),
        "source_g_domain_valid": int(source_g_domain_valid),
        "g_domain_valid": int(g_domain_valid),
        "prompt_ref_len_sec": float(prompt_ref_len_sec),
        "prompt_speech_ratio": float(prompt_speech_ratio),
        "speech_ratio_valid": int(speech_ratio_valid),
        "ref_len_valid": int(ref_len_valid),
    }


def _summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = rows

    def _subset(predicate) -> list[dict[str, Any]]:
        return [row for row in frame if predicate(row)]

    def _metric(subset: list[dict[str, Any]], *, x_key: str, y_key: str) -> dict[str, float]:
        filtered = [row for row in subset if x_key in row and y_key in row]
        if not filtered:
            return {
                "spearman": float("nan"),
                "robust_slope": float("nan"),
                "r2_like": float("nan"),
                "count": 0.0,
            }
        return tempo_explainability(
            [row[x_key] for row in filtered],
            [row[y_key] for row in filtered],
        )

    valid_rows = _subset(lambda row: int(row["g_domain_valid"]) > 0)
    cross_rows = _subset(lambda row: int(row["same_text_reference"]) == 0)
    valid_cross_rows = _subset(lambda row: int(row["g_domain_valid"]) > 0 and int(row["same_text_reference"]) == 0)
    same_rows = _subset(lambda row: int(row["same_text_reference"]) > 0)
    clean_total_claim_rows = _subset(lambda row: str(row.get("protocol_slice", "")) == "clean_total_claim")
    valid_clean_total_claim_rows = _subset(
        lambda row: int(row["g_domain_valid"]) > 0 and str(row.get("protocol_slice", "")) == "clean_total_claim"
    )
    hostile_protocol_rows = _subset(
        lambda row: str(row.get("protocol_slice", "")) == "cross_text_prompt_vs_cross_speaker_target"
    )
    valid_hostile_protocol_rows = _subset(
        lambda row: int(row["g_domain_valid"]) > 0 and str(row.get("protocol_slice", "")) == "cross_text_prompt_vs_cross_speaker_target"
    )

    overall_total = _metric(frame, x_key="delta_g_ref_minus_src_utt", y_key="zbar_sp_star")
    valid_total = _metric(valid_rows, x_key="delta_g_ref_minus_src_utt", y_key="zbar_sp_star")
    overall_total_mean = _metric(frame, x_key="delta_g_ref_minus_src_utt", y_key="zmean_sp_star")
    valid_total_mean = _metric(valid_rows, x_key="delta_g_ref_minus_src_utt", y_key="zmean_sp_star")
    overall_total_logratio = _metric(frame, x_key="delta_g_ref_minus_src_utt", y_key="speech_total_logratio_star")
    valid_total_logratio = _metric(valid_rows, x_key="delta_g_ref_minus_src_utt", y_key="speech_total_logratio_star")
    cross_total = _metric(cross_rows, x_key="delta_g_ref_minus_src_utt", y_key="zbar_sp_star")
    valid_cross_total = _metric(valid_cross_rows, x_key="delta_g_ref_minus_src_utt", y_key="zbar_sp_star")
    same_total = _metric(same_rows, x_key="delta_g_ref_minus_src_utt", y_key="zbar_sp_star")
    clean_total_claim_total = _metric(
        clean_total_claim_rows,
        x_key="delta_g_ref_minus_src_utt",
        y_key="zbar_sp_star",
    )
    valid_clean_total_claim_total = _metric(
        valid_clean_total_claim_rows,
        x_key="delta_g_ref_minus_src_utt",
        y_key="zbar_sp_star",
    )
    valid_clean_total_claim_total_mean = _metric(
        valid_clean_total_claim_rows,
        x_key="delta_g_ref_minus_src_utt",
        y_key="zmean_sp_star",
    )
    valid_clean_total_claim_total_logratio = _metric(
        valid_clean_total_claim_rows,
        x_key="delta_g_ref_minus_src_utt",
        y_key="speech_total_logratio_star",
    )
    hostile_protocol_total = _metric(
        hostile_protocol_rows,
        x_key="delta_g_ref_minus_src_utt",
        y_key="zbar_sp_star",
    )
    valid_hostile_protocol_total = _metric(
        valid_hostile_protocol_rows,
        x_key="delta_g_ref_minus_src_utt",
        y_key="zbar_sp_star",
    )
    overall_analytic = _metric(frame, x_key="delta_g_ref_minus_src_utt", y_key="abar_sp_star")
    valid_analytic = _metric(valid_rows, x_key="delta_g_ref_minus_src_utt", y_key="abar_sp_star")
    overall_analytic_mean = _metric(frame, x_key="delta_g_ref_minus_src_utt", y_key="amean_sp_star")
    valid_analytic_mean = _metric(valid_rows, x_key="delta_g_ref_minus_src_utt", y_key="amean_sp_star")
    overall_analytic_runtime = _metric(frame, x_key="delta_g_ref_minus_src_utt", y_key="abar_sp_star_runtime")
    valid_analytic_runtime = _metric(valid_rows, x_key="delta_g_ref_minus_src_utt", y_key="abar_sp_star_runtime")
    overall_analytic_runtime_mean = _metric(frame, x_key="delta_g_ref_minus_src_utt", y_key="amean_sp_star_runtime")
    valid_analytic_runtime_mean = _metric(valid_rows, x_key="delta_g_ref_minus_src_utt", y_key="amean_sp_star_runtime")
    overall_residual = _metric(frame, x_key="delta_g_ref_minus_src_utt", y_key="c_star")
    valid_residual = _metric(valid_rows, x_key="delta_g_ref_minus_src_utt", y_key="c_star")
    overall_residual_mean = _metric(frame, x_key="delta_g_ref_minus_src_utt", y_key="cmean_sp_star")
    valid_residual_mean = _metric(valid_rows, x_key="delta_g_ref_minus_src_utt", y_key="cmean_sp_star")
    overall_residual_runtime = _metric(frame, x_key="delta_g_ref_minus_src_utt", y_key="c_star_runtime")
    valid_residual_runtime = _metric(valid_rows, x_key="delta_g_ref_minus_src_utt", y_key="c_star_runtime")
    overall_residual_runtime_mean = _metric(frame, x_key="delta_g_ref_minus_src_utt", y_key="cmean_sp_star_runtime")
    valid_residual_runtime_mean = _metric(valid_rows, x_key="delta_g_ref_minus_src_utt", y_key="cmean_sp_star_runtime")
    overall_residual_runtime_affine = _metric(frame, x_key="delta_g_ref_minus_src_utt", y_key="c_star_runtime_affine")
    valid_residual_runtime_affine = _metric(valid_rows, x_key="delta_g_ref_minus_src_utt", y_key="c_star_runtime_affine")
    overall_residual_runtime_affine_mean = _metric(
        frame,
        x_key="delta_g_ref_minus_src_utt",
        y_key="cmean_sp_star_runtime_affine",
    )
    valid_residual_runtime_affine_mean = _metric(
        valid_rows,
        x_key="delta_g_ref_minus_src_utt",
        y_key="cmean_sp_star_runtime_affine",
    )
    valid_prefix_total = _metric(valid_rows, x_key="delta_g_ref_minus_src_prefix_final", y_key="zbar_sp_star")
    valid_prefix_analytic = _metric(valid_rows, x_key="delta_g_ref_minus_src_prefix_final", y_key="abar_sp_star")
    valid_prefix_analytic_runtime = _metric(valid_rows, x_key="delta_g_ref_minus_src_prefix_final", y_key="abar_sp_star_runtime")
    valid_prefix_residual = _metric(valid_rows, x_key="delta_g_ref_minus_src_prefix_final", y_key="c_star")
    valid_prefix_residual_runtime = _metric(valid_rows, x_key="delta_g_ref_minus_src_prefix_final", y_key="c_star_runtime")
    return {
        "candidate_token": int(frame[0]["candidate_token"]),
        "drop_edge_runs_for_g": int(frame[0]["drop_edge_runs_for_g"]),
        "reference_mode": str(frame[0].get("reference_mode", "manifest")),
        "src_prefix_stat_mode": str(frame[0].get("src_prefix_stat_mode", "")),
        "item_count": len(frame),
        "finite_pairs_signal": int(sum(1 for row in frame if np.isfinite(float(row["delta_g_ref_minus_src_utt"])) and np.isfinite(float(row["zbar_sp_star"])))),
        "finite_pairs_analytic": int(sum(1 for row in frame if np.isfinite(float(row["delta_g_ref_minus_src_utt"])) and np.isfinite(float(row["abar_sp_star"])))),
        "finite_pairs_residual": int(sum(1 for row in frame if np.isfinite(float(row["delta_g_ref_minus_src_utt"])) and np.isfinite(float(row["c_star"])))),
        "g_domain_valid_items": int(sum(1 for row in frame if int(row["g_domain_valid"]) > 0)),
        "source_g_domain_valid_items": int(sum(1 for row in frame if int(row.get("source_g_domain_valid", 0)) > 0)),
        "same_text_items": int(sum(1 for row in frame if int(row["same_text_reference"]) > 0)),
        "cross_text_items": int(sum(1 for row in frame if int(row["same_text_reference"]) == 0)),
        "clean_total_claim_items": len(clean_total_claim_rows),
        "valid_clean_total_claim_items": len(valid_clean_total_claim_rows),
        "protocol_misaligned_items": len(hostile_protocol_rows),
        "valid_protocol_misaligned_items": len(valid_hostile_protocol_rows),
        "mean_support_count": float(np.mean([float(row["support_count"]) for row in frame])),
        "mean_source_support_count": float(
            np.nanmean(np.asarray([row.get("source_support_count", float("nan")) for row in frame], dtype=np.float32))
        ),
        "mean_clean_count": float(np.mean([float(row["clean_count"]) for row in frame])),
        "mean_source_clean_count": float(
            np.nanmean(np.asarray([row.get("source_clean_count", float("nan")) for row in frame], dtype=np.float32))
        ),
        "mean_prompt_speech_ratio": float(np.nanmean(np.asarray([row["prompt_speech_ratio"] for row in frame], dtype=np.float32))),
        "overall_total_signal_spearman": float(overall_total["spearman"]),
        "overall_total_signal_robust_slope": float(overall_total["robust_slope"]),
        "overall_total_signal_r2_like": float(overall_total["r2_like"]),
        "overall_total_signal_count": float(overall_total["count"]),
        "valid_total_signal_spearman": float(valid_total["spearman"]),
        "valid_total_signal_robust_slope": float(valid_total["robust_slope"]),
        "valid_total_signal_r2_like": float(valid_total["r2_like"]),
        "valid_total_signal_count": float(valid_total["count"]),
        "overall_total_mean_signal_spearman": float(overall_total_mean["spearman"]),
        "overall_total_mean_signal_robust_slope": float(overall_total_mean["robust_slope"]),
        "overall_total_mean_signal_r2_like": float(overall_total_mean["r2_like"]),
        "overall_total_mean_signal_count": float(overall_total_mean["count"]),
        "valid_total_mean_signal_spearman": float(valid_total_mean["spearman"]),
        "valid_total_mean_signal_robust_slope": float(valid_total_mean["robust_slope"]),
        "valid_total_mean_signal_r2_like": float(valid_total_mean["r2_like"]),
        "valid_total_mean_signal_count": float(valid_total_mean["count"]),
        "overall_total_logratio_signal_spearman": float(overall_total_logratio["spearman"]),
        "overall_total_logratio_signal_robust_slope": float(overall_total_logratio["robust_slope"]),
        "overall_total_logratio_signal_r2_like": float(overall_total_logratio["r2_like"]),
        "overall_total_logratio_signal_count": float(overall_total_logratio["count"]),
        "valid_total_logratio_signal_spearman": float(valid_total_logratio["spearman"]),
        "valid_total_logratio_signal_robust_slope": float(valid_total_logratio["robust_slope"]),
        "valid_total_logratio_signal_r2_like": float(valid_total_logratio["r2_like"]),
        "valid_total_logratio_signal_count": float(valid_total_logratio["count"]),
        "cross_total_signal_spearman": float(cross_total["spearman"]),
        "cross_total_signal_robust_slope": float(cross_total["robust_slope"]),
        "cross_total_signal_r2_like": float(cross_total["r2_like"]),
        "cross_total_signal_count": float(cross_total["count"]),
        "valid_cross_total_signal_spearman": float(valid_cross_total["spearman"]),
        "valid_cross_total_signal_robust_slope": float(valid_cross_total["robust_slope"]),
        "valid_cross_total_signal_r2_like": float(valid_cross_total["r2_like"]),
        "valid_cross_total_signal_count": float(valid_cross_total["count"]),
        "same_total_signal_spearman": float(same_total["spearman"]),
        "same_total_signal_robust_slope": float(same_total["robust_slope"]),
        "same_total_signal_r2_like": float(same_total["r2_like"]),
        "same_total_signal_count": float(same_total["count"]),
        "clean_total_claim_signal_spearman": float(clean_total_claim_total["spearman"]),
        "clean_total_claim_signal_robust_slope": float(clean_total_claim_total["robust_slope"]),
        "clean_total_claim_signal_r2_like": float(clean_total_claim_total["r2_like"]),
        "clean_total_claim_signal_count": float(clean_total_claim_total["count"]),
        "valid_clean_total_claim_signal_spearman": float(valid_clean_total_claim_total["spearman"]),
        "valid_clean_total_claim_signal_robust_slope": float(valid_clean_total_claim_total["robust_slope"]),
        "valid_clean_total_claim_signal_r2_like": float(valid_clean_total_claim_total["r2_like"]),
        "valid_clean_total_claim_signal_count": float(valid_clean_total_claim_total["count"]),
        "valid_clean_total_claim_mean_signal_spearman": float(valid_clean_total_claim_total_mean["spearman"]),
        "valid_clean_total_claim_mean_signal_robust_slope": float(valid_clean_total_claim_total_mean["robust_slope"]),
        "valid_clean_total_claim_mean_signal_r2_like": float(valid_clean_total_claim_total_mean["r2_like"]),
        "valid_clean_total_claim_mean_signal_count": float(valid_clean_total_claim_total_mean["count"]),
        "valid_clean_total_claim_logratio_signal_spearman": float(valid_clean_total_claim_total_logratio["spearman"]),
        "valid_clean_total_claim_logratio_signal_robust_slope": float(valid_clean_total_claim_total_logratio["robust_slope"]),
        "valid_clean_total_claim_logratio_signal_r2_like": float(valid_clean_total_claim_total_logratio["r2_like"]),
        "valid_clean_total_claim_logratio_signal_count": float(valid_clean_total_claim_total_logratio["count"]),
        "protocol_misaligned_signal_spearman": float(hostile_protocol_total["spearman"]),
        "protocol_misaligned_signal_robust_slope": float(hostile_protocol_total["robust_slope"]),
        "protocol_misaligned_signal_r2_like": float(hostile_protocol_total["r2_like"]),
        "protocol_misaligned_signal_count": float(hostile_protocol_total["count"]),
        "valid_protocol_misaligned_signal_spearman": float(valid_hostile_protocol_total["spearman"]),
        "valid_protocol_misaligned_signal_robust_slope": float(valid_hostile_protocol_total["robust_slope"]),
        "valid_protocol_misaligned_signal_r2_like": float(valid_hostile_protocol_total["r2_like"]),
        "valid_protocol_misaligned_signal_count": float(valid_hostile_protocol_total["count"]),
        "overall_analytic_signal_spearman": float(overall_analytic["spearman"]),
        "overall_analytic_signal_robust_slope": float(overall_analytic["robust_slope"]),
        "overall_analytic_signal_r2_like": float(overall_analytic["r2_like"]),
        "overall_analytic_signal_count": float(overall_analytic["count"]),
        "valid_analytic_signal_spearman": float(valid_analytic["spearman"]),
        "valid_analytic_signal_robust_slope": float(valid_analytic["robust_slope"]),
        "valid_analytic_signal_r2_like": float(valid_analytic["r2_like"]),
        "valid_analytic_signal_count": float(valid_analytic["count"]),
        "overall_analytic_mean_signal_spearman": float(overall_analytic_mean["spearman"]),
        "overall_analytic_mean_signal_robust_slope": float(overall_analytic_mean["robust_slope"]),
        "overall_analytic_mean_signal_r2_like": float(overall_analytic_mean["r2_like"]),
        "overall_analytic_mean_signal_count": float(overall_analytic_mean["count"]),
        "valid_analytic_mean_signal_spearman": float(valid_analytic_mean["spearman"]),
        "valid_analytic_mean_signal_robust_slope": float(valid_analytic_mean["robust_slope"]),
        "valid_analytic_mean_signal_r2_like": float(valid_analytic_mean["r2_like"]),
        "valid_analytic_mean_signal_count": float(valid_analytic_mean["count"]),
        "overall_analytic_runtime_signal_spearman": float(overall_analytic_runtime["spearman"]),
        "overall_analytic_runtime_signal_robust_slope": float(overall_analytic_runtime["robust_slope"]),
        "overall_analytic_runtime_signal_r2_like": float(overall_analytic_runtime["r2_like"]),
        "overall_analytic_runtime_signal_count": float(overall_analytic_runtime["count"]),
        "valid_analytic_runtime_signal_spearman": float(valid_analytic_runtime["spearman"]),
        "valid_analytic_runtime_signal_robust_slope": float(valid_analytic_runtime["robust_slope"]),
        "valid_analytic_runtime_signal_r2_like": float(valid_analytic_runtime["r2_like"]),
        "valid_analytic_runtime_signal_count": float(valid_analytic_runtime["count"]),
        "overall_analytic_runtime_mean_signal_spearman": float(overall_analytic_runtime_mean["spearman"]),
        "overall_analytic_runtime_mean_signal_robust_slope": float(overall_analytic_runtime_mean["robust_slope"]),
        "overall_analytic_runtime_mean_signal_r2_like": float(overall_analytic_runtime_mean["r2_like"]),
        "overall_analytic_runtime_mean_signal_count": float(overall_analytic_runtime_mean["count"]),
        "valid_analytic_runtime_mean_signal_spearman": float(valid_analytic_runtime_mean["spearman"]),
        "valid_analytic_runtime_mean_signal_robust_slope": float(valid_analytic_runtime_mean["robust_slope"]),
        "valid_analytic_runtime_mean_signal_r2_like": float(valid_analytic_runtime_mean["r2_like"]),
        "valid_analytic_runtime_mean_signal_count": float(valid_analytic_runtime_mean["count"]),
        "overall_residual_signal_spearman": float(overall_residual["spearman"]),
        "overall_residual_signal_robust_slope": float(overall_residual["robust_slope"]),
        "overall_residual_signal_r2_like": float(overall_residual["r2_like"]),
        "overall_residual_signal_count": float(overall_residual["count"]),
        "valid_residual_signal_spearman": float(valid_residual["spearman"]),
        "valid_residual_signal_robust_slope": float(valid_residual["robust_slope"]),
        "valid_residual_signal_r2_like": float(valid_residual["r2_like"]),
        "valid_residual_signal_count": float(valid_residual["count"]),
        "overall_residual_mean_signal_spearman": float(overall_residual_mean["spearman"]),
        "overall_residual_mean_signal_robust_slope": float(overall_residual_mean["robust_slope"]),
        "overall_residual_mean_signal_r2_like": float(overall_residual_mean["r2_like"]),
        "overall_residual_mean_signal_count": float(overall_residual_mean["count"]),
        "valid_residual_mean_signal_spearman": float(valid_residual_mean["spearman"]),
        "valid_residual_mean_signal_robust_slope": float(valid_residual_mean["robust_slope"]),
        "valid_residual_mean_signal_r2_like": float(valid_residual_mean["r2_like"]),
        "valid_residual_mean_signal_count": float(valid_residual_mean["count"]),
        "overall_residual_runtime_signal_spearman": float(overall_residual_runtime["spearman"]),
        "overall_residual_runtime_signal_robust_slope": float(overall_residual_runtime["robust_slope"]),
        "overall_residual_runtime_signal_r2_like": float(overall_residual_runtime["r2_like"]),
        "overall_residual_runtime_signal_count": float(overall_residual_runtime["count"]),
        "valid_residual_runtime_signal_spearman": float(valid_residual_runtime["spearman"]),
        "valid_residual_runtime_signal_robust_slope": float(valid_residual_runtime["robust_slope"]),
        "valid_residual_runtime_signal_r2_like": float(valid_residual_runtime["r2_like"]),
        "valid_residual_runtime_signal_count": float(valid_residual_runtime["count"]),
        "overall_residual_runtime_mean_signal_spearman": float(overall_residual_runtime_mean["spearman"]),
        "overall_residual_runtime_mean_signal_robust_slope": float(overall_residual_runtime_mean["robust_slope"]),
        "overall_residual_runtime_mean_signal_r2_like": float(overall_residual_runtime_mean["r2_like"]),
        "overall_residual_runtime_mean_signal_count": float(overall_residual_runtime_mean["count"]),
        "valid_residual_runtime_mean_signal_spearman": float(valid_residual_runtime_mean["spearman"]),
        "valid_residual_runtime_mean_signal_robust_slope": float(valid_residual_runtime_mean["robust_slope"]),
        "valid_residual_runtime_mean_signal_r2_like": float(valid_residual_runtime_mean["r2_like"]),
        "valid_residual_runtime_mean_signal_count": float(valid_residual_runtime_mean["count"]),
        "overall_residual_runtime_affine_signal_spearman": float(overall_residual_runtime_affine["spearman"]),
        "overall_residual_runtime_affine_signal_robust_slope": float(overall_residual_runtime_affine["robust_slope"]),
        "overall_residual_runtime_affine_signal_r2_like": float(overall_residual_runtime_affine["r2_like"]),
        "overall_residual_runtime_affine_signal_count": float(overall_residual_runtime_affine["count"]),
        "valid_residual_runtime_affine_signal_spearman": float(valid_residual_runtime_affine["spearman"]),
        "valid_residual_runtime_affine_signal_robust_slope": float(valid_residual_runtime_affine["robust_slope"]),
        "valid_residual_runtime_affine_signal_r2_like": float(valid_residual_runtime_affine["r2_like"]),
        "valid_residual_runtime_affine_signal_count": float(valid_residual_runtime_affine["count"]),
        "overall_residual_runtime_affine_mean_signal_spearman": float(overall_residual_runtime_affine_mean["spearman"]),
        "overall_residual_runtime_affine_mean_signal_robust_slope": float(overall_residual_runtime_affine_mean["robust_slope"]),
        "overall_residual_runtime_affine_mean_signal_r2_like": float(overall_residual_runtime_affine_mean["r2_like"]),
        "overall_residual_runtime_affine_mean_signal_count": float(overall_residual_runtime_affine_mean["count"]),
        "valid_residual_runtime_affine_mean_signal_spearman": float(valid_residual_runtime_affine_mean["spearman"]),
        "valid_residual_runtime_affine_mean_signal_robust_slope": float(valid_residual_runtime_affine_mean["robust_slope"]),
        "valid_residual_runtime_affine_mean_signal_r2_like": float(valid_residual_runtime_affine_mean["r2_like"]),
        "valid_residual_runtime_affine_mean_signal_count": float(valid_residual_runtime_affine_mean["count"]),
        "valid_prefix_total_signal_spearman": float(valid_prefix_total["spearman"]),
        "valid_prefix_total_signal_robust_slope": float(valid_prefix_total["robust_slope"]),
        "valid_prefix_total_signal_r2_like": float(valid_prefix_total["r2_like"]),
        "valid_prefix_total_signal_count": float(valid_prefix_total["count"]),
        "valid_prefix_analytic_signal_spearman": float(valid_prefix_analytic["spearman"]),
        "valid_prefix_analytic_signal_robust_slope": float(valid_prefix_analytic["robust_slope"]),
        "valid_prefix_analytic_signal_r2_like": float(valid_prefix_analytic["r2_like"]),
        "valid_prefix_analytic_signal_count": float(valid_prefix_analytic["count"]),
        "valid_prefix_analytic_runtime_signal_spearman": float(valid_prefix_analytic_runtime["spearman"]),
        "valid_prefix_analytic_runtime_signal_robust_slope": float(valid_prefix_analytic_runtime["robust_slope"]),
        "valid_prefix_analytic_runtime_signal_r2_like": float(valid_prefix_analytic_runtime["r2_like"]),
        "valid_prefix_analytic_runtime_signal_count": float(valid_prefix_analytic_runtime["count"]),
        "valid_prefix_residual_signal_spearman": float(valid_prefix_residual["spearman"]),
        "valid_prefix_residual_signal_robust_slope": float(valid_prefix_residual["robust_slope"]),
        "valid_prefix_residual_signal_r2_like": float(valid_prefix_residual["r2_like"]),
        "valid_prefix_residual_signal_count": float(valid_prefix_residual["count"]),
        "valid_prefix_residual_runtime_signal_spearman": float(valid_prefix_residual_runtime["spearman"]),
        "valid_prefix_residual_runtime_signal_robust_slope": float(valid_prefix_residual_runtime["robust_slope"]),
        "valid_prefix_residual_runtime_signal_r2_like": float(valid_prefix_residual_runtime["r2_like"]),
        "valid_prefix_residual_runtime_signal_count": float(valid_prefix_residual_runtime["count"]),
        "mean_analytic_gap_clip": float(np.nanmean(np.asarray([row["analytic_gap_clip"] for row in frame], dtype=np.float32))),
        "mean_analytic_saturation_rate": float(np.nanmean(np.asarray([row["analytic_saturation_rate"] for row in frame], dtype=np.float32))),
        "valid_mean_analytic_saturation_rate": float(np.nanmean(np.asarray([row["analytic_saturation_rate"] for row in valid_rows], dtype=np.float32))) if valid_rows else float("nan"),
        "mean_beta1_runtime_fit": float(np.nanmean(np.asarray([row["beta1_runtime_fit"] for row in frame], dtype=np.float32))),
        "valid_mean_beta1_runtime_fit": float(np.nanmean(np.asarray([row["beta1_runtime_fit"] for row in valid_rows], dtype=np.float32))) if valid_rows else float("nan"),
        "overall_signal_spearman": float(overall_total["spearman"]),
        "overall_signal_robust_slope": float(overall_total["robust_slope"]),
        "overall_signal_r2_like": float(overall_total["r2_like"]),
        "overall_signal_count": float(overall_total["count"]),
        "valid_signal_spearman": float(valid_total["spearman"]),
        "valid_signal_robust_slope": float(valid_total["robust_slope"]),
        "valid_signal_r2_like": float(valid_total["r2_like"]),
        "valid_signal_count": float(valid_total["count"]),
        "cross_signal_spearman": float(cross_total["spearman"]),
        "cross_signal_robust_slope": float(cross_total["robust_slope"]),
        "cross_signal_r2_like": float(cross_total["r2_like"]),
        "cross_signal_count": float(cross_total["count"]),
        "valid_cross_signal_spearman": float(valid_cross_total["spearman"]),
        "valid_cross_signal_robust_slope": float(valid_cross_total["robust_slope"]),
        "valid_cross_signal_r2_like": float(valid_cross_total["r2_like"]),
        "valid_cross_signal_count": float(valid_cross_total["count"]),
        "same_signal_spearman": float(same_total["spearman"]),
        "same_signal_robust_slope": float(same_total["robust_slope"]),
        "same_signal_r2_like": float(same_total["r2_like"]),
        "same_signal_count": float(same_total["count"]),
        "overall_residual_spearman": float(overall_residual["spearman"]),
        "overall_residual_robust_slope": float(overall_residual["robust_slope"]),
        "overall_residual_r2_like": float(overall_residual["r2_like"]),
        "overall_residual_count": float(overall_residual["count"]),
        "valid_residual_spearman": float(valid_residual["spearman"]),
        "valid_residual_robust_slope": float(valid_residual["robust_slope"]),
        "valid_residual_r2_like": float(valid_residual["r2_like"]),
        "valid_residual_count": float(valid_residual["count"]),
        "valid_prefix_signal_spearman": float(valid_prefix_total["spearman"]),
        "valid_prefix_signal_robust_slope": float(valid_prefix_total["robust_slope"]),
        "valid_prefix_signal_r2_like": float(valid_prefix_total["r2_like"]),
        "valid_prefix_signal_count": float(valid_prefix_total["count"]),
        "valid_zero_total_median_items": int(sum(1 for row in valid_rows if abs(float(row.get("zbar_sp_star", float("nan")))) <= 1.0e-9)),
    }


def main() -> None:
    args = _parse_args()
    set_hparams(
        config=args.config,
        hparams_str=_compose_hparams_override(args),
        global_hparams=True,
        print_hparams=False,
        reset=True,
    )

    splits = _parse_splits(args.splits)
    candidate_tokens = _parse_int_list(args.candidate_tokens)
    drop_edge_values = _parse_int_list(args.drop_edge_runs)
    reference_modes = _parse_reference_modes(args.reference_modes)

    rows: list[dict[str, Any]] = []
    for split in splits:
        ds = ConanDataset(prefix=split, shuffle=False)
        pair_entries = getattr(ds, "_pair_entries", None)
        if not pair_entries:
            continue
        for fetch_index in range(len(pair_entries)):
            for candidate_token in candidate_tokens:
                for drop_edge_runs in drop_edge_values:
                    for reference_mode in reference_modes:
                        rows.append(
                            _compute_pair_row(
                                ds=ds,
                                split=split,
                                fetch_index=fetch_index,
                                candidate_token=int(candidate_token),
                                drop_edge_runs=int(drop_edge_runs),
                                reference_mode=str(reference_mode),
                            )
                        )

    grouped: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (
                int(row["candidate_token"]),
                int(row["drop_edge_runs_for_g"]),
                str(row.get("reference_mode", "manifest")),
            ),
            [],
        ).append(row)
    summary = [_summarize_group(grouped[key]) for key in sorted(grouped.keys())]

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    _write_csv(output_csv, rows)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            {
                "config": {
                    "config": args.config,
                    "splits": splits,
                    "candidate_tokens": candidate_tokens,
                    "drop_edge_runs": drop_edge_values,
                    "reference_modes": reference_modes,
                    "counterfactual_emit_silence_runs": False,
                },
                "summary": summary,
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"[counterfactual-static-gate0] rows={len(rows)}")
    print(f"[counterfactual-static-gate0] wrote_csv={output_csv}")
    print(f"[counterfactual-static-gate0] wrote_json={output_json}")
    for row in summary:
        print(
            "[counterfactual-static-gate0] "
            f"token={int(row['candidate_token'])} drop_edge={int(row['drop_edge_runs_for_g'])} "
            f"reference_mode={row.get('reference_mode', 'manifest')} "
            f"valid={int(row['g_domain_valid_items'])}/{int(row['item_count'])} "
            f"overall_signal_slope={float(row['overall_signal_robust_slope']):.4f} "
            f"valid_signal_slope={float(row['valid_signal_robust_slope']):.4f} "
            f"valid_prefix_signal_slope={float(row['valid_prefix_signal_robust_slope']):.4f} "
            f"clean_total_claim_slope={float(row['valid_clean_total_claim_signal_robust_slope']):.4f} "
            f"hostile_slice_slope={float(row['valid_protocol_misaligned_signal_robust_slope']):.4f}"
        )


if __name__ == "__main__":
    main()
