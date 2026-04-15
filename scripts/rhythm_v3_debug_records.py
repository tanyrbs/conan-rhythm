#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.plot.rhythm_v3_viz import load_debug_records, record_summary
from utils.plot.rhythm_v3_viz import (
    build_monotonicity_table,
    build_prefix_silence_review_table,
    build_ref_crop_table,
    save_review_figure_bundle,
    save_validation_gate_bundle,
    summarize_falsification_ladder,
)


_GATE1_MONOTONICITY_RATE_MIN = 0.95
_REQUIRED_NEGATIVE_CONTROLS = ("source_only", "random_ref", "shuffled_ref")
_GATE0_SIGNAL_SLOPE_MIN = 0.0
_GATE1_NEGATIVE_CONTROL_GAP_MIN = 0.0
_GATE1_SAME_TEXT_GAP_MAX = 0.10
_GATE1_TRANSFER_SLOPE_MIN = 0.10
_GATE1_EFFECT_SIZE_MIN = 0.02
_GATE1_MAX_CLIP_HIT_RATE = 0.50
_GATE1_MAX_BOUNDARY_HIT_RATE = 0.40
_GATE1_PROJECTED_REAL_RANGE_MIN = 0.05
_GATE1_ANALYTIC_SATURATION_RATE_MAX = 0.75
_GATE1_PROJECTOR_BUCKET_COUNT_MIN = 3.0
_GATE1_TIE_RATE_MAX = 0.05
_GATE1_INVALID_G_RATE_MAX = 0.0
_GATE1_ANTI_MONO_RATE_MAX = 0.10
_GATE0_VALID_ITEMS_MIN = 50
_GATE1_TRIPLETS_MIN = 30
_GATE0_ALIGNMENT_LOCAL_MARGIN_P10_MIN = 0.02
_ALIGNMENT_MEAN_LOCAL_CONF_MIN = 0.55
_ALIGNMENT_MEAN_COARSE_CONF_MIN = 0.60
_GATE2_RUNTIME_LIMITS = {
    "silence_leakage": 0.05,
    "prefix_discrepancy": 0.05,
    "budget_hit_rate": 0.05,
    "cumulative_drift": 0.25,
}
_GATE2_MONOTONICITY_DROP_TOL = 0.01
_GATE2_TRANSFER_SLOPE_DROP_TOL = 0.05
_GATE2_SPEECH_GAIN_MIN = 0.0
_GATE3_RUNTIME_DEGRADATION_TOLERANCES = dict(_GATE2_RUNTIME_LIMITS)
_GATE3_MONOTONICITY_DROP_TOL = 0.01
_GATE3_TRANSFER_SLOPE_DROP_TOL = 0.05
_GATE3_SPEECH_GAIN_MIN = 0.0
_GATE3_RESIDUAL_CORR_MIN = 0.10
_GATE3_COARSE_CORR_DROP_TOL = 0.05
_GATE3_RESIDUAL_BIAS_SHARE_MAX = 0.25
_GATE3_LOCAL_SILENCE_DELTA_SHARE_MAX = 0.02
_REAL_REF_BINS = {"slow", "mid", "fast"}
_REAL_REF_CONDITIONS = {"", "nan", "real", "real_reference"}


def _normalize_ref_condition(value) -> str:
    return str(value or "").strip().lower()


def _normalize_ref_bin(value, *, fallback_condition=None) -> str:
    ref_bin = str(value or "").strip().lower()
    if ref_bin in _REAL_REF_BINS:
        return ref_bin
    fallback = _normalize_ref_condition(fallback_condition)
    if fallback in _REAL_REF_BINS:
        return fallback
    return ref_bin


def _is_real_reference_condition(value, *, ref_bin=None) -> bool:
    ref_condition = _normalize_ref_condition(value)
    if ref_condition in _REQUIRED_NEGATIVE_CONTROLS:
        return False
    if ref_condition in _REAL_REF_CONDITIONS or ref_condition in _REAL_REF_BINS:
        return True
    return _normalize_ref_bin(ref_bin, fallback_condition=ref_condition) in _REAL_REF_BINS


def _resolve_ref_bin_column(frame):
    if "ref_bin" in getattr(frame, "columns", []):
        return frame["ref_bin"].map(_normalize_ref_bin)
    if "ref_condition" in getattr(frame, "columns", []):
        return frame["ref_condition"].map(lambda value: _normalize_ref_bin("", fallback_condition=value))
    return None


def _mean_for_eval_mode(frame, *, column: str, eval_mode: str) -> float:
    if column not in frame.columns or "eval_mode" not in frame.columns:
        return float("nan")
    mode_mask = frame["eval_mode"].astype(str).str.strip().str.lower() == str(eval_mode).strip().lower()
    values = pd.to_numeric(frame.loc[mode_mask, column], errors="coerce").to_numpy(dtype=np.float32)
    return float(np.nanmean(values)) if values.size > 0 and np.isfinite(values).any() else float("nan")


def _max_for_eval_mode(frame, *, column: str, eval_mode: str) -> float:
    if column not in frame.columns or "eval_mode" not in frame.columns:
        return float("nan")
    mode_mask = frame["eval_mode"].astype(str).str.strip().str.lower() == str(eval_mode).strip().lower()
    values = pd.to_numeric(frame.loc[mode_mask, column], errors="coerce").to_numpy(dtype=np.float32)
    return float(np.nanmax(values)) if values.size > 0 and np.isfinite(values).any() else float("nan")


def _corr_for_eval_mode(frame, *, x_col: str, y_col: str, eval_mode: str) -> float:
    if any(column not in frame.columns for column in (x_col, y_col, "eval_mode")):
        return float("nan")
    mode_mask = frame["eval_mode"].astype(str).str.strip().str.lower() == str(eval_mode).strip().lower()
    subset = frame.loc[mode_mask, [x_col, y_col]].copy()
    if subset.empty:
        return float("nan")
    subset[x_col] = pd.to_numeric(subset[x_col], errors="coerce")
    subset[y_col] = pd.to_numeric(subset[y_col], errors="coerce")
    subset = subset.dropna()
    if int(subset.shape[0]) <= 1:
        return float("nan")
    return float(subset[x_col].corr(subset[y_col], method="spearman"))


def _fast_slow_effect_for_eval_mode(frame, *, eval_mode: str) -> float:
    if "fast_minus_slow" in getattr(frame, "columns", []):
        value = _mean_for_eval_mode(frame, column="fast_minus_slow", eval_mode=eval_mode)
        if np.isfinite(value):
            return value
    if not {"eval_mode", "ref_condition", "tempo_out"}.issubset(getattr(frame, "columns", [])):
        return float("nan")
    mode_mask = frame["eval_mode"].astype(str).str.strip().str.lower() == str(eval_mode).strip().lower()
    ref_bins = _resolve_ref_bin_column(frame)
    if ref_bins is None:
        return float("nan")
    ref_condition = frame["ref_condition"].map(_normalize_ref_condition)
    real_mask = np.asarray(
        [
            _is_real_reference_condition(cond, ref_bin=bin_value)
            for cond, bin_value in zip(ref_condition.tolist(), ref_bins.tolist())
        ],
        dtype=bool,
    )
    slow_vals = pd.to_numeric(
        frame.loc[mode_mask & real_mask & (ref_bins == "slow"), "tempo_out"],
        errors="coerce",
    ).to_numpy(dtype=np.float32)
    fast_vals = pd.to_numeric(
        frame.loc[mode_mask & real_mask & (ref_bins == "fast"), "tempo_out"],
        errors="coerce",
    ).to_numpy(dtype=np.float32)
    if slow_vals.size <= 0 or fast_vals.size <= 0:
        return float("nan")
    slow_mean = float(np.nanmean(slow_vals)) if np.isfinite(slow_vals).any() else float("nan")
    fast_mean = float(np.nanmean(fast_vals)) if np.isfinite(fast_vals).any() else float("nan")
    if not np.isfinite(slow_mean) or not np.isfinite(fast_mean):
        return float("nan")
    return float(fast_mean - slow_mean)


def _real_tempo_range_for_eval_mode(frame, *, eval_mode: str) -> float:
    if not {"eval_mode", "ref_condition", "tempo_out"}.issubset(getattr(frame, "columns", [])):
        return float("nan")
    mode_mask = frame["eval_mode"].astype(str).str.strip().str.lower() == str(eval_mode).strip().lower()
    ref_bins = _resolve_ref_bin_column(frame)
    ref_condition = frame["ref_condition"].map(_normalize_ref_condition)
    real_mask = np.asarray(
        [
            _is_real_reference_condition(cond, ref_bin=(None if ref_bins is None else ref_bins.iloc[idx]))
            for idx, cond in enumerate(ref_condition.tolist())
        ],
        dtype=bool,
    )
    values = pd.to_numeric(frame.loc[mode_mask & real_mask, "tempo_out"], errors="coerce").to_numpy(dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size <= 0:
        return float("nan")
    return float(np.max(values) - np.min(values))


def _resolve_monotonicity_column(frame) -> str:
    if hasattr(frame, "columns") and "tempo_monotonicity_rate" in frame.columns:
        return "tempo_monotonicity_rate"
    return "monotonicity_rate"


def _resolve_speech_metric_column(frame) -> str:
    if hasattr(frame, "columns") and "speech_weighted_mae" in frame.columns:
        return "speech_weighted_mae"
    return "speech_mae"


def _resolve_gate0_signal_column(frame) -> str:
    if hasattr(frame, "columns") and "signal_explainability_slope" in frame.columns:
        return "signal_explainability_slope"
    return "explainability_slope"


def _resolve_gate0_decomp_column(frame) -> str:
    if hasattr(frame, "columns") and "coarse_residual_slope" in frame.columns:
        return "coarse_residual_slope"
    return "explainability_slope"


def _invalid_g_rate_for_eval_mode(frame, *, eval_mode: str) -> float:
    if "invalid_g_rate" in getattr(frame, "columns", []):
        value = _mean_for_eval_mode(frame, column="invalid_g_rate", eval_mode=eval_mode)
        if np.isfinite(value):
            return value
    domain_column = "g_domain_valid" if "g_domain_valid" in getattr(frame, "columns", []) else "g_valid"
    if domain_column not in getattr(frame, "columns", []) or "eval_mode" not in getattr(frame, "columns", []):
        return float("nan")
    mode_mask = frame["eval_mode"].astype(str).str.strip().str.lower() == str(eval_mode).strip().lower()
    values = pd.to_numeric(frame.loc[mode_mask, domain_column], errors="coerce").to_numpy(dtype=np.float32)
    if values.size <= 0 or not np.isfinite(values).any():
        return float("nan")
    finite = values[np.isfinite(values)]
    if finite.size <= 0:
        return float("nan")
    return float(np.mean(finite < 0.5))


def _collect_missing_metadata_issues(frame) -> list[str]:
    if pd is None or frame is None or not hasattr(frame, "columns"):
        return []
    total = int(frame.shape[0])
    if total <= 0:
        return []
    required = (
        "pair_id",
        "same_text_reference",
        "same_text_target",
        "lexical_mismatch",
        "ref_len_sec",
        "speech_ratio",
        "alignment_kind",
        "target_duration_surface",
        "g_support_count",
        "g_support_ratio_vs_speech",
        "g_support_ratio_vs_valid",
        "g_valid",
        "g_trim_ratio",
        "prompt_global_weight_present",
        "prompt_unit_log_prior_present",
        "alignment_unmatched_speech_ratio",
        "alignment_mean_local_confidence_speech",
        "alignment_mean_coarse_confidence_speech",
        "projector_boundary_hit_rate",
        "projector_boundary_decay_rate",
    )
    issues = []
    for column in required:
        if column not in frame.columns:
            issues.append(f"{column}=missing")
            continue
        observed = int(frame[column].notna().sum())
        if observed <= 0:
            issues.append(f"{column}=0/{total}")
    return issues


def collect_gate_issues(frame) -> list[str]:
    if pd is None or frame is None or not hasattr(frame, "columns"):
        return []
    total = int(frame.shape[0])
    if total <= 0:
        return ["summary_rows=empty"]
    quality_issues: list[str] = []
    domain_column = "g_domain_valid" if "g_domain_valid" in frame.columns else "g_valid"
    if domain_column in frame.columns:
        g_valid = pd.to_numeric(frame[domain_column], errors="coerce")
        g_valid_rate = float(np.nanmean(g_valid.to_numpy(dtype=np.float32)))
        if np.isfinite(g_valid_rate) and g_valid_rate < 0.95:
            quality_issues.append(f"{domain_column}_mean={g_valid_rate:.3f}<0.950")
    if "gate0_row_dropped" in frame.columns:
        gate0_drop = pd.to_numeric(frame["gate0_row_dropped"], errors="coerce").to_numpy(dtype=np.float32)
        gate0_drop_rate = float(np.nanmean(gate0_drop)) if gate0_drop.size > 0 else float("nan")
        if np.isfinite(gate0_drop_rate) and gate0_drop_rate > 0.05:
            quality_issues.append(f"gate0_drop_rate={gate0_drop_rate:.3f}>0.050")
        if "gate0_drop_reason" in frame.columns:
            drop_reason = frame.loc[pd.to_numeric(frame["gate0_row_dropped"], errors="coerce") > 0.5, "gate0_drop_reason"]
            if not drop_reason.empty:
                reason_counts = (
                    drop_reason.astype(str)
                    .str.strip()
                    .replace({"": "missing"})
                    .value_counts()
                    .head(5)
                )
                quality_issues.append(
                    "gate0_drop_top="
                    + "|".join(f"{key}:{int(value)}" for key, value in reason_counts.items())
                )
    if "alignment_unmatched_speech_ratio" in frame.columns:
        unmatched = pd.to_numeric(frame["alignment_unmatched_speech_ratio"], errors="coerce").to_numpy(dtype=np.float32)
        unmatched_p95 = float(np.nanpercentile(unmatched, 95)) if np.isfinite(unmatched).any() else float("nan")
        if np.isfinite(unmatched_p95) and unmatched_p95 > 0.15:
            quality_issues.append(f"alignment_unmatched_speech_ratio_p95={unmatched_p95:.3f}>0.150")
    for column, threshold in (
        ("alignment_mean_local_confidence_speech", _ALIGNMENT_MEAN_LOCAL_CONF_MIN),
        ("alignment_mean_coarse_confidence_speech", _ALIGNMENT_MEAN_COARSE_CONF_MIN),
    ):
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float32)
        mean_value = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
        if np.isfinite(mean_value) and mean_value < threshold:
            quality_issues.append(f"{column}_mean={mean_value:.3f}<{threshold:.3f}")
    if "alignment_kind" in frame.columns:
        alignment_kind = frame["alignment_kind"].astype(str).str.strip().str.lower()
        observed = alignment_kind[(alignment_kind != "") & (alignment_kind != "nan")]
        alignment_valid_rate = float(observed.shape[0]) / float(total)
        if np.isfinite(alignment_valid_rate) and alignment_valid_rate < 0.95:
            quality_issues.append(f"alignment_valid_rate={alignment_valid_rate:.3f}<0.950")
        if not observed.empty:
            continuous_mask = observed.str.startswith("continuous")
            if bool((~continuous_mask).any()):
                fraction = float(continuous_mask.mean())
                quality_issues.append(f"continuous_alignment_coverage={fraction:.3f}<1.000")
    for column in ("g_compute_status", "g_src_compute_status"):
        if column not in frame.columns:
            continue
        failures = frame[column].astype(str).str.strip().str.lower()
        failures = failures[(failures != "") & (failures != "ok") & (failures != "nan")]
        if not failures.empty:
            top = failures.value_counts().head(3)
            quality_issues.append(
                f"{column}_top=" + "|".join(f"{key}:{int(value)}" for key, value in top.items())
            )
    if {"eval_mode"}.issubset(frame.columns):
        observed_modes = {
            str(mode).strip()
            for mode in frame["eval_mode"].dropna().tolist()
            if str(mode).strip() and str(mode).strip().lower() != "nan"
        }
        analytic_mask = frame["eval_mode"].astype(str).str.strip().str.lower() == "analytic"
        analytic_valid_items = (
            int(frame.loc[analytic_mask, "sample_id"].dropna().nunique())
            if "sample_id" in frame.columns
            else int(analytic_mask.sum())
        )
        analytic_triplet_count = (
            int(frame.loc[analytic_mask, "triplet_id"].dropna().nunique())
            if "triplet_id" in frame.columns
            else 0
        )
        if analytic_valid_items < _GATE0_VALID_ITEMS_MIN:
            quality_issues.append(
                f"analytic_valid_items={analytic_valid_items}<{_GATE0_VALID_ITEMS_MIN}"
            )
        if analytic_triplet_count < _GATE1_TRIPLETS_MIN:
            quality_issues.append(
                f"analytic_triplet_count={analytic_triplet_count}<{_GATE1_TRIPLETS_MIN}"
            )
        missing_modes = [mode for mode in ("analytic", "coarse_only", "learned") if mode not in observed_modes]
        if missing_modes:
            quality_issues.append("missing_eval_modes=" + "|".join(missing_modes))
        analytic_mono = _mean_for_eval_mode(frame, column="tempo_monotonicity_rate", eval_mode="analytic")
        if "analytic" in observed_modes:
            if not np.isfinite(analytic_mono):
                quality_issues.append("analytic_tempo_monotonicity_rate=missing")
            elif analytic_mono < _GATE1_MONOTONICITY_RATE_MIN:
                quality_issues.append(
                    f"analytic_tempo_monotonicity_rate={analytic_mono:.3f}<{_GATE1_MONOTONICITY_RATE_MIN:.3f}"
                )
            signal_column = _resolve_gate0_signal_column(frame)
            analytic_slope = _mean_for_eval_mode(frame, column=signal_column, eval_mode="analytic")
            if not np.isfinite(analytic_slope):
                quality_issues.append("analytic_signal_explainability_slope=missing")
            elif analytic_slope <= _GATE0_SIGNAL_SLOPE_MIN:
                quality_issues.append(
                    f"analytic_signal_explainability_slope={analytic_slope:.3f}<={_GATE0_SIGNAL_SLOPE_MIN:.3f}"
                )
            analytic_neg_gap = _mean_for_eval_mode(frame, column="negative_control_gap", eval_mode="analytic")
            if not np.isfinite(analytic_neg_gap):
                quality_issues.append("analytic_negative_control_gap=missing")
            elif analytic_neg_gap <= _GATE1_NEGATIVE_CONTROL_GAP_MIN:
                quality_issues.append(
                    f"analytic_negative_control_gap={analytic_neg_gap:.3f}<={_GATE1_NEGATIVE_CONTROL_GAP_MIN:.3f}"
                )
            analytic_same_text_gap = _mean_for_eval_mode(frame, column="same_text_gap", eval_mode="analytic")
            analytic_same_text_gap_max = _max_for_eval_mode(frame, column="same_text_gap", eval_mode="analytic")
            if np.isfinite(analytic_same_text_gap_max) and analytic_same_text_gap_max > _GATE1_SAME_TEXT_GAP_MAX:
                quality_issues.append(
                    f"analytic_same_text_gap_max={analytic_same_text_gap_max:.3f}>{_GATE1_SAME_TEXT_GAP_MAX:.3f}"
                )
            analytic_margin = _mean_for_eval_mode(
                frame, column="alignment_local_margin_p10", eval_mode="analytic"
            )
            if not np.isfinite(analytic_margin):
                quality_issues.append("analytic_alignment_local_margin_p10=missing")
            elif analytic_margin < _GATE0_ALIGNMENT_LOCAL_MARGIN_P10_MIN:
                quality_issues.append(
                    f"analytic_alignment_local_margin_p10={analytic_margin:.3f}<{_GATE0_ALIGNMENT_LOCAL_MARGIN_P10_MIN:.3f}"
                )
            analytic_transfer = _mean_for_eval_mode(frame, column="tempo_transfer_slope", eval_mode="analytic")
            if not np.isfinite(analytic_transfer):
                quality_issues.append("analytic_tempo_transfer_slope=missing")
            elif analytic_transfer < _GATE1_TRANSFER_SLOPE_MIN:
                quality_issues.append(
                    f"analytic_tempo_transfer_slope={analytic_transfer:.3f}<{_GATE1_TRANSFER_SLOPE_MIN:.3f}"
                )
            analytic_effect = _fast_slow_effect_for_eval_mode(frame, eval_mode="analytic")
            if np.isfinite(analytic_effect) and analytic_effect < _GATE1_EFFECT_SIZE_MIN:
                quality_issues.append(
                    f"analytic_fast_slow_effect={analytic_effect:.3f}<{_GATE1_EFFECT_SIZE_MIN:.3f}"
                )
            analytic_clip = _mean_for_eval_mode(frame, column="analytic_saturation_rate", eval_mode="analytic")
            if np.isfinite(analytic_clip) and analytic_clip > _GATE1_MAX_CLIP_HIT_RATE:
                quality_issues.append(
                    f"analytic_clip_hit_rate={analytic_clip:.3f}>{_GATE1_MAX_CLIP_HIT_RATE:.3f}"
                )
            if np.isfinite(analytic_clip) and analytic_clip > _GATE1_ANALYTIC_SATURATION_RATE_MAX:
                quality_issues.append(
                    f"analytic_mean_saturation={analytic_clip:.3f}>{_GATE1_ANALYTIC_SATURATION_RATE_MAX:.3f}"
                )
            analytic_boundary = _mean_for_eval_mode(frame, column="projector_boundary_hit_rate", eval_mode="analytic")
            if np.isfinite(analytic_boundary) and analytic_boundary > _GATE1_MAX_BOUNDARY_HIT_RATE:
                quality_issues.append(
                    f"analytic_boundary_hit_rate={analytic_boundary:.3f}>{_GATE1_MAX_BOUNDARY_HIT_RATE:.3f}"
                )
            analytic_projected_range = _real_tempo_range_for_eval_mode(frame, eval_mode="analytic")
            if not np.isfinite(analytic_projected_range):
                quality_issues.append("analytic_projected_real_range=missing")
            elif analytic_projected_range < _GATE1_PROJECTED_REAL_RANGE_MIN:
                quality_issues.append(
                    f"analytic_projected_real_range={analytic_projected_range:.3f}<{_GATE1_PROJECTED_REAL_RANGE_MIN:.3f}"
                )
            analytic_bucket_count = _mean_for_eval_mode(
                frame,
                column="projector_bucket_count",
                eval_mode="analytic",
            )
            if not np.isfinite(analytic_bucket_count):
                quality_issues.append("analytic_mean_bucket_count=missing")
            elif analytic_bucket_count < _GATE1_PROJECTOR_BUCKET_COUNT_MIN:
                quality_issues.append(
                    f"analytic_mean_bucket_count={analytic_bucket_count:.3f}<{_GATE1_PROJECTOR_BUCKET_COUNT_MIN:.3f}"
                )
            analytic_tie_rate = _mean_for_eval_mode(frame, column="tempo_tie_rate", eval_mode="analytic")
            if not np.isfinite(analytic_tie_rate):
                quality_issues.append("analytic_tempo_tie_rate=missing")
            elif analytic_tie_rate > _GATE1_TIE_RATE_MAX:
                quality_issues.append(
                    f"analytic_tempo_tie_rate={analytic_tie_rate:.3f}>{_GATE1_TIE_RATE_MAX:.3f}"
                )
            analytic_anti_mono_rate = _mean_for_eval_mode(frame, column="anti_monotonicity_rate", eval_mode="analytic")
            if np.isfinite(analytic_anti_mono_rate) and analytic_anti_mono_rate > _GATE1_ANTI_MONO_RATE_MAX:
                quality_issues.append(
                    f"analytic_anti_monotonicity_rate={analytic_anti_mono_rate:.3f}>{_GATE1_ANTI_MONO_RATE_MAX:.3f}"
                )
            analytic_invalid_g_rate = _invalid_g_rate_for_eval_mode(frame, eval_mode="analytic")
            if not np.isfinite(analytic_invalid_g_rate):
                quality_issues.append("analytic_invalid_g_rate=missing")
            elif analytic_invalid_g_rate > _GATE1_INVALID_G_RATE_MAX:
                quality_issues.append(
                    f"analytic_invalid_g_rate={analytic_invalid_g_rate:.3f}>{_GATE1_INVALID_G_RATE_MAX:.3f}"
                )
        mono_column = _resolve_monotonicity_column(frame)
        speech_metric_column = _resolve_speech_metric_column(frame)
        if "coarse_only" in observed_modes:
            for column, limit in _GATE2_RUNTIME_LIMITS.items():
                if column not in frame.columns:
                    continue
                coarse_mean = _mean_for_eval_mode(frame, column=column, eval_mode="coarse_only")
                if np.isfinite(coarse_mean) and coarse_mean > float(limit):
                    quality_issues.append(
                        f"coarse_only_{column}_limit={coarse_mean:.3f}>{float(limit):.3f}"
                    )
            coarse_mono = _mean_for_eval_mode(frame, column=mono_column, eval_mode="coarse_only")
            if "analytic" in observed_modes and np.isfinite(analytic_mono) and np.isfinite(coarse_mono):
                if coarse_mono < (analytic_mono - _GATE2_MONOTONICITY_DROP_TOL):
                    quality_issues.append(
                        f"coarse_only_monotonicity_rate_regression={coarse_mono:.3f}<{analytic_mono:.3f}-{_GATE2_MONOTONICITY_DROP_TOL:.3f}"
                    )
            analytic_transfer = _mean_for_eval_mode(frame, column="tempo_transfer_slope", eval_mode="analytic")
            coarse_transfer = _mean_for_eval_mode(frame, column="tempo_transfer_slope", eval_mode="coarse_only")
            if np.isfinite(analytic_transfer) and np.isfinite(coarse_transfer):
                if coarse_transfer < (analytic_transfer - _GATE2_TRANSFER_SLOPE_DROP_TOL):
                    quality_issues.append(
                        f"coarse_only_tempo_transfer_slope_regression={coarse_transfer:.3f}<{analytic_transfer:.3f}-{_GATE2_TRANSFER_SLOPE_DROP_TOL:.3f}"
                    )
            if speech_metric_column in frame.columns:
                analytic_speech = _mean_for_eval_mode(frame, column=speech_metric_column, eval_mode="analytic")
                coarse_speech = _mean_for_eval_mode(frame, column=speech_metric_column, eval_mode="coarse_only")
                if np.isfinite(analytic_speech) and np.isfinite(coarse_speech):
                    gain = analytic_speech - coarse_speech
                    if gain <= _GATE2_SPEECH_GAIN_MIN:
                        quality_issues.append(
                            f"coarse_only_{speech_metric_column}_gain={gain:.3f}<={_GATE2_SPEECH_GAIN_MIN:.3f}"
                        )
        if {"coarse_only", "learned"}.issubset(observed_modes):
            for column, tolerance in _GATE3_RUNTIME_DEGRADATION_TOLERANCES.items():
                if column not in frame.columns:
                    quality_issues.append(f"{column}=missing")
                    continue
                coarse_mean = _mean_for_eval_mode(frame, column=column, eval_mode="coarse_only")
                learned_mean = _mean_for_eval_mode(frame, column=column, eval_mode="learned")
                if not np.isfinite(coarse_mean) or not np.isfinite(learned_mean):
                    quality_issues.append(f"{column}_mode_mean=missing")
                    continue
                if learned_mean > (coarse_mean + float(tolerance)):
                    quality_issues.append(
                        f"learned_{column}_regression={learned_mean:.3f}>{coarse_mean:.3f}+{float(tolerance):.3f}"
                    )
            coarse_mono = _mean_for_eval_mode(frame, column=mono_column, eval_mode="coarse_only")
            learned_mono = _mean_for_eval_mode(frame, column=mono_column, eval_mode="learned")
            if not np.isfinite(coarse_mono) or not np.isfinite(learned_mono):
                quality_issues.append("monotonicity_rate_mode_mean=missing")
            elif learned_mono < (coarse_mono - _GATE3_MONOTONICITY_DROP_TOL):
                quality_issues.append(
                    f"learned_monotonicity_rate_regression={learned_mono:.3f}<{coarse_mono:.3f}-{_GATE3_MONOTONICITY_DROP_TOL:.3f}"
                )
            coarse_transfer = _mean_for_eval_mode(frame, column="tempo_transfer_slope", eval_mode="coarse_only")
            learned_transfer = _mean_for_eval_mode(frame, column="tempo_transfer_slope", eval_mode="learned")
            if not np.isfinite(coarse_transfer) or not np.isfinite(learned_transfer):
                quality_issues.append("tempo_transfer_slope_mode_mean=missing")
            elif learned_transfer < (coarse_transfer - _GATE3_TRANSFER_SLOPE_DROP_TOL):
                quality_issues.append(
                    f"learned_tempo_transfer_slope_regression={learned_transfer:.3f}<{coarse_transfer:.3f}-{_GATE3_TRANSFER_SLOPE_DROP_TOL:.3f}"
                )
            if speech_metric_column in frame.columns:
                coarse_speech = _mean_for_eval_mode(frame, column=speech_metric_column, eval_mode="coarse_only")
                learned_speech = _mean_for_eval_mode(frame, column=speech_metric_column, eval_mode="learned")
                if np.isfinite(coarse_speech) and np.isfinite(learned_speech):
                    gain = coarse_speech - learned_speech
                    if gain <= _GATE3_SPEECH_GAIN_MIN:
                        quality_issues.append(
                            f"learned_{speech_metric_column}_gain={gain:.3f}<={_GATE3_SPEECH_GAIN_MIN:.3f}"
                        )
            learned_residual_corr = _mean_for_eval_mode(frame, column="residual_target_corr", eval_mode="learned")
            if np.isfinite(learned_residual_corr) and learned_residual_corr < _GATE3_RESIDUAL_CORR_MIN:
                quality_issues.append(
                    f"learned_residual_target_corr={learned_residual_corr:.3f}<{_GATE3_RESIDUAL_CORR_MIN:.3f}"
                )
            learned_residual_bias_share = _mean_for_eval_mode(frame, column="residual_bias_share", eval_mode="learned")
            if np.isfinite(learned_residual_bias_share) and learned_residual_bias_share > _GATE3_RESIDUAL_BIAS_SHARE_MAX:
                quality_issues.append(
                    f"learned_residual_bias_share={learned_residual_bias_share:.3f}>{_GATE3_RESIDUAL_BIAS_SHARE_MAX:.3f}"
                )
            learned_local_silence_delta_share = _mean_for_eval_mode(
                frame,
                column="local_silence_delta_share",
                eval_mode="learned",
            )
            if np.isfinite(learned_local_silence_delta_share) and learned_local_silence_delta_share > _GATE3_LOCAL_SILENCE_DELTA_SHARE_MAX:
                quality_issues.append(
                    f"learned_local_silence_delta_share={learned_local_silence_delta_share:.3f}>{_GATE3_LOCAL_SILENCE_DELTA_SHARE_MAX:.3f}"
                )
            coarse_corr = _corr_for_eval_mode(
                frame,
                x_col="oracle_bias",
                y_col="predicted_bias",
                eval_mode="coarse_only",
            )
            learned_corr = _corr_for_eval_mode(
                frame,
                x_col="oracle_bias",
                y_col="predicted_bias",
                eval_mode="learned",
            )
            if np.isfinite(coarse_corr) and np.isfinite(learned_corr):
                if learned_corr < (coarse_corr - _GATE3_COARSE_CORR_DROP_TOL):
                    quality_issues.append(
                        f"learned_coarse_target_corr_drop={learned_corr:.3f}<{coarse_corr:.3f}-{_GATE3_COARSE_CORR_DROP_TOL:.3f}"
                    )
    if {"src_id", "eval_mode", "ref_bin"}.issubset(frame.columns):
        triplet_frame = frame.copy()
        if "ref_condition" in triplet_frame.columns:
            ref_condition = triplet_frame["ref_condition"].astype(str).str.strip().str.lower()
            ref_bins = triplet_frame["ref_bin"].map(_normalize_ref_bin)
            real_mask = pd.Series(
                [
                    _is_real_reference_condition(cond, ref_bin=bin_value)
                    for cond, bin_value in zip(ref_condition.tolist(), ref_bins.tolist())
                ],
                index=triplet_frame.index,
            )
            triplet_frame = triplet_frame[real_mask]
        triplet_frame = triplet_frame[triplet_frame["ref_bin"].map(_normalize_ref_bin).isin(_REAL_REF_BINS)]
        if not triplet_frame.empty:
            triplet_counts = (
                triplet_frame.assign(ref_bin=triplet_frame["ref_bin"].astype(str).str.strip().str.lower())
                .groupby(["src_id", "eval_mode"])["ref_bin"]
                .nunique()
            )
            incomplete = int((triplet_counts < 3).sum())
            if incomplete > 0:
                quality_issues.append(f"incomplete_real_triplets={incomplete}")
    if "ref_condition" not in frame.columns:
        quality_issues.append("ref_condition=missing")
    else:
        ref_conditions = {
            str(value).strip().lower()
            for value in frame["ref_condition"].dropna().tolist()
            if str(value).strip() and str(value).strip().lower() != "nan"
        }
        missing_controls = [
            control
            for control in _REQUIRED_NEGATIVE_CONTROLS
            if control not in ref_conditions
        ]
        if missing_controls:
            quality_issues.append("missing_negative_controls=" + "|".join(missing_controls))
    return quality_issues


def collect_review_issues(frame) -> list[str]:
    return _collect_missing_metadata_issues(frame) + collect_gate_issues(frame)


def build_gate_status(frame) -> dict[str, object]:
    if pd is None or frame is None or not hasattr(frame, "columns") or int(frame.shape[0]) <= 0:
        return {
            "gate0a_pass": False,
            "gate0b_pass": False,
            "gate0c_pass": False,
            "gate0_pass": False,
            "gate1_pass": False,
            "gate2_pass": False,
            "gate3_pass": False,
            "missing_controls": list(_REQUIRED_NEGATIVE_CONTROLS),
            "missing_eval_modes": ["analytic", "coarse_only", "learned"],
            "incomplete_triplets": 0,
            "continuous_alignment_coverage": float("nan"),
            "g_domain_valid_mean": float("nan"),
            "gate0_drop_rate": float("nan"),
            "analytic_signal_explainability_slope": float("nan"),
            "analytic_explainability_slope": float("nan"),
            "analytic_coarse_residual_slope": float("nan"),
            "analytic_negative_control_gap": float("nan"),
            "analytic_fast_slow_effect": float("nan"),
            "analytic_clip_hit_rate": float("nan"),
            "analytic_boundary_hit_rate": float("nan"),
            "analytic_projected_real_range": float("nan"),
            "analytic_mean_saturation": float("nan"),
            "analytic_mean_bucket_count": float("nan"),
            "analytic_same_text_gap": float("nan"),
            "analytic_same_text_gap_max": float("nan"),
            "analytic_tempo_monotonicity_rate": float("nan"),
            "analytic_tempo_transfer_slope": float("nan"),
            "analytic_tempo_tie_rate": float("nan"),
            "analytic_invalid_g_rate": float("nan"),
            "analytic_anti_monotonicity_rate": float("nan"),
            "analytic_valid_items": 0,
            "analytic_triplet_count": 0,
            "alignment_mean_local_confidence_speech": float("nan"),
            "alignment_mean_coarse_confidence_speech": float("nan"),
            "unmatched_speech_ratio_p95": float("nan"),
            "coarse_only_runtime_metrics": {},
            "coarse_only_runtime_limit_violations": [],
            "coarse_only_control_regressions": [],
            "coarse_only_speech_metric": float("nan"),
            "coarse_only_coarse_target_corr": float("nan"),
            "learned_runtime_metrics": {},
            "learned_runtime_regressions": [],
            "learned_control_regressions": [],
            "learned_speech_metric": float("nan"),
            "learned_coarse_target_corr": float("nan"),
            "learned_residual_target_corr": float("nan"),
            "learned_residual_bias_share": float("nan"),
            "learned_local_silence_delta_share": float("nan"),
            "warnings": ["summary_rows=empty"],
        }
    issues = collect_gate_issues(frame)
    missing_controls: list[str] = []
    missing_eval_modes: list[str] = []
    incomplete_triplets = 0
    continuous_alignment_coverage = float("nan")
    g_domain_valid_mean = float("nan")
    gate0_drop_rate = float("nan")
    analytic_signal_explainability_slope = float("nan")
    analytic_explainability_slope = float("nan")
    analytic_coarse_residual_slope = float("nan")
    analytic_negative_control_gap = float("nan")
    analytic_same_text_gap = float("nan")
    analytic_same_text_gap_max = float("nan")
    analytic_alignment_local_margin_p10 = float("nan")
    analytic_tempo_monotonicity_rate = float("nan")
    analytic_tempo_transfer_slope = float("nan")
    analytic_tempo_tie_rate = float("nan")
    analytic_invalid_g_rate = float("nan")
    analytic_anti_monotonicity_rate = float("nan")
    analytic_projected_real_range = float("nan")
    analytic_mean_saturation = float("nan")
    analytic_mean_bucket_count = float("nan")
    analytic_valid_items = 0
    analytic_triplet_count = 0
    alignment_mean_local_confidence_speech = float("nan")
    alignment_mean_coarse_confidence_speech = float("nan")
    unmatched_speech_ratio_p95 = float("nan")
    coarse_only_runtime_metrics: dict[str, float] = {}
    coarse_only_runtime_limit_violations: list[str] = []
    coarse_only_control_regressions: list[str] = []
    learned_runtime_metrics: dict[str, float] = {}
    learned_runtime_regressions: list[str] = []
    learned_control_regressions: list[str] = []
    speech_metric_column = _resolve_speech_metric_column(frame)
    mono_column = _resolve_monotonicity_column(frame)
    domain_column = "g_domain_valid" if "g_domain_valid" in frame.columns else "g_valid"
    if domain_column in frame.columns:
        g_domain_valid_mean = float(
            np.nanmean(pd.to_numeric(frame[domain_column], errors="coerce").to_numpy(dtype=np.float32))
        )
    if "gate0_row_dropped" in frame.columns:
        gate0_drop_rate = float(
            np.nanmean(pd.to_numeric(frame["gate0_row_dropped"], errors="coerce").to_numpy(dtype=np.float32))
        )
    if "alignment_unmatched_speech_ratio" in frame.columns:
        unmatched = pd.to_numeric(frame["alignment_unmatched_speech_ratio"], errors="coerce").to_numpy(dtype=np.float32)
        unmatched_speech_ratio_p95 = float(np.nanpercentile(unmatched, 95)) if np.isfinite(unmatched).any() else float("nan")
    if "alignment_mean_local_confidence_speech" in frame.columns:
        values = pd.to_numeric(frame["alignment_mean_local_confidence_speech"], errors="coerce").to_numpy(dtype=np.float32)
        alignment_mean_local_confidence_speech = (
            float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
        )
    if "alignment_mean_coarse_confidence_speech" in frame.columns:
        values = pd.to_numeric(frame["alignment_mean_coarse_confidence_speech"], errors="coerce").to_numpy(dtype=np.float32)
        alignment_mean_coarse_confidence_speech = (
            float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
        )
    if "alignment_kind" in frame.columns:
        alignment_kind = frame["alignment_kind"].astype(str).str.strip().str.lower()
        observed = alignment_kind[(alignment_kind != "") & (alignment_kind != "nan")]
        if not observed.empty:
            continuous_alignment_coverage = float(observed.str.startswith("continuous").mean())
    for issue in issues:
        if issue.startswith("missing_eval_modes="):
            missing_eval_modes = [part for part in issue.split("=", 1)[1].split("|") if part]
        elif issue.startswith("missing_negative_controls="):
            missing_controls.extend(
                [part for part in issue.split("=", 1)[1].split("|") if part]
            )
        elif issue == "ref_condition=missing":
            missing_controls.extend(list(_REQUIRED_NEGATIVE_CONTROLS))
        elif issue.startswith("incomplete_real_triplets="):
            try:
                incomplete_triplets = int(issue.split("=", 1)[1])
            except Exception:
                incomplete_triplets = 0
    missing_controls = list(dict.fromkeys(missing_controls))
    observed_modes = {
        str(mode).strip()
        for mode in frame.get("eval_mode", []).dropna().tolist()
        if str(mode).strip() and str(mode).strip().lower() != "nan"
    } if "eval_mode" in frame.columns else set()
    if "eval_mode" in frame.columns:
        analytic_mask = frame["eval_mode"].astype(str).str.strip().str.lower() == "analytic"
        analytic_valid_items = (
            int(frame.loc[analytic_mask, "sample_id"].dropna().nunique())
            if "sample_id" in frame.columns
            else int(analytic_mask.sum())
        )
        analytic_triplet_count = (
            int(frame.loc[analytic_mask, "triplet_id"].dropna().nunique())
            if "triplet_id" in frame.columns
            else 0
        )
    if "analytic" in observed_modes:
        signal_column = _resolve_gate0_signal_column(frame)
        analytic_signal_explainability_slope = _mean_for_eval_mode(
            frame,
            column=signal_column,
            eval_mode="analytic",
        )
        analytic_explainability_slope = _mean_for_eval_mode(
            frame,
            column="explainability_slope",
            eval_mode="analytic",
        )
        analytic_coarse_residual_slope = _mean_for_eval_mode(
            frame,
            column=_resolve_gate0_decomp_column(frame),
            eval_mode="analytic",
        )
        analytic_negative_control_gap = _mean_for_eval_mode(
            frame,
            column="negative_control_gap",
            eval_mode="analytic",
        )
        analytic_same_text_gap = _mean_for_eval_mode(
            frame,
            column="same_text_gap",
            eval_mode="analytic",
        )
        analytic_same_text_gap_max = _max_for_eval_mode(
            frame,
            column="same_text_gap",
            eval_mode="analytic",
        )
        analytic_alignment_local_margin_p10 = _mean_for_eval_mode(
            frame,
            column="alignment_local_margin_p10",
            eval_mode="analytic",
        )
        analytic_tempo_monotonicity_rate = _mean_for_eval_mode(
            frame,
            column=mono_column,
            eval_mode="analytic",
        )
        analytic_tempo_transfer_slope = _mean_for_eval_mode(
            frame,
            column="tempo_transfer_slope",
            eval_mode="analytic",
        )
        analytic_tempo_tie_rate = _mean_for_eval_mode(
            frame,
            column="tempo_tie_rate",
            eval_mode="analytic",
        )
        analytic_anti_monotonicity_rate = _mean_for_eval_mode(
            frame,
            column="anti_monotonicity_rate",
            eval_mode="analytic",
        )
        analytic_invalid_g_rate = _invalid_g_rate_for_eval_mode(frame, eval_mode="analytic")
    analytic_fast_slow_effect = _fast_slow_effect_for_eval_mode(frame, eval_mode="analytic")
    if not np.isfinite(analytic_fast_slow_effect):
        analytic_fast_slow_effect = analytic_tempo_transfer_slope
    analytic_clip_hit_rate = _mean_for_eval_mode(
        frame,
        column="analytic_saturation_rate",
        eval_mode="analytic",
    )
    analytic_mean_saturation = analytic_clip_hit_rate
    analytic_boundary_hit_rate = _mean_for_eval_mode(
        frame,
        column="projector_boundary_hit_rate",
        eval_mode="analytic",
    )
    analytic_projected_real_range = _real_tempo_range_for_eval_mode(frame, eval_mode="analytic")
    analytic_mean_bucket_count = _mean_for_eval_mode(
        frame,
        column="projector_bucket_count",
        eval_mode="analytic",
    )
    analytic_transfer = analytic_tempo_transfer_slope
    analytic_speech_metric = _mean_for_eval_mode(frame, column=speech_metric_column, eval_mode="analytic")
    for column, limit in _GATE2_RUNTIME_LIMITS.items():
        coarse_only_runtime_metrics[column] = _mean_for_eval_mode(frame, column=column, eval_mode="coarse_only")
        coarse_mean = coarse_only_runtime_metrics[column]
        if np.isfinite(coarse_mean) and coarse_mean > float(limit):
            coarse_only_runtime_limit_violations.append(column)
    coarse_mono = _mean_for_eval_mode(frame, column=mono_column, eval_mode="coarse_only")
    coarse_transfer = _mean_for_eval_mode(frame, column="tempo_transfer_slope", eval_mode="coarse_only")
    coarse_only_speech_metric = _mean_for_eval_mode(frame, column=speech_metric_column, eval_mode="coarse_only")
    if np.isfinite(analytic_tempo_monotonicity_rate) and np.isfinite(coarse_mono):
        if coarse_mono < (analytic_tempo_monotonicity_rate - _GATE2_MONOTONICITY_DROP_TOL):
            coarse_only_control_regressions.append("monotonicity_rate")
    if np.isfinite(analytic_transfer) and np.isfinite(coarse_transfer):
        if coarse_transfer < (analytic_transfer - _GATE2_TRANSFER_SLOPE_DROP_TOL):
            coarse_only_control_regressions.append("tempo_transfer_slope")
    if np.isfinite(analytic_speech_metric) and np.isfinite(coarse_only_speech_metric):
        if (analytic_speech_metric - coarse_only_speech_metric) <= _GATE2_SPEECH_GAIN_MIN:
            coarse_only_control_regressions.append(speech_metric_column)
    coarse_only_coarse_target_corr = _corr_for_eval_mode(
        frame,
        x_col="oracle_bias",
        y_col="predicted_bias",
        eval_mode="coarse_only",
    )
    for column, tolerance in _GATE3_RUNTIME_DEGRADATION_TOLERANCES.items():
        learned_runtime_metrics[column] = _mean_for_eval_mode(frame, column=column, eval_mode="learned")
        coarse_mean = coarse_only_runtime_metrics[column]
        learned_mean = learned_runtime_metrics[column]
        if np.isfinite(coarse_mean) and np.isfinite(learned_mean) and learned_mean > (coarse_mean + float(tolerance)):
            learned_runtime_regressions.append(column)
    learned_mono = _mean_for_eval_mode(frame, column=mono_column, eval_mode="learned")
    if np.isfinite(coarse_mono) and np.isfinite(learned_mono) and learned_mono < (coarse_mono - _GATE3_MONOTONICITY_DROP_TOL):
        learned_control_regressions.append("monotonicity_rate")
    learned_transfer = _mean_for_eval_mode(frame, column="tempo_transfer_slope", eval_mode="learned")
    if np.isfinite(coarse_transfer) and np.isfinite(learned_transfer) and learned_transfer < (coarse_transfer - _GATE3_TRANSFER_SLOPE_DROP_TOL):
        learned_control_regressions.append("tempo_transfer_slope")
    learned_speech_metric = _mean_for_eval_mode(frame, column=speech_metric_column, eval_mode="learned")
    if np.isfinite(coarse_only_speech_metric) and np.isfinite(learned_speech_metric):
        if (coarse_only_speech_metric - learned_speech_metric) <= _GATE3_SPEECH_GAIN_MIN:
            learned_control_regressions.append(speech_metric_column)
    learned_coarse_target_corr = _corr_for_eval_mode(
        frame,
        x_col="oracle_bias",
        y_col="predicted_bias",
        eval_mode="learned",
    )
    learned_residual_target_corr = _mean_for_eval_mode(frame, column="residual_target_corr", eval_mode="learned")
    learned_residual_bias_share = _mean_for_eval_mode(frame, column="residual_bias_share", eval_mode="learned")
    learned_local_silence_delta_share = _mean_for_eval_mode(
        frame,
        column="local_silence_delta_share",
        eval_mode="learned",
    )
    gate0a_pass = (
        analytic_valid_items >= _GATE0_VALID_ITEMS_MIN
        and
        np.isfinite(g_domain_valid_mean)
        and g_domain_valid_mean >= 0.95
        and np.isfinite(gate0_drop_rate)
        and gate0_drop_rate <= 0.05
    )
    gate0b_pass = (
        np.isfinite(continuous_alignment_coverage)
        and continuous_alignment_coverage >= 1.0 - 1.0e-6
        and np.isfinite(alignment_mean_local_confidence_speech)
        and alignment_mean_local_confidence_speech >= _ALIGNMENT_MEAN_LOCAL_CONF_MIN
        and np.isfinite(alignment_mean_coarse_confidence_speech)
        and alignment_mean_coarse_confidence_speech >= _ALIGNMENT_MEAN_COARSE_CONF_MIN
        and np.isfinite(analytic_alignment_local_margin_p10)
        and analytic_alignment_local_margin_p10 >= _GATE0_ALIGNMENT_LOCAL_MARGIN_P10_MIN
    )
    gate0c_pass = (
        np.isfinite(analytic_signal_explainability_slope)
        and analytic_signal_explainability_slope > _GATE0_SIGNAL_SLOPE_MIN
    )
    gate0_pass = gate0a_pass and gate0b_pass and gate0c_pass
    gate1_criteria_pass = (
        "analytic" in observed_modes
        and analytic_triplet_count >= _GATE1_TRIPLETS_MIN
        and incomplete_triplets == 0
        and not missing_controls
        and np.isfinite(analytic_tempo_monotonicity_rate)
        and analytic_tempo_monotonicity_rate >= _GATE1_MONOTONICITY_RATE_MIN
        and np.isfinite(analytic_negative_control_gap)
        and analytic_negative_control_gap > _GATE1_NEGATIVE_CONTROL_GAP_MIN
        and np.isfinite(analytic_tempo_transfer_slope)
        and analytic_tempo_transfer_slope >= _GATE1_TRANSFER_SLOPE_MIN
        and np.isfinite(analytic_fast_slow_effect)
        and analytic_fast_slow_effect >= _GATE1_EFFECT_SIZE_MIN
        and np.isfinite(analytic_invalid_g_rate)
        and analytic_invalid_g_rate <= _GATE1_INVALID_G_RATE_MAX
        and (
            (not np.isfinite(analytic_clip_hit_rate))
            or analytic_clip_hit_rate <= _GATE1_MAX_CLIP_HIT_RATE
        )
        and (
            (not np.isfinite(analytic_boundary_hit_rate))
            or analytic_boundary_hit_rate <= _GATE1_MAX_BOUNDARY_HIT_RATE
        )
        and np.isfinite(analytic_projected_real_range)
        and analytic_projected_real_range >= _GATE1_PROJECTED_REAL_RANGE_MIN
        and (
            (not np.isfinite(analytic_mean_saturation))
            or analytic_mean_saturation <= _GATE1_ANALYTIC_SATURATION_RATE_MAX
        )
        and np.isfinite(analytic_mean_bucket_count)
        and analytic_mean_bucket_count >= _GATE1_PROJECTOR_BUCKET_COUNT_MIN
        and (
            (not np.isfinite(analytic_tempo_tie_rate))
            or analytic_tempo_tie_rate <= _GATE1_TIE_RATE_MAX
        )
        and (
            (not np.isfinite(analytic_anti_monotonicity_rate))
            or analytic_anti_monotonicity_rate <= _GATE1_ANTI_MONO_RATE_MAX
        )
        and (
            (not np.isfinite(analytic_same_text_gap_max))
            or analytic_same_text_gap_max <= _GATE1_SAME_TEXT_GAP_MAX
        )
    )
    gate1_pass = gate0_pass and gate1_criteria_pass
    gate2_criteria_pass = (
        "coarse_only" in observed_modes
        and np.isfinite(unmatched_speech_ratio_p95)
        and unmatched_speech_ratio_p95 <= 0.15
        and not coarse_only_runtime_limit_violations
        and not coarse_only_control_regressions
    )
    gate2_pass = gate1_pass and gate2_criteria_pass
    gate3_local_metrics_ok = True
    if "residual_target_corr" in frame.columns:
        gate3_local_metrics_ok = (
            gate3_local_metrics_ok
            and np.isfinite(learned_residual_target_corr)
            and learned_residual_target_corr >= _GATE3_RESIDUAL_CORR_MIN
        )
    if "residual_bias_share" in frame.columns:
        gate3_local_metrics_ok = (
            gate3_local_metrics_ok
            and np.isfinite(learned_residual_bias_share)
            and learned_residual_bias_share <= _GATE3_RESIDUAL_BIAS_SHARE_MAX
        )
    if "local_silence_delta_share" in frame.columns:
        gate3_local_metrics_ok = (
            gate3_local_metrics_ok
            and np.isfinite(learned_local_silence_delta_share)
            and learned_local_silence_delta_share <= _GATE3_LOCAL_SILENCE_DELTA_SHARE_MAX
        )
    coarse_corr_ok = True
    if np.isfinite(coarse_only_coarse_target_corr) and np.isfinite(learned_coarse_target_corr):
        coarse_corr_ok = learned_coarse_target_corr >= (
            coarse_only_coarse_target_corr - _GATE3_COARSE_CORR_DROP_TOL
        )
    gate3_criteria_pass = (
        gate2_pass
        and "learned" in observed_modes
        and not learned_runtime_regressions
        and not learned_control_regressions
        and gate3_local_metrics_ok
        and coarse_corr_ok
    )
    gate3_pass = gate3_criteria_pass
    return {
        "gate0a_pass": bool(gate0a_pass),
        "gate0b_pass": bool(gate0b_pass),
        "gate0c_pass": bool(gate0c_pass),
        "gate0_pass": bool(gate0_pass),
        "gate1_pass": bool(gate1_pass),
        "gate2_pass": bool(gate2_pass),
        "gate3_pass": bool(gate3_pass),
        "missing_controls": missing_controls,
        "missing_eval_modes": missing_eval_modes,
        "incomplete_triplets": int(incomplete_triplets),
        "continuous_alignment_coverage": continuous_alignment_coverage,
        "g_domain_valid_mean": g_domain_valid_mean,
        "gate0_drop_rate": gate0_drop_rate,
        "analytic_signal_explainability_slope": analytic_signal_explainability_slope,
        "analytic_explainability_slope": analytic_explainability_slope,
        "analytic_coarse_residual_slope": analytic_coarse_residual_slope,
        "analytic_negative_control_gap": analytic_negative_control_gap,
        "analytic_fast_slow_effect": analytic_fast_slow_effect,
        "analytic_clip_hit_rate": analytic_clip_hit_rate,
        "analytic_boundary_hit_rate": analytic_boundary_hit_rate,
        "analytic_projected_real_range": analytic_projected_real_range,
        "analytic_mean_saturation": analytic_mean_saturation,
        "analytic_mean_bucket_count": analytic_mean_bucket_count,
        "analytic_same_text_gap": analytic_same_text_gap,
        "analytic_same_text_gap_max": analytic_same_text_gap_max,
        "analytic_alignment_local_margin_p10": analytic_alignment_local_margin_p10,
        "analytic_tempo_monotonicity_rate": analytic_tempo_monotonicity_rate,
        "analytic_tempo_transfer_slope": analytic_tempo_transfer_slope,
        "analytic_tempo_tie_rate": analytic_tempo_tie_rate,
        "analytic_invalid_g_rate": analytic_invalid_g_rate,
        "analytic_anti_monotonicity_rate": analytic_anti_monotonicity_rate,
        "analytic_valid_items": int(analytic_valid_items),
        "analytic_triplet_count": int(analytic_triplet_count),
        "alignment_mean_local_confidence_speech": alignment_mean_local_confidence_speech,
        "alignment_mean_coarse_confidence_speech": alignment_mean_coarse_confidence_speech,
        "unmatched_speech_ratio_p95": unmatched_speech_ratio_p95,
        "coarse_only_runtime_metrics": coarse_only_runtime_metrics,
        "coarse_only_runtime_limit_violations": coarse_only_runtime_limit_violations,
        "coarse_only_control_regressions": coarse_only_control_regressions,
        "coarse_only_speech_metric": coarse_only_speech_metric,
        "coarse_only_coarse_target_corr": coarse_only_coarse_target_corr,
        "learned_runtime_metrics": learned_runtime_metrics,
        "learned_runtime_regressions": learned_runtime_regressions,
        "learned_control_regressions": learned_control_regressions,
        "learned_speech_metric": learned_speech_metric,
        "learned_coarse_target_corr": learned_coarse_target_corr,
        "learned_residual_target_corr": learned_residual_target_corr,
        "learned_residual_bias_share": learned_residual_bias_share,
        "learned_local_silence_delta_share": learned_local_silence_delta_share,
        "warnings": issues,
    }


def _warn_sparse_review_metadata(frame) -> list[str]:
    missing_issues = _collect_missing_metadata_issues(frame)
    if missing_issues:
        joined = ", ".join(missing_issues)
        print(
            "[rhythm_v3_debug_records] warning: review metadata is incomplete "
            f"({joined}). Summary export will still succeed, but Gate-0/Panel-C "
            "style slices or boundary/provenance review may be partially degenerate.",
            file=sys.stderr,
        )
    quality_issues = collect_gate_issues(frame)
    if quality_issues:
        print(
            "[rhythm_v3_debug_records] warning: gate quality checks found potential evidence gaps "
            f"({', '.join(quality_issues)}). Partial exports still succeed, but this bundle should "
            "not be read as a full gate pass.",
            file=sys.stderr,
        )
    return missing_issues + quality_issues


def _infer_single_meta(records, key, default=None):
    values = []
    prefixed_key = f"rhythm_v3_{key}"
    for record in records:
        meta = dict(getattr(record, "metadata", {}) or {})
        value = meta.get(key)
        if value is None:
            value = meta.get(prefixed_key)
        if value is not None and str(value).strip() != "":
            values.append(value)
    if not values:
        return default
    uniq = {str(value) for value in values}
    if len(uniq) > 1:
        raise RuntimeError(
            f"Mixed metadata for {key}: {sorted(uniq)}. Pass it explicitly on CLI."
        )
    return values[0]


def _coerce_contract_fingerprint_value(key: str, value):
    if value is None:
        return None
    if key in {
        "rhythm_v3_use_continuous_alignment",
        "rhythm_v3_minimal_v1_profile",
        "rhythm_v3_strict_minimal_claim_profile",
    }:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if key in {"rhythm_v3_drop_edge_runs_for_g", "rhythm_v3_src_prefix_min_support"}:
        return int(value)
    if key in {"rhythm_v3_g_trim_ratio", "rhythm_v3_min_boundary_confidence_for_g"}:
        return float(value)
    return str(value)


def _build_gate_contract_fingerprint(*, records, args) -> dict[str, object]:
    alignment_kind = str(
        _infer_single_meta(records, "alignment_kind", default="")
    ).strip().lower()
    use_continuous_alignment = None
    if alignment_kind:
        use_continuous_alignment = bool(alignment_kind.startswith("continuous"))
    fingerprint = {
        "rhythm_v3_g_variant": args.g_variant,
        "rhythm_v3_g_trim_ratio": args.g_trim_ratio,
        "rhythm_v3_drop_edge_runs_for_g": args.drop_edge_runs,
        "rhythm_v3_min_boundary_confidence_for_g": _infer_single_meta(
            records,
            "min_boundary_confidence_for_g",
            default=None,
        ),
        "rhythm_v3_src_prefix_stat_mode": _infer_single_meta(
            records,
            "src_prefix_stat_mode",
            default=None,
        ),
        "rhythm_v3_src_prefix_min_support": _infer_single_meta(
            records,
            "src_prefix_min_support",
            default=None,
        ),
        "rhythm_v3_src_rate_init_mode": _infer_single_meta(
            records,
            "src_rate_init_mode",
            default=None,
        ),
        "rhythm_v3_use_continuous_alignment": _infer_single_meta(
            records,
            "use_continuous_alignment",
            default=use_continuous_alignment,
        ),
        "rhythm_v3_alignment_mode": _infer_single_meta(
            records,
            "alignment_mode",
            default=None,
        ),
        "rhythm_v3_minimal_v1_profile": _infer_single_meta(
            records,
            "minimal_v1_profile",
            default=None,
        ),
        "rhythm_v3_strict_minimal_claim_profile": _infer_single_meta(
            records,
            "strict_minimal_claim_profile",
            default=None,
        ),
    }
    return {
        key: _coerce_contract_fingerprint_value(key, value)
        for key, value in fingerprint.items()
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single maintained review/export CLI for rhythm_v3 debug-record bundles.",
    )
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        required=True,
        help="Debug bundle file (.pt/.pth/.npz) or directory containing bundles. Can be specified multiple times.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--unit-step-ms",
        type=float,
        default=20.0,
        help="Unit step in milliseconds for reference-duration summaries. Default: 20.",
    )
    parser.add_argument(
        "--local-rate-decay",
        type=float,
        default=0.95,
        help="Decay used when reconstructing source prefix tempo if source_rate_seq is absent.",
    )
    parser.add_argument(
        "--silence-tau",
        type=float,
        default=0.35,
        help="Silence coarse clip used for oracle pseudo-target reconstruction.",
    )
    parser.add_argument(
        "--g-variant",
        default=None,
        help="Global-cue statistic variant used for summary/review reconstruction.",
    )
    parser.add_argument(
        "--g-trim-ratio",
        type=float,
        default=None,
        help="Trim ratio for trimmed_mean analysis variants.",
    )
    parser.add_argument(
        "--drop-edge-runs",
        type=int,
        default=None,
        help="Drop this many speech runs at each edge when reconstructing analysis-side g statistics.",
    )
    parser.add_argument(
        "--review-dir",
        default=None,
        help="Optional directory for exporting the retained five-figure review bundle and gate-oriented review bundle.",
    )
    parser.add_argument(
        "--strict-gates",
        action="store_true",
        help="Fail non-zero when gate-quality evidence is incomplete or below threshold.",
    )
    parser.add_argument(
        "--allow-partial-gates",
        action="store_true",
        help="Allow warning-only review exports even when --review-dir or --gate-status-json is used.",
    )
    parser.add_argument(
        "--gate-status-json",
        default=None,
        help="Optional path for writing gate_status.json style audit output.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    records = []
    for raw in args.inputs:
        records.extend(load_debug_records(raw))
    if not records:
        raise RuntimeError("No debug records found.")
    if args.g_variant is None:
        args.g_variant = str(
            _infer_single_meta(records, "g_variant", default="raw_median")
        )
    if args.g_trim_ratio is None:
        args.g_trim_ratio = float(
            _infer_single_meta(records, "g_trim_ratio", default=0.2)
        )
    if args.drop_edge_runs is None:
        args.drop_edge_runs = int(
            _infer_single_meta(
                records,
                "drop_edge_runs_for_g",
                default=_infer_single_meta(records, "g_drop_edge_runs", default=0),
            )
        )

    rows = [
        record_summary(
            record,
            unit_step_ms=args.unit_step_ms,
            local_rate_decay=args.local_rate_decay,
            silence_tau=args.silence_tau,
            g_variant=args.g_variant,
            g_trim_ratio=args.g_trim_ratio,
            drop_edge_runs=args.drop_edge_runs,
        )
        for record in records
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    strict_gate_mode = bool(
        args.strict_gates
        or (
            (args.review_dir is not None or args.gate_status_json is not None)
            and not args.allow_partial_gates
        )
    )
    if pd is not None:
        summary_df = pd.DataFrame(rows)
        ref_crop_df = build_ref_crop_table(
            records,
            g_variant=args.g_variant,
            g_trim_ratio=args.g_trim_ratio,
            drop_edge_runs=args.drop_edge_runs,
        )
        if not ref_crop_df.empty and {"sample_id", "eval_mode"}.issubset(summary_df.columns):
            crop_cols = [
                "sample_id",
                "eval_mode",
                "pair_id",
                "g_crop",
                "g_full",
                "g_crop_abs_err",
                "has_crop_comparison",
            ]
            available_crop_cols = [column for column in crop_cols if column in ref_crop_df.columns]
            merge_keys_crop = [key for key in ("sample_id", "eval_mode", "pair_id") if key in available_crop_cols and key in summary_df.columns]
            value_cols_crop = [column for column in available_crop_cols if column not in merge_keys_crop]
            if merge_keys_crop and value_cols_crop:
                summary_df = summary_df.drop(columns=value_cols_crop, errors="ignore").merge(
                    ref_crop_df[merge_keys_crop + value_cols_crop].drop_duplicates(subset=merge_keys_crop),
                    on=merge_keys_crop,
                    how="left",
                )
        prefix_silence_df = build_prefix_silence_review_table(records)
        if not prefix_silence_df.empty and {"sample_id", "eval_mode"}.issubset(summary_df.columns):
            keep_cols = [
                "sample_id",
                "eval_mode",
                "prefix_discrepancy",
                "budget_hit_rate",
                "budget_hit_pos_rate",
                "budget_hit_neg_rate",
                "cumulative_drift",
                "silence_leakage",
            ]
            summary_df = summary_df.drop(
                columns=[
                    column
                    for column in keep_cols[2:]
                    if column in summary_df.columns
                ],
                errors="ignore",
            ).merge(
                prefix_silence_df[keep_cols],
                on=["sample_id", "eval_mode"],
                how="left",
            )
        monotonicity_df = build_monotonicity_table(
            records,
            g_variant=args.g_variant,
            g_trim_ratio=args.g_trim_ratio,
            drop_edge_runs=args.drop_edge_runs,
        )
        if not monotonicity_df.empty:
            merge_keys_triplet = [
                key
                for key in ("src_id", "eval_mode", "ref_bin")
                if key in summary_df.columns and key in monotonicity_df.columns
            ]
            if merge_keys_triplet == ["src_id", "eval_mode", "ref_bin"]:
                summary_df = summary_df.merge(
                    monotonicity_df[
                        [
                            "src_id",
                            "eval_mode",
                            "ref_bin",
                            "mono_triplet_ok",
                            "tempo_delta",
                        ]
                    ].drop_duplicates(subset=["src_id", "eval_mode", "ref_bin"]),
                    on=merge_keys_triplet,
                    how="left",
                )
            mono_summary = (
                monotonicity_df.drop_duplicates(subset=["src_id", "eval_mode"])[["src_id", "eval_mode", "mono_triplet_ok"]]
                .rename(columns={"mono_triplet_ok": "tempo_monotonicity_rate"})
            )
            merge_keys = [key for key in ("src_id", "eval_mode") if key in summary_df.columns]
            if merge_keys == ["src_id", "eval_mode"]:
                summary_df = summary_df.merge(mono_summary, on=merge_keys, how="left")
        ladder_df = summarize_falsification_ladder(ref_crop_df, monotonicity_df, prefix_silence_df)
        if not ladder_df.empty and "eval_mode" in summary_df.columns and "eval_mode" in ladder_df.columns:
            ladder_cols = [
                "eval_mode",
                "signal_explainability_spearman",
                "signal_explainability_slope",
                "signal_explainability_r2_like",
                "analytic_signal_spearman",
                "analytic_signal_slope",
                "analytic_signal_r2_like",
                "coarse_residual_spearman",
                "coarse_residual_slope",
                "coarse_residual_r2_like",
                "prefix_signal_explainability_spearman",
                "prefix_signal_explainability_slope",
                "prefix_signal_explainability_r2_like",
                "explainability_spearman",
                "explainability_slope",
                "explainability_r2_like",
                "monotonicity_rate",
                "anti_monotonicity_rate",
                "tempo_tie_rate",
                "tempo_transfer_slope",
                "tempo_transfer_spearman",
                "invalid_g_rate",
                "negative_control_gap",
                "same_text_gap",
            ]
            available_ladder_cols = [column for column in ladder_cols if column in ladder_df.columns]
            summary_df = summary_df.drop(
                columns=[column for column in available_ladder_cols if column != "eval_mode" and column in summary_df.columns],
                errors="ignore",
            ).merge(
                ladder_df[available_ladder_cols].drop_duplicates(subset=["eval_mode"]),
                on=["eval_mode"],
                how="left",
            )
        issues = _warn_sparse_review_metadata(summary_df)
        summary_df.to_csv(output, index=False)
        gate_status = build_gate_status(summary_df)
        gate_status["contract_fingerprint"] = _build_gate_contract_fingerprint(
            records=records,
            args=args,
        )
        gate_status_path = None
        if args.gate_status_json:
            gate_status_path = Path(args.gate_status_json)
        elif args.review_dir:
            gate_status_path = Path(args.review_dir) / "gate_status.json"
        if gate_status_path is not None:
            gate_status_path.parent.mkdir(parents=True, exist_ok=True)
            gate_status_path.write_text(
                json.dumps(gate_status, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
        if strict_gate_mode and issues:
            msg = "[rhythm_v3_debug_records] strict gate failure: " + ", ".join(issues)
            print(msg, file=sys.stderr)
            raise SystemExit(2)
    else:  # pragma: no cover
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    if args.review_dir:
        review_dir = Path(args.review_dir)
        save_review_figure_bundle(
            records,
            output_dir=review_dir,
            g_variant=args.g_variant,
            g_trim_ratio=args.g_trim_ratio,
            drop_edge_runs=args.drop_edge_runs,
            silence_tau=args.silence_tau,
        )
        save_validation_gate_bundle(
            records,
            output_dir=review_dir,
            g_variant=args.g_variant,
            g_trim_ratio=args.g_trim_ratio,
            drop_edge_runs=args.drop_edge_runs,
        )

    print(f"[rhythm_v3_debug_records] wrote {len(rows)} rows -> {output}")


if __name__ == "__main__":
    main()
