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
)


_GATE1_MONOTONICITY_RATE_MIN = 0.95
_REQUIRED_NEGATIVE_CONTROLS = ("source_only", "random_ref", "shuffled_ref")
_GATE0_EXPLAINABILITY_SLOPE_MIN = 0.0
_GATE1_NEGATIVE_CONTROL_GAP_MIN = 0.0
_GATE1_SAME_TEXT_GAP_MAX = 0.10
_GATE0_ALIGNMENT_LOCAL_MARGIN_P10_MIN = 0.02
_GATE2_RUNTIME_DEGRADATION_TOLERANCES = {
    "silence_leakage": 0.05,
    "prefix_discrepancy": 0.05,
    "budget_hit_rate": 0.05,
    "cumulative_drift": 0.25,
}
_GATE2_MONOTONICITY_DROP_TOL = 0.01
_GATE2_TRANSFER_SLOPE_DROP_TOL = 0.05


def _mean_for_eval_mode(frame, *, column: str, eval_mode: str) -> float:
    if column not in frame.columns or "eval_mode" not in frame.columns:
        return float("nan")
    mode_mask = frame["eval_mode"].astype(str).str.strip().str.lower() == str(eval_mode).strip().lower()
    values = pd.to_numeric(frame.loc[mode_mask, column], errors="coerce").to_numpy(dtype=np.float32)
    return float(np.nanmean(values)) if values.size > 0 and np.isfinite(values).any() else float("nan")


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
            analytic_slope = _mean_for_eval_mode(frame, column="explainability_slope", eval_mode="analytic")
            if not np.isfinite(analytic_slope):
                quality_issues.append("analytic_explainability_slope=missing")
            elif analytic_slope <= _GATE0_EXPLAINABILITY_SLOPE_MIN:
                quality_issues.append(
                    f"analytic_explainability_slope={analytic_slope:.3f}<={_GATE0_EXPLAINABILITY_SLOPE_MIN:.3f}"
                )
            analytic_neg_gap = _mean_for_eval_mode(frame, column="negative_control_gap", eval_mode="analytic")
            if not np.isfinite(analytic_neg_gap):
                quality_issues.append("analytic_negative_control_gap=missing")
            elif analytic_neg_gap <= _GATE1_NEGATIVE_CONTROL_GAP_MIN:
                quality_issues.append(
                    f"analytic_negative_control_gap={analytic_neg_gap:.3f}<={_GATE1_NEGATIVE_CONTROL_GAP_MIN:.3f}"
                )
            analytic_same_text_gap = _mean_for_eval_mode(frame, column="same_text_gap", eval_mode="analytic")
            if np.isfinite(analytic_same_text_gap) and analytic_same_text_gap > _GATE1_SAME_TEXT_GAP_MAX:
                quality_issues.append(
                    f"analytic_same_text_gap={analytic_same_text_gap:.3f}>{_GATE1_SAME_TEXT_GAP_MAX:.3f}"
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
        if {"coarse_only", "learned"}.issubset(observed_modes):
            for column, tolerance in _GATE2_RUNTIME_DEGRADATION_TOLERANCES.items():
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
            coarse_mono = _mean_for_eval_mode(frame, column="monotonicity_rate", eval_mode="coarse_only")
            learned_mono = _mean_for_eval_mode(frame, column="monotonicity_rate", eval_mode="learned")
            if not np.isfinite(coarse_mono) or not np.isfinite(learned_mono):
                quality_issues.append("monotonicity_rate_mode_mean=missing")
            elif learned_mono < (coarse_mono - _GATE2_MONOTONICITY_DROP_TOL):
                quality_issues.append(
                    f"learned_monotonicity_rate_regression={learned_mono:.3f}<{coarse_mono:.3f}-{_GATE2_MONOTONICITY_DROP_TOL:.3f}"
                )
            coarse_transfer = _mean_for_eval_mode(frame, column="tempo_transfer_slope", eval_mode="coarse_only")
            learned_transfer = _mean_for_eval_mode(frame, column="tempo_transfer_slope", eval_mode="learned")
            if not np.isfinite(coarse_transfer) or not np.isfinite(learned_transfer):
                quality_issues.append("tempo_transfer_slope_mode_mean=missing")
            elif learned_transfer < (coarse_transfer - _GATE2_TRANSFER_SLOPE_DROP_TOL):
                quality_issues.append(
                    f"learned_tempo_transfer_slope_regression={learned_transfer:.3f}<{coarse_transfer:.3f}-{_GATE2_TRANSFER_SLOPE_DROP_TOL:.3f}"
                )
    if {"src_id", "eval_mode", "ref_bin"}.issubset(frame.columns):
        triplet_frame = frame.copy()
        if "ref_condition" in triplet_frame.columns:
            ref_condition = triplet_frame["ref_condition"].astype(str).str.strip().str.lower()
            real_mask = ref_condition.isin({"", "nan", "real", "real_reference"})
            triplet_frame = triplet_frame[real_mask]
        triplet_frame = triplet_frame[triplet_frame["ref_bin"].astype(str).str.strip().str.lower().isin({"slow", "mid", "fast"})]
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
            "gate0_pass": False,
            "gate1_pass": False,
            "gate2_pass": False,
            "missing_controls": list(_REQUIRED_NEGATIVE_CONTROLS),
            "missing_eval_modes": ["analytic", "coarse_only", "learned"],
            "incomplete_triplets": 0,
            "continuous_alignment_coverage": float("nan"),
            "g_domain_valid_mean": float("nan"),
            "gate0_drop_rate": float("nan"),
            "analytic_explainability_slope": float("nan"),
            "analytic_negative_control_gap": float("nan"),
            "analytic_same_text_gap": float("nan"),
            "analytic_tempo_monotonicity_rate": float("nan"),
            "unmatched_speech_ratio_p95": float("nan"),
            "coarse_only_runtime_metrics": {},
            "learned_runtime_metrics": {},
            "learned_runtime_regressions": [],
            "learned_control_regressions": [],
            "warnings": ["summary_rows=empty"],
        }
    issues = collect_gate_issues(frame)
    missing_controls: list[str] = []
    missing_eval_modes: list[str] = []
    incomplete_triplets = 0
    continuous_alignment_coverage = float("nan")
    g_domain_valid_mean = float("nan")
    gate0_drop_rate = float("nan")
    analytic_explainability_slope = float("nan")
    analytic_negative_control_gap = float("nan")
    analytic_same_text_gap = float("nan")
    analytic_alignment_local_margin_p10 = float("nan")
    analytic_tempo_monotonicity_rate = float("nan")
    unmatched_speech_ratio_p95 = float("nan")
    coarse_only_runtime_metrics: dict[str, float] = {}
    learned_runtime_metrics: dict[str, float] = {}
    learned_runtime_regressions: list[str] = []
    learned_control_regressions: list[str] = []
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
    if "analytic" in observed_modes:
        analytic_explainability_slope = _mean_for_eval_mode(
            frame,
            column="explainability_slope",
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
        analytic_alignment_local_margin_p10 = _mean_for_eval_mode(
            frame,
            column="alignment_local_margin_p10",
            eval_mode="analytic",
        )
        analytic_tempo_monotonicity_rate = _mean_for_eval_mode(
            frame,
            column="tempo_monotonicity_rate",
            eval_mode="analytic",
        )
    for column, tolerance in _GATE2_RUNTIME_DEGRADATION_TOLERANCES.items():
        coarse_only_runtime_metrics[column] = _mean_for_eval_mode(frame, column=column, eval_mode="coarse_only")
        learned_runtime_metrics[column] = _mean_for_eval_mode(frame, column=column, eval_mode="learned")
        coarse_mean = coarse_only_runtime_metrics[column]
        learned_mean = learned_runtime_metrics[column]
        if np.isfinite(coarse_mean) and np.isfinite(learned_mean) and learned_mean > (coarse_mean + float(tolerance)):
            learned_runtime_regressions.append(column)
    coarse_mono = _mean_for_eval_mode(frame, column="monotonicity_rate", eval_mode="coarse_only")
    learned_mono = _mean_for_eval_mode(frame, column="monotonicity_rate", eval_mode="learned")
    if np.isfinite(coarse_mono) and np.isfinite(learned_mono) and learned_mono < (coarse_mono - _GATE2_MONOTONICITY_DROP_TOL):
        learned_control_regressions.append("monotonicity_rate")
    coarse_transfer = _mean_for_eval_mode(frame, column="tempo_transfer_slope", eval_mode="coarse_only")
    learned_transfer = _mean_for_eval_mode(frame, column="tempo_transfer_slope", eval_mode="learned")
    if np.isfinite(coarse_transfer) and np.isfinite(learned_transfer) and learned_transfer < (coarse_transfer - _GATE2_TRANSFER_SLOPE_DROP_TOL):
        learned_control_regressions.append("tempo_transfer_slope")
    gate0_pass = (
        np.isfinite(g_domain_valid_mean)
        and g_domain_valid_mean >= 0.95
        and np.isfinite(gate0_drop_rate)
        and gate0_drop_rate <= 0.05
        and np.isfinite(continuous_alignment_coverage)
        and continuous_alignment_coverage >= 1.0 - 1.0e-6
        and np.isfinite(analytic_explainability_slope)
        and analytic_explainability_slope > _GATE0_EXPLAINABILITY_SLOPE_MIN
        and np.isfinite(analytic_alignment_local_margin_p10)
        and analytic_alignment_local_margin_p10 >= _GATE0_ALIGNMENT_LOCAL_MARGIN_P10_MIN
    )
    gate1_criteria_pass = (
        "analytic" in observed_modes
        and incomplete_triplets == 0
        and not missing_controls
        and np.isfinite(analytic_tempo_monotonicity_rate)
        and analytic_tempo_monotonicity_rate >= _GATE1_MONOTONICITY_RATE_MIN
        and np.isfinite(analytic_negative_control_gap)
        and analytic_negative_control_gap > _GATE1_NEGATIVE_CONTROL_GAP_MIN
        and (
            (not np.isfinite(analytic_same_text_gap))
            or analytic_same_text_gap <= _GATE1_SAME_TEXT_GAP_MAX
        )
    )
    gate1_pass = gate0_pass and gate1_criteria_pass
    gate2_criteria_pass = (
        {"coarse_only", "learned"}.issubset(observed_modes)
        and np.isfinite(unmatched_speech_ratio_p95)
        and unmatched_speech_ratio_p95 <= 0.15
        and not learned_runtime_regressions
        and not learned_control_regressions
    )
    gate2_pass = gate1_pass and gate2_criteria_pass
    return {
        "gate0_pass": bool(gate0_pass),
        "gate1_pass": bool(gate1_pass),
        "gate2_pass": bool(gate2_pass),
        "missing_controls": missing_controls,
        "missing_eval_modes": missing_eval_modes,
        "incomplete_triplets": int(incomplete_triplets),
        "continuous_alignment_coverage": continuous_alignment_coverage,
        "g_domain_valid_mean": g_domain_valid_mean,
        "gate0_drop_rate": gate0_drop_rate,
        "analytic_explainability_slope": analytic_explainability_slope,
        "analytic_negative_control_gap": analytic_negative_control_gap,
        "analytic_same_text_gap": analytic_same_text_gap,
        "analytic_alignment_local_margin_p10": analytic_alignment_local_margin_p10,
        "analytic_tempo_monotonicity_rate": analytic_tempo_monotonicity_rate,
        "unmatched_speech_ratio_p95": unmatched_speech_ratio_p95,
        "coarse_only_runtime_metrics": coarse_only_runtime_metrics,
        "learned_runtime_metrics": learned_runtime_metrics,
        "learned_runtime_regressions": learned_runtime_regressions,
        "learned_control_regressions": learned_control_regressions,
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
        default="raw_median",
        help="Global-cue statistic variant used for summary/review reconstruction.",
    )
    parser.add_argument(
        "--g-trim-ratio",
        type=float,
        default=0.2,
        help="Trim ratio for trimmed_mean analysis variants.",
    )
    parser.add_argument(
        "--drop-edge-runs",
        type=int,
        default=0,
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
        issues = _warn_sparse_review_metadata(summary_df)
        summary_df.to_csv(output, index=False)
        gate_status = build_gate_status(summary_df)
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
