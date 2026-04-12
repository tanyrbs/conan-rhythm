#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
    save_review_figure_bundle,
    save_validation_gate_bundle,
)


def _warn_sparse_review_metadata(frame) -> None:
    if pd is None or frame is None or not hasattr(frame, "columns"):
        return
    total = int(frame.shape[0])
    if total <= 0:
        return
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
    if issues:
        joined = ", ".join(issues)
        print(
            "[rhythm_v3_debug_records] warning: review metadata is incomplete "
            f"({joined}). Summary export will still succeed, but Gate-0/Panel-C "
            "style slices or boundary/provenance review may be partially degenerate.",
            file=sys.stderr,
        )
    quality_issues = []
    if "g_valid" in frame.columns:
        g_valid = pd.to_numeric(frame["g_valid"], errors="coerce")
        g_valid_rate = float(np.nanmean(g_valid.to_numpy(dtype=np.float32)))
        if np.isfinite(g_valid_rate) and g_valid_rate < 0.95:
            quality_issues.append(f"g_valid_mean={g_valid_rate:.3f}<0.950")
    if "gate0_row_dropped" in frame.columns:
        gate0_drop = pd.to_numeric(frame["gate0_row_dropped"], errors="coerce").to_numpy(dtype=np.float32)
        gate0_nan_rate = float(np.nanmean(gate0_drop)) if gate0_drop.size > 0 else float("nan")
        if np.isfinite(gate0_nan_rate) and gate0_nan_rate > 0.05:
            quality_issues.append(f"gate0_nan_rate={gate0_nan_rate:.3f}>0.050")
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
        if not ref_conditions.intersection({"source_only", "random_ref", "shuffled_ref"}):
            quality_issues.append("negative_control_reference=missing")
    if quality_issues:
        print(
            "[rhythm_v3_debug_records] warning: gate quality checks found potential evidence gaps "
            f"({', '.join(quality_issues)}). Partial exports still succeed, but this bundle should "
            "not be read as a full gate pass.",
            file=sys.stderr,
        )


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
    if pd is not None:
        summary_df = pd.DataFrame(rows)
        _warn_sparse_review_metadata(summary_df)
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
        summary_df.to_csv(output, index=False)
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
