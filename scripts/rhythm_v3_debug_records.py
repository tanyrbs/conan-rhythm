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
from utils.plot.rhythm_v3_viz.review import (
    _is_real_reference_condition,
    _normalize_ref_bin,
    _normalize_ref_condition,
)
from tasks.Conan.rhythm.duration_v3.gate_contract import build_runtime_contract_id_from_values
from tasks.Conan.rhythm.duration_v3.gate_status import (
    build_gate_status,
    build_review_contract_fingerprint,
    collect_review_issues,
)

def _warn_sparse_review_metadata(frame) -> list[str]:
    issues = collect_review_issues(frame)
    if issues:
        print(
            "[rhythm_v3_debug_records] warning: review or gate evidence is incomplete "
            f"({', '.join(issues)}). Partial exports still succeed, but this bundle should "
            "not be read as a full gate pass.",
            file=sys.stderr,
        )
    return issues


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
        if "ref_bin" in summary_df.columns:
            ref_bin_series = summary_df["ref_bin"].map(_normalize_ref_bin)
            if "ref_condition" not in summary_df.columns:
                summary_df["ref_condition"] = ref_bin_series
            else:
                ref_condition = summary_df["ref_condition"].astype(str).str.strip()
                missing_ref_condition = ref_condition.eq("") | ref_condition.str.lower().eq("nan")
                summary_df.loc[missing_ref_condition, "ref_condition"] = ref_bin_series.loc[missing_ref_condition]
        if "triplet_id" not in summary_df.columns and {"src_id", "eval_mode", "ref_bin"}.issubset(summary_df.columns):
            ref_bin_series = summary_df["ref_bin"].map(_normalize_ref_bin)
            ref_condition = (
                summary_df["ref_condition"].map(_normalize_ref_condition)
                if "ref_condition" in summary_df.columns
                else ref_bin_series.map(_normalize_ref_condition)
            )
            real_mask = np.asarray(
                [
                    _is_real_reference_condition(cond, ref_bin=bin_value)
                    for cond, bin_value in zip(ref_condition.tolist(), ref_bin_series.tolist())
                ],
                dtype=bool,
            )
            summary_df["triplet_id"] = np.where(
                real_mask,
                summary_df["src_id"].astype(str) + "::" + summary_df["eval_mode"].astype(str),
                np.nan,
            )
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
                "control_contract_id",
                "g_crop",
                "g_full",
                "g_crop_abs_err",
                "has_crop_comparison",
            ]
            available_crop_cols = [column for column in crop_cols if column in ref_crop_df.columns]
            merge_keys_crop = [
                key
                for key in ("sample_id", "eval_mode", "pair_id", "control_contract_id")
                if key in available_crop_cols and key in summary_df.columns
            ]
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
                "control_contract_id",
                "prefix_discrepancy",
                "z_prefix_discrepancy",
                "preproj_exec_prefix_discrepancy",
                "disc_exec_prefix_discrepancy",
                "budget_hit_rate",
                "budget_hit_pos_rate",
                "budget_hit_neg_rate",
                "cumulative_drift",
                "cumulative_drift_mean_abs",
                "final_prefix_drift_abs_mean",
                "final_prefix_offset_abs_mean",
                "max_prefix_offset_abs",
                "silence_leakage",
            ]
            available_prefix_cols = [column for column in keep_cols if column in prefix_silence_df.columns]
            prefix_merge_keys = [
                key
                for key in ("sample_id", "eval_mode", "control_contract_id")
                if key in available_prefix_cols and key in summary_df.columns
            ]
            value_cols_prefix = [column for column in available_prefix_cols if column not in prefix_merge_keys]
            if prefix_merge_keys and value_cols_prefix:
                prefix_summary_df = (
                    prefix_silence_df[prefix_merge_keys + value_cols_prefix]
                    .groupby(prefix_merge_keys, as_index=False)
                    .mean(numeric_only=True)
                )
                summary_df = summary_df.drop(
                    columns=[column for column in value_cols_prefix if column in summary_df.columns],
                    errors="ignore",
                ).merge(
                    prefix_summary_df,
                    on=prefix_merge_keys,
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
                for key in ("src_id", "eval_mode", "ref_bin", "control_contract_id")
                if key in summary_df.columns and key in monotonicity_df.columns
            ]
            if {"src_id", "eval_mode", "ref_bin"}.issubset(set(merge_keys_triplet)):
                summary_df = summary_df.merge(
                    monotonicity_df[
                        [
                            "src_id",
                            "eval_mode",
                            "ref_bin",
                            *(
                                ["control_contract_id"]
                                if "control_contract_id" in monotonicity_df.columns
                                else []
                            ),
                            "mono_triplet_ok",
                            "mono_triplet_ok_raw",
                            "mono_triplet_ok_preproj",
                            "mono_triplet_ok_exec",
                            "tempo_tie_triplet_raw",
                            "tempo_tie_triplet_preproj",
                            "tempo_tie_triplet_exec",
                            "tempo_delta",
                            "tempo_delta_raw",
                            "tempo_delta_preproj",
                            "tempo_delta_exec",
                        ]
                    ].drop_duplicates(subset=merge_keys_triplet),
                    on=merge_keys_triplet,
                    how="left",
                )
            mono_summary = (
                monotonicity_df.drop_duplicates(
                    subset=[key for key in ("src_id", "eval_mode", "control_contract_id") if key in monotonicity_df.columns]
                )[[
                    "src_id",
                    "eval_mode",
                    *(
                        ["control_contract_id"]
                        if "control_contract_id" in monotonicity_df.columns
                        else []
                    ),
                    "mono_triplet_ok",
                ]]
                .rename(columns={"mono_triplet_ok": "tempo_monotonicity_rate"})
            )
            merge_keys = [key for key in ("src_id", "eval_mode", "control_contract_id") if key in summary_df.columns and key in mono_summary.columns]
            if {"src_id", "eval_mode"}.issubset(set(merge_keys)):
                summary_df = summary_df.merge(mono_summary, on=merge_keys, how="left")
        ladder_df = summarize_falsification_ladder(ref_crop_df, monotonicity_df, prefix_silence_df)
        if not ladder_df.empty and "eval_mode" in summary_df.columns and "eval_mode" in ladder_df.columns:
            ladder_cols = [
                "eval_mode",
                "control_contract_id",
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
                "monotonicity_rate_raw",
                "monotonicity_rate_preproj",
                "monotonicity_rate_exec",
                "anti_monotonicity_rate",
                "tempo_tie_rate",
                "tempo_tie_rate_raw",
                "tempo_tie_rate_preproj",
                "tempo_tie_rate_exec",
                "tempo_transfer_slope",
                "tempo_transfer_slope_raw",
                "tempo_transfer_slope_preproj",
                "tempo_transfer_slope_exec",
                "tempo_transfer_spearman",
                "invalid_g_rate",
                "negative_control_gap",
                "same_text_gap",
            ]
            available_ladder_cols = [column for column in ladder_cols if column in ladder_df.columns]
            summary_df = summary_df.drop(
                columns=[
                    column
                    for column in available_ladder_cols
                    if column not in {"eval_mode", "control_contract_id"} and column in summary_df.columns
                ],
                errors="ignore",
            ).merge(
                ladder_df[available_ladder_cols].drop_duplicates(
                    subset=[key for key in ("eval_mode", "control_contract_id") if key in available_ladder_cols]
                ),
                on=[key for key in ("eval_mode", "control_contract_id") if key in available_ladder_cols and key in summary_df.columns],
                how="left",
            )
        issues = _warn_sparse_review_metadata(summary_df)
        summary_df.to_csv(output, index=False)
        gate_status = build_gate_status(summary_df)
        contract_fingerprint = build_review_contract_fingerprint(
            records=records,
            args=args,
        )
        gate_status["contract_fingerprint"] = contract_fingerprint
        gate_status["contract_id"] = build_runtime_contract_id_from_values(contract_fingerprint)
        gate_status["control_contract_id"] = str(
            gate_status.get("control_contract_id", gate_status["contract_id"]) or gate_status["contract_id"]
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
