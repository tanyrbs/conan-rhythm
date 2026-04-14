from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_INPUT_GLOB = "tmp/gate1_counterfactual_probe/anti_control/*_summary.json"
DEFAULT_DIRECTION_REPORT = "tmp/gate1_boundary_audit/counterfactual_static_gate0_direction_report.json"
DEFAULT_OUTPUT_JSON = "tmp/gate1_counterfactual_probe/anti_control/runtime_anti_control_report.json"
DEFAULT_MIN_REAL_RANGE = 0.01
DEFAULT_MIN_CONTROL_GAP = 0.01


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate runtime anti-control evidence from Gate-1 counterfactual probe summaries."
    )
    parser.add_argument("--input_glob", default=DEFAULT_INPUT_GLOB, help="Glob for summary JSONs emitted by probe_rhythm_v3_gate1_silent_counterfactual.py.")
    parser.add_argument("--direction_report", default=DEFAULT_DIRECTION_REPORT, help="Optional static Gate-0 directionality report JSON.")
    parser.add_argument("--output_json", default=DEFAULT_OUTPUT_JSON, help="Where to write the aggregated report.")
    parser.add_argument("--min_real_range", type=float, default=DEFAULT_MIN_REAL_RANGE, help="Minimum real_tempo_range to count as a non-flat runtime response.")
    parser.add_argument("--min_control_gap", type=float, default=DEFAULT_MIN_CONTROL_GAP, help="Minimum gap versus controls to count as a meaningful runtime response.")
    return parser.parse_args()


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict JSON payload: {path}")
    return payload


def _load_direction_index(path: Path | None) -> dict[tuple[int, int], dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = _load_json(path)
    summary = payload.get("summary")
    if not isinstance(summary, list):
        return {}
    index: dict[tuple[int, int], dict[str, Any]] = {}
    for row in summary:
        if not isinstance(row, dict):
            continue
        token = int(row.get("candidate_token", 0) or 0)
        drop_edge = int(row.get("drop_edge_runs_for_g", 0) or 0)
        index[(token, drop_edge)] = row
    return index


def _max_control_gap(row: dict[str, Any]) -> float:
    gap_source = _safe_float(row.get("max_gap_vs_source_only"))
    gap_random = _safe_float(row.get("max_gap_vs_random_ref"))
    gaps = [value for value in (gap_source, gap_random) if np.isfinite(value)]
    if not gaps:
        return float("nan")
    return float(max(gaps))


def _runtime_flags(row: dict[str, Any], *, min_real_range: float, min_control_gap: float) -> dict[str, Any]:
    real_range = _safe_float(row.get("real_tempo_range"))
    max_gap = _max_control_gap(row)
    valid_real_count = int(_safe_float(row.get("valid_real_row_count")))
    if valid_real_count <= 0:
        valid_real_count = 3
    active = bool(
        valid_real_count >= 3
        and np.isfinite(real_range)
        and real_range >= float(min_real_range)
        and np.isfinite(max_gap)
        and max_gap >= float(min_control_gap)
    )
    prompt_slope = _safe_float(row.get("transfer_slope"))
    delta_slope = _safe_float(row.get("delta_g_transfer_slope"))
    anti_slope = _safe_float(row.get("neg_delta_g_transfer_slope"))
    prompt = bool(active and row.get("monotone_by_prompt_tempo") and np.isfinite(prompt_slope) and prompt_slope > 0.0)
    delta = bool(active and row.get("monotone_by_delta_g") and np.isfinite(delta_slope) and delta_slope > 0.0)
    anti = bool(active and row.get("monotone_by_neg_delta_g") and np.isfinite(anti_slope) and anti_slope > 0.0)
    return {
        "active_runtime_response": active,
        "prompt_control_response": prompt,
        "delta_g_control_response": delta,
        "anti_control_response": anti,
        "max_control_gap": max_gap,
    }


def _source_speaker(source_name: str) -> str:
    return str(source_name).split("_", 1)[0] if source_name else ""


def _summarize_file(
    path: Path,
    *,
    direction_index: dict[tuple[int, int], dict[str, Any]],
    min_real_range: float,
    min_control_gap: float,
) -> dict[str, Any]:
    payload = _load_json(path)
    rows = payload.get("rows")
    summary_rows = payload.get("summary")
    if not isinstance(rows, list) or not isinstance(summary_rows, list):
        raise ValueError(f"Unexpected summary payload: {path}")
    candidate_token = int(payload.get("candidate_token", 0) or 0)
    drop_edge = 0
    for raw_row in rows:
        if isinstance(raw_row, dict):
            drop_edge = int(_safe_float(raw_row.get("g_drop_edge_runs")))
            if drop_edge:
                break

    groups: list[dict[str, Any]] = []
    prompt_count = 0
    delta_count = 0
    anti_count = 0
    active_count = 0
    anti_only_sources: list[str] = []
    prompt_only_sources: list[str] = []
    flat_sources: list[str] = []
    for row in summary_rows:
        if not isinstance(row, dict):
            continue
        source_name = str(row.get("source_name", ""))
        flags = _runtime_flags(
            row,
            min_real_range=float(min_real_range),
            min_control_gap=float(min_control_gap),
        )
        if flags["active_runtime_response"]:
            active_count += 1
        else:
            flat_sources.append(source_name)
        if flags["prompt_control_response"]:
            prompt_count += 1
        if flags["delta_g_control_response"]:
            delta_count += 1
        if flags["anti_control_response"]:
            anti_count += 1
        if flags["anti_control_response"] and not flags["prompt_control_response"]:
            anti_only_sources.append(source_name)
        if flags["prompt_control_response"] and not flags["anti_control_response"]:
            prompt_only_sources.append(source_name)
        group = {
            "source_name": source_name,
            "speaker": _source_speaker(source_name),
            "valid_real_row_count": int(_safe_float(row.get("valid_real_row_count"))),
            "total_real_row_count": int(_safe_float(row.get("total_real_row_count"))),
            "all_real_domain_valid": bool(row.get("all_real_domain_valid")),
            "real_tempo_range": _safe_float(row.get("real_tempo_range")),
            "max_control_gap": flags["max_control_gap"],
            "prompt_control_response": flags["prompt_control_response"],
            "delta_g_control_response": flags["delta_g_control_response"],
            "anti_control_response": flags["anti_control_response"],
            "prompt_slope": _safe_float(row.get("transfer_slope")),
            "delta_g_slope": _safe_float(row.get("delta_g_transfer_slope")),
            "anti_control_slope": _safe_float(row.get("neg_delta_g_transfer_slope")),
            "monotone_by_prompt_tempo": bool(row.get("monotone_by_prompt_tempo")),
            "monotone_by_delta_g": bool(row.get("monotone_by_delta_g")),
            "monotone_by_neg_delta_g": bool(row.get("monotone_by_neg_delta_g")),
            "prompt_sorted_ref_conditions": row.get("prompt_sorted_ref_conditions"),
            "delta_g_sorted_ref_conditions": row.get("delta_g_sorted_ref_conditions"),
            "anti_control_sorted_ref_conditions": row.get("anti_control_sorted_ref_conditions"),
        }
        groups.append(group)

    directionality = direction_index.get((candidate_token, drop_edge), {})
    verdict = "anti_control_not_needed_or_uninformative"
    if anti_count > 0 and prompt_count <= 0:
        verdict = "anti_control_matches_runtime_residuals_but_prompt_control_does_not"
    elif anti_count > prompt_count:
        verdict = "anti_control_covers_more_runtime_residuals_than_prompt_control"
    elif anti_count > 0 and prompt_count > 0:
        verdict = "anti_control_and_prompt_both_cover_some_runtime_residuals"
    elif active_count > 0:
        verdict = "runtime_residuals_exist_but_anti_control_does_not_rescue_them"

    return {
        "artifact": str(path),
        "artifact_name": path.stem,
        "candidate_token": candidate_token,
        "drop_edge_runs_for_g": drop_edge,
        "trim_head_runs": int(payload.get("trim_head_runs", 0) or 0),
        "trim_tail_runs": int(payload.get("trim_tail_runs", 0) or 0),
        "min_real_range": float(min_real_range),
        "min_control_gap": float(min_control_gap),
        "source_count": len(groups),
        "active_runtime_source_count": active_count,
        "prompt_control_response_count": prompt_count,
        "delta_g_control_response_count": delta_count,
        "anti_control_response_count": anti_count,
        "anti_only_sources": anti_only_sources,
        "prompt_only_sources": prompt_only_sources,
        "flat_or_control_indistinct_sources": flat_sources,
        "verdict": verdict,
        "static_directionality": {
            "overall_same_sign_rate": _safe_float(directionality.get("overall", {}).get("same_sign_rate")),
            "overall_opposite_sign_rate": _safe_float(directionality.get("overall", {}).get("opposite_sign_rate")),
            "overall_original_slope": _safe_float(directionality.get("overall", {}).get("original_robust_slope")),
            "overall_flipped_slope": _safe_float(directionality.get("overall", {}).get("flipped_robust_slope")),
            "valid_same_sign_rate": _safe_float(directionality.get("valid_only", {}).get("same_sign_rate")),
            "valid_opposite_sign_rate": _safe_float(directionality.get("valid_only", {}).get("opposite_sign_rate")),
            "valid_original_slope": _safe_float(directionality.get("valid_only", {}).get("original_robust_slope")),
            "valid_flipped_slope": _safe_float(directionality.get("valid_only", {}).get("flipped_robust_slope")),
        },
        "groups": groups,
    }


def main() -> None:
    args = _parse_args()
    matched = sorted(Path(path) for path in glob.glob(args.input_glob))
    direction_index = _load_direction_index(Path(args.direction_report) if args.direction_report else None)
    summaries = [
        _summarize_file(
            path,
            direction_index=direction_index,
            min_real_range=float(args.min_real_range),
            min_control_gap=float(args.min_control_gap),
        )
        for path in matched
    ]
    payload = {
        "input_glob": args.input_glob,
        "matched_count": len(matched),
        "min_real_range": float(args.min_real_range),
        "min_control_gap": float(args.min_control_gap),
        "direction_report": args.direction_report,
        "summaries": summaries,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[runtime-anti-control] matched={len(matched)}")
    print(f"[runtime-anti-control] wrote_json={output_json}")
    for summary in summaries:
        print(
            "[runtime-anti-control] "
            f"{summary['artifact_name']} token={summary['candidate_token']} drop_edge={summary['drop_edge_runs_for_g']} "
            f"active={summary['active_runtime_source_count']}/{summary['source_count']} "
            f"prompt={summary['prompt_control_response_count']} "
            f"delta={summary['delta_g_control_response_count']} "
            f"anti={summary['anti_control_response_count']} "
            f"verdict={summary['verdict']}"
        )


if __name__ == "__main__":
    main()
