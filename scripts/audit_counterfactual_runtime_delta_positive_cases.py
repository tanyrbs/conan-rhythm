from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np

from tasks.Conan.rhythm.duration_v3.metrics import tempo_explainability


DEFAULT_RUNTIME_REPORT = "tmp/gate1_counterfactual_probe/full_sweep/runtime_anti_control_report.json"
DEFAULT_STATIC_CSV = "tmp/gate1_boundary_audit/counterfactual_static_gate0_rows.csv"
DEFAULT_OUTPUT_JSON = "tmp/gate1_counterfactual_probe/full_sweep/delta_positive_audit_report.json"
DEFAULT_SIGN_EPS = 0.05


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the tiny number of runtime delta_g-positive cases against row-level sign/oracle evidence."
    )
    parser.add_argument("--runtime_report", default=DEFAULT_RUNTIME_REPORT, help="Aggregated runtime anti-control report JSON.")
    parser.add_argument("--static_csv", default=DEFAULT_STATIC_CSV, help="Optional static Gate-0 rows CSV for cross-reference.")
    parser.add_argument("--output_json", default=DEFAULT_OUTPUT_JSON, help="Where to write the audit report.")
    parser.add_argument("--sign_eps", type=float, default=DEFAULT_SIGN_EPS, help="Minimum absolute magnitude for sign-alignment stats.")
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


def _load_static_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _same_sign_rate(pairs: list[tuple[float, float]], *, eps: float) -> dict[str, float]:
    usable: list[tuple[float, float]] = []
    same = 0
    opposite = 0
    for left, right in pairs:
        if not (np.isfinite(left) and np.isfinite(right)):
            continue
        if abs(left) <= float(eps) or abs(right) <= float(eps):
            continue
        usable.append((left, right))
        if (left > 0.0) == (right > 0.0):
            same += 1
        else:
            opposite += 1
    count = len(usable)
    return {
        "count": float(count),
        "same_sign_rate": float(same) / float(count) if count > 0 else float("nan"),
        "opposite_sign_rate": float(opposite) / float(count) if count > 0 else float("nan"),
    }


def _find_summary_group(runtime_summary: dict[str, Any], source_name: str) -> dict[str, Any]:
    for group in runtime_summary.get("groups", []):
        if str(group.get("source_name", "")) == str(source_name):
            return group
    raise KeyError(f"Missing group for source '{source_name}'.")


def _row_domain_valid(row: dict[str, Any]) -> bool:
    domain_valid = _safe_float(row.get("counterfactual_domain_valid", row.get("g_domain_valid")))
    if np.isfinite(domain_valid):
        return bool(domain_valid > 0.5)
    return bool(_safe_float(row.get("gate0_row_dropped")) <= 0.5)


def _reconcile_case(
    *,
    runtime_summary: dict[str, Any],
    source_name: str,
    static_rows: list[dict[str, Any]],
    sign_eps: float,
) -> dict[str, Any]:
    artifact_path = Path(str(runtime_summary["artifact"]))
    payload = _load_json(artifact_path)
    real_rows = [
        row
        for row in payload.get("rows", [])
        if str(row.get("source_name", "")) == str(source_name)
        and str(row.get("ref_condition", "")) in {"slow", "mid", "fast"}
        and _row_domain_valid(row)
    ]
    group = _find_summary_group(runtime_summary, source_name)

    delta_g = [_safe_float(row.get("delta_g")) for row in real_rows]
    c_star = [_safe_float(row.get("c_star")) for row in real_rows]
    prompt_tempo_ref = [_safe_float(row.get("prompt_tempo_ref")) for row in real_rows]
    tempo_out = [_safe_float(row.get("tempo_out")) for row in real_rows]
    support_counts = [_safe_float(row.get("g_support_count")) for row in real_rows]
    unique_tempo_out = sorted({round(value, 6) for value in tempo_out if np.isfinite(value)})
    explain = tempo_explainability(delta_g, c_star)
    explain_flip = tempo_explainability([-value for value in delta_g], c_star)
    sign_stats = _same_sign_rate(list(zip(delta_g, c_star)), eps=float(sign_eps))

    static_hits = [
        row
        for row in static_rows
        if int(_safe_float(row.get("candidate_token"))) == int(runtime_summary["candidate_token"])
        and int(_safe_float(row.get("drop_edge_runs_for_g"))) == int(runtime_summary["drop_edge_runs_for_g"])
        and str(row.get("src_item_name", "")) == str(source_name)
    ]
    static_same_ref_hits = [
        row
        for row in static_hits
        if str(row.get("ref_item_name", "")) in {str(real_row.get("ref_name", "")) for real_row in real_rows}
    ]

    verdict_bits: list[str] = []
    if len(real_rows) < 3:
        verdict_bits.append(f"insufficient_domain_valid_real_rows:{len(real_rows)}")
    if bool(group.get("monotone_by_delta_g")) and not bool(group.get("monotone_by_prompt_tempo")):
        verdict_bits.append("delta_rank_passes_only_after_prompt_order_breaks")
    if np.isfinite(float(explain.get("robust_slope", float("nan")))) and float(explain["robust_slope"]) < 0.0:
        verdict_bits.append("local_delta_g_to_c_star_is_negative")
    if np.isfinite(float(explain_flip.get("robust_slope", float("nan")))) and float(explain_flip["robust_slope"]) > 0.0:
        verdict_bits.append("sign_flipped_delta_g_matches_local_c_star")
    if len(unique_tempo_out) <= 2:
        verdict_bits.append("runtime_response_has_only_two_unique_levels")
    if np.isfinite(float(group.get("real_tempo_range", float("nan")))) and float(group["real_tempo_range"]) < 0.02:
        verdict_bits.append("runtime_range_is_tiny")
    if np.isfinite(min(support_counts)) and min(support_counts) <= 1.0:
        verdict_bits.append("contains_singleton_support_ref")
    if static_same_ref_hits:
        first = static_same_ref_hits[0]
        static_delta = _safe_float(first.get("delta_g"))
        static_c = _safe_float(first.get("c_star"))
        if np.isfinite(static_delta) and np.isfinite(static_c) and static_delta * static_c < 0.0:
            verdict_bits.append("static_same_ref_row_is_sign_mismatched")

    return {
        "candidate_token": int(runtime_summary["candidate_token"]),
        "source_name": str(source_name),
        "artifact": str(artifact_path),
        "runtime_group": group,
        "real_rows": [
            {
                "ref_condition": str(row.get("ref_condition", "")),
                "ref_name": str(row.get("ref_name", "")),
                "prompt_tempo_ref": _safe_float(row.get("prompt_tempo_ref")),
                "delta_g": _safe_float(row.get("delta_g")),
                "tempo_out": _safe_float(row.get("tempo_out")),
                "tempo_src": _safe_float(row.get("tempo_src")),
                "tempo_delta": _safe_float(row.get("tempo_delta")),
                "c_star": _safe_float(row.get("c_star")),
                "g_support_count": _safe_float(row.get("g_support_count")),
                "g_domain_valid": _safe_float(row.get("g_domain_valid")),
            }
            for row in real_rows
        ],
        "local_delta_to_c_star": {
            "original": explain,
            "flipped": explain_flip,
            "same_sign_stats": sign_stats,
        },
        "runtime_shape": {
            "unique_tempo_out_count": len(unique_tempo_out),
            "unique_tempo_out": unique_tempo_out,
            "min_support_count": float(min(support_counts)) if support_counts else float("nan"),
            "max_support_count": float(max(support_counts)) if support_counts else float("nan"),
        },
        "static_cross_reference": {
            "source_row_count": len(static_hits),
            "same_ref_row_count": len(static_same_ref_hits),
            "same_ref_rows": [
                {
                    "ref_item_name": str(row.get("ref_item_name", "")),
                    "delta_g": _safe_float(row.get("delta_g")),
                    "c_star": _safe_float(row.get("c_star")),
                    "g_domain_valid": _safe_float(row.get("g_domain_valid")),
                }
                for row in static_same_ref_hits
            ],
        },
        "verdict_flags": verdict_bits,
    }


def main() -> None:
    args = _parse_args()
    runtime_payload = _load_json(Path(args.runtime_report))
    static_rows = _load_static_rows(Path(args.static_csv) if args.static_csv else None)
    cases: list[dict[str, Any]] = []
    for runtime_summary in runtime_payload.get("summaries", []):
        if not isinstance(runtime_summary, dict):
            continue
        for group in runtime_summary.get("groups", []):
            if not isinstance(group, dict):
                continue
            if bool(group.get("delta_g_control_response")):
                cases.append(
                    _reconcile_case(
                        runtime_summary=runtime_summary,
                        source_name=str(group.get("source_name", "")),
                        static_rows=static_rows,
                        sign_eps=float(args.sign_eps),
                    )
                )

    payload = {
        "runtime_report": args.runtime_report,
        "static_csv": args.static_csv,
        "case_count": len(cases),
        "cases": cases,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[delta-positive-audit] cases={len(cases)}")
    print(f"[delta-positive-audit] wrote_json={output_json}")
    for case in cases:
        orig = case["local_delta_to_c_star"]["original"]["robust_slope"]
        flip = case["local_delta_to_c_star"]["flipped"]["robust_slope"]
        print(
            "[delta-positive-audit] "
            f"token={case['candidate_token']} source={case['source_name']} "
            f"runtime_delta_pass={case['runtime_group']['delta_g_control_response']} "
            f"prompt_pass={case['runtime_group']['prompt_control_response']} "
            f"local_orig_slope={orig:.4f} local_flip_slope={flip:.4f} "
            f"flags={case['verdict_flags']}"
        )


if __name__ == "__main__":
    main()
