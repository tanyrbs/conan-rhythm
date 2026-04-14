from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.commons.single_thread_env import apply_single_thread_env

apply_single_thread_env()

import numpy as np

from tasks.Conan.rhythm.duration_v3.metrics import tempo_explainability


DEFAULT_INPUT_CSV = "tmp/gate1_boundary_audit/counterfactual_static_gate0_rows.csv"
DEFAULT_OUTPUT_JSON = "tmp/gate1_boundary_audit/counterfactual_static_gate0_direction_report.json"
DEFAULT_SIGN_EPS = 0.05


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Directionality audit for counterfactual static Gate 0 rows."
    )
    parser.add_argument("--input_csv", default=DEFAULT_INPUT_CSV, help="Row CSV from audit_rhythm_v3_counterfactual_static_gate0.py")
    parser.add_argument("--output_json", default=DEFAULT_OUTPUT_JSON, help="Where to write the directionality summary JSON.")
    parser.add_argument("--sign_eps", type=float, default=DEFAULT_SIGN_EPS, help="Minimum absolute magnitude to count a row in sign-agreement stats.")
    return parser.parse_args()


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _safe_int(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sign_stats(rows: list[dict[str, Any]], *, x_key: str, y_key: str, eps: float) -> dict[str, float]:
    pairs: list[tuple[float, float]] = []
    same = 0
    opposite = 0
    for row in rows:
        x_value = _safe_float(row.get(x_key))
        y_value = _safe_float(row.get(y_key))
        if not (np.isfinite(x_value) and np.isfinite(y_value)):
            continue
        if abs(x_value) <= float(eps) or abs(y_value) <= float(eps):
            continue
        pairs.append((x_value, y_value))
        sign_delta = 1 if x_value > 0.0 else -1
        sign_c = 1 if y_value > 0.0 else -1
        if sign_delta == sign_c:
            same += 1
        elif sign_delta == -sign_c:
            opposite += 1
    count = len(pairs)
    if count <= 0:
        return {
            "count": 0.0,
            "same_sign_rate": float("nan"),
            "opposite_sign_rate": float("nan"),
            "same_minus_opposite": float("nan"),
        }
    same_rate = float(same) / float(count)
    opposite_rate = float(opposite) / float(count)
    return {
        "count": float(count),
        "same_sign_rate": same_rate,
        "opposite_sign_rate": opposite_rate,
        "same_minus_opposite": same_rate - opposite_rate,
    }


def _explainability(rows: list[dict[str, Any]], *, x_key: str, y_key: str, flipped: bool = False) -> dict[str, float]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        x = _safe_float(row.get(x_key))
        y = _safe_float(row.get(y_key))
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        xs.append(-x if flipped else x)
        ys.append(y)
    if not xs:
        return {
            "spearman": float("nan"),
            "robust_slope": float("nan"),
            "r2_like": float("nan"),
            "count": 0.0,
        }
    return tempo_explainability(xs, ys)


def _speaker_from_row(row: dict[str, Any]) -> str:
    source = str(row.get("src_spk") or "").strip()
    if source:
        return source
    item_name = str(row.get("src_item_name") or "").strip()
    return item_name.split("_", 1)[0] if item_name else ""


def _subset(rows: list[dict[str, Any]], *, valid_only: bool = False) -> list[dict[str, Any]]:
    if not valid_only:
        return list(rows)
    return [row for row in rows if _safe_int(row.get("g_domain_valid")) > 0]


def _surface_summary(
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    sign_eps: float,
) -> dict[str, Any]:
    original = _explainability(rows, x_key=x_key, y_key=y_key, flipped=False)
    flipped = _explainability(rows, x_key=x_key, y_key=y_key, flipped=True)
    sign = _sign_stats(rows, x_key=x_key, y_key=y_key, eps=sign_eps)
    return {
        "x_key": x_key,
        "y_key": y_key,
        "count": len(rows),
        "sign_count": sign["count"],
        "same_sign_rate": sign["same_sign_rate"],
        "opposite_sign_rate": sign["opposite_sign_rate"],
        "same_minus_opposite": sign["same_minus_opposite"],
        "original_spearman": float(original["spearman"]),
        "original_robust_slope": float(original["robust_slope"]),
        "original_r2_like": float(original["r2_like"]),
        "flipped_spearman": float(flipped["spearman"]),
        "flipped_robust_slope": float(flipped["robust_slope"]),
        "flipped_r2_like": float(flipped["r2_like"]),
    }


def _summarize_rows(rows: list[dict[str, Any]], *, sign_eps: float) -> dict[str, Any]:
    return {
        "delta_g_to_total_signal": _surface_summary(
            rows,
            x_key="delta_g_ref_minus_src_utt",
            y_key="zbar_sp_star",
            sign_eps=sign_eps,
        ),
        "delta_g_to_coarse_residual": _surface_summary(
            rows,
            x_key="delta_g_ref_minus_src_utt",
            y_key="c_star",
            sign_eps=sign_eps,
        ),
        "delta_g_prefix_to_total_signal": _surface_summary(
            rows,
            x_key="delta_g_ref_minus_src_prefix",
            y_key="zbar_sp_star",
            sign_eps=sign_eps,
        ),
        "delta_g_prefix_to_coarse_residual": _surface_summary(
            rows,
            x_key="delta_g_ref_minus_src_prefix",
            y_key="c_star",
            sign_eps=sign_eps,
        ),
    }


def main() -> None:
    args = _parse_args()
    input_csv = Path(args.input_csv)
    rows = _load_rows(input_csv)
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(_safe_int(row.get("candidate_token")), _safe_int(row.get("drop_edge_runs_for_g")))].append(row)

    summary: list[dict[str, Any]] = []
    for key in sorted(grouped.keys()):
        candidate_token, drop_edge_runs = key
        group_rows = grouped[key]
        valid_rows = _subset(group_rows, valid_only=True)
        speaker_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in valid_rows:
            speaker_groups[_speaker_from_row(row)].append(row)
        speaker_summary = {
            speaker: _summarize_rows(speaker_rows, sign_eps=float(args.sign_eps))
            for speaker, speaker_rows in sorted(speaker_groups.items())
            if speaker_rows
        }
        summary.append(
            {
                "candidate_token": int(candidate_token),
                "drop_edge_runs_for_g": int(drop_edge_runs),
                "overall": _summarize_rows(group_rows, sign_eps=float(args.sign_eps)),
                "valid_only": _summarize_rows(valid_rows, sign_eps=float(args.sign_eps)),
                "speaker_valid_only": speaker_summary,
            }
        )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            {
                "input_csv": str(input_csv),
                "sign_eps": float(args.sign_eps),
                "summary": summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"[counterfactual-static-direction] rows={len(rows)}")
    print(f"[counterfactual-static-direction] wrote_json={output_json}")
    for row in summary:
        overall = row["overall"]
        valid = row["valid_only"]
        print(
            "[counterfactual-static-direction] "
            f"token={int(row['candidate_token'])} drop_edge={int(row['drop_edge_runs_for_g'])} "
            f"overall_signal_slope={overall['delta_g_to_total_signal']['original_robust_slope']:.4f} "
            f"overall_resid_slope={overall['delta_g_to_coarse_residual']['original_robust_slope']:.4f} "
            f"valid_signal_slope={valid['delta_g_to_total_signal']['original_robust_slope']:.4f} "
            f"valid_prefix_signal_slope={valid['delta_g_prefix_to_total_signal']['original_robust_slope']:.4f}"
        )


if __name__ == "__main__":
    main()
