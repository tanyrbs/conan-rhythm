from __future__ import annotations

import argparse
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
DEFAULT_OUTPUT_JSON = "tmp/gate1_counterfactual_probe/full_sweep/active_local_oracle_report.json"
DEFAULT_SIGN_EPS = 0.05


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit local delta_g->c_star directionality for all runtime-active counterfactual sources."
    )
    parser.add_argument("--runtime_report", default=DEFAULT_RUNTIME_REPORT, help="Runtime anti-control report JSON.")
    parser.add_argument("--output_json", default=DEFAULT_OUTPUT_JSON, help="Where to write the aggregated audit report.")
    parser.add_argument("--sign_eps", type=float, default=DEFAULT_SIGN_EPS, help="Minimum absolute magnitude for local sign-agreement stats.")
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


def _active_runtime_source(group: dict[str, Any]) -> bool:
    real_tempo_range = _safe_float(group.get("real_tempo_range"))
    max_control_gap = _safe_float(group.get("max_control_gap"))
    return bool(np.isfinite(real_tempo_range) and real_tempo_range >= 0.01 and np.isfinite(max_control_gap) and max_control_gap >= 0.01)


def _row_domain_valid(row: dict[str, Any]) -> bool:
    domain_valid = _safe_float(row.get("counterfactual_domain_valid", row.get("g_domain_valid")))
    if np.isfinite(domain_valid):
        return bool(domain_valid > 0.5)
    return bool(_safe_float(row.get("gate0_row_dropped")) <= 0.5)


def _same_sign_stats(delta_g: list[float], c_star: list[float], *, eps: float) -> dict[str, float]:
    same = 0
    opposite = 0
    count = 0
    for left, right in zip(delta_g, c_star):
        if not (np.isfinite(left) and np.isfinite(right)):
            continue
        if abs(left) <= float(eps) or abs(right) <= float(eps):
            continue
        count += 1
        if (left > 0.0) == (right > 0.0):
            same += 1
        else:
            opposite += 1
    return {
        "count": float(count),
        "same_sign_rate": float(same) / float(count) if count > 0 else float("nan"),
        "opposite_sign_rate": float(opposite) / float(count) if count > 0 else float("nan"),
    }


def _response_bucket(group: dict[str, Any]) -> str:
    prompt = bool(group.get("prompt_control_response"))
    delta = bool(group.get("delta_g_control_response"))
    anti = bool(group.get("anti_control_response"))
    if delta and not prompt and not anti:
        return "delta_only"
    if delta and prompt and anti:
        return "prompt_delta_anti"
    if delta and prompt:
        return "prompt_delta"
    if delta and anti:
        return "delta_anti"
    if prompt and anti:
        return "prompt_anti"
    if prompt:
        return "prompt_only"
    if anti:
        return "anti_only"
    return "none"


def _order_signature(real_rows: list[dict[str, Any]], values: list[float]) -> list[str]:
    indices = sorted(range(len(real_rows)), key=lambda idx: (values[idx], idx))
    return [str(real_rows[idx].get("ref_condition", "")) for idx in indices]


def _summarize_bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "prompt_pos_local_slope_count": 0,
            "neg_local_slope_count": 0,
            "pos_local_slope_count": 0,
            "prompt_pos_delta_neg_count": 0,
            "median_local_slope": float("nan"),
            "median_prompt_slope": float("nan"),
            "median_flipped_slope": float("nan"),
        }
    local = np.asarray([_safe_float(row["local_orig_slope"]) for row in rows], dtype=np.float32)
    prompt = np.asarray([_safe_float(row["prompt_orig_slope"]) for row in rows], dtype=np.float32)
    flip = np.asarray([_safe_float(row["local_flip_slope"]) for row in rows], dtype=np.float32)
    return {
        "count": len(rows),
        "prompt_pos_local_slope_count": int(np.sum(np.isfinite(prompt) & (prompt > 0.0))),
        "neg_local_slope_count": int(np.sum(np.isfinite(local) & (local < 0.0))),
        "pos_local_slope_count": int(np.sum(np.isfinite(local) & (local > 0.0))),
        "prompt_pos_delta_neg_count": int(np.sum((np.isfinite(prompt) & (prompt > 0.0)) & (np.isfinite(local) & (local < 0.0)))),
        "median_local_slope": float(np.nanmedian(local)),
        "median_prompt_slope": float(np.nanmedian(prompt)),
        "median_flipped_slope": float(np.nanmedian(flip)),
    }


def main() -> None:
    args = _parse_args()
    runtime_report = _load_json(Path(args.runtime_report))
    summaries: list[dict[str, Any]] = []
    for runtime_summary in runtime_report.get("summaries", []):
        if not isinstance(runtime_summary, dict):
            continue
        artifact_payload = _load_json(Path(str(runtime_summary["artifact"])))
        artifact_rows = artifact_payload.get("rows", [])
        active_rows: list[dict[str, Any]] = []
        for group in runtime_summary.get("groups", []):
            if not isinstance(group, dict) or not _active_runtime_source(group):
                continue
            source_name = str(group.get("source_name", ""))
            real_rows = [
                row
                for row in artifact_rows
                if str(row.get("source_name", "")) == source_name
                and str(row.get("ref_condition", "")) in {"slow", "mid", "fast"}
                and _row_domain_valid(row)
            ]
            if len(real_rows) < 3:
                continue
            delta_g = [_safe_float(row.get("delta_g")) for row in real_rows]
            prompt = [_safe_float(row.get("prompt_tempo_ref")) for row in real_rows]
            g_ref = [_safe_float(row.get("g_ref")) for row in real_rows]
            c_star = [_safe_float(row.get("c_star")) for row in real_rows]
            prompt_original = tempo_explainability(prompt, c_star)
            prompt_to_delta = tempo_explainability(prompt, delta_g)
            prompt_to_g_ref = tempo_explainability(prompt, g_ref)
            original = tempo_explainability(delta_g, c_star)
            flipped = tempo_explainability([-value for value in delta_g], c_star)
            sign_stats = _same_sign_stats(delta_g, c_star, eps=float(args.sign_eps))
            c_order = _order_signature(real_rows, c_star)
            prompt_order = _order_signature(real_rows, prompt)
            delta_order = _order_signature(real_rows, delta_g)
            g_ref_order = _order_signature(real_rows, g_ref)
            anti_order = _order_signature(real_rows, [-value for value in delta_g])
            active_rows.append(
                {
                    "source_name": source_name,
                    "speaker": str(group.get("speaker", "")),
                    "response_bucket": _response_bucket(group),
                    "prompt_control_response": bool(group.get("prompt_control_response")),
                    "delta_g_control_response": bool(group.get("delta_g_control_response")),
                    "anti_control_response": bool(group.get("anti_control_response")),
                    "real_tempo_range": _safe_float(group.get("real_tempo_range")),
                    "max_control_gap": _safe_float(group.get("max_control_gap")),
                    "prompt_orig_slope": _safe_float(prompt_original.get("robust_slope")),
                    "prompt_orig_spearman": _safe_float(prompt_original.get("spearman")),
                    "prompt_to_delta_slope": _safe_float(prompt_to_delta.get("robust_slope")),
                    "prompt_to_delta_spearman": _safe_float(prompt_to_delta.get("spearman")),
                    "prompt_to_g_ref_slope": _safe_float(prompt_to_g_ref.get("robust_slope")),
                    "prompt_to_g_ref_spearman": _safe_float(prompt_to_g_ref.get("spearman")),
                    "local_orig_slope": _safe_float(original.get("robust_slope")),
                    "local_flip_slope": _safe_float(flipped.get("robust_slope")),
                    "local_orig_spearman": _safe_float(original.get("spearman")),
                    "local_flip_spearman": _safe_float(flipped.get("spearman")),
                    "same_sign_rate": sign_stats["same_sign_rate"],
                    "opposite_sign_rate": sign_stats["opposite_sign_rate"],
                    "sign_count": sign_stats["count"],
                    "c_order": c_order,
                    "prompt_order": prompt_order,
                    "delta_order": delta_order,
                    "g_ref_order": g_ref_order,
                    "anti_order": anti_order,
                    "prompt_exact_order_match": bool(prompt_order == c_order),
                    "delta_exact_order_match": bool(delta_order == c_order),
                    "prompt_delta_order_match": bool(prompt_order == delta_order),
                    "prompt_g_ref_order_match": bool(prompt_order == g_ref_order),
                    "anti_exact_order_match": bool(anti_order == c_order),
                    "prompt_pos_delta_neg": bool(
                        np.isfinite(_safe_float(prompt_original.get("robust_slope")))
                        and _safe_float(prompt_original.get("robust_slope")) > 0.0
                        and np.isfinite(_safe_float(original.get("robust_slope")))
                        and _safe_float(original.get("robust_slope")) < 0.0
                    ),
                }
            )

        local_orig = np.asarray([_safe_float(row["local_orig_slope"]) for row in active_rows], dtype=np.float32)
        prompt_orig = np.asarray([_safe_float(row["prompt_orig_slope"]) for row in active_rows], dtype=np.float32)
        prompt_to_delta = np.asarray([_safe_float(row["prompt_to_delta_slope"]) for row in active_rows], dtype=np.float32)
        prompt_to_g_ref = np.asarray([_safe_float(row["prompt_to_g_ref_slope"]) for row in active_rows], dtype=np.float32)
        local_flip = np.asarray([_safe_float(row["local_flip_slope"]) for row in active_rows], dtype=np.float32)
        same_sign = np.asarray([_safe_float(row["same_sign_rate"]) for row in active_rows], dtype=np.float32)
        buckets: dict[str, list[dict[str, Any]]] = {}
        for row in active_rows:
            buckets.setdefault(str(row["response_bucket"]), []).append(row)
        summaries.append(
            {
                "candidate_token": int(runtime_summary["candidate_token"]),
                "artifact": runtime_summary["artifact"],
                "active_source_count": len(active_rows),
                "prompt_pos_local_slope_count": int(np.sum(np.isfinite(prompt_orig) & (prompt_orig > 0.0))),
                "prompt_neg_local_slope_count": int(np.sum(np.isfinite(prompt_orig) & (prompt_orig < 0.0))),
                "prompt_to_delta_pos_count": int(np.sum(np.isfinite(prompt_to_delta) & (prompt_to_delta > 0.0))),
                "prompt_to_delta_neg_count": int(np.sum(np.isfinite(prompt_to_delta) & (prompt_to_delta < 0.0))),
                "prompt_to_g_ref_pos_count": int(np.sum(np.isfinite(prompt_to_g_ref) & (prompt_to_g_ref > 0.0))),
                "prompt_to_g_ref_neg_count": int(np.sum(np.isfinite(prompt_to_g_ref) & (prompt_to_g_ref < 0.0))),
                "neg_local_slope_count": int(np.sum(np.isfinite(local_orig) & (local_orig < 0.0))),
                "pos_local_slope_count": int(np.sum(np.isfinite(local_orig) & (local_orig > 0.0))),
                "prompt_pos_delta_neg_count": int(
                    np.sum((np.isfinite(prompt_orig) & (prompt_orig > 0.0)) & (np.isfinite(local_orig) & (local_orig < 0.0)))
                ),
                "prompt_exact_order_match_count": int(np.sum([bool(row["prompt_exact_order_match"]) for row in active_rows])),
                "delta_exact_order_match_count": int(np.sum([bool(row["delta_exact_order_match"]) for row in active_rows])),
                "prompt_delta_order_match_count": int(np.sum([bool(row["prompt_delta_order_match"]) for row in active_rows])),
                "prompt_g_ref_order_match_count": int(np.sum([bool(row["prompt_g_ref_order_match"]) for row in active_rows])),
                "anti_exact_order_match_count": int(np.sum([bool(row["anti_exact_order_match"]) for row in active_rows])),
                "median_prompt_slope": float(np.nanmedian(prompt_orig)) if active_rows else float("nan"),
                "median_prompt_to_delta_slope": float(np.nanmedian(prompt_to_delta)) if active_rows else float("nan"),
                "median_prompt_to_g_ref_slope": float(np.nanmedian(prompt_to_g_ref)) if active_rows else float("nan"),
                "median_local_slope": float(np.nanmedian(local_orig)) if active_rows else float("nan"),
                "median_flipped_slope": float(np.nanmedian(local_flip)) if active_rows else float("nan"),
                "flip_positive_count": int(np.sum(np.isfinite(local_flip) & (local_flip > 0.0))),
                "same_sign_majority_count": int(np.sum(np.isfinite(same_sign) & (same_sign >= 0.5))),
                "bucket_summary": {
                    bucket: _summarize_bucket(rows)
                    for bucket, rows in sorted(buckets.items())
                },
                "active_rows": active_rows,
            }
        )

    payload = {
        "runtime_report": args.runtime_report,
        "sign_eps": float(args.sign_eps),
        "summaries": summaries,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[active-local-oracle] wrote_json={output_json}")
    for summary in summaries:
        print(
            "[active-local-oracle] "
            f"token={summary['candidate_token']} active={summary['active_source_count']} "
            f"prompt_pos={summary['prompt_pos_local_slope_count']} prompt_neg={summary['prompt_neg_local_slope_count']} "
            f"prompt_to_delta_pos={summary['prompt_to_delta_pos_count']} prompt_to_delta_neg={summary['prompt_to_delta_neg_count']} "
            f"neg_local={summary['neg_local_slope_count']} pos_local={summary['pos_local_slope_count']} "
            f"prompt_pos_delta_neg={summary['prompt_pos_delta_neg_count']} "
            f"prompt_exact={summary['prompt_exact_order_match_count']} delta_exact={summary['delta_exact_order_match_count']} prompt_delta_exact={summary['prompt_delta_order_match_count']} anti_exact={summary['anti_exact_order_match_count']} "
            f"median_local={summary['median_local_slope']:.4f} median_flip={summary['median_flipped_slope']:.4f} "
            f"same_sign_majority={summary['same_sign_majority_count']}"
        )


if __name__ == "__main__":
    main()
