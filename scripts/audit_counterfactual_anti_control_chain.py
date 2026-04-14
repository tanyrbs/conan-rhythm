from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT_JSON = "tmp/gate1_counterfactual_probe/full_sweep/active_local_oracle_report.json"
DEFAULT_OUTPUT_JSON = "tmp/gate1_counterfactual_probe/full_sweep/anti_control_chain_report.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize coherent anti-control vs maintained-control chains from active local oracle audit."
    )
    parser.add_argument("--input_json", default=DEFAULT_INPUT_JSON, help="Active local oracle audit JSON.")
    parser.add_argument("--output_json", default=DEFAULT_OUTPUT_JSON, help="Where to write the chain-coherence report.")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict JSON payload: {path}")
    return payload


def _bool(value: Any) -> bool:
    return bool(value)


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def main() -> None:
    args = _parse_args()
    payload = _load_json(Path(args.input_json))
    summaries: list[dict[str, Any]] = []
    for summary in payload.get("summaries", []):
        if not isinstance(summary, dict):
            continue
        active_rows = list(summary.get("active_rows", []))
        anti_chain_rows: list[str] = []
        anti_chain_strict_rows: list[str] = []
        weak_anti_rows: list[str] = []
        maintained_chain_rows: list[str] = []
        maintained_chain_strict_rows: list[str] = []
        for row in active_rows:
            if not isinstance(row, dict):
                continue
            source_name = str(row.get("source_name", ""))
            prompt_pos = _float(row.get("prompt_orig_slope")) > 0.0
            prompt_to_delta_neg = _float(row.get("prompt_to_delta_slope")) < 0.0
            prompt_to_delta_pos = _float(row.get("prompt_to_delta_slope")) > 0.0
            delta_pos = _float(row.get("local_orig_slope")) > 0.0
            anti_pos = _float(row.get("local_flip_slope")) > 0.0
            prompt_exact = _bool(row.get("prompt_exact_order_match"))
            delta_exact = _bool(row.get("delta_exact_order_match"))
            anti_exact = _bool(row.get("anti_exact_order_match"))

            if prompt_to_delta_neg and anti_pos:
                weak_anti_rows.append(source_name)
            if prompt_pos and prompt_to_delta_neg and anti_pos:
                anti_chain_rows.append(source_name)
            if prompt_pos and prompt_to_delta_neg and anti_pos and prompt_exact and anti_exact:
                anti_chain_strict_rows.append(source_name)
            if prompt_pos and prompt_to_delta_pos and delta_pos:
                maintained_chain_rows.append(source_name)
            if prompt_pos and prompt_to_delta_pos and delta_pos and prompt_exact and delta_exact:
                maintained_chain_strict_rows.append(source_name)

        active_source_count = int(summary.get("active_source_count", len(active_rows)) or len(active_rows))
        out = {
            "candidate_token": int(summary.get("candidate_token", 0) or 0),
            "active_source_count": active_source_count,
            "weak_anti_chain_count": len(weak_anti_rows),
            "anti_chain_count": len(anti_chain_rows),
            "anti_chain_strict_count": len(anti_chain_strict_rows),
            "maintained_chain_count": len(maintained_chain_rows),
            "maintained_chain_strict_count": len(maintained_chain_strict_rows),
            "weak_anti_chain_sources": weak_anti_rows,
            "anti_chain_sources": anti_chain_rows,
            "anti_chain_strict_sources": anti_chain_strict_rows,
            "maintained_chain_sources": maintained_chain_rows,
            "maintained_chain_strict_sources": maintained_chain_strict_rows,
        }
        summaries.append(out)

    report = {
        "input_json": args.input_json,
        "summaries": summaries,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[anti-control-chain] wrote_json={output_json}")
    for summary in summaries:
        print(
            "[anti-control-chain] "
            f"token={summary['candidate_token']} active={summary['active_source_count']} "
            f"weak_anti={summary['weak_anti_chain_count']} anti={summary['anti_chain_count']} "
            f"anti_strict={summary['anti_chain_strict_count']} "
            f"maintained={summary['maintained_chain_count']} maintained_strict={summary['maintained_chain_strict_count']}"
        )


if __name__ == "__main__":
    main()
