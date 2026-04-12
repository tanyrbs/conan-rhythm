#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.plot.rhythm_v3_viz import load_debug_records, record_summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize exported rhythm_v3 debug-record bundles into CSV/JSON-friendly tables.",
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
        )
        for record in records
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if pd is not None:
        pd.DataFrame(rows).to_csv(output, index=False)
    else:  # pragma: no cover
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    print(f"[rhythm_v3_debug_records] wrote {len(rows)} rows -> {output}")


if __name__ == "__main__":
    main()
