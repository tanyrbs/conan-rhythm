#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path


VALIDATION_PATTERN = re.compile(r"Validation results@(\d+):\s*(\{.*\})")

S_KEYS = [
    "total_loss",
    "L_exec_pause",
    "L_prefix_state",
    "rhythm_metric_prefix_drift_l1",
    "rhythm_metric_pause_event_f1",
    "rhythm_metric_budget_projection_repair_ratio_mean",
    "L_base",
    "L_pitch",
    "rhythm_metric_module_only_objective",
    "rhythm_metric_skip_acoustic_objective",
    "rhythm_metric_disable_acoustic_train_path",
]

A_KEYS = [
    "L_exec_speech",
    "L_exec_pause_value",
    "L_pause_event",
    "L_pause_support",
    "L_budget",
    "L_stream_state",
    "L_rhythm_exec",
    "rhythm_metric_exec_total_corr",
    "rhythm_metric_exec_pause_l1",
    "rhythm_metric_exec_speech_l1",
    "rhythm_metric_pause_event_precision",
    "rhythm_metric_pause_event_recall",
    "rhythm_metric_budget_violation_mean",
    "rhythm_metric_phase_nonretro_rate",
]

B_KEYS = [
    "rhythm_metric_local_rate_transfer_corr",
    "rhythm_metric_boundary_trace_corr",
    "rhythm_metric_pause_share_mean",
    "rhythm_metric_expand_ratio_mean",
    "rhythm_metric_budget_projection_repair_active_rate",
]


def parse_validations(log_path: Path):
    rows = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = VALIDATION_PATTERN.search(line)
            if not match:
                continue
            step = int(match.group(1))
            payload = ast.literal_eval(match.group(2))
            rows.append((step, payload))
    return rows


def _fmt(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _print_group(name: str, payload: dict, keys: list[str]) -> None:
    print(f"[{name}]")
    for key in keys:
        print(f"  {key:50s} {_fmt(payload.get(key))}")


def _delta_summary(current: dict, previous: dict, keys: list[str]) -> None:
    print("[delta vs prev val]")
    for key in keys:
        cur = current.get(key)
        prev = previous.get(key)
        if not isinstance(cur, (int, float)) or not isinstance(prev, (int, float)):
            continue
        delta = float(cur) - float(prev)
        print(f"  {key:50s} {delta:+.4f}")


def main():
    parser = argparse.ArgumentParser(description="Summarize stage-1 validation metrics from a training log.")
    parser.add_argument("--log", required=True, help="Training log path.")
    parser.add_argument("--tail", type=int, default=5, help="How many latest validation windows to show.")
    args = parser.parse_args()

    log_path = Path(args.log)
    rows = parse_validations(log_path)
    if not rows:
        raise SystemExit(f"No validation rows found in {log_path}")

    print(f"log={log_path}")
    print(f"num_validations={len(rows)}")
    print("")

    tail_rows = rows[-max(1, args.tail):]
    for step, payload in tail_rows:
        print(f"===== step {step} =====")
        _print_group("S", payload, S_KEYS)
        _print_group("A", payload, A_KEYS)
        _print_group("B", payload, B_KEYS)
        print("")

    if len(rows) >= 2:
        prev_step, prev = rows[-2]
        cur_step, cur = rows[-1]
        print(f"===== latest delta: {prev_step} -> {cur_step} =====")
        _delta_summary(cur, prev, S_KEYS + A_KEYS + B_KEYS)


if __name__ == "__main__":
    main()
