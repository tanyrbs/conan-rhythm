from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.commons.single_thread_env import apply_single_thread_env

apply_single_thread_env()

import numpy as np


DEFAULT_METADATA_PATH = "C:/project/00-2 project/data/processed/vc/metadata.json"
DEFAULT_CANDIDATES = "71,72,63,12,4,57"
DEFAULT_OUTPUT_JSON = "tmp/gate1_boundary_audit/candidate_token_structure_report.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether candidate HuBERT tokens behave like pause cues or boundary artifacts."
    )
    parser.add_argument("--metadata_path", default=DEFAULT_METADATA_PATH, help="Path to processed metadata.json.")
    parser.add_argument("--candidate_tokens", default=DEFAULT_CANDIDATES, help="Comma-separated token ids to audit.")
    parser.add_argument("--output_json", default=DEFAULT_OUTPUT_JSON, help="Where to write the audit report.")
    return parser.parse_args()


def _parse_tokens(text: str) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        token = int(part)
        if token in seen:
            continue
        seen.add(token)
        values.append(token)
    if not values:
        raise ValueError("candidate_tokens must contain at least one token id.")
    return values


def _tokenize_hubert(value: Any) -> list[int]:
    if isinstance(value, str):
        return [int(float(token)) for token in value.split() if str(token).strip()]
    arr = np.asarray(value).reshape(-1)
    return [int(token) for token in arr.tolist()]


def _run_encode(tokens: list[int]) -> list[tuple[int, int, int, int]]:
    runs: list[tuple[int, int, int, int]] = []
    if not tokens:
        return runs
    start = 0
    current = tokens[0]
    for idx in range(1, len(tokens)):
        if tokens[idx] != current:
            runs.append((current, start, idx - 1, idx - start))
            current = tokens[idx]
            start = idx
    runs.append((current, start, len(tokens) - 1, len(tokens) - start))
    return runs


def _safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def _safe_percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))


def _speaker_breakdown(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    by_speaker: dict[str, dict[str, int]] = {}
    for row in rows:
        speaker = str(row["speaker"])
        slot = by_speaker.setdefault(
            speaker,
            {
                "items": 0,
                "items_with_token": 0,
                "items_start": 0,
                "items_end": 0,
                "items_internal": 0,
            },
        )
        slot["items"] += 1
        slot["items_with_token"] += int(row["has_token"])
        slot["items_start"] += int(row["has_start"])
        slot["items_end"] += int(row["has_end"])
        slot["items_internal"] += int(row["has_internal"])
    return by_speaker


def _prompt_breakdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_prompt[str(row["prompt_id"])].append(row)
    summary_rows: list[dict[str, Any]] = []
    for prompt_id, prompt_rows in sorted(by_prompt.items()):
        speakers = sorted({str(row["speaker"]) for row in prompt_rows})
        last_tokens = sorted({int(row["last_token"]) for row in prompt_rows})
        summary_rows.append(
            {
                "prompt_id": prompt_id,
                "speaker_count": len(speakers),
                "speakers": speakers,
                "token_start_speakers": sum(int(row["has_start"]) for row in prompt_rows),
                "token_end_speakers": sum(int(row["has_end"]) for row in prompt_rows),
                "token_internal_speakers": sum(int(row["has_internal"]) for row in prompt_rows),
                "first_tokens": sorted({int(row["first_token"]) for row in prompt_rows}),
                "last_tokens": last_tokens,
                "last_token_uniform": len(last_tokens) == 1,
            }
        )
    return {
        "prompt_count": len(summary_rows),
        "all_start_prompts": sum(1 for row in summary_rows if int(row["token_start_speakers"]) == int(row["speaker_count"])),
        "all_end_prompts": sum(1 for row in summary_rows if int(row["token_end_speakers"]) == int(row["speaker_count"])),
        "mixed_end_prompts": sum(1 for row in summary_rows if 0 < int(row["token_end_speakers"]) < int(row["speaker_count"])),
        "all_internal_prompts": sum(1 for row in summary_rows if int(row["token_internal_speakers"]) == int(row["speaker_count"])),
        "rows": summary_rows,
    }


def _audit_candidate(items: list[dict[str, Any]], candidate_token: int) -> dict[str, Any]:
    item_rows: list[dict[str, Any]] = []
    occurrence_pos: list[float] = []
    run_pos: list[float] = []
    run_len_all: list[float] = []
    run_len_start: list[float] = []
    run_len_end: list[float] = []
    run_len_internal: list[float] = []
    occurrence_internal = 0
    occurrence_start = 0
    occurrence_end = 0
    run_start = 0
    run_end = 0
    run_internal = 0
    for item in items:
        tokens = _tokenize_hubert(item["hubert"])
        runs = _run_encode(tokens)
        n = len(tokens)
        norm = max(1, n - 1)
        candidate_occ_indices = [idx for idx, token in enumerate(tokens) if token == int(candidate_token)]
        has_token = bool(candidate_occ_indices)
        has_start = bool(tokens and tokens[0] == int(candidate_token))
        has_end = bool(tokens and tokens[-1] == int(candidate_token))
        has_internal = any(0 < idx < (n - 1) for idx in candidate_occ_indices)
        for idx in candidate_occ_indices:
            pos = float(idx) / float(norm)
            occurrence_pos.append(pos)
            if idx == 0:
                occurrence_start += 1
            elif idx == n - 1:
                occurrence_end += 1
            else:
                occurrence_internal += 1
        for run_idx, (token, start, end, length) in enumerate(runs):
            if token != int(candidate_token):
                continue
            pos = float(run_idx) / float(max(1, len(runs) - 1))
            run_pos.append(pos)
            run_len_all.append(float(length))
            if run_idx == 0:
                run_start += 1
                run_len_start.append(float(length))
            elif run_idx == len(runs) - 1:
                run_end += 1
                run_len_end.append(float(length))
            else:
                run_internal += 1
                run_len_internal.append(float(length))
        item_rows.append(
            {
                "item_name": str(item["item_name"]),
                "prompt_id": str(item.get("prompt_id", item["item_name"])),
                "speaker": str(item.get("speaker", str(item["item_name"]).split("_", 1)[0])),
                "split": str(item.get("split", "")),
                "first_token": int(tokens[0]) if tokens else -1,
                "last_token": int(tokens[-1]) if tokens else -1,
                "has_token": int(has_token),
                "has_start": int(has_start),
                "has_end": int(has_end),
                "has_internal": int(has_internal),
                "token_count": len(candidate_occ_indices),
                "start_run_length": float(run_len_start[-1]) if has_start and run_len_start else float("nan"),
                "end_run_length": float(run_len_end[-1]) if has_end and run_len_end else float("nan"),
            }
        )
    occurrence_total = occurrence_start + occurrence_end + occurrence_internal
    run_total = run_start + run_end + run_internal
    return {
        "candidate_token": int(candidate_token),
        "item_count": len(items),
        "items_with_token": sum(int(row["has_token"]) for row in item_rows),
        "items_start": sum(int(row["has_start"]) for row in item_rows),
        "items_end": sum(int(row["has_end"]) for row in item_rows),
        "items_internal": sum(int(row["has_internal"]) for row in item_rows),
        "occurrence_count": occurrence_total,
        "occurrence_start": occurrence_start,
        "occurrence_end": occurrence_end,
        "occurrence_internal": occurrence_internal,
        "occurrence_edge_fraction": float((occurrence_start + occurrence_end) / max(1, occurrence_total)),
        "run_count": run_total,
        "run_start": run_start,
        "run_end": run_end,
        "run_internal": run_internal,
        "run_edge_fraction": float((run_start + run_end) / max(1, run_total)),
        "run_len_mean": _safe_mean(run_len_all),
        "run_len_start_mean": _safe_mean(run_len_start),
        "run_len_end_mean": _safe_mean(run_len_end),
        "run_len_internal_mean": _safe_mean(run_len_internal),
        "occurrence_pos_p05": _safe_percentile(occurrence_pos, 5),
        "occurrence_pos_p50": _safe_percentile(occurrence_pos, 50),
        "occurrence_pos_p95": _safe_percentile(occurrence_pos, 95),
        "run_pos_p05": _safe_percentile(run_pos, 5),
        "run_pos_p50": _safe_percentile(run_pos, 50),
        "run_pos_p95": _safe_percentile(run_pos, 95),
        "speaker_slices": _speaker_breakdown(item_rows),
        "prompt_slices": _prompt_breakdown(item_rows),
        "item_rows": item_rows,
    }


def main() -> None:
    args = _parse_args()
    metadata_path = Path(args.metadata_path)
    items = json.loads(metadata_path.read_text(encoding="utf-8"))
    candidates = _parse_tokens(args.candidate_tokens)
    report = {
        "metadata_path": str(metadata_path),
        "candidate_tokens": candidates,
        "item_count": len(items),
        "candidates": [_audit_candidate(items, token) for token in candidates],
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[candidate-structure] metadata={metadata_path}")
    print(f"[candidate-structure] candidates={candidates}")
    print(f"[candidate-structure] wrote_json={output_path}")
    for candidate in report["candidates"]:
        print(
            f"[candidate-structure] token={candidate['candidate_token']} "
            f"items_with={candidate['items_with_token']}/{candidate['item_count']} "
            f"items_start={candidate['items_start']} items_end={candidate['items_end']} "
            f"items_internal={candidate['items_internal']} run_edge_fraction={candidate['run_edge_fraction']:.3f} "
            f"run_len_start_mean={candidate['run_len_start_mean']:.3f} run_len_end_mean={candidate['run_len_end_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
