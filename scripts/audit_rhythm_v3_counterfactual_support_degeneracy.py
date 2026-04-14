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

from utils.commons.single_thread_env import apply_single_thread_env

apply_single_thread_env()

import numpy as np
import torch

from modules.Conan.rhythm_v3.g_stats import summarize_global_rate_support
from modules.Conan.rhythm_v3.source_cache import build_source_rhythm_cache_v3
from tasks.Conan.dataset import ConanDataset
from utils.commons.hparams import set_hparams


DEFAULT_CONFIG = "egs/local_arctic_rhythm_v3_quick.yaml"
DEFAULT_SPLITS = "train,valid,test"
DEFAULT_CANDIDATES = "71,72,63"
DEFAULT_DROP_EDGES = "1,3"
DEFAULT_OUTPUT_CSV = "tmp/gate1_boundary_audit/counterfactual_support_degeneracy_rows.csv"
DEFAULT_OUTPUT_JSON = "tmp/gate1_boundary_audit/counterfactual_support_degeneracy_report.json"
DEFAULT_RELAXED_HPARAMS = (
    "use_pitch_embed=False,"
    "rhythm_v3_gate_quality_strict=False,"
    "rhythm_v3_minimal_v1_profile=False,"
    "rhythm_v3_strict_minimal_claim_profile=False"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit full-split support degeneracy for counterfactual prompt-side silent-token candidates."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Config YAML.")
    parser.add_argument("--splits", default=DEFAULT_SPLITS, help="Comma-separated dataset splits.")
    parser.add_argument("--candidate_tokens", default=DEFAULT_CANDIDATES, help="Comma-separated candidate token ids.")
    parser.add_argument("--drop_edge_runs", default=DEFAULT_DROP_EDGES, help="Comma-separated drop_edge_runs_for_g values.")
    parser.add_argument("--binary_data_dir", default="", help="Optional binary_data_dir override.")
    parser.add_argument("--processed_data_dir", default="", help="Optional processed_data_dir override.")
    parser.add_argument("--hparams", default="", help="Extra hparam overrides appended after relaxed audit defaults.")
    parser.add_argument("--output_csv", default=DEFAULT_OUTPUT_CSV, help="Where to write flattened rows.")
    parser.add_argument("--output_json", default=DEFAULT_OUTPUT_JSON, help="Where to write the structured report.")
    return parser.parse_args()


def _compose_hparams_override(args: argparse.Namespace) -> str:
    parts = [DEFAULT_RELAXED_HPARAMS]
    if args.binary_data_dir:
        parts.append(f"binary_data_dir='{args.binary_data_dir}'")
    if args.processed_data_dir:
        parts.append(f"processed_data_dir='{args.processed_data_dir}'")
    if args.hparams:
        parts.append(args.hparams)
    return ",".join(part for part in parts if part)


def _parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _parse_splits(text: str) -> list[str]:
    splits = [part.strip() for part in str(text).split(",") if part.strip()]
    if not splits:
        raise ValueError("Expected at least one split.")
    return splits


def _scalar_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    arr = np.asarray(value)
    if arr.size <= 0:
        return float("nan")
    try:
        return float(arr.reshape(-1)[0])
    except (TypeError, ValueError):
        return float("nan")


def _safe_item_tokens(value: Any) -> list[int]:
    if isinstance(value, str):
        return [int(float(token)) for token in value.split() if str(token).strip()]
    arr = np.asarray(value).reshape(-1)
    return [int(token) for token in arr.tolist()]


def _min_edge_distance(indices: np.ndarray, prompt_units: int) -> float:
    if prompt_units <= 0 or indices.size <= 0:
        return float("nan")
    distances = np.minimum(indices, (prompt_units - 1) - indices).astype(np.float32)
    return float(distances.min()) if distances.size > 0 else float("nan")


def _mean_edge_distance(indices: np.ndarray, prompt_units: int) -> float:
    if prompt_units <= 0 or indices.size <= 0:
        return float("nan")
    distances = np.minimum(indices, (prompt_units - 1) - indices).astype(np.float32)
    return float(distances.mean()) if distances.size > 0 else float("nan")


def _support_span(indices: np.ndarray) -> float:
    if indices.size <= 0:
        return float("nan")
    return float(indices.max() - indices.min())


def _is_valid_ref_len(value: float, *, min_ref_len_sec: float, max_ref_len_sec: float) -> bool:
    return bool(np.isfinite(value) and value >= min_ref_len_sec and value <= max_ref_len_sec)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _materialize_raw_items(ds: ConanDataset) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for local_idx in range(len(ds.avail_idxs)):
        raw_item = ds._get_raw_item_cached(local_idx)
        if not isinstance(raw_item, dict):
            continue
        items.append(raw_item)
    return items


def _audit_item(
    *,
    ds: ConanDataset,
    split: str,
    raw_item: dict[str, Any],
    candidate_token: int,
    drop_edge_runs: int,
    min_boundary_confidence: float,
    min_prompt_speech_ratio: float,
    min_prompt_ref_len_sec: float,
    max_prompt_ref_len_sec: float,
) -> dict[str, Any]:
    item_name = str(raw_item.get("item_name", "<item>"))
    hubert = np.asarray(raw_item["hubert"])
    cache = build_source_rhythm_cache_v3(
        hubert,
        silent_token=int(candidate_token),
        separator_aware=bool(ds.hparams.get("rhythm_separator_aware", True)),
        tail_open_units=int(ds.hparams.get("rhythm_tail_open_units", 1)),
        emit_silence_runs=False,
        debounce_min_run_frames=int(ds.hparams.get("rhythm_v3_debounce_min_run_frames", 2)),
        phrase_boundary_threshold=float(ds.hparams.get("rhythm_source_phrase_threshold", 0.55)),
        unit_prior_path=None,
    )
    prompt_item = dict(raw_item)
    prompt_item.update(cache)
    conditioning = ds._build_reference_prompt_unit_conditioning(
        prompt_item,
        target_mode=ds._resolve_rhythm_target_mode(),
    )

    prompt_duration_obs = np.asarray(conditioning["prompt_duration_obs"], dtype=np.float32).reshape(-1)
    prompt_valid_mask = np.asarray(
        conditioning.get("prompt_valid_mask", conditioning.get("prompt_unit_mask")),
        dtype=np.float32,
    ).reshape(-1)
    prompt_speech_mask = np.asarray(conditioning["prompt_speech_mask"], dtype=np.float32).reshape(-1)
    prompt_closed_mask = np.asarray(conditioning["prompt_closed_mask"], dtype=np.float32).reshape(-1)
    prompt_boundary_confidence = np.asarray(
        conditioning["prompt_boundary_confidence"],
        dtype=np.float32,
    ).reshape(-1)

    prompt_ref_len_sec = _scalar_or_nan(conditioning.get("prompt_ref_len_sec"))
    if not np.isfinite(prompt_ref_len_sec):
        prompt_ref_len_sec = _scalar_or_nan(raw_item.get("duration"))
    prompt_speech_ratio = _scalar_or_nan(conditioning.get("prompt_speech_ratio_scalar"))

    stats = summarize_global_rate_support(
        speech_mask=torch.as_tensor(prompt_speech_mask, dtype=torch.float32),
        valid_mask=torch.as_tensor(prompt_valid_mask, dtype=torch.float32),
        duration_obs=torch.as_tensor(prompt_duration_obs, dtype=torch.float32),
        drop_edge_runs=int(drop_edge_runs),
        min_speech_ratio=float(min_prompt_speech_ratio),
        closed_mask=torch.as_tensor(prompt_closed_mask, dtype=torch.float32),
        boundary_confidence=torch.as_tensor(prompt_boundary_confidence, dtype=torch.float32),
        min_boundary_confidence=float(min_boundary_confidence),
    )
    support_mask = stats.support_mask.reshape(-1).detach().cpu().numpy().astype(bool)
    support_indices = np.flatnonzero(support_mask).astype(np.int64)
    prompt_units = int(prompt_duration_obs.shape[0])
    support_count = float(stats.support_count.reshape(-1)[0].item())
    support_seed_count = float(stats.support_seed_count.reshape(-1)[0].item())
    clean_count = float(stats.clean_count.reshape(-1)[0].item())
    support_domain_valid = float(stats.domain_valid.reshape(-1)[0].item())
    support_fraction = float(stats.support_fraction.reshape(-1)[0].item())
    edge_runs_dropped = float(stats.edge_runs_dropped.reshape(-1)[0].item())

    speech_ratio_valid = bool(
        np.isfinite(prompt_speech_ratio)
        and prompt_speech_ratio >= (float(min_prompt_speech_ratio) - 1.0e-6)
    )
    ref_len_valid = _is_valid_ref_len(
        prompt_ref_len_sec,
        min_ref_len_sec=float(min_prompt_ref_len_sec),
        max_ref_len_sec=float(max_prompt_ref_len_sec),
    )
    g_valid_support = bool(support_count > 0.5)
    g_domain_valid = bool(
        g_valid_support
        and clean_count > 0.5
        and support_domain_valid > 0.5
        and speech_ratio_valid
        and ref_len_valid
    )

    raw_candidate_count = sum(1 for token in _safe_item_tokens(raw_item["hubert"]) if token == int(candidate_token))
    return {
        "split": split,
        "item_name": item_name,
        "speaker": str(raw_item.get("speaker", item_name.split("_", 1)[0])),
        "prompt_id": str(raw_item.get("prompt_id", item_name)),
        "candidate_token": int(candidate_token),
        "drop_edge_runs_for_g": int(drop_edge_runs),
        "prompt_units": prompt_units,
        "raw_candidate_count": int(raw_candidate_count),
        "sep_nonzero_count": int((np.asarray(cache["sep_hint"]).reshape(-1) > 0.5).sum()),
        "support_seed_count": support_seed_count,
        "support_count": support_count,
        "clean_count": clean_count,
        "support_fraction": support_fraction,
        "edge_runs_dropped": edge_runs_dropped,
        "singleton_seed": int(support_seed_count <= 1.0 + 1.0e-6),
        "singleton_support": int(support_count <= 1.0 + 1.0e-6),
        "multi_support": int(support_count >= 2.0 - 1.0e-6),
        "support_min_edge_distance": _min_edge_distance(support_indices, prompt_units),
        "support_mean_edge_distance": _mean_edge_distance(support_indices, prompt_units),
        "support_span": _support_span(support_indices),
        "support_indices": json.dumps(support_indices.tolist(), ensure_ascii=True),
        "prompt_ref_len_sec": float(prompt_ref_len_sec),
        "prompt_speech_ratio": float(prompt_speech_ratio),
        "speech_ratio_valid": int(speech_ratio_valid),
        "ref_len_valid": int(ref_len_valid),
        "support_domain_valid": support_domain_valid,
        "g_domain_valid": int(g_domain_valid),
    }


def _mean(values: list[float]) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float32)
    if finite.size <= 0:
        return float("nan")
    return float(finite.mean())


def _summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "candidate_token": int(rows[0]["candidate_token"]),
        "drop_edge_runs_for_g": int(rows[0]["drop_edge_runs_for_g"]),
        "item_count": len(rows),
        "raw_candidate_items": int(sum(1 for row in rows if int(row["raw_candidate_count"]) > 0)),
        "sep_nonzero_items": int(sum(1 for row in rows if int(row["sep_nonzero_count"]) > 0)),
        "support_positive_items": int(sum(1 for row in rows if float(row["support_count"]) > 0.5)),
        "clean_positive_items": int(sum(1 for row in rows if float(row["clean_count"]) > 0.5)),
        "singleton_seed_items": int(sum(1 for row in rows if int(row["singleton_seed"]) > 0)),
        "singleton_support_items": int(sum(1 for row in rows if int(row["singleton_support"]) > 0)),
        "multi_support_items": int(sum(1 for row in rows if int(row["multi_support"]) > 0)),
        "g_domain_valid_items": int(sum(1 for row in rows if int(row["g_domain_valid"]) > 0)),
        "mean_support_seed_count": _mean([float(row["support_seed_count"]) for row in rows]),
        "mean_support_count": _mean([float(row["support_count"]) for row in rows]),
        "mean_support_fraction": _mean([float(row["support_fraction"]) for row in rows]),
        "mean_edge_runs_dropped": _mean([float(row["edge_runs_dropped"]) for row in rows]),
        "mean_support_min_edge_distance": _mean([float(row["support_min_edge_distance"]) for row in rows]),
        "mean_support_mean_edge_distance": _mean([float(row["support_mean_edge_distance"]) for row in rows]),
        "mean_support_span": _mean([float(row["support_span"]) for row in rows]),
    }


def main() -> None:
    args = _parse_args()
    set_hparams(
        config=args.config,
        hparams_str=_compose_hparams_override(args),
        global_hparams=True,
        print_hparams=False,
        reset=True,
    )

    splits = _parse_splits(args.splits)
    candidate_tokens = _parse_int_list(args.candidate_tokens)
    drop_edge_values = _parse_int_list(args.drop_edge_runs)
    datasets = {split: ConanDataset(prefix=split, shuffle=False) for split in splits}
    ds0 = next(iter(datasets.values()))
    min_boundary_confidence = float(ds0.hparams.get("rhythm_v3_min_boundary_confidence_for_g", 0.5) or 0.5)
    min_prompt_speech_ratio = float(ds0.hparams.get("rhythm_v3_min_prompt_speech_ratio", 0.6) or 0.6)
    min_prompt_ref_len_sec = float(ds0.hparams.get("rhythm_v3_min_prompt_ref_len_sec", 3.0) or 3.0)
    max_prompt_ref_len_sec = float(ds0.hparams.get("rhythm_v3_max_prompt_ref_len_sec", 8.0) or 8.0)

    rows: list[dict[str, Any]] = []
    for split, ds in datasets.items():
        for raw_item in _materialize_raw_items(ds):
            for candidate_token in candidate_tokens:
                for drop_edge_runs in drop_edge_values:
                    rows.append(
                        _audit_item(
                            ds=ds,
                            split=split,
                            raw_item=raw_item,
                            candidate_token=int(candidate_token),
                            drop_edge_runs=int(drop_edge_runs),
                            min_boundary_confidence=min_boundary_confidence,
                            min_prompt_speech_ratio=min_prompt_speech_ratio,
                            min_prompt_ref_len_sec=min_prompt_ref_len_sec,
                            max_prompt_ref_len_sec=max_prompt_ref_len_sec,
                        )
                    )

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["candidate_token"]), int(row["drop_edge_runs_for_g"]))
        grouped.setdefault(key, []).append(row)
    summary = [_summarize_group(grouped[key]) for key in sorted(grouped.keys())]

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    _write_csv(output_csv, rows)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            {
                "config": {
                    "config": args.config,
                    "splits": splits,
                    "candidate_tokens": candidate_tokens,
                    "drop_edge_runs": drop_edge_values,
                    "min_boundary_confidence_for_g": min_boundary_confidence,
                    "min_prompt_speech_ratio": min_prompt_speech_ratio,
                    "min_prompt_ref_len_sec": min_prompt_ref_len_sec,
                    "max_prompt_ref_len_sec": max_prompt_ref_len_sec,
                },
                "summary": summary,
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"[counterfactual-support-degeneracy] rows={len(rows)}")
    print(f"[counterfactual-support-degeneracy] wrote_csv={output_csv}")
    print(f"[counterfactual-support-degeneracy] wrote_json={output_json}")
    for row in summary:
        print(
            "[counterfactual-support-degeneracy] "
            f"token={int(row['candidate_token'])} drop_edge={int(row['drop_edge_runs_for_g'])} "
            f"singleton_seed={int(row['singleton_seed_items'])}/{int(row['item_count'])} "
            f"singleton_support={int(row['singleton_support_items'])}/{int(row['item_count'])} "
            f"mean_support={float(row['mean_support_count']):.3f} "
            f"mean_edge_distance={float(row['mean_support_min_edge_distance']):.3f}"
        )


if __name__ == "__main__":
    main()
