from __future__ import annotations

import argparse
import os
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
from tasks.Conan.dataset import ConanDataset
from utils.commons.hparams import set_hparams


# Maintained default should not depend on local quick configs with machine-specific paths.
DEFAULT_CONFIG = os.environ.get("CONAN_RHYTHM_CONFIG", "egs/conan_emformer_rhythm_v3.yaml")
DEFAULT_SPLITS = "train,valid,test"
DEFAULT_THRESHOLDS = "0.5,0.45,0.4,0.35,0.3,0.0"
DEFAULT_OUTPUT_CSV = "tmp/gate1_boundary_audit/full_split_boundary_audit_rows.csv"
DEFAULT_OUTPUT_JSON = "tmp/gate1_boundary_audit/full_split_boundary_audit_report.json"
DEFAULT_RELAXED_HPARAMS = (
    "use_pitch_embed=False,"
    "rhythm_v3_gate_quality_strict=False,"
    "rhythm_v3_strict_eval_invalid_g=True"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit prompt-side boundary clean-support viability for rhythm_v3 zero-training falsification."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Config YAML.")
    parser.add_argument("--splits", default=DEFAULT_SPLITS, help="Comma-separated dataset splits.")
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS, help="Comma-separated boundary thresholds to sweep.")
    parser.add_argument("--binary_data_dir", default="", help="Optional binary_data_dir override.")
    parser.add_argument("--processed_data_dir", default="", help="Optional processed_data_dir override.")
    parser.add_argument("--hparams", default="", help="Extra hparam overrides appended after relaxed audit defaults.")
    parser.add_argument("--output_csv", default=DEFAULT_OUTPUT_CSV, help="Where to write flattened item rows.")
    parser.add_argument("--output_json", default=DEFAULT_OUTPUT_JSON, help="Where to write the structured audit report.")
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


def _parse_splits(text: str) -> list[str]:
    splits = [part.strip() for part in str(text).split(",") if part.strip()]
    if not splits:
        raise ValueError("At least one split is required.")
    return splits


def _parse_thresholds(text: str) -> list[float]:
    values: list[float] = []
    seen: set[float] = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        value = round(float(part), 6)
        if value in seen:
            continue
        values.append(value)
        seen.add(value)
    if not values:
        raise ValueError("At least one threshold is required.")
    return values


def _merge_thresholds(thresholds: list[float], extra: float | None) -> list[float]:
    if extra is None:
        return thresholds
    value = round(float(extra), 6)
    if any(abs(item - value) <= 1.0e-8 for item in thresholds):
        return thresholds
    return [value, *thresholds]


def _threshold_key(value: float) -> str:
    return f"{float(value):.2f}"


def _as_tokens(content: Any) -> list[int]:
    if isinstance(content, str):
        return [int(float(token)) for token in content.split() if str(token).strip() != ""]
    array = np.asarray(content).reshape(-1)
    return [int(token) for token in array.tolist()]


def _scalar_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    array = np.asarray(value)
    if array.size <= 0:
        return float("nan")
    try:
        return float(array.reshape(-1)[0])
    except (TypeError, ValueError):
        return float("nan")


def _array_1d(value: Any, *, dtype, default: np.ndarray | None = None) -> np.ndarray:
    if value is None:
        if default is None:
            raise ValueError("Missing required array value.")
        return np.asarray(default, dtype=dtype).reshape(-1)
    return np.asarray(value, dtype=dtype).reshape(-1)


def _finite_percentile(values: list[float], q: float) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float32)
    if finite.size <= 0:
        return float("nan")
    return float(np.percentile(finite, q))


def _finite_mean(values: list[float]) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float32)
    if finite.size <= 0:
        return float("nan")
    return float(finite.mean())


def _is_valid_ref_len(value: float, *, min_ref_len_sec: float, max_ref_len_sec: float) -> bool:
    return bool(np.isfinite(value) and value >= min_ref_len_sec and value <= max_ref_len_sec)


def _audit_item(
    *,
    ds: ConanDataset,
    split: str,
    local_idx: int,
    thresholds: list[float],
    drop_edge_runs: int,
    min_prompt_speech_ratio: float,
    min_prompt_ref_len_sec: float,
    max_prompt_ref_len_sec: float,
    silent_token: int | None,
    phrase_boundary_threshold: float,
) -> dict[str, Any]:
    raw_item = ds._get_raw_item_cached(local_idx)
    item_name = str(raw_item.get("item_name", f"{split}_{local_idx}"))
    speaker = str(raw_item.get("speaker", item_name.split("_", 1)[0]))
    prompt_id = str(raw_item.get("prompt_id", item_name))

    try:
        ref_item = ds._materialize_rhythm_cache_compat(raw_item, item_name=item_name)
        conditioning = ds._build_reference_prompt_unit_conditioning(
            ref_item,
            target_mode=ds._resolve_rhythm_target_mode(),
        )
    except Exception as exc:  # pragma: no cover - audit should keep going on bad items
        return {
            "split": split,
            "item_name": item_name,
            "speaker": speaker,
            "prompt_id": prompt_id,
            "error": str(exc),
        }

    prompt_duration_obs = _array_1d(conditioning["prompt_duration_obs"], dtype=np.float32)
    prompt_valid_mask = _array_1d(
        conditioning.get("prompt_valid_mask", conditioning.get("prompt_unit_mask")),
        dtype=np.float32,
        default=np.ones_like(prompt_duration_obs, dtype=np.float32),
    )
    prompt_speech_mask = _array_1d(
        conditioning.get("prompt_speech_mask"),
        dtype=np.float32,
        default=np.ones_like(prompt_duration_obs, dtype=np.float32),
    )
    prompt_closed_mask = _array_1d(
        conditioning.get("prompt_closed_mask"),
        dtype=np.float32,
        default=np.ones_like(prompt_duration_obs, dtype=np.float32),
    )
    prompt_boundary_confidence = _array_1d(
        conditioning.get("prompt_boundary_confidence"),
        dtype=np.float32,
        default=np.zeros_like(prompt_duration_obs, dtype=np.float32),
    )
    prompt_ref_len_sec = _scalar_or_nan(conditioning.get("prompt_ref_len_sec"))
    if not np.isfinite(prompt_ref_len_sec):
        prompt_ref_len_sec = _scalar_or_nan(raw_item.get("duration"))
    prompt_speech_ratio = _scalar_or_nan(conditioning.get("prompt_speech_ratio_scalar"))
    source_run_stability = _array_1d(
        ref_item.get("source_run_stability"),
        dtype=np.float32,
        default=np.ones_like(prompt_duration_obs, dtype=np.float32),
    )
    open_run_mask = _array_1d(
        ref_item.get("open_run_mask"),
        dtype=np.float32,
        default=np.zeros_like(prompt_duration_obs, dtype=np.float32),
    )
    sep_hint = _array_1d(
        ref_item.get("sep_hint"),
        dtype=np.float32,
        default=np.zeros_like(prompt_duration_obs, dtype=np.float32),
    )
    source_silence_mask = _array_1d(
        ref_item.get("source_silence_mask"),
        dtype=np.float32,
        default=np.zeros_like(prompt_duration_obs, dtype=np.float32),
    )
    source_boundary_cue = _array_1d(
        ref_item.get("source_boundary_cue"),
        dtype=np.float32,
        default=np.zeros_like(prompt_duration_obs, dtype=np.float32),
    )

    raw_tokens = _as_tokens(raw_item.get("hubert", []))
    raw_silent_token_count = (
        sum(1 for token in raw_tokens if token == int(silent_token))
        if silent_token is not None
        else 0
    )

    threshold_stats: dict[str, dict[str, Any]] = {}
    for threshold in thresholds:
        stats = summarize_global_rate_support(
            speech_mask=torch.as_tensor(prompt_speech_mask, dtype=torch.float32),
            valid_mask=torch.as_tensor(prompt_valid_mask, dtype=torch.float32),
            duration_obs=torch.as_tensor(prompt_duration_obs, dtype=torch.float32),
            drop_edge_runs=drop_edge_runs,
            min_speech_ratio=min_prompt_speech_ratio,
            closed_mask=torch.as_tensor(prompt_closed_mask, dtype=torch.float32),
            boundary_confidence=torch.as_tensor(prompt_boundary_confidence, dtype=torch.float32),
            min_boundary_confidence=float(threshold),
        )
        weight = ds._build_prompt_global_weight(
            prompt_speech_mask=prompt_speech_mask,
            run_stability=source_run_stability,
            open_run_mask=open_run_mask,
            prompt_closed_mask=prompt_closed_mask,
            prompt_boundary_confidence=prompt_boundary_confidence,
            min_boundary_confidence=float(threshold),
            drop_edge_runs=drop_edge_runs,
            allow_shape_repair=False,
        )
        support_count = float(stats.support_count.reshape(-1)[0].item())
        clean_count = float(stats.clean_count.reshape(-1)[0].item())
        support_seed_count = float(stats.support_seed_count.reshape(-1)[0].item())
        support_weight = float(np.asarray(weight, dtype=np.float32).sum())
        support_domain_valid = bool(stats.domain_valid.reshape(-1)[0].item() > 0.5)
        ref_len_valid = _is_valid_ref_len(
            prompt_ref_len_sec,
            min_ref_len_sec=min_prompt_ref_len_sec,
            max_ref_len_sec=max_prompt_ref_len_sec,
        )
        speech_ratio_valid = bool(
            np.isfinite(prompt_speech_ratio)
            and prompt_speech_ratio >= (min_prompt_speech_ratio - 1.0e-6)
        )
        g_valid_support = support_count > 0.5 and support_weight > 1.0e-6
        g_domain_valid = (
            g_valid_support
            and clean_count > 0.5
            and speech_ratio_valid
            and ref_len_valid
        )
        threshold_stats[_threshold_key(threshold)] = {
            "threshold": float(threshold),
            "support_count": support_count,
            "support_seed_count": support_seed_count,
            "clean_count": clean_count,
            "support_weight": support_weight,
            "support_domain_valid": float(support_domain_valid),
            "g_domain_valid": float(g_domain_valid),
        }

    bc_finite = prompt_boundary_confidence[np.isfinite(prompt_boundary_confidence)]
    if bc_finite.size <= 0:
        bc_max = float("nan")
        bc_mean = float("nan")
        bc_p90 = float("nan")
    else:
        bc_max = float(bc_finite.max())
        bc_mean = float(bc_finite.mean())
        bc_p90 = float(np.percentile(bc_finite, 90))
    source_boundary_finite = source_boundary_cue[np.isfinite(source_boundary_cue)]
    source_boundary_max = float(source_boundary_finite.max()) if source_boundary_finite.size > 0 else float("nan")

    return {
        "split": split,
        "item_name": item_name,
        "speaker": speaker,
        "prompt_id": prompt_id,
        "num_prompt_runs": int(prompt_duration_obs.shape[0]),
        "prompt_ref_len_sec": float(prompt_ref_len_sec),
        "prompt_ref_len_valid": float(
            _is_valid_ref_len(
                prompt_ref_len_sec,
                min_ref_len_sec=min_prompt_ref_len_sec,
                max_ref_len_sec=max_prompt_ref_len_sec,
            )
        ),
        "prompt_speech_ratio": float(prompt_speech_ratio),
        "prompt_speech_ratio_valid": float(
            np.isfinite(prompt_speech_ratio)
            and prompt_speech_ratio >= (min_prompt_speech_ratio - 1.0e-6)
        ),
        "speech_run_count": float((prompt_speech_mask * prompt_valid_mask).sum()),
        "valid_run_count": float(prompt_valid_mask.sum()),
        "closed_run_count": float((prompt_closed_mask > 0.5).sum()),
        "open_run_count": float((open_run_mask > 0.5).sum()),
        "source_silence_run_count": float((source_silence_mask > 0.5).sum()),
        "sep_nonzero_count": int((sep_hint > 0.5).sum()),
        "raw_silent_token_count": int(raw_silent_token_count),
        "bc_max": bc_max,
        "bc_mean": bc_mean,
        "bc_p90": bc_p90,
        "source_boundary_max": source_boundary_max,
        "source_phrase_threshold": float(phrase_boundary_threshold),
        "source_phrase_reachable": float(
            np.isfinite(source_boundary_max) and source_boundary_max >= (float(phrase_boundary_threshold) - 1.0e-6)
        ),
        "threshold_stats": threshold_stats,
    }


def _flatten_row(row: dict[str, Any], *, configured_key: str) -> dict[str, Any]:
    flat = {key: value for key, value in row.items() if key != "threshold_stats"}
    threshold_stats = row.get("threshold_stats")
    if not isinstance(threshold_stats, dict):
        return flat
    config_stats = threshold_stats.get(configured_key, {})
    for field in ("support_count", "support_seed_count", "clean_count", "support_weight", "support_domain_valid", "g_domain_valid"):
        flat[f"configured_{field}"] = config_stats.get(field)
    for threshold_key, stats in threshold_stats.items():
        for field, value in stats.items():
            if field == "threshold":
                continue
            flat[f"{field}_{threshold_key.replace('.', '')}"] = value
    return flat


def _speaker_summary(rows: list[dict[str, Any]], *, thresholds: list[float], configured_key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        speaker = str(row.get("speaker", ""))
        grouped.setdefault(speaker, []).append(row)
    summary: dict[str, Any] = {}
    for speaker, speaker_rows in sorted(grouped.items()):
        summary[speaker] = _summarize_rows(
            speaker_rows,
            thresholds=thresholds,
            configured_key=configured_key,
            include_speaker_breakdown=False,
        )
    return summary


def _summarize_rows(
    rows: list[dict[str, Any]],
    *,
    thresholds: list[float],
    configured_key: str,
    include_speaker_breakdown: bool,
) -> dict[str, Any]:
    valid_rows = [row for row in rows if "threshold_stats" in row]
    threshold_keys = [_threshold_key(threshold) for threshold in thresholds]
    summary: dict[str, Any] = {
        "item_count": len(rows),
        "ok_items": len(valid_rows),
        "error_items": len(rows) - len(valid_rows),
        "raw_silent_token_items": sum(1 for row in valid_rows if int(row.get("raw_silent_token_count", 0)) > 0),
        "sep_nonzero_items": sum(1 for row in valid_rows if int(row.get("sep_nonzero_count", 0)) > 0),
        "source_silence_items": sum(1 for row in valid_rows if float(row.get("source_silence_run_count", 0.0)) > 0.5),
        "ref_len_valid_items": sum(1 for row in valid_rows if float(row.get("prompt_ref_len_valid", 0.0)) > 0.5),
        "speech_ratio_valid_items": sum(1 for row in valid_rows if float(row.get("prompt_speech_ratio_valid", 0.0)) > 0.5),
        "bc_max_mean": _finite_mean([float(row.get("bc_max", float("nan"))) for row in valid_rows]),
        "bc_max_p50": _finite_percentile([float(row.get("bc_max", float("nan"))) for row in valid_rows], 50),
        "bc_max_p95": _finite_percentile([float(row.get("bc_max", float("nan"))) for row in valid_rows], 95),
        "source_boundary_max_mean": _finite_mean([float(row.get("source_boundary_max", float("nan"))) for row in valid_rows]),
        "source_boundary_max_p95": _finite_percentile(
            [float(row.get("source_boundary_max", float("nan"))) for row in valid_rows], 95
        ),
        "source_phrase_reachable_items": sum(
            1 for row in valid_rows if float(row.get("source_phrase_reachable", 0.0)) > 0.5
        ),
        "configured_threshold": configured_key,
    }
    config_support_positive = 0
    config_clean_positive = 0
    config_domain_valid = 0
    sweep: dict[str, Any] = {}
    for threshold_key in threshold_keys:
        clean_positive = 0
        support_positive = 0
        domain_valid = 0
        mean_clean: list[float] = []
        mean_support: list[float] = []
        for row in valid_rows:
            stats = row["threshold_stats"].get(threshold_key)
            if not isinstance(stats, dict):
                continue
            support_count = float(stats.get("support_count", float("nan")))
            clean_count = float(stats.get("clean_count", float("nan")))
            domain_value = float(stats.get("g_domain_valid", 0.0))
            mean_clean.append(clean_count)
            mean_support.append(support_count)
            if support_count > 0.5:
                support_positive += 1
            if clean_count > 0.5:
                clean_positive += 1
            if domain_value > 0.5:
                domain_valid += 1
        sweep[threshold_key] = {
            "support_positive_items": support_positive,
            "clean_positive_items": clean_positive,
            "g_domain_valid_items": domain_valid,
            "mean_support_count": _finite_mean(mean_support),
            "mean_clean_count": _finite_mean(mean_clean),
        }
        if threshold_key == configured_key:
            config_support_positive = support_positive
            config_clean_positive = clean_positive
            config_domain_valid = domain_valid
    summary["support_positive_items"] = config_support_positive
    summary["clean_positive_items"] = config_clean_positive
    summary["g_domain_valid_items"] = config_domain_valid
    summary["threshold_sweep"] = sweep
    if include_speaker_breakdown:
        summary["speaker_slices"] = _speaker_summary(
            valid_rows,
            thresholds=thresholds,
            configured_key=configured_key,
        )
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]], *, configured_key: str) -> None:
    flat_rows = [_flatten_row(row, configured_key=configured_key) for row in rows]
    fieldnames = sorted({key for row in flat_rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in flat_rows:
            writer.writerow(row)


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
    thresholds = _parse_thresholds(args.thresholds)
    rows: list[dict[str, Any]] = []
    split_summaries: dict[str, Any] = {}
    configured_threshold: float | None = None
    phrase_boundary_threshold: float | None = None
    config_snapshot: dict[str, Any] | None = None

    for split in splits:
        ds = ConanDataset(prefix=split, shuffle=False)
        if configured_threshold is None:
            configured_threshold = float(ds.hparams.get("rhythm_v3_min_boundary_confidence_for_g", 0.0) or 0.0)
            phrase_boundary_threshold = float(ds.hparams.get("rhythm_source_phrase_threshold", 0.55) or 0.55)
            thresholds = _merge_thresholds(thresholds, configured_threshold)
            config_snapshot = {
                "config": args.config,
                "binary_data_dir": ds.hparams.get("binary_data_dir"),
                "processed_data_dir": ds.hparams.get("processed_data_dir"),
                "silent_token": ds.hparams.get("silent_token", 57),
                "separator_aware": bool(ds.hparams.get("rhythm_separator_aware", True)),
                "emit_silence_runs": bool(ds.hparams.get("rhythm_v3_emit_silence_runs", True)),
                "drop_edge_runs_for_g": int(ds.hparams.get("rhythm_v3_drop_edge_runs_for_g", 0) or 0),
                "min_prompt_speech_ratio": float(ds.hparams.get("rhythm_v3_min_prompt_speech_ratio", 0.6) or 0.0),
                "min_prompt_ref_len_sec": float(ds.hparams.get("rhythm_v3_min_prompt_ref_len_sec", 3.0) or 0.0),
                "max_prompt_ref_len_sec": float(ds.hparams.get("rhythm_v3_max_prompt_ref_len_sec", 8.0) or 0.0),
                "configured_min_boundary_confidence_for_g": configured_threshold,
                "source_phrase_threshold": phrase_boundary_threshold,
                "thresholds": thresholds,
            }
        drop_edge_runs = int(ds.hparams.get("rhythm_v3_drop_edge_runs_for_g", 0) or 0)
        min_prompt_speech_ratio = float(ds.hparams.get("rhythm_v3_min_prompt_speech_ratio", 0.6) or 0.0)
        min_prompt_ref_len_sec = float(ds.hparams.get("rhythm_v3_min_prompt_ref_len_sec", 3.0) or 0.0)
        max_prompt_ref_len_sec = float(ds.hparams.get("rhythm_v3_max_prompt_ref_len_sec", 8.0) or 0.0)
        silent_token = ds.hparams.get("silent_token", 57)
        split_phrase_boundary_threshold = float(ds.hparams.get("rhythm_source_phrase_threshold", 0.55) or 0.55)

        split_rows = [
            _audit_item(
                ds=ds,
                split=split,
                local_idx=local_idx,
                thresholds=thresholds,
                drop_edge_runs=drop_edge_runs,
                min_prompt_speech_ratio=min_prompt_speech_ratio,
                min_prompt_ref_len_sec=min_prompt_ref_len_sec,
                max_prompt_ref_len_sec=max_prompt_ref_len_sec,
                silent_token=silent_token,
                phrase_boundary_threshold=split_phrase_boundary_threshold,
            )
            for local_idx in range(len(ds.avail_idxs))
        ]
        rows.extend(split_rows)
        split_summaries[split] = _summarize_rows(
            split_rows,
            thresholds=thresholds,
            configured_key=_threshold_key(configured_threshold),
            include_speaker_breakdown=True,
        )

    configured_key = _threshold_key(configured_threshold if configured_threshold is not None else 0.0)
    overall_summary = _summarize_rows(
        rows,
        thresholds=thresholds,
        configured_key=configured_key,
        include_speaker_breakdown=False,
    )

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    _write_csv(output_csv, rows, configured_key=configured_key)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": config_snapshot,
                "rows": rows,
                "summary": {
                    "overall": overall_summary,
                    "splits": split_summaries,
                },
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"[boundary-audit] splits={','.join(splits)} items={len(rows)} "
        f"configured_threshold={configured_key}"
    )
    print(f"[boundary-audit] wrote_csv={output_csv}")
    print(f"[boundary-audit] wrote_json={output_json}")
    for split in splits:
        split_summary = split_summaries[split]
        print(
            f"[boundary-audit] split={split} items={split_summary['item_count']} "
            f"raw_silent_token_items={split_summary['raw_silent_token_items']} "
            f"sep_nonzero_items={split_summary['sep_nonzero_items']} "
            f"clean_positive_items={split_summary['clean_positive_items']} "
            f"g_domain_valid_items={split_summary['g_domain_valid_items']}"
        )
        for threshold_key, stats in split_summary["threshold_sweep"].items():
            print(
                f"[boundary-audit] split={split} threshold={threshold_key} "
                f"support_positive={stats['support_positive_items']} "
                f"clean_positive={stats['clean_positive_items']} "
                f"g_domain_valid={stats['g_domain_valid_items']}"
            )


if __name__ == "__main__":
    main()
