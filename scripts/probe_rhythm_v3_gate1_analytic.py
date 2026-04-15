from __future__ import annotations

import argparse
import copy
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
import torch

from tasks.Conan.Conan import ConanTask
from tasks.Conan.dataset import ConanDataset
from tasks.Conan.rhythm.dataset_errors import RhythmDatasetPrefilterDrop
from utils.commons.hparams import set_hparams
from utils.commons.tensor_utils import move_to_cuda
from utils.plot.rhythm_v3_viz.core import (
    build_debug_records_from_batch,
    record_summary,
    save_debug_records,
)
from utils.plot.rhythm_v3_viz.review import (
    bootstrap_ci,
)
from scripts.rhythm_v3_probe_cases import build_auto_gate1_cases, compute_conditioning_runtime_control


DEFAULT_CONFIG = "egs/local_arctic_rhythm_v3_quick_gate1.yaml"
DEFAULT_SPLIT = "train"
DEFAULT_OUTPUT_CSV = "tmp/gate1_analytic_probe/results.csv"
DEFAULT_OUTPUT_JSON = "tmp/gate1_analytic_probe/summary.json"
DEFAULT_MIN_REAL_RANGE = 0.05
DEFAULT_RELAXED_HPARAMS = (
    "use_pitch_embed=False,"
    "rhythm_v3_gate_quality_strict=False,"
    "rhythm_v3_strict_eval_invalid_g=True,"
    "rhythm_v3_alignment_prefilter_bad_samples=False,"
    "rhythm_v3_alignment_unmatched_speech_ratio_max=1.0,"
    "rhythm_v3_alignment_local_margin_p10_min=0.0,"
    "rhythm_v3_alignment_mean_local_confidence_speech_min=0.0,"
    "rhythm_v3_alignment_mean_coarse_confidence_speech_min=0.0"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-training Gate 1 analytic probe with manual prompt overrides.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Config YAML.")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Dataset split to probe.")
    parser.add_argument("--binary_data_dir", default="", help="Optional binary_data_dir override.")
    parser.add_argument("--processed_data_dir", default="", help="Optional processed_data_dir override.")
    parser.add_argument("--hparams", default="", help="Extra hparam overrides appended after relaxed probe defaults.")
    parser.add_argument("--load_ckpt", default="", help="Optional checkpoint path or directory to load before probing.")
    parser.add_argument(
        "--eval_mode",
        default="",
        help="Optional runtime eval_mode override. Typical values: analytic, coarse_only, learned.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, cuda, or cuda:<index>.",
    )
    parser.add_argument(
        "--cases_json",
        default="",
        help="Optional JSON file with probe cases. Defaults to built-in quick-train same-speaker different-text cases.",
    )
    parser.add_argument("--output_csv", default=DEFAULT_OUTPUT_CSV, help="Where to write row-level results.")
    parser.add_argument("--output_json", default=DEFAULT_OUTPUT_JSON, help="Where to write grouped summary.")
    parser.add_argument("--output_bundle", default="", help="Optional .pt path for raw debug-record export.")
    parser.add_argument("--min_real_range", type=float, default=DEFAULT_MIN_REAL_RANGE, help="Minimum tempo span required to count as a meaningful analytic control response.")
    parser.add_argument("--max_sources_per_speaker", type=int, default=3, help="How many source items to probe per speaker.")
    parser.add_argument("--min_ref_gap", type=float, default=0.08, help="Minimum slow/fast prompt gap required when auto-building cases.")
    parser.add_argument("--min_fast_slow_effect", type=float, default=0.02, help="Minimum projected fast-slow effect size required for a positive control response.")
    parser.add_argument("--max_clip_hit_rate", type=float, default=0.50, help="Maximum allowed mean analytic clip-hit rate for a source-level pass.")
    parser.add_argument("--max_boundary_hit_rate", type=float, default=0.40, help="Maximum allowed mean projector boundary-hit rate for a source-level pass.")
    return parser.parse_args()


def _compose_hparams_override(args: argparse.Namespace) -> str:
    parts = [DEFAULT_RELAXED_HPARAMS]
    if args.binary_data_dir:
        parts.append(f"binary_data_dir='{args.binary_data_dir}'")
    if args.processed_data_dir:
        parts.append(f"processed_data_dir='{args.processed_data_dir}'")
    if args.load_ckpt:
        parts.append(f"load_ckpt='{args.load_ckpt}'")
    if args.eval_mode:
        parts.append(f"rhythm_v3_eval_mode='{args.eval_mode}'")
    if args.hparams:
        parts.append(args.hparams)
    return ",".join(part for part in parts if part)


def _resolve_device(spec: str) -> tuple[str, int | None]:
    normalized = str(spec or "auto").strip().lower()
    if normalized in {"", "auto"}:
        if torch.cuda.is_available():
            return "cuda:0", 0
        return "cpu", None
    if normalized == "cpu":
        return "cpu", None
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
        return "cuda:0", 0
    if normalized.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
        try:
            gpu_id = int(normalized.split(":", 1)[1])
        except Exception as exc:
            raise ValueError(f"Unsupported device spec: {spec!r}") from exc
        return f"cuda:{gpu_id}", gpu_id
    raise ValueError(f"Unsupported device spec: {spec!r}")


def _load_cases(args: argparse.Namespace, ds: ConanDataset) -> list[dict[str, Any]]:
    if not args.cases_json:
        return build_auto_gate1_cases(
            ds,
            max_sources_per_speaker=int(args.max_sources_per_speaker),
            min_slow_fast_gap=float(args.min_ref_gap),
        )
    with open(args.cases_json, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("cases_json must contain a list of case objects.")
    return payload


def _item_name_to_local_index(ds: ConanDataset) -> dict[str, int]:
    resolver = getattr(ds, "_item_name_to_local_index", None)
    if callable(resolver):
        return dict(resolver())
    mapping: dict[str, int] = {}
    for local_idx in range(len(ds.avail_idxs)):
        raw_item = ds._get_raw_item_cached(local_idx)
        item_name = str(raw_item["item_name"])
        if item_name in mapping:
            raise RuntimeError(f"Duplicate item_name detected inside split '{ds.prefix}': {item_name}")
        mapping[item_name] = int(local_idx)
    return mapping


def _source_local_to_fetch_index(ds: ConanDataset) -> dict[int, int]:
    pair_entries = getattr(ds, "_pair_entries", None)
    if not pair_entries:
        return {int(local_idx): int(local_idx) for local_idx in range(len(ds.avail_idxs))}
    mapping: dict[int, int] = {}
    for pair_index, entry in enumerate(pair_entries):
        src_local = int(entry["src_local"])
        mapping.setdefault(src_local, int(pair_index))
    return mapping


def _prompt_tensor(key: str, value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if key == "prompt_content_units" or key.endswith("_vocab_size"):
            return value.long()
        return value.float()
    if key == "prompt_content_units" or key.endswith("_vocab_size"):
        return torch.as_tensor(value, dtype=torch.long)
    return torch.as_tensor(value, dtype=torch.float32)


def _build_probe_sample(
    ds: ConanDataset,
    *,
    source_fetch_index: int,
    ref_local_index: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    sample = copy.deepcopy(ds[source_fetch_index])
    raw_ref_item = ds._get_raw_item_cached(ref_local_index)
    ref_item = ds._materialize_rhythm_cache_compat(
        raw_ref_item,
        item_name=str(raw_ref_item.get("item_name", "<unknown-ref-item>")),
    )
    target_mode = ds._resolve_rhythm_target_mode()
    conditioning = ds._build_reference_prompt_unit_conditioning(ref_item, target_mode=target_mode)
    if not conditioning:
        raise RuntimeError(f"Failed to build prompt conditioning for ref '{raw_ref_item.get('item_name')}'.")

    for key in list(sample.keys()):
        if key.startswith("prompt_"):
            sample.pop(key)
    for key, value in conditioning.items():
        sample[key] = _prompt_tensor(key, value)

    ref_name = str(raw_ref_item["item_name"])
    source_local = int(sample["item_id"])
    sample["ref_item_id"] = int(ref_local_index)
    sample["_raw_ref_item"] = raw_ref_item
    sample["ref_mel"] = ds._mel_to_tensor(raw_ref_item["mel"], max_frames=ds.hparams["max_frames"])
    sample["rhythm_reference_is_self"] = torch.tensor(
        [1.0 if source_local == int(ref_local_index) else 0.0],
        dtype=torch.float32,
    )
    return sample, conditioning, raw_ref_item, ref_item


def _run_condition(
    *,
    ds: ConanDataset,
    task: ConanTask,
    cuda_gpu_id: int | None,
    source_name: str,
    source_fetch_index: int,
    ref_name: str,
    ref_local_index: int,
    ref_condition: str,
) -> tuple[dict[str, Any], Any | None]:
    try:
        sample, conditioning, raw_ref_item, _ = _build_probe_sample(
            ds,
            source_fetch_index=source_fetch_index,
            ref_local_index=ref_local_index,
        )
    except (RhythmDatasetPrefilterDrop, RuntimeError) as exc:
        raw_ref_item = ds._get_raw_item_cached(ref_local_index)
        summary = {
            "g_domain_valid": 0.0,
            "gate0_row_dropped": 1.0,
            "tempo_out": float("nan"),
            "probe_error": str(exc),
            "source_name": source_name,
            "ref_name": ref_name,
            "ref_condition": ref_condition,
            "prompt_tempo_ref_runtime": float("nan"),
            "prompt_g_ref": float("nan"),
            "prompt_g_status": "conditioning_error",
            "prompt_total_units": int(np.asarray(raw_ref_item.get("dur_anchor_src", [])).reshape(-1).shape[0]),
            "same_speaker_reference": int(source_name.split("_", 1)[0] == ref_name.split("_", 1)[0]),
            "same_text_reference": int(source_name == ref_name),
            "ref_pair_item_name": str(raw_ref_item.get("item_name", ref_name)),
            "src_prefix_stat_mode": str(ds.hparams.get("rhythm_v3_src_prefix_stat_mode", "ema")),
        }
        return summary, None
    raw_source_item = sample.get("_raw_item")
    raw_paired_target_item = sample.get("_raw_paired_target_item")
    batch = ds.collater([sample])
    if cuda_gpu_id is not None:
        batch = move_to_cuda(batch, gpu_id=int(cuda_gpu_id))
    if (
        "rhythm_v3_cache_meta" not in batch
        and isinstance(raw_source_item, dict)
        and raw_source_item.get("rhythm_v3_cache_meta") is not None
    ):
        batch["rhythm_v3_cache_meta"] = [copy.deepcopy(raw_source_item["rhythm_v3_cache_meta"])]
    metadata = {
        "sample_id": f"{source_name}::{ref_condition}",
        "src_id": source_name,
        "src_item_name": source_name,
        "src_prompt_id": source_name,
        "src_spk": source_name.split("_", 1)[0],
        "ref_item_name": ref_name,
        "ref_prompt_id": ref_name,
        "ref_spk": ref_name.split("_", 1)[0],
        "ref_condition": ref_condition,
        "ref_bin": ref_condition,
        "case_source": source_name,
        "same_text_reference": int(source_name == ref_name),
        "same_speaker_reference": int(source_name.split("_", 1)[0] == ref_name.split("_", 1)[0]),
    }
    if raw_source_item is not None:
        metadata["source_text_signature"] = ds._extract_text_signature(raw_source_item)
        metadata.setdefault("src_wav", raw_source_item.get("wav_fn"))
    if raw_ref_item is not None:
        metadata["reference_text_signature"] = ds._extract_text_signature(raw_ref_item)
        metadata.setdefault("ref_wav", raw_ref_item.get("wav_fn"))
    if raw_paired_target_item is not None:
        metadata["paired_target_item_name"] = str(raw_paired_target_item.get("item_name", ""))
        metadata["paired_target_text_signature"] = ds._extract_text_signature(raw_paired_target_item)
        metadata["tgt_prompt_id"] = str(raw_paired_target_item.get("item_name", ""))
        metadata["tgt_spk"] = str(raw_paired_target_item.get("item_name", "")).split("_", 1)[0]
    try:
        with torch.no_grad():
            _, output = task.run_model(batch, infer=True, test=True)
    except ValueError as exc:
        message = str(exc)
        if "non-empty closed/boundary-clean support for g" not in message:
            raise
        summary = {
            "g_domain_valid": 0.0,
            "gate0_row_dropped": 1.0,
            "tempo_out": float("nan"),
            "probe_error": message,
        }
        record = None
    else:
        record = build_debug_records_from_batch(
            sample=batch,
            model_output=output,
            metadata=metadata,
        )[0]
        summary = record_summary(record)
    control = compute_conditioning_runtime_control(ds, conditioning)
    prompt_g_ref = float(summary.get("g_ref", float("nan")))
    if not np.isfinite(prompt_g_ref):
        prompt_g_ref = float(control["prompt_g_ref"])
    prompt_tempo_ref_runtime = float(summary.get("tempo_ref_runtime", float("nan")))
    if not np.isfinite(prompt_tempo_ref_runtime):
        prompt_tempo_ref_runtime = float(control["prompt_tempo_ref_runtime"])
    summary.update(
        {
            "source_name": source_name,
            "ref_name": ref_name,
            "ref_condition": ref_condition,
            "prompt_g_ref": float(prompt_g_ref),
            "prompt_tempo_ref_runtime": float(prompt_tempo_ref_runtime),
            "prompt_g_status": str(control["prompt_g_status"]),
            "prompt_total_units": int(np.asarray(conditioning["prompt_duration_obs"]).reshape(-1).shape[0]),
            "same_speaker_reference": int(source_name.split("_", 1)[0] == ref_name.split("_", 1)[0]),
            "same_text_reference": int(source_name == ref_name),
            "ref_pair_item_name": str(raw_ref_item.get("item_name", ref_name)),
            "src_prefix_stat_mode": str(
                summary.get("src_prefix_stat_mode")
                or ds.hparams.get("rhythm_v3_src_prefix_stat_mode", "ema")
            ),
        }
    )
    return summary, record


def _row_domain_valid(row: dict[str, Any]) -> bool:
    try:
        domain_valid = float(row.get("g_domain_valid", float("nan")))
    except Exception:
        domain_valid = float("nan")
    if np.isfinite(domain_valid):
        return bool(domain_valid > 0.5)
    try:
        gate0_row_dropped = float(row.get("gate0_row_dropped", 0.0) or 0.0)
    except Exception:
        gate0_row_dropped = 0.0
    return bool(gate0_row_dropped <= 0.5)


def _series_summary(
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    min_real_range: float,
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: float(row[x_key]))
    x_vals = [float(row[x_key]) for row in ordered]
    y_vals = [float(row[y_key]) for row in ordered]

    monotone = len(ordered) >= 3
    for left, right in zip(y_vals, y_vals[1:]):
        if not (np.isfinite(left) and np.isfinite(right) and right >= (left - 1.0e-6)):
            monotone = False
            break

    slope = float("nan")
    if len(ordered) >= 2:
        x = np.asarray(x_vals, dtype=np.float32)
        y = np.asarray(y_vals, dtype=np.float32)
        if np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and float(np.max(x) - np.min(x)) > 1.0e-6:
            slope = float(np.polyfit(x, y, deg=1)[0])

    real_range = float("nan")
    if y_vals:
        y = np.asarray(y_vals, dtype=np.float32)
        if np.all(np.isfinite(y)):
            real_range = float(np.max(y) - np.min(y))

    control_ok = bool(
        monotone
        and np.isfinite(slope)
        and slope > 0.0
        and np.isfinite(real_range)
        and real_range >= float(min_real_range)
    )
    return {
        "x_sorted": x_vals,
        "y_sorted": y_vals,
        "monotone": bool(monotone),
        "transfer_slope": slope,
        "real_range": real_range,
        "control_ok": control_ok,
    }


def _monotonicity_summary(
    rows: list[dict[str, Any]],
    *,
    min_real_range: float,
    min_fast_slow_effect: float,
    max_clip_hit_rate: float,
    max_boundary_hit_rate: float,
) -> dict[str, Any]:
    all_real_rows = [row for row in rows if row["ref_condition"] in {"slow", "mid", "fast"}]
    real_rows = [row for row in all_real_rows if _row_domain_valid(row)]

    def _empty_summary() -> dict[str, Any]:
        return {
            "x_sorted": [],
            "y_sorted": [],
            "monotone": False,
            "transfer_slope": float("nan"),
            "real_range": float("nan"),
            "control_ok": False,
        }

    raw_summary = (
        _series_summary(
            real_rows,
            x_key="prompt_tempo_ref_runtime",
            y_key="tempo_out_raw" if "tempo_out_raw" in real_rows[0] else "tempo_out",
            min_real_range=min_real_range,
        )
        if real_rows
        else _empty_summary()
    )
    preproj_summary = (
        _series_summary(
            real_rows,
            x_key="prompt_tempo_ref_runtime",
            y_key="tempo_out_preproj" if "tempo_out_preproj" in real_rows[0] else "tempo_out",
            min_real_range=min_real_range,
        )
        if real_rows
        else _empty_summary()
    )
    exec_summary = (
        _series_summary(
            real_rows,
            x_key="prompt_tempo_ref_runtime",
            y_key="tempo_out_exec" if "tempo_out_exec" in real_rows[0] else "tempo_out",
            min_real_range=min_real_range,
        )
        if real_rows
        else _empty_summary()
    )
    controls = {
        row["ref_condition"]: float(row.get("tempo_out_exec", row.get("tempo_out", float("nan"))))
        for row in rows
        if row["ref_condition"] not in {"slow", "mid", "fast"}
    }

    mean_saturation = float(
        np.nanmean(np.asarray([row.get("analytic_clip_hit_rate", float("nan")) for row in real_rows], dtype=np.float32))
    ) if real_rows else float("nan")
    mean_boundary_hit = float(
        np.nanmean(np.asarray([row.get("projector_boundary_hit_rate", float("nan")) for row in real_rows], dtype=np.float32))
    ) if real_rows else float("nan")
    mean_bucket_count = float(
        np.nanmean(np.asarray([row.get("projector_bucket_count", float("nan")) for row in real_rows], dtype=np.float32))
    ) if real_rows else float("nan")
    exec_fast_minus_slow = float("nan")
    if len(exec_summary["y_sorted"]) >= 2:
        left = exec_summary["y_sorted"][-1]
        right = exec_summary["y_sorted"][0]
        if np.isfinite(left) and np.isfinite(right):
            exec_fast_minus_slow = float(left - right)
    exec_pass = bool(
        exec_summary["control_ok"]
        and np.isfinite(exec_fast_minus_slow)
        and exec_fast_minus_slow >= float(min_fast_slow_effect)
        and (
            (not np.isfinite(mean_saturation))
            or mean_saturation <= float(max_clip_hit_rate)
        )
        and (
            (not np.isfinite(mean_boundary_hit))
            or mean_boundary_hit <= float(max_boundary_hit_rate)
        )
    )
    return {
        "source_name": rows[0]["source_name"] if rows else "",
        "src_prefix_stat_mode": rows[0].get("src_prefix_stat_mode", "") if rows else "",
        "total_real_row_count": len(all_real_rows),
        "valid_real_row_count": len(real_rows),
        "all_real_domain_valid": bool(len(real_rows) == len(all_real_rows) and len(all_real_rows) > 0),
        "tempo_ref_runtime_sorted": exec_summary["x_sorted"],
        "monotone_by_prompt_tempo_raw": raw_summary["control_ok"],
        "transfer_slope_raw": raw_summary["transfer_slope"],
        "real_tempo_range_raw": raw_summary["real_range"],
        "tempo_out_raw_sorted": raw_summary["y_sorted"],
        "monotone_by_prompt_tempo_preproj": preproj_summary["control_ok"],
        "transfer_slope_preproj": preproj_summary["transfer_slope"],
        "real_tempo_range_preproj": preproj_summary["real_range"],
        "tempo_out_preproj_sorted": preproj_summary["y_sorted"],
        "monotone_by_prompt_tempo_exec": exec_pass,
        "transfer_slope_exec": exec_summary["transfer_slope"],
        "real_tempo_range_exec": exec_summary["real_range"],
        "tempo_out_exec_sorted": exec_summary["y_sorted"],
        "monotone_by_prompt_tempo": exec_pass,
        "transfer_slope": exec_summary["transfer_slope"],
        "real_tempo_range": exec_summary["real_range"],
        "fast_minus_slow": exec_fast_minus_slow,
        "tempo_ref_sorted": exec_summary["x_sorted"],
        "tempo_out_sorted": exec_summary["y_sorted"],
        "mean_analytic_saturation_rate": mean_saturation,
        "mean_projector_boundary_hit_rate": mean_boundary_hit,
        "mean_projector_bucket_count": mean_bucket_count,
        "relaxed_probe": True,
        "controls": controls,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    fieldnames = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = _parse_args()
    device_spec, cuda_gpu_id = _resolve_device(args.device)
    set_hparams(
        config=args.config,
        hparams_str=_compose_hparams_override(args),
        global_hparams=True,
        print_hparams=False,
        reset=True,
    )
    ds = ConanDataset(prefix=args.split, shuffle=False)
    cases = _load_cases(args, ds)
    task = ConanTask()
    task.build_model()
    task.global_step = 0
    task.eval()
    if device_spec != "cpu":
        task = task.to(torch.device(device_spec))

    name_to_local = _item_name_to_local_index(ds)
    source_to_fetch = _source_local_to_fetch_index(ds)

    rows: list[dict[str, Any]] = []
    records: list[Any] = []
    for case in cases:
        source_name = str(case["source"])
        refs = dict(case["refs"])
        refs.setdefault("source_only", source_name)

        if source_name not in name_to_local:
            raise KeyError(f"Source item not found in split '{args.split}': {source_name}")
        source_local = int(name_to_local[source_name])
        if source_local not in source_to_fetch:
            raise KeyError(f"No fetch index found for source local id {source_local} ({source_name}).")
        source_fetch_index = int(source_to_fetch[source_local])

        for ref_condition, ref_name in refs.items():
            ref_name = str(ref_name)
            if ref_name not in name_to_local:
                raise KeyError(f"Reference item not found in split '{args.split}': {ref_name}")
            row, record = _run_condition(
                ds=ds,
                task=task,
                cuda_gpu_id=cuda_gpu_id,
                source_name=source_name,
                source_fetch_index=source_fetch_index,
                ref_name=ref_name,
                ref_local_index=int(name_to_local[ref_name]),
                ref_condition=str(ref_condition),
            )
            rows.append(row)
            if record is not None:
                records.append(record)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_name"])].append(row)
    summary_rows = [
        _monotonicity_summary(
            grouped[source_name],
            min_real_range=float(args.min_real_range),
            min_fast_slow_effect=float(args.min_fast_slow_effect),
            max_clip_hit_rate=float(args.max_clip_hit_rate),
            max_boundary_hit_rate=float(args.max_boundary_hit_rate),
        )
        for source_name in sorted(grouped.keys())
    ]
    slopes = [
        float(row["transfer_slope"])
        for row in summary_rows
        if np.isfinite(float(row.get("transfer_slope", float("nan"))))
    ]
    effects = [
        float(row["fast_minus_slow"])
        for row in summary_rows
        if np.isfinite(float(row.get("fast_minus_slow", float("nan"))))
    ]
    aggregate = {
        "source_count": int(len(summary_rows)),
        "monotone_source_count": int(sum(1 for row in summary_rows if bool(row["monotone_by_prompt_tempo"]))),
        "transfer_slope_ci": bootstrap_ci(slopes) if slopes else (float("nan"), float("nan"), float("nan")),
        "fast_minus_slow_ci": bootstrap_ci(effects) if effects else (float("nan"), float("nan"), float("nan")),
        "max_sources_per_speaker": int(args.max_sources_per_speaker),
        "min_ref_gap": float(args.min_ref_gap),
        "min_fast_slow_effect": float(args.min_fast_slow_effect),
        "max_clip_hit_rate": float(args.max_clip_hit_rate),
        "max_boundary_hit_rate": float(args.max_boundary_hit_rate),
        "relaxed_probe": True,
    }

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    _write_csv(output_csv, rows)
    if args.output_bundle:
        save_debug_records(records, args.output_bundle)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "rows": rows,
                "summary": summary_rows,
                "aggregate": aggregate,
                "record_count": len(records),
                "output_bundle": str(args.output_bundle) if args.output_bundle else "",
                "device": device_spec,
                "load_ckpt": str(args.load_ckpt or ""),
                "eval_mode": str(args.eval_mode or ds.hparams.get("rhythm_v3_eval_mode", "analytic")),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"[gate1-probe] split={args.split} cases={len(cases)} rows={len(rows)} "
        f"records={len(records)} device={device_spec} "
        f"eval_mode={str(args.eval_mode or ds.hparams.get('rhythm_v3_eval_mode', 'analytic'))}"
    )
    print(f"[gate1-probe] wrote_csv={output_csv}")
    print(f"[gate1-probe] wrote_json={output_json}")
    if args.output_bundle:
        print(f"[gate1-probe] wrote_bundle={args.output_bundle}")
    for row in summary_rows:
        source_name = row["source_name"]
        mono_pre = "PASS" if row["monotone_by_prompt_tempo_preproj"] else "FAIL"
        mono_exec = "PASS" if row["monotone_by_prompt_tempo_exec"] else "FAIL"
        s_pre = row["transfer_slope_preproj"]
        s_exec = row["transfer_slope_exec"]
        s_pre_str = "nan" if not np.isfinite(s_pre) else f"{s_pre:.4f}"
        s_exec_str = "nan" if not np.isfinite(s_exec) else f"{s_exec:.4f}"
        print(
            f"[gate1-probe] {source_name} "
            f"preproj={mono_pre}/{s_pre_str} exec={mono_exec}/{s_exec_str} "
            f"range_pre={row['real_tempo_range_preproj']:.4f} range_exec={row['real_tempo_range_exec']:.4f} "
            f"valid_real={row['valid_real_row_count']}/{row['total_real_row_count']} "
            f"tempo_ref={row['tempo_ref_runtime_sorted']} "
            f"tempo_preproj={row['tempo_out_preproj_sorted']} tempo_exec={row['tempo_out_exec_sorted']}"
        )
        if row["controls"]:
            print(f"[gate1-probe] {source_name} controls={row['controls']}")


if __name__ == "__main__":
    main()
