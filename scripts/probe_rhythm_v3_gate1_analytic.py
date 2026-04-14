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
from utils.plot.rhythm_v3_viz.core import build_debug_records_from_batch, record_summary
from utils.plot.rhythm_v3_viz.review import (
    compute_source_global_rate_for_analysis,
    compute_speech_tempo_for_analysis,
)
from scripts.rhythm_v3_probe_cases import build_auto_gate1_cases


DEFAULT_CONFIG = "egs/local_arctic_rhythm_v3_quick_gate1.yaml"
DEFAULT_SPLIT = "train"
DEFAULT_OUTPUT_CSV = "tmp/gate1_analytic_probe/results.csv"
DEFAULT_OUTPUT_JSON = "tmp/gate1_analytic_probe/summary.json"
DEFAULT_MIN_REAL_RANGE = 0.01
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
    parser.add_argument(
        "--cases_json",
        default="",
        help="Optional JSON file with probe cases. Defaults to built-in quick-train same-speaker different-text cases.",
    )
    parser.add_argument("--output_csv", default=DEFAULT_OUTPUT_CSV, help="Where to write row-level results.")
    parser.add_argument("--output_json", default=DEFAULT_OUTPUT_JSON, help="Where to write grouped summary.")
    parser.add_argument("--min_real_range", type=float, default=DEFAULT_MIN_REAL_RANGE, help="Minimum tempo span required to count as a meaningful analytic control response.")
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


def _load_cases(args: argparse.Namespace, ds: ConanDataset) -> list[dict[str, Any]]:
    if not args.cases_json:
        return build_auto_gate1_cases(ds)
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


def _conditioning_tempo(ds: ConanDataset, conditioning: dict[str, np.ndarray]) -> float:
    return compute_speech_tempo_for_analysis(
        source_duration_obs=conditioning.get("prompt_duration_obs"),
        source_speech_mask=conditioning.get("prompt_speech_mask"),
        source_valid_mask=conditioning.get("prompt_valid_mask"),
        source_weight=conditioning.get("prompt_global_weight"),
        source_unit_ids=conditioning.get("prompt_content_units"),
        source_closed_mask=conditioning.get("prompt_closed_mask"),
        source_boundary_confidence=conditioning.get("prompt_boundary_confidence"),
        min_boundary_confidence=ds.hparams.get("rhythm_v3_min_boundary_confidence_for_g"),
        g_variant=str(ds.hparams.get("rhythm_v3_g_variant", "raw_median")),
        g_trim_ratio=float(ds.hparams.get("rhythm_v3_g_trim_ratio", 0.2) or 0.2),
        drop_edge_runs=int(ds.hparams.get("rhythm_v3_drop_edge_runs_for_g", 0) or 0),
    )


def _conditioning_g(ds: ConanDataset, conditioning: dict[str, np.ndarray]) -> tuple[float, str]:
    return compute_source_global_rate_for_analysis(
        source_duration_obs=conditioning.get("prompt_duration_obs"),
        source_speech_mask=conditioning.get("prompt_speech_mask"),
        source_valid_mask=conditioning.get("prompt_valid_mask"),
        source_weight=conditioning.get("prompt_global_weight"),
        source_unit_ids=conditioning.get("prompt_content_units"),
        source_closed_mask=conditioning.get("prompt_closed_mask"),
        source_boundary_confidence=conditioning.get("prompt_boundary_confidence"),
        g_variant=str(ds.hparams.get("rhythm_v3_g_variant", "raw_median")),
        g_trim_ratio=float(ds.hparams.get("rhythm_v3_g_trim_ratio", 0.2) or 0.2),
        drop_edge_runs=int(ds.hparams.get("rhythm_v3_drop_edge_runs_for_g", 0) or 0),
        min_boundary_confidence=ds.hparams.get("rhythm_v3_min_boundary_confidence_for_g"),
        require_explicit_speech_mask=False,
        return_status=True,
    )


def _run_condition(
    *,
    ds: ConanDataset,
    task: ConanTask,
    source_name: str,
    source_fetch_index: int,
    ref_name: str,
    ref_local_index: int,
    ref_condition: str,
) -> dict[str, Any]:
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
            "prompt_tempo_ref": float("nan"),
            "prompt_g_ref": float("nan"),
            "prompt_g_status": "conditioning_error",
            "prompt_total_units": int(np.asarray(raw_ref_item.get("dur_anchor_src", [])).reshape(-1).shape[0]),
            "same_speaker_reference": int(source_name.split("_", 1)[0] == ref_name.split("_", 1)[0]),
            "same_text_reference": int(source_name == ref_name),
            "ref_pair_item_name": str(raw_ref_item.get("item_name", ref_name)),
            "src_prefix_stat_mode": str(ds.hparams.get("rhythm_v3_src_prefix_stat_mode", "ema")),
        }
        return summary
    raw_source_item = sample.get("_raw_item")
    raw_paired_target_item = sample.get("_raw_paired_target_item")
    batch = ds.collater([sample])
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
    else:
        record = build_debug_records_from_batch(
            sample=batch,
            model_output=output,
            metadata=metadata,
        )[0]
        summary = record_summary(record)
    prompt_g_ref, prompt_g_status = _conditioning_g(ds, conditioning)
    summary.update(
        {
            "source_name": source_name,
            "ref_name": ref_name,
            "ref_condition": ref_condition,
            "prompt_tempo_ref": _conditioning_tempo(ds, conditioning),
            "prompt_g_ref": float(prompt_g_ref),
            "prompt_g_status": str(prompt_g_status),
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
    return summary


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


def _tempo_curve_summary(
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    tempo_key: str,
    min_real_range: float,
    direction: str = "increasing",
) -> dict[str, Any]:
    tempo_ref = [float(row.get(x_key, float("nan"))) for row in rows]
    tempo_out = [float(row.get(tempo_key, float("nan"))) for row in rows]
    monotone = len(rows) >= 3
    for left, right in zip(tempo_out, tempo_out[1:]):
        if direction == "decreasing":
            ok = np.isfinite(left) and np.isfinite(right) and right <= (left + 1.0e-6)
        else:
            ok = np.isfinite(left) and np.isfinite(right) and right >= (left - 1.0e-6)
        if not ok:
            monotone = False
            break
    slope = float("nan")
    if len(rows) >= 2:
        x = np.asarray(tempo_ref, dtype=np.float32)
        y = np.asarray(tempo_out, dtype=np.float32)
        if np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and float(np.max(x) - np.min(x)) > 1.0e-6:
            slope = float(np.polyfit(x, y, deg=1)[0])
    real_range = float("nan")
    if tempo_out:
        y = np.asarray(tempo_out, dtype=np.float32)
        if np.all(np.isfinite(y)):
            real_range = float(np.max(y) - np.min(y))
    slope_ok = np.isfinite(slope) and (
        slope < 0.0 if direction == "decreasing" else slope > 0.0
    )
    passed = bool(
        monotone
        and slope_ok
        and np.isfinite(real_range)
        and real_range >= float(min_real_range)
    )
    return {
        "x_sorted": tempo_ref,
        "tempo_out_sorted": tempo_out,
        "order_monotone": bool(monotone),
        "pass": passed,
        "transfer_slope": slope,
        "real_tempo_range": real_range,
        "direction": direction,
    }


def _monotonicity_summary(rows: list[dict[str, Any]], *, min_real_range: float) -> dict[str, Any]:
    all_real_rows = [row for row in rows if row["ref_condition"] in {"slow", "mid", "fast"}]
    real_rows = [row for row in all_real_rows if _row_domain_valid(row)]
    real_rows = sorted(real_rows, key=lambda row: float(row.get("prompt_g_ref", float("nan"))))
    controls = {
        row["ref_condition"]: float(row["tempo_out_projected"] if "tempo_out_projected" in row else row["tempo_out"])
        for row in rows
        if row["ref_condition"] not in {"slow", "mid", "fast"}
    }
    preclip = _tempo_curve_summary(
        real_rows,
        x_key="prompt_g_ref",
        tempo_key="tempo_out_preclip",
        min_real_range=min_real_range,
        direction="decreasing",
    )
    continuous = _tempo_curve_summary(
        real_rows,
        x_key="prompt_g_ref",
        tempo_key="tempo_out_continuous",
        min_real_range=min_real_range,
        direction="decreasing",
    )
    projected = _tempo_curve_summary(
        real_rows,
        x_key="prompt_g_ref",
        tempo_key="tempo_out_projected",
        min_real_range=min_real_range,
        direction="decreasing",
    )
    count_ratio_continuous = _tempo_curve_summary(
        real_rows,
        x_key="prompt_g_ref",
        tempo_key="speech_exec_ratio_continuous",
        min_real_range=0.0,
        direction="increasing",
    )
    count_ratio_projected = _tempo_curve_summary(
        real_rows,
        x_key="prompt_g_ref",
        tempo_key="speech_exec_ratio_projected",
        min_real_range=0.0,
        direction="increasing",
    )
    logstretch_continuous = _tempo_curve_summary(
        real_rows,
        x_key="prompt_g_ref",
        tempo_key="mean_speech_logstretch_continuous",
        min_real_range=0.0,
        direction="increasing",
    )
    logstretch_projected = _tempo_curve_summary(
        real_rows,
        x_key="prompt_g_ref",
        tempo_key="mean_speech_logstretch_projected",
        min_real_range=0.0,
        direction="increasing",
    )
    mean_saturation = float(
        np.nanmean(np.asarray([row.get("analytic_saturation_rate", float("nan")) for row in real_rows], dtype=np.float32))
    ) if real_rows else float("nan")
    mean_boundary_hit = float(
        np.nanmean(np.asarray([row.get("projector_boundary_hit_rate", float("nan")) for row in real_rows], dtype=np.float32))
    ) if real_rows else float("nan")
    mean_bucket_count = float(
        np.nanmean(np.asarray([row.get("projector_bucket_count", float("nan")) for row in real_rows], dtype=np.float32))
    ) if real_rows else float("nan")
    return {
        "source_name": rows[0]["source_name"] if rows else "",
        "src_prefix_stat_mode": rows[0].get("src_prefix_stat_mode", "") if rows else "",
        "total_real_row_count": len(all_real_rows),
        "valid_real_row_count": len(real_rows),
        "all_real_domain_valid": bool(len(real_rows) == len(all_real_rows) and len(all_real_rows) > 0),
        "g_ref_sorted": projected["x_sorted"],
        "prompt_tempo_ref_sorted": [float(row.get("prompt_tempo_ref", float("nan"))) for row in real_rows],
        "order_monotone_by_prompt_g": projected["order_monotone"],
        "monotone_by_prompt_g": projected["pass"],
        "order_monotone_by_prompt_tempo": projected["order_monotone"],
        "monotone_by_prompt_tempo": projected["pass"],
        "transfer_slope": projected["transfer_slope"],
        "real_tempo_range": projected["real_tempo_range"],
        "tempo_ref_sorted": projected["x_sorted"],
        "tempo_out_sorted": projected["tempo_out_sorted"],
        "preclip_monotone_by_prompt_tempo": preclip["pass"],
        "preclip_transfer_slope": preclip["transfer_slope"],
        "preclip_real_tempo_range": preclip["real_tempo_range"],
        "tempo_out_preclip_sorted": preclip["tempo_out_sorted"],
        "continuous_monotone_by_prompt_tempo": continuous["pass"],
        "continuous_transfer_slope": continuous["transfer_slope"],
        "continuous_real_tempo_range": continuous["real_tempo_range"],
        "tempo_out_continuous_sorted": continuous["tempo_out_sorted"],
        "projected_monotone_by_prompt_tempo": projected["pass"],
        "projected_transfer_slope": projected["transfer_slope"],
        "projected_real_tempo_range": projected["real_tempo_range"],
        "tempo_out_projected_sorted": projected["tempo_out_sorted"],
        "continuous_exec_ratio_slope": count_ratio_continuous["transfer_slope"],
        "continuous_exec_ratio_range": count_ratio_continuous["real_tempo_range"],
        "continuous_exec_ratio_sorted": count_ratio_continuous["tempo_out_sorted"],
        "projected_exec_ratio_slope": count_ratio_projected["transfer_slope"],
        "projected_exec_ratio_range": count_ratio_projected["real_tempo_range"],
        "projected_exec_ratio_sorted": count_ratio_projected["tempo_out_sorted"],
        "continuous_exec_logstretch_slope": logstretch_continuous["transfer_slope"],
        "continuous_exec_logstretch_range": logstretch_continuous["real_tempo_range"],
        "continuous_exec_logstretch_sorted": logstretch_continuous["tempo_out_sorted"],
        "projected_exec_logstretch_slope": logstretch_projected["transfer_slope"],
        "projected_exec_logstretch_range": logstretch_projected["real_tempo_range"],
        "projected_exec_logstretch_sorted": logstretch_projected["tempo_out_sorted"],
        "mean_analytic_saturation_rate": mean_saturation,
        "mean_projector_boundary_hit_rate": mean_boundary_hit,
        "mean_projector_bucket_count": mean_bucket_count,
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
    task.build_tts_model()
    task.global_step = 0
    task.model.eval()

    name_to_local = _item_name_to_local_index(ds)
    source_to_fetch = _source_local_to_fetch_index(ds)

    rows: list[dict[str, Any]] = []
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
            rows.append(
                _run_condition(
                    ds=ds,
                    task=task,
                    source_name=source_name,
                    source_fetch_index=source_fetch_index,
                    ref_name=ref_name,
                    ref_local_index=int(name_to_local[ref_name]),
                    ref_condition=str(ref_condition),
                )
            )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_name"])].append(row)
    summary_rows = [
        _monotonicity_summary(grouped[source_name], min_real_range=float(args.min_real_range))
        for source_name in sorted(grouped.keys())
    ]

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    _write_csv(output_csv, rows)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump({"rows": rows, "summary": summary_rows}, handle, indent=2, ensure_ascii=False)

    print(f"[gate1-probe] split={args.split} cases={len(cases)} rows={len(rows)}")
    print(f"[gate1-probe] wrote_csv={output_csv}")
    print(f"[gate1-probe] wrote_json={output_json}")
    for row in summary_rows:
        source_name = row["source_name"]
        monotone = "PASS" if row["monotone_by_prompt_tempo"] else "FAIL"
        slope = row["transfer_slope"]
        slope_str = "nan" if not np.isfinite(slope) else f"{slope:.4f}"
        cont_slope = row.get("continuous_transfer_slope", float("nan"))
        cont_slope_str = "nan" if not np.isfinite(cont_slope) else f"{cont_slope:.4f}"
        preclip_slope = row.get("preclip_transfer_slope", float("nan"))
        preclip_slope_str = "nan" if not np.isfinite(preclip_slope) else f"{preclip_slope:.4f}"
        exec_ratio_slope = row.get("projected_exec_ratio_slope", float("nan"))
        exec_ratio_slope_str = "nan" if not np.isfinite(exec_ratio_slope) else f"{exec_ratio_slope:.4f}"
        exec_logstretch_slope = row.get("projected_exec_logstretch_slope", float("nan"))
        exec_logstretch_slope_str = "nan" if not np.isfinite(exec_logstretch_slope) else f"{exec_logstretch_slope:.4f}"
        print(
            f"[gate1-probe] {source_name} monotone={monotone} "
            f"preclip_slope={preclip_slope_str} continuous_slope={cont_slope_str} projected_slope={slope_str} "
            f"projected_exec_ratio_slope={exec_ratio_slope_str} "
            f"projected_exec_logstretch_slope={exec_logstretch_slope_str} "
            f"projected_range={row['real_tempo_range']:.4f} saturation={row.get('mean_analytic_saturation_rate', float('nan')):.4f} "
            f"valid_real={row['valid_real_row_count']}/{row['total_real_row_count']} "
            f"tempo_ref={row['tempo_ref_sorted']} tempo_out={row['tempo_out_sorted']}"
        )
        if row["controls"]:
            print(f"[gate1-probe] {source_name} controls={row['controls']}")


if __name__ == "__main__":
    main()
