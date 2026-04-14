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

from modules.Conan.rhythm_v3.source_cache import build_source_phrase_cache, build_source_rhythm_cache_v3
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
DEFAULT_OUTPUT_CSV = "tmp/gate1_counterfactual_probe/results.csv"
DEFAULT_OUTPUT_JSON = "tmp/gate1_counterfactual_probe/summary.json"
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
    parser = argparse.ArgumentParser(description="Gate 1 analytic runtime probe with counterfactual prompt-side silent token.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Config YAML.")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Dataset split to probe.")
    parser.add_argument("--candidate_token", type=int, required=True, help="Counterfactual silent token id for prompt cache rebuild.")
    parser.add_argument("--trim_head_runs", type=int, default=0, help="Trim this many prompt head runs before rebuilding conditioning.")
    parser.add_argument("--trim_tail_runs", type=int, default=0, help="Trim this many prompt tail runs before rebuilding conditioning.")
    parser.add_argument("--binary_data_dir", default="", help="Optional binary_data_dir override.")
    parser.add_argument("--processed_data_dir", default="", help="Optional processed_data_dir override.")
    parser.add_argument("--hparams", default="", help="Extra hparam overrides appended after relaxed probe defaults.")
    parser.add_argument("--cases_json", default="", help="Optional JSON file with probe cases.")
    parser.add_argument("--output_csv", default=DEFAULT_OUTPUT_CSV, help="Where to write row-level results.")
    parser.add_argument("--output_json", default=DEFAULT_OUTPUT_JSON, help="Where to write grouped summary.")
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
    if isinstance(payload, dict):
        payload = payload.get("cases")
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
        mapping[item_name] = int(local_idx)
    return mapping


def _source_local_to_fetch_index(ds: ConanDataset) -> dict[int, int]:
    pair_entries = getattr(ds, "_pair_entries", None)
    if not pair_entries:
        return {int(local_idx): int(local_idx) for local_idx in range(len(ds.avail_idxs))}
    mapping: dict[int, int] = {}
    for pair_index, entry in enumerate(pair_entries):
        mapping.setdefault(int(entry["src_local"]), int(pair_index))
    return mapping


def _prompt_tensor(key: str, value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if key == "prompt_content_units" or key.endswith("_vocab_size"):
            return value.long()
        return value.float()
    if key == "prompt_content_units" or key.endswith("_vocab_size"):
        return torch.as_tensor(value, dtype=torch.long)
    return torch.as_tensor(value, dtype=torch.float32)


def _resolve_ref_len_sec_from_duration_obs(ds: ConanDataset, duration_obs) -> float | None:
    hop_size = float(ds.hparams.get("hop_size", 0.0) or 0.0)
    sample_rate = float(ds.hparams.get("audio_sample_rate", 0.0) or 0.0)
    if hop_size <= 0.0 or sample_rate <= 0.0:
        return None
    total_frames = float(np.asarray(duration_obs, dtype=np.float32).sum(dtype=np.float64))
    if total_frames <= 0.0:
        return None
    return total_frames * hop_size / sample_rate


def _trim_counterfactual_cache_edges(
    *,
    ds: ConanDataset,
    cache: dict[str, Any],
    trim_head_runs: int,
    trim_tail_runs: int,
) -> dict[str, Any]:
    trim_head_runs = max(0, int(trim_head_runs))
    trim_tail_runs = max(0, int(trim_tail_runs))
    if trim_head_runs <= 0 and trim_tail_runs <= 0:
        return cache

    run_count = int(np.asarray(cache["dur_anchor_src"]).reshape(-1).shape[0])
    if run_count <= (trim_head_runs + trim_tail_runs):
        raise ValueError(
            "Cannot trim prompt cache beyond available runs: "
            f"run_count={run_count}, trim_head_runs={trim_head_runs}, trim_tail_runs={trim_tail_runs}."
        )

    start = trim_head_runs
    stop = run_count - trim_tail_runs
    trimmed = dict(cache)
    slice_keys = (
        "content_units",
        "dur_anchor_src",
        "source_silence_mask",
        "open_run_mask",
        "sealed_mask",
        "sep_hint",
        "boundary_confidence",
        "source_run_stability",
        "source_boundary_cue",
        "phrase_group_pos",
        "phrase_final_mask",
        "unit_log_prior",
        "unit_anchor_base",
        "unit_rate_log_base",
    )
    for key in slice_keys:
        value = trimmed.get(key)
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.ndim >= 1 and int(arr.shape[0]) == run_count:
            trimmed[key] = arr[start:stop].copy()

    dur_anchor_src = np.asarray(trimmed["dur_anchor_src"], dtype=np.int64)
    sep_hint = np.asarray(trimmed.get("sep_hint", np.zeros_like(dur_anchor_src)), dtype=np.int64)
    open_run_mask = np.asarray(trimmed.get("open_run_mask", np.zeros_like(dur_anchor_src)), dtype=np.int64)
    sealed_mask = np.asarray(trimmed.get("sealed_mask", np.ones_like(dur_anchor_src)), dtype=np.int64)
    boundary_confidence = np.asarray(
        trimmed.get("boundary_confidence", np.zeros_like(dur_anchor_src, dtype=np.float32)),
        dtype=np.float32,
    )
    trimmed.update(
        build_source_phrase_cache(
            dur_anchor_src=dur_anchor_src,
            sep_hint=sep_hint,
            open_run_mask=open_run_mask,
            sealed_mask=sealed_mask,
            boundary_confidence=boundary_confidence,
            phrase_boundary_threshold=float(ds.hparams.get("rhythm_source_phrase_threshold", 0.55)),
        )
    )

    trimmed_ref_len_sec = _resolve_ref_len_sec_from_duration_obs(ds, trimmed["dur_anchor_src"])
    trimmed.pop("dur_sec", None)
    if trimmed_ref_len_sec is None:
        trimmed.pop("ref_len_sec", None)
        trimmed.pop("prompt_ref_len_sec", None)
    else:
        trimmed["ref_len_sec"] = float(trimmed_ref_len_sec)
        trimmed["prompt_ref_len_sec"] = float(trimmed_ref_len_sec)
    return trimmed


def _build_counterfactual_conditioning(
    *,
    ds: ConanDataset,
    raw_ref_item: dict[str, Any],
    candidate_token: int,
    trim_head_runs: int,
    trim_tail_runs: int,
) -> dict[str, Any]:
    hubert = np.asarray(raw_ref_item["hubert"])
    cache = build_source_rhythm_cache_v3(
        hubert,
        silent_token=int(candidate_token),
        separator_aware=bool(ds.hparams.get("rhythm_separator_aware", True)),
        tail_open_units=int(ds.hparams.get("rhythm_tail_open_units", 1)),
        emit_silence_runs=bool(ds.hparams.get("rhythm_v3_emit_silence_runs", True)),
        debounce_min_run_frames=int(ds.hparams.get("rhythm_v3_debounce_min_run_frames", 2)),
        phrase_boundary_threshold=float(ds.hparams.get("rhythm_source_phrase_threshold", 0.55)),
        unit_prior_path=None,
        mel=raw_ref_item.get("mel"),
    )
    cache = _trim_counterfactual_cache_edges(
        ds=ds,
        cache=cache,
        trim_head_runs=trim_head_runs,
        trim_tail_runs=trim_tail_runs,
    )
    prompt_item = dict(raw_ref_item)
    prompt_item.update(cache)
    return ds._build_reference_prompt_unit_conditioning(
        prompt_item,
        target_mode=ds._resolve_rhythm_target_mode(),
    )


def _build_probe_sample(
    ds: ConanDataset,
    *,
    source_fetch_index: int,
    ref_local_index: int,
    candidate_token: int,
    trim_head_runs: int,
    trim_tail_runs: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    sample = copy.deepcopy(ds[source_fetch_index])
    raw_ref_item = ds._get_raw_item_cached(ref_local_index)
    ref_item = ds._materialize_rhythm_cache_compat(
        raw_ref_item,
        item_name=str(raw_ref_item.get("item_name", "<unknown-ref-item>")),
    )
    conditioning = _build_counterfactual_conditioning(
        ds=ds,
        raw_ref_item=raw_ref_item,
        candidate_token=candidate_token,
        trim_head_runs=trim_head_runs,
        trim_tail_runs=trim_tail_runs,
    )
    if not conditioning:
        raise RuntimeError(f"Failed to build counterfactual prompt conditioning for ref '{raw_ref_item.get('item_name')}'.")

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


def _conditioning_tempo(conditioning: dict[str, Any]) -> float:
    return compute_speech_tempo_for_analysis(
        source_duration_obs=conditioning.get("prompt_duration_obs"),
        source_speech_mask=conditioning.get("prompt_speech_mask"),
        source_valid_mask=conditioning.get("prompt_valid_mask"),
        source_unit_ids=conditioning.get("prompt_content_units"),
    )


def _conditioning_g(ds: ConanDataset, conditioning: dict[str, Any]) -> tuple[float, str]:
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
    candidate_token: int,
    trim_head_runs: int,
    trim_tail_runs: int,
) -> dict[str, Any]:
    try:
        sample, conditioning, raw_ref_item, _ = _build_probe_sample(
            ds,
            source_fetch_index=source_fetch_index,
            ref_local_index=ref_local_index,
            candidate_token=candidate_token,
            trim_head_runs=trim_head_runs,
            trim_tail_runs=trim_tail_runs,
        )
    except (RhythmDatasetPrefilterDrop, RuntimeError) as exc:
        raw_ref_item = ds._get_raw_item_cached(ref_local_index)
        return {
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
            "prompt_total_units": 0,
            "same_speaker_reference": int(source_name.split("_", 1)[0] == ref_name.split("_", 1)[0]),
            "same_text_reference": int(source_name == ref_name),
            "ref_pair_item_name": str(raw_ref_item.get("item_name", ref_name)),
            "counterfactual_silent_token": int(candidate_token),
            "counterfactual_clean_count": float("nan"),
            "counterfactual_domain_valid": 0.0,
            "trim_head_runs": int(trim_head_runs),
            "trim_tail_runs": int(trim_tail_runs),
            "src_prefix_stat_mode": str(ds.hparams.get("rhythm_v3_src_prefix_stat_mode", "ema")),
        }
    raw_source_item = sample.get("_raw_item")
    raw_paired_target_item = sample.get("_raw_paired_target_item")
    batch = ds.collater([sample])
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
        "counterfactual_silent_token": int(candidate_token),
    }
    if raw_source_item is not None:
        metadata["source_text_signature"] = ds._extract_text_signature(raw_source_item)
    if raw_ref_item is not None:
        metadata["reference_text_signature"] = ds._extract_text_signature(raw_ref_item)
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
            "prompt_tempo_ref": _conditioning_tempo(conditioning),
            "prompt_g_ref": float(prompt_g_ref),
            "prompt_g_status": str(prompt_g_status),
            "prompt_total_units": int(np.asarray(conditioning["prompt_duration_obs"]).reshape(-1).shape[0]),
            "same_speaker_reference": int(source_name.split("_", 1)[0] == ref_name.split("_", 1)[0]),
            "same_text_reference": int(source_name == ref_name),
            "ref_pair_item_name": str(raw_ref_item.get("item_name", ref_name)),
            "counterfactual_silent_token": int(candidate_token),
            "counterfactual_clean_count": float(summary.get("g_clean_count", float("nan"))),
            "counterfactual_domain_valid": float(summary.get("g_domain_valid", float("nan"))),
            "trim_head_runs": int(trim_head_runs),
            "trim_tail_runs": int(trim_tail_runs),
            "src_prefix_stat_mode": str(
                summary.get("src_prefix_stat_mode")
                or ds.hparams.get("rhythm_v3_src_prefix_stat_mode", "ema")
            ),
        }
    )
    return summary


def _monotonicity_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def _row_domain_valid(row: dict[str, Any]) -> bool:
        try:
            domain_valid = float(row.get("counterfactual_domain_valid", row.get("g_domain_valid", float("nan"))))
        except Exception:
            domain_valid = float("nan")
        if np.isfinite(domain_valid):
            return bool(domain_valid > 0.5)
        try:
            gate0_row_dropped = float(row.get("gate0_row_dropped", 0.0) or 0.0)
        except Exception:
            gate0_row_dropped = 0.0
        return bool(gate0_row_dropped <= 0.5)

    all_real_rows = [row for row in rows if row["ref_condition"] in {"slow", "mid", "fast"}]
    real_rows = [row for row in all_real_rows if _row_domain_valid(row)]

    def _ordered_control_summary(value_key: str, *, flip_sign: bool = False) -> dict[str, Any]:
        ordered_rows = sorted(
            real_rows,
            key=lambda row: -float(row[value_key]) if flip_sign else float(row[value_key]),
        )
        x_values = [(-float(row[value_key]) if flip_sign else float(row[value_key])) for row in ordered_rows]
        y_values = [float(row["tempo_out"]) for row in ordered_rows]
        monotone = len(ordered_rows) >= 3
        for left, right in zip(y_values, y_values[1:]):
            if not (np.isfinite(left) and np.isfinite(right) and right >= (left - 1.0e-6)):
                monotone = False
                break
        slope = float("nan")
        if len(ordered_rows) >= 2:
            x = np.asarray(x_values, dtype=np.float32)
            y = np.asarray(y_values, dtype=np.float32)
            if np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and float(np.max(x) - np.min(x)) > 1.0e-6:
                slope = float(np.polyfit(x, y, deg=1)[0])
        return {
            "sorted_ref_conditions": [str(row["ref_condition"]) for row in ordered_rows],
            "x_sorted": x_values,
            "tempo_out_sorted": y_values,
            "monotone": bool(monotone),
            "transfer_slope": slope,
        }

    prompt_control = _ordered_control_summary("prompt_g_ref", flip_sign=False)
    delta_g_control = _ordered_control_summary("delta_g", flip_sign=False)
    anti_control = _ordered_control_summary("delta_g", flip_sign=True)
    tempo_ref = list(prompt_control["x_sorted"])
    tempo_out = list(prompt_control["tempo_out_sorted"])
    monotone = bool(prompt_control["monotone"])
    slope = float(prompt_control["transfer_slope"])
    controls = {
        row["ref_condition"]: float(row["tempo_out"])
        for row in rows
        if row["ref_condition"] not in {"slow", "mid", "fast"}
    }
    real_range = float("nan")
    if tempo_out:
        tempo_out_np = np.asarray(tempo_out, dtype=np.float32)
        if np.all(np.isfinite(tempo_out_np)):
            real_range = float(np.max(tempo_out_np) - np.min(tempo_out_np))
            if len(real_rows) < 3 or real_range < float(DEFAULT_MIN_REAL_RANGE):
                real_range = float("nan")

    def _max_gap_to_control(control_name: str) -> float:
        control_value = controls.get(control_name)
        if control_value is None or not np.isfinite(control_value):
            return float("nan")
        real_values = np.asarray(tempo_out, dtype=np.float32)
        real_values = real_values[np.isfinite(real_values)]
        if real_values.size <= 0:
            return float("nan")
        return float(np.max(np.abs(real_values - float(control_value))))

    return {
        "source_name": rows[0]["source_name"] if rows else "",
        "src_prefix_stat_mode": rows[0].get("src_prefix_stat_mode", "") if rows else "",
        "counterfactual_silent_token": rows[0]["counterfactual_silent_token"] if rows else None,
        "trim_head_runs": rows[0]["trim_head_runs"] if rows else 0,
        "trim_tail_runs": rows[0]["trim_tail_runs"] if rows else 0,
        "total_real_row_count": len(all_real_rows),
        "valid_real_row_count": len(real_rows),
        "all_real_domain_valid": bool(len(real_rows) == len(all_real_rows) and len(all_real_rows) > 0),
        "g_ref_sorted": tempo_ref,
        "prompt_tempo_ref_sorted": [
            float(row.get("prompt_tempo_ref", float("nan")))
            for row in sorted(real_rows, key=lambda row: float(row.get("prompt_g_ref", float("nan"))))
        ],
        "monotone_by_prompt_g": bool(monotone),
        "monotone_by_prompt_tempo": bool(monotone),
        "transfer_slope": slope,
        "tempo_ref_sorted": tempo_ref,
        "tempo_out_sorted": tempo_out,
        "prompt_sorted_ref_conditions": prompt_control["sorted_ref_conditions"],
        "delta_g_sorted_ref_conditions": delta_g_control["sorted_ref_conditions"],
        "delta_g_sorted": delta_g_control["x_sorted"],
        "tempo_out_by_delta_g": delta_g_control["tempo_out_sorted"],
        "monotone_by_delta_g": bool(delta_g_control["monotone"]),
        "delta_g_transfer_slope": float(delta_g_control["transfer_slope"]),
        "anti_control_sorted_ref_conditions": anti_control["sorted_ref_conditions"],
        "neg_delta_g_sorted": anti_control["x_sorted"],
        "tempo_out_by_neg_delta_g": anti_control["tempo_out_sorted"],
        "monotone_by_neg_delta_g": bool(anti_control["monotone"]),
        "neg_delta_g_transfer_slope": float(anti_control["transfer_slope"]),
        "real_tempo_range": real_range,
        "max_gap_vs_source_only": _max_gap_to_control("source_only"),
        "max_gap_vs_random_ref": _max_gap_to_control("random_ref"),
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

        source_local = int(name_to_local[source_name])
        source_fetch_index = int(source_to_fetch[source_local])
        for ref_condition, ref_name in refs.items():
            ref_name = str(ref_name)
            rows.append(
                _run_condition(
                    ds=ds,
                    task=task,
                    source_name=source_name,
                    source_fetch_index=source_fetch_index,
                    ref_name=ref_name,
                    ref_local_index=int(name_to_local[ref_name]),
                    ref_condition=str(ref_condition),
                    candidate_token=int(args.candidate_token),
                    trim_head_runs=int(args.trim_head_runs),
                    trim_tail_runs=int(args.trim_tail_runs),
                )
            )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_name"])].append(row)
    summary_rows = [_monotonicity_summary(grouped[source_name]) for source_name in sorted(grouped.keys())]

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    _write_csv(output_csv, rows)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "candidate_token": int(args.candidate_token),
                "trim_head_runs": int(args.trim_head_runs),
                "trim_tail_runs": int(args.trim_tail_runs),
                "rows": rows,
                "summary": summary_rows,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"[gate1-counterfactual] split={args.split} candidate_token={int(args.candidate_token)} "
        f"trim_head_runs={int(args.trim_head_runs)} trim_tail_runs={int(args.trim_tail_runs)} rows={len(rows)}"
    )
    print(f"[gate1-counterfactual] wrote_csv={output_csv}")
    print(f"[gate1-counterfactual] wrote_json={output_json}")
    for row in summary_rows:
        monotone = "PASS" if row["monotone_by_prompt_tempo"] else "FAIL"
        slope = row["transfer_slope"]
        slope_str = "nan" if not np.isfinite(slope) else f"{slope:.4f}"
        print(
            f"[gate1-counterfactual] token={int(args.candidate_token)} {row['source_name']} "
            f"trim=({int(row['trim_head_runs'])},{int(row['trim_tail_runs'])}) monotone={monotone} "
            f"slope={slope_str} real_range={row['real_tempo_range']:.4f} "
            f"gap_source_only={row['max_gap_vs_source_only']:.4f} gap_random_ref={row['max_gap_vs_random_ref']:.4f} "
            f"tempo_ref={row['tempo_ref_sorted']} tempo_out={row['tempo_out_sorted']}"
        )
        neg_delta_slope = row["neg_delta_g_transfer_slope"]
        neg_delta_slope_str = "nan" if not np.isfinite(neg_delta_slope) else f"{neg_delta_slope:.4f}"
        print(
            f"[gate1-counterfactual] {row['source_name']} "
            f"delta_g_order={row['delta_g_sorted_ref_conditions']} delta_g={row['delta_g_sorted']} "
            f"tempo_by_delta={row['tempo_out_by_delta_g']} mono_by_delta={'PASS' if row['monotone_by_delta_g'] else 'FAIL'} "
            f"anti_order={row['anti_control_sorted_ref_conditions']} neg_delta={row['neg_delta_g_sorted']} "
            f"tempo_by_neg_delta={row['tempo_out_by_neg_delta_g']} mono_by_neg_delta={'PASS' if row['monotone_by_neg_delta_g'] else 'FAIL'} "
            f"neg_delta_slope={neg_delta_slope_str}"
        )
        if row["controls"]:
            print(f"[gate1-counterfactual] {row['source_name']} controls={row['controls']}")


if __name__ == "__main__":
    main()
