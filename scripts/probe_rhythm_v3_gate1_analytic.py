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
from utils.plot.rhythm_v3_viz.review import compute_speech_tempo_for_analysis


DEFAULT_CONFIG = "egs/local_arctic_rhythm_v3_quick_gate1.yaml"
DEFAULT_SPLIT = "train"
DEFAULT_OUTPUT_CSV = "tmp/gate1_analytic_probe/results.csv"
DEFAULT_OUTPUT_JSON = "tmp/gate1_analytic_probe/summary.json"
DEFAULT_MIN_REAL_RANGE = 0.01
DEFAULT_RELAXED_HPARAMS = (
    "use_pitch_embed=False,"
    "rhythm_v3_gate_quality_strict=False,"
    "rhythm_v3_minimal_v1_profile=False,"
    "rhythm_v3_strict_minimal_claim_profile=False,"
    "rhythm_v3_alignment_prefilter_bad_samples=False,"
    "rhythm_v3_alignment_local_margin_p10_min=0.0,"
    "rhythm_v3_alignment_mean_local_confidence_speech_min=0.0,"
    "rhythm_v3_alignment_mean_coarse_confidence_speech_min=0.0,"
    "rhythm_v3_disallow_same_text_reference=False"
)


DEFAULT_CASES = [
    {
        "speaker": "aba",
        "source": "aba_train_arctic_a0010",
        "refs": {
            "slow": "aba_train_arctic_a0012",
            "mid": "aba_train_arctic_a0015",
            "fast": "aba_train_arctic_a0016",
            "random_ref": "bdl_train_arctic_a0012",
        },
    },
    {
        "speaker": "asi",
        "source": "asi_train_arctic_a0011",
        "refs": {
            "slow": "asi_train_arctic_a0015",
            "mid": "asi_train_arctic_a0014",
            "fast": "asi_train_arctic_a0013",
            "random_ref": "bdl_train_arctic_a0012",
        },
    },
    {
        "speaker": "bdl",
        "source": "bdl_train_arctic_a0014",
        "refs": {
            "slow": "bdl_train_arctic_a0012",
            "mid": "bdl_train_arctic_a0013",
            "fast": "bdl_train_arctic_a0016",
            "random_ref": "aba_train_arctic_a0012",
        },
    },
    {
        "speaker": "slt",
        "source": "slt_train_arctic_a0014",
        "refs": {
            "slow": "slt_train_arctic_a0010",
            "mid": "slt_train_arctic_a0013",
            "fast": "slt_train_arctic_a0016",
            "random_ref": "aba_train_arctic_a0012",
        },
    },
]


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


def _load_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.cases_json:
        return copy.deepcopy(DEFAULT_CASES)
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


def _conditioning_tempo(conditioning: dict[str, np.ndarray]) -> float:
    return compute_speech_tempo_for_analysis(
        source_duration_obs=conditioning.get("prompt_duration_obs"),
        source_speech_mask=conditioning.get("prompt_speech_mask"),
        source_valid_mask=conditioning.get("prompt_valid_mask"),
        source_unit_ids=conditioning.get("prompt_content_units"),
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
    except RhythmDatasetPrefilterDrop as exc:
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
            "prompt_total_units": int(np.asarray(raw_ref_item.get("dur_anchor_src", [])).reshape(-1).shape[0]),
            "same_speaker_reference": int(source_name.split("_", 1)[0] == ref_name.split("_", 1)[0]),
            "same_text_reference": int(source_name == ref_name),
            "ref_pair_item_name": str(raw_ref_item.get("item_name", ref_name)),
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
    summary.update(
        {
            "source_name": source_name,
            "ref_name": ref_name,
            "ref_condition": ref_condition,
            "prompt_tempo_ref": _conditioning_tempo(conditioning),
            "prompt_total_units": int(np.asarray(conditioning["prompt_duration_obs"]).reshape(-1).shape[0]),
            "same_speaker_reference": int(source_name.split("_", 1)[0] == ref_name.split("_", 1)[0]),
            "same_text_reference": int(source_name == ref_name),
            "ref_pair_item_name": str(raw_ref_item.get("item_name", ref_name)),
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


def _monotonicity_summary(rows: list[dict[str, Any]], *, min_real_range: float) -> dict[str, Any]:
    all_real_rows = [row for row in rows if row["ref_condition"] in {"slow", "mid", "fast"}]
    real_rows = [row for row in all_real_rows if _row_domain_valid(row)]
    real_rows = sorted(real_rows, key=lambda row: float(row["prompt_tempo_ref"]))
    tempo_ref = [float(row["prompt_tempo_ref"]) for row in real_rows]
    tempo_out = [float(row["tempo_out"]) for row in real_rows]
    monotone = len(real_rows) >= 3
    for left, right in zip(tempo_out, tempo_out[1:]):
        if not (np.isfinite(left) and np.isfinite(right) and right >= (left - 1.0e-6)):
            monotone = False
            break
    slope = float("nan")
    if len(real_rows) >= 2:
        x = np.asarray(tempo_ref, dtype=np.float32)
        y = np.asarray(tempo_out, dtype=np.float32)
        if np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and float(np.max(x) - np.min(x)) > 1.0e-6:
            slope = float(np.polyfit(x, y, deg=1)[0])
    real_range = float("nan")
    if tempo_out:
        y = np.asarray(tempo_out, dtype=np.float32)
        if np.all(np.isfinite(y)):
            real_range = float(np.max(y) - np.min(y))
    controls = {
        row["ref_condition"]: float(row["tempo_out"])
        for row in rows
        if row["ref_condition"] not in {"slow", "mid", "fast"}
    }
    prompt_control_response = bool(
        monotone
        and np.isfinite(slope)
        and slope > 0.0
        and np.isfinite(real_range)
        and real_range >= float(min_real_range)
    )
    return {
        "source_name": rows[0]["source_name"] if rows else "",
        "total_real_row_count": len(all_real_rows),
        "valid_real_row_count": len(real_rows),
        "all_real_domain_valid": bool(len(real_rows) == len(all_real_rows) and len(all_real_rows) > 0),
        "order_monotone_by_prompt_tempo": bool(monotone),
        "monotone_by_prompt_tempo": prompt_control_response,
        "transfer_slope": slope,
        "real_tempo_range": real_range,
        "tempo_ref_sorted": tempo_ref,
        "tempo_out_sorted": tempo_out,
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
    cases = _load_cases(args)
    ds = ConanDataset(prefix=args.split, shuffle=False)
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
        print(
            f"[gate1-probe] {source_name} monotone={monotone} "
            f"slope={slope_str} range={row['real_tempo_range']:.4f} "
            f"valid_real={row['valid_real_row_count']}/{row['total_real_row_count']} "
            f"tempo_ref={row['tempo_ref_sorted']} tempo_out={row['tempo_out_sorted']}"
        )
        if row["controls"]:
            print(f"[gate1-probe] {source_name} controls={row['controls']}")


if __name__ == "__main__":
    main()
