from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
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
from utils.plot.rhythm_v3_viz.review import compute_source_global_rate_for_analysis


DEFAULT_CONFIG = "egs/local_arctic_rhythm_v3_quick.yaml"
DEFAULT_SPLITS = "train,valid,test"
DEFAULT_OUTPUT_CSV = "tmp/gate1_boundary_audit/silent_token_sweep_rows.csv"
DEFAULT_OUTPUT_JSON = "tmp/gate1_boundary_audit/silent_token_sweep_report.json"
DEFAULT_TOPK = 16
DEFAULT_RELAXED_HPARAMS = (
    "use_pitch_embed=False,"
    "rhythm_v3_gate_quality_strict=False,"
    "rhythm_v3_minimal_v1_profile=False,"
    "rhythm_v3_strict_minimal_claim_profile=False"
)
DEFAULT_CASES = [
    {
        "speaker": "aba",
        "source": "aba_train_arctic_a0010",
        "refs": {
            "slow": "aba_train_arctic_a0012",
            "mid": "aba_train_arctic_a0015",
            "fast": "aba_train_arctic_a0016",
        },
    },
    {
        "speaker": "asi",
        "source": "asi_train_arctic_a0011",
        "refs": {
            "slow": "asi_train_arctic_a0015",
            "mid": "asi_train_arctic_a0014",
            "fast": "asi_train_arctic_a0013",
        },
    },
    {
        "speaker": "bdl",
        "source": "bdl_train_arctic_a0014",
        "refs": {
            "slow": "bdl_train_arctic_a0012",
            "mid": "bdl_train_arctic_a0013",
            "fast": "bdl_train_arctic_a0016",
        },
    },
    {
        "speaker": "slt",
        "source": "slt_train_arctic_a0014",
        "refs": {
            "slow": "slt_train_arctic_a0010",
            "mid": "slt_train_arctic_a0013",
            "fast": "slt_train_arctic_a0016",
        },
    },
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Counterfactual silent-token sweep for rhythm_v3 boundary-support falsification."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Config YAML.")
    parser.add_argument("--splits", default=DEFAULT_SPLITS, help="Comma-separated dataset splits to audit.")
    parser.add_argument(
        "--candidate_tokens",
        default="",
        help="Optional comma-separated token ids. Defaults to top-k frequent raw tokens plus configured silent_token.",
    )
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK, help="Top-k frequent tokens to include when candidate_tokens is empty.")
    parser.add_argument("--binary_data_dir", default="", help="Optional binary_data_dir override.")
    parser.add_argument("--processed_data_dir", default="", help="Optional processed_data_dir override.")
    parser.add_argument("--hparams", default="", help="Extra hparam overrides appended after relaxed audit defaults.")
    parser.add_argument("--output_csv", default=DEFAULT_OUTPUT_CSV, help="Where to write flattened candidate rows.")
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


def _parse_splits(text: str) -> list[str]:
    splits = [part.strip() for part in str(text).split(",") if part.strip()]
    if not splits:
        raise ValueError("At least one split is required.")
    return splits


def _safe_item_tokens(value: Any) -> list[int]:
    if isinstance(value, str):
        return [int(float(token)) for token in value.split() if str(token).strip()]
    arr = np.asarray(value).reshape(-1)
    return [int(token) for token in arr.tolist()]


def _build_candidate_tokens(items: list[dict[str, Any]], *, configured_silent_token: int, topk: int, explicit: str) -> list[int]:
    if explicit.strip():
        out: list[int] = []
        seen: set[int] = set()
        for token in explicit.split(","):
            value = int(token.strip())
            if value in seen:
                continue
            seen.add(value)
            out.append(value)
        return out
    counter: Counter[int] = Counter()
    for item in items:
        counter.update(_safe_item_tokens(item["hubert"]))
    candidates = [int(token) for token, _ in counter.most_common(max(1, int(topk)))]
    if int(configured_silent_token) not in candidates:
        candidates.append(int(configured_silent_token))
    return candidates


def _scalar_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    arr = np.asarray(value)
    if arr.size <= 0:
        return float("nan")
    return float(arr.reshape(-1)[0])


def _build_custom_prompt_item(
    *,
    raw_item: dict[str, Any],
    cache: dict[str, np.ndarray],
) -> dict[str, Any]:
    prompt_item = dict(raw_item)
    prompt_item.update(cache)
    return prompt_item


def _audit_candidate_on_item(
    *,
    ds: ConanDataset,
    raw_item: dict[str, Any],
    candidate_token: int,
    drop_edge_runs: int,
    min_boundary_confidence: float,
    min_prompt_speech_ratio: float,
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
    prompt_item = _build_custom_prompt_item(raw_item=raw_item, cache=cache)
    conditioning = ds._build_reference_prompt_unit_conditioning(
        prompt_item,
        target_mode=ds._resolve_rhythm_target_mode(),
    )
    prompt_duration_obs = np.asarray(conditioning["prompt_duration_obs"], dtype=np.float32).reshape(-1)
    prompt_valid_mask = np.asarray(conditioning["prompt_valid_mask"], dtype=np.float32).reshape(-1)
    prompt_speech_mask = np.asarray(conditioning["prompt_speech_mask"], dtype=np.float32).reshape(-1)
    prompt_closed_mask = np.asarray(conditioning["prompt_closed_mask"], dtype=np.float32).reshape(-1)
    prompt_boundary_confidence = np.asarray(conditioning["prompt_boundary_confidence"], dtype=np.float32).reshape(-1)
    prompt_ref_len_sec = _scalar_or_nan(conditioning.get("prompt_ref_len_sec"))
    if not np.isfinite(prompt_ref_len_sec):
        prompt_ref_len_sec = _scalar_or_nan(raw_item.get("duration"))
    prompt_speech_ratio = _scalar_or_nan(conditioning.get("prompt_speech_ratio_scalar"))
    stats = summarize_global_rate_support(
        speech_mask=torch.as_tensor(prompt_speech_mask, dtype=torch.float32),
        valid_mask=torch.as_tensor(prompt_valid_mask, dtype=torch.float32),
        duration_obs=torch.as_tensor(prompt_duration_obs, dtype=torch.float32),
        drop_edge_runs=drop_edge_runs,
        min_speech_ratio=min_prompt_speech_ratio,
        closed_mask=torch.as_tensor(prompt_closed_mask, dtype=torch.float32),
        boundary_confidence=torch.as_tensor(prompt_boundary_confidence, dtype=torch.float32),
        min_boundary_confidence=min_boundary_confidence,
    )
    support_count = float(stats.support_count.reshape(-1)[0].item())
    clean_count = float(stats.clean_count.reshape(-1)[0].item())
    support_domain_valid = float(stats.domain_valid.reshape(-1)[0].item())
    g_ref = compute_source_global_rate_for_analysis(
        source_duration_obs=prompt_duration_obs,
        source_speech_mask=prompt_speech_mask,
        source_valid_mask=prompt_valid_mask,
        g_variant=str(ds.hparams.get("rhythm_v3_g_variant", "raw_median")),
        g_trim_ratio=float(ds.hparams.get("rhythm_v3_g_trim_ratio", 0.2) or 0.2),
        drop_edge_runs=drop_edge_runs,
        source_unit_ids=np.asarray(conditioning["prompt_content_units"], dtype=np.int64).reshape(-1),
        source_closed_mask=prompt_closed_mask,
        source_boundary_confidence=prompt_boundary_confidence,
        min_boundary_confidence=min_boundary_confidence,
    )
    return {
        "item_name": item_name,
        "speaker": str(raw_item.get("speaker", item_name.split("_", 1)[0])),
        "split": str(raw_item.get("split", "")),
        "candidate_token": int(candidate_token),
        "raw_token_count": int(np.asarray(hubert).reshape(-1).shape[0]),
        "raw_candidate_count": int(sum(1 for token in _safe_item_tokens(raw_item["hubert"]) if token == int(candidate_token))),
        "sep_nonzero_count": int((np.asarray(cache["sep_hint"]).reshape(-1) > 0).sum()),
        "source_silence_run_count": int((np.asarray(cache["source_silence_mask"]).reshape(-1) > 0.5).sum()),
        "support_count": support_count,
        "clean_count": clean_count,
        "support_domain_valid": support_domain_valid,
        "prompt_ref_len_sec": float(prompt_ref_len_sec),
        "prompt_speech_ratio": float(prompt_speech_ratio),
        "g_ref": float(g_ref) if np.isfinite(g_ref) else float("nan"),
        "bc_max": float(np.nanmax(prompt_boundary_confidence)) if prompt_boundary_confidence.size > 0 else float("nan"),
    }


def _run_case_g_order(
    *,
    ds: ConanDataset,
    item_map: dict[str, dict[str, Any]],
    candidate_token: int,
    drop_edge_runs: int,
    min_boundary_confidence: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in DEFAULT_CASES:
        source_name = str(case["source"])
        for ref_condition, ref_name in case["refs"].items():
            raw_item = item_map.get(ref_name)
            if raw_item is None:
                continue
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
            prompt_item = _build_custom_prompt_item(raw_item=raw_item, cache=cache)
            conditioning = ds._build_reference_prompt_unit_conditioning(
                prompt_item,
                target_mode=ds._resolve_rhythm_target_mode(),
            )
            prompt_duration_obs = np.asarray(conditioning["prompt_duration_obs"], dtype=np.float32).reshape(-1)
            prompt_valid_mask = np.asarray(conditioning["prompt_valid_mask"], dtype=np.float32).reshape(-1)
            prompt_speech_mask = np.asarray(conditioning["prompt_speech_mask"], dtype=np.float32).reshape(-1)
            prompt_closed_mask = np.asarray(conditioning["prompt_closed_mask"], dtype=np.float32).reshape(-1)
            prompt_boundary_confidence = np.asarray(conditioning["prompt_boundary_confidence"], dtype=np.float32).reshape(-1)
            g_ref = compute_source_global_rate_for_analysis(
                source_duration_obs=prompt_duration_obs,
                source_speech_mask=prompt_speech_mask,
                source_valid_mask=prompt_valid_mask,
                g_variant=str(ds.hparams.get("rhythm_v3_g_variant", "raw_median")),
                g_trim_ratio=float(ds.hparams.get("rhythm_v3_g_trim_ratio", 0.2) or 0.2),
                drop_edge_runs=drop_edge_runs,
                source_unit_ids=np.asarray(conditioning["prompt_content_units"], dtype=np.int64).reshape(-1),
                source_closed_mask=prompt_closed_mask,
                source_boundary_confidence=prompt_boundary_confidence,
                min_boundary_confidence=min_boundary_confidence,
            )
            rows.append(
                {
                    "candidate_token": int(candidate_token),
                    "source_name": source_name,
                    "ref_condition": str(ref_condition),
                    "ref_name": ref_name,
                    "g_ref": float(g_ref) if np.isfinite(g_ref) else float("nan"),
                    "sep_nonzero_count": int((np.asarray(cache["sep_hint"]).reshape(-1) > 0).sum()),
                    "clean_count": float(
                        summarize_global_rate_support(
                            speech_mask=torch.as_tensor(prompt_speech_mask, dtype=torch.float32),
                            valid_mask=torch.as_tensor(prompt_valid_mask, dtype=torch.float32),
                            duration_obs=torch.as_tensor(prompt_duration_obs, dtype=torch.float32),
                            drop_edge_runs=drop_edge_runs,
                            closed_mask=torch.as_tensor(prompt_closed_mask, dtype=torch.float32),
                            boundary_confidence=torch.as_tensor(prompt_boundary_confidence, dtype=torch.float32),
                            min_boundary_confidence=min_boundary_confidence,
                        ).clean_count.reshape(-1)[0].item()
                    ),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summarize_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate = int(rows[0]["candidate_token"])
    return {
        "candidate_token": candidate,
        "item_count": len(rows),
        "raw_candidate_items": int(sum(1 for row in rows if int(row["raw_candidate_count"]) > 0)),
        "sep_nonzero_items": int(sum(1 for row in rows if int(row["sep_nonzero_count"]) > 0)),
        "silence_run_items": int(sum(1 for row in rows if int(row["source_silence_run_count"]) > 0)),
        "clean_positive_items": int(sum(1 for row in rows if float(row["clean_count"]) > 0.5)),
        "support_positive_items": int(sum(1 for row in rows if float(row["support_count"]) > 0.5)),
        "domain_valid_items": int(sum(1 for row in rows if float(row["support_domain_valid"]) > 0.5)),
        "mean_sep_nonzero_count": float(np.mean([float(row["sep_nonzero_count"]) for row in rows])),
        "mean_bc_max": float(np.mean([float(row["bc_max"]) for row in rows])),
        "mean_g_ref": float(np.nanmean(np.asarray([row["g_ref"] for row in rows], dtype=np.float32))),
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
    datasets = {split: ConanDataset(prefix=split, shuffle=False) for split in splits}
    metadata_path = Path(str(next(iter(datasets.values())).hparams["processed_data_dir"])) / "metadata.json"
    metadata_items = json.loads(metadata_path.read_text(encoding="utf-8"))
    configured_silent_token = int(next(iter(datasets.values())).hparams.get("silent_token", 57))
    candidates = _build_candidate_tokens(
        metadata_items,
        configured_silent_token=configured_silent_token,
        topk=args.topk,
        explicit=args.candidate_tokens,
    )

    ds0 = next(iter(datasets.values()))
    drop_edge_runs = int(ds0.hparams.get("rhythm_v3_drop_edge_runs_for_g", 0) or 0)
    min_boundary_confidence = float(ds0.hparams.get("rhythm_v3_min_boundary_confidence_for_g", 0.5) or 0.0)
    min_prompt_speech_ratio = float(ds0.hparams.get("rhythm_v3_min_prompt_speech_ratio", 0.6) or 0.0)

    rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        for split, ds in datasets.items():
            del split
            for local_idx in range(len(ds.avail_idxs)):
                raw_item = ds._get_raw_item_cached(local_idx)
                rows.append(
                    _audit_candidate_on_item(
                        ds=ds,
                        raw_item=raw_item,
                        candidate_token=int(candidate),
                        drop_edge_runs=drop_edge_runs,
                        min_boundary_confidence=min_boundary_confidence,
                        min_prompt_speech_ratio=min_prompt_speech_ratio,
                    )
                )
        train_ds = datasets.get("train")
        if train_ds is not None:
            item_map = {
                str(train_ds._get_raw_item_cached(local_idx)["item_name"]): train_ds._get_raw_item_cached(local_idx)
                for local_idx in range(len(train_ds.avail_idxs))
            }
            case_rows.extend(
                _run_case_g_order(
                    ds=train_ds,
                    item_map=item_map,
                    candidate_token=int(candidate),
                    drop_edge_runs=drop_edge_runs,
                    min_boundary_confidence=min_boundary_confidence,
                )
            )

    summary_rows = [
        _summarize_candidate([row for row in rows if int(row["candidate_token"]) == int(candidate)])
        for candidate in candidates
    ]
    summary_rows = sorted(
        summary_rows,
        key=lambda row: (
            -int(row["domain_valid_items"]),
            -int(row["clean_positive_items"]),
            -int(row["sep_nonzero_items"]),
            -int(row["raw_candidate_items"]),
        ),
    )

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    _write_csv(output_csv, rows)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": {
                    "config": args.config,
                    "metadata_path": str(metadata_path),
                    "configured_silent_token": configured_silent_token,
                    "candidate_tokens": candidates,
                    "counterfactual_emit_silence_runs": False,
                    "min_boundary_confidence_for_g": min_boundary_confidence,
                    "drop_edge_runs_for_g": drop_edge_runs,
                },
                "summary": summary_rows,
                "rows": rows,
                "case_rows": case_rows,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"[silent-sweep] candidates={len(candidates)} configured_silent_token={configured_silent_token} "
        f"threshold={min_boundary_confidence:.2f}"
    )
    print(f"[silent-sweep] wrote_csv={output_csv}")
    print(f"[silent-sweep] wrote_json={output_json}")
    for row in summary_rows[: min(10, len(summary_rows))]:
        print(
            f"[silent-sweep] token={row['candidate_token']} raw_items={row['raw_candidate_items']} "
            f"sep_items={row['sep_nonzero_items']} clean_items={row['clean_positive_items']} "
            f"domain_items={row['domain_valid_items']} mean_bc_max={row['mean_bc_max']:.4f}"
        )


if __name__ == "__main__":
    main()
