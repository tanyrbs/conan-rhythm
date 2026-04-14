from __future__ import annotations

import argparse
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

from modules.Conan.rhythm_v3.g_stats import summarize_global_rate_support
from tasks.Conan.dataset import ConanDataset
from utils.commons.hparams import set_hparams

from scripts.probe_rhythm_v3_gate1_silent_counterfactual import (
    _build_counterfactual_conditioning,
    _compose_hparams_override,
    _item_name_to_local_index,
)
from utils.plot.rhythm_v3_viz.review import compute_speech_tempo_for_analysis


DEFAULT_CONFIG = "egs/local_arctic_rhythm_v3_quick_gate1.yaml"
DEFAULT_SPLIT = "train"
DEFAULT_OUTPUT_JSON = "tmp/gate1_counterfactual_probe/full_sweep/cases.json"
DEFAULT_RANDOM_REF_STRATEGY = "round_robin"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate slow/mid/fast/random cases for counterfactual Gate-1 runtime sweeps."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Config YAML.")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Dataset split.")
    parser.add_argument("--candidate_token", type=int, required=True, help="Counterfactual silent token id used to score prompt refs.")
    parser.add_argument("--trim_head_runs", type=int, default=0, help="Optional prompt-cache head trim, matching the runtime probe.")
    parser.add_argument("--trim_tail_runs", type=int, default=0, help="Optional prompt-cache tail trim, matching the runtime probe.")
    parser.add_argument("--drop_edge_runs", type=int, default=-1, help="Optional drop_edge_runs_for_g override for prompt-domain validity checks. Defaults to config value.")
    parser.add_argument("--binary_data_dir", default="", help="Optional binary_data_dir override.")
    parser.add_argument("--processed_data_dir", default="", help="Optional processed_data_dir override.")
    parser.add_argument("--hparams", default="", help="Extra hparam overrides appended after relaxed defaults.")
    parser.add_argument("--random_ref_strategy", default=DEFAULT_RANDOM_REF_STRATEGY, choices=["round_robin"], help="How to assign random_ref from another speaker.")
    parser.add_argument("--output_json", default=DEFAULT_OUTPUT_JSON, help="Where to write the generated cases JSON.")
    return parser.parse_args()


def _speaker_from_name(item_name: str) -> str:
    return str(item_name).split("_", 1)[0].strip().lower()


def _prompt_stats(
    *,
    ds: ConanDataset,
    raw_ref_item: dict[str, Any],
    candidate_token: int,
    trim_head_runs: int,
    trim_tail_runs: int,
    drop_edge_runs: int,
) -> dict[str, Any]:
    conditioning = _build_counterfactual_conditioning(
        ds=ds,
        raw_ref_item=raw_ref_item,
        candidate_token=int(candidate_token),
        trim_head_runs=int(trim_head_runs),
        trim_tail_runs=int(trim_tail_runs),
    )
    prompt_duration_obs = np.asarray(conditioning["prompt_duration_obs"], dtype=np.float32).reshape(-1)
    prompt_valid_mask = np.asarray(
        conditioning.get("prompt_valid_mask", conditioning.get("prompt_unit_mask")),
        dtype=np.float32,
    ).reshape(-1)
    prompt_speech_mask = np.asarray(conditioning["prompt_speech_mask"], dtype=np.float32).reshape(-1)
    prompt_closed_mask = np.asarray(conditioning["prompt_closed_mask"], dtype=np.float32).reshape(-1)
    prompt_boundary_confidence = np.asarray(conditioning["prompt_boundary_confidence"], dtype=np.float32).reshape(-1)
    min_boundary_confidence = ds.hparams.get("rhythm_v3_min_boundary_confidence_for_g", None)
    if min_boundary_confidence is not None:
        min_boundary_confidence = float(min_boundary_confidence)
    support_stats = summarize_global_rate_support(
        speech_mask=torch.as_tensor(prompt_speech_mask, dtype=torch.float32),
        valid_mask=torch.as_tensor(prompt_valid_mask, dtype=torch.float32),
        duration_obs=torch.as_tensor(prompt_duration_obs, dtype=torch.float32),
        drop_edge_runs=int(drop_edge_runs),
        min_speech_ratio=float(ds.hparams.get("rhythm_v3_min_prompt_speech_ratio", 0.6) or 0.6),
        closed_mask=torch.as_tensor(prompt_closed_mask, dtype=torch.float32),
        boundary_confidence=torch.as_tensor(prompt_boundary_confidence, dtype=torch.float32),
        min_boundary_confidence=min_boundary_confidence,
    )
    prompt_tempo = compute_speech_tempo_for_analysis(
        source_duration_obs=conditioning.get("prompt_duration_obs"),
        source_speech_mask=conditioning.get("prompt_speech_mask"),
        source_valid_mask=conditioning.get("prompt_valid_mask"),
        source_unit_ids=conditioning.get("prompt_content_units"),
    )
    return {
        "prompt_tempo_ref": float(prompt_tempo),
        "g_domain_valid": float(support_stats.domain_valid.reshape(-1)[0].item()),
        "g_clean_count": float(support_stats.clean_count.reshape(-1)[0].item()),
        "prompt_total_units": int(prompt_duration_obs.shape[0]),
    }


def _choose_triplet(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["prompt_tempo_ref"]),
            -float(row["g_clean_count"]),
            str(row["ref_name"]),
        ),
    )
    if len(ordered) < 3:
        raise ValueError(f"Need at least 3 refs to choose slow/mid/fast, got {len(ordered)}.")
    slow = ordered[0]
    fast = ordered[-1]
    mid = ordered[len(ordered) // 2]
    if slow["ref_name"] == mid["ref_name"] or fast["ref_name"] == mid["ref_name"] or slow["ref_name"] == fast["ref_name"]:
        distinct: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in ordered:
            ref_name = str(row["ref_name"])
            if ref_name in seen:
                continue
            seen.add(ref_name)
            distinct.append(row)
        if len(distinct) < 3:
            raise ValueError(f"Need 3 distinct refs after dedupe, got {len(distinct)}.")
        slow = distinct[0]
        fast = distinct[-1]
        mid = distinct[len(distinct) // 2]
    return slow, mid, fast


def _pick_random_ref(
    *,
    all_names_by_speaker: dict[str, list[str]],
    source_speaker: str,
    blocked_names: set[str],
    source_position: int,
) -> str:
    other_speakers = [speaker for speaker in sorted(all_names_by_speaker.keys()) if speaker != source_speaker]
    if not other_speakers:
        raise ValueError(f"No alternate speaker available for source speaker '{source_speaker}'.")
    speaker = other_speakers[source_position % len(other_speakers)]
    pool = [name for name in all_names_by_speaker[speaker] if name not in blocked_names]
    if not pool:
        raise ValueError(f"No random_ref candidate left for speaker '{speaker}'.")
    return str(pool[source_position % len(pool)])


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
    name_to_local = _item_name_to_local_index(ds)
    drop_edge_runs = (
        int(args.drop_edge_runs)
        if int(args.drop_edge_runs) >= 0
        else int(ds.hparams.get("rhythm_v3_drop_edge_runs_for_g", 1) or 1)
    )

    items_by_speaker: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for item_name, local_idx in sorted(name_to_local.items()):
        items_by_speaker[_speaker_from_name(item_name)].append((str(item_name), int(local_idx)))

    cases: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for speaker, items in sorted(items_by_speaker.items()):
        item_names_sorted = [name for name, _ in sorted(items)]
        for source_position, (source_name, source_local) in enumerate(sorted(items)):
            ref_rows: list[dict[str, Any]] = []
            for ref_name, ref_local in sorted(items):
                if str(ref_name) == str(source_name):
                    continue
                raw_ref_item = ds._get_raw_item_cached(int(ref_local))
                stats = _prompt_stats(
                    ds=ds,
                    raw_ref_item=raw_ref_item,
                    candidate_token=int(args.candidate_token),
                    trim_head_runs=int(args.trim_head_runs),
                    trim_tail_runs=int(args.trim_tail_runs),
                    drop_edge_runs=int(drop_edge_runs),
                )
                ref_rows.append(
                    {
                        "source_name": str(source_name),
                        "source_local": int(source_local),
                        "speaker": str(speaker),
                        "ref_name": str(ref_name),
                        "ref_local": int(ref_local),
                        **stats,
                    }
                )
            valid_rows = [row for row in ref_rows if float(row["g_domain_valid"]) > 0.5 and np.isfinite(float(row["prompt_tempo_ref"]))]
            chosen_pool = valid_rows if len(valid_rows) >= 3 else [row for row in ref_rows if np.isfinite(float(row["prompt_tempo_ref"]))]
            slow, mid, fast = _choose_triplet(chosen_pool)
            blocked = {str(source_name), str(slow["ref_name"]), str(mid["ref_name"]), str(fast["ref_name"])}
            random_ref = _pick_random_ref(
                all_names_by_speaker={k: [name for name, _ in sorted(v)] for k, v in items_by_speaker.items()},
                source_speaker=str(speaker),
                blocked_names=blocked,
                source_position=int(source_position),
            )
            case = {
                "speaker": str(speaker),
                "source": str(source_name),
                "refs": {
                    "slow": str(slow["ref_name"]),
                    "mid": str(mid["ref_name"]),
                    "fast": str(fast["ref_name"]),
                    "random_ref": str(random_ref),
                    "source_only": str(source_name),
                },
            }
            cases.append(case)
            coverage_rows.append(
                {
                    "speaker": str(speaker),
                    "source": str(source_name),
                    "same_speaker_candidate_count": len(ref_rows),
                    "valid_same_speaker_candidate_count": len(valid_rows),
                    "selected_pool_kind": "valid_only" if len(valid_rows) >= 3 else "tempo_fallback",
                    "slow_ref": str(slow["ref_name"]),
                    "slow_prompt_tempo_ref": float(slow["prompt_tempo_ref"]),
                    "slow_clean_count": float(slow["g_clean_count"]),
                    "mid_ref": str(mid["ref_name"]),
                    "mid_prompt_tempo_ref": float(mid["prompt_tempo_ref"]),
                    "mid_clean_count": float(mid["g_clean_count"]),
                    "fast_ref": str(fast["ref_name"]),
                    "fast_prompt_tempo_ref": float(fast["prompt_tempo_ref"]),
                    "fast_clean_count": float(fast["g_clean_count"]),
                    "random_ref": str(random_ref),
                }
            )

    payload = {
        "split": str(args.split),
        "candidate_token": int(args.candidate_token),
        "trim_head_runs": int(args.trim_head_runs),
        "trim_tail_runs": int(args.trim_tail_runs),
        "drop_edge_runs": int(drop_edge_runs),
        "case_count": len(cases),
        "coverage": coverage_rows,
        "cases": cases,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[counterfactual-casegen] split={args.split} candidate_token={int(args.candidate_token)} "
        f"trim_head_runs={int(args.trim_head_runs)} trim_tail_runs={int(args.trim_tail_runs)} "
        f"drop_edge_runs={int(drop_edge_runs)} cases={len(cases)}"
    )
    print(f"[counterfactual-casegen] wrote_json={output_json}")
    if coverage_rows:
        valid_counts = np.asarray([row["valid_same_speaker_candidate_count"] for row in coverage_rows], dtype=np.int64)
        print(
            "[counterfactual-casegen] "
            f"valid_same_speaker_refs min={int(valid_counts.min())} "
            f"median={int(np.median(valid_counts))} max={int(valid_counts.max())}"
        )


if __name__ == "__main__":
    main()
