from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any

import numpy as np

from tasks.Conan.dataset import ConanDataset
from utils.plot.rhythm_v3_viz.review import compute_source_global_rate_for_analysis


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


def _collect_valid_prompt_rows(ds: ConanDataset) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target_mode = ds._resolve_rhythm_target_mode()
    for local_idx in range(len(ds.avail_idxs)):
        raw_item = ds._get_raw_item_cached(local_idx)
        item_name = str(raw_item.get("item_name", ""))
        try:
            prompt_item = ds._materialize_rhythm_cache_compat(raw_item, item_name=item_name)
            conditioning = ds._build_reference_prompt_unit_conditioning(prompt_item, target_mode=target_mode)
        except Exception:
            continue
        if not conditioning:
            continue
        g_ref, g_status = _conditioning_g(ds, conditioning)
        if not np.isfinite(g_ref):
            continue
        rows.append(
            {
                "item_name": item_name,
                "speaker": str(raw_item.get("speaker", "")),
                "g_ref": float(g_ref),
                "g_status": str(g_status),
            }
        )
    return rows


def _pick_midpoint_item(rows: list[dict[str, Any]], *, target_tempo: float) -> dict[str, Any]:
    return min(
        rows,
        key=lambda row: (
            abs(float(row["g_ref"]) - float(target_tempo)),
            str(row["item_name"]),
        ),
    )


def build_auto_gate1_cases(ds: ConanDataset) -> list[dict[str, Any]]:
    rows = _collect_valid_prompt_rows(ds)
    if not rows:
        raise RuntimeError(f"No valid prompt-conditioning items were found in split '{ds.prefix}'.")

    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_speaker[str(row["speaker"])].append(copy.deepcopy(row))

    all_rows = sorted(rows, key=lambda row: (str(row["speaker"]), float(row["g_ref"]), str(row["item_name"])))
    cases: list[dict[str, Any]] = []
    for speaker in sorted(by_speaker.keys()):
        speaker_rows = sorted(
            by_speaker[speaker],
            key=lambda row: (float(row["g_ref"]), str(row["item_name"])),
        )
        if len(speaker_rows) < 4:
            continue

        distinct_rows: list[dict[str, Any]] = []
        last_tempo: float | None = None
        for row in speaker_rows:
            tempo = float(row["g_ref"])
            if last_tempo is None or abs(tempo - last_tempo) > 1.0e-6:
                distinct_rows.append(row)
                last_tempo = tempo
        if len(distinct_rows) < 3:
            continue

        slow = distinct_rows[0]
        fast = distinct_rows[-1]
        midpoint = 0.5 * (float(slow["g_ref"]) + float(fast["g_ref"]))
        mid_candidates = distinct_rows[1:-1]
        mid = _pick_midpoint_item(mid_candidates, target_tempo=midpoint)

        ref_ids = {str(slow["item_name"]), str(mid["item_name"]), str(fast["item_name"])}
        source_candidates = [row for row in speaker_rows if str(row["item_name"]) not in ref_ids]
        if not source_candidates:
            continue
        source = _pick_midpoint_item(source_candidates, target_tempo=midpoint)

        random_candidates = [row for row in all_rows if str(row["speaker"]) != speaker]
        random_ref = (
            _pick_midpoint_item(random_candidates, target_tempo=float(source["g_ref"]))
            if random_candidates
            else None
        )

        refs = {
            "slow": str(slow["item_name"]),
            "mid": str(mid["item_name"]),
            "fast": str(fast["item_name"]),
        }
        if random_ref is not None:
            refs["random_ref"] = str(random_ref["item_name"])
        cases.append(
            {
                "speaker": speaker,
                "source": str(source["item_name"]),
                "refs": refs,
            }
        )
    if not cases:
        raise RuntimeError(
            f"No same-speaker slow/mid/fast probe cases could be auto-selected in split '{ds.prefix}'."
        )
    return cases
