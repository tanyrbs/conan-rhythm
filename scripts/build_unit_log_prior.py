#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from modules.Conan.rhythm_v3.source_cache import (
    duration_v3_cache_meta_signature,
    resolve_duration_v3_cache_meta,
)


def _iter_input_files(paths: list[str]) -> Iterable[Path]:
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.suffix.lower() in {".pt", ".pth", ".npz", ".npy"}:
                    yield child
            continue
        yield path


def _load_payload(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=True)
    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            return {key: data[key] for key in data.files}
    if suffix in {".pt", ".pth"}:
        return torch.load(path, map_location="cpu", weights_only=False)
    raise ValueError(f"Unsupported input format: {path}")


def _maybe_yield_record(payload, *, _visited: set[int] | None = None):
    if _visited is None:
        _visited = set()
    obj_id = id(payload)
    if obj_id in _visited:
        return
    _visited.add(obj_id)
    if isinstance(payload, dict):
        has_units = "content_units" in payload
        has_duration = "dur_anchor_src" in payload or "source_duration_obs" in payload
        if has_units and has_duration:
            yield payload
            return
        for value in payload.values():
            yield from _maybe_yield_record(value, _visited=_visited)
        return
    if isinstance(payload, (list, tuple)):
        for item in payload:
            yield from _maybe_yield_record(item, _visited=_visited)
        return


def _as_array(value, *, dtype) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=dtype)
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(dtype, copy=False)
    return np.asarray(value, dtype=dtype)


def _drop_edge_valid_mask(valid: np.ndarray, *, drop_edge_runs: int) -> np.ndarray:
    drop = max(0, int(drop_edge_runs))
    if drop <= 0:
        return valid
    keep = np.asarray(valid, dtype=bool).copy()
    active = np.flatnonzero(keep)
    if active.size <= (2 * drop):
        return keep
    keep[active[:drop]] = False
    keep[active[-drop:]] = False
    return keep


def _collapse_scalar(values: set[object], *, missing="missing", mixed="mixed"):
    cleaned = {value for value in values if value is not None}
    if not cleaned:
        return missing
    if len(cleaned) == 1:
        return next(iter(cleaned))
    return mixed


def _collect_frontend_meta(record) -> dict[str, object]:
    cache_meta = resolve_duration_v3_cache_meta(record)
    signature = duration_v3_cache_meta_signature(cache_meta)
    meta = {
        "frontend_meta_signature": signature,
        "silent_token": None,
        "separator_aware": None,
        "tail_open_units": None,
        "emit_silence_runs": None,
        "debounce_min_run_frames": None,
        "phrase_boundary_threshold": None,
    }
    if cache_meta is None:
        return meta
    meta.update(
        {
            "silent_token": cache_meta.get("silent_token"),
            "separator_aware": cache_meta.get("separator_aware"),
            "tail_open_units": cache_meta.get("tail_open_units"),
            "emit_silence_runs": cache_meta.get("emit_silence_runs"),
            "debounce_min_run_frames": cache_meta.get("debounce_min_run_frames"),
            "phrase_boundary_threshold": cache_meta.get("phrase_boundary_threshold"),
        }
    )
    return meta


def _build_prior(
    inputs: list[str],
    *,
    min_count: int,
    default_policy: str,
    global_backoff: str,
    exclude_open_runs: bool,
    only_sealed_runs: bool,
    drop_edge_runs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, dict[str, object]]:
    buckets: dict[int, list[float]] = defaultdict(list)
    all_values: list[float] = []
    frontend_meta_values = {
        "frontend_meta_signature": set(),
        "silent_token": set(),
        "separator_aware": set(),
        "tail_open_units": set(),
        "emit_silence_runs": set(),
        "debounce_min_run_frames": set(),
        "phrase_boundary_threshold": set(),
    }
    for path in _iter_input_files(inputs):
        payload = _load_payload(path)
        for record in _maybe_yield_record(payload):
            record_meta = _collect_frontend_meta(record)
            for key, value in record_meta.items():
                frontend_meta_values[key].add(value)
            units = _as_array(record.get("content_units"), dtype=np.int64)
            duration = _as_array(record.get("dur_anchor_src", record.get("source_duration_obs")), dtype=np.float32)
            if units is None or duration is None:
                continue
            units = units.reshape(-1)
            duration = duration.reshape(-1)
            if units.shape[0] != duration.shape[0]:
                raise ValueError(
                    f"content_units/duration length mismatch in {path}: "
                    f"units={int(units.shape[0])} duration={int(duration.shape[0])}"
                )
            if bool((units < 0).any()):
                raise ValueError(f"Negative unit ids are not supported in {path}")
            silence = _as_array(record.get("source_silence_mask"), dtype=np.float32)
            if silence is None:
                silence = np.zeros_like(duration, dtype=np.float32)
            silence = silence.reshape(-1)
            if silence.shape[0] != duration.shape[0]:
                raise ValueError(
                    "source_silence_mask length mismatch in record: "
                    f"silence={int(silence.shape[0])} duration={int(duration.shape[0])}"
                )
            valid = (duration > 0.0) & (silence <= 0.5)
            if exclude_open_runs:
                open_mask = _as_array(record.get("open_run_mask"), dtype=np.float32)
                if open_mask is not None:
                    open_mask = open_mask.reshape(-1)
                    if open_mask.shape[0] != duration.shape[0]:
                        raise ValueError(
                            "open_run_mask length mismatch in record: "
                            f"open={int(open_mask.shape[0])} duration={int(duration.shape[0])}"
                        )
                    valid &= open_mask <= 0.5
            if only_sealed_runs:
                sealed_mask = _as_array(record.get("sealed_mask"), dtype=np.float32)
                if sealed_mask is not None:
                    sealed_mask = sealed_mask.reshape(-1)
                    if sealed_mask.shape[0] != duration.shape[0]:
                        raise ValueError(
                            "sealed_mask length mismatch in record: "
                            f"sealed={int(sealed_mask.shape[0])} duration={int(duration.shape[0])}"
                        )
                    valid &= sealed_mask > 0.5
            valid = _drop_edge_valid_mask(valid, drop_edge_runs=drop_edge_runs)
            for unit_id, value in zip(units[valid], duration[valid]):
                log_value = float(np.log(max(float(value), 1.0e-4)))
                buckets[int(unit_id)].append(log_value)
                all_values.append(log_value)
    if not buckets:
        raise RuntimeError("No usable speech runs were found in the provided inputs.")
    if default_policy != "global_median":
        raise ValueError(f"Unsupported default prior policy: {default_policy!r}")
    if global_backoff not in {"hard", "linear"}:
        raise ValueError(f"Unsupported global backoff policy: {global_backoff!r}")
    vocab_size = max(buckets.keys()) + 1
    global_default = float(np.median(np.asarray(all_values, dtype=np.float32)))
    prior = np.full((vocab_size,), global_default, dtype=np.float32)
    counts = np.zeros((vocab_size,), dtype=np.int64)
    is_default = np.ones((vocab_size,), dtype=np.int64)
    backoff_weight = np.zeros((vocab_size,), dtype=np.float32)
    min_count = max(1, int(min_count))
    for unit_id, values in buckets.items():
        arr = np.asarray(values, dtype=np.float32)
        count = int(arr.shape[0])
        counts[unit_id] = count
        local = float(np.median(arr))
        if global_backoff == "hard":
            alpha = 1.0 if count >= min_count else 0.0
        else:
            alpha = min(1.0, float(count) / float(min_count))
        backoff_weight[unit_id] = float(alpha)
        prior[unit_id] = float((alpha * local) + ((1.0 - alpha) * global_default))
        if count >= min_count and alpha >= 1.0 - 1.0e-6:
            is_default[unit_id] = 0
    frontend_meta = {
        "frontend_meta_signature": _collapse_scalar(frontend_meta_values["frontend_meta_signature"]),
        "silent_token": _collapse_scalar(frontend_meta_values["silent_token"]),
        "separator_aware": _collapse_scalar(frontend_meta_values["separator_aware"]),
        "tail_open_units": _collapse_scalar(frontend_meta_values["tail_open_units"]),
        "emit_silence_runs": _collapse_scalar(frontend_meta_values["emit_silence_runs"]),
        "debounce_min_run_frames": _collapse_scalar(frontend_meta_values["debounce_min_run_frames"]),
        "phrase_boundary_threshold": _collapse_scalar(frontend_meta_values["phrase_boundary_threshold"]),
    }
    return prior, counts, is_default, backoff_weight, global_default, frontend_meta


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a repo-native unit_log_prior bundle for rhythm_v3 unit_norm experiments.",
    )
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        required=True,
        help="Source-cache file or directory (.pt/.pth/.npz/.npy). Can be specified multiple times.",
    )
    parser.add_argument("--output", required=True, help="Output .npz path.")
    parser.add_argument(
        "--source",
        default="unspecified",
        help="Human-readable provenance label stored in the bundle.",
    )
    parser.add_argument(
        "--version",
        default="manual",
        help="Version string stored in the bundle.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=5,
        help="Minimum speech-run observations required for a unit-specific median. Default: 5.",
    )
    parser.add_argument(
        "--default-prior",
        choices=("global_median",),
        default="global_median",
        help="Fallback policy for rare/unseen units. Default: global_median.",
    )
    parser.add_argument(
        "--global-backoff",
        choices=("linear", "hard"),
        default="linear",
        help="Backoff policy for low-count units. Default: linear shrinkage toward the global prior.",
    )
    parser.add_argument(
        "--exclude-open-runs",
        action="store_true",
        help="Exclude runs marked by open_run_mask>0.5 when present.",
    )
    parser.add_argument(
        "--only-sealed-runs",
        action="store_true",
        help="Keep only runs with sealed_mask>0.5 when present.",
    )
    parser.add_argument(
        "--drop-edge-runs",
        type=int,
        default=0,
        help="Drop the first/last N eligible speech runs per record before aggregation. Default: 0.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    prior, counts, is_default, backoff_weight, global_default, frontend_meta = _build_prior(
        args.inputs,
        min_count=int(args.min_count),
        default_policy=str(args.default_prior),
        global_backoff=str(args.global_backoff),
        exclude_open_runs=bool(args.exclude_open_runs),
        only_sealed_runs=bool(args.only_sealed_runs),
        drop_edge_runs=int(args.drop_edge_runs),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        unit_log_prior=prior.astype(np.float32, copy=False),
        unit_count=counts.astype(np.int64, copy=False),
        unit_log_prior_is_default=is_default.astype(np.int64, copy=False),
        unit_prior_backoff_weight=backoff_weight.astype(np.float32, copy=False),
        global_speech_log_prior=np.asarray([float(global_default)], dtype=np.float32),
        unit_prior_min_count=np.asarray([int(max(1, args.min_count))], dtype=np.int64),
        unit_prior_default_value=np.asarray([float(global_default)], dtype=np.float32),
        unit_prior_default_policy=np.asarray([str(args.default_prior)], dtype=object),
        unit_prior_global_backoff=np.asarray([str(args.global_backoff)], dtype=object),
        unit_prior_default_count=np.asarray([int(is_default.sum())], dtype=np.int64),
        unit_prior_observed_count=np.asarray([int((counts > 0).sum())], dtype=np.int64),
        unit_prior_low_count_count=np.asarray(
            [int(((counts > 0) & (counts < max(1, args.min_count))).sum())],
            dtype=np.int64,
        ),
        unit_prior_filter_exclude_open_runs=np.asarray([bool(args.exclude_open_runs)], dtype=np.bool_),
        unit_prior_filter_only_sealed_runs=np.asarray([bool(args.only_sealed_runs)], dtype=np.bool_),
        unit_prior_filter_drop_edge_runs=np.asarray([int(max(0, args.drop_edge_runs))], dtype=np.int64),
        unit_prior_source=np.asarray([str(args.source)], dtype=object),
        unit_prior_version=np.asarray([str(args.version)], dtype=object),
        unit_prior_vocab_size=np.asarray([int(prior.shape[0])], dtype=np.int64),
        unit_prior_frontend_signature=np.asarray([str(frontend_meta["frontend_meta_signature"])], dtype=object),
        unit_prior_silent_token=np.asarray([frontend_meta["silent_token"]], dtype=object),
        unit_prior_separator_aware=np.asarray([frontend_meta["separator_aware"]], dtype=object),
        unit_prior_tail_open_units=np.asarray([frontend_meta["tail_open_units"]], dtype=object),
        unit_prior_emit_silence_runs=np.asarray([frontend_meta["emit_silence_runs"]], dtype=object),
        unit_prior_debounce_min_run_frames=np.asarray([frontend_meta["debounce_min_run_frames"]], dtype=object),
        unit_prior_phrase_boundary_threshold=np.asarray([frontend_meta["phrase_boundary_threshold"]], dtype=object),
    )
    print(
        f"[build_unit_log_prior] wrote vocab_size={int(prior.shape[0])} "
        f"observed_units={int((counts > 0).sum())} "
        f"default_units={int(is_default.sum())} "
        f"global_default={float(global_default):.6f} "
        f"backoff={args.global_backoff} "
        f"frontend_signature={frontend_meta['frontend_meta_signature']} -> {output}"
    )


if __name__ == "__main__":
    main()
