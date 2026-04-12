#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Iterable
import sys

import numpy as np
import torch


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


def _maybe_yield_record(payload):
    if isinstance(payload, dict):
        has_units = "content_units" in payload
        has_duration = "dur_anchor_src" in payload or "source_duration_obs" in payload
        if has_units and has_duration:
            yield payload
        for value in payload.values():
            yield from _maybe_yield_record(value)
        return
    if isinstance(payload, (list, tuple)):
        for item in payload:
            yield from _maybe_yield_record(item)
        return


def _as_array(value, *, dtype) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=dtype)
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(dtype, copy=False)
    return np.asarray(value, dtype=dtype)


def _build_prior(
    inputs: list[str],
    *,
    min_count: int,
    default_policy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    buckets: dict[int, list[float]] = defaultdict(list)
    all_values: list[float] = []
    for path in _iter_input_files(inputs):
        payload = _load_payload(path)
        for record in _maybe_yield_record(payload):
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
            for unit_id, value in zip(units[valid], duration[valid]):
                log_value = float(np.log(max(float(value), 1.0e-4)))
                buckets[int(unit_id)].append(log_value)
                all_values.append(log_value)
    if not buckets:
        raise RuntimeError("No usable speech runs were found in the provided inputs.")
    if default_policy != "global_median":
        raise ValueError(f"Unsupported default prior policy: {default_policy!r}")
    vocab_size = max(buckets.keys()) + 1
    global_default = float(np.median(np.asarray(all_values, dtype=np.float32)))
    prior = np.full((vocab_size,), global_default, dtype=np.float32)
    counts = np.zeros((vocab_size,), dtype=np.int64)
    is_default = np.ones((vocab_size,), dtype=np.int64)
    min_count = max(1, int(min_count))
    for unit_id, values in buckets.items():
        arr = np.asarray(values, dtype=np.float32)
        counts[unit_id] = int(arr.shape[0])
        if int(arr.shape[0]) >= min_count:
            prior[unit_id] = float(np.median(arr))
            is_default[unit_id] = 0
    return prior, counts, is_default, global_default


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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    prior, counts, is_default, global_default = _build_prior(
        args.inputs,
        min_count=int(args.min_count),
        default_policy=str(args.default_prior),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        unit_log_prior=prior.astype(np.float32, copy=False),
        unit_count=counts.astype(np.int64, copy=False),
        unit_log_prior_is_default=is_default.astype(np.int64, copy=False),
        global_speech_log_prior=np.asarray([float(global_default)], dtype=np.float32),
        unit_prior_min_count=np.asarray([int(max(1, args.min_count))], dtype=np.int64),
        unit_prior_default_value=np.asarray([float(global_default)], dtype=np.float32),
        unit_prior_default_policy=np.asarray([str(args.default_prior)], dtype=object),
        unit_prior_default_count=np.asarray([int(is_default.sum())], dtype=np.int64),
        unit_prior_observed_count=np.asarray([int((counts > 0).sum())], dtype=np.int64),
        unit_prior_low_count_count=np.asarray(
            [int(((counts > 0) & (counts < max(1, args.min_count))).sum())],
            dtype=np.int64,
        ),
        unit_prior_source=np.asarray([str(args.source)], dtype=object),
        unit_prior_version=np.asarray([str(args.version)], dtype=object),
        unit_prior_vocab_size=np.asarray([int(prior.shape[0])], dtype=np.int64),
    )
    print(
        f"[build_unit_log_prior] wrote vocab_size={int(prior.shape[0])} "
        f"observed_units={int((counts > 0).sum())} "
        f"default_units={int(is_default.sum())} "
        f"global_default={float(global_default):.6f} -> {output}"
    )


if __name__ == "__main__":
    main()
