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


def _build_prior(inputs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    buckets: dict[int, list[float]] = defaultdict(list)
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
                continue
            silence = _as_array(record.get("source_silence_mask"), dtype=np.float32)
            if silence is None:
                silence = np.zeros_like(duration, dtype=np.float32)
            silence = silence.reshape(-1)
            if silence.shape[0] != duration.shape[0]:
                silence_resized = np.zeros_like(duration, dtype=np.float32)
                limit = min(int(silence.shape[0]), int(duration.shape[0]))
                silence_resized[:limit] = silence[:limit]
                silence = silence_resized
            valid = (duration > 0.0) & (silence <= 0.5)
            for unit_id, value in zip(units[valid], duration[valid]):
                buckets[int(unit_id)].append(float(np.log(max(float(value), 1.0e-4))))
    if not buckets:
        raise RuntimeError("No usable speech runs were found in the provided inputs.")
    vocab_size = max(buckets.keys()) + 1
    prior = np.zeros((vocab_size,), dtype=np.float32)
    counts = np.zeros((vocab_size,), dtype=np.int64)
    for unit_id, values in buckets.items():
        arr = np.asarray(values, dtype=np.float32)
        prior[unit_id] = float(np.median(arr))
        counts[unit_id] = int(arr.shape[0])
    return prior, counts


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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    prior, counts = _build_prior(args.inputs)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        unit_log_prior=prior.astype(np.float32, copy=False),
        unit_count=counts.astype(np.int64, copy=False),
        unit_prior_source=np.asarray([str(args.source)], dtype=object),
        unit_prior_version=np.asarray([str(args.version)], dtype=object),
        unit_prior_vocab_size=np.asarray([int(prior.shape[0])], dtype=np.int64),
    )
    print(
        f"[build_unit_log_prior] wrote vocab_size={int(prior.shape[0])} "
        f"nonzero_units={int((counts > 0).sum())} -> {output}"
    )


if __name__ == "__main__":
    main()
