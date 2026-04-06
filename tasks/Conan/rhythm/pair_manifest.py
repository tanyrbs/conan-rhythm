from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _maybe_load_yaml(text: str):
    try:
        import yaml  # type: ignore
    except Exception:
        return None
    return yaml.safe_load(text)


def _load_structured_file(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        data = _maybe_load_yaml(text)
        if data is None:
            raise RuntimeError(
                f"Pair manifest '{path}' uses YAML, but PyYAML is unavailable in this environment."
            )
        return data
    try:
        return json.loads(text)
    except Exception:
        data = _maybe_load_yaml(text)
        if data is not None:
            return data
        raise RuntimeError(
            f"Unsupported pair manifest format for '{path}'. Use JSON, JSONL, or YAML."
        )


def _normalize_ref_entry(ref_entry: Any) -> dict[str, Any]:
    if isinstance(ref_entry, str):
        return {"ref_item_name": ref_entry}
    if not isinstance(ref_entry, dict):
        raise RuntimeError(
            f"Unsupported pair manifest ref entry type: {type(ref_entry)!r}. "
            "Expected a string item name or a mapping."
        )
    ref_name = (
        ref_entry.get("ref_item_name")
        or ref_entry.get("ref")
        or ref_entry.get("item_name")
        or ref_entry.get("name")
    )
    if not ref_name:
        raise RuntimeError(f"Pair manifest ref entry is missing ref_item_name/ref/item_name: {ref_entry!r}")
    normalized = dict(ref_entry)
    normalized["ref_item_name"] = str(ref_name)
    return normalized


def _normalize_manifest_entries(raw_entries: Any) -> list[dict[str, Any]]:
    if raw_entries is None:
        return []
    if isinstance(raw_entries, dict):
        if "pairs" in raw_entries:
            raw_entries = raw_entries["pairs"]
        else:
            raw_entries = [raw_entries]
    if not isinstance(raw_entries, list):
        raise RuntimeError("Pair manifest root must be a list, a dict with 'pairs', or a split-keyed mapping.")

    normalized: list[dict[str, Any]] = []
    for entry in raw_entries:
        if isinstance(entry, str):
            raise RuntimeError(
                "Pair manifest entries must be mappings. "
                "For grouped expansion use {source/refs}, for flat mode use {source, ref}."
            )
        if not isinstance(entry, dict):
            raise RuntimeError(f"Unsupported pair manifest entry type: {type(entry)!r}")
        source_name = (
            entry.get("source_item_name")
            or entry.get("source")
            or entry.get("src")
            or entry.get("item_name")
        )
        if not source_name:
            raise RuntimeError(f"Pair manifest entry is missing source/source_item_name: {entry!r}")
        source_name = str(source_name)
        group_id = str(entry.get("group_id", source_name))
        refs = entry.get("refs", None)
        if refs is None:
            ref_name = (
                entry.get("ref_item_name")
                or entry.get("ref")
                or entry.get("reference")
            )
            if not ref_name:
                raise RuntimeError(
                    f"Flat pair manifest entry must include ref/ref_item_name/reference: {entry!r}"
                )
            refs = [{"ref_item_name": str(ref_name)}]
        if not isinstance(refs, list):
            raise RuntimeError(f"Pair manifest refs must be a list for source={source_name!r}.")
        include_self = bool(entry.get("include_self", False))
        expanded_refs = [_normalize_ref_entry(ref) for ref in refs]
        if include_self and all(str(ref["ref_item_name"]) != source_name for ref in expanded_refs):
            expanded_refs = [{"ref_item_name": source_name, "pair_label": "self"}] + expanded_refs
        for rank, ref_entry in enumerate(expanded_refs):
            pair_label = (
                ref_entry.get("pair_label")
                or ref_entry.get("label")
                or ref_entry.get("bucket")
                or entry.get("pair_label")
                or entry.get("label")
            )
            normalized.append(
                {
                    "source_item_name": source_name,
                    "ref_item_name": str(ref_entry["ref_item_name"]),
                    "group_id": group_id,
                    "pair_rank": int(ref_entry.get("pair_rank", rank)),
                    "pair_label": None if pair_label in {None, ""} else str(pair_label),
                }
            )
    return normalized


def load_pair_manifest(path: str | Path, *, prefix: str | None = None) -> list[dict[str, Any]]:
    manifest_path = Path(path)
    payload = _load_structured_file(manifest_path)
    if isinstance(payload, dict) and prefix is not None and prefix in payload:
        payload = payload[prefix]
    return _normalize_manifest_entries(payload)


__all__ = ["load_pair_manifest"]
