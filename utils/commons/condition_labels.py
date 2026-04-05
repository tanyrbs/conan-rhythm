from __future__ import annotations

import json
import os
from typing import Any, Iterable

CONDITION_FIELDS = ("style", "emotion", "accent")

_CONDITION_ARTIFACT_CANDIDATES = {
    field: (f"{field}_map.json", f"{field}_set.json")
    for field in CONDITION_FIELDS
}


def _looks_int_like(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return False
        if stripped[0] in {"+", "-"}:
            stripped = stripped[1:]
        return stripped.isdigit()
    return False


def resolve_condition_artifact_paths(
    data_dir: str | os.PathLike[str] | None,
    *,
    fields: Iterable[str] = CONDITION_FIELDS,
) -> dict[str, str | None]:
    resolved: dict[str, str | None] = {}
    normalized_dir = str(data_dir or "").strip()
    for field in fields:
        path = None
        if normalized_dir:
            for filename in _CONDITION_ARTIFACT_CANDIDATES.get(field, ()):
                candidate = os.path.join(normalized_dir, filename)
                if os.path.exists(candidate):
                    path = candidate
                    break
        resolved[str(field)] = path
    return resolved


def _normalize_condition_mapping(payload: Any, *, field: str, path: str) -> dict[str, int]:
    if isinstance(payload, list):
        return {str(label): int(idx) for idx, label in enumerate(payload)}
    if isinstance(payload, dict):
        if all(_looks_int_like(value) for value in payload.values()):
            return {str(label): int(value) for label, value in payload.items()}
        if all(_looks_int_like(key) for key in payload.keys()):
            return {str(label): int(idx) for idx, label in payload.items()}
    raise ValueError(
        f"Unsupported {field} condition artifact schema at {path}. "
        "Expected a list, a label->id dict, or an id->label dict."
    )


def load_condition_id_maps(
    candidate_dirs: Iterable[str | os.PathLike[str] | None],
    *,
    fields: Iterable[str] = CONDITION_FIELDS,
) -> dict[str, dict[str, int]]:
    resolved: dict[str, dict[str, int]] = {}
    normalized_fields = tuple(str(field) for field in fields)
    candidate_dirs = tuple(candidate_dirs)
    for field in normalized_fields:
        mapping: dict[str, int] = {}
        for data_dir in candidate_dirs:
            path = resolve_condition_artifact_paths(data_dir, fields=(field,)).get(field)
            if path is None:
                continue
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            mapping = _normalize_condition_mapping(payload, field=field, path=path)
            break
        resolved[field] = mapping
    return resolved
