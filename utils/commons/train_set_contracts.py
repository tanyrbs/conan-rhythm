from __future__ import annotations

import json
import os
from typing import Any, Iterable

from utils.commons.condition_labels import (
    CONDITION_FIELDS,
    load_condition_id_maps,
    resolve_condition_artifact_paths,
)

_SHARED_JSON_ARTIFACTS = (
    "phone_set.json",
    "word_set.json",
    "spk_map.json",
    "spker_set.json",
)


def normalize_train_set_dirs(raw_value: str | Iterable[str] | None) -> list[str]:
    if raw_value in (None, ""):
        return []
    if isinstance(raw_value, str):
        entries = raw_value.split("|")
    else:
        entries = raw_value
    normalized: list[str] = []
    for entry in entries:
        candidate = str(entry or "").strip()
        if candidate:
            normalized.append(candidate)
    return normalized


def _load_json_artifact(path: str) -> tuple[bool, Any | None, str | None]:
    if not os.path.exists(path):
        return False, None, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return True, json.load(f), None
    except Exception as exc:
        return True, None, f"{type(exc).__name__}: {exc}"


def collect_shared_json_artifact_issues(
    base_dir: str | os.PathLike[str] | None,
    train_set_dirs: str | Iterable[str] | None,
) -> list[str]:
    normalized_dirs = normalize_train_set_dirs(train_set_dirs)
    if not base_dir or not normalized_dirs:
        return []

    base_dir = str(base_dir)
    issues: list[str] = []
    for filename in _SHARED_JSON_ARTIFACTS:
        base_path = os.path.join(base_dir, filename)
        any_present = os.path.exists(base_path) or any(
            os.path.exists(os.path.join(train_dir, filename))
            for train_dir in normalized_dirs
        )
        if not any_present:
            continue
        base_exists, base_payload, base_error = _load_json_artifact(base_path)
        if not base_exists:
            issues.append(
                f"Shared artifact '{filename}' is missing from binary_data_dir '{base_dir}' while train_sets use it."
            )
            continue
        if base_error is not None:
            issues.append(
                f"Failed to read shared artifact '{filename}' from binary_data_dir '{base_dir}': {base_error}"
            )
            continue

        for train_dir in normalized_dirs:
            train_path = os.path.join(train_dir, filename)
            train_exists, train_payload, train_error = _load_json_artifact(train_path)
            if not train_exists:
                issues.append(
                    f"Shared artifact '{filename}' is missing from train_set '{train_dir}'."
                )
                continue
            if train_error is not None:
                issues.append(
                    f"Failed to read shared artifact '{filename}' from train_set '{train_dir}': {train_error}"
                )
                continue
            if train_payload != base_payload:
                issues.append(
                    f"Shared artifact '{filename}' in train_set '{train_dir}' does not match binary_data_dir '{base_dir}'."
                )
    return issues


def collect_condition_map_issues(
    base_dir: str | os.PathLike[str] | None,
    train_set_dirs: str | Iterable[str] | None,
    *,
    fields: Iterable[str] = CONDITION_FIELDS,
) -> list[str]:
    normalized_dirs = normalize_train_set_dirs(train_set_dirs)
    if not base_dir or not normalized_dirs:
        return []

    normalized_fields = tuple(str(field) for field in fields)
    base_dir = str(base_dir)
    issues: list[str] = []

    try:
        base_paths = resolve_condition_artifact_paths(base_dir, fields=normalized_fields)
        base_maps = load_condition_id_maps([base_dir], fields=normalized_fields)
    except Exception as exc:
        return [f"Failed to resolve base condition artifacts from '{base_dir}': {exc}"]

    for train_dir in normalized_dirs:
        try:
            train_paths = resolve_condition_artifact_paths(train_dir, fields=normalized_fields)
            train_maps = load_condition_id_maps([train_dir], fields=normalized_fields)
        except Exception as exc:
            issues.append(f"Failed to resolve condition artifacts from train_set '{train_dir}': {exc}")
            continue
        for field in normalized_fields:
            base_has = base_paths.get(field) is not None
            train_has = train_paths.get(field) is not None
            if not base_has and not train_has:
                continue
            if base_has != train_has:
                issues.append(
                    f"{field} condition artifact presence differs between binary_data_dir '{base_dir}' and train_set '{train_dir}'."
                )
                continue
            if dict(train_maps.get(field, {})) != dict(base_maps.get(field, {})):
                issues.append(
                    f"{field} condition map in train_set '{train_dir}' does not match binary_data_dir '{base_dir}'."
                )
    return issues
