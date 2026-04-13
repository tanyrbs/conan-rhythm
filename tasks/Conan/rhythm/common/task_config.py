from __future__ import annotations

from pathlib import Path

from modules.Conan.rhythm.policy import (
    normalize_distill_surface,
    normalize_primary_target_surface,
    normalize_retimed_target_mode,
    normalize_rhythm_target_mode,
    parse_optional_bool,
    resolve_pause_boundary_weight,
)


def _normalize_public_surface(value, *, key: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = tuple(part.strip() for part in value.split(",") if part.strip())
    elif isinstance(value, (list, tuple, set)):
        normalized = tuple(str(part).strip() for part in value if str(part).strip())
    else:
        raise ValueError(f"{key} must be a list/tuple/set or comma-separated string for rhythm_v3.")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{key} contains duplicate entries for rhythm_v3.")
    return normalized


def _validate_required_public_surface(
    values: tuple[str, ...] | None,
    *,
    key: str,
    required: tuple[str, ...],
    forbidden: tuple[str, ...] = (),
) -> None:
    if values is None:
        return
    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError(f"{key} is missing required rhythm_v3 entries: {missing}")
    stale = [name for name in forbidden if name in values]
    if stale:
        raise ValueError(f"{key} contains legacy rhythm_v2/slot entries that are removed for rhythm_v3: {stale}")


def _normalize_optional_path(value, *, key: str) -> str | None:
    del key
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    return normalized


def _validate_optional_existing_path(value, *, key: str) -> str | None:
    normalized = _normalize_optional_path(value, key=key)
    if normalized is None:
        return None
    if not Path(normalized).exists():
        raise ValueError(f"{key} must point to an existing path for rhythm_v3: {normalized}")
    return normalized


def resolve_task_pause_boundary_weight(hparams) -> float:
    return resolve_pause_boundary_weight(hparams)


def parse_task_optional_bool(value):
    return parse_optional_bool(value)


def resolve_task_target_mode(hparams) -> str:
    return normalize_rhythm_target_mode(hparams.get("rhythm_dataset_target_mode", "prefer_cache"))


def resolve_task_primary_target_surface(hparams) -> str:
    return normalize_primary_target_surface(hparams.get("rhythm_primary_target_surface", "guidance"))


def resolve_task_distill_surface(hparams) -> str:
    return normalize_distill_surface(hparams.get("rhythm_distill_surface", "auto"))


def resolve_task_retimed_target_mode(hparams) -> str:
    return normalize_retimed_target_mode(hparams.get("rhythm_retimed_target_mode", "cached"))


__all__ = [
    "_normalize_optional_path",
    "_validate_optional_existing_path",
    "_normalize_public_surface",
    "_validate_required_public_surface",
    "parse_task_optional_bool",
    "resolve_task_pause_boundary_weight",
    "resolve_task_target_mode",
    "resolve_task_primary_target_surface",
    "resolve_task_distill_surface",
    "resolve_task_retimed_target_mode",
]
