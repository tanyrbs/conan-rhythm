from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

_AUTO_TOKENS = {"auto", "detect", "adaptive"}
_TRUE_TOKENS = {"1", "true", "yes", "y", "on"}
_FALSE_TOKENS = {"0", "false", "no", "n", "off"}


def parse_ddp_mode(value: Any, *, default: bool | None = None) -> bool | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _AUTO_TOKENS:
            return None
        if normalized in _TRUE_TOKENS:
            return True
        if normalized in _FALSE_TOKENS:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def resolve_ddp_runtime_config(
    find_unused_value: Any,
    static_graph_value: Any,
    *,
    ddp_logging_data: Mapping[str, Any] | None = None,
    default_find_unused: bool = True,
    default_static_graph: bool = False,
) -> tuple[bool, bool]:
    ddp_logging_data = ddp_logging_data or {}
    can_set_static_graph = bool(ddp_logging_data.get("can_set_static_graph", False))

    static_graph_mode = parse_ddp_mode(static_graph_value, default=None)
    if static_graph_mode is None:
        static_graph = can_set_static_graph or default_static_graph
    else:
        static_graph = static_graph_mode

    find_unused_mode = parse_ddp_mode(find_unused_value, default=None)
    if find_unused_mode is None:
        find_unused = default_find_unused if not static_graph else False
    else:
        find_unused = find_unused_mode
    return bool(find_unused), bool(static_graph)


def load_ddp_logging_data(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file():
        return {}
    try:
        with file_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, Mapping):
        return {str(k): _json_safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(v) for v in value]
    return str(value)


def save_ddp_logging_data(
    path: str | Path,
    logging_data: Mapping[str, Any],
    *,
    global_step: int,
    epoch: int,
) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(k): _json_safe_value(v) for k, v in logging_data.items()}
    payload["saved_global_step"] = int(global_step)
    payload["saved_epoch"] = int(epoch)
    tmp_path = file_path.with_suffix(file_path.suffix + ".part")
    with tmp_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)
    tmp_path.replace(file_path)


__all__ = [
    "load_ddp_logging_data",
    "parse_ddp_mode",
    "resolve_ddp_runtime_config",
    "save_ddp_logging_data",
]
