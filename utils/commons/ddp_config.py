from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

_AUTO_TOKENS = {"auto", "detect", "adaptive"}
_TRUE_TOKENS = {"1", "true", "yes", "y", "on"}
_FALSE_TOKENS = {"0", "false", "no", "n", "off"}
_DDP_AUTO_SIGNATURE_KEYS = (
    "ddp_find_unused_parameters",
    "ddp_static_graph",
    "disc_start_steps",
    "rhythm_apply_train_override",
    "rhythm_apply_valid_override",
    "rhythm_distill_surface",
    "rhythm_enable_algorithmic_teacher",
    "rhythm_enable_dual_mode_teacher",
    "rhythm_enable_learned_offline_teacher",
    "rhythm_fastpath_disable_acoustic_when_module_only",
    "rhythm_online_retimed_target_start_steps",
    "rhythm_optimize_module_only",
    "rhythm_require_retimed_cache",
    "rhythm_retimed_target_start_steps",
    "rhythm_runtime_enable_learned_offline_teacher",
    "rhythm_schedule_only_stage",
    "rhythm_stage",
    "rhythm_teacher_as_main",
    "rhythm_teacher_only_stage",
    "rhythm_use_retimed_target_if_available",
)


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


def build_ddp_auto_signature(
    *,
    hparams: Mapping[str, Any] | None = None,
    task: Any | None = None,
) -> str:
    payload: dict[str, Any] = {
        "task_module": task.__class__.__module__ if task is not None else None,
        "task_qualname": task.__class__.__qualname__ if task is not None else None,
    }
    if hparams is not None:
        payload["hparams"] = {
            key: _json_safe_value(hparams.get(key))
            for key in _DDP_AUTO_SIGNATURE_KEYS
            if key in hparams
        }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()


def get_ddp_auto_min_step(hparams: Mapping[str, Any] | None = None) -> int:
    hparams = hparams or {}
    dynamic_steps = [
        int(hparams.get("disc_start_steps", 0) or 0),
        int(hparams.get("rhythm_train_render_start_steps", 0) or 0),
        int(hparams.get("rhythm_retimed_target_start_steps", 0) or 0),
        int(
            hparams.get(
                "rhythm_online_retimed_target_start_steps",
                hparams.get("rhythm_retimed_target_start_steps", 0),
            )
            or 0
        ),
    ]
    return max(dynamic_steps) if dynamic_steps else 0


def select_ddp_logging_hint(
    logging_data: Mapping[str, Any] | None,
    *,
    signature: str | None = None,
    min_saved_global_step: int = 0,
) -> dict[str, Any]:
    logging_data = dict(logging_data or {})
    if not logging_data:
        return {}
    if signature is not None and logging_data.get("ddp_auto_signature") != signature:
        return {}
    saved_global_step = int(logging_data.get("saved_global_step", -1) or -1)
    if saved_global_step < int(min_saved_global_step):
        return {}
    return logging_data


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
    signature: str | None = None,
) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(k): _json_safe_value(v) for k, v in logging_data.items()}
    payload["saved_global_step"] = int(global_step)
    payload["saved_epoch"] = int(epoch)
    if signature is not None:
        payload["ddp_auto_signature"] = str(signature)
    tmp_path = file_path.with_suffix(file_path.suffix + ".part")
    with tmp_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)
    tmp_path.replace(file_path)


__all__ = [
    "build_ddp_auto_signature",
    "get_ddp_auto_min_step",
    "load_ddp_logging_data",
    "parse_ddp_mode",
    "resolve_ddp_runtime_config",
    "select_ddp_logging_hint",
    "save_ddp_logging_data",
]
