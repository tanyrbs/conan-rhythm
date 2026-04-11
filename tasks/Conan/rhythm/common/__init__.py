"""Shared Conan rhythm task utilities."""

from __future__ import annotations

from importlib import import_module

_LAZY_EXPORTS = {
    "CommonRhythmDatasetMixin": (".dataset_mixin", "CommonRhythmDatasetMixin"),
    "CommonRhythmTaskMixin": (".task_mixin", "CommonRhythmTaskMixin"),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = list(_LAZY_EXPORTS)
