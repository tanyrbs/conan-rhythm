from __future__ import annotations

from importlib import import_module

__all__ = ["RhythmConanTaskMixin"]


def __getattr__(name: str):
    if name != "RhythmConanTaskMixin":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module("tasks.Conan.rhythm.task_mixin")
    value = getattr(module, name)
    globals()[name] = value
    return value
