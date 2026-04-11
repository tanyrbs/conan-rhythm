from __future__ import annotations

"""Compatibility shim.

Canonical rhythm task metric surfaces now live under:
- tasks.Conan.rhythm.rhythm_v2.metrics
- tasks.Conan.rhythm.duration_v3.metrics
- tasks.Conan.rhythm.common.metrics_impl
"""

from .common import metrics_impl as _impl

globals().update({name: getattr(_impl, name) for name in dir(_impl) if not name.startswith("__")})

__all__ = [name for name in globals() if not name.startswith("__")]
