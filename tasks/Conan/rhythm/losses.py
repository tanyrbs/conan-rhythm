from __future__ import annotations

"""Compatibility shim.

Canonical rhythm task loss surfaces now live under:
- tasks.Conan.rhythm.rhythm_v2.losses
- tasks.Conan.rhythm.duration_v3.losses
- tasks.Conan.rhythm.common.losses_impl
"""

from .common import losses_impl as _impl

globals().update({name: getattr(_impl, name) for name in dir(_impl) if not name.startswith("__")})

__all__ = [name for name in globals() if not name.startswith("__")]
