from __future__ import annotations

import numpy as np

# NumPy 2 removed several aliases that older third-party audio packages still
# touch during import. Restore only the small legacy surface we rely on before
# importing those optional dependencies.
_LEGACY_NUMPY_ALIASES = {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": np.complex128,
    "object": object,
    "str": str,
    "unicode": str,
}


def ensure_legacy_numpy_aliases() -> None:
    for name, value in _LEGACY_NUMPY_ALIASES.items():
        if name not in np.__dict__:
            setattr(np, name, value)


__all__ = ["ensure_legacy_numpy_aliases"]
