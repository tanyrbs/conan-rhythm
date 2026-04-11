from __future__ import annotations

from modules.Conan.rhythm_v3.reference_memory import normalize_duration_v3_conditioning


def build_duration_v3_ref_conditioning(sample, *, explicit=None):
    if explicit is not None and not isinstance(explicit, dict):
        return explicit
    source = explicit if isinstance(explicit, dict) else sample
    return normalize_duration_v3_conditioning(source)


__all__ = [
    "build_duration_v3_ref_conditioning",
]
