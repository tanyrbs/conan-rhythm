from __future__ import annotations


class RhythmDatasetPrefilterDrop(RuntimeError):
    """Raised when a sample is intentionally dropped and should be retried."""


__all__ = ["RhythmDatasetPrefilterDrop"]
