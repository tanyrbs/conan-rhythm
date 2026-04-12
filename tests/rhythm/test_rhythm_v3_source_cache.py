from __future__ import annotations

import pytest

from modules.Conan.rhythm_v3.source_cache import (
    DURATION_V3_CACHE_META_KEY,
    assert_duration_v3_cache_meta_compatible,
    build_source_rhythm_cache_v3,
    duration_v3_cache_meta_signature,
)


def test_build_source_rhythm_cache_v3_emits_meta_contract():
    cache = build_source_rhythm_cache_v3(
        [1, 1, 57, 57, 2, 2],
        silent_token=57,
        separator_aware=True,
        tail_open_units=2,
        emit_silence_runs=True,
        debounce_min_run_frames=3,
        phrase_boundary_threshold=0.61,
    )
    meta = cache[DURATION_V3_CACHE_META_KEY]
    assert meta["cache_version"] == 3
    assert meta["silent_token"] == 57
    assert meta["separator_aware"] is True
    assert meta["tail_open_units"] == 2
    assert meta["emit_silence_runs"] is True
    assert meta["debounce_min_run_frames"] == 3
    assert meta["phrase_boundary_threshold"] == pytest.approx(0.61)
    assert duration_v3_cache_meta_signature(meta) != "missing"


def test_duration_v3_cache_meta_mismatch_fails_fast():
    cache = build_source_rhythm_cache_v3(
        [1, 1, 57, 2, 2],
        silent_token=57,
        separator_aware=True,
        tail_open_units=1,
        emit_silence_runs=True,
        debounce_min_run_frames=2,
        phrase_boundary_threshold=0.55,
    )
    with pytest.raises(ValueError, match="meta mismatch"):
        assert_duration_v3_cache_meta_compatible(
            cache[DURATION_V3_CACHE_META_KEY],
            silent_token=57,
            separator_aware=True,
            tail_open_units=1,
            emit_silence_runs=False,
            debounce_min_run_frames=2,
            phrase_boundary_threshold=0.55,
        )


def test_build_source_rhythm_cache_v3_defaults_follow_mainline_surface():
    cache = build_source_rhythm_cache_v3(
        [1, 1, 57, 57, 2],
        silent_token=57,
    )
    meta = cache[DURATION_V3_CACHE_META_KEY]
    assert meta["emit_silence_runs"] is True
    assert meta["debounce_min_run_frames"] == 2
    assert cache["source_silence_mask"].tolist() == [0.0, 1.0, 0.0]
