from __future__ import annotations

import pytest
import numpy as np

from modules.Conan.rhythm_v3.source_cache import (
    DURATION_V3_CACHE_META_KEY,
    UNIT_LOG_PRIOR_META_KEY,
    assert_duration_v3_cache_meta_compatible,
    attach_unit_log_prior_to_source_cache,
    build_source_rhythm_cache_v3,
    duration_v3_cache_meta_signature,
    load_unit_log_prior_bundle,
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


def test_attach_unit_log_prior_to_source_cache_maps_vocab_prior_to_runs():
    cache = build_source_rhythm_cache_v3(
        [1, 1, 57, 2],
        silent_token=57,
    )
    attach_unit_log_prior_to_source_cache(
        cache,
        unit_log_prior=np.asarray([0.0, 0.2, 0.4] + [0.0] * 55, dtype=np.float32),
        unit_prior_source="unit-test",
        unit_prior_version="v1",
    )
    assert cache["unit_log_prior"].shape == cache["content_units"].shape
    expected = np.asarray([0.2 if int(unit_id) == 1 else 0.4 if int(unit_id) == 2 else 0.0 for unit_id in cache["content_units"]], dtype=np.float32)
    assert np.allclose(cache["unit_log_prior"], expected)
    assert cache[UNIT_LOG_PRIOR_META_KEY]["unit_prior_source"] == "unit-test"
    assert cache[UNIT_LOG_PRIOR_META_KEY]["unit_prior_version"] == "v1"


def test_build_source_rhythm_cache_v3_can_attach_prior_from_path(tmp_path):
    prior_path = tmp_path / "unit_prior.npz"
    np.savez(
        prior_path,
        unit_log_prior=np.asarray([0.0, 0.3, 0.7] + [0.0] * 55, dtype=np.float32),
        unit_prior_source=np.asarray(["native_native_demo"], dtype=object),
        unit_prior_version=np.asarray(["2026-04-12"], dtype=object),
        unit_prior_vocab_size=np.asarray([58], dtype=np.int64),
    )
    cache = build_source_rhythm_cache_v3(
        [1, 57, 2],
        silent_token=57,
        unit_prior_path=str(prior_path),
    )
    expected = np.asarray([0.3 if int(unit_id) == 1 else 0.7 if int(unit_id) == 2 else 0.0 for unit_id in cache["content_units"]], dtype=np.float32)
    assert np.allclose(cache["unit_log_prior"], expected)
    assert cache[UNIT_LOG_PRIOR_META_KEY]["unit_prior_vocab_size"] == 58
    assert "unit_prior_path" in cache[UNIT_LOG_PRIOR_META_KEY]


def test_load_unit_log_prior_bundle_reads_npz_metadata(tmp_path):
    prior_path = tmp_path / "unit_prior_bundle.npz"
    np.savez(
        prior_path,
        unit_log_prior=np.asarray([0.0, 0.2, 0.4], dtype=np.float32),
        unit_prior_source=np.asarray(["demo"], dtype=object),
        unit_prior_version=np.asarray(["v2"], dtype=object),
        unit_prior_vocab_size=np.asarray([3], dtype=np.int64),
    )
    bundle = load_unit_log_prior_bundle(str(prior_path))
    assert np.allclose(bundle["unit_log_prior"], np.asarray([0.0, 0.2, 0.4], dtype=np.float32))
    assert bundle["unit_prior_source"] == "demo"
    assert bundle["unit_prior_version"] == "v2"
    assert bundle["unit_prior_vocab_size"] == 3
