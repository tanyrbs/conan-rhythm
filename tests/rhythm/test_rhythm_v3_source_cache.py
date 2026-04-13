from __future__ import annotations

import os
import pytest
import numpy as np
import subprocess
import sys
import torch
from pathlib import Path

from modules.Conan.rhythm_v3.source_cache import (
    DURATION_V3_CACHE_META_KEY,
    UNIT_LOG_PRIOR_META_KEY,
    assert_duration_v3_cache_meta_compatible,
    attach_unit_log_prior_to_source_cache,
    build_duration_v3_frontend_signature,
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
    assert cache[UNIT_LOG_PRIOR_META_KEY]["unit_prior_unseen_count"] == 0
    assert np.array_equal(cache["unit_log_prior_is_default"], np.zeros_like(cache["content_units"], dtype=np.int64))
    assert np.array_equal(cache["unit_log_prior_count"], np.zeros_like(cache["content_units"], dtype=np.int64))


def test_attach_unit_log_prior_to_source_cache_uses_global_default_for_unseen_units():
    cache = {
        "content_units": np.asarray([1, 4], dtype=np.int64),
    }
    default_value = float(np.log(6.0))

    attach_unit_log_prior_to_source_cache(
        cache,
        unit_prior_bundle={
            "unit_log_prior": np.asarray([0.0, np.log(2.0)], dtype=np.float32),
            "global_speech_log_prior": np.asarray([default_value], dtype=np.float32),
            "unit_log_prior_is_default": np.asarray([1, 0], dtype=np.int64),
            "unit_prior_min_count": np.asarray([5], dtype=np.int64),
            "unit_prior_default_policy": np.asarray(["global_median"], dtype=object),
            "unit_prior_source": np.asarray(["unit-test"], dtype=object),
            "unit_prior_version": np.asarray(["v2"], dtype=object),
        },
    )

    assert np.allclose(cache["unit_log_prior"], np.asarray([np.log(2.0), default_value], dtype=np.float32))
    assert np.array_equal(cache["unit_log_prior_is_default"], np.asarray([0, 1], dtype=np.int64))
    meta = cache[UNIT_LOG_PRIOR_META_KEY]
    assert meta["unit_prior_min_count"] == 5
    assert meta["unit_prior_default_policy"] == "global_median"
    assert meta["unit_prior_default_value"] == pytest.approx(default_value)
    assert meta["unit_prior_default_count"] == 1
    assert meta["unit_prior_unseen_count"] == 1
    assert meta["unit_prior_out_of_vocab_count"] == 1
    assert np.array_equal(cache["unit_log_prior_count"], np.asarray([0, 0], dtype=np.int64))


def test_attach_unit_log_prior_to_source_cache_rejects_frontend_signature_mismatch():
    cache = build_source_rhythm_cache_v3(
        [1, 57, 2],
        silent_token=57,
        emit_silence_runs=True,
        debounce_min_run_frames=2,
    )
    mismatched_signature = duration_v3_cache_meta_signature(
        {
            "cache_version": 3,
            "silent_token": 57,
            "separator_aware": True,
            "tail_open_units": 1,
            "emit_silence_runs": False,
            "debounce_min_run_frames": 2,
            "phrase_boundary_threshold": 0.55,
        }
    )

    with pytest.raises(ValueError, match="frontend signature mismatch"):
        attach_unit_log_prior_to_source_cache(
            cache,
            unit_prior_bundle={
                "unit_log_prior": np.asarray([0.0, 0.2, 0.4] + [0.0] * 55, dtype=np.float32),
                "unit_prior_frontend_signature": np.asarray([mismatched_signature], dtype=object),
            },
            overwrite=True,
        )


def test_attach_unit_log_prior_to_source_cache_records_cache_side_provenance():
    cache = build_source_rhythm_cache_v3(
        [1, 57, 2],
        silent_token=57,
        separator_aware=True,
        tail_open_units=1,
        emit_silence_runs=True,
        debounce_min_run_frames=2,
        phrase_boundary_threshold=0.55,
    )
    attach_unit_log_prior_to_source_cache(
        cache,
        unit_prior_bundle={
            "unit_log_prior": np.asarray([0.0, 0.2, 0.4] + [0.0] * 55, dtype=np.float32),
            "unit_prior_frontend_signature": np.asarray(
                [duration_v3_cache_meta_signature(cache[DURATION_V3_CACHE_META_KEY])],
                dtype=object,
            ),
            "unit_prior_silent_token": np.asarray([57], dtype=object),
            "unit_prior_emit_silence_runs": np.asarray([True], dtype=object),
            "unit_prior_debounce_min_run_frames": np.asarray([2], dtype=object),
            "unit_prior_filter_min_run_stability": np.asarray([0.4], dtype=np.float32),
            "unit_prior_sha1": np.asarray(["abc123"], dtype=object),
            "unit_prior_build_cmd": np.asarray(["python scripts/build_unit_log_prior.py"], dtype=object),
            "unit_prior_input_manifest_sha1": np.asarray(["manifest123"], dtype=object),
            "unit_prior_input_count": np.asarray([8], dtype=np.int64),
            "unit_prior_special_token_policy": np.asarray(["exclude_silence_runs_keep_special_backoff"], dtype=object),
            "unit_prior_accepted_run_count": np.asarray([6], dtype=np.int64),
        },
        overwrite=True,
    )
    meta = cache[UNIT_LOG_PRIOR_META_KEY]
    assert meta["frontend_meta_signature"] == duration_v3_cache_meta_signature(cache[DURATION_V3_CACHE_META_KEY])
    assert meta["silent_token"] == 57
    assert meta["emit_silence_runs"] is True
    assert meta["debounce_min_run_frames"] == 2
    assert meta["unit_prior_frontend_signature"] == meta["frontend_meta_signature"]
    assert meta["unit_prior_filter_min_run_stability"] == pytest.approx(0.4)
    assert meta["unit_prior_sha1"] == "abc123"
    assert meta["unit_prior_build_cmd"] == "python scripts/build_unit_log_prior.py"
    assert meta["unit_prior_input_manifest_sha1"] == "manifest123"
    assert meta["unit_prior_input_count"] == 8
    assert meta["unit_prior_special_token_policy"] == "exclude_silence_runs_keep_special_backoff"
    assert meta["unit_prior_accepted_run_count"] == 6


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
    assert "unit_prior_default_value" in cache[UNIT_LOG_PRIOR_META_KEY]


def test_attach_unit_log_prior_to_source_cache_preserves_provenance_counts_and_path():
    cache = {
        "content_units": np.asarray([1, 2, 9], dtype=np.int64),
    }

    attach_unit_log_prior_to_source_cache(
        cache,
        unit_prior_bundle={
            "unit_log_prior": np.asarray([0.0, 0.2, 0.4], dtype=np.float32),
            "unit_count": np.asarray([0, 7, 2], dtype=np.int64),
            "unit_log_prior_is_default": np.asarray([1, 0, 1], dtype=np.int64),
            "global_speech_log_prior": np.asarray([0.3], dtype=np.float32),
            "unit_prior_min_count": np.asarray([5], dtype=np.int64),
            "unit_prior_default_policy": np.asarray(["global_median"], dtype=object),
            "unit_prior_source": np.asarray(["unit-test"], dtype=object),
            "unit_prior_version": np.asarray(["v3"], dtype=object),
            "unit_prior_path": np.asarray(["D:/tmp/unit_prior.npz"], dtype=object),
        },
    )

    meta = cache[UNIT_LOG_PRIOR_META_KEY]
    assert meta["unit_prior_source"] == "unit-test"
    assert meta["unit_prior_version"] == "v3"
    assert meta["unit_prior_path"] == "D:/tmp/unit_prior.npz"
    assert meta["unit_prior_default_policy"] == "global_median"
    assert meta["unit_prior_min_count"] == 5
    assert meta["unit_prior_default_count"] == 2
    assert meta["unit_prior_observed_count"] == 2
    assert meta["unit_prior_low_count_count"] == 1
    assert meta["unit_prior_unseen_count"] == 2
    assert meta["unit_prior_out_of_vocab_count"] == 1
    assert np.array_equal(cache["unit_log_prior_is_default"], np.asarray([0, 1, 1], dtype=np.int64))
    assert np.array_equal(cache["unit_log_prior_count"], np.asarray([7, 2, 0], dtype=np.int64))


def test_load_unit_log_prior_bundle_reads_npz_metadata(tmp_path):
    prior_path = tmp_path / "unit_prior_bundle.npz"
    np.savez(
        prior_path,
        unit_log_prior=np.asarray([0.0, 0.2, 0.4], dtype=np.float32),
        unit_prior_source=np.asarray(["demo"], dtype=object),
        unit_prior_version=np.asarray(["v2"], dtype=object),
        unit_prior_vocab_size=np.asarray([3], dtype=np.int64),
        global_speech_log_prior=np.asarray([0.31], dtype=np.float32),
        unit_prior_min_count=np.asarray([7], dtype=np.int64),
        unit_prior_default_policy=np.asarray(["global_median"], dtype=object),
        unit_log_prior_is_default=np.asarray([1, 0, 0], dtype=np.int64),
        unit_prior_filter_min_run_stability=np.asarray([0.35], dtype=np.float32),
        unit_prior_sha1=np.asarray(["deadbeef"], dtype=object),
        unit_prior_build_cmd=np.asarray(["python scripts/build_unit_log_prior.py --input demo"], dtype=object),
        unit_prior_input_manifest_sha1=np.asarray(["manifest"], dtype=object),
        unit_prior_input_count=np.asarray([3], dtype=np.int64),
        unit_prior_special_token_policy=np.asarray(["exclude_silence_runs_keep_special_backoff"], dtype=object),
        unit_prior_accepted_run_count=np.asarray([2], dtype=np.int64),
    )
    bundle = load_unit_log_prior_bundle(str(prior_path))
    assert np.allclose(bundle["unit_log_prior"], np.asarray([0.0, 0.2, 0.4], dtype=np.float32))
    assert bundle["unit_prior_source"] == "demo"
    assert bundle["unit_prior_version"] == "v2"
    assert bundle["unit_prior_vocab_size"] == 3
    assert bundle["unit_prior_min_count"] == 7
    assert bundle["unit_prior_default_policy"] == "global_median"
    assert bundle["unit_prior_default_value"] == pytest.approx(0.31)
    assert np.array_equal(bundle["unit_log_prior_is_default"], np.asarray([True, False, False]))
    assert bundle["unit_prior_filter_min_run_stability"] == pytest.approx(0.35)
    assert bundle["unit_prior_sha1"] == "deadbeef"
    assert bundle["unit_prior_build_cmd"] == "python scripts/build_unit_log_prior.py --input demo"
    assert bundle["unit_prior_input_manifest_sha1"] == "manifest"
    assert bundle["unit_prior_input_count"] == 3
    assert bundle["unit_prior_special_token_policy"] == "exclude_silence_runs_keep_special_backoff"
    assert bundle["unit_prior_accepted_run_count"] == 2


def test_load_unit_log_prior_bundle_reloads_when_path_stat_changes(tmp_path):
    prior_path = tmp_path / "unit_prior_bundle.npz"
    np.savez(
        prior_path,
        unit_log_prior=np.asarray([0.1], dtype=np.float32),
        unit_prior_version=np.asarray(["v1"], dtype=object),
    )
    first = load_unit_log_prior_bundle(str(prior_path))

    np.savez(
        prior_path,
        unit_log_prior=np.asarray([0.9], dtype=np.float32),
        unit_prior_version=np.asarray(["v2"], dtype=object),
    )
    bumped_mtime_ns = int(prior_path.stat().st_mtime_ns) + 1_000_000_000
    os.utime(prior_path, ns=(bumped_mtime_ns, bumped_mtime_ns))
    second = load_unit_log_prior_bundle(str(prior_path))

    assert first["unit_prior_version"] == "v1"
    assert float(first["unit_log_prior"][0]) == pytest.approx(0.1)
    assert second["unit_prior_version"] == "v2"
    assert float(second["unit_log_prior"][0]) == pytest.approx(0.9)
    assert second["unit_prior_path_mtime_ns"] == bumped_mtime_ns


def test_load_unit_log_prior_bundle_rejects_default_mask_size_mismatch(tmp_path):
    prior_path = tmp_path / "bad_prior.npz"
    np.savez(
        prior_path,
        unit_log_prior=np.asarray([0.0, 0.2, 0.4], dtype=np.float32),
        unit_log_prior_is_default=np.asarray([1, 0], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="size mismatch"):
        load_unit_log_prior_bundle(str(prior_path))


def test_load_unit_log_prior_bundle_inferrs_default_mask_from_counts(tmp_path):
    prior_path = tmp_path / "prior_counts.npz"
    np.savez(
        prior_path,
        unit_log_prior=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        unit_count=np.asarray([0, 3, 8], dtype=np.int64),
        unit_prior_min_count=np.asarray([5], dtype=np.int64),
    )

    bundle = load_unit_log_prior_bundle(str(prior_path))

    assert np.array_equal(bundle["unit_log_prior_is_default"], np.asarray([True, True, False]))
    assert bundle["unit_prior_observed_count"] == 2
    assert bundle["unit_prior_low_count_count"] == 1
    assert bundle["unit_prior_default_count"] == 2


def test_build_duration_v3_frontend_signature_tracks_global_stat_knobs(tmp_path):
    meta = build_source_rhythm_cache_v3(
        [1, 57, 2],
        silent_token=57,
    )[DURATION_V3_CACHE_META_KEY]
    prior_path = tmp_path / "unit_prior.npz"
    np.savez(prior_path, unit_log_prior=np.asarray([0.0, 0.2, 0.4], dtype=np.float32))
    signature_a = build_duration_v3_frontend_signature(
        meta,
        g_variant="raw_median",
        drop_edge_runs_for_g=0,
        unit_prior_path=None,
        summary_pool_speech_only=True,
        emit_silence_runs=True,
    )
    signature_b = build_duration_v3_frontend_signature(
        meta,
        g_variant="unit_norm",
        drop_edge_runs_for_g=1,
        unit_prior_path=str(prior_path),
        summary_pool_speech_only=False,
        emit_silence_runs=True,
    )
    assert signature_a != signature_b
    assert '"g_variant":"unit_norm"' in signature_b
    assert '"drop_edge_runs_for_g":1' in signature_b
    assert str(prior_path.resolve()).replace("\\", "\\\\") in signature_b
    assert '"unit_prior_bundle"' in signature_b


def test_build_duration_v3_frontend_signature_binds_prior_bundle_metadata(tmp_path):
    meta = build_source_rhythm_cache_v3(
        [1, 57, 2],
        silent_token=57,
        emit_silence_runs=True,
        debounce_min_run_frames=2,
    )[DURATION_V3_CACHE_META_KEY]
    prior_path = tmp_path / "unit_prior_meta.npz"
    np.savez(
        prior_path,
        unit_log_prior=np.asarray([0.0, 0.2, 0.4], dtype=np.float32),
        unit_prior_source=np.asarray(["demo"], dtype=object),
        unit_prior_version=np.asarray(["v3"], dtype=object),
        unit_prior_frontend_signature=np.asarray([duration_v3_cache_meta_signature(meta)], dtype=object),
        unit_prior_min_count=np.asarray([7], dtype=np.int64),
        unit_prior_default_policy=np.asarray(["global_median"], dtype=object),
        unit_prior_global_backoff=np.asarray(["linear"], dtype=object),
        unit_prior_emit_silence_runs=np.asarray([True], dtype=np.bool_),
        unit_prior_debounce_min_run_frames=np.asarray([2], dtype=np.int64),
        unit_prior_silent_token=np.asarray([57], dtype=np.int64),
        unit_prior_filter_only_sealed_runs=np.asarray([True], dtype=np.bool_),
        unit_prior_filter_drop_edge_runs=np.asarray([1], dtype=np.int64),
        unit_prior_filter_min_run_stability=np.asarray([0.45], dtype=np.float32),
        unit_prior_sha1=np.asarray(["deadbeef"], dtype=object),
        unit_prior_special_token_policy=np.asarray(["exclude_silence_runs_keep_special_backoff"], dtype=object),
    )

    signature = build_duration_v3_frontend_signature(
        meta,
        g_variant="unit_norm",
        drop_edge_runs_for_g=1,
        unit_prior_path=str(prior_path),
        summary_pool_speech_only=True,
        emit_silence_runs=True,
    )

    assert '"unit_prior_bundle"' in signature
    assert '"version":"v3"' in signature
    assert '"frontend_meta_signature"' in signature
    assert '"min_count":7' in signature
    assert '"default_policy":"global_median"' in signature
    assert '"global_backoff":"linear"' in signature
    assert '"filter_only_sealed_runs":true' in signature
    assert '"filter_min_run_stability":' in signature
    assert '"sha1":"deadbeef"' in signature
    assert '"special_token_policy":"exclude_silence_runs_keep_special_backoff"' in signature


def test_build_unit_log_prior_script_writes_global_default_metadata(tmp_path):
    input_path = tmp_path / "source_cache.pt"
    output_path = tmp_path / "unit_prior.npz"
    repo_root = Path(__file__).resolve().parents[2]
    torch.save(
        {
            "content_units": np.asarray([1, 1, 2], dtype=np.int64),
            "dur_anchor_src": np.asarray([1.0, 9.0, 100.0], dtype=np.float32),
            "source_silence_mask": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            "open_run_mask": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            "sealed_mask": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            "source_run_stability": np.asarray([0.9, 0.9, 0.9], dtype=np.float32),
            DURATION_V3_CACHE_META_KEY: {
                "cache_version": 3,
                "silent_token": 57,
                "separator_aware": True,
                "tail_open_units": 1,
                "emit_silence_runs": True,
                "debounce_min_run_frames": 2,
                "phrase_boundary_threshold": 0.55,
            },
        },
        input_path,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/build_unit_log_prior.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--min-count",
            "2",
            "--default-prior",
            "global_median",
            "--global-backoff",
            "linear",
            "--exclude-open-runs",
            "--min-run-stability",
            "0.5",
        ],
        check=True,
        cwd=str(repo_root),
        env={
            **os.environ,
            "PYTHONPATH": str(repo_root) + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
        },
    )

    with np.load(output_path, allow_pickle=True) as bundle:
        prior = bundle["unit_log_prior"]
        is_default = bundle["unit_log_prior_is_default"]
        assert bundle["unit_prior_min_count"].reshape(-1)[0] == 2
        assert bundle["unit_prior_default_policy"].reshape(-1)[0] == "global_median"
        assert bundle["unit_prior_global_backoff"].reshape(-1)[0] == "linear"
        assert bool(bundle["unit_prior_filter_exclude_open_runs"].reshape(-1)[0])
        assert bundle["unit_prior_filter_min_run_stability"].reshape(-1)[0] == pytest.approx(0.5)
        assert bundle["unit_prior_frontend_signature"].reshape(-1)[0] != "missing"
        assert int(bundle["unit_prior_silent_token"].reshape(-1)[0]) == 57
        assert bool(bundle["unit_prior_emit_silence_runs"].reshape(-1)[0])
        assert int(bundle["unit_prior_debounce_min_run_frames"].reshape(-1)[0]) == 2
        assert bundle["unit_prior_special_token_policy"].reshape(-1)[0] == "exclude_silence_runs_keep_special_backoff"
        assert int(bundle["unit_prior_input_count"].reshape(-1)[0]) == 1
        assert int(bundle["unit_prior_accepted_run_count"].reshape(-1)[0]) == 3
        assert len(str(bundle["unit_prior_input_manifest_sha1"].reshape(-1)[0])) == 40
        assert len(str(bundle["unit_prior_sha1"].reshape(-1)[0])) == 40
        assert "build_unit_log_prior.py" in str(bundle["unit_prior_build_cmd"].reshape(-1)[0])
        expected_global = np.log(9.0)
        assert bundle["global_speech_log_prior"].reshape(-1)[0] == pytest.approx(expected_global)
        assert bundle["unit_prior_observed_count"].reshape(-1)[0] == 2
        assert bundle["unit_prior_low_count_count"].reshape(-1)[0] == 1
        assert bundle["unit_prior_default_count"].reshape(-1)[0] == 2
        assert prior[1] == pytest.approx(np.log(3.0))
        expected_backoff = 0.5 * np.log(100.0) + 0.5 * expected_global
        assert prior[2] == pytest.approx(expected_backoff)
        assert np.array_equal(is_default, np.asarray([1, 0, 1], dtype=np.int64))


def test_build_unit_log_prior_script_filters_low_stability_runs(tmp_path):
    input_path = tmp_path / "source_cache.pt"
    output_path = tmp_path / "unit_prior_filtered.npz"
    repo_root = Path(__file__).resolve().parents[2]
    torch.save(
        {
            "content_units": np.asarray([1, 1, 2], dtype=np.int64),
            "dur_anchor_src": np.asarray([1.0, 9.0, 100.0], dtype=np.float32),
            "source_silence_mask": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            "open_run_mask": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            "sealed_mask": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            "source_run_stability": np.asarray([0.2, 0.9, 0.9], dtype=np.float32),
            DURATION_V3_CACHE_META_KEY: {
                "cache_version": 3,
                "silent_token": 57,
                "separator_aware": True,
                "tail_open_units": 1,
                "emit_silence_runs": True,
                "debounce_min_run_frames": 2,
                "phrase_boundary_threshold": 0.55,
            },
        },
        input_path,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/build_unit_log_prior.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--min-count",
            "1",
            "--min-run-stability",
            "0.5",
        ],
        check=True,
        cwd=str(repo_root),
        env={
            **os.environ,
            "PYTHONPATH": str(repo_root) + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
        },
    )

    with np.load(output_path, allow_pickle=True) as bundle:
        assert bundle["unit_prior_filter_min_run_stability"].reshape(-1)[0] == pytest.approx(0.5)
        assert int(bundle["unit_prior_accepted_run_count"].reshape(-1)[0]) == 2
        assert int(bundle["unit_prior_observed_count"].reshape(-1)[0]) == 2
        assert bundle["unit_log_prior"][1] == pytest.approx(np.log(9.0))
