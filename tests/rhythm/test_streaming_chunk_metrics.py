from __future__ import annotations

from types import SimpleNamespace

from tasks.Conan.rhythm.metrics import build_streaming_chunk_metrics


def test_streaming_chunk_metrics_returns_zeros_for_empty_history():
    metrics = build_streaming_chunk_metrics(
        SimpleNamespace(
            mel_lengths=[],
            commit_history=[],
            committed_mel_lengths=[],
            prefix_exec_deltas=[],
            backlog_history=[],
            clock_history=[],
            blank_ratio_history=[],
        )
    )
    assert metrics["stream_num_chunks"] == 0.0
    assert metrics["stream_no_rollback_violations"] == 0.0


def test_streaming_chunk_metrics_reports_clean_monotonic_history():
    metrics = build_streaming_chunk_metrics(
        SimpleNamespace(
            mel_lengths=[10, 20, 28],
            commit_history=[[2], [4], [5]],
            committed_mel_lengths=[8, 16, 24],
            prefix_exec_deltas=[0.0, 0.0, 0.0],
            backlog_history=[0.1, 0.2, 0.15],
            clock_history=[0.0, -0.1, 0.05],
            blank_ratio_history=[0.0, 0.0, 0.0],
        )
    )
    assert metrics["stream_commit_monotonic_violations"] == 0.0
    assert metrics["stream_committed_mel_rollback_violations"] == 0.0
    assert metrics["stream_no_rollback_violations"] == 0.0


def test_streaming_chunk_metrics_reports_rollbacks():
    metrics = build_streaming_chunk_metrics(
        SimpleNamespace(
            mel_lengths=[10, 18, 26],
            commit_history=[[3], [2], [5]],
            committed_mel_lengths=[8, 7, 20],
            prefix_exec_deltas=[0.0, 0.2, 0.0],
            backlog_history=[0.1, 0.4, 0.2],
            clock_history=[0.0, -0.3, 0.1],
            blank_ratio_history=[0.0, 0.1, 0.0],
        )
    )
    assert metrics["stream_commit_monotonic_violations"] > 0.0
    assert metrics["stream_committed_mel_rollback_violations"] > 0.0
    assert metrics["stream_no_rollback_violations"] > 0.0
