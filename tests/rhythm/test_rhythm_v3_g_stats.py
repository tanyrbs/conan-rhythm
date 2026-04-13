from __future__ import annotations

import torch

import pytest

from modules.Conan.rhythm_v3.g_stats import (
    build_global_rate_support_mask,
    compute_global_rate,
    summarize_global_rate_support,
    true_median_1d,
    weighted_median_1d,
)


def test_compute_global_rate_raw_median_ignores_silence_duration_changes():
    speech_mask = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32)
    valid_mask = torch.ones_like(speech_mask)
    log_dur_a = torch.log(torch.tensor([[4.0, 1.0, 8.0]], dtype=torch.float32))
    log_dur_b = torch.log(torch.tensor([[4.0, 100.0, 8.0]], dtype=torch.float32))

    g_a = compute_global_rate(
        log_dur=log_dur_a,
        speech_mask=speech_mask,
        valid_mask=valid_mask,
        variant="raw_median",
    )
    g_b = compute_global_rate(
        log_dur=log_dur_b,
        speech_mask=speech_mask,
        valid_mask=valid_mask,
        variant="raw_median",
    )

    assert torch.allclose(g_a, g_b)


def test_build_global_rate_support_mask_does_not_fall_back_to_valid_only_support():
    speech_mask = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    valid_mask = torch.ones_like(speech_mask)

    support = build_global_rate_support_mask(
        speech_mask=speech_mask,
        valid_mask=valid_mask,
        drop_edge_runs=0,
    )

    assert torch.equal(support, torch.zeros_like(speech_mask, dtype=torch.bool))


def test_build_global_rate_support_mask_keeps_raw_speech_support_when_edge_drop_empties_it():
    speech_mask = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32)
    valid_mask = torch.ones_like(speech_mask)

    support = build_global_rate_support_mask(
        speech_mask=speech_mask,
        valid_mask=valid_mask,
        drop_edge_runs=1,
    )

    assert torch.equal(support, torch.tensor([[True, False, True]]))


def test_compute_global_rate_raises_when_no_speech_support_exists():
    speech_mask = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    valid_mask = torch.ones_like(speech_mask)
    log_dur = torch.log(torch.tensor([[4.0, 1.0, 8.0]], dtype=torch.float32))

    with pytest.raises(ValueError, match="No valid speech duration|at least one valid item per batch row"):
        compute_global_rate(
            log_dur=log_dur,
            speech_mask=speech_mask,
            valid_mask=valid_mask,
            variant="raw_median",
        )


def test_compute_global_rate_raw_median_batch_fast_path_matches_expected_support():
    log_dur = torch.log(torch.tensor([[4.0, 1.0, 8.0], [2.0, 3.0, 9.0]], dtype=torch.float32))
    speech_mask = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=torch.float32)
    valid_mask = torch.ones_like(speech_mask)

    g = compute_global_rate(
        log_dur=log_dur,
        speech_mask=speech_mask,
        valid_mask=valid_mask,
        variant="raw_median",
    )

    expected = torch.stack(
        [
            0.5 * (torch.log(torch.tensor(4.0)) + torch.log(torch.tensor(8.0))),
            0.5 * (torch.log(torch.tensor(3.0)) + torch.log(torch.tensor(9.0))),
        ]
    ).reshape(2, 1)
    assert torch.allclose(g, expected)


def test_true_median_uses_average_for_even_support():
    values = torch.log(torch.tensor([2.0, 8.0], dtype=torch.float32))

    median = true_median_1d(values)

    expected = 0.5 * (torch.log(torch.tensor(2.0)) + torch.log(torch.tensor(8.0)))
    assert torch.allclose(median, expected)


def test_weighted_median_with_unit_weights_matches_raw_true_median():
    values = torch.log(torch.tensor([2.0, 4.0, 8.0, 16.0], dtype=torch.float32))
    weights = torch.ones_like(values)

    weighted = weighted_median_1d(values, weights)
    raw = true_median_1d(values)

    assert torch.allclose(weighted, raw)


def test_compute_global_rate_trimmed_mean_batch_fast_path_ignores_trimmed_extrema():
    log_dur = torch.log(torch.tensor([[1.0, 2.0, 100.0, 200.0]], dtype=torch.float32))
    speech_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
    valid_mask = torch.ones_like(speech_mask)

    g = compute_global_rate(
        log_dur=log_dur,
        speech_mask=speech_mask,
        valid_mask=valid_mask,
        variant="trimmed_mean",
        trim_ratio=0.25,
    )

    expected = (
        torch.log(torch.tensor([[2.0]], dtype=torch.float32))
        + torch.log(torch.tensor([[100.0]], dtype=torch.float32))
    ) / 2.0
    assert torch.allclose(g, expected)


def test_compute_global_rate_raw_median_batch_fast_path_supports_edge_drop():
    log_dur = torch.log(
        torch.tensor(
            [
                [2.0, 4.0, 8.0, 16.0],
                [3.0, 5.0, 7.0, 11.0],
            ],
            dtype=torch.float32,
        )
    )
    speech_mask = torch.ones_like(log_dur)
    valid_mask = torch.ones_like(log_dur)

    g = compute_global_rate(
        log_dur=log_dur,
        speech_mask=speech_mask,
        valid_mask=valid_mask,
        variant="raw_median",
        drop_edge_runs=1,
    )

    expected = torch.stack(
        [
            0.5 * (torch.log(torch.tensor(4.0)) + torch.log(torch.tensor(8.0))),
            0.5 * (torch.log(torch.tensor(5.0)) + torch.log(torch.tensor(7.0))),
        ]
    ).reshape(2, 1)
    assert torch.allclose(g, expected)


def test_compute_global_rate_weighted_median_prefers_high_confidence_support():
    log_dur = torch.log(torch.tensor([[2.0, 10.0, 30.0]], dtype=torch.float32))
    speech_mask = torch.ones_like(log_dur)
    weight = torch.tensor([[8.0, 1.0, 1.0]], dtype=torch.float32)

    g = compute_global_rate(
        log_dur=log_dur,
        speech_mask=speech_mask,
        weight=weight,
        variant="weighted_median",
    )

    assert torch.allclose(g, torch.log(torch.tensor([[2.0]], dtype=torch.float32)))


def test_compute_global_rate_weighted_median_with_drop_edge_runs_uses_trimmed_support():
    log_dur = torch.log(
        torch.tensor(
            [
                [2.0, 4.0, 8.0, 16.0],
                [3.0, 5.0, 7.0, 11.0],
            ],
            dtype=torch.float32,
        )
    )
    speech_mask = torch.ones_like(log_dur)
    weight = torch.ones_like(log_dur)

    g = compute_global_rate(
        log_dur=log_dur,
        speech_mask=speech_mask,
        weight=weight,
        variant="weighted_median",
        drop_edge_runs=1,
    )

    expected = torch.stack(
        [
            0.5 * (torch.log(torch.tensor(4.0)) + torch.log(torch.tensor(8.0))),
            0.5 * (torch.log(torch.tensor(5.0)) + torch.log(torch.tensor(7.0))),
        ]
    ).reshape(2, 1)
    assert torch.allclose(g, expected)


def test_build_global_rate_support_mask_can_prefer_boundary_clean_closed_support():
    speech_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
    closed_mask = torch.tensor([[0.0, 1.0, 1.0, 0.0]], dtype=torch.float32)
    boundary_confidence = torch.tensor([[0.2, 0.8, 0.9, 0.3]], dtype=torch.float32)

    support = build_global_rate_support_mask(
        speech_mask=speech_mask,
        closed_mask=closed_mask,
        boundary_confidence=boundary_confidence,
        min_boundary_confidence=0.7,
    )

    assert torch.equal(support, torch.tensor([[False, True, True, False]]))


def test_build_global_rate_support_mask_falls_back_when_boundary_clean_support_is_empty():
    speech_mask = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32)
    closed_mask = torch.zeros_like(speech_mask)
    boundary_confidence = torch.zeros_like(speech_mask)

    support = build_global_rate_support_mask(
        speech_mask=speech_mask,
        closed_mask=closed_mask,
        boundary_confidence=boundary_confidence,
        min_boundary_confidence=0.9,
    )

    assert torch.equal(support, torch.tensor([[True, False, True]]))


def test_summarize_global_rate_support_reports_clean_count():
    speech_mask = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32)
    closed_mask = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    boundary_confidence = torch.tensor([[0.8, 0.9, 0.0]], dtype=torch.float32)

    summary = summarize_global_rate_support(
        speech_mask=speech_mask,
        closed_mask=closed_mask,
        boundary_confidence=boundary_confidence,
        min_boundary_confidence=0.7,
    )

    assert torch.equal(summary.clean_mask, torch.tensor([[True, False, False]]))
    assert torch.allclose(summary.clean_count, torch.tensor([[1.0]]))


def test_compute_global_rate_unit_norm_accepts_default_filled_run_prior():
    log_dur = torch.log(torch.tensor([[4.0, 8.0]], dtype=torch.float32))
    speech_mask = torch.ones_like(log_dur)
    unit_ids = torch.tensor([[1, 99]], dtype=torch.long)
    run_prior = torch.log(torch.tensor([[2.0, 4.0]], dtype=torch.float32))

    g = compute_global_rate(
        log_dur=log_dur,
        speech_mask=speech_mask,
        unit_ids=unit_ids,
        unit_prior=run_prior,
        variant="unit_norm",
    )

    expected = torch.log(torch.tensor([[2.0]], dtype=torch.float32))
    assert torch.allclose(g, expected)


def test_compute_global_rate_unit_norm_uses_default_for_oov_vocab_prior():
    log_dur = torch.log(torch.tensor([[4.0, 8.0]], dtype=torch.float32))
    speech_mask = torch.ones_like(log_dur)
    unit_ids = torch.tensor([[1, 99]], dtype=torch.long)
    unit_prior = torch.log(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))

    g = compute_global_rate(
        log_dur=log_dur,
        speech_mask=speech_mask,
        unit_ids=unit_ids,
        unit_prior=unit_prior,
        unit_prior_default_value=float(torch.log(torch.tensor(4.0)).item()),
        variant="unit_norm",
    )

    expected = true_median_1d(
        torch.tensor(
            [
                torch.log(torch.tensor(4.0)) - torch.log(torch.tensor(2.0)),
                torch.log(torch.tensor(8.0)) - torch.log(torch.tensor(4.0)),
            ],
            dtype=torch.float32,
        )
    ).reshape(1, 1)
    assert torch.allclose(g, expected)


def test_compute_global_rate_unit_norm_applies_count_aware_backoff():
    log_dur = torch.log(torch.tensor([[8.0]], dtype=torch.float32))
    speech_mask = torch.ones_like(log_dur)
    unit_ids = torch.tensor([[2]], dtype=torch.long)
    unit_prior = torch.log(torch.tensor([1.0, 2.0, 16.0], dtype=torch.float32))
    unit_count = torch.tensor([10, 10, 1], dtype=torch.long)
    global_backoff = float(torch.log(torch.tensor(4.0)).item())

    g = compute_global_rate(
        log_dur=log_dur,
        speech_mask=speech_mask,
        unit_ids=unit_ids,
        unit_prior=unit_prior,
        unit_count=unit_count,
        unit_prior_min_count=4,
        unit_prior_global_backoff=global_backoff,
        unit_prior_default_value=global_backoff,
        variant="unit_norm",
    )

    alpha = 1.0 / 4.0
    resolved_prior = (alpha * float(torch.log(torch.tensor(16.0)).item())) + (
        (1.0 - alpha) * global_backoff
    )
    expected = torch.tensor(
        [[float(torch.log(torch.tensor(8.0)).item()) - resolved_prior]],
        dtype=torch.float32,
    )
    assert torch.allclose(g, expected)


def test_compute_global_rate_unit_norm_rejects_oov_vocab_prior_without_default():
    log_dur = torch.log(torch.tensor([[4.0, 8.0]], dtype=torch.float32))
    speech_mask = torch.ones_like(log_dur)
    unit_ids = torch.tensor([[1, 99]], dtype=torch.long)
    unit_prior = torch.log(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))

    with pytest.raises(ValueError, match="unit_prior_default_value"):
        compute_global_rate(
            log_dur=log_dur,
            speech_mask=speech_mask,
            unit_ids=unit_ids,
            unit_prior=unit_prior,
            variant="unit_norm",
        )
