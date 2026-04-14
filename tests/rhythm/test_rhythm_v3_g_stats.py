from __future__ import annotations

import torch

import pytest

from modules.Conan.rhythm_v3.g_stats import (
    build_global_rate_support_mask,
    build_softclean_weights,
    compute_global_rate,
    masked_weighted_median_batch,
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


def test_weighted_median_raises_when_total_weight_is_nonpositive_by_default():
    values = torch.log(torch.tensor([2.0, 4.0, 8.0], dtype=torch.float32))
    weights = torch.zeros_like(values)

    with pytest.raises(ValueError, match="positive finite total weight"):
        weighted_median_1d(values, weights)


def test_weighted_median_can_return_nan_for_invalid_total_weight():
    values = torch.log(torch.tensor([2.0, 4.0, 8.0], dtype=torch.float32))
    weights = torch.zeros_like(values)

    median = weighted_median_1d(
        values,
        weights,
        invalid_weight_behavior="nan",
    )

    assert torch.isnan(median)


def test_weighted_median_can_explicitly_use_legacy_fallback_for_invalid_total_weight():
    values = torch.log(torch.tensor([2.0, 4.0, 8.0, 16.0], dtype=torch.float32))
    weights = torch.zeros_like(values)

    median = weighted_median_1d(
        values,
        weights,
        invalid_weight_behavior="fallback",
    )

    assert torch.allclose(median, true_median_1d(values))


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


def test_build_softclean_weights_applies_closed_mask_and_boundary_soft_floor():
    speech_mask = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
    valid_mask = torch.ones_like(speech_mask)
    closed_mask = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32)
    boundary_confidence = torch.tensor([[1.0, 0.2, 0.5]], dtype=torch.float32)

    weights = build_softclean_weights(
        speech_mask=speech_mask,
        valid_mask=valid_mask,
        closed_mask=closed_mask,
        boundary_confidence=boundary_confidence,
        weight_floor=0.2,
    )

    expected = torch.tensor([[1.0, 0.0, 0.6]], dtype=torch.float32)
    assert torch.allclose(weights, expected)


def test_compute_global_rate_softclean_wmed_prefers_high_weight_rows_without_clean_hard_gate():
    log_dur = torch.log(torch.tensor([[2.0, 10.0, 30.0]], dtype=torch.float32))
    speech_mask = torch.ones_like(log_dur)
    weight = torch.tensor([[1.0, 0.0, 0.6]], dtype=torch.float32)

    g = compute_global_rate(
        log_dur=log_dur,
        speech_mask=speech_mask,
        weight=weight,
        variant="softclean_wmed",
        support_mask=torch.ones_like(speech_mask, dtype=torch.bool),
    )

    assert torch.allclose(g, torch.log(torch.tensor([[2.0]], dtype=torch.float32)))


def test_compute_global_rate_softclean_wtmean_uses_weighted_trimmed_mean():
    log_dur = torch.log(torch.tensor([[2.0, 10.0, 30.0]], dtype=torch.float32))
    speech_mask = torch.ones_like(log_dur)
    weight = torch.tensor([[1.0, 0.0, 0.6]], dtype=torch.float32)

    g = compute_global_rate(
        log_dur=log_dur,
        speech_mask=speech_mask,
        weight=weight,
        variant="softclean_wtmean",
        trim_ratio=0.0,
        support_mask=torch.ones_like(speech_mask, dtype=torch.bool),
    )

    expected = (
        (torch.log(torch.tensor(2.0)) * 1.0)
        + (torch.log(torch.tensor(30.0)) * 0.6)
    ) / 1.6
    assert torch.allclose(g, expected.reshape(1, 1))


def test_compute_global_rate_weighted_median_raises_for_zero_weight_support_by_default():
    log_dur = torch.log(torch.tensor([[2.0, 10.0, 30.0]], dtype=torch.float32))
    speech_mask = torch.ones_like(log_dur)
    weight = torch.zeros_like(log_dur)

    with pytest.raises(ValueError, match="positive finite total weight"):
        compute_global_rate(
            log_dur=log_dur,
            speech_mask=speech_mask,
            weight=weight,
            variant="weighted_median",
        )


def test_compute_global_rate_weighted_median_can_return_nan_for_zero_weight_support():
    log_dur = torch.log(torch.tensor([[2.0, 10.0, 30.0]], dtype=torch.float32))
    speech_mask = torch.ones_like(log_dur)
    weight = torch.zeros_like(log_dur)

    g = compute_global_rate(
        log_dur=log_dur,
        speech_mask=speech_mask,
        weight=weight,
        variant="weighted_median",
        invalid_weight_behavior="nan",
    )

    assert bool(torch.isnan(g).all().item())


def test_compute_global_rate_weighted_median_can_explicitly_use_legacy_fallback():
    log_dur = torch.log(torch.tensor([[2.0, 4.0, 8.0, 16.0]], dtype=torch.float32))
    speech_mask = torch.ones_like(log_dur)
    weight = torch.zeros_like(log_dur)

    g = compute_global_rate(
        log_dur=log_dur,
        speech_mask=speech_mask,
        weight=weight,
        variant="weighted_median",
        invalid_weight_behavior="fallback",
    )

    expected = true_median_1d(log_dur.reshape(-1)).reshape(1, 1)
    assert torch.allclose(g, expected)


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
    assert torch.allclose(summary.support_seed_count, torch.tensor([[1.0]]))
    assert torch.allclose(summary.support_fraction, torch.tensor([[0.5]]))
    assert torch.allclose(summary.edge_runs_dropped, torch.tensor([[0.0]]))


def test_summarize_global_rate_support_reports_edge_drop_from_support_seed():
    speech_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)

    summary = summarize_global_rate_support(
        speech_mask=speech_mask,
        drop_edge_runs=1,
    )

    assert torch.equal(summary.support_mask, torch.tensor([[False, True, True, False]]))
    assert torch.allclose(summary.support_seed_count, torch.tensor([[4.0]]))
    assert torch.allclose(summary.support_count, torch.tensor([[2.0]]))
    assert torch.allclose(summary.support_fraction, torch.tensor([[0.5]]))
    assert torch.allclose(summary.edge_runs_dropped, torch.tensor([[2.0]]))


def test_summarize_global_rate_support_tracks_duration_weighted_and_count_speech_ratios():
    speech_mask = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32)
    valid_mask = torch.ones_like(speech_mask)
    duration_obs = torch.tensor([[1.0, 100.0, 1.0]], dtype=torch.float32)

    summary = summarize_global_rate_support(
        speech_mask=speech_mask,
        valid_mask=valid_mask,
        duration_obs=duration_obs,
        min_speech_ratio=0.5,
    )

    assert torch.allclose(summary.speech_ratio_count, torch.tensor([[2.0 / 3.0]], dtype=torch.float32))
    assert torch.allclose(summary.speech_ratio, torch.tensor([[2.0 / 102.0]], dtype=torch.float32))
    assert torch.allclose(summary.domain_valid, torch.tensor([[0.0]], dtype=torch.float32))


def test_summarize_global_rate_support_exposes_control_valid_dynamic_range_gate():
    speech_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
    duration_obs = torch.tensor([[4.0, 4.02, 4.01, 4.03]], dtype=torch.float32)

    summary = summarize_global_rate_support(
        speech_mask=speech_mask,
        duration_obs=duration_obs,
        min_speech_runs=3,
        min_support_log_iqr=0.08,
        min_support_log_span=0.18,
        min_support_unique_count=3,
    )

    assert torch.allclose(summary.domain_valid, torch.tensor([[1.0]], dtype=torch.float32))
    assert torch.allclose(summary.control_valid, torch.tensor([[0.0]], dtype=torch.float32))
    assert float(summary.support_log_span.reshape(-1)[0].item()) < 0.18


def test_compute_global_rate_accepts_reused_support_mask_for_weighted_median():
    log_dur = torch.log(torch.tensor([[2.0, 4.0, 8.0, 16.0]], dtype=torch.float32))
    speech_mask = torch.ones_like(log_dur)
    weight = torch.tensor([[1.0, 10.0, 1.0, 1.0]], dtype=torch.float32)
    support = build_global_rate_support_mask(
        speech_mask=speech_mask,
        drop_edge_runs=1,
    )

    g = compute_global_rate(
        log_dur=log_dur,
        speech_mask=speech_mask,
        weight=weight,
        variant="weighted_median",
        support_mask=support,
    )

    assert torch.allclose(g, torch.log(torch.tensor([[4.0]], dtype=torch.float32)))


def test_masked_weighted_median_batch_nan_policy_flags_only_invalid_rows():
    values = torch.log(torch.tensor([[2.0, 4.0], [8.0, 16.0]], dtype=torch.float32))
    mask = torch.ones_like(values, dtype=torch.bool)
    weights = torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.float32)

    median = masked_weighted_median_batch(
        values,
        mask,
        weights,
        invalid_weight_behavior="nan",
    )

    assert torch.isnan(median[0, 0])
    assert torch.allclose(median[1], torch.log(torch.tensor([16.0], dtype=torch.float32)))


def test_compute_global_rate_accepts_support_stats_for_unit_norm():
    log_dur = torch.log(torch.tensor([[4.0, 8.0, 16.0]], dtype=torch.float32))
    speech_mask = torch.ones_like(log_dur)
    unit_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    unit_prior = torch.log(torch.tensor([2.0, 4.0, 8.0], dtype=torch.float32))
    support_stats = summarize_global_rate_support(
        speech_mask=speech_mask,
        drop_edge_runs=1,
    )

    g = compute_global_rate(
        log_dur=log_dur,
        speech_mask=speech_mask,
        unit_ids=unit_ids,
        unit_prior=unit_prior,
        variant="unit_norm",
        support_stats=support_stats,
    )

    assert torch.allclose(g, torch.log(torch.tensor([[2.0]], dtype=torch.float32)))


def test_compute_global_rate_rejects_conflicting_support_inputs():
    log_dur = torch.log(torch.tensor([[4.0, 8.0]], dtype=torch.float32))
    speech_mask = torch.ones_like(log_dur)
    support_stats = summarize_global_rate_support(speech_mask=speech_mask)

    with pytest.raises(ValueError, match="at most one of support_mask or support_stats"):
        compute_global_rate(
            log_dur=log_dur,
            speech_mask=speech_mask,
            support_mask=support_stats.support_mask,
            support_stats=support_stats,
        )


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


def test_compute_global_rate_unit_norm_weighted_raises_for_zero_weight_support_by_default():
    log_dur = torch.log(torch.tensor([[4.0, 8.0]], dtype=torch.float32))
    speech_mask = torch.ones_like(log_dur)
    unit_ids = torch.tensor([[0, 1]], dtype=torch.long)
    unit_prior = torch.log(torch.tensor([2.0, 4.0], dtype=torch.float32))
    weight = torch.zeros_like(log_dur)

    with pytest.raises(ValueError, match="positive finite total weight"):
        compute_global_rate(
            log_dur=log_dur,
            speech_mask=speech_mask,
            unit_ids=unit_ids,
            unit_prior=unit_prior,
            weight=weight,
            variant="unit_norm",
        )


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
