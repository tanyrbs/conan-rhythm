from __future__ import annotations

import torch

import pytest

from modules.Conan.rhythm_v3.g_stats import build_global_rate_support_mask, compute_global_rate


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

    with pytest.raises(ValueError, match="No valid speech duration"):
        compute_global_rate(
            log_dur=log_dur,
            speech_mask=speech_mask,
            valid_mask=valid_mask,
            variant="raw_median",
        )
