from __future__ import annotations

import torch

from modules.Conan.rhythm_v3.math_utils import build_causal_source_prefix_rate_seq


def test_dual_timescale_prefix_rate_is_smoother_than_ema_on_alternating_signal():
    observed_log = torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]], dtype=torch.float32)
    speech_mask = torch.ones_like(observed_log)

    ema_seq, _ = build_causal_source_prefix_rate_seq(
        observed_log=observed_log,
        speech_mask=speech_mask,
        init_rate=None,
        default_init_rate=0.0,
        stat_mode="ema",
        decay=0.5,
    )
    dual_seq, _ = build_causal_source_prefix_rate_seq(
        observed_log=observed_log,
        speech_mask=speech_mask,
        init_rate=None,
        default_init_rate=0.0,
        stat_mode="dual_timescale",
        decay=0.5,
        decay_fast=0.35,
        decay_slow=0.95,
        slow_mix=0.8,
    )

    ema_std = float(ema_seq.std(unbiased=False).item())
    dual_std = float(dual_seq.std(unbiased=False).item())
    assert dual_std < ema_std
