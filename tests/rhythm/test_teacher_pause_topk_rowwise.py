from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.Conan.rhythm.teacher import build_algorithmic_teacher_targets


def _sample_a_inputs():
    dur_anchor_src = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
    unit_mask = torch.ones_like(dur_anchor_src)
    ref_rhythm_stats = torch.tensor([[0.20, 1.0, 1.0, 0.0, 0.10, 0.8]], dtype=torch.float32)
    ref_rhythm_trace = torch.tensor(
        [[
            [0.1, 0.0, 0.1, 0.0, 1.0],
            [0.1, 0.0, 0.2, 0.0, 1.0],
            [0.1, 0.0, 0.3, 0.0, 1.0],
            [0.1, 0.0, 0.4, 0.0, 1.0],
        ]],
        dtype=torch.float32,
    )
    return dur_anchor_src, ref_rhythm_stats, ref_rhythm_trace, unit_mask


def test_first_sample_pause_targets_are_batch_local() -> None:
    dur_a, stats_a, trace_a, mask_a = _sample_a_inputs()
    single = build_algorithmic_teacher_targets(
        dur_anchor_src=dur_a,
        ref_rhythm_stats=stats_a,
        ref_rhythm_trace=trace_a,
        unit_mask=mask_a,
    )

    dur_b = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
    stats_b = torch.tensor([[0.20, 1.0, 1.0, 0.0, 0.95, 0.8]], dtype=torch.float32)
    trace_b = trace_a.clone()
    batch = build_algorithmic_teacher_targets(
        dur_anchor_src=torch.cat([dur_a, dur_b], dim=0),
        ref_rhythm_stats=torch.cat([stats_a, stats_b], dim=0),
        ref_rhythm_trace=torch.cat([trace_a, trace_b], dim=0),
        unit_mask=torch.cat([mask_a, mask_a], dim=0),
    )

    assert torch.allclose(single.pause_exec_tgt[0], batch.pause_exec_tgt[0], atol=1e-6, rtol=1e-6)
