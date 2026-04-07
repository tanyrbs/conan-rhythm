from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.Conan.rhythm.module import StreamingRhythmModule


def _build_inputs():
    content_units = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    dur_anchor_src = torch.tensor([[2.0, 3.0, 2.0, 1.0]], dtype=torch.float32)
    unit_mask = torch.ones_like(dur_anchor_src)
    ref_rhythm_stats = torch.tensor(
        [[0.20, 1.50, 3.00, 0.00, 0.40, 0.80]],
        dtype=torch.float32,
    )
    ref_rhythm_trace = torch.rand((1, 24, 5), dtype=torch.float32)
    return content_units, dur_anchor_src, unit_mask, ref_rhythm_stats, ref_rhythm_trace


def test_sample_trace_pair_returns_reliability_sidecar() -> None:
    module = StreamingRhythmModule(enable_trace_exhaustion_fallback=True)
    content_units, dur_anchor_src, unit_mask, ref_rhythm_stats, ref_rhythm_trace = _build_inputs()
    ref_conditioning = module.build_reference_conditioning(
        ref_rhythm_stats=ref_rhythm_stats,
        ref_rhythm_trace=ref_rhythm_trace,
    )
    state = module.init_state(batch_size=1, device=dur_anchor_src.device)
    outputs = module._sample_trace_pair(
        ref_conditioning=ref_conditioning,
        phase_ptr=state.phase_ptr,
        window_size=content_units.size(1),
        unit_mask=unit_mask,
        dur_anchor_src=dur_anchor_src,
        horizon=None,
        state=state,
    )
    assert len(outputs) == 3
    trace_context, planner_trace_context, reliability = outputs
    assert trace_context.shape[:2] == content_units.shape
    assert planner_trace_context.shape[:2] == content_units.shape
    assert isinstance(reliability, dict)
    assert "local_gate" in reliability
    assert "blend" in reliability


def test_forward_exposes_trace_reliability_after_unpack() -> None:
    module = StreamingRhythmModule(enable_trace_exhaustion_fallback=True)
    content_units, dur_anchor_src, unit_mask, ref_rhythm_stats, ref_rhythm_trace = _build_inputs()
    execution = module(
        content_units=content_units,
        dur_anchor_src=dur_anchor_src,
        ref_rhythm_stats=ref_rhythm_stats,
        ref_rhythm_trace=ref_rhythm_trace,
        unit_mask=unit_mask,
        open_run_mask=torch.zeros_like(unit_mask),
        sealed_mask=torch.ones_like(unit_mask),
    )
    assert hasattr(execution, "trace_reliability")
    assert "local_gate" in execution.trace_reliability
    assert "blend" in execution.trace_reliability
