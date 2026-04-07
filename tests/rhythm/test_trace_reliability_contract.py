from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.Conan.rhythm.contracts import TraceReliabilityBundle
from modules.Conan.rhythm.factory import build_streaming_rhythm_module_from_hparams


class TraceReliabilityContractTests(unittest.TestCase):
    @staticmethod
    def _dummy_reference(trace_bins: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
        stats = torch.tensor(
            [[0.20, 2.0, 4.0, 0.10, 0.30, 0.80]],
            dtype=torch.float32,
        )
        trace = torch.zeros((1, trace_bins, 5), dtype=torch.float32)
        trace[:, :, 0] = 0.10
        trace[:, :, 1] = torch.linspace(0.0, 1.0, trace_bins)
        trace[:, :, 2] = torch.linspace(0.0, 1.0, trace_bins)
        trace[:, :, 3] = torch.linspace(-0.2, 0.2, trace_bins)
        trace[:, :, 4] = 1.0
        return stats, trace

    def _build_module(self, *, with_offline_teacher: bool) -> object:
        hparams = {
            "hidden_size": 16,
            "rhythm_hidden_size": 16,
            "content_vocab_size": 32,
            "rhythm_trace_bins": 8,
            "rhythm_emit_reference_sidecar": True,
            "rhythm_trace_reliability_enable": True,
            "rhythm_trace_exhaustion_gap_start": 0.08,
            "rhythm_trace_exhaustion_gap_end": 0.22,
            "rhythm_trace_exhaustion_local_floor": 0.20,
            "rhythm_trace_exhaustion_boundary_floor": 0.05,
        }
        if with_offline_teacher:
            hparams.update(
                {
                    "rhythm_stage": "teacher_offline",
                    "rhythm_enable_learned_offline_teacher": True,
                    "rhythm_runtime_enable_learned_offline_teacher": True,
                }
            )
        return build_streaming_rhythm_module_from_hparams(hparams)

    def test_sample_trace_pair_returns_trace_reliability_bundle(self) -> None:
        module = self._build_module(with_offline_teacher=False)
        stats, trace = self._dummy_reference(trace_bins=8)
        conditioning = module.build_reference_conditioning(
            ref_rhythm_stats=stats,
            ref_rhythm_trace=trace,
        )
        unit_mask = torch.ones((1, 4), dtype=torch.float32)
        dur_anchor_src = torch.tensor([[2.0, 3.0, 1.0, 2.0]], dtype=torch.float32)
        state = module.init_state(batch_size=1, device=dur_anchor_src.device)
        state.phase_ptr = torch.tensor([0.85], dtype=torch.float32)
        state.phase_anchor = torch.tensor([[2.0, 8.0]], dtype=torch.float32)

        trace_context, planner_trace_context, reliability = module._sample_trace_pair(
            ref_conditioning=conditioning,
            phase_ptr=state.phase_ptr,
            window_size=unit_mask.size(1),
            unit_mask=unit_mask,
            dur_anchor_src=dur_anchor_src,
            horizon=0.35,
            state=state,
        )

        self.assertEqual(tuple(trace_context.shape), (1, 4, 5))
        self.assertEqual(tuple(planner_trace_context.shape), (1, 4, 2))
        self.assertIsInstance(reliability, TraceReliabilityBundle)
        self.assertEqual(tuple(reliability.trace_reliability.shape), (1,))
        self.assertEqual(tuple(reliability.local_trace_path_weight.shape), (1,))

    def test_forward_uses_three_value_trace_pair_contract(self) -> None:
        module = self._build_module(with_offline_teacher=False)
        stats, trace = self._dummy_reference(trace_bins=8)
        execution = module(
            content_units=torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            dur_anchor_src=torch.tensor([[2.0, 3.0, 1.0, 2.0]], dtype=torch.float32),
            ref_rhythm_stats=stats,
            ref_rhythm_trace=trace,
            unit_mask=torch.ones((1, 4), dtype=torch.float32),
        )
        self.assertEqual(tuple(execution.speech_duration_exec.shape), (1, 4))
        self.assertIsNotNone(execution.next_state)

    def test_forward_teacher_uses_three_value_trace_pair_contract(self) -> None:
        module = self._build_module(with_offline_teacher=True)
        stats, trace = self._dummy_reference(trace_bins=8)
        execution, confidence = module.forward_teacher(
            content_units=torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            dur_anchor_src=torch.tensor([[2.0, 3.0, 1.0, 2.0]], dtype=torch.float32),
            ref_rhythm_stats=stats,
            ref_rhythm_trace=trace,
            unit_mask=torch.ones((1, 4), dtype=torch.float32),
        )
        self.assertEqual(tuple(execution.speech_duration_exec.shape), (1, 4))
        self.assertTrue(confidence)
        self.assertIn("overall", confidence)


if __name__ == "__main__":
    unittest.main()
