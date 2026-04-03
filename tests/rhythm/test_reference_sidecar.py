from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.Conan.rhythm.factory import build_streaming_rhythm_module_from_hparams
from tasks.Conan.rhythm.runtime_modes import build_rhythm_ref_conditioning


class ReferenceSidecarTests(unittest.TestCase):
    @staticmethod
    def _dummy_inputs(trace_bins: int = 8):
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

    def test_factory_can_enable_reference_sidecar(self) -> None:
        module = build_streaming_rhythm_module_from_hparams(
            {
                "hidden_size": 16,
                "rhythm_hidden_size": 16,
                "rhythm_trace_bins": 8,
                "rhythm_emit_reference_sidecar": True,
            }
        )
        self.assertTrue(module.reference_descriptor.emit_reference_sidecar)
        self.assertIsNotNone(module.reference_descriptor.selector)

    def test_build_reference_conditioning_from_cached_surfaces_emits_sidecar_when_enabled(self) -> None:
        module = build_streaming_rhythm_module_from_hparams(
            {
                "hidden_size": 16,
                "rhythm_hidden_size": 16,
                "rhythm_trace_bins": 8,
                "rhythm_emit_reference_sidecar": True,
            }
        )
        stats, trace = self._dummy_inputs(trace_bins=8)
        conditioning = module.build_reference_conditioning(
            ref_rhythm_stats=stats,
            ref_rhythm_trace=trace,
        )
        self.assertIn("slow_rhythm_memory", conditioning)
        self.assertIn("slow_rhythm_summary", conditioning)
        self.assertIn("planner_slow_rhythm_memory", conditioning)
        self.assertIn("planner_slow_rhythm_summary", conditioning)
        self.assertEqual(tuple(conditioning["planner_slow_rhythm_memory"].shape[-2:]), (6, 2))

    def test_build_reference_conditioning_respects_input_sidecars(self) -> None:
        module = build_streaming_rhythm_module_from_hparams(
            {
                "hidden_size": 16,
                "rhythm_hidden_size": 16,
                "rhythm_trace_bins": 8,
                "rhythm_emit_reference_sidecar": True,
            }
        )
        stats, trace = self._dummy_inputs(trace_bins=8)
        explicit_summary = torch.full((1, 2), 0.25, dtype=torch.float32)
        conditioning = module.build_reference_conditioning(
            ref_conditioning={
                "ref_rhythm_stats": stats,
                "ref_rhythm_trace": trace,
                "planner_slow_rhythm_summary": explicit_summary,
            }
        )
        self.assertTrue(torch.allclose(conditioning["planner_slow_rhythm_summary"], explicit_summary))
        self.assertIn("slow_rhythm_memory", conditioning)

    def test_runtime_ref_conditioning_keeps_planner_sidecars(self) -> None:
        stats, trace = self._dummy_inputs(trace_bins=8)
        planner_memory = torch.randn(1, 6, 2)
        planner_summary = planner_memory.mean(dim=1)
        conditioning = build_rhythm_ref_conditioning(
            {
                "ref_rhythm_stats": stats,
                "ref_rhythm_trace": trace,
                "planner_slow_rhythm_memory": planner_memory,
                "planner_slow_rhythm_summary": planner_summary,
            }
        )
        self.assertIn("planner_slow_rhythm_memory", conditioning)
        self.assertIn("planner_slow_rhythm_summary", conditioning)
        self.assertTrue(torch.allclose(conditioning["planner_slow_rhythm_summary"], planner_summary))


if __name__ == "__main__":
    unittest.main()
