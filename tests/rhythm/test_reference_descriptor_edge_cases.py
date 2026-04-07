from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.Conan.rhythm.factorization import _sync_raw_reference_contract
from modules.Conan.rhythm.reference_descriptor import RefRhythmDescriptor
from modules.Conan.rhythm.reference_encoder import ReferenceRhythmEncoder
from tasks.Conan.rhythm.targets import resolve_reference_descriptor_targets_from_sample


class ReferenceDescriptorEdgeCaseTests(unittest.TestCase):
    def test_zero_speech_encoder_uses_uniform_progress_fallback(self) -> None:
        encoder = ReferenceRhythmEncoder(trace_bins=8, pause_energy_threshold_std=0.5)
        ref_mel = torch.zeros((1, 12, 80), dtype=torch.float32)
        packed = encoder(ref_mel)
        trace = packed["ref_rhythm_trace"]
        self.assertFalse(torch.isnan(trace).any())
        # segment_duration_bias should stay near zero under uniform-progress fallback.
        self.assertTrue(torch.allclose(trace[:, :, 3], torch.zeros_like(trace[:, :, 3]), atol=1e-5))

    def test_zero_mean_speech_maps_to_zero_global_rate(self) -> None:
        stats = torch.tensor([[1.0, 12.0, 0.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
        trace = torch.zeros((1, 8, 5), dtype=torch.float32)
        compact = RefRhythmDescriptor.from_stats_trace(stats, trace)
        self.assertTrue(torch.allclose(compact["global_rate"], torch.zeros((1, 1))))

    def test_targets_fill_zero_global_rate_from_cached_contract(self) -> None:
        sample = {
            "ref_rhythm_stats": torch.tensor([[1.0, 12.0, 0.0, 0.0, 1.0, 0.0]], dtype=torch.float32),
            "ref_rhythm_trace": torch.zeros((1, 8, 5), dtype=torch.float32),
        }
        ref_global_rate, _, _, _ = resolve_reference_descriptor_targets_from_sample(sample, detach=False)
        self.assertIsNotNone(ref_global_rate)
        self.assertTrue(torch.allclose(ref_global_rate, torch.zeros((1, 1))))

    def test_sync_raw_reference_contract_preserves_zero_global_rate_semantics(self) -> None:
        ref_conditioning = {
            "ref_rhythm_stats": torch.tensor([[0.5, 2.0, 3.0, 0.0, 0.5, 0.5]], dtype=torch.float32),
            "global_rate": torch.zeros((1, 1), dtype=torch.float32),
        }
        _sync_raw_reference_contract(ref_conditioning)
        self.assertEqual(float(ref_conditioning["ref_rhythm_stats"][0, 2].item()), 0.0)


if __name__ == "__main__":
    unittest.main()
