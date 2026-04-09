from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.Conan.rhythm.factory import build_streaming_rhythm_module_from_hparams
from modules.Conan.rhythm.contracts import StreamingRhythmState


class PhaseDecoupledSegmentShapeTests(unittest.TestCase):
    def test_anchor_local_rho_is_monotonic_inside_open_tail(self) -> None:
        dur_anchor_src = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
        unit_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
        open_tail_mask = torch.tensor([[0.0, 1.0, 1.0, 1.0]], dtype=torch.float32)

        rho = build_streaming_rhythm_module_from_hparams({})._compute_anchor_local_rho(
            dur_anchor_src=dur_anchor_src,
            unit_mask=unit_mask,
            open_tail_mask=open_tail_mask,
        )

        self.assertTrue(torch.allclose(rho[0, :1], torch.zeros_like(rho[0, :1])))
        self.assertGreaterEqual(float(rho[0, 2].item()), float(rho[0, 1].item()))
        self.assertGreaterEqual(float(rho[0, 3].item()), float(rho[0, 2].item()))
        self.assertAlmostEqual(float(rho[0, 3].item()), 1.0, places=6)

    def test_phase_decoupled_segment_shape_context_soft_rolls_to_next_entry(self) -> None:
        module = build_streaming_rhythm_module_from_hparams(
            {
                "rhythm_phase_decoupled_timing": True,
                "rhythm_phase_decoupled_rollover_start": 0.40,
                "rhythm_phase_decoupled_rollover_end": 0.80,
            }
        )
        ref_phrase_trace = torch.tensor(
            [
                [
                    [
                        [0.0, 1.0, 0.1, 0.2, 1.0],
                        [0.0, 1.1, 0.2, 0.2, 1.0],
                        [0.0, 1.2, 0.3, 0.3, 1.0],
                        [0.0, 1.3, 0.4, 0.3, 1.0],
                    ],
                    [
                        [0.0, 4.0, 0.9, 0.8, 1.0],
                        [0.0, 4.1, 0.9, 0.8, 1.0],
                        [0.0, 4.2, 0.9, 0.8, 1.0],
                        [0.0, 4.3, 0.9, 0.8, 1.0],
                    ],
                ]
            ],
            dtype=torch.float32,
        )
        phrase_selection = {
            "selected_ref_phrase_trace": ref_phrase_trace[:, 0],
            "selected_ref_phrase_index": torch.tensor([0], dtype=torch.long),
        }
        ref_conditioning = {
            "ref_phrase_trace": ref_phrase_trace,
            "ref_phrase_valid": torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        }
        state = StreamingRhythmState(
            phase_ptr=torch.zeros((1,), dtype=torch.float32),
            clock_delta=torch.zeros((1,), dtype=torch.float32),
            commit_frontier=torch.tensor([1], dtype=torch.long),
        )
        dur_anchor_src = torch.tensor([[1.0, 1.0, 1.0, 3.0]], dtype=torch.float32)
        unit_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)

        bundle = module._build_phase_decoupled_segment_shape_context(
            ref_conditioning=ref_conditioning,
            phrase_selection=phrase_selection,
            dur_anchor_src=dur_anchor_src,
            unit_mask=unit_mask,
            state=state,
        )

        self.assertIn("segment_shape_context_unit", bundle)
        self.assertIn("segment_roll_alpha_unit", bundle)
        self.assertIn("local_rho_prior_unit", bundle)
        self.assertTrue(torch.allclose(bundle["open_tail_mask_unit"][0, :1], torch.tensor([0.0])))
        self.assertGreater(float(bundle["segment_roll_alpha_unit"][0, 3].item()), 0.5)
        last_shape = bundle["segment_shape_context_unit"][0, 3]
        # Tail rollover should pull the final open unit toward the next phrase entry.
        self.assertGreater(float(last_shape[0].item()), 2.0)
        self.assertGreater(float(last_shape[1].item()), 0.5)
        self.assertGreater(float(last_shape[2].item()), 0.5)


if __name__ == "__main__":
    unittest.main()
