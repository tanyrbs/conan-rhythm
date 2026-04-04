from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.Conan.rhythm.factorization import (
    CompactPlannerIntervention,
    apply_compact_reference_intervention,
    collect_planner_surface_bundle,
)


class FactorizationContractTests(unittest.TestCase):
    def _build_ref_conditioning(self) -> dict[str, torch.Tensor]:
        ref_rhythm_stats = torch.tensor(
            [[0.20, 1.50, 2.00, 0.10, 0.30, 0.70]],
            dtype=torch.float32,
        )
        ref_rhythm_trace = torch.tensor(
            [
                [
                    [0.00, 0.60, 0.10, -0.10, 1.00],
                    [0.10, 0.80, 0.30, 0.00, 1.00],
                    [0.00, 0.40, 0.20, 0.10, 1.00],
                ]
            ],
            dtype=torch.float32,
        )
        global_rate = torch.tensor([[0.50]], dtype=torch.float32)
        pause_ratio = torch.tensor([[0.20]], dtype=torch.float32)
        local_rate_trace = ref_rhythm_trace[:, :, 1:2].clone()
        boundary_trace = ref_rhythm_trace[:, :, 2:3].clone()
        return {
            "ref_rhythm_stats": ref_rhythm_stats,
            "ref_rhythm_trace": ref_rhythm_trace,
            "global_rate": global_rate,
            "pause_ratio": pause_ratio,
            "local_rate_trace": local_rate_trace,
            "boundary_trace": boundary_trace,
            "planner_ref_stats": torch.cat([global_rate, pause_ratio], dim=-1),
            "planner_ref_trace": torch.cat([local_rate_trace, boundary_trace], dim=-1),
            "slow_rhythm_memory": torch.ones((1, 2, 5), dtype=torch.float32),
            "slow_rhythm_summary": torch.full((1, 5), 9.0, dtype=torch.float32),
            "slow_rhythm_summary_source": "stale",
            "planner_slow_rhythm_memory": torch.ones((1, 2, 2), dtype=torch.float32),
            "planner_slow_rhythm_summary": torch.full((1, 2), 9.0, dtype=torch.float32),
            "planner_slow_rhythm_summary_source": "stale",
            "selector_meta_indices": torch.tensor([[0, 2]], dtype=torch.long),
            "selector_meta_scores": torch.tensor([[0.4, 0.9]], dtype=torch.float32),
            "selector_meta_starts": torch.tensor([[0, 2]], dtype=torch.long),
            "selector_meta_ends": torch.tensor([[1, 2]], dtype=torch.long),
        }

    def test_apply_compact_reference_intervention_keeps_raw_contract_in_sync(self) -> None:
        ref_conditioning = self._build_ref_conditioning()
        original_stats = ref_conditioning["ref_rhythm_stats"].clone()
        original_trace = ref_conditioning["ref_rhythm_trace"].clone()

        updated = apply_compact_reference_intervention(
            ref_conditioning,
            CompactPlannerIntervention(
                name="contract_sync",
                global_rate_scale=1.20,
                pause_ratio_delta=0.10,
                local_rate_scale=1.10,
                local_rate_bias=0.05,
                boundary_trace_scale=1.00,
                boundary_trace_bias=0.25,
            ),
        )

        self.assertTrue(torch.allclose(ref_conditioning["ref_rhythm_stats"], original_stats))
        self.assertTrue(torch.allclose(ref_conditioning["ref_rhythm_trace"], original_trace))
        self.assertTrue(
            torch.allclose(
                updated["planner_ref_stats"],
                torch.cat([updated["global_rate"], updated["pause_ratio"]], dim=-1),
            )
        )
        self.assertTrue(
            torch.allclose(
                updated["planner_ref_trace"],
                torch.cat([updated["local_rate_trace"], updated["boundary_trace"]], dim=-1),
            )
        )
        self.assertTrue(torch.allclose(updated["ref_rhythm_stats"][:, 0:1], updated["pause_ratio"]))
        self.assertTrue(
            torch.allclose(
                updated["ref_rhythm_stats"][:, 2:3],
                torch.reciprocal(updated["global_rate"].clamp_min(1e-6)),
            )
        )
        self.assertTrue(
            torch.allclose(
                updated["ref_rhythm_trace"][:, :, 1:2],
                updated["local_rate_trace"],
            )
        )
        self.assertTrue(
            torch.allclose(
                updated["ref_rhythm_trace"][:, :, 2:3],
                updated["boundary_trace"],
            )
        )
        for stale_key in (
            "slow_rhythm_memory",
            "slow_rhythm_summary",
            "slow_rhythm_summary_source",
            "planner_slow_rhythm_memory",
            "planner_slow_rhythm_summary",
            "planner_slow_rhythm_summary_source",
            "selector_meta_indices",
            "selector_meta_scores",
            "selector_meta_starts",
            "selector_meta_ends",
        ):
            self.assertNotIn(stale_key, updated)

    def test_collect_planner_surface_bundle_accepts_boundary_latent_fallback(self) -> None:
        execution = SimpleNamespace(
            planner=SimpleNamespace(
                speech_budget_win=torch.tensor([[2.0]], dtype=torch.float32),
                pause_budget_win=torch.tensor([[1.0]], dtype=torch.float32),
                dur_shape_unit=torch.tensor([[0.6, 0.4]], dtype=torch.float32),
                pause_shape_unit=torch.tensor([[0.2, 0.8]], dtype=torch.float32),
                boundary_latent=torch.tensor([[0.1, 0.9]], dtype=torch.float32),
            ),
            speech_duration_exec=torch.tensor([[2.0, 0.0]], dtype=torch.float32),
            pause_after_exec=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            commit_frontier=torch.tensor([1], dtype=torch.long),
        )

        bundle = collect_planner_surface_bundle(execution)

        self.assertTrue(torch.allclose(bundle["boundary_score_unit"], execution.planner.boundary_latent))

    def test_collect_planner_surface_bundle_uses_zero_boundary_when_surface_missing(self) -> None:
        execution = SimpleNamespace(
            planner=SimpleNamespace(
                speech_budget_win=torch.tensor([[2.0]], dtype=torch.float32),
                pause_budget_win=torch.tensor([[1.0]], dtype=torch.float32),
                dur_shape_unit=torch.tensor([[0.6, 0.4]], dtype=torch.float32),
                pause_shape_unit=torch.tensor([[0.2, 0.8]], dtype=torch.float32),
            ),
            speech_duration_exec=torch.tensor([[2.0, 0.0]], dtype=torch.float32),
            pause_after_exec=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            commit_frontier=torch.tensor([1], dtype=torch.long),
        )

        bundle = collect_planner_surface_bundle(execution)

        self.assertTrue(
            torch.allclose(
                bundle["boundary_score_unit"],
                torch.zeros_like(execution.planner.pause_shape_unit),
            )
        )


if __name__ == "__main__":
    unittest.main()
