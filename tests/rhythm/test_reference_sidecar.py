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
from modules.Conan.rhythm.reference_encoder import sample_progress_trace
from tasks.Conan.rhythm.dataset_mixin import RhythmConanDatasetMixin
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

    class _DummyDataset(RhythmConanDatasetMixin):
        def __init__(self):
            self.hparams = {
                "rhythm_enable_v2": True,
                "rhythm_export_debug_sidecars": True,
                "rhythm_trace_bins": 8,
                "rhythm_trace_dim": 5,
                "rhythm_slow_topk": 6,
                "rhythm_selector_cell_size": 3,
                "rhythm_source_phrase_threshold": 0.55,
                "rhythm_reference_mode_id": 0,
            }
            self.prefix = "train"

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

    def test_build_reference_conditioning_tags_summary_provenance(self) -> None:
        module = build_streaming_rhythm_module_from_hparams(
            {
                "hidden_size": 16,
                "rhythm_hidden_size": 16,
                "rhythm_trace_bins": 8,
                "rhythm_emit_reference_sidecar": False,
            }
        )
        stats, trace = self._dummy_inputs(trace_bins=8)
        conditioning = module.build_reference_conditioning(
            ref_conditioning={
                "ref_rhythm_stats": stats,
                "ref_rhythm_trace": trace,
            }
        )
        self.assertEqual(conditioning["slow_rhythm_summary_source"], "absent")
        self.assertEqual(conditioning["planner_slow_rhythm_summary_source"], "planner_ref_trace_mean")

    def test_build_reference_conditioning_rejects_malformed_planner_memory(self) -> None:
        module = build_streaming_rhythm_module_from_hparams(
            {
                "hidden_size": 16,
                "rhythm_hidden_size": 16,
                "rhythm_trace_bins": 8,
                "rhythm_emit_reference_sidecar": True,
            }
        )
        stats, trace = self._dummy_inputs(trace_bins=8)
        with self.assertRaises(ValueError):
            module.build_reference_conditioning(
                ref_conditioning={
                    "ref_rhythm_stats": stats,
                    "ref_rhythm_trace": trace,
                    "planner_slow_rhythm_memory": torch.randn(1, 6, 2, 1),
                }
            )

    def test_runtime_ref_conditioning_keeps_planner_sidecars(self) -> None:
        stats, trace = self._dummy_inputs(trace_bins=8)
        planner_memory = torch.randn(1, 6, 2)
        planner_summary = planner_memory.mean(dim=1)
        conditioning = build_rhythm_ref_conditioning(
            {
                "ref_rhythm_stats": stats,
                "ref_rhythm_trace": trace,
                "global_rate": torch.tensor([[0.25]], dtype=torch.float32),
                "pause_ratio": torch.tensor([[0.20]], dtype=torch.float32),
                "local_rate_trace": trace[:, :, 1:2],
                "boundary_trace": trace[:, :, 2:3],
                "planner_ref_stats": torch.tensor([[0.25, 0.20]], dtype=torch.float32),
                "planner_ref_trace": torch.cat([trace[:, :, 1:2], trace[:, :, 2:3]], dim=-1),
                "planner_slow_rhythm_memory": planner_memory,
                "planner_slow_rhythm_summary": planner_summary,
            }
        )
        self.assertIn("global_rate", conditioning)
        self.assertIn("pause_ratio", conditioning)
        self.assertIn("planner_ref_stats", conditioning)
        self.assertIn("planner_ref_trace", conditioning)
        self.assertIn("planner_slow_rhythm_memory", conditioning)
        self.assertIn("planner_slow_rhythm_summary", conditioning)
        self.assertTrue(torch.allclose(conditioning["planner_slow_rhythm_summary"], planner_summary))

    def test_trace_reliability_gate_preserves_raw_trace_and_downweights_local_paths(self) -> None:
        module = build_streaming_rhythm_module_from_hparams(
            {
                "hidden_size": 16,
                "rhythm_hidden_size": 16,
                "rhythm_trace_bins": 8,
                "rhythm_emit_reference_sidecar": True,
                "rhythm_trace_reliability_enable": True,
                "rhythm_trace_exhaustion_gap_start": 0.08,
                "rhythm_trace_exhaustion_gap_end": 0.22,
                "rhythm_trace_exhaustion_local_floor": 0.20,
            }
        )
        stats, trace = self._dummy_inputs(trace_bins=8)
        trace[:, :, 1] = 1.0
        trace[:, :, 2] = 1.0
        conditioning = module.build_reference_conditioning(
            ref_conditioning={
                "ref_rhythm_stats": stats,
                "ref_rhythm_trace": trace,
                "slow_rhythm_summary": torch.full((1, 5), 0.25, dtype=torch.float32),
                "planner_slow_rhythm_summary": torch.full((1, 2), 0.10, dtype=torch.float32),
            }
        )
        trace_context, _, trace_reliability = module._sample_trace_pair(
            ref_conditioning=conditioning,
            phase_ptr=torch.tensor([0.95], dtype=torch.float32),
            window_size=4,
            unit_mask=torch.ones((1, 4), dtype=torch.float32),
            dur_anchor_src=torch.ones((1, 4), dtype=torch.float32),
            horizon=0.35,
            state=StreamingRhythmState(
                phase_ptr=torch.tensor([0.95], dtype=torch.float32),
                clock_delta=torch.zeros((1,), dtype=torch.float32),
                commit_frontier=torch.zeros((1,), dtype=torch.long),
                phase_anchor=torch.tensor([[1.0, 4.0]], dtype=torch.float32),
                trace_tail_reuse_count=torch.tensor([3], dtype=torch.long),
            ),
        )
        self.assertGreater(float(trace_context.mean()), 0.55)
        self.assertAlmostEqual(float(trace_reliability.local_trace_path_weight.item()), 0.20, places=5)
        self.assertLess(
            float(trace_reliability.boundary_trace_path_weight.item()),
            float(trace_reliability.local_trace_path_weight.item()),
        )

    def test_anchor_aware_trace_sampling_uses_duration_weighted_offsets(self) -> None:
        trace = torch.linspace(0.0, 1.0, 5, dtype=torch.float32).view(1, 5, 1)
        phase_ptr = torch.tensor([0.0], dtype=torch.float32)
        visible_sizes = torch.tensor([4], dtype=torch.long)
        uniform = sample_progress_trace(
            trace,
            phase_ptr=phase_ptr,
            window_size=4,
            horizon=1.0,
            visible_sizes=visible_sizes,
        )
        anchored = sample_progress_trace(
            trace,
            phase_ptr=phase_ptr,
            window_size=4,
            horizon=1.0,
            visible_sizes=visible_sizes,
            anchor_durations=torch.tensor([[1.0, 3.0, 1.0, 1.0]], dtype=torch.float32),
        )
        self.assertLess(float(anchored[0, 1, 0]), float(uniform[0, 1, 0]))
        self.assertGreater(float(anchored[0, 2, 0]), float(uniform[0, 2, 0]))

    def test_dataset_contract_rejects_partial_planner_sidecar(self) -> None:
        dataset = self._DummyDataset()
        stats, trace = self._dummy_inputs(trace_bins=8)
        with self.assertRaises(RuntimeError):
            dataset._rhythm_cache_contract().validate_reference_conditioning_shapes(
                {
                    "ref_rhythm_stats": stats.numpy()[0],
                    "ref_rhythm_trace": trace.numpy()[0],
                    "planner_slow_rhythm_memory": torch.randn(6, 5, dtype=torch.float32).numpy(),
                },
                item_name="broken-ref",
            )

    def test_dataset_cached_reference_keeps_planner_sidecars(self) -> None:
        dataset = self._DummyDataset()
        stats, trace = self._dummy_inputs(trace_bins=8)
        ref_item = {
            "ref_rhythm_stats": stats.numpy()[0],
            "ref_rhythm_trace": trace.numpy()[0],
            "slow_rhythm_memory": torch.randn(6, 5, dtype=torch.float32).numpy(),
            "slow_rhythm_summary": torch.randn(5, dtype=torch.float32).numpy(),
            "planner_slow_rhythm_memory": torch.randn(6, 5, dtype=torch.float32).numpy(),
            "planner_slow_rhythm_summary": torch.randn(5, dtype=torch.float32).numpy(),
            "selector_meta_indices": torch.arange(6, dtype=torch.long).numpy(),
            "selector_meta_scores": torch.ones(6, dtype=torch.float32).numpy(),
            "selector_meta_starts": torch.arange(6, dtype=torch.long).numpy(),
            "selector_meta_ends": torch.arange(6, dtype=torch.long).numpy(),
        }
        conditioning = dataset._get_reference_rhythm_conditioning(
            ref_item,
            sample={"ref_mel": torch.zeros((10, 80), dtype=torch.float32)},
            target_mode="runtime_only",
        )
        self.assertIn("planner_slow_rhythm_memory", conditioning)
        self.assertIn("planner_slow_rhythm_summary", conditioning)
        self.assertIn("planner_slow_rhythm_memory", dataset._resolve_optional_sample_keys())
        self.assertIn("planner_slow_rhythm_summary", dataset._build_optional_collate_spec())
        planner_memory = dataset._rhythm_sample_assembler()._tensorize_optional_value(
            "planner_slow_rhythm_memory",
            ref_item["planner_slow_rhythm_memory"],
        )
        self.assertEqual(planner_memory.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
