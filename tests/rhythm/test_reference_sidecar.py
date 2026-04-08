from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.Conan.rhythm.controller import ChunkStateBundle
from modules.Conan.rhythm.factory import build_streaming_rhythm_module_from_hparams
from modules.Conan.rhythm.contracts import StreamingRhythmState
from modules.Conan.rhythm.reference_encoder import sample_progress_trace
from modules.Conan.rhythm.reference_selector import ReferenceSelector
from modules.Conan.rhythm.supervision import build_source_phrase_cache
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
                "rhythm_runtime_phrase_bank_enable": True,
            }
        )
        self.assertTrue(module.reference_descriptor.emit_reference_sidecar)
        self.assertIsNotNone(module.reference_descriptor.selector)

    def test_factory_auto_enables_reference_sidecar_for_external_reference_bootstrap(self) -> None:
        module = build_streaming_rhythm_module_from_hparams(
            {
                "hidden_size": 16,
                "rhythm_hidden_size": 16,
                "rhythm_trace_bins": 8,
                "rhythm_cached_reference_policy": "sample_ref",
                "rhythm_require_external_reference": True,
            }
        )
        self.assertTrue(module.reference_descriptor.emit_reference_sidecar)
        self.assertIsNotNone(module.reference_descriptor.selector)

    def test_factory_explicit_sidecar_flag_overrides_external_reference_auto_enable(self) -> None:
        module = build_streaming_rhythm_module_from_hparams(
            {
                "hidden_size": 16,
                "rhythm_hidden_size": 16,
                "rhythm_trace_bins": 8,
                "rhythm_cached_reference_policy": "sample_ref",
                "rhythm_require_external_reference": True,
                "rhythm_emit_reference_sidecar": False,
            }
        )
        self.assertFalse(module.reference_descriptor.emit_reference_sidecar)
        self.assertIsNone(module.reference_descriptor.selector)

    def test_factory_legacy_trace_exhaustion_alias_enables_sidecar_and_reliability(self) -> None:
        module = build_streaming_rhythm_module_from_hparams(
            {
                "hidden_size": 16,
                "rhythm_hidden_size": 16,
                "rhythm_trace_bins": 8,
                "rhythm_enable_trace_exhaustion_fallback": True,
            }
        )
        self.assertTrue(module.reference_descriptor.emit_reference_sidecar)
        self.assertTrue(module.trace_reliability_enable)

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

    def test_reference_sidecar_phrase_bank_shapes_are_well_formed(self) -> None:
        module = build_streaming_rhythm_module_from_hparams(
            {
                "hidden_size": 16,
                "rhythm_hidden_size": 16,
                "rhythm_trace_bins": 8,
                "rhythm_emit_reference_sidecar": True,
            }
        )
        stats, trace = self._dummy_inputs(trace_bins=8)
        trace[:, :, 0] = torch.tensor([[0.0, 0.1, 0.9, 0.0, 0.2, 0.8, 0.0, 0.4]], dtype=torch.float32)
        trace[:, :, 2] = torch.tensor([[0.1, 0.3, 0.9, 0.2, 0.2, 0.8, 0.1, 1.0]], dtype=torch.float32)
        conditioning = module.build_reference_conditioning(
            ref_rhythm_stats=stats,
            ref_rhythm_trace=trace,
        )
        self.assertIn("ref_phrase_trace", conditioning)
        self.assertIn("planner_ref_phrase_trace", conditioning)
        self.assertIn("ref_phrase_valid", conditioning)
        self.assertIn("ref_phrase_lengths", conditioning)
        self.assertEqual(conditioning["ref_phrase_trace"].dim(), 4)
        self.assertEqual(conditioning["planner_ref_phrase_trace"].size(-1), 2)
        self.assertEqual(conditioning["ref_phrase_valid"].dim(), 2)
        self.assertEqual(conditioning["ref_phrase_lengths"].shape, conditioning["ref_phrase_valid"].shape)

    def test_select_monotonic_phrase_prefers_boundary_strong_candidate(self) -> None:
        selector = ReferenceSelector()
        ref_phrase_trace = torch.zeros((1, 3, 5), dtype=torch.float32)
        planner_ref_phrase_trace = torch.zeros((1, 3, 2), dtype=torch.float32)
        selection = ReferenceSelector.select_monotonic_phrase(
            ref_phrase_trace=ref_phrase_trace,
            planner_ref_phrase_trace=planner_ref_phrase_trace,
            ref_phrase_valid=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
            ref_phrase_lengths=torch.tensor([[5.0, 3.0, 1.0]], dtype=torch.float32),
            ref_phrase_starts=torch.tensor([[0.0, 0.5, 1.0]], dtype=torch.float32),
            ref_phrase_ends=torch.tensor([[1.0, 1.5, 2.0]], dtype=torch.float32),
            ref_phrase_boundary_strength=torch.tensor([[0.1, 0.9, 0.2]], dtype=torch.float32),
            ref_phrase_stats=None,
            ref_phrase_ptr=torch.tensor([0], dtype=torch.long),
        )
        self.assertEqual(int(selection["selected_ref_phrase_index"][0].item()), 1)

    def test_select_phrase_bank_is_monotonic_and_clamps_to_last_valid_phrase(self) -> None:
        module = build_streaming_rhythm_module_from_hparams(
            {
                "hidden_size": 16,
                "rhythm_hidden_size": 16,
                "rhythm_trace_bins": 8,
                "rhythm_emit_reference_sidecar": True,
            }
        )
        stats, trace = self._dummy_inputs(trace_bins=8)
        trace[:, :, 0] = torch.tensor([[0.0, 0.1, 0.9, 0.0, 0.2, 0.8, 0.0, 0.4]], dtype=torch.float32)
        trace[:, :, 2] = torch.tensor([[0.1, 0.3, 0.9, 0.2, 0.2, 0.8, 0.1, 1.0]], dtype=torch.float32)
        conditioning = module.build_reference_conditioning(
            ref_rhythm_stats=stats,
            ref_rhythm_trace=trace,
        )
        valid_count = int(conditioning["ref_phrase_valid"].sum().item())
        first = module.reference_descriptor.select_phrase_bank(
            conditioning,
            ref_phrase_ptr=torch.tensor([0], dtype=torch.long),
        )
        last = module.reference_descriptor.select_phrase_bank(
            conditioning,
            ref_phrase_ptr=torch.tensor([valid_count + 3], dtype=torch.long),
        )
        self.assertEqual(int(first["selected_ref_phrase_index"][0].item()), 0)
        self.assertEqual(int(last["selected_ref_phrase_index"][0].item()), max(valid_count - 1, 0))
        self.assertGreaterEqual(
            int(last["selected_ref_phrase_index"][0].item()),
            int(first["selected_ref_phrase_index"][0].item()),
        )

    def test_select_phrase_bank_can_advance_with_boundary_aware_query(self) -> None:
        module = build_streaming_rhythm_module_from_hparams(
            {
                "hidden_size": 16,
                "rhythm_hidden_size": 16,
                "rhythm_trace_bins": 8,
                "rhythm_emit_reference_sidecar": True,
                "rhythm_runtime_phrase_bank_enable": True,
                "rhythm_runtime_phrase_select_window": 3,
            }
        )
        stats, trace = self._dummy_inputs(trace_bins=8)
        trace[:, :, 0] = torch.tensor([[0.0, 0.0, 0.7, 0.0, 0.0, 0.2, 0.1, 0.2]], dtype=torch.float32)
        trace[:, :, 2] = torch.tensor([[0.1, 0.2, 0.95, 0.1, 0.1, 0.3, 0.2, 0.4]], dtype=torch.float32)
        conditioning = module.build_reference_conditioning(
            ref_rhythm_stats=stats,
            ref_rhythm_trace=trace,
        )
        base = module.reference_descriptor.select_phrase_bank(
            conditioning,
            ref_phrase_ptr=torch.tensor([0], dtype=torch.long),
        )
        guided = module.reference_descriptor.select_phrase_bank(
            conditioning,
            ref_phrase_ptr=torch.tensor([0], dtype=torch.long),
            query_chunk_summary=torch.tensor([[0.1, 0.2, 0.8, 0.8, 0.1, 0.9]], dtype=torch.float32),
            query_commit_confidence=torch.tensor([[0.9]], dtype=torch.float32),
            query_phrase_close_prob=torch.tensor([[0.95]], dtype=torch.float32),
        )
        self.assertGreaterEqual(
            int(guided["selected_ref_phrase_index"][0].item()),
            int(base["selected_ref_phrase_index"][0].item()),
        )
        self.assertIn("selected_phrase_prototype_summary", guided)
        self.assertIn("selected_phrase_prototype_valid", guided)

    def test_select_phrase_bank_exposes_phrase_prototype_metadata(self) -> None:
        module = build_streaming_rhythm_module_from_hparams(
            {
                "hidden_size": 16,
                "rhythm_hidden_size": 16,
                "rhythm_trace_bins": 8,
                "rhythm_emit_reference_sidecar": True,
            }
        )
        stats, trace = self._dummy_inputs(trace_bins=8)
        trace[:, :, 2] = torch.tensor([[0.0, 0.1, 0.8, 0.0, 0.4, 0.9, 0.0, 0.2]], dtype=torch.float32)
        conditioning = module.build_reference_conditioning(ref_rhythm_stats=stats, ref_rhythm_trace=trace)
        selection = module.reference_descriptor.select_phrase_bank(
            conditioning,
            ref_phrase_ptr=torch.tensor([1], dtype=torch.long),
        )
        self.assertIn("selected_phrase_prototype_summary", selection)
        self.assertIn("selected_phrase_prototype_stats", selection)
        self.assertIn("selected_phrase_prototype_valid", selection)
        summary = selection["selected_phrase_prototype_summary"]
        stats_proto = selection["selected_phrase_prototype_stats"]
        valid = selection["selected_phrase_prototype_valid"]
        self.assertEqual(summary.dim(), 2)
        self.assertEqual(stats_proto.size(-1), 2)
        self.assertTrue(torch.eq(valid, valid.clamp(0.0, 1.0)).all())

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

    def test_source_phrase_cache_builds_monotonic_phrase_groups(self) -> None:
        phrase_cache = build_source_phrase_cache(
            dur_anchor_src=[1, 1, 1, 1, 1],
            sep_hint=[0, 1, 0, 0, 0],
            open_run_mask=[0, 0, 0, 0, 0],
            sealed_mask=[1, 1, 1, 1, 1],
            boundary_confidence=[0.1, 0.9, 0.2, 0.8, 0.1],
            phrase_boundary_threshold=0.55,
        )
        phrase_group_index = torch.tensor(phrase_cache["phrase_group_index"], dtype=torch.long)
        phrase_group_pos = torch.tensor(phrase_cache["phrase_group_pos"], dtype=torch.float32)
        phrase_final_mask = torch.tensor(phrase_cache["phrase_final_mask"], dtype=torch.float32)

        self.assertEqual(tuple(phrase_group_index.shape), (5,))
        self.assertTrue(torch.all(phrase_group_index[1:] >= phrase_group_index[:-1]))
        final_positions = torch.nonzero(phrase_final_mask > 0.5, as_tuple=False).flatten().tolist()
        self.assertGreaterEqual(len(final_positions), 1)
        self.assertEqual(final_positions[-1], 4)

        phrase_start = 0
        for phrase_end in final_positions:
            phrase_pos = phrase_group_pos[phrase_start : phrase_end + 1]
            self.assertAlmostEqual(float(phrase_pos[0].item()), 0.0, places=6)
            self.assertAlmostEqual(float(phrase_pos[-1].item()), 1.0, places=6)
            if phrase_pos.numel() > 1:
                self.assertTrue(torch.all(phrase_pos[1:] >= phrase_pos[:-1]))
            phrase_start = phrase_end + 1

    def test_reference_phrase_bank_sidecar_shapes_when_available(self) -> None:
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
        ref_phrase_trace = conditioning.get("ref_phrase_trace")
        ref_phrase_stats = conditioning.get("ref_phrase_stats")
        if ref_phrase_trace is None or ref_phrase_stats is None:
            self.skipTest("reference phrase bank sidecar not wired yet")
        self.assertEqual(ref_phrase_trace.dim(), 4)
        self.assertEqual(ref_phrase_stats.dim(), 3)
        self.assertEqual(ref_phrase_trace.size(0), ref_phrase_stats.size(0))
        self.assertEqual(ref_phrase_trace.size(1), ref_phrase_stats.size(1))
        ref_phrase_index = conditioning.get("ref_phrase_index")
        if ref_phrase_index is not None and ref_phrase_index.size(1) > 1:
            self.assertTrue(torch.all(ref_phrase_index[:, 1:] >= ref_phrase_index[:, :-1]))

    def test_build_reference_conditioning_recomputes_public_compact_contract(self) -> None:
        module = build_streaming_rhythm_module_from_hparams(
            {
                "hidden_size": 16,
                "rhythm_hidden_size": 16,
                "rhythm_trace_bins": 8,
                "rhythm_emit_reference_sidecar": True,
            }
        )
        stats, trace = self._dummy_inputs(trace_bins=8)
        stale_trace = torch.full((1, 8, 2), 9.0, dtype=torch.float32)
        planner_memory = torch.randn(1, 6, 2)
        conditioning = module.build_reference_conditioning(
            ref_conditioning={
                "ref_rhythm_stats": stats,
                "ref_rhythm_trace": trace,
                "planner_ref_trace": stale_trace,
                "planner_ref_stats": torch.full((1, 2), 7.0, dtype=torch.float32),
                "global_rate": torch.full((1, 1), 5.0, dtype=torch.float32),
                "pause_ratio": torch.full((1, 1), 6.0, dtype=torch.float32),
                "planner_slow_rhythm_memory": planner_memory,
            }
        )
        self.assertTrue(torch.allclose(conditioning["global_rate"], module.reference_descriptor.from_stats_trace(stats, trace)["global_rate"]))
        self.assertTrue(torch.allclose(conditioning["pause_ratio"], stats[:, 0:1]))
        self.assertFalse(torch.allclose(conditioning["planner_ref_trace"], stale_trace))
        self.assertTrue(torch.allclose(conditioning["planner_slow_rhythm_memory"], planner_memory))

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
        ref_phrase_trace = torch.randn(1, 2, 4, 5)
        planner_ref_phrase_trace = ref_phrase_trace[:, :, :, 1:3]
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
                "ref_phrase_trace": ref_phrase_trace,
                "planner_ref_phrase_trace": planner_ref_phrase_trace,
                "ref_phrase_valid": torch.tensor([[1.0, 1.0]], dtype=torch.float32),
                "ref_phrase_lengths": torch.tensor([[3.0, 5.0]], dtype=torch.float32),
                "ref_phrase_starts": torch.tensor([[0.0, 0.5]], dtype=torch.float32),
                "ref_phrase_ends": torch.tensor([[0.5, 1.0]], dtype=torch.float32),
                "ref_phrase_boundary_strength": torch.tensor([[0.2, 0.9]], dtype=torch.float32),
                "ref_phrase_stats": torch.randn(1, 2, 5),
            }
        )
        self.assertIn("global_rate", conditioning)
        self.assertIn("pause_ratio", conditioning)
        self.assertIn("planner_ref_stats", conditioning)
        self.assertIn("planner_ref_trace", conditioning)
        self.assertIn("planner_slow_rhythm_memory", conditioning)
        self.assertIn("planner_slow_rhythm_summary", conditioning)
        self.assertIn("ref_phrase_trace", conditioning)
        self.assertIn("planner_ref_phrase_trace", conditioning)
        self.assertTrue(torch.allclose(conditioning["planner_slow_rhythm_summary"], planner_summary))

    def test_scheduler_phrase_selection_can_reuse_cached_phrase_sidecar_without_runtime_bank_flag(self) -> None:
        module = build_streaming_rhythm_module_from_hparams(
            {
                "hidden_size": 16,
                "rhythm_hidden_size": 16,
                "rhythm_trace_bins": 8,
                "rhythm_emit_reference_sidecar": True,
                "rhythm_runtime_phrase_bank_enable": False,
            }
        )
        stats, trace = self._dummy_inputs(trace_bins=8)
        trace[:, :, 2] = torch.tensor([[0.0, 0.2, 0.9, 0.0, 0.1, 0.8, 0.0, 0.2]], dtype=torch.float32)
        conditioning = module.build_reference_conditioning(
            ref_rhythm_stats=stats,
            ref_rhythm_trace=trace,
        )
        selection = module._select_scheduler_phrase_bank(
            ref_conditioning=conditioning,
            state=module.init_state(batch_size=1, device=torch.device("cpu")),
            batch_size=1,
            device=torch.device("cpu"),
        )
        self.assertIsNotNone(selection)
        assert selection is not None
        self.assertIn("selected_ref_phrase_index", selection)

    def test_trace_reliability_gate_uses_hierarchical_fallback_when_local_path_is_unreliable(self) -> None:
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
        self.assertGreater(float(trace_context.mean()), 0.25)
        self.assertLess(float(trace_context.mean()), 0.55)
        self.assertAlmostEqual(float(trace_reliability.local_trace_path_weight.item()), 0.20, places=5)
        self.assertLess(
            float(trace_reliability.boundary_trace_path_weight.item()),
            float(trace_reliability.local_trace_path_weight.item()),
        )
        self.assertGreaterEqual(float(trace_reliability.global_blend.item()), 0.0)

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

    def test_sample_trace_window_active_tail_only_keeps_current_tail(self) -> None:
        trace = torch.linspace(0.0, 1.0, 8, dtype=torch.float32).view(1, 8, 1)
        phase_ptr = torch.tensor([0.0], dtype=torch.float32)
        anchor_short = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
        anchor_full = torch.tensor([[1.0, 1.0, 1.0, 10.0]], dtype=torch.float32)
        visible_short = torch.tensor([3], dtype=torch.long)
        visible_full = torch.tensor([4], dtype=torch.long)

        baseline = sample_progress_trace(
            trace,
            phase_ptr=phase_ptr,
            window_size=4,
            horizon=1.0,
            visible_sizes=visible_short,
            anchor_durations=anchor_short,
        )
        tail_only = sample_progress_trace(
            trace,
            phase_ptr=phase_ptr,
            window_size=4,
            horizon=1.0,
            visible_sizes=visible_full,
            anchor_durations=anchor_full,
            commit_frontier=torch.tensor([0], dtype=torch.long),
            lookahead_units=3,
            active_tail_only=True,
        )
        self.assertTrue(torch.allclose(baseline[0, :3, 0], tail_only[0, :3, 0], atol=1e-4))
        self.assertGreater(float(tail_only[0, 3, 0]), 0.0)

    def test_sample_trace_window_lookahead_limits_offsets(self) -> None:
        trace = torch.linspace(0.0, 1.0, 8, dtype=torch.float32).view(1, 8, 1)
        phase_ptr = torch.tensor([0.0], dtype=torch.float32)
        anchor = torch.tensor([[1.0] * 8], dtype=torch.float32)
        visible = torch.tensor([8], dtype=torch.long)
        lookahead = sample_progress_trace(
            trace,
            phase_ptr=phase_ptr,
            window_size=8,
            horizon=1.0,
            visible_sizes=visible,
            anchor_durations=anchor,
            commit_frontier=torch.tensor([0], dtype=torch.long),
            lookahead_units=3,
            active_tail_only=True,
        )
        self.assertTrue(torch.allclose(lookahead[0, 3:, 0], lookahead[0, 2, 0]))

    def test_sample_trace_window_active_tail_handles_long_prefix(self) -> None:
        trace = torch.linspace(0.0, 1.0, 16, dtype=torch.float32).view(1, 16, 1)
        phase_ptr = torch.tensor([0.0], dtype=torch.float32)
        baseline = sample_progress_trace(
            trace,
            phase_ptr=phase_ptr,
            window_size=4,
            horizon=1.0,
            visible_sizes=torch.tensor([3], dtype=torch.long),
            anchor_durations=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
        )
        tail_only = sample_progress_trace(
            trace,
            phase_ptr=phase_ptr,
            window_size=4,
            horizon=1.0,
            visible_sizes=torch.tensor([10], dtype=torch.long),
            anchor_durations=torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0]], dtype=torch.float32),
            commit_frontier=torch.tensor([6], dtype=torch.long),
            lookahead_units=3,
            active_tail_only=True,
        )
        self.assertTrue(torch.allclose(baseline[0, :3, 0], tail_only[0, :3, 0], atol=1e-4))
        self.assertTrue(torch.allclose(tail_only[0, 3:, 0], tail_only[0, 2, 0], atol=1e-4))

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
