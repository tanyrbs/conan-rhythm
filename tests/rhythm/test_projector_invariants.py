from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.Conan.rhythm.contracts import RhythmPlannerOutputs, StreamingRhythmState
from modules.Conan.rhythm.controller import ChunkStateBundle, resolve_budget_views_from_total_and_pause_share
from modules.Conan.rhythm.feasibility import lift_projector_budgets_to_feasible_region
from modules.Conan.rhythm.offline_teacher import OfflineRhythmTeacherPlanner, OfflineTeacherConfig
from modules.Conan.rhythm.pause_features import build_pause_support_feature_bundle
from modules.Conan.rhythm.prefix_state import (
    build_prefix_state_from_exec_numpy,
    build_prefix_state_from_exec_torch,
)
from modules.Conan.rhythm.factory import build_streaming_rhythm_module_from_hparams
from modules.Conan.rhythm.projector import (
    ProjectorConfig,
    StreamingRhythmProjector,
    _project_pause_impl,
    _project_pause_simple_impl,
    _resolve_allocation_mask,
)
from modules.Conan.rhythm_v3.projector import (
    StreamingDurationProjector as StreamingDurationV3Projector,
)
from modules.Conan.rhythm.scheduler import MonotonicRhythmScheduler


class _ChunkStateHeadStub(nn.Module):
    def __init__(self, chunk_state: ChunkStateBundle):
        super().__init__()
        self._chunk_state = chunk_state

    def forward(self, *args, **kwargs) -> ChunkStateBundle:
        return self._chunk_state


class DurationV3ProjectorHotPathTests(unittest.TestCase):
    def test_duration_v3_prefix_projection_keeps_carry_budget_and_boundary_semantics(self):
        projected, residual, prefix_offset, boundary_hit, boundary_decay = (
            StreamingDurationV3Projector._project_duration_prefix(
                unit_duration_exec=torch.tensor([[2.6, 0.2, 4.7, 3.6]], dtype=torch.float32),
                source_duration_obs=torch.tensor([[2.0, 5.0, 4.0, 3.0]], dtype=torch.float32),
                commit_mask=torch.ones((1, 4), dtype=torch.float32),
                speech_commit_mask=torch.tensor([[1.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
                coarse_only_commit_mask=torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=torch.float32),
                source_boundary_cue=torch.tensor([[0.0, 0.0, 0.7, 0.0]], dtype=torch.float32),
                phrase_final_mask=torch.zeros((1, 4), dtype=torch.float32),
                residual_prev=torch.tensor([[0.4]], dtype=torch.float32),
                prefix_unit_offset_prev=torch.zeros((1, 1), dtype=torch.float32),
                committed_units_prev=None,
                cached_duration_exec_prev=None,
                budget_pos=1,
                budget_neg=1,
                boundary_carry_decay=0.5,
                boundary_reset_thresh=0.5,
            )
        )

        assert torch.allclose(projected, torch.tensor([[3.0, 5.0, 4.0, 3.0]], dtype=torch.float32))
        assert torch.allclose(residual, torch.tensor([[0.95]], dtype=torch.float32))
        assert torch.allclose(prefix_offset, torch.tensor([[0.5]], dtype=torch.float32))
        assert torch.allclose(boundary_hit, torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=torch.float32))
        assert torch.allclose(boundary_decay, torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=torch.float32))


class ProjectorInvariantTests(unittest.TestCase):
    @staticmethod
    def _make_invariance_projector() -> StreamingRhythmProjector:
        return StreamingRhythmProjector(
            ProjectorConfig(
                min_speech_frames=0.0,
                tail_hold_units=1,
                use_boundary_commit_guard=False,
                pause_selection_mode="simple",
                build_render_plan=False,
            )
        )

    @staticmethod
    def _run_projector(
        projector: StreamingRhythmProjector,
        *,
        state: StreamingRhythmState,
        dur_anchor_src: list[float],
        open_run_mask: list[int],
        reuse_prefix: bool,
        speech_budget: float = 3.0,
        pause_budget: float = 1.0,
    ):
        width = len(dur_anchor_src)
        pause_seed = [0.0, 1.0, 0.0, 0.0, 0.0]
        planner = RhythmPlannerOutputs(
            speech_budget_win=torch.tensor([[speech_budget]], dtype=torch.float32),
            pause_budget_win=torch.tensor([[pause_budget]], dtype=torch.float32),
            dur_logratio_unit=torch.zeros((1, width), dtype=torch.float32),
            pause_weight_unit=torch.tensor([pause_seed[:width]], dtype=torch.float32),
            boundary_score_unit=torch.zeros((1, width), dtype=torch.float32),
            trace_context=torch.zeros((1, width, 2), dtype=torch.float32),
            source_boundary_cue=torch.zeros((1, width), dtype=torch.float32),
        )
        return projector(
            dur_anchor_src=torch.tensor([dur_anchor_src], dtype=torch.float32),
            unit_mask=torch.ones((1, width), dtype=torch.float32),
            speech_budget_win=planner.speech_budget_win,
            pause_budget_win=planner.pause_budget_win,
            dur_logratio_unit=planner.dur_logratio_unit,
            pause_weight_unit=planner.pause_weight_unit,
            boundary_score_unit=planner.boundary_score_unit,
            state=state,
            open_run_mask=torch.tensor([open_run_mask], dtype=torch.long),
            planner=planner,
            reuse_prefix=reuse_prefix,
            force_full_commit=False,
        )

    def test_backlog_is_derived_from_signed_clock_delta(self) -> None:
        state = StreamingRhythmState(
            phase_ptr=torch.tensor([0.1, 0.2], dtype=torch.float32),
            clock_delta=torch.tensor([1.5, -0.5], dtype=torch.float32),
            commit_frontier=torch.tensor([0, 0], dtype=torch.long),
        )
        self.assertTrue(torch.allclose(state.backlog, torch.tensor([1.5, 0.0], dtype=torch.float32)))

    def test_phase_ptr_gap_is_derived_from_anchor_progress_ratio(self) -> None:
        state = StreamingRhythmState(
            phase_ptr=torch.tensor([0.60], dtype=torch.float32),
            clock_delta=torch.tensor([0.0], dtype=torch.float32),
            commit_frontier=torch.tensor([0], dtype=torch.long),
            phase_anchor=torch.tensor([[3.0, 10.0]], dtype=torch.float32),
        )
        self.assertTrue(torch.allclose(state.phase_progress_ratio, torch.tensor([0.30], dtype=torch.float32)))
        self.assertTrue(torch.allclose(state.phase_ptr_gap, torch.tensor([0.30], dtype=torch.float32)))

    def test_projector_advance_state_applies_debt_leak(self) -> None:
        projector = StreamingRhythmProjector(
            ProjectorConfig(
                build_render_plan=False,
                debt_leak=0.25,
                debt_max_abs=20.0,
                debt_correction_horizon=10.0,
            )
        )
        state = StreamingRhythmState(
            phase_ptr=torch.tensor([0.0], dtype=torch.float32),
            clock_delta=torch.tensor([4.0], dtype=torch.float32),
            commit_frontier=torch.tensor([0], dtype=torch.long),
        )
        next_state = projector._advance_state(
            state=state,
            dur_anchor_src=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            unit_mask=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            effective_duration_exec=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            commit_frontier=torch.tensor([1], dtype=torch.long),
            speech_duration_exec=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            pause_after_exec=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        )
        self.assertAlmostEqual(float(next_state.clock_delta.item()), 3.0, places=5)

    def test_projector_advance_state_clamps_debt_correction_horizon(self) -> None:
        projector = StreamingRhythmProjector(
            ProjectorConfig(
                build_render_plan=False,
                debt_leak=0.0,
                debt_max_abs=20.0,
                debt_correction_horizon=1.5,
            )
        )
        state = StreamingRhythmState(
            phase_ptr=torch.tensor([0.0], dtype=torch.float32),
            clock_delta=torch.tensor([0.0], dtype=torch.float32),
            commit_frontier=torch.tensor([0], dtype=torch.long),
        )
        next_state = projector._advance_state(
            state=state,
            dur_anchor_src=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            unit_mask=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            effective_duration_exec=torch.tensor([[5.0, 0.0]], dtype=torch.float32),
            commit_frontier=torch.tensor([1], dtype=torch.long),
            speech_duration_exec=torch.tensor([[5.0, 0.0]], dtype=torch.float32),
            pause_after_exec=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        )
        self.assertAlmostEqual(float(next_state.clock_delta.item()), 1.5, places=5)

    def test_projector_advance_state_clips_debt_max_abs(self) -> None:
        projector = StreamingRhythmProjector(
            ProjectorConfig(
                build_render_plan=False,
                debt_leak=0.0,
                debt_max_abs=2.5,
                debt_correction_horizon=10.0,
            )
        )
        state = StreamingRhythmState(
            phase_ptr=torch.tensor([0.0], dtype=torch.float32),
            clock_delta=torch.tensor([1.0], dtype=torch.float32),
            commit_frontier=torch.tensor([0], dtype=torch.long),
        )
        next_state = projector._advance_state(
            state=state,
            dur_anchor_src=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            unit_mask=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            effective_duration_exec=torch.tensor([[5.0, 0.0]], dtype=torch.float32),
            commit_frontier=torch.tensor([1], dtype=torch.long),
            speech_duration_exec=torch.tensor([[5.0, 0.0]], dtype=torch.float32),
            pause_after_exec=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        )
        self.assertAlmostEqual(float(next_state.clock_delta.item()), 2.5, places=5)

    def test_total_budget_and_pause_share_are_derived_surfaces(self) -> None:
        planner = RhythmPlannerOutputs(
            speech_budget_win=torch.tensor([[2.0], [3.0]], dtype=torch.float32),
            pause_budget_win=torch.tensor([[1.0], [1.5]], dtype=torch.float32),
            dur_logratio_unit=torch.zeros((2, 3), dtype=torch.float32),
            pause_weight_unit=torch.ones((2, 3), dtype=torch.float32),
            boundary_score_unit=torch.zeros((2, 3), dtype=torch.float32),
            trace_context=torch.zeros((2, 3, 5), dtype=torch.float32),
        )
        self.assertTrue(torch.allclose(planner.total_budget_win, torch.tensor([[3.0], [4.5]], dtype=torch.float32)))
        self.assertTrue(torch.allclose(planner.pause_share_win, torch.tensor([[1.0 / 3.0], [1.0 / 3.0]], dtype=torch.float32)))

    def test_budget_helper_preserves_raw_under_budget_proposal(self) -> None:
        budget_views = resolve_budget_views_from_total_and_pause_share(
            total_budget=torch.tensor([[0.3]], dtype=torch.float32),
            pause_share=torch.tensor([[1.0 / 3.0]], dtype=torch.float32),
        )
        self.assertTrue(
            torch.allclose(
                budget_views["raw_speech_budget_win"],
                torch.tensor([[0.2]], dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.allclose(
                budget_views["raw_pause_budget_win"],
                torch.tensor([[0.1]], dtype=torch.float32),
            )
        )
        self.assertTrue(torch.allclose(budget_views["speech_budget_win"], budget_views["raw_speech_budget_win"]))
        self.assertTrue(torch.allclose(budget_views["pause_budget_win"], budget_views["raw_pause_budget_win"]))

    def test_scheduler_preserves_raw_budget_views_on_planner_output(self) -> None:
        scheduler = MonotonicRhythmScheduler(hidden_size=4, stats_dim=2, trace_dim=2)
        raw_speech = torch.tensor([[0.2]], dtype=torch.float32)
        raw_pause = torch.tensor([[0.1]], dtype=torch.float32)
        scheduler.window_budget.forward = lambda *args, **kwargs: {
            "raw_speech_budget_win": raw_speech,
            "raw_pause_budget_win": raw_pause,
            "speech_budget_win": raw_speech,
            "pause_budget_win": raw_pause,
        }
        scheduler.unit_redistribution.forward = lambda *args, **kwargs: {
            "dur_logratio_unit": torch.zeros((1, 3), dtype=torch.float32),
            "pause_weight_unit": torch.full((1, 3), 1.0 / 3.0, dtype=torch.float32),
        }
        planner = scheduler(
            unit_states=torch.zeros((1, 3, 4), dtype=torch.float32),
            dur_anchor_src=torch.ones((1, 3), dtype=torch.float32),
            unit_mask=torch.ones((1, 3), dtype=torch.float32),
            ref_conditioning={"planner_ref_stats": torch.tensor([[1.0, 0.25]], dtype=torch.float32)},
            trace_context=torch.zeros((1, 3, 2), dtype=torch.float32),
            planner_trace_context=torch.zeros((1, 3, 2), dtype=torch.float32),
            state=StreamingRhythmState(
                phase_ptr=torch.zeros((1,), dtype=torch.float32),
                clock_delta=torch.zeros((1,), dtype=torch.float32),
                commit_frontier=torch.zeros((1,), dtype=torch.long),
            ),
        )
        self.assertTrue(torch.allclose(planner.raw_speech_budget_win, raw_speech))
        self.assertTrue(torch.allclose(planner.raw_pause_budget_win, raw_pause))

    def test_speech_lower_bound_uses_reallocation_before_total_lift(self) -> None:
        projection = lift_projector_budgets_to_feasible_region(
            dur_anchor_src=torch.tensor([[0.4, 0.4, 0.4]], dtype=torch.float32),
            unit_mask=torch.ones((1, 3), dtype=torch.float32),
            speech_budget_win=torch.tensor([[1.0]], dtype=torch.float32),
            pause_budget_win=torch.tensor([[5.0]], dtype=torch.float32),
            previous_speech_exec=None,
            previous_pause_exec=None,
            commit_frontier=torch.tensor([0], dtype=torch.long),
            reuse_prefix=True,
            min_speech_frames=1.0,
            max_speech_expand=3.0,
        )
        self.assertTrue(torch.allclose(projection.speech_budget_win, torch.tensor([[3.0]], dtype=torch.float32)))
        self.assertTrue(torch.allclose(projection.pause_budget_win, torch.tensor([[3.0]], dtype=torch.float32)))
        self.assertTrue(torch.allclose(projection.speech_budget_delta, torch.tensor([[2.0]], dtype=torch.float32)))
        self.assertTrue(torch.allclose(projection.pause_budget_delta, torch.tensor([[0.0]], dtype=torch.float32)))
        self.assertTrue(torch.allclose(projection.total_budget_delta, torch.tensor([[0.0]], dtype=torch.float32)))

    def test_committed_prefix_pause_shortage_only_repairs_pause_branch(self) -> None:
        projection = lift_projector_budgets_to_feasible_region(
            dur_anchor_src=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
            unit_mask=torch.ones((1, 3), dtype=torch.float32),
            speech_budget_win=torch.tensor([[4.0]], dtype=torch.float32),
            pause_budget_win=torch.tensor([[1.0]], dtype=torch.float32),
            previous_speech_exec=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
            previous_pause_exec=torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float32),
            commit_frontier=torch.tensor([1], dtype=torch.long),
            reuse_prefix=True,
            min_speech_frames=1.0,
            max_speech_expand=3.0,
        )
        self.assertTrue(torch.allclose(projection.speech_budget_win, torch.tensor([[3.0]], dtype=torch.float32)))
        self.assertTrue(torch.allclose(projection.pause_budget_win, torch.tensor([[2.0]], dtype=torch.float32)))
        self.assertTrue(torch.allclose(projection.speech_budget_delta, torch.tensor([[0.0]], dtype=torch.float32)))
        self.assertTrue(torch.allclose(projection.pause_budget_delta, torch.tensor([[1.0]], dtype=torch.float32)))
        self.assertTrue(torch.allclose(projection.total_budget_delta, torch.tensor([[0.0]], dtype=torch.float32)))

    def test_total_budget_is_lifted_only_when_raw_total_is_infeasible(self) -> None:
        projection = lift_projector_budgets_to_feasible_region(
            dur_anchor_src=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
            unit_mask=torch.ones((1, 3), dtype=torch.float32),
            speech_budget_win=torch.tensor([[2.5]], dtype=torch.float32),
            pause_budget_win=torch.tensor([[1.0]], dtype=torch.float32),
            previous_speech_exec=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
            previous_pause_exec=torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float32),
            commit_frontier=torch.tensor([1], dtype=torch.long),
            reuse_prefix=True,
            min_speech_frames=1.0,
            max_speech_expand=3.0,
        )
        self.assertTrue(torch.allclose(projection.speech_budget_win, torch.tensor([[3.0]], dtype=torch.float32)))
        self.assertTrue(torch.allclose(projection.pause_budget_win, torch.tensor([[2.0]], dtype=torch.float32)))
        self.assertTrue(torch.allclose(projection.speech_budget_delta, torch.tensor([[0.5]], dtype=torch.float32)))
        self.assertTrue(torch.allclose(projection.pause_budget_delta, torch.tensor([[1.0]], dtype=torch.float32)))
        self.assertTrue(torch.allclose(projection.total_budget_delta, torch.tensor([[1.5]], dtype=torch.float32)))

    def test_per_unit_upper_bound_never_falls_below_lower_bound(self) -> None:
        projector = StreamingRhythmProjector(
            ProjectorConfig(
                min_speech_frames=1.0,
                max_speech_expand=0.25,
                build_render_plan=False,
            )
        )
        state = projector.init_state(batch_size=1, device=torch.device("cpu"))
        speech = projector._project_speech(
            dur_anchor_src=torch.tensor([[0.2, 0.4]], dtype=torch.float32),
            dur_logratio_unit=torch.zeros((1, 2), dtype=torch.float32),
            unit_mask=torch.ones((1, 2), dtype=torch.float32),
            speech_budget_win=torch.tensor([[2.0]], dtype=torch.float32),
            state=state,
            reuse_prefix=False,
        )
        self.assertTrue(torch.all(speech >= 1.0))
        self.assertTrue(torch.allclose(speech.sum(dim=1), torch.tensor([2.0], dtype=torch.float32)))

    def test_speech_projection_only_allocates_tail_mass_inside_segment_mask(self) -> None:
        projector = StreamingRhythmProjector(
            ProjectorConfig(
                min_speech_frames=1.0,
                max_speech_expand=3.0,
                build_render_plan=False,
            )
        )
        state = StreamingRhythmState(
            phase_ptr=torch.tensor([0.2], dtype=torch.float32),
            clock_delta=torch.tensor([0.0], dtype=torch.float32),
            commit_frontier=torch.tensor([1], dtype=torch.long),
            previous_speech_exec=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        )
        speech = projector._project_speech(
            dur_anchor_src=torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32),
            dur_logratio_unit=torch.zeros((1, 4), dtype=torch.float32),
            unit_mask=torch.ones((1, 4), dtype=torch.float32),
            speech_budget_win=torch.tensor([[4.0]], dtype=torch.float32),
            state=state,
            reuse_prefix=True,
            segment_mask=torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
        )
        self.assertAlmostEqual(float(speech[0, 0].item()), 1.0, places=6)
        self.assertAlmostEqual(float(speech[0, 1].item()), 0.0, places=6)
        self.assertGreater(float(speech[0, 2].item()), 0.0)
        self.assertGreater(float(speech[0, 3].item()), 0.0)
        self.assertTrue(torch.allclose(speech.sum(dim=1), torch.tensor([4.0], dtype=torch.float32)))

    def test_forward_prefers_phrase_projection_budgets_when_present(self) -> None:
        projector = StreamingRhythmProjector(
            ProjectorConfig(
                min_speech_frames=0.0,
                max_speech_expand=10.0,
                pause_selection_mode="simple",
                build_render_plan=False,
            )
        )
        planner = RhythmPlannerOutputs(
            speech_budget_win=torch.tensor([[5.0]], dtype=torch.float32),
            pause_budget_win=torch.tensor([[3.0]], dtype=torch.float32),
            dur_logratio_unit=torch.zeros((1, 4), dtype=torch.float32),
            pause_weight_unit=torch.tensor([[0.0, 0.1, 0.2, 0.7]], dtype=torch.float32),
            boundary_score_unit=torch.tensor([[0.0, 0.2, 0.8, 1.0]], dtype=torch.float32),
            trace_context=torch.zeros((1, 4, 2), dtype=torch.float32),
            source_boundary_cue=torch.zeros((1, 4), dtype=torch.float32),
            phrase_speech_budget_win=torch.tensor([[2.0]], dtype=torch.float32),
            phrase_pause_budget_win=torch.tensor([[1.0]], dtype=torch.float32),
            segment_mask_unit=torch.ones((1, 4), dtype=torch.float32),
            pause_segment_mask_unit=torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
        )
        execution = projector(
            dur_anchor_src=torch.ones((1, 4), dtype=torch.float32),
            unit_mask=torch.ones((1, 4), dtype=torch.float32),
            speech_budget_win=planner.speech_budget_win,
            pause_budget_win=planner.pause_budget_win,
            dur_logratio_unit=planner.dur_logratio_unit,
            pause_weight_unit=planner.pause_weight_unit,
            boundary_score_unit=planner.boundary_score_unit,
            state=projector.init_state(batch_size=1, device=torch.device("cpu")),
            planner=planner,
            reuse_prefix=False,
            force_full_commit=True,
        )
        self.assertTrue(torch.allclose(execution.speech_duration_exec.sum(dim=1), torch.tensor([2.0])))
        self.assertTrue(torch.allclose(execution.pause_after_exec.sum(dim=1), torch.tensor([1.0])))
        self.assertTrue(torch.allclose(execution.planner.phrase_speech_budget_win, torch.tensor([[2.0]])))
        self.assertTrue(torch.allclose(execution.planner.phrase_pause_budget_win, torch.tensor([[1.0]])))

    def test_forward_sanitizes_pause_mask_to_segment_tail(self) -> None:
        projector = StreamingRhythmProjector(
            ProjectorConfig(
                min_speech_frames=0.0,
                max_speech_expand=10.0,
                pause_selection_mode="simple",
                build_render_plan=False,
            )
        )
        state = StreamingRhythmState(
            phase_ptr=torch.tensor([0.0], dtype=torch.float32),
            clock_delta=torch.tensor([0.0], dtype=torch.float32),
            commit_frontier=torch.tensor([1], dtype=torch.long),
        )
        planner = RhythmPlannerOutputs(
            speech_budget_win=torch.tensor([[2.0]], dtype=torch.float32),
            pause_budget_win=torch.tensor([[1.0]], dtype=torch.float32),
            dur_logratio_unit=torch.zeros((1, 4), dtype=torch.float32),
            pause_weight_unit=torch.ones((1, 4), dtype=torch.float32),
            boundary_score_unit=torch.zeros((1, 4), dtype=torch.float32),
            trace_context=torch.zeros((1, 4, 2), dtype=torch.float32),
            segment_mask_unit=torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
            pause_segment_mask_unit=torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32),
        )
        execution = projector(
            dur_anchor_src=torch.ones((1, 4), dtype=torch.float32),
            unit_mask=torch.ones((1, 4), dtype=torch.float32),
            speech_budget_win=planner.speech_budget_win,
            pause_budget_win=planner.pause_budget_win,
            dur_logratio_unit=planner.dur_logratio_unit,
            pause_weight_unit=planner.pause_weight_unit,
            boundary_score_unit=planner.boundary_score_unit,
            state=state,
            planner=planner,
            reuse_prefix=True,
            force_full_commit=False,
        )
        self.assertAlmostEqual(float(execution.pause_after_exec[0, :2].sum().item()), 0.0, places=6)
        self.assertTrue(torch.allclose(execution.planner.segment_mask_unit, torch.tensor([[0.0, 0.0, 1.0, 1.0]])))
        self.assertTrue(
            torch.allclose(
                execution.planner.pause_segment_mask_unit,
                torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
            )
        )

    def test_pause_simple_zero_scores_fallbacks_to_last_eligible_slot(self) -> None:
        pause = _project_pause_simple_impl(
            pause_weight_unit=torch.zeros((1, 4), dtype=torch.float32),
            boundary_score_unit=torch.zeros((1, 4), dtype=torch.float32),
            unit_mask=torch.ones((1, 4), dtype=torch.float32),
            pause_budget_win=torch.tensor([[1.0]], dtype=torch.float32),
            previous_pause_exec=None,
            commit_frontier=torch.tensor([1], dtype=torch.long),
            reuse_prefix=True,
            pause_min_boundary_weight=0.1,
            pause_boundary_bias_weight=0.15,
            allocation_mask=torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
        )
        self.assertAlmostEqual(float(pause[0, 2].item()), 0.0, places=6)
        self.assertAlmostEqual(float(pause[0, 3].item()), 1.0, places=6)

    def test_commit_frontier_never_rolls_back(self) -> None:
        projector = StreamingRhythmProjector(
            ProjectorConfig(
                tail_hold_units=1,
                use_boundary_commit_guard=False,
                build_render_plan=False,
            )
        )
        state = StreamingRhythmState(
            phase_ptr=torch.tensor([0.4], dtype=torch.float32),
            clock_delta=torch.tensor([0.0], dtype=torch.float32),
            commit_frontier=torch.tensor([2], dtype=torch.long),
        )
        commit_frontier = projector._compute_commit_frontier(
            state=state,
            unit_mask=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
            open_run_mask=torch.tensor([[0, 1, 1]], dtype=torch.long),
            boundary_score_unit=None,
            force_full_commit=False,
        )
        self.assertTrue(torch.equal(commit_frontier, torch.tensor([2], dtype=torch.long)))

    def test_phase_ptr_does_not_roll_back_when_visible_total_grows(self) -> None:
        projector = StreamingRhythmProjector(
            ProjectorConfig(
                build_render_plan=False,
            )
        )
        state = StreamingRhythmState(
            phase_ptr=torch.tensor([0.60], dtype=torch.float32),
            clock_delta=torch.tensor([0.0], dtype=torch.float32),
            commit_frontier=torch.tensor([3], dtype=torch.long),
            phase_anchor=torch.tensor([[3.0, 5.0]], dtype=torch.float32),
        )
        next_state = projector._advance_state(
            state=state,
            dur_anchor_src=torch.tensor([[1.0, 1.0, 1.0, 10.0]], dtype=torch.float32),
            unit_mask=torch.ones((1, 4), dtype=torch.float32),
            effective_duration_exec=torch.tensor([[1.0, 1.0, 1.0, 0.0]], dtype=torch.float32),
            commit_frontier=torch.tensor([3], dtype=torch.long),
            speech_duration_exec=torch.tensor([[1.0, 1.0, 1.0, 0.0]], dtype=torch.float32),
            pause_after_exec=torch.zeros((1, 4), dtype=torch.float32),
        )
        self.assertTrue(torch.allclose(next_state.phase_ptr, torch.tensor([0.60], dtype=torch.float32)))
        self.assertTrue(torch.allclose(next_state.phase_progress_ratio, torch.tensor([0.60], dtype=torch.float32)))

    def test_phase_anchor_total_uses_masked_visible_anchor(self) -> None:
        projector = StreamingRhythmProjector(ProjectorConfig(build_render_plan=False))
        state = StreamingRhythmState(
            phase_ptr=torch.tensor([0.25], dtype=torch.float32),
            clock_delta=torch.tensor([0.0], dtype=torch.float32),
            commit_frontier=torch.tensor([1], dtype=torch.long),
            phase_anchor=torch.tensor([[1.0, 4.0]], dtype=torch.float32),
        )
        next_state = projector._advance_state(
            state=state,
            dur_anchor_src=torch.tensor([[1.0, 1.0, 1.0, 10.0]], dtype=torch.float32),
            unit_mask=torch.tensor([[1.0, 1.0, 1.0, 0.0]], dtype=torch.float32),
            effective_duration_exec=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            commit_frontier=torch.tensor([1], dtype=torch.long),
            speech_duration_exec=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            pause_after_exec=torch.zeros((1, 4), dtype=torch.float32),
        )
        self.assertTrue(torch.allclose(next_state.phase_anchor[:, 1], torch.tensor([4.0], dtype=torch.float32)))
        self.assertTrue(torch.allclose(next_state.phase_progress_ratio, torch.tensor([0.25], dtype=torch.float32)))

    def test_pause_projection_preserves_reused_prefix_mass(self) -> None:
        pause = _project_pause_impl(
            pause_weight_unit=torch.tensor([[0.1, 0.3, 0.6]], dtype=torch.float32),
            boundary_score_unit=torch.zeros((1, 3), dtype=torch.float32),
            unit_mask=torch.ones((1, 3), dtype=torch.float32),
            pause_budget_win=torch.tensor([[3.0]], dtype=torch.float32),
            previous_pause_exec=torch.tensor([[2.0, 0.0]], dtype=torch.float32),
            commit_frontier=torch.tensor([1], dtype=torch.long),
            reuse_prefix=True,
            soft_pause_selection=False,
            topk_ratio=1.0,
            pause_min_boundary_weight=0.10,
            pause_boundary_bias_weight=0.15,
            temperature=0.12,
        )
        self.assertTrue(torch.allclose(pause[:, :1], torch.tensor([[2.0]], dtype=torch.float32)))
        self.assertTrue(torch.allclose(pause.sum(dim=1), torch.tensor([3.0], dtype=torch.float32)))

    def test_sparse_pause_topk_ignores_reused_prefix_slots(self) -> None:
        pause = _project_pause_impl(
            pause_weight_unit=torch.tensor([[0.95, 0.80, 0.40, 0.20]], dtype=torch.float32),
            boundary_score_unit=torch.zeros((1, 4), dtype=torch.float32),
            unit_mask=torch.ones((1, 4), dtype=torch.float32),
            allocation_mask=torch.ones((1, 4), dtype=torch.float32),
            pause_budget_win=torch.tensor([[2.0]], dtype=torch.float32),
            previous_pause_exec=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            commit_frontier=torch.tensor([2], dtype=torch.long),
            reuse_prefix=True,
            soft_pause_selection=False,
            topk_ratio=0.25,
            pause_min_boundary_weight=0.10,
            pause_boundary_bias_weight=0.0,
            temperature=0.20,
        )
        self.assertTrue(torch.allclose(pause[:, :2], torch.tensor([[1.0, 0.0]], dtype=torch.float32)))
        self.assertGreater(float(pause[0, 2].item()), 0.99)
        self.assertAlmostEqual(float(pause[0, 3].item()), 0.0, places=7)

    def test_append_only_invariance_keeps_committed_prefix_execution(self) -> None:
        projector = self._make_invariance_projector()
        short = self._run_projector(
            projector,
            state=projector.init_state(batch_size=1, device=torch.device("cpu")),
            dur_anchor_src=[2.0, 1.0, 0.0],
            open_run_mask=[0, 0, 1],
            reuse_prefix=False,
        )
        extended = self._run_projector(
            projector,
            state=short.next_state,
            dur_anchor_src=[2.0, 1.0, 0.0, 0.0, 0.0],
            open_run_mask=[0, 0, 1, 1, 1],
            reuse_prefix=True,
        )
        committed = int(short.commit_frontier[0].item())
        self.assertGreater(committed, 0)
        self.assertTrue(
            torch.allclose(
                short.speech_duration_exec[:, :committed],
                extended.speech_duration_exec[:, :committed],
            )
        )
        self.assertTrue(
            torch.allclose(
                short.pause_after_exec[:, :committed],
                extended.pause_after_exec[:, :committed],
            )
        )
        self.assertEqual(int(extended.commit_frontier[0].item()), committed)

    def test_chunking_invariance_matches_single_pass_on_committed_prefix(self) -> None:
        projector = self._make_invariance_projector()
        base_state = projector.init_state(batch_size=1, device=torch.device("cpu"))
        single_pass = self._run_projector(
            projector,
            state=base_state,
            dur_anchor_src=[2.0, 1.0, 0.0, 0.0, 0.0],
            open_run_mask=[0, 0, 1, 1, 1],
            reuse_prefix=False,
        )
        first_chunk = self._run_projector(
            projector,
            state=base_state,
            dur_anchor_src=[2.0, 1.0, 0.0],
            open_run_mask=[0, 0, 1],
            reuse_prefix=False,
        )
        stitched = self._run_projector(
            projector,
            state=first_chunk.next_state,
            dur_anchor_src=[2.0, 1.0, 0.0, 0.0, 0.0],
            open_run_mask=[0, 0, 1, 1, 1],
            reuse_prefix=True,
        )
        committed = int(single_pass.commit_frontier[0].item())
        self.assertEqual(committed, int(stitched.commit_frontier[0].item()))
        self.assertTrue(
            torch.allclose(
                single_pass.speech_duration_exec[:, :committed],
                stitched.speech_duration_exec[:, :committed],
            )
        )
        self.assertTrue(
            torch.allclose(
                single_pass.pause_after_exec[:, :committed],
                stitched.pause_after_exec[:, :committed],
            )
        )

    def test_reuse_prefix_rejects_shorter_current_chunk_than_committed_frontier(self) -> None:
        projector = self._make_invariance_projector()
        state = StreamingRhythmState(
            phase_ptr=torch.tensor([0.0], dtype=torch.float32),
            clock_delta=torch.tensor([0.0], dtype=torch.float32),
            commit_frontier=torch.tensor([2], dtype=torch.long),
            previous_speech_exec=torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float32),
            previous_pause_exec=torch.tensor([[0.0, 0.5, 0.0]], dtype=torch.float32),
        )
        with self.assertRaises(ValueError):
            self._run_projector(
                projector,
                state=state,
                dur_anchor_src=[2.0],
                open_run_mask=[0],
                reuse_prefix=True,
            )

    def test_reuse_prefix_rejects_holes_inside_committed_prefix(self) -> None:
        projector = self._make_invariance_projector()
        planner = RhythmPlannerOutputs(
            speech_budget_win=torch.tensor([[3.0]], dtype=torch.float32),
            pause_budget_win=torch.tensor([[1.0]], dtype=torch.float32),
            dur_logratio_unit=torch.zeros((1, 4), dtype=torch.float32),
            pause_weight_unit=torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32),
            boundary_score_unit=torch.zeros((1, 4), dtype=torch.float32),
            trace_context=torch.zeros((1, 4, 2), dtype=torch.float32),
            source_boundary_cue=torch.zeros((1, 4), dtype=torch.float32),
        )
        state = StreamingRhythmState(
            phase_ptr=torch.tensor([0.0], dtype=torch.float32),
            clock_delta=torch.tensor([0.0], dtype=torch.float32),
            commit_frontier=torch.tensor([2], dtype=torch.long),
            previous_speech_exec=torch.tensor([[2.0, 1.0, 0.0, 0.0]], dtype=torch.float32),
            previous_pause_exec=torch.tensor([[0.0, 0.5, 0.0, 0.0]], dtype=torch.float32),
        )
        with self.assertRaises(ValueError):
            projector(
                dur_anchor_src=torch.tensor([[2.0, 1.0, 0.0, 0.0]], dtype=torch.float32),
                unit_mask=torch.tensor([[1.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
                speech_budget_win=planner.speech_budget_win,
                pause_budget_win=planner.pause_budget_win,
                dur_logratio_unit=planner.dur_logratio_unit,
                pause_weight_unit=planner.pause_weight_unit,
                boundary_score_unit=planner.boundary_score_unit,
                state=state,
                open_run_mask=torch.tensor([[0, 0, 1, 1]], dtype=torch.long),
                planner=planner,
                reuse_prefix=True,
                force_full_commit=False,
            )

    def test_pause_locality_keeps_pause_mass_inside_boundary_like_slots(self) -> None:
        pause = _project_pause_impl(
            pause_weight_unit=torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]], dtype=torch.float32),
            boundary_score_unit=torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0]], dtype=torch.float32),
            unit_mask=torch.ones((1, 5), dtype=torch.float32),
            allocation_mask=torch.tensor([[1.0, 1.0, 0.0, 1.0, 0.0]], dtype=torch.float32),
            pause_budget_win=torch.tensor([[1.5]], dtype=torch.float32),
            previous_pause_exec=torch.tensor([[0.5, 0.0]], dtype=torch.float32),
            commit_frontier=torch.tensor([2], dtype=torch.long),
            reuse_prefix=True,
            soft_pause_selection=False,
            topk_ratio=0.40,
            pause_min_boundary_weight=0.10,
            pause_boundary_bias_weight=0.0,
            temperature=0.20,
        )
        self.assertTrue(torch.allclose(pause[:, :2], torch.tensor([[0.5, 0.0]], dtype=torch.float32)))
        self.assertAlmostEqual(float(pause[0, 2].item()), 0.0, places=7)
        self.assertGreater(float(pause[0, 3].item()), 0.99)
        self.assertAlmostEqual(float(pause[0, 4].item()), 0.0, places=7)

    def test_sparse_pause_boundary_gain_does_not_create_absolute_floor(self) -> None:
        pause = _project_pause_impl(
            pause_weight_unit=torch.tensor([[0.020, 0.000, 0.019, 0.018]], dtype=torch.float32),
            boundary_score_unit=torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32),
            unit_mask=torch.ones((1, 4), dtype=torch.float32),
            pause_budget_win=torch.tensor([[1.0]], dtype=torch.float32),
            previous_pause_exec=None,
            commit_frontier=torch.tensor([0], dtype=torch.long),
            reuse_prefix=False,
            soft_pause_selection=False,
            topk_ratio=0.25,
            pause_min_boundary_weight=0.10,
            pause_boundary_bias_weight=0.18,
            temperature=0.12,
        )
        self.assertGreater(float(pause[0, 0].item()), 0.99)
        self.assertAlmostEqual(float(pause[0, 1].item()), 0.0, places=7)

    def test_simple_pause_boundary_gain_does_not_create_absolute_floor(self) -> None:
        pause = _project_pause_simple_impl(
            pause_weight_unit=torch.tensor([[0.020, 0.000, 0.019, 0.018]], dtype=torch.float32),
            boundary_score_unit=torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32),
            unit_mask=torch.ones((1, 4), dtype=torch.float32),
            pause_budget_win=torch.tensor([[1.0]], dtype=torch.float32),
            previous_pause_exec=None,
            commit_frontier=torch.tensor([0], dtype=torch.long),
            reuse_prefix=False,
            pause_min_boundary_weight=0.10,
            pause_boundary_bias_weight=0.18,
        )
        self.assertAlmostEqual(float(pause[0, 1].item()), 0.0, places=7)
        self.assertGreater(float(pause[0, 0].item()), float(pause[0, 2].item()))

    def test_pause_fallback_prefers_boundary_slots_inside_allocation_mask(self) -> None:
        pause = _project_pause_impl(
            pause_weight_unit=torch.zeros((1, 4), dtype=torch.float32),
            boundary_score_unit=torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32),
            unit_mask=torch.ones((1, 4), dtype=torch.float32),
            allocation_mask=torch.tensor([[1.0, 1.0, 1.0, 0.0]], dtype=torch.float32),
            pause_budget_win=torch.tensor([[1.0]], dtype=torch.float32),
            previous_pause_exec=None,
            commit_frontier=torch.tensor([0], dtype=torch.long),
            reuse_prefix=False,
            soft_pause_selection=False,
            topk_ratio=0.25,
            pause_min_boundary_weight=0.10,
            pause_boundary_bias_weight=0.18,
            temperature=0.12,
        )
        self.assertGreater(float(pause[0, 1].item()), 0.99)
        self.assertAlmostEqual(float(pause[0, 3].item()), 0.0, places=7)

    def test_sparse_pause_topk_ratio_uses_allocation_domain_not_full_visible_domain(self) -> None:
        pause = _project_pause_impl(
            pause_weight_unit=torch.tensor([[0.0, 0.90, 0.0, 0.60, 0.0, 0.0]], dtype=torch.float32),
            boundary_score_unit=torch.zeros((1, 6), dtype=torch.float32),
            unit_mask=torch.ones((1, 6), dtype=torch.float32),
            allocation_mask=torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0, 0.0]], dtype=torch.float32),
            pause_budget_win=torch.tensor([[1.0]], dtype=torch.float32),
            previous_pause_exec=None,
            commit_frontier=torch.tensor([0], dtype=torch.long),
            reuse_prefix=False,
            soft_pause_selection=False,
            topk_ratio=0.50,
            pause_min_boundary_weight=0.10,
            pause_boundary_bias_weight=0.0,
            temperature=0.20,
        )
        self.assertAlmostEqual(float(pause[0, 1].item()), 1.0, places=6)
        self.assertAlmostEqual(float(pause[0, 3].item()), 0.0, places=6)
        self.assertAlmostEqual(float(pause[0, [0, 2, 4, 5]].sum().item()), 0.0, places=6)

    def test_forward_reports_sanitized_planned_commit_frontier_not_stale_input(self) -> None:
        projector = StreamingRhythmProjector(ProjectorConfig(build_render_plan=False, pause_selection_mode="simple"))
        planner = RhythmPlannerOutputs(
            speech_budget_win=torch.tensor([[4.0]], dtype=torch.float32),
            pause_budget_win=torch.tensor([[1.0]], dtype=torch.float32),
            dur_logratio_unit=torch.zeros((1, 4), dtype=torch.float32),
            pause_weight_unit=torch.tensor([[0.0, 0.0, 0.2, 0.8]], dtype=torch.float32),
            boundary_score_unit=torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
            trace_context=torch.zeros((1, 4, 2), dtype=torch.float32),
            source_boundary_cue=torch.zeros((1, 4), dtype=torch.float32),
            planned_commit_frontier=torch.tensor([[99.0]], dtype=torch.float32),
            segment_mask_unit=torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
            pause_segment_mask_unit=torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
        )
        state = StreamingRhythmState(
            phase_ptr=torch.tensor([0.0], dtype=torch.float32),
            clock_delta=torch.tensor([0.0], dtype=torch.float32),
            commit_frontier=torch.tensor([2], dtype=torch.long),
            previous_speech_exec=torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32),
            previous_pause_exec=torch.tensor([[0.2, 0.0, 0.0, 0.0]], dtype=torch.float32),
        )
        execution = projector(
            dur_anchor_src=torch.ones((1, 4), dtype=torch.float32),
            unit_mask=torch.tensor([[1.0, 1.0, 1.0, 0.0]], dtype=torch.float32),
            speech_budget_win=planner.speech_budget_win,
            pause_budget_win=planner.pause_budget_win,
            dur_logratio_unit=planner.dur_logratio_unit,
            pause_weight_unit=planner.pause_weight_unit,
            boundary_score_unit=planner.boundary_score_unit,
            state=state,
            planner=planner,
            reuse_prefix=True,
            force_full_commit=False,
        )
        self.assertEqual(int(execution.commit_frontier[0].item()), 3)
        self.assertEqual(int(execution.planner.planned_commit_frontier[0].item()), 3)

    def test_forward_prefers_planned_commit_frontier_and_segment_masks(self) -> None:
        projector = StreamingRhythmProjector(ProjectorConfig(build_render_plan=False, pause_selection_mode="simple"))
        planner = RhythmPlannerOutputs(
            speech_budget_win=torch.tensor([[4.0]], dtype=torch.float32),
            pause_budget_win=torch.tensor([[1.0]], dtype=torch.float32),
            dur_logratio_unit=torch.zeros((1, 4), dtype=torch.float32),
            pause_weight_unit=torch.tensor([[0.0, 0.0, 0.2, 0.8]], dtype=torch.float32),
            boundary_score_unit=torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
            trace_context=torch.zeros((1, 4, 2), dtype=torch.float32),
            source_boundary_cue=torch.zeros((1, 4), dtype=torch.float32),
            planned_commit_frontier=torch.tensor([4], dtype=torch.long),
            segment_mask_unit=torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
            pause_segment_mask_unit=torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
            active_phrase_start=torch.tensor([2], dtype=torch.long),
            active_phrase_end=torch.tensor([4], dtype=torch.long),
        )
        state = StreamingRhythmState(
            phase_ptr=torch.tensor([0.0], dtype=torch.float32),
            clock_delta=torch.tensor([0.0], dtype=torch.float32),
            commit_frontier=torch.tensor([2], dtype=torch.long),
            previous_speech_exec=torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32),
            previous_pause_exec=torch.tensor([[0.2, 0.0, 0.0, 0.0]], dtype=torch.float32),
        )
        execution = projector(
            dur_anchor_src=torch.ones((1, 4), dtype=torch.float32),
            unit_mask=torch.ones((1, 4), dtype=torch.float32),
            speech_budget_win=planner.speech_budget_win,
            pause_budget_win=planner.pause_budget_win,
            dur_logratio_unit=planner.dur_logratio_unit,
            pause_weight_unit=planner.pause_weight_unit,
            boundary_score_unit=planner.boundary_score_unit,
            state=state,
            planner=planner,
            reuse_prefix=True,
            force_full_commit=False,
        )
        self.assertEqual(int(execution.commit_frontier[0].item()), 4)
        self.assertTrue(torch.allclose(execution.speech_duration_exec[0, :2], torch.tensor([1.0, 1.0])))
        self.assertAlmostEqual(float(execution.pause_after_exec[0, 2].item()), 0.0, places=6)
        self.assertGreater(float(execution.pause_after_exec[0, 3].item()), 0.7)
        self.assertIsNotNone(execution.planner.local_rho_unit)
        self.assertIsNotNone(execution.planner.intra_phrase_alpha)
        self.assertGreaterEqual(float(execution.planner.intra_phrase_alpha[0, 0].item()), 0.0)
        self.assertEqual(int(execution.planner.active_phrase_start[0].item()), 2)
        self.assertEqual(int(execution.planner.active_phrase_end[0].item()), 4)

    def test_scheduler_exposes_chunk_state_and_phrase_prototype(self) -> None:
        scheduler = MonotonicRhythmScheduler(hidden_size=8, stats_dim=2, trace_dim=2)
        chunk_state = ChunkStateBundle(
            chunk_summary=torch.tensor([[0.4, 0.6, 0.75, 0.5, 0.1, 0.7]], dtype=torch.float32),
            structure_progress=torch.tensor([0.3], dtype=torch.float32),
            commit_now_prob=torch.tensor([0.8], dtype=torch.float32),
            phrase_open_prob=torch.tensor([0.15], dtype=torch.float32),
            phrase_close_prob=torch.tensor([0.65], dtype=torch.float32),
            phrase_role_prob=torch.tensor([[0.15, 0.20, 0.65]], dtype=torch.float32),
            active_tail_mask=torch.ones((1, 3), dtype=torch.float32),
        )
        scheduler.chunk_state_head = _ChunkStateHeadStub(chunk_state)
        phrase_selection = {
            "selected_phrase_prototype_summary": torch.tensor([[0.5, 0.2]], dtype=torch.float32),
            "selected_phrase_prototype_stats": torch.tensor([[0.15, 0.25]], dtype=torch.float32),
            "selected_phrase_prototype_valid": torch.tensor([[1.0]], dtype=torch.float32),
        }
        planner = scheduler(
            unit_states=torch.zeros((1, 3, 8), dtype=torch.float32),
            dur_anchor_src=torch.ones((1, 3), dtype=torch.float32),
            unit_mask=torch.ones((1, 3), dtype=torch.float32),
            ref_conditioning={"planner_ref_stats": torch.tensor([[1.0, 0.25]], dtype=torch.float32)},
            trace_context=torch.zeros((1, 3, 2), dtype=torch.float32),
            planner_trace_context=torch.zeros((1, 3, 2), dtype=torch.float32),
            state=StreamingRhythmState(
                phase_ptr=torch.zeros((1,), dtype=torch.float32),
                clock_delta=torch.zeros((1,), dtype=torch.float32),
                commit_frontier=torch.zeros((1,), dtype=torch.long),
            ),
            phrase_selection=phrase_selection,
        )
        self.assertTrue(torch.allclose(planner.chunk_commit_prob, chunk_state.commit_now_prob))
        self.assertTrue(torch.allclose(planner.chunk_structure_progress, chunk_state.structure_progress))
        self.assertIsNotNone(planner.phrase_prototype_summary)
        self.assertIsNotNone(planner.phrase_prototype_stats)

    def test_projector_passes_phrase_state_to_execution(self) -> None:
        scheduler = MonotonicRhythmScheduler(hidden_size=8, stats_dim=2, trace_dim=2)
        chunk_state = ChunkStateBundle(
            chunk_summary=torch.tensor([[0.2, 0.4, 0.5, 0.2, 0.3, 0.4]], dtype=torch.float32),
            structure_progress=torch.tensor([0.4], dtype=torch.float32),
            commit_now_prob=torch.tensor([0.6], dtype=torch.float32),
            phrase_open_prob=torch.tensor([0.1], dtype=torch.float32),
            phrase_close_prob=torch.tensor([0.6], dtype=torch.float32),
            phrase_role_prob=torch.tensor([[0.1, 0.3, 0.6]], dtype=torch.float32),
            active_tail_mask=torch.ones((1, 3), dtype=torch.float32),
        )
        scheduler.chunk_state_head = _ChunkStateHeadStub(chunk_state)
        phrase_selection = {
            "selected_phrase_prototype_summary": torch.tensor([[0.4, 0.1]], dtype=torch.float32),
            "selected_phrase_prototype_stats": torch.tensor([[0.2, 0.2]], dtype=torch.float32),
            "selected_phrase_prototype_valid": torch.tensor([[1.0]], dtype=torch.float32),
        }
        planner = scheduler(
            unit_states=torch.zeros((1, 3, 8), dtype=torch.float32),
            dur_anchor_src=torch.ones((1, 3), dtype=torch.float32),
            unit_mask=torch.ones((1, 3), dtype=torch.float32),
            ref_conditioning={"planner_ref_stats": torch.tensor([[1.0, 0.25]], dtype=torch.float32)},
            trace_context=torch.zeros((1, 3, 2), dtype=torch.float32),
            planner_trace_context=torch.zeros((1, 3, 2), dtype=torch.float32),
            state=StreamingRhythmState(
                phase_ptr=torch.zeros((1,), dtype=torch.float32),
                clock_delta=torch.zeros((1,), dtype=torch.float32),
                commit_frontier=torch.zeros((1,), dtype=torch.long),
            ),
            phrase_selection=phrase_selection,
        )
        projector = StreamingRhythmProjector(ProjectorConfig(build_render_plan=False))
        execution = projector(
            dur_anchor_src=torch.ones((1, 3), dtype=torch.float32),
            unit_mask=torch.ones((1, 3), dtype=torch.float32),
            speech_budget_win=planner.speech_budget_win,
            pause_budget_win=planner.pause_budget_win,
            dur_logratio_unit=planner.dur_logratio_unit,
            pause_weight_unit=planner.pause_weight_unit,
            boundary_score_unit=planner.boundary_score_unit,
            state=projector.init_state(batch_size=1, device=torch.device("cpu")),
            planner=planner,
            reuse_prefix=False,
            force_full_commit=False,
        )
        self.assertTrue(torch.allclose(execution.planner.chunk_commit_prob, chunk_state.commit_now_prob))
        self.assertTrue(torch.allclose(execution.planner.chunk_structure_progress, chunk_state.structure_progress))
        self.assertIsNotNone(execution.planner.phrase_prototype_summary)
        self.assertIsNotNone(execution.planner.phrase_prototype_stats)

    def test_forward_treats_fractional_masks_as_binary_when_binary_sanitize_is_available(self) -> None:
        sanitized = _resolve_allocation_mask(
            torch.ones((1, 4), dtype=torch.float32),
            torch.tensor([[0.49, 0.51, 0.0, 1.0]], dtype=torch.float32),
        )
        if not torch.all((sanitized == 0.0) | (sanitized == 1.0)):
            self.skipTest("binary allocation-mask sanitize not landed yet")

        projector = StreamingRhythmProjector(ProjectorConfig(build_render_plan=False, pause_selection_mode="simple"))
        planner = RhythmPlannerOutputs(
            speech_budget_win=torch.tensor([[4.0]], dtype=torch.float32),
            pause_budget_win=torch.tensor([[1.0]], dtype=torch.float32),
            dur_logratio_unit=torch.zeros((1, 4), dtype=torch.float32),
            pause_weight_unit=torch.tensor([[0.2, 0.4, 0.1, 0.3]], dtype=torch.float32),
            boundary_score_unit=torch.zeros((1, 4), dtype=torch.float32),
            trace_context=torch.zeros((1, 4, 2), dtype=torch.float32),
            source_boundary_cue=torch.zeros((1, 4), dtype=torch.float32),
            segment_mask_unit=torch.tensor([[0.49, 0.51, 0.0, 1.0]], dtype=torch.float32),
            pause_segment_mask_unit=torch.tensor([[0.49, 0.51, 0.0, 1.0]], dtype=torch.float32),
        )
        execution = projector(
            dur_anchor_src=torch.ones((1, 4), dtype=torch.float32),
            unit_mask=torch.ones((1, 4), dtype=torch.float32),
            speech_budget_win=planner.speech_budget_win,
            pause_budget_win=planner.pause_budget_win,
            dur_logratio_unit=planner.dur_logratio_unit,
            pause_weight_unit=planner.pause_weight_unit,
            boundary_score_unit=planner.boundary_score_unit,
            state=projector.init_state(batch_size=1, device=torch.device("cpu")),
            planner=planner,
            reuse_prefix=False,
            force_full_commit=True,
        )
        inactive = ~(sanitized[0] > 0.5)
        self.assertTrue(
            torch.allclose(
                execution.speech_duration_exec[0, inactive],
                torch.zeros_like(execution.speech_duration_exec[0, inactive]),
            )
        )
        self.assertTrue(
            torch.allclose(
                execution.pause_after_exec[0, inactive],
                torch.zeros_like(execution.pause_after_exec[0, inactive]),
            )
        )

    def test_forward_sanitizes_planned_commit_frontier_before_state_update(self) -> None:
        projector = StreamingRhythmProjector(ProjectorConfig(build_render_plan=False, pause_selection_mode="simple"))
        planner = RhythmPlannerOutputs(
            speech_budget_win=torch.tensor([[3.0], [3.0]], dtype=torch.float32),
            pause_budget_win=torch.tensor([[1.0], [1.0]], dtype=torch.float32),
            dur_logratio_unit=torch.zeros((2, 4), dtype=torch.float32),
            pause_weight_unit=torch.tensor([[0.0, 0.0, 0.2, 0.8], [0.0, 0.0, 0.2, 0.8]], dtype=torch.float32),
            boundary_score_unit=torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
            trace_context=torch.zeros((2, 4, 2), dtype=torch.float32),
            source_boundary_cue=torch.zeros((2, 4), dtype=torch.float32),
            planned_commit_frontier=torch.tensor([10, 1], dtype=torch.long),
        )
        state = StreamingRhythmState(
            phase_ptr=torch.tensor([0.0, 0.0], dtype=torch.float32),
            clock_delta=torch.tensor([0.0, 0.0], dtype=torch.float32),
            commit_frontier=torch.tensor([2, 2], dtype=torch.long),
            previous_speech_exec=torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]], dtype=torch.float32),
            previous_pause_exec=torch.tensor([[0.2, 0.0, 0.0, 0.0], [0.2, 0.0, 0.0, 0.0]], dtype=torch.float32),
        )
        execution = projector(
            dur_anchor_src=torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0]], dtype=torch.float32),
            unit_mask=torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0]], dtype=torch.float32),
            speech_budget_win=planner.speech_budget_win,
            pause_budget_win=planner.pause_budget_win,
            dur_logratio_unit=planner.dur_logratio_unit,
            pause_weight_unit=planner.pause_weight_unit,
            boundary_score_unit=planner.boundary_score_unit,
            state=state,
            planner=planner,
            reuse_prefix=True,
            force_full_commit=False,
        )
        self.assertTrue(torch.equal(execution.commit_frontier, torch.tensor([3, 2], dtype=torch.long)))
        self.assertTrue(torch.equal(execution.next_state.commit_frontier, torch.tensor([3, 2], dtype=torch.long)))
        self.assertTrue(torch.equal(execution.planner.planned_commit_frontier, torch.tensor([3, 2], dtype=torch.long)))

    def test_local_rho_is_monotonic_inside_segment_when_available(self) -> None:
        projector = self._make_invariance_projector()
        execution = self._run_projector(
            projector,
            state=projector.init_state(batch_size=1, device=torch.device("cpu")),
            dur_anchor_src=[2.0, 1.0, 0.0, 0.0, 0.0],
            open_run_mask=[0, 0, 1, 1, 1],
            reuse_prefix=False,
        )
        local_rho = getattr(execution.planner, "local_rho_unit", None)
        segment_mask = getattr(execution.planner, "segment_mask_unit", None)
        if local_rho is None:
            local_rho = getattr(execution.next_state, "local_rho_unit", None)
        if local_rho is None or segment_mask is None:
            self.skipTest("local_rho_unit / segment_mask_unit not wired yet")
        local_rho = local_rho.float()
        segment_mask = segment_mask.float() > 0.5
        segment_values = local_rho[0, segment_mask[0]]
        if segment_values.numel() > 1:
            self.assertTrue(torch.all(segment_values[1:] >= segment_values[:-1] - 1.0e-6))
        self.assertTrue(
            torch.allclose(
                local_rho[0, ~segment_mask[0]],
                torch.zeros_like(local_rho[0, ~segment_mask[0]]),
            )
        )
        intra_phrase_alpha = getattr(execution.planner, "intra_phrase_alpha", None)
        if intra_phrase_alpha is not None:
            self.assertGreaterEqual(float(intra_phrase_alpha[0, 0].item()), 0.0)
            self.assertLessEqual(float(intra_phrase_alpha[0, 0].item()), 1.0)

    def test_forward_samples_local_trace_context_from_selected_phrase_trace(self) -> None:
        projector = StreamingRhythmProjector(
            ProjectorConfig(
                min_speech_frames=0.0,
                max_speech_expand=10.0,
                pause_selection_mode="simple",
                build_render_plan=False,
            )
        )
        planner = RhythmPlannerOutputs(
            speech_budget_win=torch.tensor([[3.0]], dtype=torch.float32),
            pause_budget_win=torch.tensor([[1.0]], dtype=torch.float32),
            dur_logratio_unit=torch.zeros((1, 4), dtype=torch.float32),
            pause_weight_unit=torch.tensor([[0.0, 0.0, 0.2, 0.8]], dtype=torch.float32),
            boundary_score_unit=torch.tensor([[0.0, 0.2, 0.8, 1.0]], dtype=torch.float32),
            trace_context=torch.zeros((1, 4, 5), dtype=torch.float32),
            source_boundary_cue=torch.zeros((1, 4), dtype=torch.float32),
            phrase_speech_budget_win=torch.tensor([[3.0]], dtype=torch.float32),
            phrase_pause_budget_win=torch.tensor([[1.0]], dtype=torch.float32),
            segment_mask_unit=torch.tensor([[0.0, 1.0, 1.0, 0.0]], dtype=torch.float32),
            pause_segment_mask_unit=torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=torch.float32),
            ref_phrase_trace=torch.tensor(
                [[
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.2, 0.2, 0.2, 0.2, 0.2],
                    [0.6, 0.6, 0.6, 0.6, 0.6],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ]],
                dtype=torch.float32,
            ),
        )
        execution = projector(
            dur_anchor_src=torch.ones((1, 4), dtype=torch.float32),
            unit_mask=torch.ones((1, 4), dtype=torch.float32),
            speech_budget_win=planner.speech_budget_win,
            pause_budget_win=planner.pause_budget_win,
            dur_logratio_unit=planner.dur_logratio_unit,
            pause_weight_unit=planner.pause_weight_unit,
            boundary_score_unit=planner.boundary_score_unit,
            state=projector.init_state(batch_size=1, device=torch.device("cpu")),
            planner=planner,
            reuse_prefix=False,
            force_full_commit=True,
        )
        local_ctx = execution.planner.local_trace_ctx_unit
        self.assertIsNotNone(local_ctx)
        assert local_ctx is not None
        self.assertTrue(torch.allclose(local_ctx[0, 0], torch.zeros_like(local_ctx[0, 0])))
        self.assertTrue(torch.allclose(local_ctx[0, 3], torch.zeros_like(local_ctx[0, 3])))
        self.assertGreater(float(local_ctx[0, 1].mean().item()), 0.0)
        self.assertGreaterEqual(float(local_ctx[0, 2].mean().item()), float(local_ctx[0, 1].mean().item()))

    def test_trace_coverage_gate_suppresses_local_on_short_prefix(self) -> None:
        module = build_streaming_rhythm_module_from_hparams(
            {
                "hidden_size": 16,
                "rhythm_hidden_size": 16,
                "rhythm_trace_bins": 8,
                "rhythm_trace_dim": 5,
                "rhythm_emit_reference_sidecar": False,
                "rhythm_trace_reliability_enable": True,
                "rhythm_trace_cold_start_min_visible_units": 3,
                "rhythm_trace_cold_start_full_visible_units": 8,
                "rhythm_trace_active_tail_only": False,
            }
        )
        phase_ptr = torch.tensor([0.1], dtype=torch.float32)
        bundle = module._build_trace_reliability(
            phase_ptr=phase_ptr,
            phase_gap_runtime=torch.tensor([0.0], dtype=torch.float32),
            phase_gap_anchor=torch.tensor([0.0], dtype=torch.float32),
            horizon=0.35,
            visible_units=torch.tensor([1.0], dtype=torch.float32),
            cold_start_min_visible_units=3,
            cold_start_full_visible_units=8,
        )
        self.assertLess(bundle.local_gate[0].item(), 0.2)

    def test_phrase_budget_views_ignore_far_future_suffix_outside_active_segment(self) -> None:
        module = build_streaming_rhythm_module_from_hparams(
            {
                "hidden_size": 16,
                "rhythm_hidden_size": 16,
                "rhythm_trace_bins": 8,
            }
        )
        state = StreamingRhythmState(
            phase_ptr=torch.tensor([0.0], dtype=torch.float32),
            clock_delta=torch.tensor([0.0], dtype=torch.float32),
            commit_frontier=torch.tensor([0], dtype=torch.long),
        )
        short_planner = RhythmPlannerOutputs(
            speech_budget_win=torch.tensor([[10.0]], dtype=torch.float32),
            pause_budget_win=torch.tensor([[4.0]], dtype=torch.float32),
            dur_logratio_unit=torch.zeros((1, 3), dtype=torch.float32),
            pause_weight_unit=torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32),
            boundary_score_unit=torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32),
            trace_context=torch.zeros((1, 3, 2), dtype=torch.float32),
            pause_support_prob_unit=torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32),
        )
        extended_planner = RhythmPlannerOutputs(
            speech_budget_win=torch.tensor([[10.0]], dtype=torch.float32),
            pause_budget_win=torch.tensor([[4.0]], dtype=torch.float32),
            dur_logratio_unit=torch.zeros((1, 5), dtype=torch.float32),
            pause_weight_unit=torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            boundary_score_unit=torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            trace_context=torch.zeros((1, 5, 2), dtype=torch.float32),
            pause_support_prob_unit=torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        )
        short_budgets = module._build_phrase_budget_views(
            planner=short_planner,
            state=state,
            dur_anchor_src=torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float32),
            unit_mask=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
            segment_mask=torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32),
            pause_segment_mask=torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32),
        )
        extended_budgets = module._build_phrase_budget_views(
            planner=extended_planner,
            state=state,
            dur_anchor_src=torch.tensor([[2.0, 1.0, 0.0, 10.0, 10.0]], dtype=torch.float32),
            unit_mask=torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]], dtype=torch.float32),
            segment_mask=torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            pause_segment_mask=torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        )
        self.assertTrue(torch.allclose(short_budgets[0], extended_budgets[0], atol=1.0e-6))
        self.assertTrue(torch.allclose(short_budgets[1], extended_budgets[1], atol=1.0e-6))
        self.assertTrue(torch.allclose(short_budgets[0], torch.tensor([[3.0]], dtype=torch.float32)))

    def test_prefix_state_torch_numpy_parity(self) -> None:
        speech_exec = torch.tensor([[2.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
        pause_exec = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
        dur_anchor_src = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
        unit_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
        torch_clock, torch_backlog = build_prefix_state_from_exec_torch(
            speech_exec=speech_exec,
            pause_exec=pause_exec,
            dur_anchor_src=dur_anchor_src,
            unit_mask=unit_mask,
        )
        numpy_clock, numpy_backlog = build_prefix_state_from_exec_numpy(
            speech_exec=speech_exec[0].numpy(),
            pause_exec=pause_exec[0].numpy(),
            dur_anchor_src=dur_anchor_src[0].numpy(),
            unit_mask=unit_mask[0].numpy(),
        )
        self.assertTrue(torch.allclose(torch_clock[0], torch.from_numpy(numpy_clock), atol=1e-6))
        self.assertTrue(torch.allclose(torch_backlog[0], torch.from_numpy(numpy_backlog), atol=1e-6))

    def test_forward_keeps_raw_budget_separate_from_effective_budget(self) -> None:
        projector = StreamingRhythmProjector(
            ProjectorConfig(
                min_speech_frames=1.0,
                max_speech_expand=3.0,
                build_render_plan=False,
            )
        )
        planner = RhythmPlannerOutputs(
            speech_budget_win=torch.tensor([[0.2]], dtype=torch.float32, requires_grad=True),
            pause_budget_win=torch.tensor([[0.1]], dtype=torch.float32, requires_grad=True),
            dur_logratio_unit=torch.zeros((1, 3), dtype=torch.float32, requires_grad=True),
            pause_weight_unit=torch.full((1, 3), 1.0 / 3.0, dtype=torch.float32, requires_grad=True),
            boundary_score_unit=torch.zeros((1, 3), dtype=torch.float32),
            trace_context=torch.zeros((1, 3, 2), dtype=torch.float32),
            source_boundary_cue=torch.zeros((1, 3), dtype=torch.float32),
        )
        execution = projector(
            dur_anchor_src=torch.ones((1, 3), dtype=torch.float32),
            unit_mask=torch.ones((1, 3), dtype=torch.float32),
            speech_budget_win=planner.speech_budget_win,
            pause_budget_win=planner.pause_budget_win,
            dur_logratio_unit=planner.dur_logratio_unit,
            pause_weight_unit=planner.pause_weight_unit,
            boundary_score_unit=planner.boundary_score_unit,
            state=projector.init_state(batch_size=1, device=torch.device("cpu")),
            planner=planner,
            reuse_prefix=False,
            force_full_commit=True,
        )
        self.assertTrue(torch.allclose(execution.planner.raw_speech_budget_win, planner.speech_budget_win))
        self.assertTrue(torch.allclose(execution.planner.raw_pause_budget_win, planner.pause_budget_win))
        self.assertGreater(
            float(execution.planner.effective_speech_budget_win.item()),
            float(planner.speech_budget_win.item()),
        )
        self.assertGreaterEqual(float(execution.planner.feasible_total_budget_delta.item()), 0.0)

    def test_forward_preserves_explicit_raw_budget_views_from_planner(self) -> None:
        projector = StreamingRhythmProjector(
            ProjectorConfig(
                min_speech_frames=1.0,
                max_speech_expand=3.0,
                build_render_plan=False,
            )
        )
        planner = RhythmPlannerOutputs(
            speech_budget_win=torch.tensor([[0.2]], dtype=torch.float32),
            pause_budget_win=torch.tensor([[0.1]], dtype=torch.float32),
            dur_logratio_unit=torch.zeros((1, 3), dtype=torch.float32),
            pause_weight_unit=torch.full((1, 3), 1.0 / 3.0, dtype=torch.float32),
            boundary_score_unit=torch.zeros((1, 3), dtype=torch.float32),
            trace_context=torch.zeros((1, 3, 2), dtype=torch.float32),
            source_boundary_cue=torch.zeros((1, 3), dtype=torch.float32),
        )
        planner.raw_speech_budget_win = torch.tensor([[0.05]], dtype=torch.float32)
        planner.raw_pause_budget_win = torch.tensor([[0.25]], dtype=torch.float32)
        execution = projector(
            dur_anchor_src=torch.ones((1, 3), dtype=torch.float32),
            unit_mask=torch.ones((1, 3), dtype=torch.float32),
            speech_budget_win=planner.speech_budget_win,
            pause_budget_win=planner.pause_budget_win,
            dur_logratio_unit=planner.dur_logratio_unit,
            pause_weight_unit=planner.pause_weight_unit,
            boundary_score_unit=planner.boundary_score_unit,
            state=projector.init_state(batch_size=1, device=torch.device("cpu")),
            planner=planner,
            reuse_prefix=False,
            force_full_commit=True,
        )
        self.assertTrue(torch.allclose(execution.planner.raw_speech_budget_win, planner.raw_speech_budget_win))
        self.assertTrue(torch.allclose(execution.planner.raw_pause_budget_win, planner.raw_pause_budget_win))

    def test_forward_can_enable_soft_pause_selection_even_with_force_full_commit(self) -> None:
        projector = StreamingRhythmProjector(
            ProjectorConfig(
                pause_selection_mode="sparse",
                pause_train_soft=True,
                build_render_plan=False,
            )
        )
        projector.train()
        planner = RhythmPlannerOutputs(
            speech_budget_win=torch.tensor([[4.0]], dtype=torch.float32),
            pause_budget_win=torch.tensor([[2.0]], dtype=torch.float32),
            dur_logratio_unit=torch.zeros((1, 4), dtype=torch.float32),
            pause_weight_unit=torch.tensor([[0.45, 0.30, 0.15, 0.10]], dtype=torch.float32),
            boundary_score_unit=torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32),
            trace_context=torch.zeros((1, 4, 2), dtype=torch.float32),
            source_boundary_cue=torch.zeros((1, 4), dtype=torch.float32),
        )
        execution = projector(
            dur_anchor_src=torch.ones((1, 4), dtype=torch.float32),
            unit_mask=torch.ones((1, 4), dtype=torch.float32),
            speech_budget_win=planner.speech_budget_win,
            pause_budget_win=planner.pause_budget_win,
            dur_logratio_unit=planner.dur_logratio_unit,
            pause_weight_unit=planner.pause_weight_unit,
            boundary_score_unit=planner.boundary_score_unit,
            state=projector.init_state(batch_size=1, device=torch.device("cpu")),
            planner=planner,
            reuse_prefix=False,
            force_full_commit=True,
            soft_pause_selection_override=True,
            pause_topk_ratio_override=0.5,
        )
        self.assertTrue(torch.allclose(execution.planner.projector_force_full_commit, torch.tensor([[1.0]])))
        self.assertTrue(torch.allclose(execution.planner.pause_soft_selection_active, torch.tensor([[1.0]])))
        self.assertTrue(torch.allclose(execution.planner.pause_topk_ratio, torch.tensor([[0.5]])))

    def test_soft_pause_selection_restores_gradient_to_below_topk_candidates(self) -> None:
        kwargs = dict(
            boundary_score_unit=torch.zeros((1, 4), dtype=torch.float32),
            unit_mask=torch.ones((1, 4), dtype=torch.float32),
            pause_budget_win=torch.tensor([[1.0]], dtype=torch.float32),
            previous_pause_exec=None,
            commit_frontier=torch.tensor([0], dtype=torch.long),
            reuse_prefix=False,
            topk_ratio=0.5,
            pause_min_boundary_weight=0.10,
            pause_boundary_bias_weight=0.0,
            temperature=0.20,
        )
        hard_scores = torch.tensor([[0.60, 0.25, 0.10, 0.05]], dtype=torch.float32, requires_grad=True)
        hard_pause = _project_pause_impl(
            pause_weight_unit=hard_scores,
            soft_pause_selection=False,
            **kwargs,
        )
        hard_pause[:, 2].sum().backward()
        hard_grad = hard_scores.grad.detach().clone()

        soft_scores = torch.tensor([[0.60, 0.25, 0.10, 0.05]], dtype=torch.float32, requires_grad=True)
        soft_pause = _project_pause_impl(
            pause_weight_unit=soft_scores,
            soft_pause_selection=True,
            **kwargs,
        )
        soft_pause[:, 2].sum().backward()
        soft_grad = soft_scores.grad.detach().clone()

        self.assertTrue(torch.allclose(hard_pause[:, 2], torch.tensor([0.0], dtype=torch.float32), atol=1e-7))
        self.assertAlmostEqual(float(hard_grad[0, 2].item()), 0.0, places=7)
        self.assertGreater(float(soft_pause[:, 2].item()), 0.0)
        self.assertGreater(abs(float(soft_grad[0, 2].item())), 1.0e-7)

    def test_pause_projection_can_decouple_support_selection_from_allocation(self) -> None:
        pause = _project_pause_impl(
            pause_weight_unit=torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float32),
            pause_support_prob_unit=torch.tensor([[0.90, 0.80, 0.10, 0.05]], dtype=torch.float32),
            pause_allocation_weight_unit=torch.tensor([[0.10, 0.20, 0.60, 0.10]], dtype=torch.float32),
            boundary_score_unit=torch.zeros((1, 4), dtype=torch.float32),
            unit_mask=torch.ones((1, 4), dtype=torch.float32),
            pause_budget_win=torch.tensor([[1.0]], dtype=torch.float32),
            previous_pause_exec=None,
            commit_frontier=torch.tensor([0], dtype=torch.long),
            reuse_prefix=False,
            soft_pause_selection=False,
            topk_ratio=0.50,
            pause_min_boundary_weight=0.10,
            pause_boundary_bias_weight=0.0,
            temperature=0.20,
        )
        self.assertAlmostEqual(float(pause[:, 2].item()), 0.0, places=7)
        self.assertGreater(float(pause[:, 1].item()), float(pause[:, 0].item()))

    def test_split_head_candidate_surface_is_not_reweighted_by_boundary_gain(self) -> None:
        pause = _project_pause_impl(
            pause_weight_unit=torch.full((1, 4), 0.25, dtype=torch.float32),
            pause_support_prob_unit=torch.tensor([[0.90, 0.80, 0.10, 0.05]], dtype=torch.float32),
            pause_allocation_weight_unit=torch.tensor([[0.40, 0.60, 0.00, 0.00]], dtype=torch.float32),
            boundary_score_unit=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            unit_mask=torch.ones((1, 4), dtype=torch.float32),
            pause_budget_win=torch.tensor([[1.0]], dtype=torch.float32),
            previous_pause_exec=None,
            commit_frontier=torch.tensor([0], dtype=torch.long),
            reuse_prefix=False,
            soft_pause_selection=False,
            topk_ratio=0.50,
            pause_min_boundary_weight=0.10,
            pause_boundary_bias_weight=2.0,
            temperature=0.20,
        )
        self.assertGreater(float(pause[:, 1].item()), float(pause[:, 0].item()))

    def test_pause_projection_respects_explicit_allocation_mask(self) -> None:
        pause = _project_pause_simple_impl(
            pause_weight_unit=torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float32),
            boundary_score_unit=torch.tensor([[0.0, 0.2, 0.8, 0.0]], dtype=torch.float32),
            unit_mask=torch.ones((1, 4), dtype=torch.float32),
            allocation_mask=torch.tensor([[0.0, 1.0, 1.0, 0.0]], dtype=torch.float32),
            pause_budget_win=torch.tensor([[1.0]], dtype=torch.float32),
            previous_pause_exec=None,
            commit_frontier=torch.tensor([0], dtype=torch.long),
            reuse_prefix=False,
            pause_min_boundary_weight=0.10,
            pause_boundary_bias_weight=0.18,
        )
        self.assertAlmostEqual(float(pause[0, 0].item()), 0.0, places=7)
        self.assertAlmostEqual(float(pause[0, 3].item()), 0.0, places=7)
        self.assertGreater(float(pause[0, 2].item()), float(pause[0, 1].item()))

    def test_pause_support_feature_bundle_tracks_run_length_and_breath_debt(self) -> None:
        bundle = build_pause_support_feature_bundle(
            dur_anchor_src=torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32),
            unit_mask=torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32),
            boundary_score_unit=torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32),
            reset_threshold=0.5,
        )
        self.assertTrue(torch.allclose(bundle.run_length_unit, torch.tensor([[0.25, 0.50, 0.25, 0.50]])))
        self.assertTrue(torch.allclose(bundle.reset_mask, torch.tensor([[0.0, 1.0, 0.0, 0.0]])))
        self.assertGreater(float(bundle.breath_debt_unit[0, 1].item()), float(bundle.breath_debt_unit[0, 0].item()))
        self.assertGreater(float(bundle.breath_debt_unit[0, 3].item()), float(bundle.breath_debt_unit[0, 2].item()))

    def test_offline_teacher_can_emit_support_allocation_and_breath_features(self) -> None:
        teacher = OfflineRhythmTeacherPlanner(
            hidden_size=8,
            stats_dim=2,
            trace_dim=2,
            config=OfflineTeacherConfig(
                pause_support_split_enable=True,
                pause_breath_features_enable=True,
            ),
        )
        planner, _ = teacher(
            unit_states=torch.zeros((1, 4, 8), dtype=torch.float32),
            dur_anchor_src=torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32),
            unit_mask=torch.ones((1, 4), dtype=torch.float32),
            ref_conditioning={"planner_ref_stats": torch.tensor([[1.0, 0.25]], dtype=torch.float32)},
            planner_trace_context=torch.zeros((1, 4, 2), dtype=torch.float32),
            full_trace_context=torch.zeros((1, 4, 2), dtype=torch.float32),
            source_boundary_cue=torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32),
        )
        self.assertIsNotNone(planner.pause_support_prob_unit)
        self.assertIsNotNone(planner.pause_allocation_weight_unit)
        self.assertIsNotNone(planner.pause_run_length_unit)
        self.assertIsNotNone(planner.pause_breath_debt_unit)
        self.assertTrue(torch.allclose(planner.pause_shape_unit, planner.pause_allocation_weight_unit))
        self.assertEqual(tuple(planner.pause_support_prob_unit.shape), (1, 4))


if __name__ == "__main__":
    unittest.main()
