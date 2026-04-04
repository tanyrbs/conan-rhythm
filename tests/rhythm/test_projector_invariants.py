from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.Conan.rhythm.contracts import RhythmPlannerOutputs, StreamingRhythmState
from modules.Conan.rhythm.controller import resolve_budget_views_from_total_and_pause_share
from modules.Conan.rhythm.feasibility import lift_projector_budgets_to_feasible_region
from modules.Conan.rhythm.prefix_state import (
    build_prefix_state_from_exec_numpy,
    build_prefix_state_from_exec_torch,
)
from modules.Conan.rhythm.projector import ProjectorConfig, StreamingRhythmProjector
from modules.Conan.rhythm.scheduler import MonotonicRhythmScheduler


class ProjectorInvariantTests(unittest.TestCase):
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
            effective_duration_exec=torch.tensor([[1.0, 1.0, 1.0, 0.0]], dtype=torch.float32),
            commit_frontier=torch.tensor([3], dtype=torch.long),
            speech_duration_exec=torch.tensor([[1.0, 1.0, 1.0, 0.0]], dtype=torch.float32),
            pause_after_exec=torch.zeros((1, 4), dtype=torch.float32),
        )
        self.assertTrue(torch.allclose(next_state.phase_ptr, torch.tensor([0.60], dtype=torch.float32)))
        self.assertTrue(torch.allclose(next_state.phase_progress_ratio, torch.tensor([0.60], dtype=torch.float32)))

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


if __name__ == "__main__":
    unittest.main()
