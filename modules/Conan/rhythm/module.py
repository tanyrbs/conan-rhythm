from __future__ import annotations

import torch
import torch.nn as nn

from .controller import UnitRedistributionHead, WindowBudgetController
from .contracts import RhythmPlannerOutputs, StreamingRhythmState
from .projector import ProjectorConfig, StreamingRhythmProjector
from .reference_encoder import ReferenceRhythmEncoder


class StreamingRhythmModule(nn.Module):
    def __init__(
        self,
        *,
        num_units: int = 128,
        hidden_size: int = 256,
        trace_bins: int = 24,
        stats_dim: int = 6,
        trace_dim: int = 5,
        trace_horizon: float = 0.35,
        max_total_logratio: float = 0.8,
        max_unit_logratio: float = 0.6,
        pause_share_max: float = 0.45,
        projector_config: ProjectorConfig | None = None,
    ) -> None:
        super().__init__()
        self.unit_embedding = nn.Embedding(num_units, hidden_size)
        self.reference_encoder = ReferenceRhythmEncoder(
            trace_bins=trace_bins,
            trace_horizon=trace_horizon,
        )
        self.budget_controller = WindowBudgetController(
            hidden_size=hidden_size,
            stats_dim=stats_dim,
            trace_dim=trace_dim,
            max_total_logratio=max_total_logratio,
            pause_share_max=pause_share_max,
        )
        self.redistribution_head = UnitRedistributionHead(
            hidden_size=hidden_size,
            trace_dim=trace_dim,
            max_unit_logratio=max_unit_logratio,
        )
        self.projector = StreamingRhythmProjector(projector_config)

    def init_state(self, batch_size: int, device: torch.device) -> StreamingRhythmState:
        return self.projector.init_state(batch_size=batch_size, device=device)

    def encode_reference(self, ref_mel: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.reference_encoder(ref_mel)

    def forward(
        self,
        *,
        content_units: torch.Tensor,
        dur_anchor_src: torch.Tensor,
        ref_rhythm_stats: torch.Tensor | None = None,
        ref_rhythm_trace: torch.Tensor | None = None,
        ref_mel: torch.Tensor | None = None,
        unit_mask: torch.Tensor | None = None,
        open_run_mask: torch.Tensor | None = None,
        state: StreamingRhythmState | None = None,
    ):
        if ref_rhythm_stats is None or ref_rhythm_trace is None:
            if ref_mel is None:
                raise ValueError('Need either (ref_rhythm_stats, ref_rhythm_trace) or ref_mel.')
            ref_conditioning = self.encode_reference(ref_mel)
            ref_rhythm_stats = ref_conditioning['ref_rhythm_stats']
            ref_rhythm_trace = ref_conditioning['ref_rhythm_trace']
        if unit_mask is None:
            unit_mask = dur_anchor_src.gt(0).float()
        if state is None:
            state = self.init_state(batch_size=content_units.size(0), device=content_units.device)

        unit_embed = self.unit_embedding(content_units.long().clamp_min(0))
        trace_context = self.reference_encoder.sample_trace_window(
            ref_rhythm_trace=ref_rhythm_trace,
            phase_ptr=state.phase_ptr,
            window_size=content_units.size(1),
        )
        budget_outputs = self.budget_controller(
            unit_states=unit_embed,
            dur_anchor_src=dur_anchor_src,
            unit_mask=unit_mask,
            ref_rhythm_stats=ref_rhythm_stats,
            trace_context=trace_context,
            phase_ptr=state.phase_ptr,
            backlog=state.backlog,
            clock_delta=state.clock_delta,
        )
        redistribution_outputs = self.redistribution_head(
            hidden=budget_outputs['hidden'],
            trace_context=trace_context,
            unit_mask=unit_mask,
        )
        planner = RhythmPlannerOutputs(
            speech_budget_win=budget_outputs['speech_budget_win'],
            pause_budget_win=budget_outputs['pause_budget_win'],
            dur_logratio_unit=redistribution_outputs['dur_logratio_unit'],
            pause_weight_unit=redistribution_outputs['pause_weight_unit'],
            total_budget_win=budget_outputs['total_budget_win'],
            pause_share_win=budget_outputs['pause_share_win'],
            anchor_gate=budget_outputs['anchor_gate'],
            boundary_latent=redistribution_outputs['boundary_latent'],
            trace_context=trace_context,
        )
        return self.projector(
            dur_anchor_src=dur_anchor_src,
            unit_mask=unit_mask,
            speech_budget_win=planner.speech_budget_win,
            pause_budget_win=planner.pause_budget_win,
            dur_logratio_unit=planner.dur_logratio_unit,
            pause_weight_unit=planner.pause_weight_unit,
            boundary_latent=planner.boundary_latent,
            state=state,
            open_run_mask=open_run_mask,
            planner=planner,
        )
