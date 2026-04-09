from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from modules.Conan.rhythm.frame_plan import build_frame_plan_from_execution

from .contracts import DurationExecution, DurationRuntimeState


@dataclass(frozen=True)
class PrefixDriftConfig:
    gain: float = 0.25
    clip: float = 0.05


class StreamingDurationProjector(nn.Module):
    def __init__(self, *, drift_gain: float = 0.25, drift_clip: float = 0.05) -> None:
        super().__init__()
        self.drift = PrefixDriftConfig(gain=float(drift_gain), clip=float(drift_clip))

    def init_state(self, *, batch_size: int, device: torch.device) -> DurationRuntimeState:
        zeros = torch.zeros((batch_size, 1), device=device)
        return DurationRuntimeState(
            committed_units=torch.zeros((batch_size,), dtype=torch.long, device=device),
            cumulative_pred_frames=zeros,
            cumulative_tgt_frames=zeros.clone(),
            cached_duration_exec=None,
        )

    def compute_drift_correction(self, state: DurationRuntimeState | None) -> torch.Tensor:
        if state is None:
            return torch.zeros((1, 1))
        if state.cumulative_tgt_frames is None:
            return state.cumulative_pred_frames.new_zeros(state.cumulative_pred_frames.shape)
        delta = state.cumulative_tgt_frames - state.cumulative_pred_frames
        return (delta * float(self.drift.gain)).clamp(-float(self.drift.clip), float(self.drift.clip))

    def build_frame_plan(
        self,
        *,
        source_runlen_src: torch.Tensor,
        unit_duration_exec: torch.Tensor,
        unit_mask: torch.Tensor,
    ):
        zero_pause = torch.zeros_like(unit_duration_exec)
        return build_frame_plan_from_execution(
            dur_anchor_src=source_runlen_src,
            speech_exec=unit_duration_exec,
            pause_exec=zero_pause,
            unit_mask=unit_mask,
        )

    def finalize_execution(
        self,
        *,
        unit_logstretch: torch.Tensor,
        unit_duration_exec: torch.Tensor,
        role_attention: torch.Tensor,
        anti_pos_logits: torch.Tensor | None,
        prompt_reconstruction: torch.Tensor | None,
        prompt_rel_stretch: torch.Tensor | None,
        prompt_mask: torch.Tensor | None,
        source_runlen_src: torch.Tensor,
        unit_mask: torch.Tensor,
        sealed_mask: torch.Tensor | None,
        state: DurationRuntimeState | None,
    ) -> DurationExecution:
        batch_size = unit_duration_exec.size(0)
        device = unit_duration_exec.device
        if state is None:
            state = self.init_state(batch_size=batch_size, device=device)
        if sealed_mask is None:
            committed_units = unit_mask.sum(dim=1).long()
        else:
            committed_units = (sealed_mask.float() * unit_mask.float()).sum(dim=1).long()
        cumulative_pred_frames = []
        cumulative_tgt_frames = []
        for batch_idx in range(batch_size):
            frontier = int(committed_units[batch_idx].item())
            cumulative_pred_frames.append(unit_duration_exec[batch_idx, :frontier].sum())
            cumulative_tgt_frames.append((source_runlen_src[batch_idx, :frontier].float()).sum())
        cumulative_pred_frames = torch.stack(cumulative_pred_frames, dim=0).reshape(batch_size, 1)
        cumulative_tgt_frames = torch.stack(cumulative_tgt_frames, dim=0).reshape(batch_size, 1)
        next_state = DurationRuntimeState(
            committed_units=committed_units,
            cumulative_pred_frames=cumulative_pred_frames.detach(),
            cumulative_tgt_frames=cumulative_tgt_frames.detach(),
            cached_duration_exec=unit_duration_exec.detach(),
        )
        frame_plan = self.build_frame_plan(
            source_runlen_src=source_runlen_src,
            unit_duration_exec=unit_duration_exec,
            unit_mask=unit_mask,
        )
        return DurationExecution(
            unit_logstretch=unit_logstretch,
            unit_duration_exec=unit_duration_exec,
            role_attention=role_attention,
            next_state=next_state,
            frame_plan=frame_plan,
            anti_pos_logits=anti_pos_logits,
            prompt_reconstruction=prompt_reconstruction,
            prompt_rel_stretch=prompt_rel_stretch,
            prompt_mask=prompt_mask,
        )
