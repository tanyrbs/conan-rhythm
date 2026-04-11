from __future__ import annotations

import math

import torch
import torch.nn as nn

from .contracts import DurationExecution, DurationRuntimeState
from .frame_plan import build_frame_plan_from_execution


class StreamingDurationProjector(nn.Module):
    def __init__(
        self,
        *,
        prefix_budget_pos: int = 24,
        prefix_budget_neg: int = 24,
    ) -> None:
        super().__init__()
        self.prefix_budget_pos = int(max(0, prefix_budget_pos))
        self.prefix_budget_neg = int(max(0, prefix_budget_neg))

    def init_state(self, *, batch_size: int, device: torch.device) -> DurationRuntimeState:
        zeros = torch.zeros((batch_size, 1), device=device)
        return DurationRuntimeState(
            committed_units=torch.zeros((batch_size,), dtype=torch.long, device=device),
            rounding_residual=zeros.clone(),
            prefix_unit_offset=zeros.clone(),
            cached_duration_exec=None,
            local_rate_ema=None,
            since_last_boundary=zeros.clone(),
        )

    @staticmethod
    def _resolve_state(
        *,
        state: DurationRuntimeState | None,
        batch_size: int,
        device: torch.device,
        init_state,
    ) -> DurationRuntimeState:
        if state is not None:
            return state
        return init_state(batch_size=batch_size, device=device)

    @staticmethod
    def _validate_prefix_commit_mask(
        *,
        unit_mask: torch.Tensor,
        commit_mask: torch.Tensor,
    ) -> None:
        if unit_mask.numel() <= 0:
            return
        steps = torch.arange(unit_mask.size(1), device=unit_mask.device)[None, :]
        visible_len = unit_mask.float().sum(dim=1).long()
        committed_len = commit_mask.float().sum(dim=1).long()
        expected = (steps < committed_len[:, None]) & (steps < visible_len[:, None])
        actual = commit_mask > 0.5
        if not torch.equal(actual, expected):
            raise ValueError("Duration V3 commit mask must form a contiguous visible prefix.")

    @classmethod
    def _build_commit_mask(
        cls,
        *,
        unit_mask: torch.Tensor,
        sealed_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        commit_mask = unit_mask.float() if sealed_mask is None else unit_mask.float() * sealed_mask.float()
        cls._validate_prefix_commit_mask(unit_mask=unit_mask.float(), commit_mask=commit_mask)
        return commit_mask

    @staticmethod
    def build_frame_plan(
        *,
        source_duration_obs: torch.Tensor,
        projected_duration_exec: torch.Tensor,
        commit_mask: torch.Tensor,
    ):
        return build_frame_plan_from_execution(
            dur_anchor_src=source_duration_obs.float(),
            speech_exec=projected_duration_exec.float(),
            pause_exec=torch.zeros_like(projected_duration_exec.float()),
            unit_mask=commit_mask.float(),
        )

    @staticmethod
    def _project_duration_prefix(
        *,
        unit_duration_exec: torch.Tensor,
        source_duration_obs: torch.Tensor,
        commit_mask: torch.Tensor,
        speech_commit_mask: torch.Tensor,
        residual_prev: torch.Tensor,
        prefix_unit_offset_prev: torch.Tensor,
        committed_units_prev: torch.Tensor | None,
        cached_duration_exec_prev: torch.Tensor | None,
        budget_pos: int,
        budget_neg: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_units = unit_duration_exec.shape
        projected = unit_duration_exec.new_zeros(unit_duration_exec.shape)
        residual_next = residual_prev.float().reshape(batch_size).clone()
        prefix_offset_next = prefix_unit_offset_prev.float().reshape(batch_size).clone()
        if isinstance(committed_units_prev, torch.Tensor):
            committed_units_prev = committed_units_prev.long().reshape(batch_size)
        else:
            committed_units_prev = torch.zeros((batch_size,), dtype=torch.long, device=unit_duration_exec.device)
        cached_prev = None
        if isinstance(cached_duration_exec_prev, torch.Tensor):
            cached_prev = cached_duration_exec_prev.to(device=unit_duration_exec.device, dtype=unit_duration_exec.dtype)
        for batch_idx in range(batch_size):
            start_unit = 0
            if cached_prev is not None and cached_prev.size(0) > batch_idx:
                prefix = min(int(committed_units_prev[batch_idx].item()), int(cached_prev.size(1)), num_units)
                if prefix > 0:
                    projected[batch_idx, :prefix] = cached_prev[batch_idx, :prefix]
                    start_unit = prefix
            carry = float(residual_next[batch_idx].item())
            prefix_offset = int(round(float(prefix_offset_next[batch_idx].item())))
            for unit_idx in range(start_unit, num_units):
                if float(commit_mask[batch_idx, unit_idx].item()) <= 0.5:
                    continue
                source_count = int(max(0.0, round(float(source_duration_obs[batch_idx, unit_idx].item()))))
                if float(speech_commit_mask[batch_idx, unit_idx].item()) <= 0.5:
                    projected[batch_idx, unit_idx] = float(source_count)
                    continue
                total = max(0.0, float(unit_duration_exec[batch_idx, unit_idx].item()) + carry)
                frames = float(math.floor(total))
                frames = max(1.0, frames)
                anchor = max(1, source_count)
                lower = max(1, int(math.ceil(float(anchor - (budget_neg + prefix_offset)))))
                upper = max(lower, int(math.floor(float(anchor + (budget_pos - prefix_offset)))))
                frames = float(min(max(int(frames), lower), upper))
                projected[batch_idx, unit_idx] = frames
                prefix_offset += int(frames) - anchor
                carry = total - frames
            residual_next[batch_idx] = carry
            prefix_offset_next[batch_idx] = float(prefix_offset)
        return projected, residual_next.reshape(batch_size, 1), prefix_offset_next.reshape(batch_size, 1)

    @staticmethod
    def _materialize_projected_duration(
        *,
        unit_duration_exec: torch.Tensor,
        projected_duration_exec: torch.Tensor,
        commit_mask: torch.Tensor,
    ) -> torch.Tensor:
        projected_prefix = projected_duration_exec * commit_mask.float()
        return unit_duration_exec + (projected_prefix - unit_duration_exec).detach()

    @staticmethod
    def _build_next_state(
        *,
        commit_mask: torch.Tensor,
        residual_next: torch.Tensor,
        prefix_unit_offset_next: torch.Tensor,
        cached_duration_exec: torch.Tensor,
        state_prev: DurationRuntimeState,
    ) -> DurationRuntimeState:
        return DurationRuntimeState(
            committed_units=commit_mask.sum(dim=1).long(),
            rounding_residual=residual_next.detach(),
            prefix_unit_offset=prefix_unit_offset_next.detach(),
            cached_duration_exec=cached_duration_exec.detach(),
            local_rate_ema=None if state_prev.local_rate_ema is None else state_prev.local_rate_ema.detach(),
            since_last_boundary=None if state_prev.since_last_boundary is None else state_prev.since_last_boundary.detach(),
        )

    def finalize_execution(
        self,
        *,
        unit_logstretch: torch.Tensor,
        unit_duration_exec: torch.Tensor,
        basis_activation: torch.Tensor,
        source_duration_obs: torch.Tensor,
        unit_mask: torch.Tensor,
        sealed_mask: torch.Tensor | None,
        speech_commit_mask: torch.Tensor,
        state: DurationRuntimeState | None,
        progress_response: torch.Tensor | None = None,
        detector_response: torch.Tensor | None = None,
        local_response: torch.Tensor | None = None,
        role_attn_unit: torch.Tensor | None = None,
        role_value_unit: torch.Tensor | None = None,
        role_var_unit: torch.Tensor | None = None,
        role_conf_unit: torch.Tensor | None = None,
        unit_logstretch_raw: torch.Tensor | None = None,
        unit_duration_raw: torch.Tensor | None = None,
        global_bias_scalar: torch.Tensor | None = None,
        global_shift_analytic: torch.Tensor | None = None,
        local_residual: torch.Tensor | None = None,
        source_rate_seq: torch.Tensor | None = None,
    ) -> DurationExecution:
        batch_size = unit_duration_exec.size(0)
        device = unit_duration_exec.device
        state = self._resolve_state(
            state=state,
            batch_size=batch_size,
            device=device,
            init_state=self.init_state,
        )
        commit_mask = self._build_commit_mask(unit_mask=unit_mask, sealed_mask=sealed_mask)
        projected_duration_exec, residual_next, prefix_unit_offset_next = self._project_duration_prefix(
            unit_duration_exec=unit_duration_exec,
            source_duration_obs=source_duration_obs,
            commit_mask=commit_mask,
            speech_commit_mask=speech_commit_mask.float(),
            residual_prev=state.rounding_residual,
            prefix_unit_offset_prev=state.prefix_unit_offset,
            committed_units_prev=state.committed_units,
            cached_duration_exec_prev=state.cached_duration_exec,
            budget_pos=self.prefix_budget_pos,
            budget_neg=self.prefix_budget_neg,
        )
        materialized_duration_exec = self._materialize_projected_duration(
            unit_duration_exec=unit_duration_exec,
            projected_duration_exec=projected_duration_exec,
            commit_mask=commit_mask,
        )
        cached_duration_exec = projected_duration_exec * commit_mask.float()
        next_state = self._build_next_state(
            commit_mask=commit_mask,
            residual_next=residual_next,
            prefix_unit_offset_next=prefix_unit_offset_next,
            cached_duration_exec=cached_duration_exec,
            state_prev=state,
        )
        frame_plan = self.build_frame_plan(
            source_duration_obs=source_duration_obs,
            projected_duration_exec=projected_duration_exec,
            commit_mask=commit_mask,
        )
        return DurationExecution(
            unit_logstretch=unit_logstretch,
            unit_duration_exec=materialized_duration_exec,
            basis_activation=basis_activation,
            commit_mask=commit_mask,
            next_state=next_state,
            progress_response=progress_response,
            detector_response=detector_response,
            local_response=local_response,
            role_attn_unit=role_attn_unit,
            role_value_unit=role_value_unit,
            role_var_unit=role_var_unit,
            role_conf_unit=role_conf_unit,
            unit_logstretch_raw=unit_logstretch_raw,
            unit_duration_raw=unit_duration_raw,
            frame_plan=frame_plan,
            global_bias_scalar=global_bias_scalar,
            global_shift_analytic=global_shift_analytic,
            local_residual=local_residual,
            source_rate_seq=source_rate_seq,
            prefix_unit_offset=prefix_unit_offset_next,
        )
