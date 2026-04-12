from __future__ import annotations

import math

import numpy as np
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
        dynamic_budget_ratio: float = 0.0,
        min_prefix_budget: int = 0,
        max_prefix_budget: int = 0,
        boundary_carry_decay: float = 0.25,
        boundary_reset_thresh: float = 0.5,
    ) -> None:
        super().__init__()
        self.prefix_budget_pos = int(max(0, prefix_budget_pos))
        self.prefix_budget_neg = int(max(0, prefix_budget_neg))
        self.dynamic_budget_ratio = float(max(0.0, dynamic_budget_ratio))
        self.min_prefix_budget = int(max(0, min_prefix_budget))
        self.max_prefix_budget = int(max(0, max_prefix_budget))
        self.boundary_carry_decay = float(max(0.0, min(1.0, boundary_carry_decay)))
        self.boundary_reset_thresh = float(max(0.0, min(1.0, boundary_reset_thresh)))

    def init_state(self, *, batch_size: int, device: torch.device) -> DurationRuntimeState:
        zeros = torch.zeros((batch_size, 1), device=device)
        return DurationRuntimeState(
            committed_units=torch.zeros((batch_size,), dtype=torch.long, device=device),
            rounding_residual=zeros.clone(),
            prefix_unit_offset=zeros.clone(),
            cached_duration_exec=None,
            local_rate_ema=None,
            since_last_boundary=zeros.clone(),
            frontend_state=None,
            consumed_content_steps=torch.zeros((batch_size, 1), dtype=torch.long, device=device),
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
    def _resolve_prefix_budget(
        *,
        source_duration_obs: torch.Tensor,
        committed_len: int,
        static_budget: int,
        dynamic_budget_ratio: float,
        min_prefix_budget: int,
        max_prefix_budget: int,
    ) -> int:
        if committed_len <= 0 or float(dynamic_budget_ratio) <= 0.0:
            return int(max(0, static_budget))
        prefix_source_total = float(source_duration_obs[:committed_len].float().sum().item())
        dynamic_budget = int(round(prefix_source_total * float(dynamic_budget_ratio)))
        dynamic_budget = max(int(max(0, min_prefix_budget)), dynamic_budget)
        if int(max_prefix_budget) > 0:
            dynamic_budget = min(dynamic_budget, int(max_prefix_budget))
        return int(max(0, dynamic_budget))

    @staticmethod
    def _project_duration_prefix(
        *,
        unit_duration_exec: torch.Tensor,
        source_duration_obs: torch.Tensor,
        commit_mask: torch.Tensor,
        speech_commit_mask: torch.Tensor,
        coarse_only_commit_mask: torch.Tensor | None = None,
        source_boundary_cue: torch.Tensor | None = None,
        phrase_final_mask: torch.Tensor | None = None,
        residual_prev: torch.Tensor,
        prefix_unit_offset_prev: torch.Tensor,
        committed_units_prev: torch.Tensor | None,
        cached_duration_exec_prev: torch.Tensor | None,
        budget_pos: int,
        budget_neg: int,
        dynamic_budget_ratio: float = 0.0,
        min_prefix_budget: int = 0,
        max_prefix_budget: int = 0,
        boundary_carry_decay: float = 0.25,
        boundary_reset_thresh: float = 0.5,
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
            committed_len = int(commit_mask[batch_idx].float().sum().item())
            if committed_len <= 0:
                continue
            row_budget_pos = StreamingDurationProjector._resolve_prefix_budget(
                source_duration_obs=source_duration_obs[batch_idx],
                committed_len=committed_len,
                static_budget=budget_pos,
                dynamic_budget_ratio=dynamic_budget_ratio,
                min_prefix_budget=min_prefix_budget,
                max_prefix_budget=max_prefix_budget,
            )
            row_budget_neg = StreamingDurationProjector._resolve_prefix_budget(
                source_duration_obs=source_duration_obs[batch_idx],
                committed_len=committed_len,
                static_budget=budget_neg,
                dynamic_budget_ratio=dynamic_budget_ratio,
                min_prefix_budget=min_prefix_budget,
                max_prefix_budget=max_prefix_budget,
            )
            start_unit = 0
            if cached_prev is not None and cached_prev.size(0) > batch_idx:
                prefix = min(int(committed_units_prev[batch_idx].item()), int(cached_prev.size(1)), committed_len)
                if prefix > 0:
                    projected[batch_idx, :prefix] = cached_prev[batch_idx, :prefix]
                    start_unit = prefix
            carry = float(residual_next[batch_idx].item())
            prefix_offset = int(round(float(prefix_offset_next[batch_idx].item())))
            if start_unit >= committed_len:
                residual_next[batch_idx] = carry
                prefix_offset_next[batch_idx] = float(prefix_offset)
                continue
            row_exec, row_source_rounded, row_speech, row_coarse, row_boundary, row_phrase_final = (
                StreamingDurationProjector._extract_projection_row_numpy(
                    batch_idx=batch_idx,
                    start_unit=start_unit,
                    committed_len=committed_len,
                    unit_duration_exec=unit_duration_exec,
                    source_duration_obs=source_duration_obs,
                    speech_commit_mask=speech_commit_mask,
                    coarse_only_commit_mask=coarse_only_commit_mask,
                    source_boundary_cue=source_boundary_cue,
                    phrase_final_mask=phrase_final_mask,
                )
            )
            row_projected = np.empty_like(row_exec, dtype=np.float32)
            for offset in range(int(row_exec.shape[0])):
                exec_value = float(row_exec[offset])
                source_count = int(max(0, int(row_source_rounded[offset])))
                is_speech = bool(row_speech[offset])
                is_coarse_only = bool(row_coarse is not None and row_coarse[offset])
                if (not is_speech) and (not is_coarse_only):
                    row_projected[offset] = np.float32(source_count)
                    continue
                total = max(0.0, float(exec_value) + carry)
                frames = float(math.floor(total + 0.5))
                frames = max(1.0, frames)
                anchor = max(1, source_count)
                lower = max(1, int(math.ceil(float(anchor - (row_budget_neg + prefix_offset)))))
                upper = max(lower, int(math.floor(float(anchor + (row_budget_pos - prefix_offset)))))
                frames = float(min(max(int(frames), lower), upper))
                row_projected[offset] = np.float32(frames)
                prefix_offset += int(frames) - anchor
                carry = total - frames
                boundary_hit = False
                if row_boundary is not None and bool(row_boundary[offset]):
                    boundary_hit = True
                if row_phrase_final is not None and bool(row_phrase_final[offset]):
                    boundary_hit = True
                if boundary_hit:
                    carry = float(carry) * float(boundary_carry_decay)
                    prefix_offset = int(round(float(prefix_offset) * float(boundary_carry_decay)))
                    prefix_offset = max(-int(row_budget_neg), min(int(row_budget_pos), int(prefix_offset)))
            projected[batch_idx, start_unit:committed_len] = torch.from_numpy(row_projected).to(
                device=projected.device,
                dtype=projected.dtype,
            )
            residual_next[batch_idx] = carry
            prefix_offset_next[batch_idx] = float(prefix_offset)
        return projected, residual_next.reshape(batch_size, 1), prefix_offset_next.reshape(batch_size, 1)

    @staticmethod
    def _extract_projection_row_numpy(
        *,
        batch_idx: int,
        start_unit: int,
        committed_len: int,
        unit_duration_exec: torch.Tensor,
        source_duration_obs: torch.Tensor,
        speech_commit_mask: torch.Tensor,
        coarse_only_commit_mask: torch.Tensor | None,
        source_boundary_cue: torch.Tensor | None,
        phrase_final_mask: torch.Tensor | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        row_slice = slice(int(start_unit), int(committed_len))
        row_exec = unit_duration_exec[batch_idx, row_slice].detach().float().cpu().numpy()
        row_source = source_duration_obs[batch_idx, row_slice].detach().float().cpu().numpy()
        row_source_rounded = np.rint(np.clip(row_source, a_min=0.0, a_max=None)).astype(np.int64, copy=False)
        row_speech = speech_commit_mask[batch_idx, row_slice].detach().cpu().numpy() > 0.5
        row_coarse = (
            coarse_only_commit_mask[batch_idx, row_slice].detach().cpu().numpy() > 0.5
            if isinstance(coarse_only_commit_mask, torch.Tensor)
            else None
        )
        row_boundary = (
            source_boundary_cue[batch_idx, row_slice].detach().cpu().numpy() >= 0.5
            if isinstance(source_boundary_cue, torch.Tensor)
            else None
        )
        row_phrase_final = (
            phrase_final_mask[batch_idx, row_slice].detach().cpu().numpy() > 0.5
            if isinstance(phrase_final_mask, torch.Tensor)
            else None
        )
        return row_exec, row_source_rounded, row_speech, row_coarse, row_boundary, row_phrase_final

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
        source_boundary_cue: torch.Tensor | None,
        phrase_final_mask: torch.Tensor | None,
        residual_next: torch.Tensor,
        prefix_unit_offset_next: torch.Tensor,
        cached_duration_exec: torch.Tensor,
        state_prev: DurationRuntimeState,
        boundary_reset_thresh: float,
    ) -> DurationRuntimeState:
        if isinstance(state_prev.since_last_boundary, torch.Tensor):
            since_last_boundary = state_prev.since_last_boundary.detach().float().clone()
        else:
            since_last_boundary = residual_next.new_zeros((commit_mask.size(0), 1))
        prev_committed_units = (
            state_prev.committed_units.long()
            if isinstance(getattr(state_prev, "committed_units", None), torch.Tensor)
            else torch.zeros((commit_mask.size(0),), dtype=torch.long, device=commit_mask.device)
        )
        committed_units = commit_mask.sum(dim=1).long()
        for batch_idx in range(commit_mask.size(0)):
            counter = int(round(float(since_last_boundary[batch_idx, 0].item())))
            start_unit = int(min(prev_committed_units[batch_idx].item(), committed_units[batch_idx].item()))
            end_unit = int(committed_units[batch_idx].item())
            for unit_idx in range(start_unit, end_unit):
                boundary_hit = False
                if isinstance(source_boundary_cue, torch.Tensor) and float(source_boundary_cue[batch_idx, unit_idx].item()) >= float(boundary_reset_thresh):
                    boundary_hit = True
                if isinstance(phrase_final_mask, torch.Tensor) and float(phrase_final_mask[batch_idx, unit_idx].item()) > 0.5:
                    boundary_hit = True
                counter = 0 if boundary_hit else (counter + 1)
            since_last_boundary[batch_idx, 0] = float(counter)
        return DurationRuntimeState(
            committed_units=committed_units,
            rounding_residual=residual_next.detach(),
            prefix_unit_offset=prefix_unit_offset_next.detach(),
            cached_duration_exec=cached_duration_exec.detach(),
            local_rate_ema=None if state_prev.local_rate_ema is None else state_prev.local_rate_ema.detach(),
            since_last_boundary=since_last_boundary.detach(),
            frontend_state=state_prev.frontend_state,
            consumed_content_steps=(
                None
                if state_prev.consumed_content_steps is None
                else state_prev.consumed_content_steps.detach().clone()
            ),
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
        coarse_only_commit_mask: torch.Tensor | None = None,
        source_boundary_cue: torch.Tensor | None = None,
        phrase_final_mask: torch.Tensor | None = None,
        role_attn_unit: torch.Tensor | None = None,
        role_value_unit: torch.Tensor | None = None,
        role_var_unit: torch.Tensor | None = None,
        role_conf_unit: torch.Tensor | None = None,
        unit_logstretch_raw: torch.Tensor | None = None,
        unit_duration_raw: torch.Tensor | None = None,
        global_bias_scalar: torch.Tensor | None = None,
        global_shift_analytic: torch.Tensor | None = None,
        coarse_logstretch: torch.Tensor | None = None,
        coarse_correction: torch.Tensor | None = None,
        local_residual: torch.Tensor | None = None,
        source_rate_seq: torch.Tensor | None = None,
        source_prefix_summary: torch.Tensor | None = None,
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
            coarse_only_commit_mask=(
                None if not isinstance(coarse_only_commit_mask, torch.Tensor) else coarse_only_commit_mask.float()
            ),
            source_boundary_cue=(
                None if not isinstance(source_boundary_cue, torch.Tensor) else source_boundary_cue.float()
            ),
            phrase_final_mask=(
                None if not isinstance(phrase_final_mask, torch.Tensor) else phrase_final_mask.float()
            ),
            residual_prev=state.rounding_residual,
            prefix_unit_offset_prev=state.prefix_unit_offset,
            committed_units_prev=state.committed_units,
            cached_duration_exec_prev=state.cached_duration_exec,
            budget_pos=self.prefix_budget_pos,
            budget_neg=self.prefix_budget_neg,
            dynamic_budget_ratio=self.dynamic_budget_ratio,
            min_prefix_budget=self.min_prefix_budget,
            max_prefix_budget=self.max_prefix_budget,
            boundary_carry_decay=self.boundary_carry_decay,
            boundary_reset_thresh=self.boundary_reset_thresh,
        )
        materialized_duration_exec = self._materialize_projected_duration(
            unit_duration_exec=unit_duration_exec,
            projected_duration_exec=projected_duration_exec,
            commit_mask=commit_mask,
        )
        cached_duration_exec = projected_duration_exec * commit_mask.float()
        next_state = self._build_next_state(
            commit_mask=commit_mask,
            source_boundary_cue=source_boundary_cue,
            phrase_final_mask=phrase_final_mask,
            residual_next=residual_next,
            prefix_unit_offset_next=prefix_unit_offset_next,
            cached_duration_exec=cached_duration_exec,
            state_prev=state,
            boundary_reset_thresh=self.boundary_reset_thresh,
        )
        frame_plan = self.build_frame_plan(
            source_duration_obs=source_duration_obs,
            projected_duration_exec=projected_duration_exec,
            commit_mask=commit_mask,
        )
        budget_hit_pos = prefix_unit_offset_next >= float(self.prefix_budget_pos - 1.0e-6)
        budget_hit_neg = prefix_unit_offset_next <= -float(self.prefix_budget_neg - 1.0e-6)
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
            coarse_logstretch=coarse_logstretch,
            coarse_correction=coarse_correction,
            local_residual=local_residual,
            source_rate_seq=source_rate_seq,
            source_prefix_summary=source_prefix_summary,
            prefix_unit_offset=prefix_unit_offset_next,
            projector_budget_hit_pos=budget_hit_pos.detach(),
            projector_budget_hit_neg=budget_hit_neg.detach(),
        )
