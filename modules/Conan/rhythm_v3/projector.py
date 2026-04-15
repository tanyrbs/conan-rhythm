from __future__ import annotations

import math

import torch
import torch.nn as nn

from .contracts import DurationExecution, DurationRuntimeState
from .frame_plan import build_frame_plan_from_execution


@torch.jit.script
def _project_row_recurrence_script(
    row_exec: torch.Tensor,
    row_source_rounded: torch.Tensor,
    row_source_budget: torch.Tensor,
    row_speech: torch.Tensor,
    row_coarse: torch.Tensor,
    row_boundary: torch.Tensor,
    row_phrase_final: torch.Tensor,
    carry_init: torch.Tensor,
    prefix_offset_init: torch.Tensor,
    row_budget_pos: int,
    row_budget_neg: int,
    boundary_carry_decay: float,
    boundary_offset_decay: float,
    boundary_reset_thresh: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_units = int(row_exec.numel())
    projected = torch.zeros_like(row_exec)
    boundary_hit = torch.zeros_like(row_exec)
    boundary_decay = torch.zeros_like(row_exec)
    carry = carry_init.reshape(1)
    prefix_offset = prefix_offset_init.reshape(1)
    for offset in range(num_units):
        is_speech = bool(row_speech[offset] > 0.5)
        is_coarse_only = bool(row_coarse[offset] > 0.5)
        anchor = torch.clamp_min(row_source_rounded[offset], 1.0)
        budget_anchor = torch.clamp_min(row_source_budget[offset], 1.0)
        if (not is_speech) and (not is_coarse_only):
            projected[offset] = anchor
            continue

        total = torch.clamp_min(row_exec[offset] + carry[0], 0.0)
        frames = torch.floor(total + 0.5)
        frames = torch.clamp_min(frames, 1.0)
        lower = torch.ceil(budget_anchor - (float(row_budget_neg) + prefix_offset[0]))
        lower = torch.clamp_min(lower, 1.0)
        upper = torch.floor(budget_anchor + (float(row_budget_pos) - prefix_offset[0]))
        upper = torch.maximum(upper, lower)
        frames = torch.minimum(torch.maximum(frames, lower), upper)
        projected[offset] = frames
        prefix_offset[0] = prefix_offset[0] + (frames - budget_anchor)
        carry[0] = total - frames

        boundary_event = bool(row_boundary[offset] >= boundary_reset_thresh) or bool(row_phrase_final[offset] > 0.5)
        if boundary_event:
            boundary_hit[offset] = 1.0
            if (
                boundary_carry_decay < (1.0 - 1.0e-6)
                or boundary_offset_decay < (1.0 - 1.0e-6)
            ):
                carry[0] = carry[0] * boundary_carry_decay
                prefix_offset[0] = torch.clamp(
                    prefix_offset[0] * boundary_offset_decay,
                    min=-float(row_budget_neg),
                    max=float(row_budget_pos),
                )
                boundary_decay[offset] = 1.0
    return projected, boundary_hit, boundary_decay, carry, prefix_offset


@torch.jit.script
def _project_row_recurrence_no_boundary_script(
    row_exec: torch.Tensor,
    row_source_rounded: torch.Tensor,
    row_source_budget: torch.Tensor,
    row_speech: torch.Tensor,
    row_coarse: torch.Tensor,
    carry_init: torch.Tensor,
    prefix_offset_init: torch.Tensor,
    row_budget_pos: int,
    row_budget_neg: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_units = int(row_exec.numel())
    projected = torch.zeros_like(row_exec)
    carry = carry_init.reshape(1)
    prefix_offset = prefix_offset_init.reshape(1)
    for offset in range(num_units):
        is_speech = bool(row_speech[offset] > 0.5)
        is_coarse_only = bool(row_coarse[offset] > 0.5)
        anchor = torch.clamp_min(row_source_rounded[offset], 1.0)
        budget_anchor = torch.clamp_min(row_source_budget[offset], 1.0)
        if (not is_speech) and (not is_coarse_only):
            projected[offset] = anchor
            continue

        total = torch.clamp_min(row_exec[offset] + carry[0], 0.0)
        frames = torch.floor(total + 0.5)
        frames = torch.clamp_min(frames, 1.0)
        lower = torch.ceil(budget_anchor - (float(row_budget_neg) + prefix_offset[0]))
        lower = torch.clamp_min(lower, 1.0)
        upper = torch.floor(budget_anchor + (float(row_budget_pos) - prefix_offset[0]))
        upper = torch.maximum(upper, lower)
        frames = torch.minimum(torch.maximum(frames, lower), upper)
        projected[offset] = frames
        prefix_offset[0] = prefix_offset[0] + (frames - budget_anchor)
        carry[0] = total - frames
    return projected, carry, prefix_offset


def _normalize_integer_projection_mode(value: str | None) -> str:
    mode = str(value or "greedy").strip().lower()
    aliases = {
        "": "greedy",
        "default": "greedy",
        "nearest": "greedy",
        "recurrent": "greedy",
        "dp": "prefix_optimal",
        "prefix": "prefix_optimal",
        "prefix_dp": "prefix_optimal",
        "closed_prefix": "prefix_optimal",
        "closed_prefix_optimal": "prefix_optimal",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"greedy", "prefix_optimal"}:
        raise ValueError(
            f"Unsupported rhythm_v3_integer_projection_mode={value!r}. "
            "Expected one of: greedy, prefix_optimal."
        )
    return mode


def _normalize_integer_projection_anchor_mode(value: str | None) -> str:
    mode = str(value or "rounded").strip().lower()
    aliases = {
        "": "rounded",
        "default": "rounded",
        "round": "rounded",
        "source_rounded": "rounded",
        "raw": "continuous",
        "float": "continuous",
        "source": "continuous",
        "source_continuous": "continuous",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"rounded", "continuous"}:
        raise ValueError(
            f"Unsupported rhythm_v3_integer_projection_anchor_mode={value!r}. "
            "Expected one of: rounded, continuous."
        )
    return mode


def _project_row_prefix_optimal_python(
    *,
    row_exec: torch.Tensor,
    row_source_rounded: torch.Tensor,
    row_source_budget: torch.Tensor,
    row_speech: torch.Tensor,
    row_coarse: torch.Tensor,
    row_boundary: torch.Tensor,
    row_phrase_final: torch.Tensor,
    carry_init: torch.Tensor,
    prefix_offset_init: torch.Tensor,
    row_budget_pos: int,
    row_budget_neg: int,
    boundary_carry_decay: float,
    boundary_offset_decay: float,
    boundary_reset_thresh: float,
    candidate_radius: int,
    max_states: int,
    terminal_carry_weight: float,
    terminal_offset_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Constrained projection over the currently closed prefix only.

    The dynamic program keeps the same streaming contract as the recurrent
    projector: cached units before this row slice are immutable, and the open
    tail is not inspected. It only replaces local greedy rounding within the
    newly committed prefix by a bounded shortest-path search over carry and
    prefix-offset states.
    """
    device = row_exec.device
    dtype = row_exec.dtype
    num_units = int(row_exec.numel())
    if num_units <= 0:
        empty = torch.zeros_like(row_exec)
        return empty, empty, empty, carry_init.reshape(1).to(device=device), prefix_offset_init.reshape(1).to(device=device)

    projected = torch.zeros_like(row_exec)
    boundary_hit = torch.zeros_like(row_exec)
    boundary_decay = torch.zeros_like(row_exec)
    candidate_radius = int(max(0, candidate_radius))
    max_states = int(max(1, max_states))
    terminal_carry_weight = float(max(0.0, terminal_carry_weight))
    terminal_offset_weight = float(max(0.0, terminal_offset_weight))
    row_budget_pos = int(max(0, row_budget_pos))
    row_budget_neg = int(max(0, row_budget_neg))

    initial_carry = float(carry_init.reshape(-1)[0].detach().cpu().item())
    initial_offset = float(prefix_offset_init.reshape(-1)[0].detach().cpu().item())
    # key -> (cost, path, carry, prefix_offset, hit_path, decay_path)
    states: dict[tuple[int, int], tuple[float, tuple[float, ...], float, float, tuple[float, ...], tuple[float, ...]]] = {
        (int(round(initial_offset * 1000.0)), int(round(initial_carry * 1000.0))): (
            0.0,
            tuple(),
            initial_carry,
            initial_offset,
            tuple(),
            tuple(),
        )
    }

    for offset in range(num_units):
        exec_value = float(row_exec[offset].detach().cpu().item())
        anchor = max(float(row_source_rounded[offset].detach().cpu().item()), 1.0)
        budget_anchor = max(float(row_source_budget[offset].detach().cpu().item()), 1.0)
        is_speech = bool(float(row_speech[offset].detach().cpu().item()) > 0.5)
        is_coarse_only = bool(float(row_coarse[offset].detach().cpu().item()) > 0.5)
        if (not is_speech) and (not is_coarse_only):
            next_states = {}
            for key, (cost, path, carry, prefix_offset, hit_path, decay_path) in states.items():
                next_states[key] = (
                    cost,
                    path + (anchor,),
                    carry,
                    prefix_offset,
                    hit_path + (0.0,),
                    decay_path + (0.0,),
                )
            states = next_states
            continue

        boundary_event = (
            bool(float(row_boundary[offset].detach().cpu().item()) >= float(boundary_reset_thresh))
            or bool(float(row_phrase_final[offset].detach().cpu().item()) > 0.5)
        )
        next_states: dict[
            tuple[int, int],
            tuple[float, tuple[float, ...], float, float, tuple[float, ...], tuple[float, ...]],
        ] = {}
        for cost, path, carry, prefix_offset, hit_path, decay_path in states.values():
            total = max(float(exec_value) + float(carry), 0.0)
            nearest = int(math.floor(total + 0.5))
            lower = int(math.ceil(budget_anchor - (float(row_budget_neg) + float(prefix_offset))))
            lower = max(lower, 1)
            upper = int(math.floor(budget_anchor + (float(row_budget_pos) - float(prefix_offset))))
            upper = max(upper, lower)
            centers = {
                nearest,
                int(math.floor(total)),
                int(math.ceil(total)),
                int(math.floor(exec_value)),
                int(math.ceil(exec_value)),
            }
            candidates: set[int] = {lower, upper}
            for center in centers:
                for delta in range(-candidate_radius, candidate_radius + 1):
                    value = int(center + delta)
                    if lower <= value <= upper:
                        candidates.add(value)
            if not candidates:
                candidates.add(lower)

            for frames_int in sorted(candidates):
                frames = float(max(1, frames_int))
                next_offset = float(prefix_offset) + (frames - budget_anchor)
                if next_offset > float(row_budget_pos) + 1.0e-6:
                    continue
                if next_offset < -float(row_budget_neg) - 1.0e-6:
                    continue
                next_carry = total - frames
                step_cost = abs(frames - total)
                hit_value = 1.0 if boundary_event else 0.0
                decay_value = 0.0
                if boundary_event:
                    if (
                        float(boundary_carry_decay) < (1.0 - 1.0e-6)
                        or float(boundary_offset_decay) < (1.0 - 1.0e-6)
                    ):
                        next_carry = next_carry * float(boundary_carry_decay)
                        next_offset = max(
                            -float(row_budget_neg),
                            min(float(row_budget_pos), next_offset * float(boundary_offset_decay)),
                        )
                        decay_value = 1.0
                next_cost = float(cost) + float(step_cost)
                state_key = (
                    int(round(next_offset * 1000.0)),
                    int(round(next_carry * 1000.0)),
                )
                state_value = (
                    next_cost,
                    path + (frames,),
                    next_carry,
                    next_offset,
                    hit_path + (hit_value,),
                    decay_path + (decay_value,),
                )
                current = next_states.get(state_key)
                if current is None or next_cost < current[0]:
                    next_states[state_key] = state_value
        if not next_states:
            # Keep the old recurrence as a safe fallback for pathological rows.
            return _project_row_recurrence_script(
                row_exec=row_exec,
                row_source_rounded=row_source_rounded,
                row_source_budget=row_source_budget,
                row_speech=row_speech.float(),
                row_coarse=row_coarse.float(),
                row_boundary=row_boundary.float(),
                row_phrase_final=row_phrase_final.float(),
                carry_init=carry_init,
                prefix_offset_init=prefix_offset_init,
                row_budget_pos=row_budget_pos,
                row_budget_neg=row_budget_neg,
                boundary_carry_decay=boundary_carry_decay,
                boundary_offset_decay=boundary_offset_decay,
                boundary_reset_thresh=boundary_reset_thresh,
            )
        ranked = sorted(
            next_states.values(),
            key=lambda state: (
                state[0]
                + terminal_carry_weight * abs(float(state[2]))
                + terminal_offset_weight * abs(float(state[3])),
                state[1],
            ),
        )
        states = {
            (
                int(round(state[3] * 1000.0)),
                int(round(state[2] * 1000.0)),
            ): state
            for state in ranked[:max_states]
        }

    best = min(
        states.values(),
        key=lambda state: (
            state[0]
            + terminal_carry_weight * abs(float(state[2]))
            + terminal_offset_weight * abs(float(state[3])),
            state[1],
        ),
    )
    _, best_path, best_carry, best_offset, best_hit, best_decay = best
    projected = torch.as_tensor(best_path, dtype=dtype, device=device).reshape_as(row_exec)
    boundary_hit = torch.as_tensor(best_hit, dtype=dtype, device=device).reshape_as(row_exec)
    boundary_decay = torch.as_tensor(best_decay, dtype=dtype, device=device).reshape_as(row_exec)
    return (
        projected,
        boundary_hit,
        boundary_decay,
        row_exec.new_tensor([best_carry]),
        row_exec.new_tensor([best_offset]),
    )


class StreamingDurationProjector(nn.Module):
    def __init__(
        self,
        *,
        prefix_budget_pos: int = 24,
        prefix_budget_neg: int = 24,
        dynamic_budget_ratio: float = 0.0,
        min_prefix_budget: int = 0,
        max_prefix_budget: int = 0,
        budget_mode: str = "total",
        boundary_carry_decay: float = 0.25,
        boundary_offset_decay: float | None = None,
        boundary_reset_thresh: float = 0.5,
        integer_projection_mode: str = "greedy",
        integer_projection_anchor_mode: str = "rounded",
        prefix_projection_candidate_radius: int = 2,
        prefix_projection_max_states: int = 256,
        prefix_projection_terminal_carry_weight: float = 0.25,
        prefix_projection_terminal_offset_weight: float = 0.05,
        export_projector_telemetry: bool = True,
    ) -> None:
        super().__init__()
        self.prefix_budget_pos = int(max(0, prefix_budget_pos))
        self.prefix_budget_neg = int(max(0, prefix_budget_neg))
        self.dynamic_budget_ratio = float(max(0.0, dynamic_budget_ratio))
        self.min_prefix_budget = int(max(0, min_prefix_budget))
        self.max_prefix_budget = int(max(0, max_prefix_budget))
        self.budget_mode = str(budget_mode or "total").strip().lower()
        if self.budget_mode not in {"total", "speech_only", "hybrid"}:
            raise ValueError(f"Unsupported rhythm_v3 budget_mode={budget_mode!r}")
        self.boundary_carry_decay = float(max(0.0, min(1.0, boundary_carry_decay)))
        if boundary_offset_decay is None:
            boundary_offset_decay = boundary_carry_decay
        self.boundary_offset_decay = float(max(0.0, min(1.0, boundary_offset_decay)))
        self.boundary_reset_thresh = float(max(0.0, min(1.0, boundary_reset_thresh)))
        self.integer_projection_mode = _normalize_integer_projection_mode(integer_projection_mode)
        self.integer_projection_anchor_mode = _normalize_integer_projection_anchor_mode(integer_projection_anchor_mode)
        self.prefix_projection_candidate_radius = int(max(0, prefix_projection_candidate_radius))
        self.prefix_projection_max_states = int(max(1, prefix_projection_max_states))
        self.prefix_projection_terminal_carry_weight = float(max(0.0, prefix_projection_terminal_carry_weight))
        self.prefix_projection_terminal_offset_weight = float(max(0.0, prefix_projection_terminal_offset_weight))
        self.export_projector_telemetry = bool(export_projector_telemetry)

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
        sealed_mask: torch.Tensor | None = None,
    ) -> None:
        if unit_mask.numel() <= 0:
            return
        steps = torch.arange(unit_mask.size(1), device=unit_mask.device)[None, :]
        visible = unit_mask.float() > 0.5
        actual = commit_mask.float() > 0.5
        visible_len = visible.float().sum(dim=1).long()
        committed_len = actual.float().sum(dim=1).long()
        expected = (steps < committed_len[:, None]) & (steps < visible_len[:, None])
        if not torch.equal(actual, expected):
            raise ValueError("Duration V3 commit mask must form a contiguous visible prefix.")
        if isinstance(sealed_mask, torch.Tensor):
            closed = (sealed_mask.float() > 0.5) & visible
            if bool((actual & ~closed).any().item()):
                raise ValueError("Duration V3 commit mask cannot include open-tail units.")

    @staticmethod
    def _build_commit_invariant_telemetry(
        *,
        unit_mask: torch.Tensor,
        commit_mask: torch.Tensor,
        sealed_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        visible = unit_mask.float() > 0.5
        committed = commit_mask.float() > 0.5
        steps = torch.arange(unit_mask.size(1), device=unit_mask.device)[None, :]
        visible_len = visible.float().sum(dim=1).long()
        committed_len = committed.float().sum(dim=1).long()
        expected = (steps < committed_len[:, None]) & (steps < visible_len[:, None])
        closed_prefix_ok = (committed == expected).all(dim=1, keepdim=True).to(dtype=commit_mask.dtype)
        if isinstance(sealed_mask, torch.Tensor):
            closed = (sealed_mask.float() > 0.5) & visible
            open_tail_commit_violation = (committed & ~closed).to(dtype=commit_mask.dtype)
        else:
            open_tail_commit_violation = commit_mask.new_zeros(commit_mask.shape)
        open_tail_commit_violation_count = open_tail_commit_violation.float().sum(dim=1, keepdim=True)
        return closed_prefix_ok, open_tail_commit_violation, open_tail_commit_violation_count

    @classmethod
    def _build_commit_mask(
        cls,
        *,
        unit_mask: torch.Tensor,
        sealed_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        commit_mask = unit_mask.float() if sealed_mask is None else unit_mask.float() * sealed_mask.float()
        cls._validate_prefix_commit_mask(
            unit_mask=unit_mask.float(),
            commit_mask=commit_mask,
            sealed_mask=sealed_mask,
        )
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
        speech_commit_mask: torch.Tensor | None,
        committed_len: int,
        static_budget: int,
        dynamic_budget_ratio: float,
        min_prefix_budget: int,
        max_prefix_budget: int,
        budget_mode: str,
    ) -> int:
        if committed_len <= 0 or float(dynamic_budget_ratio) <= 0.0:
            return int(max(0, static_budget))
        prefix_source_total = float(
            StreamingDurationProjector._resolve_prefix_support_mass_tensor(
                source_duration_obs=source_duration_obs.reshape(1, -1),
                speech_commit_mask=(
                    None
                    if speech_commit_mask is None
                    else speech_commit_mask.reshape(1, -1)
                ),
                committed_len=torch.as_tensor([committed_len], device=source_duration_obs.device, dtype=torch.long),
                budget_mode=budget_mode,
            )[0].item()
        )
        dynamic_budget = int(round(prefix_source_total * float(dynamic_budget_ratio)))
        dynamic_budget = max(int(max(0, min_prefix_budget)), dynamic_budget)
        if int(max_prefix_budget) > 0:
            dynamic_budget = min(dynamic_budget, int(max_prefix_budget))
        return int(max(0, dynamic_budget))

    @staticmethod
    def _resolve_prefix_support_mass_tensor(
        *,
        source_duration_obs: torch.Tensor,
        speech_commit_mask: torch.Tensor | None,
        committed_len: torch.Tensor,
        budget_mode: str,
    ) -> torch.Tensor:
        committed_len = committed_len.long().reshape(-1)
        source_duration = source_duration_obs.float()
        source_cumsum = torch.cumsum(source_duration, dim=1)
        gather_idx = (committed_len.clamp_min(1) - 1).unsqueeze(1)
        prefix_total = torch.gather(source_cumsum, 1, gather_idx).squeeze(1)
        prefix_total = torch.where(committed_len > 0, prefix_total, torch.zeros_like(prefix_total))
        resolved_mode = str(budget_mode or "total").strip().lower()
        if resolved_mode == "total" or speech_commit_mask is None:
            return prefix_total
        speech_duration = source_duration * speech_commit_mask.float()
        speech_cumsum = torch.cumsum(speech_duration, dim=1)
        prefix_speech = torch.gather(speech_cumsum, 1, gather_idx).squeeze(1)
        prefix_speech = torch.where(committed_len > 0, prefix_speech, torch.zeros_like(prefix_speech))
        if resolved_mode == "speech_only":
            return prefix_speech
        if resolved_mode == "hybrid":
            prefix_silence = torch.clamp(prefix_total - prefix_speech, min=0.0)
            return prefix_speech + (0.5 * prefix_silence)
        raise ValueError(f"Unsupported rhythm_v3 budget_mode={budget_mode!r}")

    @staticmethod
    def _resolve_prefix_budget_tensor(
        *,
        source_duration_obs: torch.Tensor,
        speech_commit_mask: torch.Tensor | None,
        committed_len: torch.Tensor,
        static_budget: int,
        dynamic_budget_ratio: float,
        min_prefix_budget: int,
        max_prefix_budget: int,
        budget_mode: str,
    ) -> torch.Tensor:
        committed_len = committed_len.long().reshape(-1)
        if float(dynamic_budget_ratio) <= 0.0:
            return source_duration_obs.new_full(committed_len.shape, float(max(0, static_budget)))
        prefix_source_total = StreamingDurationProjector._resolve_prefix_support_mass_tensor(
            source_duration_obs=source_duration_obs,
            speech_commit_mask=speech_commit_mask,
            committed_len=committed_len,
            budget_mode=budget_mode,
        )
        dynamic_budget = torch.round(prefix_source_total * float(dynamic_budget_ratio))
        dynamic_budget = torch.clamp(dynamic_budget, min=float(max(0, min_prefix_budget)))
        if int(max_prefix_budget) > 0:
            dynamic_budget = torch.clamp(dynamic_budget, max=float(max_prefix_budget))
        return torch.clamp(dynamic_budget, min=0.0)

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
        budget_mode: str = "total",
        boundary_carry_decay: float = 0.25,
        boundary_offset_decay: float | None = None,
        boundary_reset_thresh: float = 0.5,
        integer_projection_mode: str = "greedy",
        integer_projection_anchor_mode: str = "rounded",
        prefix_projection_candidate_radius: int = 2,
        prefix_projection_max_states: int = 256,
        prefix_projection_terminal_carry_weight: float = 0.25,
        prefix_projection_terminal_offset_weight: float = 0.05,
        committed_len: torch.Tensor | None = None,
        budget_pos_tensor: torch.Tensor | None = None,
        budget_neg_tensor: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_units = unit_duration_exec.shape
        projected = unit_duration_exec.new_zeros(unit_duration_exec.shape)
        boundary_hit = unit_duration_exec.new_zeros(unit_duration_exec.shape)
        boundary_decay_applied = unit_duration_exec.new_zeros(unit_duration_exec.shape)
        resolved_boundary_offset_decay = (
            float(boundary_carry_decay)
            if boundary_offset_decay is None
            else float(boundary_offset_decay)
        )
        source_anchor_rounded = torch.round(source_duration_obs.detach().float().clamp_min(0.0)).to(
            device=unit_duration_exec.device
        )
        anchor_mode = _normalize_integer_projection_anchor_mode(integer_projection_anchor_mode)
        source_anchor_budget = (
            source_duration_obs.detach().float().clamp_min(0.0).to(device=unit_duration_exec.device)
            if anchor_mode == "continuous"
            else source_anchor_rounded
        )
        residual_next = residual_prev.float().reshape(batch_size).detach().clone().to(device=unit_duration_exec.device)
        prefix_offset_next = (
            prefix_unit_offset_prev.float().reshape(batch_size).detach().clone().to(device=unit_duration_exec.device)
        )
        committed_len_tensor = (
            commit_mask.float().sum(dim=1).long()
            if committed_len is None
            else committed_len.long().reshape(batch_size)
        )
        budget_pos_values = (
            StreamingDurationProjector._resolve_prefix_budget_tensor(
                source_duration_obs=source_duration_obs,
                speech_commit_mask=speech_commit_mask,
                committed_len=committed_len_tensor,
                static_budget=budget_pos,
                dynamic_budget_ratio=dynamic_budget_ratio,
                min_prefix_budget=min_prefix_budget,
                max_prefix_budget=max_prefix_budget,
                budget_mode=budget_mode,
            )
            if budget_pos_tensor is None
            else budget_pos_tensor.float().reshape(batch_size)
        )
        budget_neg_values = (
            StreamingDurationProjector._resolve_prefix_budget_tensor(
                source_duration_obs=source_duration_obs,
                speech_commit_mask=speech_commit_mask,
                committed_len=committed_len_tensor,
                static_budget=budget_neg,
                dynamic_budget_ratio=dynamic_budget_ratio,
                min_prefix_budget=min_prefix_budget,
                max_prefix_budget=max_prefix_budget,
                budget_mode=budget_mode,
            )
            if budget_neg_tensor is None
            else budget_neg_tensor.float().reshape(batch_size)
        )
        if isinstance(committed_units_prev, torch.Tensor):
            committed_units_prev = committed_units_prev.long().reshape(batch_size)
        else:
            committed_units_prev = torch.zeros((batch_size,), dtype=torch.long, device=unit_duration_exec.device)
        cached_prev = None
        prev_prefix = torch.zeros((batch_size,), dtype=torch.long, device=unit_duration_exec.device)
        if isinstance(cached_duration_exec_prev, torch.Tensor):
            cached_prev = cached_duration_exec_prev.to(device=unit_duration_exec.device, dtype=unit_duration_exec.dtype)
            cached_width = min(int(cached_prev.size(1)), int(projected.size(1)))
            prev_prefix = torch.minimum(
                committed_units_prev.clamp_min(0),
                committed_len_tensor,
            )
            if cached_width > 0:
                prev_prefix = torch.minimum(
                    prev_prefix,
                    torch.full_like(prev_prefix, cached_width),
                )
                positions = torch.arange(cached_width, device=unit_duration_exec.device).reshape(1, cached_width)
                prefix_mask = positions < prev_prefix.unsqueeze(1)
                projected[:, :cached_width] = torch.where(
                    prefix_mask,
                    cached_prev[:, :cached_width],
                    projected[:, :cached_width],
                )
        active_row_tensor = torch.nonzero(committed_len_tensor > prev_prefix, as_tuple=False).reshape(-1)
        budget_pos_values_rounded = torch.round(budget_pos_values.detach()).long()
        budget_neg_values_rounded = torch.round(budget_neg_values.detach()).long()
        has_boundary_inputs = isinstance(source_boundary_cue, torch.Tensor) or isinstance(phrase_final_mask, torch.Tensor)
        projection_mode = _normalize_integer_projection_mode(integer_projection_mode)
        for row_idx in active_row_tensor:
            batch_idx = int(row_idx.item())
            row_committed_len = int(committed_len_tensor[batch_idx].item())
            row_budget_pos = int(budget_pos_values_rounded[batch_idx].item())
            row_budget_neg = int(budget_neg_values_rounded[batch_idx].item())
            start_unit = int(prev_prefix[batch_idx].item())
            carry = float(residual_next[batch_idx].item())
            prefix_offset = float(prefix_offset_next[batch_idx].item())
            row_exec, row_source_rounded, row_source_budget, row_speech, row_coarse, row_boundary, row_phrase_final = (
                StreamingDurationProjector._extract_projection_row_torch(
                    batch_idx=batch_idx,
                    start_unit=start_unit,
                    committed_len=row_committed_len,
                    unit_duration_exec=unit_duration_exec,
                    source_anchor_rounded=source_anchor_rounded,
                    source_anchor_budget=source_anchor_budget,
                    speech_commit_mask=speech_commit_mask,
                    coarse_only_commit_mask=coarse_only_commit_mask,
                    source_boundary_cue=source_boundary_cue,
                    phrase_final_mask=phrase_final_mask,
                )
            )
            row_coarse_float = row_coarse.float() if isinstance(row_coarse, torch.Tensor) else torch.zeros_like(row_exec)
            if projection_mode == "prefix_optimal":
                (
                    row_projected,
                    row_boundary_hit,
                    row_boundary_decay,
                    carry_tensor,
                    prefix_offset_tensor,
                ) = _project_row_prefix_optimal_python(
                    row_exec=row_exec,
                    row_source_rounded=row_source_rounded,
                    row_source_budget=row_source_budget,
                    row_speech=row_speech.float(),
                    row_coarse=row_coarse_float,
                    row_boundary=(row_boundary if isinstance(row_boundary, torch.Tensor) else torch.zeros_like(row_exec)),
                    row_phrase_final=(
                        row_phrase_final.float()
                        if isinstance(row_phrase_final, torch.Tensor)
                        else torch.zeros_like(row_exec)
                    ),
                    carry_init=row_exec.new_tensor([carry]),
                    prefix_offset_init=row_exec.new_tensor([prefix_offset]),
                    row_budget_pos=row_budget_pos,
                    row_budget_neg=row_budget_neg,
                    boundary_carry_decay=boundary_carry_decay,
                    boundary_offset_decay=resolved_boundary_offset_decay,
                    boundary_reset_thresh=boundary_reset_thresh,
                    candidate_radius=prefix_projection_candidate_radius,
                    max_states=prefix_projection_max_states,
                    terminal_carry_weight=prefix_projection_terminal_carry_weight,
                    terminal_offset_weight=prefix_projection_terminal_offset_weight,
                )
                if not has_boundary_inputs:
                    row_boundary_hit = torch.zeros_like(row_exec)
                    row_boundary_decay = torch.zeros_like(row_exec)
            elif not has_boundary_inputs:
                row_projected, carry_tensor, prefix_offset_tensor = _project_row_recurrence_no_boundary_script(
                    row_exec=row_exec,
                    row_source_rounded=row_source_rounded,
                    row_source_budget=row_source_budget,
                    row_speech=row_speech.float(),
                    row_coarse=row_coarse_float,
                    carry_init=row_exec.new_tensor([carry]),
                    prefix_offset_init=row_exec.new_tensor([prefix_offset]),
                    row_budget_pos=row_budget_pos,
                    row_budget_neg=row_budget_neg,
                )
                row_boundary_hit = torch.zeros_like(row_exec)
                row_boundary_decay = torch.zeros_like(row_exec)
            else:
                (
                    row_projected,
                    row_boundary_hit,
                    row_boundary_decay,
                    carry_tensor,
                    prefix_offset_tensor,
                ) = _project_row_recurrence_script(
                    row_exec=row_exec,
                    row_source_rounded=row_source_rounded,
                    row_source_budget=row_source_budget,
                    row_speech=row_speech.float(),
                    row_coarse=row_coarse_float,
                    row_boundary=(row_boundary if isinstance(row_boundary, torch.Tensor) else torch.zeros_like(row_exec)),
                    row_phrase_final=(
                        row_phrase_final.float()
                        if isinstance(row_phrase_final, torch.Tensor)
                        else torch.zeros_like(row_exec)
                    ),
                    carry_init=row_exec.new_tensor([carry]),
                    prefix_offset_init=row_exec.new_tensor([prefix_offset]),
                    row_budget_pos=row_budget_pos,
                    row_budget_neg=row_budget_neg,
                    boundary_carry_decay=boundary_carry_decay,
                    boundary_offset_decay=resolved_boundary_offset_decay,
                    boundary_reset_thresh=boundary_reset_thresh,
                )
            projected[batch_idx, start_unit:row_committed_len] = row_projected.to(
                device=projected.device,
                dtype=projected.dtype,
            )
            boundary_hit[batch_idx, start_unit:row_committed_len] = row_boundary_hit.to(
                device=boundary_hit.device,
                dtype=boundary_hit.dtype,
            )
            boundary_decay_applied[batch_idx, start_unit:row_committed_len] = row_boundary_decay.to(
                device=boundary_decay_applied.device,
                dtype=boundary_decay_applied.dtype,
            )
            residual_next[batch_idx] = carry_tensor.reshape(-1)[0].to(
                device=residual_next.device,
                dtype=residual_next.dtype,
            )
            prefix_offset_next[batch_idx] = prefix_offset_tensor.reshape(-1)[0].to(
                device=prefix_offset_next.device,
                dtype=prefix_offset_next.dtype,
            )
        return (
            projected,
            residual_next.reshape(batch_size, 1),
            prefix_offset_next.reshape(batch_size, 1),
            boundary_hit,
            boundary_decay_applied,
        )

    @staticmethod
    def _extract_projection_row_torch(
        *,
        batch_idx: int,
        start_unit: int,
        committed_len: int,
        unit_duration_exec: torch.Tensor,
        source_anchor_rounded: torch.Tensor,
        source_anchor_budget: torch.Tensor,
        speech_commit_mask: torch.Tensor,
        coarse_only_commit_mask: torch.Tensor | None,
        source_boundary_cue: torch.Tensor | None,
        phrase_final_mask: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        row_slice = slice(int(start_unit), int(committed_len))
        row_exec = unit_duration_exec[batch_idx, row_slice].detach().float()
        row_source_rounded = source_anchor_rounded[batch_idx, row_slice]
        row_source_budget = source_anchor_budget[batch_idx, row_slice]
        row_speech = speech_commit_mask[batch_idx, row_slice].detach().float() > 0.5
        row_coarse = (
            (coarse_only_commit_mask[batch_idx, row_slice].detach().float() > 0.5)
            if isinstance(coarse_only_commit_mask, torch.Tensor)
            else None
        )
        row_boundary = (
            source_boundary_cue[batch_idx, row_slice].detach().float()
            if isinstance(source_boundary_cue, torch.Tensor)
            else None
        )
        row_phrase_final = (
            phrase_final_mask[batch_idx, row_slice].detach().float() > 0.5
            if isinstance(phrase_final_mask, torch.Tensor)
            else None
        )
        return row_exec, row_source_rounded, row_source_budget, row_speech, row_coarse, row_boundary, row_phrase_final

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
        boundary_hit: torch.Tensor | None = None,
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
        active_row_tensor = torch.nonzero(committed_units > prev_committed_units, as_tuple=False).reshape(-1)
        boundary_reset_thresh_float = float(boundary_reset_thresh)
        for row_idx in active_row_tensor:
            batch_idx = int(row_idx.item())
            counter = int(round(float(since_last_boundary[batch_idx, 0].item())))
            end_unit = int(committed_units[batch_idx].item())
            start_unit = int(min(int(prev_committed_units[batch_idx].item()), end_unit))
            span = max(0, end_unit - start_unit)
            if span <= 0:
                since_last_boundary[batch_idx, 0] = float(counter)
                continue
            if isinstance(boundary_hit, torch.Tensor):
                row_hits = boundary_hit[batch_idx, start_unit:end_unit].reshape(-1)
                hit_indices = torch.nonzero(row_hits > 0.5, as_tuple=False).reshape(-1)
                if int(hit_indices.numel()) > 0:
                    last_hit = int(hit_indices[-1].item())
                    counter = int(span - 1 - last_hit)
                else:
                    counter += int(span)
            else:
                row_boundary_hit = None
                if isinstance(source_boundary_cue, torch.Tensor):
                    row_boundary_hit = (
                        source_boundary_cue[batch_idx, start_unit:end_unit].reshape(-1) >= boundary_reset_thresh_float
                    )
                if isinstance(phrase_final_mask, torch.Tensor):
                    row_phrase_final = phrase_final_mask[batch_idx, start_unit:end_unit].reshape(-1) > 0.5
                    row_boundary_hit = (
                        row_phrase_final
                        if row_boundary_hit is None
                        else (row_boundary_hit | row_phrase_final)
                    )
                if isinstance(row_boundary_hit, torch.Tensor):
                    hit_indices = torch.nonzero(row_boundary_hit, as_tuple=False).reshape(-1)
                    if int(hit_indices.numel()) > 0:
                        last_hit = int(hit_indices[-1].item())
                        counter = int(span - 1 - last_hit)
                    else:
                        counter += int(span)
                else:
                    counter += int(span)
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
        coarse_path_logstretch: torch.Tensor | None = None,
        coarse_correction: torch.Tensor | None = None,
        coarse_correction_pred: torch.Tensor | None = None,
        local_residual: torch.Tensor | None = None,
        local_residual_pred: torch.Tensor | None = None,
        speech_pred: torch.Tensor | None = None,
        silence_pred: torch.Tensor | None = None,
        source_rate_seq: torch.Tensor | None = None,
        source_prefix_summary: torch.Tensor | None = None,
        analytic_gap_raw: torch.Tensor | None = None,
        analytic_gap_clipped: torch.Tensor | None = None,
        analytic_clip_hit: torch.Tensor | None = None,
        analytic_clip_hit_rate: torch.Tensor | None = None,
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
        commit_closed_prefix_ok, open_tail_commit_violation, open_tail_commit_violation_count = (
            self._build_commit_invariant_telemetry(
                unit_mask=unit_mask.float(),
                commit_mask=commit_mask,
                sealed_mask=sealed_mask,
            )
        )
        committed_len = commit_mask.float().sum(dim=1).long()
        budget_pos_used = self._resolve_prefix_budget_tensor(
            source_duration_obs=source_duration_obs,
            speech_commit_mask=speech_commit_mask.float(),
            committed_len=committed_len,
            static_budget=self.prefix_budget_pos,
            dynamic_budget_ratio=self.dynamic_budget_ratio,
            min_prefix_budget=self.min_prefix_budget,
            max_prefix_budget=self.max_prefix_budget,
            budget_mode=self.budget_mode,
        ).unsqueeze(1)
        budget_neg_used = self._resolve_prefix_budget_tensor(
            source_duration_obs=source_duration_obs,
            speech_commit_mask=speech_commit_mask.float(),
            committed_len=committed_len,
            static_budget=self.prefix_budget_neg,
            dynamic_budget_ratio=self.dynamic_budget_ratio,
            min_prefix_budget=self.min_prefix_budget,
            max_prefix_budget=self.max_prefix_budget,
            budget_mode=self.budget_mode,
        ).unsqueeze(1)
        projected_duration_exec, residual_next, prefix_unit_offset_next, boundary_hit, boundary_decay_applied = self._project_duration_prefix(
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
            budget_mode=self.budget_mode,
            boundary_carry_decay=self.boundary_carry_decay,
            boundary_offset_decay=self.boundary_offset_decay,
            boundary_reset_thresh=self.boundary_reset_thresh,
            integer_projection_mode=self.integer_projection_mode,
            integer_projection_anchor_mode=self.integer_projection_anchor_mode,
            prefix_projection_candidate_radius=self.prefix_projection_candidate_radius,
            prefix_projection_max_states=self.prefix_projection_max_states,
            prefix_projection_terminal_carry_weight=self.prefix_projection_terminal_carry_weight,
            prefix_projection_terminal_offset_weight=self.prefix_projection_terminal_offset_weight,
            committed_len=committed_len,
            budget_pos_tensor=budget_pos_used.reshape(batch_size),
            budget_neg_tensor=budget_neg_used.reshape(batch_size),
        )
        projected_prefix_cumsum = (
            torch.cumsum(projected_duration_exec * commit_mask.float(), dim=1)
            if self.export_projector_telemetry
            else None
        )
        preclamp_duration_exec = unit_duration_exec.float() * commit_mask.float()
        preclamp_prefix_cumsum = (
            torch.cumsum(preclamp_duration_exec, dim=1)
            if self.export_projector_telemetry
            else None
        )
        source_prefix_cumsum = (
            torch.cumsum(source_duration_obs.float() * commit_mask.float(), dim=1)
            if self.export_projector_telemetry
            else None
        )
        preclamp_rounded_exec = torch.clamp(torch.round(preclamp_duration_exec), min=1.0) * commit_mask.float()
        projector_preclamp_exec = (
            preclamp_duration_exec.detach() if self.export_projector_telemetry else None
        )
        clamp_delta = (
            (projected_duration_exec.float() - preclamp_duration_exec) * commit_mask.float()
        )
        projector_clamp_delta = clamp_delta.detach() if self.export_projector_telemetry else None
        projector_clamp_mass = (
            (projected_duration_exec.float() - preclamp_rounded_exec).abs().detach()
            if self.export_projector_telemetry
            else None
        )
        projector_rounding_regret = (
            (projected_duration_exec.float() - preclamp_duration_exec).abs().detach()
            if self.export_projector_telemetry
            else None
        )
        projector_projection_regret = projector_rounding_regret
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
            boundary_hit=boundary_hit,
        )
        frame_plan = self.build_frame_plan(
            source_duration_obs=source_duration_obs,
            projected_duration_exec=projected_duration_exec,
            commit_mask=commit_mask,
        )
        budget_hit_pos = prefix_unit_offset_next >= (budget_pos_used - 1.0e-6)
        budget_hit_neg = prefix_unit_offset_next <= -(budget_neg_used - 1.0e-6)
        budget_hit_mask = budget_hit_pos | budget_hit_neg
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
            coarse_path_logstretch=coarse_path_logstretch if coarse_path_logstretch is not None else coarse_logstretch,
            coarse_correction=coarse_correction,
            coarse_correction_pred=(
                coarse_correction_pred if coarse_correction_pred is not None else coarse_correction
            ),
            local_residual=local_residual,
            local_residual_pred=(
                local_residual_pred if local_residual_pred is not None else local_residual
            ),
            speech_pred=speech_pred,
            silence_pred=silence_pred,
            source_rate_seq=source_rate_seq,
            source_prefix_summary=source_prefix_summary,
            analytic_gap_raw=analytic_gap_raw,
            analytic_gap_clipped=analytic_gap_clipped,
            analytic_clip_hit=analytic_clip_hit,
            analytic_clip_hit_rate=analytic_clip_hit_rate,
            prefix_unit_offset=prefix_unit_offset_next,
            projector_rounding_residual=residual_next.detach(),
            projector_budget_pos_used=budget_pos_used.detach(),
            projector_budget_neg_used=budget_neg_used.detach(),
            projector_budget_hit_pos=budget_hit_pos.detach(),
            projector_budget_hit_neg=budget_hit_neg.detach(),
            projector_budget_hit_mask=budget_hit_mask.detach(),
            projector_boundary_hit=(boundary_hit.detach() if self.export_projector_telemetry else None),
            projector_boundary_decay_applied=(
                boundary_decay_applied.detach() if self.export_projector_telemetry else None
            ),
            projector_since_last_boundary=(
                next_state.since_last_boundary.detach() if self.export_projector_telemetry else None
            ),
            projector_budget_mode=self.budget_mode,
            projector_prefix_drift=prefix_unit_offset_next.detach(),
            projector_preclamp_exec=projector_preclamp_exec,
            projector_preclamp_duration_exec=projector_preclamp_exec,
            projector_clamp_delta=projector_clamp_delta,
            projector_clamp_mass=projector_clamp_mass,
            projector_rounding_regret=projector_rounding_regret,
            projector_projection_regret=projector_rounding_regret,
            commit_closed_prefix_ok=commit_closed_prefix_ok.detach(),
            open_tail_commit_violation=open_tail_commit_violation.detach(),
            open_tail_commit_violation_count=open_tail_commit_violation_count.detach(),
            projected_prefix_cumsum=(
                None if projected_prefix_cumsum is None else projected_prefix_cumsum.detach()
            ),
            projector_preclamp_prefix_cumsum=(
                None if preclamp_prefix_cumsum is None else preclamp_prefix_cumsum.detach()
            ),
            source_prefix_cumsum=(
                None if source_prefix_cumsum is None else source_prefix_cumsum.detach()
            ),
        )
