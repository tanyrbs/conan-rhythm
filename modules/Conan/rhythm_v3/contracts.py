from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from modules.Conan.rhythm.frame_plan import RhythmFramePlan


@dataclass
class SourceUnitBatch:
    content_units: torch.Tensor
    source_runlen_src: torch.Tensor
    unit_anchor_base: torch.Tensor
    unit_mask: torch.Tensor
    edge_cue: torch.Tensor
    open_run_mask: Optional[torch.Tensor] = None
    sealed_mask: Optional[torch.Tensor] = None
    sep_hint: Optional[torch.Tensor] = None
    boundary_confidence: Optional[torch.Tensor] = None

    @property
    def dur_anchor_src(self) -> torch.Tensor:
        return self.source_runlen_src


@dataclass
class ReferenceDurationMemory:
    global_rate: torch.Tensor
    role_keys: torch.Tensor
    role_value: torch.Tensor
    role_coverage: torch.Tensor
    prompt_role_feat: Optional[torch.Tensor] = None
    prompt_rel_stretch: Optional[torch.Tensor] = None
    prompt_mask: Optional[torch.Tensor] = None
    prompt_role_attention: Optional[torch.Tensor] = None
    prompt_reconstruction: Optional[torch.Tensor] = None
    role_summary: Optional[torch.Tensor] = None
    raw_stats: Optional[torch.Tensor] = None
    raw_trace: Optional[torch.Tensor] = None


@dataclass
class DurationRuntimeState:
    committed_units: torch.Tensor
    cumulative_pred_frames: torch.Tensor
    cumulative_tgt_frames: Optional[torch.Tensor] = None
    cached_duration_exec: Optional[torch.Tensor] = None

    @property
    def commit_frontier(self) -> torch.Tensor:
        return self.committed_units.long()

    @property
    def clock_delta(self) -> torch.Tensor:
        if self.cumulative_tgt_frames is None:
            return self.cumulative_pred_frames.new_zeros(self.cumulative_pred_frames.shape)
        return self.cumulative_pred_frames - self.cumulative_tgt_frames

    @property
    def backlog(self) -> torch.Tensor:
        return self.clock_delta.clamp_min(0.0)


@dataclass
class DurationExecution:
    unit_logstretch: torch.Tensor
    unit_duration_exec: torch.Tensor
    role_attention: torch.Tensor
    next_state: DurationRuntimeState
    frame_plan: Optional["RhythmFramePlan"] = None
    anti_pos_logits: Optional[torch.Tensor] = None
    prompt_reconstruction: Optional[torch.Tensor] = None
    prompt_rel_stretch: Optional[torch.Tensor] = None
    prompt_mask: Optional[torch.Tensor] = None

    @property
    def speech_duration_exec(self) -> torch.Tensor:
        return self.unit_duration_exec

    @property
    def blank_duration_exec(self) -> torch.Tensor:
        return torch.zeros_like(self.unit_duration_exec)

    @property
    def pause_after_exec(self) -> torch.Tensor:
        return self.blank_duration_exec

    @property
    def effective_duration_exec(self) -> torch.Tensor:
        return self.unit_duration_exec

    @property
    def commit_frontier(self) -> torch.Tensor:
        return self.next_state.commit_frontier

    @property
    def planner(self):
        return None


def _move_tensor(value, *, device: torch.device, dtype: torch.dtype | None = None):
    if value is None or not isinstance(value, torch.Tensor):
        return value
    kwargs = {"device": device}
    if dtype is not None and value.is_floating_point():
        kwargs["dtype"] = dtype
    return value.to(**kwargs)


def _tensor_on_device_dtype(
    value,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> bool:
    if value is None or not isinstance(value, torch.Tensor):
        return True
    if value.device != device:
        return False
    if dtype is not None and value.is_floating_point() and value.dtype != dtype:
        return False
    return True


def move_source_unit_batch(
    batch: SourceUnitBatch,
    *,
    device: torch.device,
) -> SourceUnitBatch:
    if all(
        _tensor_on_device_dtype(value, device=device)
        for value in (
            batch.content_units,
            batch.source_runlen_src,
            batch.unit_anchor_base,
            batch.unit_mask,
            batch.edge_cue,
            batch.open_run_mask,
            batch.sealed_mask,
            batch.sep_hint,
            batch.boundary_confidence,
        )
    ):
        return batch
    return SourceUnitBatch(
        content_units=_move_tensor(batch.content_units, device=device),
        source_runlen_src=_move_tensor(batch.source_runlen_src, device=device),
        unit_anchor_base=_move_tensor(batch.unit_anchor_base, device=device),
        unit_mask=_move_tensor(batch.unit_mask, device=device),
        edge_cue=_move_tensor(batch.edge_cue, device=device),
        open_run_mask=_move_tensor(batch.open_run_mask, device=device),
        sealed_mask=_move_tensor(batch.sealed_mask, device=device),
        sep_hint=_move_tensor(batch.sep_hint, device=device),
        boundary_confidence=_move_tensor(batch.boundary_confidence, device=device),
    )


def move_reference_duration_memory(
    memory: ReferenceDurationMemory,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> ReferenceDurationMemory:
    if all(
        _tensor_on_device_dtype(value, device=device, dtype=dtype)
        for value in (
            memory.global_rate,
            memory.role_keys,
            memory.role_value,
            memory.role_coverage,
            memory.prompt_role_feat,
            memory.prompt_rel_stretch,
            memory.prompt_mask,
            memory.prompt_role_attention,
            memory.prompt_reconstruction,
            memory.role_summary,
            memory.raw_stats,
            memory.raw_trace,
        )
    ):
        return memory
    return ReferenceDurationMemory(
        global_rate=_move_tensor(memory.global_rate, device=device, dtype=dtype),
        role_keys=_move_tensor(memory.role_keys, device=device, dtype=dtype),
        role_value=_move_tensor(memory.role_value, device=device, dtype=dtype),
        role_coverage=_move_tensor(memory.role_coverage, device=device, dtype=dtype),
        prompt_role_feat=_move_tensor(memory.prompt_role_feat, device=device, dtype=dtype),
        prompt_rel_stretch=_move_tensor(memory.prompt_rel_stretch, device=device, dtype=dtype),
        prompt_mask=_move_tensor(memory.prompt_mask, device=device, dtype=dtype),
        prompt_role_attention=_move_tensor(memory.prompt_role_attention, device=device, dtype=dtype),
        prompt_reconstruction=_move_tensor(memory.prompt_reconstruction, device=device, dtype=dtype),
        role_summary=_move_tensor(memory.role_summary, device=device, dtype=dtype),
        raw_stats=_move_tensor(memory.raw_stats, device=device, dtype=dtype),
        raw_trace=_move_tensor(memory.raw_trace, device=device, dtype=dtype),
    )


def ensure_reference_duration_memory_batch(
    memory: ReferenceDurationMemory,
    *,
    batch_size: int,
) -> ReferenceDurationMemory:
    current_batch = int(memory.global_rate.size(0))
    if current_batch == batch_size:
        return memory
    if current_batch != 1:
        raise ValueError(
            f"ReferenceDurationMemory batch mismatch: source_batch={batch_size}, ref_batch={current_batch}."
        )

    def _expand(value):
        if value is None or not isinstance(value, torch.Tensor):
            return value
        if value.dim() <= 0 or value.size(0) != 1:
            return value
        repeat_dims = [batch_size] + [1] * (value.dim() - 1)
        return value.repeat(*repeat_dims)

    return ReferenceDurationMemory(
        global_rate=_expand(memory.global_rate),
        role_keys=memory.role_keys,
        role_value=_expand(memory.role_value),
        role_coverage=_expand(memory.role_coverage),
        prompt_role_feat=_expand(memory.prompt_role_feat),
        prompt_rel_stretch=_expand(memory.prompt_rel_stretch),
        prompt_mask=_expand(memory.prompt_mask),
        prompt_role_attention=_expand(memory.prompt_role_attention),
        prompt_reconstruction=_expand(memory.prompt_reconstruction),
        role_summary=_expand(memory.role_summary),
        raw_stats=_expand(memory.raw_stats),
        raw_trace=_expand(memory.raw_trace),
    )


def move_duration_runtime_state(
    state: DurationRuntimeState | None,
    *,
    device: torch.device,
) -> DurationRuntimeState | None:
    if state is None:
        return None
    if all(
        _tensor_on_device_dtype(value, device=device)
        for value in (
            state.committed_units,
            state.cumulative_pred_frames,
            state.cumulative_tgt_frames,
            state.cached_duration_exec,
        )
    ):
        return state
    return replace(
        state,
        committed_units=_move_tensor(state.committed_units, device=device),
        cumulative_pred_frames=_move_tensor(state.cumulative_pred_frames, device=device),
        cumulative_tgt_frames=_move_tensor(state.cumulative_tgt_frames, device=device),
        cached_duration_exec=_move_tensor(state.cached_duration_exec, device=device),
    )
