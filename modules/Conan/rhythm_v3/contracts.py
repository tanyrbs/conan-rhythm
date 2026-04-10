from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from modules.Conan.rhythm.frame_plan import RhythmFramePlan


@dataclass
class SourceUnitBatch:
    content_units: torch.Tensor
    source_duration_obs: torch.Tensor
    unit_anchor_base: torch.Tensor
    unit_mask: torch.Tensor
    sealed_mask: torch.Tensor
    sep_mask: torch.Tensor


DURATION_V3_SOURCE_CACHE_REQUIRED_KEYS = (
    "content_units",
    "source_duration_obs",
)

DURATION_V3_SOURCE_CACHE_OPTIONAL_KEYS = (
    "unit_mask",
    "sealed_mask",
    "sep_mask",
    "unit_anchor_base",
)

DURATION_V3_SOURCE_CACHE_KEYS = (
    DURATION_V3_SOURCE_CACHE_REQUIRED_KEYS
    + DURATION_V3_SOURCE_CACHE_OPTIONAL_KEYS
)


def collect_duration_v3_source_cache(source, *, prefix: str = "") -> dict[str, torch.Tensor] | None:
    payload: dict[str, torch.Tensor] = {}
    for key in DURATION_V3_SOURCE_CACHE_KEYS:
        source_key = f"{prefix}{key}"
        if isinstance(source, Mapping):
            value = source.get(source_key)
        else:
            value = getattr(source, source_key, None)
        if value is not None:
            payload[key] = value
    if any(key not in payload for key in DURATION_V3_SOURCE_CACHE_REQUIRED_KEYS):
        return None
    return payload


def export_duration_v3_source_cache(batch: SourceUnitBatch) -> dict[str, torch.Tensor]:
    payload = collect_duration_v3_source_cache(batch)
    if payload is None:
        raise ValueError("Duration-v3 source batch is missing required cache fields.")
    return payload


@dataclass
class StructuredDurationOperatorMemory:
    operator_coeff: torch.Tensor


@dataclass
class StructuredCoarseDurationMemory:
    coarse_profile: torch.Tensor


@dataclass
class PromptConditioningEvidence:
    prompt_basis_activation: Optional[torch.Tensor] = None
    prompt_random_target: Optional[torch.Tensor] = None
    prompt_mask: Optional[torch.Tensor] = None
    prompt_fit_mask: Optional[torch.Tensor] = None
    prompt_eval_mask: Optional[torch.Tensor] = None
    prompt_operator_fit: Optional[torch.Tensor] = None
    prompt_operator_cv_fit: Optional[torch.Tensor] = None
    prompt_log_base: Optional[torch.Tensor] = None
    prompt_log_duration: Optional[torch.Tensor] = None
    prompt_log_residual: Optional[torch.Tensor] = None
    prompt_coarse_fit: Optional[torch.Tensor] = None


@dataclass
class ReferenceDurationMemory:
    global_rate: torch.Tensor
    operator: StructuredDurationOperatorMemory
    coarse: StructuredCoarseDurationMemory | None = None
    prompt: Optional[PromptConditioningEvidence] = None

    @property
    def global_stretch(self) -> torch.Tensor:
        return self.global_rate

    @property
    def operator_coeff(self) -> torch.Tensor:
        return self.operator.operator_coeff

    @property
    def coarse_profile(self) -> Optional[torch.Tensor]:
        return None if self.coarse is None else self.coarse.coarse_profile

    @property
    def prompt_basis_activation(self) -> Optional[torch.Tensor]:
        return None if self.prompt is None else self.prompt.prompt_basis_activation

    @property
    def prompt_random_target(self) -> Optional[torch.Tensor]:
        return None if self.prompt is None else self.prompt.prompt_random_target

    @property
    def prompt_mask(self) -> Optional[torch.Tensor]:
        return None if self.prompt is None else self.prompt.prompt_mask

    @property
    def prompt_operator_fit(self) -> Optional[torch.Tensor]:
        return None if self.prompt is None else self.prompt.prompt_operator_fit

    @property
    def prompt_operator_cv_fit(self) -> Optional[torch.Tensor]:
        return None if self.prompt is None else self.prompt.prompt_operator_cv_fit

    @property
    def prompt_fit_mask(self) -> Optional[torch.Tensor]:
        return None if self.prompt is None else self.prompt.prompt_fit_mask

    @property
    def prompt_eval_mask(self) -> Optional[torch.Tensor]:
        return None if self.prompt is None else self.prompt.prompt_eval_mask

    @property
    def prompt_log_base(self) -> Optional[torch.Tensor]:
        return None if self.prompt is None else self.prompt.prompt_log_base

    @property
    def prompt_log_duration(self) -> Optional[torch.Tensor]:
        return None if self.prompt is None else self.prompt.prompt_log_duration

    @property
    def prompt_log_residual(self) -> Optional[torch.Tensor]:
        return None if self.prompt is None else self.prompt.prompt_log_residual

    @property
    def prompt_coarse_fit(self) -> Optional[torch.Tensor]:
        return None if self.prompt is None else self.prompt.prompt_coarse_fit


@dataclass
class DurationRuntimeState:
    committed_units: torch.Tensor
    rounding_residual: torch.Tensor
    cached_duration_exec: Optional[torch.Tensor] = None

    @property
    def commit_frontier(self) -> torch.Tensor:
        return self.committed_units.long()

    @property
    def clock_delta(self) -> torch.Tensor:
        return self.rounding_residual.new_zeros(self.rounding_residual.shape)

    @property
    def backlog(self) -> torch.Tensor:
        return self.rounding_residual.new_zeros(self.rounding_residual.shape)


@dataclass
class DurationExecution:
    unit_logstretch: torch.Tensor
    unit_duration_exec: torch.Tensor
    basis_activation: torch.Tensor
    commit_mask: torch.Tensor
    next_state: DurationRuntimeState
    coarse_response: Optional[torch.Tensor] = None
    local_response: Optional[torch.Tensor] = None
    frame_plan: Optional["RhythmFramePlan"] = None

    @property
    def speech_duration_exec(self) -> torch.Tensor:
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
            batch.source_duration_obs,
            batch.unit_anchor_base,
            batch.unit_mask,
            batch.sealed_mask,
            batch.sep_mask,
        )
    ):
        return batch
    return SourceUnitBatch(
        content_units=_move_tensor(batch.content_units, device=device),
        source_duration_obs=_move_tensor(batch.source_duration_obs, device=device),
        unit_anchor_base=_move_tensor(batch.unit_anchor_base, device=device),
        unit_mask=_move_tensor(batch.unit_mask, device=device),
        sealed_mask=_move_tensor(batch.sealed_mask, device=device),
        sep_mask=_move_tensor(batch.sep_mask, device=device),
    )


def move_structured_duration_operator_memory(
    operator: StructuredDurationOperatorMemory,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> StructuredDurationOperatorMemory:
    return StructuredDurationOperatorMemory(
        operator_coeff=_move_tensor(operator.operator_coeff, device=device, dtype=dtype),
    )


def move_structured_coarse_duration_memory(
    coarse: StructuredCoarseDurationMemory | None,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> StructuredCoarseDurationMemory | None:
    if coarse is None:
        return None
    return StructuredCoarseDurationMemory(
        coarse_profile=_move_tensor(coarse.coarse_profile, device=device, dtype=dtype),
    )


def move_prompt_conditioning_evidence(
    prompt: PromptConditioningEvidence | None,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> PromptConditioningEvidence | None:
    if prompt is None:
        return None
    return PromptConditioningEvidence(
        prompt_basis_activation=_move_tensor(prompt.prompt_basis_activation, device=device, dtype=dtype),
        prompt_random_target=_move_tensor(prompt.prompt_random_target, device=device, dtype=dtype),
        prompt_mask=_move_tensor(prompt.prompt_mask, device=device, dtype=dtype),
        prompt_fit_mask=_move_tensor(prompt.prompt_fit_mask, device=device, dtype=dtype),
        prompt_eval_mask=_move_tensor(prompt.prompt_eval_mask, device=device, dtype=dtype),
        prompt_operator_fit=_move_tensor(prompt.prompt_operator_fit, device=device, dtype=dtype),
        prompt_operator_cv_fit=_move_tensor(prompt.prompt_operator_cv_fit, device=device, dtype=dtype),
        prompt_log_base=_move_tensor(prompt.prompt_log_base, device=device, dtype=dtype),
        prompt_log_duration=_move_tensor(prompt.prompt_log_duration, device=device, dtype=dtype),
        prompt_log_residual=_move_tensor(prompt.prompt_log_residual, device=device, dtype=dtype),
        prompt_coarse_fit=_move_tensor(prompt.prompt_coarse_fit, device=device, dtype=dtype),
    )


def move_reference_duration_memory(
    memory: ReferenceDurationMemory,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> ReferenceDurationMemory:
    validate_reference_duration_memory(memory)
    if all(
        _tensor_on_device_dtype(value, device=device, dtype=dtype)
        for value in (
            memory.global_rate,
            memory.operator_coeff,
            memory.coarse_profile,
            memory.prompt_basis_activation,
            memory.prompt_random_target,
            memory.prompt_mask,
            memory.prompt_fit_mask,
            memory.prompt_eval_mask,
            memory.prompt_operator_fit,
            memory.prompt_operator_cv_fit,
            memory.prompt_log_base,
            memory.prompt_log_duration,
            memory.prompt_log_residual,
            memory.prompt_coarse_fit,
        )
    ):
        return memory
    return ReferenceDurationMemory(
        global_rate=_move_tensor(memory.global_rate, device=device, dtype=dtype),
        operator=move_structured_duration_operator_memory(memory.operator, device=device, dtype=dtype),
        coarse=move_structured_coarse_duration_memory(memory.coarse, device=device, dtype=dtype),
        prompt=move_prompt_conditioning_evidence(memory.prompt, device=device, dtype=dtype),
    )


def validate_structured_duration_operator_memory(
    operator: StructuredDurationOperatorMemory,
    *,
    batch_size: int,
) -> StructuredDurationOperatorMemory:
    coeff = operator.operator_coeff
    if not isinstance(coeff, torch.Tensor) or coeff.dim() != 2:
        raise ValueError(
            "StructuredDurationOperatorMemory.operator_coeff must have shape [B, K], "
            f"got {getattr(coeff, 'shape', None)}"
        )
    if coeff.size(0) != batch_size:
        raise ValueError(
            "StructuredDurationOperatorMemory.operator_coeff batch mismatch: "
            f"expected {batch_size}, got {tuple(coeff.shape)}"
        )
    return operator


def validate_structured_coarse_duration_memory(
    coarse: StructuredCoarseDurationMemory | None,
    *,
    batch_size: int,
) -> StructuredCoarseDurationMemory | None:
    if coarse is None:
        return None
    profile = coarse.coarse_profile
    if not isinstance(profile, torch.Tensor) or profile.dim() != 2:
        raise ValueError(
            "StructuredCoarseDurationMemory.coarse_profile must have shape [B, M], "
            f"got {getattr(profile, 'shape', None)}"
        )
    if profile.size(0) != batch_size:
        raise ValueError(
            "StructuredCoarseDurationMemory.coarse_profile batch mismatch: "
            f"expected {batch_size}, got {tuple(profile.shape)}"
        )
    return coarse


def validate_prompt_conditioning_evidence(
    prompt: PromptConditioningEvidence | None,
    *,
    batch_size: int,
    operator_rank: int,
) -> PromptConditioningEvidence | None:
    if prompt is None:
        return None

    def _check_batch(name: str, value: torch.Tensor | None, *, dims: int | None = None):
        if value is None:
            return
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"PromptConditioningEvidence.{name} must be a tensor when provided, got {type(value)!r}")
        if dims is not None and value.dim() != dims:
            raise ValueError(
                f"PromptConditioningEvidence.{name} must be rank-{dims}, got shape={tuple(value.shape)}"
            )
        if value.dim() > 0 and value.size(0) != batch_size:
            raise ValueError(
                f"PromptConditioningEvidence.{name} batch mismatch: expected {batch_size}, got {tuple(value.shape)}"
            )

    _check_batch("prompt_basis_activation", prompt.prompt_basis_activation, dims=3)
    _check_batch("prompt_random_target", prompt.prompt_random_target, dims=2)
    _check_batch("prompt_mask", prompt.prompt_mask, dims=2)
    _check_batch("prompt_fit_mask", prompt.prompt_fit_mask, dims=2)
    _check_batch("prompt_eval_mask", prompt.prompt_eval_mask, dims=2)
    _check_batch("prompt_operator_fit", prompt.prompt_operator_fit, dims=2)
    _check_batch("prompt_operator_cv_fit", prompt.prompt_operator_cv_fit, dims=2)
    _check_batch("prompt_log_base", prompt.prompt_log_base, dims=2)
    _check_batch("prompt_log_duration", prompt.prompt_log_duration, dims=2)
    _check_batch("prompt_log_residual", prompt.prompt_log_residual, dims=2)
    _check_batch("prompt_coarse_fit", prompt.prompt_coarse_fit, dims=2)

    if prompt.prompt_basis_activation is not None and prompt.prompt_basis_activation.size(-1) != operator_rank:
        raise ValueError(
            "PromptConditioningEvidence.prompt_basis_activation operator-rank mismatch: "
            f"expected last dim {operator_rank}, got {tuple(prompt.prompt_basis_activation.shape)}"
        )
    if prompt.prompt_basis_activation is not None and prompt.prompt_mask is not None:
        if tuple(prompt.prompt_basis_activation.shape[:2]) != tuple(prompt.prompt_mask.shape):
            raise ValueError(
                "PromptConditioningEvidence.prompt_basis_activation/prompt_mask shape mismatch: "
                f"{tuple(prompt.prompt_basis_activation.shape[:2])} vs {tuple(prompt.prompt_mask.shape)}"
            )
    if prompt.prompt_basis_activation is not None and prompt.prompt_random_target is not None:
        if tuple(prompt.prompt_basis_activation.shape[:2]) != tuple(prompt.prompt_random_target.shape):
            raise ValueError(
                "PromptConditioningEvidence.prompt_basis_activation/prompt_random_target shape mismatch: "
                f"{tuple(prompt.prompt_basis_activation.shape[:2])} vs {tuple(prompt.prompt_random_target.shape)}"
            )
    if prompt.prompt_random_target is not None and prompt.prompt_mask is not None:
        if tuple(prompt.prompt_random_target.shape) != tuple(prompt.prompt_mask.shape):
            raise ValueError(
                "PromptConditioningEvidence.prompt_random_target/prompt_mask shape mismatch: "
                f"{tuple(prompt.prompt_random_target.shape)} vs {tuple(prompt.prompt_mask.shape)}"
            )
    for name, value in (
        ("prompt_fit_mask", prompt.prompt_fit_mask),
        ("prompt_eval_mask", prompt.prompt_eval_mask),
    ):
        if value is not None and prompt.prompt_mask is not None:
            if tuple(value.shape) != tuple(prompt.prompt_mask.shape):
                raise ValueError(
                    f"PromptConditioningEvidence.{name}/prompt_mask shape mismatch: "
                    f"{tuple(value.shape)} vs {tuple(prompt.prompt_mask.shape)}"
                )
    if prompt.prompt_operator_fit is not None and prompt.prompt_random_target is not None:
        if tuple(prompt.prompt_operator_fit.shape) != tuple(prompt.prompt_random_target.shape):
            raise ValueError(
                "PromptConditioningEvidence.prompt_operator_fit/prompt_random_target shape mismatch: "
                f"{tuple(prompt.prompt_operator_fit.shape)} vs {tuple(prompt.prompt_random_target.shape)}"
            )
    if prompt.prompt_operator_cv_fit is not None and prompt.prompt_random_target is not None:
        if tuple(prompt.prompt_operator_cv_fit.shape) != tuple(prompt.prompt_random_target.shape):
            raise ValueError(
                "PromptConditioningEvidence.prompt_operator_cv_fit/prompt_random_target shape mismatch: "
                f"{tuple(prompt.prompt_operator_cv_fit.shape)} vs {tuple(prompt.prompt_random_target.shape)}"
            )
    for name, value in (
        ("prompt_log_base", prompt.prompt_log_base),
        ("prompt_log_duration", prompt.prompt_log_duration),
        ("prompt_log_residual", prompt.prompt_log_residual),
        ("prompt_coarse_fit", prompt.prompt_coarse_fit),
        ("prompt_operator_fit", prompt.prompt_operator_fit),
        ("prompt_operator_cv_fit", prompt.prompt_operator_cv_fit),
        ("prompt_random_target", prompt.prompt_random_target),
    ):
        if value is not None and prompt.prompt_mask is not None:
            if tuple(value.shape) != tuple(prompt.prompt_mask.shape):
                raise ValueError(
                    f"PromptConditioningEvidence.{name}/prompt_mask shape mismatch: "
                    f"{tuple(value.shape)} vs {tuple(prompt.prompt_mask.shape)}"
                )
    return prompt


def validate_reference_duration_memory(
    memory: ReferenceDurationMemory,
) -> ReferenceDurationMemory:
    if not isinstance(memory.global_rate, torch.Tensor) or memory.global_rate.dim() != 2 or memory.global_rate.size(1) != 1:
        raise ValueError(
            f"ReferenceDurationMemory.global_rate must have shape [B, 1], got {getattr(memory.global_rate, 'shape', None)}"
        )
    batch_size = int(memory.global_rate.size(0))
    validate_structured_duration_operator_memory(memory.operator, batch_size=batch_size)
    validate_structured_coarse_duration_memory(memory.coarse, batch_size=batch_size)
    validate_prompt_conditioning_evidence(
        memory.prompt,
        batch_size=batch_size,
        operator_rank=int(memory.operator_coeff.size(1)),
    )
    return memory


def ensure_reference_duration_memory_batch(
    memory: ReferenceDurationMemory,
    *,
    batch_size: int,
) -> ReferenceDurationMemory:
    validate_reference_duration_memory(memory)
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
        return value.expand(batch_size, *value.shape[1:])

    expanded = ReferenceDurationMemory(
        global_rate=_expand(memory.global_rate),
        operator=StructuredDurationOperatorMemory(
            operator_coeff=_expand(memory.operator_coeff),
        ),
        coarse=(
            None
            if memory.coarse is None
            else StructuredCoarseDurationMemory(
                coarse_profile=_expand(memory.coarse_profile),
            )
        ),
        prompt=(
            None
            if memory.prompt is None
            else PromptConditioningEvidence(
                prompt_basis_activation=_expand(memory.prompt_basis_activation),
                prompt_random_target=_expand(memory.prompt_random_target),
                prompt_mask=_expand(memory.prompt_mask),
                prompt_fit_mask=_expand(memory.prompt_fit_mask),
                prompt_eval_mask=_expand(memory.prompt_eval_mask),
                prompt_operator_fit=_expand(memory.prompt_operator_fit),
                prompt_operator_cv_fit=_expand(memory.prompt_operator_cv_fit),
                prompt_log_base=_expand(memory.prompt_log_base),
                prompt_log_duration=_expand(memory.prompt_log_duration),
                prompt_log_residual=_expand(memory.prompt_log_residual),
                prompt_coarse_fit=_expand(memory.prompt_coarse_fit),
            )
        ),
    )
    return validate_reference_duration_memory(expanded)


def move_duration_runtime_state(
    state: DurationRuntimeState | None,
    *,
    device: torch.device,
) -> DurationRuntimeState | None:
    if state is None:
        return None
    return replace(
        state,
        committed_units=_move_tensor(state.committed_units, device=device),
        rounding_residual=_move_tensor(state.rounding_residual, device=device),
        cached_duration_exec=_move_tensor(state.cached_duration_exec, device=device),
    )


def ensure_duration_runtime_state_batch(
    state: DurationRuntimeState | None,
    *,
    batch_size: int,
) -> DurationRuntimeState | None:
    if state is None:
        return None
    current_batch = int(state.committed_units.size(0))
    if current_batch != batch_size:
        raise ValueError(
            f"DurationRuntimeState batch mismatch: source_batch={batch_size}, state_batch={current_batch}."
        )
    for name in ("committed_units", "rounding_residual", "cached_duration_exec"):
        value = getattr(state, name)
        if value is None or not isinstance(value, torch.Tensor):
            continue
        if value.size(0) != batch_size:
            raise ValueError(
                f"DurationRuntimeState.{name} batch mismatch: source_batch={batch_size}, tensor_shape={tuple(value.shape)}."
            )
    if state.rounding_residual.dim() != 2 or state.rounding_residual.size(1) != 1:
        raise ValueError(
            f"DurationRuntimeState.rounding_residual must be rank-2 [B, 1], got shape={tuple(state.rounding_residual.shape)}."
        )
    return state
