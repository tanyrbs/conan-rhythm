from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Optional, TYPE_CHECKING

import torch
from .unitizer import StreamingUnitizerRowState, StreamingUnitizerState

if TYPE_CHECKING:
    from .frame_plan import RhythmFramePlan


@dataclass
class SourceUnitBatch:
    content_units: torch.Tensor
    source_duration_obs: torch.Tensor
    unit_anchor_base: torch.Tensor
    unit_mask: torch.Tensor
    sealed_mask: torch.Tensor
    sep_mask: torch.Tensor
    unit_rate_log_base: Optional[torch.Tensor] = None
    source_silence_mask: Optional[torch.Tensor] = None
    source_boundary_cue: Optional[torch.Tensor] = None
    source_run_stability: Optional[torch.Tensor] = None
    phrase_group_index: Optional[torch.Tensor] = None
    phrase_group_pos: Optional[torch.Tensor] = None
    phrase_final_mask: Optional[torch.Tensor] = None


DURATION_V3_SOURCE_CACHE_REQUIRED_KEYS = (
    "content_units",
    "source_duration_obs",
)

DURATION_V3_SOURCE_CACHE_OPTIONAL_KEYS = (
    "unit_mask",
    "sealed_mask",
    "sep_mask",
    "unit_anchor_base",
    "unit_rate_log_base",
    "source_silence_mask",
    "source_boundary_cue",
    "source_run_stability",
    "phrase_group_index",
    "phrase_group_pos",
    "phrase_final_mask",
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
class StructuredProgressDurationMemory:
    progress_profile: torch.Tensor


@dataclass
class StructuredDetectorDurationMemory:
    detector_coeff: torch.Tensor


@dataclass
class StructuredRoleDurationMemory:
    role_value: torch.Tensor
    role_var: torch.Tensor
    role_coverage: torch.Tensor


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
    prompt_progress_fit: Optional[torch.Tensor] = None
    prompt_operator_support: Optional[torch.Tensor] = None
    prompt_operator_condition_number: Optional[torch.Tensor] = None
    prompt_short_fallback: Optional[torch.Tensor] = None
    prompt_operator_coeff_norm: Optional[torch.Tensor] = None
    prompt_detector_fit: Optional[torch.Tensor] = None
    prompt_detector_support: Optional[torch.Tensor] = None
    prompt_detector_condition_number: Optional[torch.Tensor] = None
    prompt_detector_coeff_norm: Optional[torch.Tensor] = None
    prompt_role_attn: Optional[torch.Tensor] = None
    prompt_role_fit: Optional[torch.Tensor] = None

@dataclass
class ReferenceDurationMemory:
    global_rate: torch.Tensor
    operator: StructuredDurationOperatorMemory
    progress: StructuredProgressDurationMemory | None = None
    detector: StructuredDetectorDurationMemory | None = None
    role: StructuredRoleDurationMemory | None = None
    prompt: Optional[PromptConditioningEvidence] = None
    summary_state: Optional[torch.Tensor] = None
    spk_embed: Optional[torch.Tensor] = None
    prompt_valid_mask: Optional[torch.Tensor] = None
    prompt_speech_mask: Optional[torch.Tensor] = None

    @property
    def global_stretch(self) -> torch.Tensor:
        return self.global_rate

    @property
    def operator_coeff(self) -> torch.Tensor:
        if self.operator is not None:
            return self.operator.operator_coeff
        if isinstance(self.summary_state, torch.Tensor):
            return self.summary_state
        raise ValueError("ReferenceDurationMemory has neither operator_coeff nor summary_state.")

    @property
    def summary_vector(self) -> Optional[torch.Tensor]:
        return self.summary_state

    @property
    def progress_profile(self) -> Optional[torch.Tensor]:
        return None if self.progress is None else self.progress.progress_profile

    @property
    def detector_coeff(self) -> Optional[torch.Tensor]:
        return None if self.detector is None else self.detector.detector_coeff

    @property
    def role_value(self) -> Optional[torch.Tensor]:
        return None if self.role is None else self.role.role_value

    @property
    def role_var(self) -> Optional[torch.Tensor]:
        return None if self.role is None else self.role.role_var

    @property
    def role_coverage(self) -> Optional[torch.Tensor]:
        return None if self.role is None else self.role.role_coverage

    @property
    def summary_value(self) -> Optional[torch.Tensor]:
        return self.role_value

    @property
    def summary_var(self) -> Optional[torch.Tensor]:
        return self.role_var

    @property
    def summary_coverage(self) -> Optional[torch.Tensor]:
        return self.role_coverage

    @property
    def prompt_basis_activation(self) -> Optional[torch.Tensor]:
        return None if self.prompt is None else self.prompt.prompt_basis_activation

    @property
    def prompt_random_target(self) -> Optional[torch.Tensor]:
        return None if self.prompt is None else self.prompt.prompt_random_target

    @property
    def prompt_mask(self) -> Optional[torch.Tensor]:
        if isinstance(self.prompt_valid_mask, torch.Tensor):
            return self.prompt_valid_mask
        return None if self.prompt is None else self.prompt.prompt_mask

    @property
    def prompt_speech(self) -> Optional[torch.Tensor]:
        return self.prompt_speech_mask

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
    def prompt_progress_fit(self) -> Optional[torch.Tensor]:
        return None if self.prompt is None else self.prompt.prompt_progress_fit

    @property
    def prompt_operator_support(self) -> Optional[torch.Tensor]:
        return None if self.prompt is None else self.prompt.prompt_operator_support

    @property
    def prompt_operator_condition_number(self) -> Optional[torch.Tensor]:
        return None if self.prompt is None else self.prompt.prompt_operator_condition_number

    @property
    def prompt_short_fallback(self) -> Optional[torch.Tensor]:
        return None if self.prompt is None else self.prompt.prompt_short_fallback

    @property
    def prompt_operator_coeff_norm(self) -> Optional[torch.Tensor]:
        return None if self.prompt is None else self.prompt.prompt_operator_coeff_norm

    @property
    def prompt_detector_fit(self) -> Optional[torch.Tensor]:
        return None if self.prompt is None else self.prompt.prompt_detector_fit

    @property
    def prompt_detector_support(self) -> Optional[torch.Tensor]:
        return None if self.prompt is None else self.prompt.prompt_detector_support

    @property
    def prompt_detector_condition_number(self) -> Optional[torch.Tensor]:
        return None if self.prompt is None else self.prompt.prompt_detector_condition_number

    @property
    def prompt_detector_coeff_norm(self) -> Optional[torch.Tensor]:
        return None if self.prompt is None else self.prompt.prompt_detector_coeff_norm

    @property
    def prompt_role_attn(self) -> Optional[torch.Tensor]:
        return None if self.prompt is None else self.prompt.prompt_role_attn

    @property
    def prompt_role_fit(self) -> Optional[torch.Tensor]:
        return None if self.prompt is None else self.prompt.prompt_role_fit

    @property
    def prompt_summary_attn(self) -> Optional[torch.Tensor]:
        return self.prompt_role_attn

    @property
    def prompt_summary_fit(self) -> Optional[torch.Tensor]:
        return self.prompt_role_fit


@dataclass
class DurationRuntimeState:
    committed_units: torch.Tensor
    rounding_residual: torch.Tensor
    prefix_unit_offset: torch.Tensor
    cached_duration_exec: Optional[torch.Tensor] = None
    local_rate_ema: Optional[torch.Tensor] = None
    since_last_boundary: Optional[torch.Tensor] = None
    frontend_state: Optional[StreamingUnitizerState] = None
    consumed_content_steps: Optional[torch.Tensor] = None

    @property
    def commit_frontier(self) -> torch.Tensor:
        return self.committed_units.long()

    @property
    def clock_delta(self) -> torch.Tensor:
        return self.prefix_unit_offset.float()

    @property
    def backlog(self) -> torch.Tensor:
        return self.prefix_unit_offset.float().clamp_min(0.0)


@dataclass
class DurationExecution:
    unit_logstretch: torch.Tensor
    unit_duration_exec: torch.Tensor
    basis_activation: torch.Tensor
    commit_mask: torch.Tensor
    next_state: DurationRuntimeState
    progress_response: Optional[torch.Tensor] = None
    detector_response: Optional[torch.Tensor] = None
    local_response: Optional[torch.Tensor] = None
    role_attn_unit: Optional[torch.Tensor] = None
    role_value_unit: Optional[torch.Tensor] = None
    role_var_unit: Optional[torch.Tensor] = None
    role_conf_unit: Optional[torch.Tensor] = None
    unit_logstretch_raw: Optional[torch.Tensor] = None
    unit_duration_raw: Optional[torch.Tensor] = None
    frame_plan: Optional["RhythmFramePlan"] = None
    global_bias_scalar: Optional[torch.Tensor] = None
    global_shift_analytic: Optional[torch.Tensor] = None
    coarse_logstretch: Optional[torch.Tensor] = None
    coarse_path_logstretch: Optional[torch.Tensor] = None
    coarse_correction: Optional[torch.Tensor] = None
    local_residual: Optional[torch.Tensor] = None
    speech_pred: Optional[torch.Tensor] = None
    silence_pred: Optional[torch.Tensor] = None
    source_rate_seq: Optional[torch.Tensor] = None
    source_prefix_summary: Optional[torch.Tensor] = None
    g_ref: Optional[torch.Tensor] = None
    g_src_prefix: Optional[torch.Tensor] = None
    eval_mode: Optional[str] = None
    prompt_speech_ratio: Optional[torch.Tensor] = None
    prompt_valid_len: Optional[torch.Tensor] = None
    prefix_unit_offset: Optional[torch.Tensor] = None
    projector_rounding_residual: Optional[torch.Tensor] = None
    projector_budget_pos_used: Optional[torch.Tensor] = None
    projector_budget_neg_used: Optional[torch.Tensor] = None
    projector_budget_hit_pos: Optional[torch.Tensor] = None
    projector_budget_hit_neg: Optional[torch.Tensor] = None
    projector_boundary_hit: Optional[torch.Tensor] = None
    projector_boundary_decay_applied: Optional[torch.Tensor] = None
    projector_since_last_boundary: Optional[torch.Tensor] = None

    @property
    def speech_duration_exec(self) -> torch.Tensor:
        return self.unit_duration_exec

    @property
    def commit_frontier(self) -> torch.Tensor:
        return self.next_state.commit_frontier

    @property
    def planner(self):
        return None

    @property
    def summary_attn_unit(self) -> Optional[torch.Tensor]:
        return self.role_attn_unit

    @property
    def summary_value_unit(self) -> Optional[torch.Tensor]:
        return self.role_value_unit

    @property
    def summary_var_unit(self) -> Optional[torch.Tensor]:
        return self.role_var_unit

    @property
    def summary_conf_unit(self) -> Optional[torch.Tensor]:
        return self.role_conf_unit

    @property
    def global_bias(self) -> Optional[torch.Tensor]:
        return self.global_bias_scalar


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
            batch.unit_rate_log_base,
            batch.unit_mask,
            batch.sealed_mask,
            batch.sep_mask,
            batch.source_silence_mask,
            batch.source_boundary_cue,
            batch.source_run_stability,
            batch.phrase_group_index,
            batch.phrase_group_pos,
            batch.phrase_final_mask,
        )
    ):
        return batch
    return SourceUnitBatch(
        content_units=_move_tensor(batch.content_units, device=device),
        source_duration_obs=_move_tensor(batch.source_duration_obs, device=device),
        unit_anchor_base=_move_tensor(batch.unit_anchor_base, device=device),
        unit_rate_log_base=_move_tensor(batch.unit_rate_log_base, device=device),
        unit_mask=_move_tensor(batch.unit_mask, device=device),
        sealed_mask=_move_tensor(batch.sealed_mask, device=device),
        sep_mask=_move_tensor(batch.sep_mask, device=device),
        source_silence_mask=_move_tensor(batch.source_silence_mask, device=device),
        source_boundary_cue=_move_tensor(batch.source_boundary_cue, device=device),
        source_run_stability=_move_tensor(batch.source_run_stability, device=device),
        phrase_group_index=_move_tensor(batch.phrase_group_index, device=device),
        phrase_group_pos=_move_tensor(batch.phrase_group_pos, device=device),
        phrase_final_mask=_move_tensor(batch.phrase_final_mask, device=device),
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


def move_structured_progress_duration_memory(
    progress: StructuredProgressDurationMemory | None,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> StructuredProgressDurationMemory | None:
    if progress is None:
        return None
    return StructuredProgressDurationMemory(
        progress_profile=_move_tensor(progress.progress_profile, device=device, dtype=dtype),
    )


def move_structured_detector_duration_memory(
    detector: StructuredDetectorDurationMemory | None,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> StructuredDetectorDurationMemory | None:
    if detector is None:
        return None
    return StructuredDetectorDurationMemory(
        detector_coeff=_move_tensor(detector.detector_coeff, device=device, dtype=dtype),
    )


def move_structured_role_duration_memory(
    role: StructuredRoleDurationMemory | None,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> StructuredRoleDurationMemory | None:
    if role is None:
        return None
    return StructuredRoleDurationMemory(
        role_value=_move_tensor(role.role_value, device=device, dtype=dtype),
        role_var=_move_tensor(role.role_var, device=device, dtype=dtype),
        role_coverage=_move_tensor(role.role_coverage, device=device, dtype=dtype),
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
        prompt_progress_fit=_move_tensor(prompt.prompt_progress_fit, device=device, dtype=dtype),
        prompt_operator_support=_move_tensor(prompt.prompt_operator_support, device=device, dtype=dtype),
        prompt_operator_condition_number=_move_tensor(prompt.prompt_operator_condition_number, device=device, dtype=dtype),
        prompt_short_fallback=_move_tensor(prompt.prompt_short_fallback, device=device, dtype=dtype),
        prompt_operator_coeff_norm=_move_tensor(prompt.prompt_operator_coeff_norm, device=device, dtype=dtype),
        prompt_detector_fit=_move_tensor(prompt.prompt_detector_fit, device=device, dtype=dtype),
        prompt_detector_support=_move_tensor(prompt.prompt_detector_support, device=device, dtype=dtype),
        prompt_detector_condition_number=_move_tensor(prompt.prompt_detector_condition_number, device=device, dtype=dtype),
        prompt_detector_coeff_norm=_move_tensor(prompt.prompt_detector_coeff_norm, device=device, dtype=dtype),
        prompt_role_attn=_move_tensor(prompt.prompt_role_attn, device=device, dtype=dtype),
        prompt_role_fit=_move_tensor(prompt.prompt_role_fit, device=device, dtype=dtype),
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
            memory.summary_state,
            memory.spk_embed,
            memory.prompt_valid_mask,
            memory.prompt_speech_mask,
            memory.operator_coeff,
            memory.progress_profile,
            memory.detector_coeff,
            memory.role_value,
            memory.role_var,
            memory.role_coverage,
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
            memory.prompt_progress_fit,
            memory.prompt.prompt_operator_support if memory.prompt is not None else None,
            memory.prompt.prompt_operator_condition_number if memory.prompt is not None else None,
            memory.prompt.prompt_short_fallback if memory.prompt is not None else None,
            memory.prompt.prompt_operator_coeff_norm if memory.prompt is not None else None,
            memory.prompt.prompt_detector_fit if memory.prompt is not None else None,
            memory.prompt.prompt_detector_support if memory.prompt is not None else None,
            memory.prompt.prompt_detector_condition_number if memory.prompt is not None else None,
            memory.prompt.prompt_detector_coeff_norm if memory.prompt is not None else None,
            memory.prompt.prompt_role_attn if memory.prompt is not None else None,
            memory.prompt.prompt_role_fit if memory.prompt is not None else None,
        )
    ):
        return memory
    return ReferenceDurationMemory(
        global_rate=_move_tensor(memory.global_rate, device=device, dtype=dtype),
        operator=move_structured_duration_operator_memory(memory.operator, device=device, dtype=dtype),
        progress=move_structured_progress_duration_memory(memory.progress, device=device, dtype=dtype),
        detector=move_structured_detector_duration_memory(memory.detector, device=device, dtype=dtype),
        role=move_structured_role_duration_memory(memory.role, device=device, dtype=dtype),
        prompt=move_prompt_conditioning_evidence(memory.prompt, device=device, dtype=dtype),
        summary_state=_move_tensor(memory.summary_state, device=device, dtype=dtype),
        spk_embed=_move_tensor(memory.spk_embed, device=device, dtype=dtype),
        prompt_valid_mask=_move_tensor(memory.prompt_valid_mask, device=device, dtype=dtype),
        prompt_speech_mask=_move_tensor(memory.prompt_speech_mask, device=device, dtype=dtype),
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


def validate_structured_progress_duration_memory(
    progress: StructuredProgressDurationMemory | None,
    *,
    batch_size: int,
) -> StructuredProgressDurationMemory | None:
    if progress is None:
        return None
    profile = progress.progress_profile
    if not isinstance(profile, torch.Tensor) or profile.dim() != 2:
        raise ValueError(
            "StructuredProgressDurationMemory.progress_profile must have shape [B, M], "
            f"got {getattr(profile, 'shape', None)}"
        )
    if profile.size(0) != batch_size:
        raise ValueError(
            "StructuredProgressDurationMemory.progress_profile batch mismatch: "
            f"expected {batch_size}, got {tuple(profile.shape)}"
        )
    return progress


def validate_structured_detector_duration_memory(
    detector: StructuredDetectorDurationMemory | None,
    *,
    batch_size: int,
) -> StructuredDetectorDurationMemory | None:
    if detector is None:
        return None
    coeff = detector.detector_coeff
    if not isinstance(coeff, torch.Tensor) or coeff.dim() != 2:
        raise ValueError(
            "StructuredDetectorDurationMemory.detector_coeff must have shape [B, D], "
            f"got {getattr(coeff, 'shape', None)}"
        )
    if coeff.size(0) != batch_size:
        raise ValueError(
            "StructuredDetectorDurationMemory.detector_coeff batch mismatch: "
            f"expected {batch_size}, got {tuple(coeff.shape)}"
        )
    return detector


def validate_structured_role_duration_memory(
    role: StructuredRoleDurationMemory | None,
    *,
    batch_size: int,
) -> StructuredRoleDurationMemory | None:
    if role is None:
        return None
    for name, value in (
        ("role_value", role.role_value),
        ("role_var", role.role_var),
        ("role_coverage", role.role_coverage),
    ):
        if not isinstance(value, torch.Tensor) or value.dim() != 2:
            raise ValueError(
                f"StructuredRoleDurationMemory.{name} must have shape [B, M], got {getattr(value, 'shape', None)}"
            )
        if value.size(0) != batch_size:
            raise ValueError(
                f"StructuredRoleDurationMemory.{name} batch mismatch: expected {batch_size}, got {tuple(value.shape)}"
            )
    if tuple(role.role_value.shape) != tuple(role.role_var.shape) or tuple(role.role_value.shape) != tuple(role.role_coverage.shape):
        raise ValueError(
            "StructuredRoleDurationMemory role_value/role_var/role_coverage must share the same shape."
        )
    return role


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
    _check_batch("prompt_progress_fit", prompt.prompt_progress_fit, dims=2)
    _check_batch("prompt_operator_support", prompt.prompt_operator_support, dims=2)
    _check_batch("prompt_operator_condition_number", prompt.prompt_operator_condition_number, dims=2)
    _check_batch("prompt_short_fallback", prompt.prompt_short_fallback, dims=2)
    _check_batch("prompt_operator_coeff_norm", prompt.prompt_operator_coeff_norm, dims=2)
    _check_batch("prompt_detector_fit", prompt.prompt_detector_fit, dims=2)
    _check_batch("prompt_detector_support", prompt.prompt_detector_support, dims=2)
    _check_batch("prompt_detector_condition_number", prompt.prompt_detector_condition_number, dims=2)
    _check_batch("prompt_detector_coeff_norm", prompt.prompt_detector_coeff_norm, dims=2)
    _check_batch("prompt_role_attn", prompt.prompt_role_attn, dims=3)
    _check_batch("prompt_role_fit", prompt.prompt_role_fit, dims=2)

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
        ("prompt_progress_fit", prompt.prompt_progress_fit),
        ("prompt_operator_fit", prompt.prompt_operator_fit),
        ("prompt_operator_cv_fit", prompt.prompt_operator_cv_fit),
        ("prompt_detector_fit", prompt.prompt_detector_fit),
        ("prompt_random_target", prompt.prompt_random_target),
        ("prompt_role_fit", prompt.prompt_role_fit),
    ):
        if value is not None and prompt.prompt_mask is not None:
            if tuple(value.shape) != tuple(prompt.prompt_mask.shape):
                raise ValueError(
                    f"PromptConditioningEvidence.{name}/prompt_mask shape mismatch: "
                    f"{tuple(value.shape)} vs {tuple(prompt.prompt_mask.shape)}"
                )
    if prompt.prompt_role_attn is not None and prompt.prompt_mask is not None:
        if tuple(prompt.prompt_role_attn.shape[:2]) != tuple(prompt.prompt_mask.shape):
            raise ValueError(
                "PromptConditioningEvidence.prompt_role_attn/prompt_mask shape mismatch: "
                f"{tuple(prompt.prompt_role_attn.shape[:2])} vs {tuple(prompt.prompt_mask.shape)}"
            )
    for name, value in (
        ("prompt_operator_support", prompt.prompt_operator_support),
        ("prompt_operator_condition_number", prompt.prompt_operator_condition_number),
        ("prompt_short_fallback", prompt.prompt_short_fallback),
        ("prompt_operator_coeff_norm", prompt.prompt_operator_coeff_norm),
        ("prompt_detector_support", prompt.prompt_detector_support),
        ("prompt_detector_condition_number", prompt.prompt_detector_condition_number),
        ("prompt_detector_coeff_norm", prompt.prompt_detector_coeff_norm),
    ):
        if value is not None and value.size(1) != 1:
            raise ValueError(
                f"PromptConditioningEvidence.{name} must have shape [B, 1], got {tuple(value.shape)}"
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
    if memory.summary_state is not None:
        if not isinstance(memory.summary_state, torch.Tensor) or memory.summary_state.dim() != 2:
            raise ValueError(
                f"ReferenceDurationMemory.summary_state must have shape [B, D], got {getattr(memory.summary_state, 'shape', None)}"
            )
        if memory.summary_state.size(0) != batch_size:
            raise ValueError(
                f"ReferenceDurationMemory.summary_state batch mismatch: expected {batch_size}, got {tuple(memory.summary_state.shape)}"
            )
    if memory.spk_embed is not None:
        if not isinstance(memory.spk_embed, torch.Tensor) or memory.spk_embed.dim() != 2:
            raise ValueError(
                f"ReferenceDurationMemory.spk_embed must have shape [B, H], got {getattr(memory.spk_embed, 'shape', None)}"
            )
        if memory.spk_embed.size(0) != batch_size:
            raise ValueError(
                f"ReferenceDurationMemory.spk_embed batch mismatch: expected {batch_size}, got {tuple(memory.spk_embed.shape)}"
            )
    for name, value in (
        ("prompt_valid_mask", memory.prompt_valid_mask),
        ("prompt_speech_mask", memory.prompt_speech_mask),
    ):
        if value is None:
            continue
        if not isinstance(value, torch.Tensor) or value.dim() != 2:
            raise ValueError(
                f"ReferenceDurationMemory.{name} must have shape [B, T], got {getattr(value, 'shape', None)}"
            )
        if value.size(0) != batch_size:
            raise ValueError(
                f"ReferenceDurationMemory.{name} batch mismatch: expected {batch_size}, got {tuple(value.shape)}"
            )
    if (
        isinstance(memory.prompt_valid_mask, torch.Tensor)
        and isinstance(memory.prompt_speech_mask, torch.Tensor)
    ):
        if tuple(memory.prompt_valid_mask.shape) != tuple(memory.prompt_speech_mask.shape):
            raise ValueError(
                "ReferenceDurationMemory.prompt_valid_mask/prompt_speech_mask shape mismatch: "
                f"{tuple(memory.prompt_valid_mask.shape)} vs {tuple(memory.prompt_speech_mask.shape)}"
            )
        if bool((memory.prompt_speech_mask > (memory.prompt_valid_mask + 1.0e-6)).any().item()):
            raise ValueError(
                "ReferenceDurationMemory.prompt_speech_mask must be bounded by prompt_valid_mask."
            )
    validate_structured_duration_operator_memory(memory.operator, batch_size=batch_size)
    validate_structured_progress_duration_memory(memory.progress, batch_size=batch_size)
    validate_structured_detector_duration_memory(memory.detector, batch_size=batch_size)
    validate_structured_role_duration_memory(memory.role, batch_size=batch_size)
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
        progress=(
            None
            if memory.progress is None
            else StructuredProgressDurationMemory(
                progress_profile=_expand(memory.progress_profile),
            )
        ),
        detector=(
            None
            if memory.detector is None
            else StructuredDetectorDurationMemory(
                detector_coeff=_expand(memory.detector_coeff),
            )
        ),
        role=(
            None
            if memory.role is None
            else StructuredRoleDurationMemory(
                role_value=_expand(memory.role_value),
                role_var=_expand(memory.role_var),
                role_coverage=_expand(memory.role_coverage),
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
                prompt_progress_fit=_expand(memory.prompt_progress_fit),
                prompt_operator_support=_expand(memory.prompt.prompt_operator_support),
                prompt_operator_condition_number=_expand(memory.prompt.prompt_operator_condition_number),
                prompt_short_fallback=_expand(memory.prompt.prompt_short_fallback),
                prompt_operator_coeff_norm=_expand(memory.prompt.prompt_operator_coeff_norm),
                prompt_detector_fit=_expand(memory.prompt.prompt_detector_fit),
                prompt_detector_support=_expand(memory.prompt.prompt_detector_support),
                prompt_detector_condition_number=_expand(memory.prompt.prompt_detector_condition_number),
                prompt_detector_coeff_norm=_expand(memory.prompt.prompt_detector_coeff_norm),
                prompt_role_attn=_expand(memory.prompt.prompt_role_attn),
                prompt_role_fit=_expand(memory.prompt.prompt_role_fit),
            )
        ),
        summary_state=_expand(memory.summary_state),
        spk_embed=_expand(memory.spk_embed),
        prompt_valid_mask=_expand(memory.prompt_valid_mask),
        prompt_speech_mask=_expand(memory.prompt_speech_mask),
    )
    return validate_reference_duration_memory(expanded)


def move_duration_runtime_state(
    state: DurationRuntimeState | None,
    *,
    device: torch.device,
) -> DurationRuntimeState | None:
    if state is None:
        return None

    def _move_row(row: StreamingUnitizerRowState) -> StreamingUnitizerRowState:
        return StreamingUnitizerRowState(
            units=_move_tensor(row.units, device=device),
            durations=_move_tensor(row.durations, device=device),
            silence_mask=_move_tensor(row.silence_mask, device=device),
            sep_hint=_move_tensor(row.sep_hint, device=device),
            last_token=int(row.last_token),
            pending_separator=bool(row.pending_separator),
        )

    frontend_state = None
    if isinstance(getattr(state, "frontend_state", None), StreamingUnitizerState):
        frontend_state = StreamingUnitizerState(
            rows=[_move_row(row) for row in state.frontend_state.rows],
        )
    return replace(
        state,
        committed_units=_move_tensor(state.committed_units, device=device),
        rounding_residual=_move_tensor(state.rounding_residual, device=device),
        prefix_unit_offset=_move_tensor(state.prefix_unit_offset, device=device),
        cached_duration_exec=_move_tensor(state.cached_duration_exec, device=device),
        local_rate_ema=_move_tensor(state.local_rate_ema, device=device),
        since_last_boundary=_move_tensor(state.since_last_boundary, device=device),
        frontend_state=frontend_state,
        consumed_content_steps=_move_tensor(state.consumed_content_steps, device=device),
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
    for name in (
        "committed_units",
        "rounding_residual",
        "prefix_unit_offset",
        "cached_duration_exec",
        "local_rate_ema",
        "since_last_boundary",
        "consumed_content_steps",
    ):
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
    if state.prefix_unit_offset.dim() != 2 or state.prefix_unit_offset.size(1) != 1:
        raise ValueError(
            f"DurationRuntimeState.prefix_unit_offset must be rank-2 [B, 1], got shape={tuple(state.prefix_unit_offset.shape)}."
        )
    for name in ("local_rate_ema", "since_last_boundary", "consumed_content_steps"):
        value = getattr(state, name)
        if value is None:
            continue
        if value.dim() != 2 or value.size(1) != 1:
            raise ValueError(
                f"DurationRuntimeState.{name} must be rank-2 [B, 1], got shape={tuple(value.shape)}."
            )
    if state.frontend_state is not None:
        if not isinstance(state.frontend_state, StreamingUnitizerState):
            raise TypeError(
                f"DurationRuntimeState.frontend_state must be a StreamingUnitizerState, got {type(state.frontend_state)!r}."
            )
        if len(state.frontend_state.rows) != batch_size:
            raise ValueError(
                "DurationRuntimeState.frontend_state row count mismatch: "
                f"expected {batch_size}, got {len(state.frontend_state.rows)}."
            )
    return state
