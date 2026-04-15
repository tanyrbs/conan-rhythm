from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import torch

from modules.Conan.rhythm_v3.g_stats import masked_true_median_batch, masked_weighted_median_batch
from modules.Conan.rhythm_v3.contracts import normalize_global_rate_variant
from modules.Conan.rhythm_v3.math_utils import (
    apply_analytic_gap_clip,
    build_causal_local_rate_seq,
    build_causal_source_prefix_rate_seq,
    resolve_default_source_rate_init,
)
from modules.Conan.rhythm_v3.silence_surface import build_silence_tau_surface
from .losses_impl import DurationV3LossTargets, RhythmLossTargets


@dataclass(frozen=True)
class RhythmTargetBuildConfig:
    primary_target_surface: str
    distill_surface: str
    lambda_guidance: float
    lambda_distill: float
    distill_exec_weight: float
    distill_budget_weight: float
    distill_allocation_weight: float
    distill_prefix_weight: float
    distill_speech_shape_weight: float
    distill_pause_shape_weight: float
    plan_local_weight: float
    plan_cum_weight: float
    pause_boundary_weight: float
    budget_raw_weight: float
    budget_exec_weight: float
    feasible_debt_weight: float
    unit_logratio_weight: float = 0.0
    srmdp_role_consistency_weight: float = 0.0
    srmdp_notimeline_weight: float = 0.0
    srmdp_memory_role_weight: float = 0.0
    plan_segment_shape_weight: float = 0.0
    plan_pause_release_weight: float = 0.0
    pause_event_weight: float = 0.0
    pause_support_weight: float = 0.0
    pause_allocation_weight: float = 0.0
    pause_event_threshold: float = 0.5
    pause_event_temperature: float = 0.25
    pause_event_pos_weight: float = 2.0
    dedupe_primary_teacher_cache_distill: bool = True
    enable_distill_context_match: bool = False
    distill_context_floor: float = 0.35
    distill_context_power: float = 1.0
    distill_context_open_run_penalty: float = 0.50

    @property
    def use_guidance(self) -> bool:
        return self.lambda_guidance > 0.0

    @property
    def use_distill(self) -> bool:
        return self.lambda_distill > 0.0

    @property
    def use_distill_exec(self) -> bool:
        return self.use_distill and self.distill_exec_weight > 0.0

    @property
    def use_distill_budget(self) -> bool:
        return self.use_distill and self.distill_budget_weight > 0.0

    @property
    def use_distill_allocation(self) -> bool:
        return self.use_distill and self.distill_allocation_weight > 0.0

    @property
    def use_distill_prefix(self) -> bool:
        return self.use_distill and self.distill_prefix_weight > 0.0

    @property
    def use_distill_speech_shape(self) -> bool:
        return self.use_distill and self.distill_speech_shape_weight > 0.0

    @property
    def use_distill_pause_shape(self) -> bool:
        return self.use_distill and self.distill_pause_shape_weight > 0.0


@dataclass(frozen=True)
class DurationV3TargetBuildConfig:
    lambda_dur: float
    lambda_op: float
    lambda_pref: float
    lambda_bias: float = 0.0
    lambda_base: float = 0.0
    lambda_cons: float = 0.0
    lambda_zero: float = 0.0
    lambda_ortho: float = 0.0
    lambda_silence_aux: float = 0.0
    strict_target_alignment: bool = True
    anchor_mode: str = "baseline"
    baseline_target_mode: str = "deglobalized"
    baseline_train_mode: str = "joint"
    silence_coarse_weight: float = 0.25
    silence_logstretch_max: float = 0.35
    local_rate_decay: float = 0.95
    local_rate_decay_fast: float = 0.80
    local_rate_decay_slow: float = 0.97
    local_rate_slow_mix: float = 0.65
    analytic_gap_clip: float = 0.35
    silence_short_gap_scale: float = 0.35
    use_log_base_rate: bool = False
    simple_global_stats: bool = False
    rate_mode: str = "log_base"
    minimal_v1_profile: bool = False
    g_variant: str = "raw_median"
    g_trim_ratio: float = 0.2
    src_prefix_stat_mode: str = "ema"
    src_prefix_min_support: int = 3
    src_rate_init_mode: str = "first_speech"
    g_drop_edge_runs: int = 0
    min_boundary_confidence_for_g: float | None = None
    min_support_log_iqr_for_g: float = 0.0
    min_support_log_span_for_g: float = 0.0
    min_support_unique_for_g: int = 1
    enable_shared_beta1_probe: bool = False
    beta1_min: float = 0.7
    beta1_max: float = 1.3
    beta1_min_points: int = 24
    beta1_min_var: float = 2.5e-3

    @property
    def lambda_mem(self) -> float:
        return float(self.lambda_op)


def _resolve_target_boundary_confidence(
    unit_batch,
    *,
    unit_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    boundary_confidence = getattr(unit_batch, "boundary_confidence", None)
    if not isinstance(boundary_confidence, torch.Tensor):
        boundary_confidence = getattr(unit_batch, "source_boundary_cue", None)
    if not isinstance(boundary_confidence, torch.Tensor):
        return None
    boundary_confidence = boundary_confidence.float()
    if isinstance(unit_mask, torch.Tensor):
        boundary_confidence = boundary_confidence * unit_mask.float()
    return boundary_confidence


@dataclass(frozen=True)
class RhythmSampleKeyBundle:
    pause_exec_key: str
    pause_budget_key: str
    guidance_pause_key: str
    teacher_pause_key: str
    teacher_pause_budget_key: str
    target_speech_key: str
    target_pause_key: str
    target_speech_budget_key: str
    target_pause_budget_key: str


@dataclass(frozen=True)
class DistillConfidenceBundle:
    shared: Optional[torch.Tensor] = None
    exec: Optional[torch.Tensor] = None
    budget: Optional[torch.Tensor] = None
    prefix: Optional[torch.Tensor] = None
    allocation: Optional[torch.Tensor] = None
    shape: Optional[torch.Tensor] = None


def build_pseudo_source_duration_context(
    gt_dur: torch.Tensor,
    unit_mask: torch.Tensor,
    sep_hint: torch.Tensor,
    *,
    silence_mask: torch.Tensor | None = None,
    global_scale_range: tuple[float, float] = (0.85, 1.15),
    local_span_prob: float = 0.20,
    local_span_scale: tuple[float, float] = (0.7, 1.3),
    mask_prob: float = 0.10,
    flatten_boundary_prob: float = 0.15,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    dur = gt_dur.float().clone()
    mask = unit_mask.float().clamp(0.0, 1.0)
    speech_mask = (
        mask
        if silence_mask is None
        else mask * (1.0 - silence_mask.float().clamp(0.0, 1.0))
    )
    sep = sep_hint.float().clamp(0.0, 1.0)
    if dur.numel() <= 0:
        return dur

    scale_lo, scale_hi = float(global_scale_range[0]), float(global_scale_range[1])
    if scale_hi < scale_lo:
        scale_lo, scale_hi = scale_hi, scale_lo
    global_scale = torch.empty((dur.size(0), 1), device=dur.device, dtype=dur.dtype)
    global_scale.uniform_(scale_lo, scale_hi, generator=generator)
    dur = dur * global_scale

    local_prob = float(max(0.0, min(1.0, local_span_prob)))
    span_lo, span_hi = float(local_span_scale[0]), float(local_span_scale[1])
    if span_hi < span_lo:
        span_lo, span_hi = span_hi, span_lo
    for batch_idx in range(int(dur.size(0))):
        if float(torch.rand((1,), generator=generator, device=dur.device).item()) >= local_prob:
            continue
        valid_units = int(speech_mask[batch_idx].sum().item())
        if valid_units <= 1:
            continue
        speech_indices = torch.nonzero(speech_mask[batch_idx] > 0.5, as_tuple=False).reshape(-1)
        span_start_max = max(int(speech_indices.numel()) - 1, 1)
        span_start_idx = int(torch.randint(0, span_start_max, (1,), generator=generator, device=dur.device).item())
        span_len = int(
            torch.randint(
                2,
                min(8, int(speech_indices.numel()) - span_start_idx) + 1,
                (1,),
                generator=generator,
                device=dur.device,
            ).item()
        )
        span_tokens = speech_indices[span_start_idx : span_start_idx + span_len]
        local_scale = float(torch.empty((1,), device=dur.device, dtype=dur.dtype).uniform_(span_lo, span_hi, generator=generator).item())
        dur[batch_idx, span_tokens] = dur[batch_idx, span_tokens] * local_scale

    mask_prob = float(max(0.0, min(1.0, mask_prob)))
    if mask_prob > 0.0:
        sample_mask = torch.rand(dur.shape, generator=generator, device=dur.device) < mask_prob
        sample_mask = sample_mask & (speech_mask > 0.5)
        mean_dur = (dur * speech_mask).sum(dim=1, keepdim=True) / speech_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        dur = torch.where(sample_mask, mean_dur.expand_as(dur), dur)

    flatten_prob = float(max(0.0, min(1.0, flatten_boundary_prob)))
    if flatten_prob > 0.0:
        boundary_gate = (
            torch.rand((dur.size(0), 1), generator=generator, device=dur.device) < flatten_prob
        ).float()
        dur = torch.where((sep > 0.5) & (speech_mask > 0.5), dur * (1.0 - 0.10 * boundary_gate), dur)

    return dur.clamp_min(1.0e-4) * mask


NormalizeConfidenceFn = Callable[..., torch.Tensor]
BuildPrefixCarryFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]
SliceSurfaceFn = Callable[..., tuple[torch.Tensor, ...]]

_DISTILL_COMPONENT_FIELDS = ("exec", "budget", "prefix", "allocation")
_DISTILL_COMPONENT_SAMPLE_KEYS = {
    "exec": "rhythm_teacher_confidence_exec",
    "budget": "rhythm_teacher_confidence_budget",
    "prefix": "rhythm_teacher_confidence_prefix",
    "allocation": "rhythm_teacher_confidence_allocation",
    "shape": "rhythm_teacher_confidence_shape",
}


def _detach_optional(value):
    if isinstance(value, torch.Tensor):
        return value.detach()
    return value


def resolve_rhythm_sample_keys(
    sample: dict,
    *,
    primary_target_surface: str,
) -> RhythmSampleKeyBundle:
    pause_exec_key = "rhythm_pause_exec_tgt" if "rhythm_pause_exec_tgt" in sample else "rhythm_blank_exec_tgt"
    pause_budget_key = "rhythm_pause_budget_tgt" if "rhythm_pause_budget_tgt" in sample else "rhythm_blank_budget_tgt"
    guidance_pause_key = (
        "rhythm_guidance_pause_tgt" if "rhythm_guidance_pause_tgt" in sample else "rhythm_guidance_blank_tgt"
    )
    teacher_pause_key = (
        "rhythm_teacher_pause_exec_tgt"
        if "rhythm_teacher_pause_exec_tgt" in sample
        else "rhythm_teacher_blank_exec_tgt"
    )
    teacher_pause_budget_key = (
        "rhythm_teacher_pause_budget_tgt"
        if "rhythm_teacher_pause_budget_tgt" in sample
        else "rhythm_teacher_blank_budget_tgt"
    )
    if primary_target_surface == "teacher":
        return RhythmSampleKeyBundle(
            pause_exec_key=pause_exec_key,
            pause_budget_key=pause_budget_key,
            guidance_pause_key=guidance_pause_key,
            teacher_pause_key=teacher_pause_key,
            teacher_pause_budget_key=teacher_pause_budget_key,
            target_speech_key="rhythm_teacher_speech_exec_tgt",
            target_pause_key=teacher_pause_key,
            target_speech_budget_key="rhythm_teacher_speech_budget_tgt",
            target_pause_budget_key=teacher_pause_budget_key,
        )
    return RhythmSampleKeyBundle(
        pause_exec_key=pause_exec_key,
        pause_budget_key=pause_budget_key,
        guidance_pause_key=guidance_pause_key,
        teacher_pause_key=teacher_pause_key,
        teacher_pause_budget_key=teacher_pause_budget_key,
        target_speech_key="rhythm_speech_exec_tgt",
        target_pause_key=pause_exec_key,
        target_speech_budget_key="rhythm_speech_budget_tgt",
        target_pause_budget_key=pause_budget_key,
    )


def _resolve_sample_confidence(sample: dict, *, primary_target_surface: str):
    if primary_target_surface == "teacher":
        return _detach_optional(
            sample.get("rhythm_teacher_confidence", sample.get("rhythm_target_confidence"))
        )
    return _detach_optional(sample.get("rhythm_target_confidence"))


def _normalize_optional_confidence(
    confidence,
    *,
    batch_size: int,
    device: torch.device,
    fallback_confidence: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    if confidence is None:
        confidence = fallback_confidence
    if confidence is None:
        return torch.ones((batch_size, 1), device=device)
    if isinstance(confidence, torch.Tensor):
        tensor = confidence.detach().float().reshape(batch_size, -1)[:, :1].to(device=device)
    else:
        tensor = torch.as_tensor(confidence, dtype=torch.float32, device=device).reshape(batch_size, -1)[:, :1]
    return tensor.clamp(min=0.0, max=1.0)


def _normalize_distill_confidences(
    *,
    confidence_bundle: DistillConfidenceBundle,
    batch_size: int,
    device: torch.device,
    normalize_distill_confidence: NormalizeConfidenceFn,
    normalize_component_confidence: NormalizeConfidenceFn,
) -> DistillConfidenceBundle:
    shared = normalize_distill_confidence(
        confidence_bundle.shared,
        batch_size=batch_size,
        device=device,
    )
    normalized_components = {
        name: normalize_component_confidence(
            getattr(confidence_bundle, name),
            fallback_confidence=shared,
            batch_size=batch_size,
            device=device,
        )
        for name in _DISTILL_COMPONENT_FIELDS
    }
    exec_confidence = normalized_components["exec"]
    shape_confidence = normalize_component_confidence(
        confidence_bundle.shape,
        fallback_confidence=exec_confidence,
        batch_size=batch_size,
        device=device,
    )
    return DistillConfidenceBundle(shared=shared, shape=shape_confidence, **normalized_components)


@dataclass(frozen=True)
class DistillSurfaceBundle:
    speech: Optional[torch.Tensor] = None
    pause: Optional[torch.Tensor] = None
    speech_budget: Optional[torch.Tensor] = None
    pause_budget: Optional[torch.Tensor] = None
    allocation: Optional[torch.Tensor] = None
    prefix_clock: Optional[torch.Tensor] = None
    prefix_backlog: Optional[torch.Tensor] = None
    confidences: DistillConfidenceBundle = field(default_factory=DistillConfidenceBundle)
    context_match: Optional[torch.Tensor] = None
    source_kind: str = "none"
    same_source_primary_exec: bool = False
    same_source_primary_budget: bool = False
    same_source_primary_prefix: bool = False
    same_source_primary_allocation: bool = False
    same_source_primary_shape: bool = False

    @property
    def is_complete(self) -> bool:
        return self.speech is not None and self.pause is not None


def _required_rhythm_target_keys_present(sample: dict, keys: RhythmSampleKeyBundle) -> bool:
    required_keys = (
        keys.target_speech_key,
        keys.target_pause_key,
        keys.target_speech_budget_key,
        keys.target_pause_budget_key,
    )
    return all(key in sample for key in required_keys)


def _build_guidance_targets(
    sample: dict,
    keys: RhythmSampleKeyBundle,
    config: RhythmTargetBuildConfig,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not config.use_guidance:
        return None, None
    return sample.get("rhythm_guidance_speech_tgt"), sample.get(keys.guidance_pause_key)


def _build_cached_distill_surface(
    sample: dict,
    keys: RhythmSampleKeyBundle,
    config: RhythmTargetBuildConfig,
) -> DistillSurfaceBundle:
    return DistillSurfaceBundle(
        speech=sample.get("rhythm_teacher_speech_exec_tgt"),
        pause=sample.get(keys.teacher_pause_key),
        speech_budget=sample.get("rhythm_teacher_speech_budget_tgt") if config.use_distill_budget else None,
        pause_budget=sample.get(keys.teacher_pause_budget_key) if config.use_distill_budget else None,
        allocation=sample.get("rhythm_teacher_allocation_tgt") if config.use_distill_allocation else None,
        prefix_clock=sample.get("rhythm_teacher_prefix_clock_tgt") if config.use_distill_prefix else None,
        prefix_backlog=sample.get("rhythm_teacher_prefix_backlog_tgt") if config.use_distill_prefix else None,
        confidences=DistillConfidenceBundle(
            shared=_detach_optional(sample.get("rhythm_teacher_confidence")),
            **{
                name: _detach_optional(sample.get(key))
                for name, key in _DISTILL_COMPONENT_SAMPLE_KEYS.items()
            },
        ),
        source_kind="cache",
    )


def _build_offline_distill_confidences(
    offline_confidences: DistillConfidenceBundle | None,
    reference_speech: torch.Tensor,
) -> DistillConfidenceBundle:
    offline_confidences = offline_confidences or DistillConfidenceBundle()
    confidences = DistillConfidenceBundle(
        shared=_detach_optional(offline_confidences.shared),
        exec=_detach_optional(offline_confidences.exec),
        budget=_detach_optional(offline_confidences.budget),
        prefix=_detach_optional(offline_confidences.prefix),
        allocation=_detach_optional(offline_confidences.allocation),
        shape=_detach_optional(offline_confidences.shape),
    )
    if confidences.shared is not None:
        return confidences
    return DistillConfidenceBundle(
        shared=reference_speech.new_ones((reference_speech.size(0), 1)),
        exec=confidences.exec,
        budget=confidences.budget,
        prefix=confidences.prefix,
        allocation=confidences.allocation,
        shape=confidences.shape,
    )


def _build_runtime_offline_distill_surface(
    runtime_teacher,
    *,
    unit_batch,
    config: RhythmTargetBuildConfig,
    offline_confidences: DistillConfidenceBundle | None,
    slice_rhythm_surface_to_student: SliceSurfaceFn,
) -> DistillSurfaceBundle:
    speech = runtime_teacher.speech_duration_exec.detach()
    pause = getattr(runtime_teacher, "blank_duration_exec", runtime_teacher.pause_after_exec).detach()
    (
        speech,
        pause,
        speech_budget,
        pause_budget,
        allocation,
        prefix_clock,
        prefix_backlog,
    ) = slice_rhythm_surface_to_student(
        speech_exec=speech,
        pause_exec=pause,
        student_units=unit_batch.dur_anchor_src.size(1),
        dur_anchor_src=unit_batch.dur_anchor_src,
        unit_mask=unit_batch.unit_mask,
    )
    if not config.use_distill_budget:
        speech_budget = None
        pause_budget = None
    if not config.use_distill_allocation:
        allocation = None
    if not config.use_distill_prefix:
        prefix_clock = None
        prefix_backlog = None
    return DistillSurfaceBundle(
        speech=speech,
        pause=pause,
        speech_budget=speech_budget,
        pause_budget=pause_budget,
        allocation=allocation,
        prefix_clock=prefix_clock,
        prefix_backlog=prefix_backlog,
        confidences=_build_offline_distill_confidences(offline_confidences, speech),
        source_kind="offline",
    )


def _build_algorithmic_distill_surface(
    algorithmic_teacher,
    config: RhythmTargetBuildConfig,
) -> DistillSurfaceBundle:
    return DistillSurfaceBundle(
        speech=algorithmic_teacher.speech_exec_tgt.detach(),
        pause=algorithmic_teacher.pause_exec_tgt.detach(),
        speech_budget=algorithmic_teacher.speech_budget_tgt.detach() if config.use_distill_budget else None,
        pause_budget=algorithmic_teacher.pause_budget_tgt.detach() if config.use_distill_budget else None,
        allocation=algorithmic_teacher.allocation_tgt.detach() if config.use_distill_allocation else None,
        prefix_clock=algorithmic_teacher.prefix_clock_tgt.detach() if config.use_distill_prefix else None,
        prefix_backlog=algorithmic_teacher.prefix_backlog_tgt.detach() if config.use_distill_prefix else None,
        confidences=DistillConfidenceBundle(shared=algorithmic_teacher.confidence.detach()),
        source_kind="algorithmic",
    )


def _maybe_slice_distill_surface_to_student(
    bundle: DistillSurfaceBundle,
    *,
    unit_batch,
    slice_rhythm_surface_to_student: SliceSurfaceFn,
) -> DistillSurfaceBundle:
    if not bundle.is_complete or bundle.speech.size(1) == unit_batch.dur_anchor_src.size(1):
        return bundle
    (
        speech,
        pause,
        speech_budget,
        pause_budget,
        allocation,
        prefix_clock,
        prefix_backlog,
    ) = slice_rhythm_surface_to_student(
        speech_exec=bundle.speech,
        pause_exec=bundle.pause,
        student_units=unit_batch.dur_anchor_src.size(1),
        dur_anchor_src=unit_batch.dur_anchor_src,
        unit_mask=unit_batch.unit_mask,
    )
    return replace(
        bundle,
        speech=speech,
        pause=pause,
        speech_budget=speech_budget,
        pause_budget=pause_budget,
        allocation=allocation,
        prefix_clock=prefix_clock,
        prefix_backlog=prefix_backlog,
    )


def _populate_distill_surface_fields(
    bundle: DistillSurfaceBundle,
    *,
    unit_batch,
    config: RhythmTargetBuildConfig,
    build_prefix_carry_from_exec: BuildPrefixCarryFn,
) -> DistillSurfaceBundle:
    if not bundle.is_complete:
        return bundle
    allocation = bundle.allocation
    prefix_clock = bundle.prefix_clock
    prefix_backlog = bundle.prefix_backlog
    if config.use_distill_allocation and allocation is None:
        allocation = (bundle.speech.float() + bundle.pause.float()) * unit_batch.unit_mask.float()
    if config.use_distill_prefix and (prefix_clock is None or prefix_backlog is None):
        prefix_clock, prefix_backlog = build_prefix_carry_from_exec(
            bundle.speech,
            bundle.pause,
            unit_batch.dur_anchor_src,
            unit_batch.unit_mask,
    )
    return replace(bundle, allocation=allocation, prefix_clock=prefix_clock, prefix_backlog=prefix_backlog)


def _maybe_dedupe_primary_teacher_cache_distill(
    bundle: DistillSurfaceBundle,
    *,
    config: RhythmTargetBuildConfig,
) -> DistillSurfaceBundle:
    if not bundle.is_complete:
        return bundle
    if not bool(config.dedupe_primary_teacher_cache_distill):
        return bundle
    if config.primary_target_surface != "teacher":
        return bundle
    if config.distill_surface not in {"auto", "cache"}:
        return bundle
    if bundle.source_kind != "cache":
        return bundle

    zero = bundle.speech.new_zeros((bundle.speech.size(0), 1))
    shape_confidence = bundle.confidences.shape
    if shape_confidence is None:
        shape_confidence = bundle.confidences.exec
    if shape_confidence is None:
        shape_confidence = bundle.confidences.shared

    return replace(
        bundle,
        speech_budget=None,
        pause_budget=None,
        allocation=None,
        prefix_clock=None,
        prefix_backlog=None,
        confidences=DistillConfidenceBundle(
            shared=bundle.confidences.shared,
            exec=zero,
            budget=zero,
            prefix=zero,
            allocation=zero,
            shape=_detach_optional(shape_confidence),
        ),
    )


def _normalize_distill_surface_confidences(
    bundle: DistillSurfaceBundle,
    *,
    unit_batch,
    normalize_distill_confidence: NormalizeConfidenceFn,
    normalize_component_confidence: NormalizeConfidenceFn,
) -> DistillSurfaceBundle:
    normalized_confidences = _normalize_distill_confidences(
        confidence_bundle=bundle.confidences,
        batch_size=unit_batch.dur_anchor_src.size(0),
        device=unit_batch.dur_anchor_src.device,
        normalize_distill_confidence=normalize_distill_confidence,
        normalize_component_confidence=normalize_component_confidence,
    )
    return replace(bundle, confidences=normalized_confidences)


def _estimate_distill_context_match(
    sample: dict,
    *,
    unit_batch,
    config: RhythmTargetBuildConfig,
) -> Optional[torch.Tensor]:
    if not bool(config.enable_distill_context_match):
        return None
    batch_size = unit_batch.dur_anchor_src.size(0)
    device = unit_batch.dur_anchor_src.device
    ratio = sample.get("rhythm_stream_prefix_ratio")
    if ratio is None:
        visible = sample.get("rhythm_stream_visible_units")
        full = sample.get("rhythm_stream_full_units")
        if visible is not None and full is not None:
            ratio = visible.float() / full.float().clamp_min(1.0)
    ratio = _normalize_optional_confidence(
        ratio,
        batch_size=batch_size,
        device=device,
        fallback_confidence=torch.ones((batch_size, 1), device=device),
    ).clamp(0.0, 1.0)
    floor = min(max(float(config.distill_context_floor), 0.0), 1.0)
    ratio_gate = ratio.pow(float(config.distill_context_power))
    gate = floor + (1.0 - floor) * ratio_gate
    open_run_mask = getattr(unit_batch, "open_run_mask", None)
    if open_run_mask is not None:
        visible_mask = unit_batch.unit_mask.float()
        open_run_ratio = (open_run_mask.float() * visible_mask).sum(dim=1, keepdim=True) / visible_mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        is_truncated = ratio < (1.0 - 1.0e-6)
        open_run_penalty = (1.0 - float(config.distill_context_open_run_penalty) * open_run_ratio).clamp(0.0, 1.0)
        gate = torch.where(is_truncated, gate * open_run_penalty, gate)
    return gate.detach().clamp(0.0, 1.0)


def _apply_distill_context_match(
    bundle: DistillSurfaceBundle,
    *,
    sample: dict,
    unit_batch,
    config: RhythmTargetBuildConfig,
) -> DistillSurfaceBundle:
    if not bundle.is_complete:
        return bundle
    context_match = _estimate_distill_context_match(
        sample,
        unit_batch=unit_batch,
        config=config,
    )
    if context_match is None:
        return bundle

    def _mul_optional(value, *, fill_if_missing: bool = False):
        if value is None:
            return context_match if fill_if_missing else None
        return _detach_optional(value) * context_match

    return replace(
        bundle,
        confidences=DistillConfidenceBundle(
            shared=_mul_optional(bundle.confidences.shared, fill_if_missing=True),
            exec=_mul_optional(bundle.confidences.exec),
            budget=_mul_optional(bundle.confidences.budget),
            prefix=_mul_optional(bundle.confidences.prefix),
            allocation=_mul_optional(bundle.confidences.allocation),
            shape=_mul_optional(bundle.confidences.shape),
        ),
        context_match=context_match,
    )


def _confidence_is_active(confidence: Optional[torch.Tensor]) -> bool:
    if confidence is None:
        return True
    if not isinstance(confidence, torch.Tensor):
        return bool(confidence)
    return bool(torch.any(confidence.detach() > 0.0).item())


def _annotate_distill_same_source_overlap(
    bundle: DistillSurfaceBundle,
    *,
    config: RhythmTargetBuildConfig,
) -> DistillSurfaceBundle:
    same_cached_teacher = (
        bundle.is_complete
        and config.primary_target_surface == "teacher"
        and bundle.source_kind == "cache"
    )
    if not same_cached_teacher:
        return bundle
    return replace(
        bundle,
        same_source_primary_exec=(
            config.use_distill_exec
            and _confidence_is_active(bundle.confidences.exec)
        ),
        same_source_primary_budget=(
            config.use_distill_budget
            and bundle.speech_budget is not None
            and bundle.pause_budget is not None
            and _confidence_is_active(bundle.confidences.budget)
        ),
        same_source_primary_prefix=(
            config.use_distill_prefix
            and (bundle.prefix_clock is not None or bundle.prefix_backlog is not None)
            and _confidence_is_active(bundle.confidences.prefix)
        ),
        same_source_primary_allocation=(
            config.use_distill_allocation
            and bundle.allocation is not None
            and _confidence_is_active(bundle.confidences.allocation)
        ),
        same_source_primary_shape=(
            (config.use_distill_speech_shape and bundle.speech is not None)
            or (config.use_distill_pause_shape and bundle.pause is not None)
        )
        and _confidence_is_active(bundle.confidences.shape),
    )


def _resolve_distill_surface_bundle(
    *,
    sample: dict,
    keys: RhythmSampleKeyBundle,
    unit_batch,
    config: RhythmTargetBuildConfig,
    runtime_teacher=None,
    algorithmic_teacher=None,
    offline_confidences: DistillConfidenceBundle | None = None,
    normalize_distill_confidence: NormalizeConfidenceFn,
    normalize_component_confidence: NormalizeConfidenceFn,
    build_prefix_carry_from_exec: BuildPrefixCarryFn,
    slice_rhythm_surface_to_student: SliceSurfaceFn,
) -> DistillSurfaceBundle:
    bundle = DistillSurfaceBundle()
    if config.use_distill and config.distill_surface in {"auto", "cache"}:
        bundle = _build_cached_distill_surface(sample, keys, config)
    if (
        config.use_distill
        and bundle.speech is None
        and config.distill_surface in {"auto", "offline"}
        and runtime_teacher is not None
    ):
        bundle = _build_runtime_offline_distill_surface(
            runtime_teacher,
            unit_batch=unit_batch,
            config=config,
            offline_confidences=offline_confidences,
            slice_rhythm_surface_to_student=slice_rhythm_surface_to_student,
        )
    if (
        config.use_distill
        and bundle.speech is None
        and config.distill_surface in {"auto", "algorithmic"}
        and algorithmic_teacher is not None
    ):
        bundle = _build_algorithmic_distill_surface(algorithmic_teacher, config)
    if not config.use_distill or not bundle.is_complete:
        return DistillSurfaceBundle()
    bundle = _maybe_slice_distill_surface_to_student(
        bundle,
        unit_batch=unit_batch,
        slice_rhythm_surface_to_student=slice_rhythm_surface_to_student,
    )
    bundle = _populate_distill_surface_fields(
        bundle,
        unit_batch=unit_batch,
        config=config,
        build_prefix_carry_from_exec=build_prefix_carry_from_exec,
    )
    bundle = _maybe_dedupe_primary_teacher_cache_distill(
        bundle,
        config=config,
    )
    bundle = _apply_distill_context_match(
        bundle,
        sample=sample,
        unit_batch=unit_batch,
        config=config,
    )
    bundle = _normalize_distill_surface_confidences(
        bundle,
        unit_batch=unit_batch,
        normalize_distill_confidence=normalize_distill_confidence,
        normalize_component_confidence=normalize_component_confidence,
    )
    return _annotate_distill_same_source_overlap(bundle, config=config)


def _build_sample_and_guidance_confidences(
    sample: dict,
    *,
    config: RhythmTargetBuildConfig,
    unit_batch,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    batch_size = unit_batch.dur_anchor_src.size(0)
    device = unit_batch.dur_anchor_src.device
    sample_confidence = _normalize_optional_confidence(
        _resolve_sample_confidence(
            sample,
            primary_target_surface=config.primary_target_surface,
        ),
        batch_size=batch_size,
        device=device,
    )
    guidance_confidence = None
    if config.use_guidance:
        guidance_confidence = _normalize_optional_confidence(
            sample.get("rhythm_guidance_confidence"),
            batch_size=batch_size,
            device=device,
            fallback_confidence=sample_confidence,
        )
    return sample_confidence, guidance_confidence


def build_rhythm_loss_targets_from_sample(
    *,
    sample: dict,
    unit_batch,
    config: RhythmTargetBuildConfig,
    runtime_teacher=None,
    algorithmic_teacher=None,
    offline_confidences: DistillConfidenceBundle | None = None,
    normalize_distill_confidence: NormalizeConfidenceFn,
    normalize_component_confidence: NormalizeConfidenceFn,
    build_prefix_carry_from_exec: BuildPrefixCarryFn,
    slice_rhythm_surface_to_student: SliceSurfaceFn,
) -> RhythmLossTargets | None:
    keys = resolve_rhythm_sample_keys(sample, primary_target_surface=config.primary_target_surface)
    if not _required_rhythm_target_keys_present(sample, keys):
        return None

    guidance_speech, guidance_pause = _build_guidance_targets(sample, keys, config)
    distill_bundle = _resolve_distill_surface_bundle(
        sample=sample,
        keys=keys,
        unit_batch=unit_batch,
        config=config,
        runtime_teacher=runtime_teacher,
        algorithmic_teacher=algorithmic_teacher,
        offline_confidences=offline_confidences,
        normalize_distill_confidence=normalize_distill_confidence,
        normalize_component_confidence=normalize_component_confidence,
        build_prefix_carry_from_exec=build_prefix_carry_from_exec,
        slice_rhythm_surface_to_student=slice_rhythm_surface_to_student,
    )
    sample_confidence, guidance_confidence = _build_sample_and_guidance_confidences(
        sample,
        config=config,
        unit_batch=unit_batch,
    )

    return RhythmLossTargets(
        speech_exec_tgt=sample[keys.target_speech_key],
        pause_exec_tgt=sample[keys.target_pause_key],
        speech_budget_tgt=sample[keys.target_speech_budget_key],
        pause_budget_tgt=sample[keys.target_pause_budget_key],
        unit_mask=unit_batch.unit_mask,
        dur_anchor_src=unit_batch.dur_anchor_src,
        unit_logratio_weight=float(config.unit_logratio_weight),
        srmdp_role_consistency_weight=float(config.srmdp_role_consistency_weight),
        srmdp_notimeline_weight=float(config.srmdp_notimeline_weight),
        srmdp_memory_role_weight=float(config.srmdp_memory_role_weight),
        srmdp_role_id_src_tgt=_detach_optional(sample.get("rhythm_srmdp_role_id_src_tgt")),
        srmdp_ref_memory_role_id_tgt=_detach_optional(sample.get("rhythm_srmdp_ref_memory_role_id_tgt")),
        srmdp_ref_memory_mask_tgt=_detach_optional(sample.get("rhythm_srmdp_ref_memory_mask_tgt")),
        plan_local_weight=float(config.plan_local_weight),
        plan_cum_weight=float(config.plan_cum_weight),
        plan_segment_shape_weight=float(config.plan_segment_shape_weight),
        plan_pause_release_weight=float(config.plan_pause_release_weight),
        sample_confidence=sample_confidence,
        guidance_speech_tgt=guidance_speech,
        guidance_pause_tgt=guidance_pause,
        guidance_confidence=guidance_confidence,
        distill_speech_tgt=distill_bundle.speech,
        distill_pause_tgt=distill_bundle.pause,
        distill_speech_budget_tgt=distill_bundle.speech_budget,
        distill_pause_budget_tgt=distill_bundle.pause_budget,
        distill_allocation_tgt=distill_bundle.allocation,
        distill_prefix_clock_tgt=distill_bundle.prefix_clock,
        distill_prefix_backlog_tgt=distill_bundle.prefix_backlog,
        distill_confidence=distill_bundle.confidences.shared,
        distill_exec_confidence=distill_bundle.confidences.exec,
        distill_budget_confidence=distill_bundle.confidences.budget,
        distill_prefix_confidence=distill_bundle.confidences.prefix,
        distill_allocation_confidence=distill_bundle.confidences.allocation,
        distill_shape_confidence=distill_bundle.confidences.shape,
        distill_context_match=distill_bundle.context_match,
        distill_exec_weight=float(config.distill_exec_weight),
        distill_budget_weight=float(config.distill_budget_weight),
        distill_allocation_weight=float(config.distill_allocation_weight),
        distill_prefix_weight=float(config.distill_prefix_weight),
        distill_speech_shape_weight=float(config.distill_speech_shape_weight),
        distill_pause_shape_weight=float(config.distill_pause_shape_weight),
        distill_same_source_exec=bool(distill_bundle.same_source_primary_exec),
        distill_same_source_budget=bool(distill_bundle.same_source_primary_budget),
        distill_same_source_prefix=bool(distill_bundle.same_source_primary_prefix),
        distill_same_source_allocation=bool(distill_bundle.same_source_primary_allocation),
        distill_same_source_shape=bool(distill_bundle.same_source_primary_shape),
        budget_raw_weight=float(config.budget_raw_weight),
        budget_exec_weight=float(config.budget_exec_weight),
        pause_boundary_weight=float(config.pause_boundary_weight),
        feasible_debt_weight=float(config.feasible_debt_weight),
        pause_event_weight=float(config.pause_event_weight),
        pause_support_weight=float(config.pause_support_weight),
        pause_allocation_weight=float(config.pause_allocation_weight),
        pause_event_threshold=float(config.pause_event_threshold),
        pause_event_temperature=float(config.pause_event_temperature),
        pause_event_pos_weight=float(config.pause_event_pos_weight),
    )


def build_duration_v3_loss_targets(
    *,
    sample: dict,
    output: dict,
    config: DurationV3TargetBuildConfig,
) -> DurationV3LossTargets | None:
    if bool(getattr(config, "minimal_v1_profile", False)):
        if float(config.lambda_op) > 0.0:
            raise ValueError("rhythm_v3_minimal_v1_profile forbids prompt summary/operator loss (lambda_op=0 required).")
        if float(config.lambda_zero) > 0.0:
            raise ValueError("rhythm_v3_minimal_v1_profile forbids zero-mean identifiability loss (lambda_zero=0 required).")
        if float(config.lambda_ortho) > 0.0:
            raise ValueError("rhythm_v3_minimal_v1_profile forbids orthogonality loss (lambda_ortho=0 required).")
        if float(config.lambda_base) > 0.0:
            raise ValueError("rhythm_v3_minimal_v1_profile forbids baseline/log-base loss (lambda_base=0 required).")
        if str(config.baseline_train_mode or "joint").strip().lower() == "pretrain":
            raise ValueError("rhythm_v3_minimal_v1_profile forbids baseline pretrain mode.")
    execution = output.get("rhythm_execution")
    unit_batch = output.get("rhythm_unit_batch")
    if execution is None or unit_batch is None:
        return None
    unit_mask = unit_batch.unit_mask.float()
    sample_confidence = _normalize_optional_confidence(
        _detach_optional(sample.get("rhythm_target_confidence")),
        batch_size=int(unit_mask.size(0)),
        device=unit_mask.device,
    )
    silence_mask = (
        getattr(unit_batch, "source_silence_mask", None).float().clamp(0.0, 1.0) * unit_mask
        if isinstance(getattr(unit_batch, "source_silence_mask", None), torch.Tensor)
        else unit_mask.new_zeros(unit_mask.shape)
    )
    speech_mask = unit_mask * (1.0 - silence_mask)
    committed_mask = getattr(execution, "commit_mask", getattr(unit_batch, "sealed_mask", unit_mask)).float() * unit_mask
    committed_speech_mask = committed_mask * speech_mask
    committed_silence_mask = committed_mask * silence_mask
    unit_duration_tgt = _resolve_duration_v3_target(
        sample=sample,
        unit_mask=unit_mask,
        config=config,
    )
    unit_anchor_base = getattr(unit_batch, "unit_anchor_base", None)
    if unit_anchor_base is None:
        return None
    prediction_anchor = _resolve_duration_v3_prediction_anchor(
        unit_batch=unit_batch,
        unit_mask=unit_mask,
        speech_mask=speech_mask,
        anchor_mode=str(config.anchor_mode or "baseline").strip().lower(),
    )
    baseline_pretrain_only = (str(config.baseline_train_mode or "joint").strip().lower() == "pretrain")
    if bool(getattr(config, "minimal_v1_profile", False)):
        baseline_pretrain_only = False
        baseline_duration_tgt, baseline_mask, baseline_global_tgt = None, None, None
    else:
        baseline_duration_tgt, baseline_mask, baseline_global_tgt = _build_duration_v3_baseline_targets(
            unit_duration_tgt=unit_duration_tgt,
            unit_anchor_base=unit_anchor_base.float().detach(),
            speech_mask=speech_mask.float(),
            baseline_target_mode=str(config.baseline_target_mode or "deglobalized").strip().lower(),
        )
    if bool(getattr(config, "minimal_v1_profile", False)):
        prompt_targets = _resolve_duration_v3_minimal_prompt_targets(
            ref_memory=output.get("rhythm_ref_conditioning"),
        )
        if not isinstance(prompt_targets["global_rate"], torch.Tensor):
            raise ValueError(
                "rhythm_v3_minimal_v1_profile requires rhythm_ref_conditioning.global_rate "
                "to build source-anchored global/coarse/local targets."
            )
    else:
        prompt_targets = _resolve_duration_v3_prompt_targets(
            ref_memory=output.get("rhythm_ref_conditioning"),
            lambda_op=(0.0 if baseline_pretrain_only else float(config.lambda_op)),
            lambda_zero=(0.0 if baseline_pretrain_only else float(config.lambda_zero)),
            lambda_ortho=(0.0 if baseline_pretrain_only else float(config.lambda_ortho)),
        )
    def _resolve_optional_unit_surface(key: str) -> torch.Tensor | None:
        if key not in sample:
            return None
        raw_value = sample.get(key)
        value_tensor = raw_value if isinstance(raw_value, torch.Tensor) else torch.as_tensor(
            raw_value,
            dtype=torch.float32,
            device=unit_mask.device,
        )
        return _align_duration_v3_surface(
            tensor=value_tensor,
            unit_mask=unit_mask,
            strict_target_alignment=False,
        ).float()

    unit_confidence_shared_tgt = _resolve_optional_unit_surface("unit_confidence_tgt")
    unit_confidence_local_tgt = _resolve_optional_unit_surface("unit_confidence_local_tgt")
    unit_confidence_coarse_tgt = _resolve_optional_unit_surface("unit_confidence_coarse_tgt")
    if unit_confidence_local_tgt is None:
        unit_confidence_local_tgt = unit_confidence_shared_tgt
    if unit_confidence_coarse_tgt is None:
        unit_confidence_coarse_tgt = unit_confidence_shared_tgt
    unit_confidence_tgt = unit_confidence_coarse_tgt
    global_shift_tgt = None
    coarse_logstretch_tgt = None
    silence_coarse_logstretch_tgt = None
    coarse_correction_tgt = None
    coarse_duration_tgt = None
    residual_logstretch_tgt = None
    global_bias_tgt = None
    global_bias_tgt_support_mass = None
    global_bias_tgt_support_count = None
    coarse_target_speech_conf_mean = None
    local_residual_tgt = None
    local_residual_tgt_center = None
    local_residual_tgt_abs_mean = None
    beta1_tgt = None
    residual_logstretch_tgt_mean = None
    prefix_duration_tgt = None
    consistency_local_residual_tgt = None
    consistency_logstretch_tgt = None
    state_prev = output.get("rhythm_state_prev")
    init_local_rate = None
    if state_prev is not None and isinstance(getattr(state_prev, "local_rate_ema", None), torch.Tensor):
        init_local_rate = state_prev.local_rate_ema.float().detach()
    if isinstance(prompt_targets["global_rate"], torch.Tensor):
        use_log_base_rate = bool(getattr(config, "use_log_base_rate", False))
        if str(getattr(config, "rate_mode", "log_base") or "log_base").strip().lower() == "simple_global":
            use_log_base_rate = False
        if bool(getattr(config, "simple_global_stats", False)):
            use_log_base_rate = False
        if (
            use_log_base_rate
            and isinstance(getattr(unit_batch, "unit_rate_log_base", None), torch.Tensor)
        ):
            log_rate_base = getattr(unit_batch, "unit_rate_log_base").float().detach()
        elif use_log_base_rate:
            log_rate_base = torch.log(unit_anchor_base.float().detach().clamp_min(1.0e-6))
        else:
            log_rate_base = None
        log_anchor = torch.log(prediction_anchor.float().clamp_min(1.0e-6))
        observed_log_anchor = (
            (log_anchor - log_rate_base) * unit_mask.float()
            if isinstance(log_rate_base, torch.Tensor)
            else log_anchor * unit_mask.float()
        )
        default_init_rate = output.get("rhythm_v3_source_rate_init")
        if init_local_rate is None:
            default_init_rate = resolve_default_source_rate_init(
                observed_log=observed_log_anchor,
                speech_mask=committed_speech_mask.float(),
                src_rate_init_mode=getattr(config, "src_rate_init_mode", "auto"),
                learned_init_rate=default_init_rate,
                auto_fallback="first_speech",
            )
        g_variant_tgt = normalize_global_rate_variant(
            getattr(config, "g_variant", "raw_median")
        )
        prefix_weight_tgt = None
        if (
            g_variant_tgt in {"weighted_median", "softclean_wmed", "softclean_wtmean"}
            and isinstance(getattr(unit_batch, "source_run_stability", None), torch.Tensor)
        ):
            prefix_weight_tgt = (
                getattr(unit_batch, "source_run_stability").float().clamp(0.0, 1.0)
                * committed_speech_mask.float()
            )
        boundary_confidence_tgt = _resolve_target_boundary_confidence(
            unit_batch,
            unit_mask=unit_mask,
        )
        local_rate_seq_tgt, _ = build_causal_source_prefix_rate_seq(
            observed_log=observed_log_anchor,
            speech_mask=committed_speech_mask.float(),
            init_rate=init_local_rate,
            default_init_rate=default_init_rate,
            stat_mode=str(getattr(config, "src_prefix_stat_mode", "ema") or "ema"),
            decay=float(config.local_rate_decay),
            decay_fast=float(getattr(config, "local_rate_decay_fast", 0.80)),
            decay_slow=float(getattr(config, "local_rate_decay_slow", 0.97)),
            slow_mix=float(getattr(config, "local_rate_slow_mix", 0.65)),
            variant=g_variant_tgt,
            trim_ratio=float(getattr(config, "g_trim_ratio", 0.2) or 0.2),
            min_support=int(getattr(config, "src_prefix_min_support", 3) or 3),
            weight=prefix_weight_tgt,
            valid_mask=unit_mask.float(),
            closed_mask=committed_mask.float(),
            boundary_confidence=boundary_confidence_tgt,
            min_boundary_confidence=getattr(config, "min_boundary_confidence_for_g", None),
            drop_edge_runs=int(getattr(config, "g_drop_edge_runs", 0) or 0),
            min_speech_ratio=0.0,
            min_support_log_iqr=float(getattr(config, "min_support_log_iqr_for_g", 0.0)),
            min_support_log_span=float(getattr(config, "min_support_log_span_for_g", 0.0)),
            min_support_unique_count=int(getattr(config, "min_support_unique_for_g", 1)),
            unit_ids=(
                getattr(unit_batch, "content_units", None).long()
                if isinstance(getattr(unit_batch, "content_units", None), torch.Tensor)
                else None
            ),
        )
        analytic_gap_tgt = apply_analytic_gap_clip(
            prompt_targets["global_rate"].float().detach() - local_rate_seq_tgt,
            getattr(config, "analytic_gap_clip", 0.0),
        )
        full_logstretch_tgt = (
            torch.log(unit_duration_tgt.float().clamp_min(1.0e-6))
            - torch.log(prediction_anchor.float().clamp_min(1.0e-6))
        ) * committed_mask.float()
        beta1_tgt = (
            _fit_batch_shared_beta1(
                analytic_gap=analytic_gap_tgt.detach(),
                full_logstretch=full_logstretch_tgt.detach(),
                speech_mask=committed_speech_mask.float(),
                weight=(
                    unit_confidence_coarse_tgt
                    if isinstance(unit_confidence_coarse_tgt, torch.Tensor)
                    else None
                ),
                beta1_min=float(getattr(config, "beta1_min", 0.7)),
                beta1_max=float(getattr(config, "beta1_max", 1.3)),
                min_points=int(getattr(config, "beta1_min_points", 24)),
                min_var=float(getattr(config, "beta1_min_var", 2.5e-3)),
            )
            if bool(getattr(config, "enable_shared_beta1_probe", False))
            else analytic_gap_tgt.new_ones((analytic_gap_tgt.size(0), 1))
        )
        global_shift_tgt = (analytic_gap_tgt * beta1_tgt) * committed_mask.float()
        residual_logstretch_tgt = (full_logstretch_tgt - global_shift_tgt).detach()
        residual_logstretch_tgt_mean = (
            (residual_logstretch_tgt * committed_speech_mask.float()).sum(dim=1, keepdim=True)
            / committed_speech_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        ).detach()
        speech_support = committed_speech_mask.float().sum(dim=1, keepdim=True)
        global_bias_tgt = residual_logstretch_tgt.new_zeros((residual_logstretch_tgt.size(0), 1))
        global_bias_tgt_support_count = speech_support.detach()
        if isinstance(unit_confidence_coarse_tgt, torch.Tensor):
            coarse_conf = unit_confidence_coarse_tgt.float() * committed_speech_mask.float()
            global_bias_tgt_support_mass = coarse_conf.sum(dim=1, keepdim=True).detach()
            coarse_target_speech_conf_mean = (
                global_bias_tgt_support_mass / global_bias_tgt_support_count.clamp_min(1.0)
            ).detach()
            valid_rows = global_bias_tgt_support_mass.squeeze(1) > 1.0e-6
        else:
            global_bias_tgt_support_mass = speech_support.detach()
            coarse_target_speech_conf_mean = torch.where(
                global_bias_tgt_support_count > 0.0,
                torch.ones_like(global_bias_tgt_support_count),
                torch.zeros_like(global_bias_tgt_support_count),
            ).detach()
            valid_rows = global_bias_tgt_support_count.squeeze(1) > 0.0
        if bool(valid_rows.any().item()):
            valid_weight = (
                unit_confidence_coarse_tgt[valid_rows]
                if isinstance(unit_confidence_coarse_tgt, torch.Tensor)
                else None
            )
            global_bias_tgt_valid = _masked_duration_v3_median(
                residual_logstretch_tgt[valid_rows],
                committed_speech_mask.float()[valid_rows],
                weight=valid_weight,
            ).detach()
            global_bias_tgt[valid_rows] = global_bias_tgt_valid
        coarse_bias_seq = global_bias_tgt.float()
        while coarse_bias_seq.dim() < residual_logstretch_tgt.dim():
            coarse_bias_seq = coarse_bias_seq.unsqueeze(-1)
        coarse_correction_tgt = coarse_bias_seq.expand_as(residual_logstretch_tgt) * committed_mask.float()
        coarse_logstretch_tgt = (global_shift_tgt + coarse_correction_tgt) * committed_mask.float()
        silence_tau_tgt = _build_duration_v3_silence_tau(
            prediction_anchor=prediction_anchor.float(),
            committed_silence_mask=committed_silence_mask.float(),
            sep_hint=(
                getattr(unit_batch, "sep_mask", None).float()
                if isinstance(getattr(unit_batch, "sep_mask", None), torch.Tensor)
                else None
            ),
            boundary_cue=_resolve_target_boundary_confidence(unit_batch),
            max_silence_logstretch=float(config.silence_logstretch_max),
            short_gap_scale=float(config.silence_short_gap_scale),
            minimal_v1_profile=bool(getattr(config, "minimal_v1_profile", False)),
        )
        if bool((committed_silence_mask > 0.5).any().item()):
            silence_coarse_logstretch_tgt = torch.clamp(
                coarse_logstretch_tgt,
                min=-silence_tau_tgt,
                max=silence_tau_tgt,
            ) * committed_silence_mask.float()
            coarse_logstretch_tgt = (
                coarse_logstretch_tgt * committed_speech_mask.float()
                + silence_coarse_logstretch_tgt
            )
        elif bool(getattr(config, "minimal_v1_profile", False)) and not isinstance(coarse_logstretch_tgt, torch.Tensor):
            raise RuntimeError(
                "minimal_v1_profile requires coarse-derived silence target; missing coarse_logstretch_tgt "
                "is not allowed when committed silence exists."
            )
        coarse_duration_tgt = (
            prediction_anchor.float()
            * torch.exp(coarse_logstretch_tgt.float())
            * committed_mask.float()
        )
        prefix_duration_tgt = coarse_duration_tgt.detach()
        local_residual_tgt = (full_logstretch_tgt - coarse_logstretch_tgt) * committed_speech_mask.float()
        if isinstance(local_residual_tgt, torch.Tensor):
            committed_speech_float = committed_speech_mask.float()
            local_residual_tgt_center = _masked_duration_v3_median(
                local_residual_tgt.detach(),
                committed_speech_float,
                weight=unit_confidence_local_tgt,
            ).detach()
            local_residual_tgt_abs_mean = (
                (local_residual_tgt.detach().abs() * committed_speech_float).sum(dim=1, keepdim=True)
                / committed_speech_float.sum(dim=1, keepdim=True).clamp_min(1.0)
            ).detach()
    if (
        bool(getattr(config, "minimal_v1_profile", False))
        and bool((committed_silence_mask > 0.5).any().item())
        and not isinstance(coarse_logstretch_tgt, torch.Tensor)
    ):
        raise RuntimeError(
            "minimal_v1_profile requires coarse-derived silence target; "
            "missing coarse_logstretch_tgt is not allowed when committed silence exists."
        )
    if prefix_duration_tgt is None:
        prefix_duration_tgt = (
            unit_duration_tgt.float() * committed_speech_mask.float()
            + prediction_anchor.float() * committed_silence_mask.float()
        ).detach()
    if coarse_duration_tgt is None:
        coarse_duration_tgt = prefix_duration_tgt.detach()
    else:
        coarse_duration_tgt = coarse_duration_tgt.detach()
    consistency_duration_tgt = None
    consistency_mask = None
    if float(config.lambda_cons) > 0.0 and not baseline_pretrain_only:
        consistency_duration_tgt, consistency_mask = _build_duration_v3_consistency_targets(
            state_prev=output.get("rhythm_state_prev"),
            unit_mask=unit_mask,
            committed_mask=committed_mask,
        )
        if (
            isinstance(consistency_duration_tgt, torch.Tensor)
            and isinstance(global_shift_tgt, torch.Tensor)
            and isinstance(prediction_anchor, torch.Tensor)
        ):
            consistency_full = (
                torch.log(consistency_duration_tgt.float().clamp_min(1.0e-6))
                - torch.log(prediction_anchor.float().clamp_min(1.0e-6))
            ) * consistency_mask.float()
            consistency_logstretch_tgt = consistency_full * (consistency_mask.float() * speech_mask.float())
            if isinstance(coarse_logstretch_tgt, torch.Tensor):
                silence_tau_cons = _build_duration_v3_silence_tau(
                    prediction_anchor=prediction_anchor.float(),
                    committed_silence_mask=(consistency_mask.float() * silence_mask.float()),
                    sep_hint=(
                        getattr(unit_batch, "sep_mask", None).float()
                        if isinstance(getattr(unit_batch, "sep_mask", None), torch.Tensor)
                        else None
                    ),
                    boundary_cue=_resolve_target_boundary_confidence(unit_batch),
                    max_silence_logstretch=float(config.silence_logstretch_max),
                    short_gap_scale=float(config.silence_short_gap_scale),
                    minimal_v1_profile=bool(getattr(config, "minimal_v1_profile", False)),
                )
                consistency_logstretch_tgt = consistency_logstretch_tgt + (
                    torch.clamp(
                        coarse_logstretch_tgt.float(),
                        min=-silence_tau_cons,
                        max=silence_tau_cons,
                    )
                    * (consistency_mask.float() * silence_mask.float())
                )
            consistency_local_residual_tgt = (
                consistency_full
                - (
                    coarse_logstretch_tgt.float()
                    if isinstance(coarse_logstretch_tgt, torch.Tensor)
                    else global_shift_tgt.float()
                )
            ) * (consistency_mask.float() * speech_mask.float())
    silence_coarse_weight = (
        0.0
        if bool(getattr(config, "minimal_v1_profile", False))
        else float(config.silence_coarse_weight)
    )
    return DurationV3LossTargets(
        unit_duration_tgt=unit_duration_tgt.float(),
        unit_anchor_base=unit_anchor_base.float().detach(),
        sample_confidence=sample_confidence,
        speech_mask=speech_mask.float(),
        silence_mask=silence_mask.float(),
        committed_speech_mask=committed_speech_mask.float(),
        committed_silence_mask=committed_silence_mask.float(),
        silence_coarse_weight=silence_coarse_weight,
        unit_confidence_local_tgt=unit_confidence_local_tgt,
        unit_confidence_coarse_tgt=unit_confidence_coarse_tgt,
        unit_confidence_tgt=unit_confidence_tgt,
        prediction_anchor=prediction_anchor.float().detach(),
        unit_mask=unit_mask,
        committed_mask=committed_mask.float(),
        baseline_duration_tgt=baseline_duration_tgt,
        baseline_mask=baseline_mask,
        baseline_global_tgt=baseline_global_tgt,
        global_rate=prompt_targets["global_rate"],
        global_shift_tgt=global_shift_tgt,
        coarse_logstretch_tgt=coarse_logstretch_tgt,
        silence_coarse_logstretch_tgt=silence_coarse_logstretch_tgt,
        coarse_correction_tgt=coarse_correction_tgt,
        coarse_duration_tgt=coarse_duration_tgt,
        residual_logstretch_tgt=residual_logstretch_tgt,
        global_bias_tgt=global_bias_tgt,
        global_bias_tgt_support_mass=global_bias_tgt_support_mass,
        global_bias_tgt_support_count=global_bias_tgt_support_count,
        coarse_target_speech_conf_mean=coarse_target_speech_conf_mean,
        local_residual_tgt=local_residual_tgt,
        local_residual_tgt_center=local_residual_tgt_center,
        local_residual_tgt_abs_mean=local_residual_tgt_abs_mean,
        beta1_tgt=beta1_tgt,
        residual_logstretch_tgt_mean=residual_logstretch_tgt_mean,
        prefix_duration_tgt=prefix_duration_tgt,
        prompt_basis_activation=prompt_targets["prompt_basis_activation"],
        prompt_random_target_tgt=prompt_targets["prompt_random_target_tgt"],
        prompt_mask=prompt_targets["prompt_mask"],
        prompt_fit_mask=prompt_targets["prompt_fit_mask"],
        prompt_eval_mask=prompt_targets["prompt_eval_mask"],
        prompt_operator_fit_pred=prompt_targets["prompt_operator_fit_pred"],
        prompt_operator_cv_fit_pred=prompt_targets["prompt_operator_cv_fit_pred"],
        prompt_role_attn=prompt_targets["prompt_role_attn"],
        prompt_role_fit_pred=prompt_targets["prompt_role_fit_pred"],
        prompt_role_value=prompt_targets["prompt_role_value"],
        prompt_role_var=prompt_targets["prompt_role_var"],
        prompt_log_duration=prompt_targets["prompt_log_duration"],
        prompt_log_residual=prompt_targets["prompt_log_residual"],
        consistency_duration_tgt=consistency_duration_tgt,
        consistency_mask=consistency_mask,
        consistency_logstretch_tgt=consistency_logstretch_tgt,
        consistency_local_residual_tgt=consistency_local_residual_tgt,
        lambda_dur=float(config.lambda_dur),
        lambda_op=float(config.lambda_op),
        lambda_pref=float(config.lambda_pref),
        lambda_bias=float(config.lambda_bias),
        lambda_base=float(config.lambda_base),
        lambda_cons=float(config.lambda_cons),
        lambda_zero=float(config.lambda_zero),
        lambda_ortho=float(config.lambda_ortho),
        lambda_silence_aux=float(config.lambda_silence_aux),
        baseline_pretrain_only=baseline_pretrain_only,
        minimal_v1_profile=bool(getattr(config, "minimal_v1_profile", False)),
    )


def _masked_duration_v3_median(
    values: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    mask_bool = mask > 0.5
    if isinstance(weight, torch.Tensor):
        return masked_weighted_median_batch(values.float(), mask_bool, weight.float())
    return masked_true_median_batch(values.float(), mask_bool)


def _build_duration_v3_prefix_median_seq(
    *,
    values: torch.Tensor,
    update_mask: torch.Tensor,
    output_mask: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size, num_units = values.shape
    seq = values.new_zeros(values.shape)
    weight_f = weight.float() if isinstance(weight, torch.Tensor) else None
    for batch_idx in range(batch_size):
        running_values: list[torch.Tensor] = []
        running_weights: list[torch.Tensor] = []
        for unit_idx in range(num_units):
            if float(update_mask[batch_idx, unit_idx].item()) > 0.5:
                running_values.append(values[batch_idx, unit_idx : unit_idx + 1])
                if weight_f is not None:
                    running_weights.append(weight_f[batch_idx, unit_idx : unit_idx + 1].clamp_min(1.0e-4))
            if not running_values:
                continue
            prefix_values = torch.cat(running_values, dim=0)
            if weight_f is None:
                seq[batch_idx, unit_idx] = masked_true_median_batch(
                    prefix_values.reshape(1, -1),
                    torch.ones_like(prefix_values.reshape(1, -1), dtype=torch.bool),
                ).reshape(())
                continue
            prefix_weight = torch.cat(running_weights, dim=0)
            seq[batch_idx, unit_idx] = masked_weighted_median_batch(
                prefix_values.reshape(1, -1),
                torch.ones_like(prefix_values.reshape(1, -1), dtype=torch.bool),
                prefix_weight.reshape(1, -1),
            ).reshape(())
    return seq * output_mask.float()


def _build_duration_v3_silence_tau(
    *,
    prediction_anchor: torch.Tensor,
    committed_silence_mask: torch.Tensor,
    sep_hint: torch.Tensor | None,
    boundary_cue: torch.Tensor | None,
    max_silence_logstretch: float,
    short_gap_scale: float = 0.35,
    minimal_v1_profile: bool = False,
) -> torch.Tensor:
    return build_silence_tau_surface(
        prediction_anchor=prediction_anchor,
        committed_silence_mask=committed_silence_mask,
        sep_hint=sep_hint,
        boundary_cue=boundary_cue,
        max_silence_logstretch=max_silence_logstretch,
        short_gap_scale=short_gap_scale,
        minimal_v1_profile=minimal_v1_profile,
    )


def _build_duration_v3_baseline_targets(
    *,
    unit_duration_tgt: torch.Tensor,
    unit_anchor_base: torch.Tensor,
    speech_mask: torch.Tensor,
    baseline_target_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized_mode = str(baseline_target_mode or "deglobalized").strip().lower()
    if normalized_mode not in {"raw", "deglobalized"}:
        raise ValueError("rhythm_v3_baseline_target_mode must be one of: raw, deglobalized")
    baseline_mask = speech_mask.float()
    baseline_global_tgt = unit_duration_tgt.new_zeros((unit_duration_tgt.size(0), 1), dtype=torch.float32)
    if normalized_mode == "deglobalized":
        log_residual = (
            torch.log(unit_duration_tgt.float().clamp_min(1.0e-6))
            - torch.log(unit_anchor_base.float().clamp_min(1.0e-6))
        ) * baseline_mask
        baseline_global_tgt = _masked_duration_v3_median(log_residual, baseline_mask).detach()
    baseline_log_target = (
        torch.log(unit_duration_tgt.float().clamp_min(1.0e-6))
        - baseline_global_tgt
    ) * baseline_mask
    baseline_duration_tgt = torch.exp(baseline_log_target) * baseline_mask
    return baseline_duration_tgt.detach(), baseline_mask.detach(), baseline_global_tgt.detach()


def _resolve_duration_v3_target(
    *,
    sample: dict,
    unit_mask: torch.Tensor,
    config: DurationV3TargetBuildConfig,
) -> torch.Tensor:
    unit_duration_tgt = sample.get("unit_duration_tgt")
    if unit_duration_tgt is None:
        raise ValueError("Duration V3 training requires an explicit unit duration target under unit_duration_tgt.")
    if not isinstance(unit_duration_tgt, torch.Tensor):
        raise TypeError(f"Duration V3 target must be a tensor, got {type(unit_duration_tgt)!r}")
    if unit_duration_tgt.dim() != 2:
        raise ValueError(f"Duration V3 target must be rank-2 [B, U], got shape={tuple(unit_duration_tgt.shape)}")
    if unit_duration_tgt.size(0) != unit_mask.size(0):
        raise ValueError(
            f"Duration V3 target batch mismatch: target={tuple(unit_duration_tgt.shape)}, unit_mask={tuple(unit_mask.shape)}"
        )
    return _align_duration_v3_surface(
        tensor=unit_duration_tgt,
        unit_mask=unit_mask,
        strict_target_alignment=bool(config.strict_target_alignment),
    )


def _align_duration_v3_surface(
    *,
    tensor: torch.Tensor,
    unit_mask: torch.Tensor,
    strict_target_alignment: bool,
) -> torch.Tensor:
    if tensor.size(1) == unit_mask.size(1):
        return tensor
    if strict_target_alignment:
        raise ValueError(
            "Duration V3 target/unit alignment mismatch: "
            f"target={tuple(tensor.shape)}, unit_mask={tuple(unit_mask.shape)}"
        )
    aligned = tensor[:, : unit_mask.size(1)]
    if aligned.size(1) < unit_mask.size(1):
        pad = aligned.new_zeros((aligned.size(0), unit_mask.size(1) - aligned.size(1)))
        aligned = torch.cat([aligned, pad], dim=1)
    return aligned


def _resolve_duration_v3_prediction_anchor(
    *,
    unit_batch,
    unit_mask: torch.Tensor,
    speech_mask: torch.Tensor,
    anchor_mode: str,
) -> torch.Tensor:
    normalized_mode = str(anchor_mode or "baseline").strip().lower()
    if normalized_mode not in {"baseline", "source_observed"}:
        raise ValueError("rhythm_v3_anchor_mode must be one of: baseline, source_observed")
    if normalized_mode == "baseline":
        return unit_batch.unit_anchor_base.float() * unit_mask.float()
    source_duration_obs = getattr(unit_batch, "source_duration_obs", None)
    if not isinstance(source_duration_obs, torch.Tensor):
        raise ValueError("rhythm_v3_anchor_mode='source_observed' requires source_duration_obs in rhythm_unit_batch.")
    observed = source_duration_obs.float().clamp_min(1.0e-4) * speech_mask.float()
    fallback = unit_batch.unit_anchor_base.float().clamp_min(1.0e-4) * speech_mask.float()
    return torch.where(observed > 0.0, observed, fallback)


def _build_duration_v3_causal_local_rate_seq(
    *,
    log_anchor: torch.Tensor,
    speech_mask: torch.Tensor,
    init_local_rate: torch.Tensor | None = None,
    default_init_rate: torch.Tensor | float | None = None,
    decay: float = 0.95,
) -> torch.Tensor:
    seq, _ = build_causal_local_rate_seq(
        observed_log=log_anchor,
        speech_mask=speech_mask,
        init_rate=init_local_rate,
        default_init_rate=default_init_rate,
        decay=decay,
    )
    return seq


def _fit_batch_shared_beta1(
    *,
    analytic_gap: torch.Tensor,
    full_logstretch: torch.Tensor,
    speech_mask: torch.Tensor,
    weight: torch.Tensor | None = None,
    beta1_min: float = 0.7,
    beta1_max: float = 1.3,
    min_points: int = 24,
    min_var: float = 2.5e-3,
) -> torch.Tensor:
    w = speech_mask.float()
    if isinstance(weight, torch.Tensor):
        w = w * weight.float()
    valid = w > 1.0e-6
    if int(valid.sum().item()) < int(min_points):
        return analytic_gap.new_ones((analytic_gap.size(0), 1))
    a = analytic_gap[valid]
    z = full_logstretch[valid]
    ww = w[valid]
    ww_sum = ww.sum().clamp_min(1.0e-6)
    mean_a = (a * ww).sum() / ww_sum
    mean_z = (z * ww).sum() / ww_sum
    var_a = (((a - mean_a) ** 2) * ww).sum() / ww_sum
    if float(var_a.item()) < float(min_var):
        beta = 1.0
    else:
        cov_az = (((a - mean_a) * (z - mean_z)) * ww).sum() / ww_sum
        beta = float((cov_az / var_a).clamp(float(beta1_min), float(beta1_max)).item())
    return analytic_gap.new_full((analytic_gap.size(0), 1), beta)


def _resolve_duration_v3_prompt_targets(
    *,
    ref_memory,
    lambda_op: float,
    lambda_zero: float,
    lambda_ortho: float,
) -> dict[str, torch.Tensor | None]:
    global_rate = getattr(ref_memory, "global_rate", None)
    prompt_basis_activation = getattr(ref_memory, "prompt_basis_activation", None)
    prompt_random_target_tgt = getattr(ref_memory, "prompt_random_target", None)
    prompt_mask = getattr(ref_memory, "prompt_mask", None)
    prompt_fit_mask = getattr(ref_memory, "prompt_fit_mask", None)
    prompt_eval_mask = getattr(ref_memory, "prompt_eval_mask", None)
    prompt_operator_fit_pred = getattr(ref_memory, "prompt_operator_fit", None)
    prompt_operator_cv_fit_pred = getattr(ref_memory, "prompt_operator_cv_fit", None)
    prompt_role_attn = getattr(ref_memory, "prompt_role_attn", None)
    prompt_role_fit_pred = getattr(ref_memory, "prompt_role_fit", None)
    prompt_role_value = getattr(ref_memory, "role_value", None)
    prompt_role_var = getattr(ref_memory, "role_var", None)
    prompt_log_duration = getattr(ref_memory, "prompt_log_duration", None)
    prompt_log_residual = getattr(ref_memory, "prompt_log_residual", None)
    if float(lambda_op) > 0.0 and not isinstance(prompt_random_target_tgt, torch.Tensor):
        residual_ready = all(
            isinstance(value, torch.Tensor)
            for value in (prompt_role_fit_pred, prompt_log_residual, prompt_mask)
        )
        role_ready = all(
            isinstance(value, torch.Tensor)
            for value in (prompt_role_attn, prompt_role_value, prompt_role_var, prompt_log_duration, prompt_mask)
        )
        if not residual_ready and not role_ready:
            raise ValueError(
                "Duration V3 prompt-memory loss requires role prompt targets or operator prompt targets in output['rhythm_ref_conditioning']."
            )
    if float(lambda_zero) > 0.0:
        if not isinstance(prompt_operator_fit_pred, torch.Tensor) or not isinstance(prompt_mask, torch.Tensor):
            raise ValueError(
                "Duration V3 zero-mean identifiability loss requires prompt_operator_fit and prompt_mask in output['rhythm_ref_conditioning']."
            )
    if float(lambda_ortho) > 0.0:
        if not isinstance(prompt_basis_activation, torch.Tensor) or not isinstance(prompt_mask, torch.Tensor):
            raise ValueError(
                "Duration V3 orthogonality loss requires prompt_basis_activation and prompt_mask in output['rhythm_ref_conditioning']."
            )
    return {
        "global_rate": global_rate.float().detach() if isinstance(global_rate, torch.Tensor) else None,
        "prompt_basis_activation": (
            prompt_basis_activation.float() if isinstance(prompt_basis_activation, torch.Tensor) else None
        ),
        "prompt_random_target_tgt": (
            prompt_random_target_tgt.float() if isinstance(prompt_random_target_tgt, torch.Tensor) else None
        ),
        "prompt_mask": prompt_mask.float() if isinstance(prompt_mask, torch.Tensor) else None,
        "prompt_fit_mask": prompt_fit_mask.float() if isinstance(prompt_fit_mask, torch.Tensor) else None,
        "prompt_eval_mask": prompt_eval_mask.float() if isinstance(prompt_eval_mask, torch.Tensor) else None,
        "prompt_operator_fit_pred": (
            prompt_operator_fit_pred.float() if isinstance(prompt_operator_fit_pred, torch.Tensor) else None
        ),
        "prompt_operator_cv_fit_pred": (
            prompt_operator_cv_fit_pred.float() if isinstance(prompt_operator_cv_fit_pred, torch.Tensor) else None
        ),
        "prompt_role_attn": prompt_role_attn.float() if isinstance(prompt_role_attn, torch.Tensor) else None,
        "prompt_role_fit_pred": (
            prompt_role_fit_pred.float() if isinstance(prompt_role_fit_pred, torch.Tensor) else None
        ),
        "prompt_role_value": prompt_role_value.float().detach() if isinstance(prompt_role_value, torch.Tensor) else None,
        "prompt_role_var": prompt_role_var.float().detach() if isinstance(prompt_role_var, torch.Tensor) else None,
        "prompt_log_duration": (
            prompt_log_duration.float().detach() if isinstance(prompt_log_duration, torch.Tensor) else None
        ),
        "prompt_log_residual": (
            prompt_log_residual.float().detach() if isinstance(prompt_log_residual, torch.Tensor) else None
        ),
    }


def _resolve_duration_v3_minimal_prompt_targets(
    *,
    ref_memory,
) -> dict[str, torch.Tensor | None]:
    global_rate = getattr(ref_memory, "global_rate", None)
    return {
        "global_rate": global_rate.float().detach() if isinstance(global_rate, torch.Tensor) else None,
        "prompt_basis_activation": None,
        "prompt_random_target_tgt": None,
        "prompt_mask": None,
        "prompt_fit_mask": None,
        "prompt_eval_mask": None,
        "prompt_operator_fit_pred": None,
        "prompt_operator_cv_fit_pred": None,
        "prompt_role_attn": None,
        "prompt_role_fit_pred": None,
        "prompt_role_value": None,
        "prompt_role_var": None,
        "prompt_log_duration": None,
        "prompt_log_residual": None,
    }


def _build_duration_v3_consistency_targets(
    *,
    state_prev,
    unit_mask: torch.Tensor,
    committed_mask: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    consistency_duration_tgt = None
    consistency_mask = None
    if state_prev is not None and isinstance(getattr(state_prev, "cached_duration_exec", None), torch.Tensor):
        cached_duration_exec = state_prev.cached_duration_exec.float().detach()
        consistency_duration_tgt = _align_duration_v3_surface(
            tensor=cached_duration_exec,
            unit_mask=unit_mask,
            strict_target_alignment=False,
        )
        prev_frontier = state_prev.committed_units.long().clamp_min(0)
        prev_frontier = torch.clamp(
            prev_frontier,
            max=int(cached_duration_exec.size(1)),
        )
        steps = torch.arange(unit_mask.size(1), device=unit_mask.device)[None, :]
        consistency_mask = (steps < prev_frontier[:, None]).float() * committed_mask
    return (
        consistency_duration_tgt.float() if isinstance(consistency_duration_tgt, torch.Tensor) else None,
        consistency_mask.float() if isinstance(consistency_mask, torch.Tensor) else None,
    )


def build_identity_rhythm_loss_targets(
    *,
    unit_batch,
    config: RhythmTargetBuildConfig,
) -> RhythmLossTargets:
    unit_mask = unit_batch.unit_mask.float()
    speech_exec_tgt = unit_batch.dur_anchor_src.float() * unit_mask
    pause_exec_tgt = torch.zeros_like(speech_exec_tgt)
    speech_budget_tgt = speech_exec_tgt.sum(dim=1, keepdim=True)
    pause_budget_tgt = torch.zeros_like(speech_budget_tgt)
    return RhythmLossTargets(
        speech_exec_tgt=speech_exec_tgt,
        pause_exec_tgt=pause_exec_tgt,
        speech_budget_tgt=speech_budget_tgt,
        pause_budget_tgt=pause_budget_tgt,
        unit_mask=unit_mask,
        dur_anchor_src=unit_batch.dur_anchor_src,
        unit_logratio_weight=float(config.unit_logratio_weight),
        srmdp_role_consistency_weight=float(config.srmdp_role_consistency_weight),
        srmdp_notimeline_weight=float(config.srmdp_notimeline_weight),
        srmdp_memory_role_weight=float(config.srmdp_memory_role_weight),
        plan_local_weight=float(config.plan_local_weight),
        plan_cum_weight=float(config.plan_cum_weight),
        plan_segment_shape_weight=float(config.plan_segment_shape_weight),
        plan_pause_release_weight=float(config.plan_pause_release_weight),
        sample_confidence=torch.ones((unit_mask.size(0), 1), device=unit_mask.device),
        distill_exec_weight=float(config.distill_exec_weight),
        distill_budget_weight=float(config.distill_budget_weight),
        distill_allocation_weight=float(config.distill_allocation_weight),
        distill_prefix_weight=float(config.distill_prefix_weight),
        distill_speech_shape_weight=float(config.distill_speech_shape_weight),
        distill_pause_shape_weight=float(config.distill_pause_shape_weight),
        budget_raw_weight=float(config.budget_raw_weight),
        budget_exec_weight=float(config.budget_exec_weight),
        pause_boundary_weight=float(config.pause_boundary_weight),
        feasible_debt_weight=float(config.feasible_debt_weight),
        pause_event_weight=float(config.pause_event_weight),
        pause_support_weight=float(config.pause_support_weight),
        pause_allocation_weight=float(config.pause_allocation_weight),
        pause_event_threshold=float(config.pause_event_threshold),
        pause_event_temperature=float(config.pause_event_temperature),
        pause_event_pos_weight=float(config.pause_event_pos_weight),
    )


def scale_rhythm_loss_terms(
    rhythm_losses: dict[str, torch.Tensor],
    *,
    hparams,
    cumplan_lambda: float,
) -> dict[str, torch.Tensor]:
    def _resolve_prefix_state_loss() -> torch.Tensor:
        prefix_state = rhythm_losses.get("rhythm_prefix_state")
        if isinstance(prefix_state, torch.Tensor):
            return prefix_state
        cumplan = rhythm_losses.get("rhythm_cumplan")
        if isinstance(cumplan, torch.Tensor):
            return cumplan
        return rhythm_losses["rhythm_carry"]

    def _scaled_detached(
        key: str,
        scale: float,
        *,
        fallback_key: str | None = None,
        allow_missing: bool = False,
    ) -> torch.Tensor:
        def _zero_like_losses() -> torch.Tensor:
            for candidate in rhythm_losses.values():
                if isinstance(candidate, torch.Tensor):
                    return candidate.new_tensor(0.0)
            return torch.tensor(0.0)

        value = rhythm_losses.get(key)
        if not isinstance(value, torch.Tensor):
            if fallback_key is not None:
                fallback = rhythm_losses.get(fallback_key)
                if not isinstance(fallback, torch.Tensor) and fallback_key == "rhythm_prefix_state":
                    fallback = _resolve_prefix_state_loss()
                if isinstance(fallback, torch.Tensor):
                    value = fallback
            if not isinstance(value, torch.Tensor):
                if allow_missing:
                    return _zero_like_losses()
                raise KeyError(key)
        return (value * float(scale)).detach()

    lambda_budget = float(hparams.get("lambda_rhythm_budget", 0.25))
    lambda_distill = float(hparams.get("lambda_rhythm_distill", 0.0) or 0.0)
    lambda_plan = float(hparams.get("lambda_rhythm_plan", 0.0) or 0.0)
    lambda_guidance = float(hparams.get("lambda_rhythm_guidance", 0.0) or 0.0)
    plan_local_weight = float(hparams.get("rhythm_plan_local_weight", 0.5))
    plan_cum_weight = float(hparams.get("rhythm_plan_cum_weight", 1.0))
    plan_segment_shape_weight = float(hparams.get("rhythm_plan_segment_shape_weight", 0.0))
    plan_pause_release_weight = float(hparams.get("rhythm_plan_pause_release_weight", 0.0))
    distill_budget_weight = float(hparams.get("rhythm_distill_budget_weight", 0.5))
    prefix_state = _resolve_prefix_state_loss()
    scaled_prefix_state = prefix_state * float(cumplan_lambda)
    scaled_distill = rhythm_losses["rhythm_distill"] * lambda_distill
    scaled = {
        "rhythm_exec_speech": rhythm_losses["rhythm_exec_speech"] * float(hparams.get("lambda_rhythm_exec_speech", 1.0)),
        "rhythm_exec_stretch": _scaled_detached(
            "rhythm_exec_stretch",
            float(hparams.get("lambda_rhythm_exec_speech", 1.0)),
            allow_missing=True,
        ),
        "rhythm_exec_stretch_base": _scaled_detached(
            "rhythm_exec_stretch_base",
            float(hparams.get("lambda_rhythm_exec_speech", 1.0)),
            allow_missing=True,
        ),
        "rhythm_srmdp_role_consistency": _scaled_detached(
            "rhythm_srmdp_role_consistency",
            float(hparams.get("lambda_rhythm_exec_speech", 1.0)),
            allow_missing=True,
        ),
        "rhythm_srmdp_notimeline": _scaled_detached(
            "rhythm_srmdp_notimeline",
            float(hparams.get("lambda_rhythm_exec_speech", 1.0)),
            allow_missing=True,
        ),
        "rhythm_srmdp_memory_role": _scaled_detached(
            "rhythm_srmdp_memory_role",
            float(hparams.get("lambda_rhythm_exec_speech", 1.0)),
            allow_missing=True,
        ),
        "rhythm_exec_pause": rhythm_losses["rhythm_exec_pause"] * float(hparams.get("lambda_rhythm_exec_pause", 1.0)),
        "rhythm_exec_pause_value": _scaled_detached(
            "rhythm_exec_pause_value",
            float(hparams.get("lambda_rhythm_exec_pause", 1.0)),
            fallback_key="rhythm_exec_pause",
        ),
        "rhythm_pause_event": _scaled_detached(
            "rhythm_pause_event",
            float(hparams.get("lambda_rhythm_exec_pause", 1.0)),
            allow_missing=True,
        ),
        "rhythm_pause_support": _scaled_detached(
            "rhythm_pause_support",
            float(hparams.get("lambda_rhythm_exec_pause", 1.0)),
            allow_missing=True,
        ),
        "rhythm_pause_allocation": _scaled_detached(
            "rhythm_pause_allocation",
            float(hparams.get("lambda_rhythm_exec_pause", 1.0)),
            allow_missing=True,
        ),
        "rhythm_budget": rhythm_losses["rhythm_budget"] * lambda_budget,
        "rhythm_budget_raw_surface": _scaled_detached(
            "rhythm_budget_raw_surface",
            lambda_budget,
            fallback_key="rhythm_budget",
        ),
        "rhythm_budget_exec_surface": _scaled_detached(
            "rhythm_budget_exec_surface",
            lambda_budget,
            fallback_key="rhythm_budget",
        ),
        "rhythm_budget_total_surface": _scaled_detached(
            "rhythm_budget_total_surface",
            lambda_budget,
            fallback_key="rhythm_budget_raw_surface",
        ),
        "rhythm_budget_pause_share_surface": _scaled_detached(
            "rhythm_budget_pause_share_surface",
            lambda_budget,
            fallback_key="rhythm_budget_exec_surface",
        ),
        "rhythm_feasible_debt": (
            rhythm_losses["rhythm_feasible_debt"]
            * lambda_budget
            * float(hparams.get("rhythm_feasible_debt_weight", 0.05))
        ).detach(),
        "rhythm_prefix_clock": _scaled_detached(
            "rhythm_prefix_clock",
            cumplan_lambda,
            fallback_key="rhythm_prefix_state",
        ),
        "rhythm_prefix_backlog": _scaled_detached(
            "rhythm_prefix_backlog",
            cumplan_lambda,
            fallback_key="rhythm_prefix_state",
        ),
        "rhythm_prefix_state": scaled_prefix_state,
        "rhythm_cumplan": scaled_prefix_state.detach(),
        "rhythm_carry": scaled_prefix_state.detach(),
        "rhythm_plan": rhythm_losses["rhythm_plan"] * lambda_plan,
        "rhythm_plan_local": (rhythm_losses["rhythm_plan_local"] * lambda_plan * plan_local_weight).detach(),
        "rhythm_plan_cum": (rhythm_losses["rhythm_plan_cum"] * lambda_plan * plan_cum_weight).detach(),
        "rhythm_plan_segment_shape": _scaled_detached(
            "rhythm_plan_segment_shape",
            lambda_plan * plan_segment_shape_weight,
            allow_missing=True,
        ),
        "rhythm_plan_pause_release": _scaled_detached(
            "rhythm_plan_pause_release",
            lambda_plan * plan_pause_release_weight,
            allow_missing=True,
        ),
        "rhythm_guidance": rhythm_losses["rhythm_guidance"] * lambda_guidance,
        "rhythm_distill": scaled_distill,
        "rhythm_distill_student": scaled_distill.detach(),
        "rhythm_distill_exec": _scaled_detached("rhythm_distill_exec", lambda_distill),
        "rhythm_distill_budget": _scaled_detached(
            "rhythm_distill_budget",
            lambda_distill * distill_budget_weight,
        ),
        "rhythm_distill_budget_raw_surface": _scaled_detached(
            "rhythm_distill_budget_raw_surface",
            lambda_distill * distill_budget_weight,
            fallback_key="rhythm_distill_budget",
        ),
        "rhythm_distill_budget_exec_surface": _scaled_detached(
            "rhythm_distill_budget_exec_surface",
            lambda_distill * distill_budget_weight,
            fallback_key="rhythm_distill_budget",
        ),
        "rhythm_distill_budget_total_surface": _scaled_detached(
            "rhythm_distill_budget_total_surface",
            lambda_distill * distill_budget_weight,
            fallback_key="rhythm_distill_budget_raw_surface",
        ),
        "rhythm_distill_budget_pause_share_surface": _scaled_detached(
            "rhythm_distill_budget_pause_share_surface",
            lambda_distill * distill_budget_weight,
            fallback_key="rhythm_distill_budget_exec_surface",
        ),
        "rhythm_distill_prefix": _scaled_detached(
            "rhythm_distill_prefix",
            lambda_distill * float(hparams.get("rhythm_distill_prefix_weight", 0.25)),
        ),
        "rhythm_distill_speech_shape": _scaled_detached(
            "rhythm_distill_speech_shape",
            lambda_distill * float(hparams.get("rhythm_distill_speech_shape_weight", 0.0)),
        ),
        "rhythm_distill_pause_shape": _scaled_detached(
            "rhythm_distill_pause_shape",
            lambda_distill * float(hparams.get("rhythm_distill_pause_shape_weight", 0.0)),
        ),
        "rhythm_distill_allocation": _scaled_detached(
            "rhythm_distill_allocation",
            lambda_distill * float(hparams.get("rhythm_distill_allocation_weight", 0.5)),
        ),
    }
    for key in (
        "rhythm_distill_same_source_exec",
        "rhythm_distill_same_source_budget",
        "rhythm_distill_same_source_prefix",
        "rhythm_distill_same_source_allocation",
        "rhythm_distill_same_source_shape",
        "rhythm_distill_same_source_any",
        "rhythm_distill_context_match",
    ):
        value = rhythm_losses.get(key)
        if isinstance(value, torch.Tensor):
            scaled[key] = value.detach()
    return scaled
