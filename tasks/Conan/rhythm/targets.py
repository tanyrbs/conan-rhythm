from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import torch

from .losses import RhythmLossTargets


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
    dedupe_primary_teacher_cache_distill: bool = True

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
        plan_local_weight=float(config.plan_local_weight),
        plan_cum_weight=float(config.plan_cum_weight),
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
        plan_local_weight=float(config.plan_local_weight),
        plan_cum_weight=float(config.plan_cum_weight),
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
    ) -> torch.Tensor:
        value = rhythm_losses.get(key)
        if not isinstance(value, torch.Tensor):
            if fallback_key is None:
                raise KeyError(key)
            fallback = rhythm_losses.get(fallback_key)
            if not isinstance(fallback, torch.Tensor) and fallback_key == "rhythm_prefix_state":
                fallback = _resolve_prefix_state_loss()
            value = fallback
        return (value * float(scale)).detach()

    lambda_budget = float(hparams.get("lambda_rhythm_budget", 0.25))
    lambda_distill = float(hparams.get("lambda_rhythm_distill", 0.0) or 0.0)
    lambda_plan = float(hparams.get("lambda_rhythm_plan", 0.0) or 0.0)
    lambda_guidance = float(hparams.get("lambda_rhythm_guidance", 0.0) or 0.0)
    plan_local_weight = float(hparams.get("rhythm_plan_local_weight", 0.5))
    plan_cum_weight = float(hparams.get("rhythm_plan_cum_weight", 1.0))
    distill_budget_weight = float(hparams.get("rhythm_distill_budget_weight", 0.5))
    prefix_state = _resolve_prefix_state_loss()
    scaled_prefix_state = prefix_state * float(cumplan_lambda)
    scaled_distill = rhythm_losses["rhythm_distill"] * lambda_distill
    scaled = {
        "rhythm_exec_speech": rhythm_losses["rhythm_exec_speech"] * float(hparams.get("lambda_rhythm_exec_speech", 1.0)),
        "rhythm_exec_pause": rhythm_losses["rhythm_exec_pause"] * float(hparams.get("lambda_rhythm_exec_pause", 1.0)),
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
    ):
        value = rhythm_losses.get(key)
        if isinstance(value, torch.Tensor):
            scaled[key] = value.detach()
    return scaled
