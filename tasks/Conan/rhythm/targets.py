from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from .losses import RhythmLossTargets


@dataclass(frozen=True)
class RhythmTargetBuildConfig:
    primary_target_surface: str
    distill_surface: str
    lambda_guidance: float
    lambda_distill: float
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

    @property
    def use_guidance(self) -> bool:
        return self.lambda_guidance > 0.0

    @property
    def use_distill(self) -> bool:
        return self.lambda_distill > 0.0

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
    return DistillConfidenceBundle(
        shared=shared,
        exec=normalize_component_confidence(
            confidence_bundle.exec,
            fallback_confidence=shared,
            batch_size=batch_size,
            device=device,
        ),
        budget=normalize_component_confidence(
            confidence_bundle.budget,
            fallback_confidence=shared,
            batch_size=batch_size,
            device=device,
        ),
        prefix=normalize_component_confidence(
            confidence_bundle.prefix,
            fallback_confidence=shared,
            batch_size=batch_size,
            device=device,
        ),
        allocation=normalize_component_confidence(
            confidence_bundle.allocation,
            fallback_confidence=shared,
            batch_size=batch_size,
            device=device,
        ),
        shape=normalize_component_confidence(
            confidence_bundle.shape,
            fallback_confidence=normalize_component_confidence(
                confidence_bundle.exec,
                fallback_confidence=shared,
                batch_size=batch_size,
                device=device,
            ),
            batch_size=batch_size,
            device=device,
        ),
    )


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
    required_keys = (
        keys.target_speech_key,
        keys.target_pause_key,
        keys.target_speech_budget_key,
        keys.target_pause_budget_key,
    )
    if not all(key in sample for key in required_keys):
        return None

    guidance_speech = sample.get("rhythm_guidance_speech_tgt") if config.use_guidance else None
    guidance_pause = sample.get(keys.guidance_pause_key) if config.use_guidance else None

    distill_speech = None
    distill_pause = None
    distill_speech_budget = None
    distill_pause_budget = None
    distill_allocation = None
    distill_prefix_clock = None
    distill_prefix_backlog = None
    distill_confidences = DistillConfidenceBundle()

    if config.use_distill and config.distill_surface in {"auto", "cache"}:
        distill_speech = sample.get("rhythm_teacher_speech_exec_tgt")
        distill_pause = sample.get(keys.teacher_pause_key)
        distill_speech_budget = sample.get("rhythm_teacher_speech_budget_tgt") if config.use_distill_budget else None
        distill_pause_budget = sample.get(keys.teacher_pause_budget_key) if config.use_distill_budget else None
        distill_allocation = sample.get("rhythm_teacher_allocation_tgt") if config.use_distill_allocation else None
        distill_prefix_clock = sample.get("rhythm_teacher_prefix_clock_tgt") if config.use_distill_prefix else None
        distill_prefix_backlog = sample.get("rhythm_teacher_prefix_backlog_tgt") if config.use_distill_prefix else None
        distill_confidences = DistillConfidenceBundle(
            shared=_detach_optional(sample.get("rhythm_teacher_confidence")),
            exec=_detach_optional(sample.get("rhythm_teacher_confidence_exec")),
            budget=_detach_optional(sample.get("rhythm_teacher_confidence_budget")),
            prefix=_detach_optional(sample.get("rhythm_teacher_confidence_prefix")),
            allocation=_detach_optional(sample.get("rhythm_teacher_confidence_allocation")),
            shape=_detach_optional(sample.get("rhythm_teacher_confidence_shape")),
        )

    if (
        config.use_distill
        and distill_speech is None
        and config.distill_surface in {"auto", "offline"}
        and runtime_teacher is not None
    ):
        distill_speech = runtime_teacher.speech_duration_exec.detach()
        distill_pause = getattr(runtime_teacher, "blank_duration_exec", runtime_teacher.pause_after_exec).detach()
        (
            distill_speech,
            distill_pause,
            distill_speech_budget,
            distill_pause_budget,
            distill_allocation,
            distill_prefix_clock,
            distill_prefix_backlog,
        ) = slice_rhythm_surface_to_student(
            speech_exec=distill_speech,
            pause_exec=distill_pause,
            student_units=unit_batch.dur_anchor_src.size(1),
            dur_anchor_src=unit_batch.dur_anchor_src,
            unit_mask=unit_batch.unit_mask,
        )
        if not config.use_distill_budget:
            distill_speech_budget = None
            distill_pause_budget = None
        if not config.use_distill_allocation:
            distill_allocation = None
        if not config.use_distill_prefix:
            distill_prefix_clock = None
            distill_prefix_backlog = None
        offline_confidences = offline_confidences or DistillConfidenceBundle()
        distill_confidences = DistillConfidenceBundle(
            shared=_detach_optional(offline_confidences.shared),
            exec=_detach_optional(offline_confidences.exec),
            budget=_detach_optional(offline_confidences.budget),
            prefix=_detach_optional(offline_confidences.prefix),
            allocation=_detach_optional(offline_confidences.allocation),
            shape=_detach_optional(offline_confidences.shape),
        )
        if distill_confidences.shared is None:
            distill_confidences = DistillConfidenceBundle(
                shared=distill_speech.new_ones((distill_speech.size(0), 1)),
                exec=distill_confidences.exec,
                budget=distill_confidences.budget,
                prefix=distill_confidences.prefix,
                allocation=distill_confidences.allocation,
                shape=distill_confidences.shape,
            )

    if (
        config.use_distill
        and distill_speech is None
        and config.distill_surface in {"auto", "algorithmic"}
        and algorithmic_teacher is not None
    ):
        distill_speech = algorithmic_teacher.speech_exec_tgt.detach()
        distill_pause = algorithmic_teacher.pause_exec_tgt.detach()
        distill_speech_budget = algorithmic_teacher.speech_budget_tgt.detach() if config.use_distill_budget else None
        distill_pause_budget = algorithmic_teacher.pause_budget_tgt.detach() if config.use_distill_budget else None
        distill_allocation = algorithmic_teacher.allocation_tgt.detach() if config.use_distill_allocation else None
        distill_prefix_clock = algorithmic_teacher.prefix_clock_tgt.detach() if config.use_distill_prefix else None
        distill_prefix_backlog = algorithmic_teacher.prefix_backlog_tgt.detach() if config.use_distill_prefix else None
        distill_confidences = DistillConfidenceBundle(shared=algorithmic_teacher.confidence.detach())

    if config.use_distill and (distill_speech is None or distill_pause is None):
        distill_speech = None
        distill_pause = None
        distill_speech_budget = None
        distill_pause_budget = None
        distill_allocation = None
        distill_prefix_clock = None
        distill_prefix_backlog = None
        distill_confidences = DistillConfidenceBundle()

    if distill_speech is not None and distill_pause is not None:
        if distill_speech.size(1) != unit_batch.dur_anchor_src.size(1):
            (
                distill_speech,
                distill_pause,
                distill_speech_budget,
                distill_pause_budget,
                distill_allocation,
                distill_prefix_clock,
                distill_prefix_backlog,
            ) = slice_rhythm_surface_to_student(
                speech_exec=distill_speech,
                pause_exec=distill_pause,
                student_units=unit_batch.dur_anchor_src.size(1),
                dur_anchor_src=unit_batch.dur_anchor_src,
                unit_mask=unit_batch.unit_mask,
            )
        if config.use_distill_allocation and distill_allocation is None:
            distill_allocation = (distill_speech.float() + distill_pause.float()) * unit_batch.unit_mask.float()
        if config.use_distill_prefix and (distill_prefix_clock is None or distill_prefix_backlog is None):
            distill_prefix_clock, distill_prefix_backlog = build_prefix_carry_from_exec(
                distill_speech,
                distill_pause,
                unit_batch.dur_anchor_src,
                unit_batch.unit_mask,
            )

    if config.use_distill:
        distill_confidences = _normalize_distill_confidences(
            confidence_bundle=distill_confidences,
            batch_size=unit_batch.dur_anchor_src.size(0),
            device=unit_batch.dur_anchor_src.device,
            normalize_distill_confidence=normalize_distill_confidence,
            normalize_component_confidence=normalize_component_confidence,
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
        sample_confidence=_normalize_optional_confidence(
            _resolve_sample_confidence(
                sample,
                primary_target_surface=config.primary_target_surface,
            ),
            batch_size=unit_batch.dur_anchor_src.size(0),
            device=unit_batch.dur_anchor_src.device,
        ),
        guidance_speech_tgt=guidance_speech,
        guidance_pause_tgt=guidance_pause,
        guidance_confidence=(
            _normalize_optional_confidence(
                sample.get("rhythm_guidance_confidence"),
                batch_size=unit_batch.dur_anchor_src.size(0),
                device=unit_batch.dur_anchor_src.device,
                fallback_confidence=_normalize_optional_confidence(
                    _resolve_sample_confidence(
                        sample,
                        primary_target_surface=config.primary_target_surface,
                    ),
                    batch_size=unit_batch.dur_anchor_src.size(0),
                    device=unit_batch.dur_anchor_src.device,
                ),
            )
            if config.use_guidance
            else None
        ),
        distill_speech_tgt=distill_speech,
        distill_pause_tgt=distill_pause,
        distill_speech_budget_tgt=distill_speech_budget,
        distill_pause_budget_tgt=distill_pause_budget,
        distill_allocation_tgt=distill_allocation,
        distill_prefix_clock_tgt=distill_prefix_clock,
        distill_prefix_backlog_tgt=distill_prefix_backlog,
        distill_confidence=distill_confidences.shared,
        distill_exec_confidence=distill_confidences.exec,
        distill_budget_confidence=distill_confidences.budget,
        distill_prefix_confidence=distill_confidences.prefix,
        distill_allocation_confidence=distill_confidences.allocation,
        distill_shape_confidence=distill_confidences.shape,
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
    lambda_distill = float(hparams.get("lambda_rhythm_distill", 0.0) or 0.0)
    prefix_state = rhythm_losses.get(
        "rhythm_prefix_state",
        rhythm_losses.get("rhythm_cumplan", rhythm_losses["rhythm_carry"]),
    )
    scaled_prefix_state = prefix_state * float(cumplan_lambda)
    return {
        "rhythm_exec_speech": rhythm_losses["rhythm_exec_speech"] * hparams.get("lambda_rhythm_exec_speech", 1.0),
        "rhythm_exec_pause": rhythm_losses["rhythm_exec_pause"] * hparams.get("lambda_rhythm_exec_pause", 1.0),
        "rhythm_budget": rhythm_losses["rhythm_budget"] * hparams.get("lambda_rhythm_budget", 0.25),
        "rhythm_budget_raw_surface": (
            rhythm_losses.get("rhythm_budget_raw_surface", rhythm_losses["rhythm_budget"])
            * hparams.get("lambda_rhythm_budget", 0.25)
        ).detach(),
        "rhythm_budget_exec_surface": (
            rhythm_losses.get("rhythm_budget_exec_surface", rhythm_losses["rhythm_budget"])
            * hparams.get("lambda_rhythm_budget", 0.25)
        ).detach(),
        "rhythm_budget_total_surface": (
            rhythm_losses.get(
                "rhythm_budget_total_surface",
                rhythm_losses.get("rhythm_budget_raw_surface", rhythm_losses["rhythm_budget"]),
            )
            * hparams.get("lambda_rhythm_budget", 0.25)
        ).detach(),
        "rhythm_budget_pause_share_surface": (
            rhythm_losses.get(
                "rhythm_budget_pause_share_surface",
                rhythm_losses.get("rhythm_budget_exec_surface", rhythm_losses["rhythm_budget"]),
            )
            * hparams.get("lambda_rhythm_budget", 0.25)
        ).detach(),
        "rhythm_feasible_debt": (
            rhythm_losses["rhythm_feasible_debt"]
            * hparams.get("lambda_rhythm_budget", 0.25)
            * float(hparams.get("rhythm_feasible_debt_weight", 0.05))
        ).detach(),
        "rhythm_prefix_state": scaled_prefix_state,
        "rhythm_cumplan": scaled_prefix_state.detach(),
        "rhythm_carry": scaled_prefix_state.detach(),
        "rhythm_plan": rhythm_losses["rhythm_plan"] * hparams.get("lambda_rhythm_plan", 0.0),
        "rhythm_guidance": rhythm_losses["rhythm_guidance"] * hparams.get("lambda_rhythm_guidance", 0.0),
        "rhythm_distill": rhythm_losses["rhythm_distill"] * lambda_distill,
        "rhythm_distill_exec": (rhythm_losses["rhythm_distill_exec"] * lambda_distill).detach(),
        "rhythm_distill_budget": (
            rhythm_losses["rhythm_distill_budget"]
            * lambda_distill
            * float(hparams.get("rhythm_distill_budget_weight", 0.5))
        ).detach(),
        "rhythm_distill_budget_raw_surface": (
            rhythm_losses.get("rhythm_distill_budget_raw_surface", rhythm_losses["rhythm_distill_budget"])
            * lambda_distill
            * float(hparams.get("rhythm_distill_budget_weight", 0.5))
        ).detach(),
        "rhythm_distill_budget_exec_surface": (
            rhythm_losses.get("rhythm_distill_budget_exec_surface", rhythm_losses["rhythm_distill_budget"])
            * lambda_distill
            * float(hparams.get("rhythm_distill_budget_weight", 0.5))
        ).detach(),
        "rhythm_distill_budget_total_surface": (
            rhythm_losses.get(
                "rhythm_distill_budget_total_surface",
                rhythm_losses.get("rhythm_distill_budget_raw_surface", rhythm_losses["rhythm_distill_budget"]),
            )
            * lambda_distill
            * float(hparams.get("rhythm_distill_budget_weight", 0.5))
        ).detach(),
        "rhythm_distill_budget_pause_share_surface": (
            rhythm_losses.get(
                "rhythm_distill_budget_pause_share_surface",
                rhythm_losses.get("rhythm_distill_budget_exec_surface", rhythm_losses["rhythm_distill_budget"]),
            )
            * lambda_distill
            * float(hparams.get("rhythm_distill_budget_weight", 0.5))
        ).detach(),
        "rhythm_distill_prefix": (
            rhythm_losses["rhythm_distill_prefix"]
            * lambda_distill
            * float(hparams.get("rhythm_distill_prefix_weight", 0.25))
        ).detach(),
        "rhythm_distill_speech_shape": (
            rhythm_losses["rhythm_distill_speech_shape"]
            * lambda_distill
            * float(hparams.get("rhythm_distill_speech_shape_weight", 0.0))
        ).detach(),
        "rhythm_distill_pause_shape": (
            rhythm_losses["rhythm_distill_pause_shape"]
            * lambda_distill
            * float(hparams.get("rhythm_distill_pause_shape_weight", 0.0))
        ).detach(),
        "rhythm_distill_allocation": (
            rhythm_losses["rhythm_distill_allocation"]
            * lambda_distill
            * float(hparams.get("rhythm_distill_allocation_weight", 0.5))
        ).detach(),
    }
