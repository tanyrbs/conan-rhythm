from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from modules.Conan.rhythm.policy import (
    RhythmHparamsPolicy,
    build_rhythm_hparams_policy,
    resolve_pause_boundary_weight,
)
from modules.Conan.rhythm.surface_metadata import RHYTHM_CACHE_VERSION

from .compat import resolve_duplicate_primary_distill_dedupe_flag
from .profile import detect_rhythm_profile

_DISTILL_COMPONENT_KEYS = (
    "rhythm_distill_exec_weight",
    "rhythm_distill_budget_weight",
    "rhythm_distill_allocation_weight",
    "rhythm_distill_prefix_weight",
    "rhythm_distill_speech_shape_weight",
    "rhythm_distill_pause_shape_weight",
)


@dataclass(frozen=True)
class RhythmStageKnobs:
    target_mode: str
    primary_target_surface: str
    distill_surface: str
    require_cached_teacher: bool
    require_retimed_cache: bool
    enable_dual_mode_teacher: bool
    enable_learned_offline_teacher: bool
    explicit_runtime_teacher_enable: Any
    runtime_offline_teacher_enabled: bool
    runtime_dual_mode_teacher_enabled: bool
    legacy_schedule_flag: bool
    strict_mainline: bool
    optimize_module_only: bool
    teacher_as_main: bool
    apply_train_override: bool
    apply_valid_override: bool
    use_retimed_target_if_available: bool
    retimed_target_mode: str
    retimed_target_start_steps: int
    online_retimed_target_start_steps: int
    use_retimed_pitch_target: bool
    disable_pitch_when_retimed: bool
    retimed_mel_source: str
    binarize_teacher_targets: bool
    disable_mel_adv_when_retimed: bool
    compact_joint_loss: bool
    lambda_distill: float
    lambda_teacher_aux: float
    lambda_mel_adv: float
    lambda_guidance: float
    lambda_plan: float
    lambda_budget: float
    distill_exec_weight: float
    distill_budget_weight: float
    distill_allocation_weight: float
    distill_prefix_weight: float
    distill_speech_shape_weight: float
    distill_pause_shape_weight: float
    explicit_inactive_distill_weights: tuple[float, ...]
    dedupe_teacher_primary_cache_distill: bool
    plan_local_weight: float
    plan_cum_weight: float
    budget_raw_weight: float
    budget_exec_weight: float
    pause_topk_ratio: float
    pause_topk_ratio_train_start: float
    pause_topk_ratio_train_end: float
    pause_topk_ratio_anneal_steps: int
    pause_topk_ratio_warmup_steps: int
    distill_conf_floor: float
    distill_conf_power: float
    source_boundary_scale: float
    source_boundary_scale_train_start: float
    source_boundary_scale_train_end: float
    source_boundary_scale_anneal_steps: int
    source_boundary_scale_warmup_steps: int
    teacher_source_boundary_scale: float
    pause_boundary_weight: float
    pause_boundary_bias_weight: float
    pause_selection_mode: str
    pause_event_weight: float
    pause_support_weight: float
    export_cache_audit_to_sample: bool
    public_losses: tuple[str, ...]
    configured_cache_version: int

    @property
    def has_train_or_valid_override(self) -> bool:
        return self.apply_train_override or self.apply_valid_override

    @property
    def shape_distill_active(self) -> bool:
        return self.distill_speech_shape_weight > 0.0 or self.distill_pause_shape_weight > 0.0

    @property
    def total_distill_component_weight(self) -> float:
        return (
            self.distill_exec_weight
            + self.distill_budget_weight
            + self.distill_allocation_weight
            + self.distill_prefix_weight
            + self.distill_speech_shape_weight
            + self.distill_pause_shape_weight
        )

    @property
    def total_budget_component_weight(self) -> float:
        return self.budget_raw_weight + self.budget_exec_weight

    @property
    def total_plan_component_weight(self) -> float:
        return self.plan_local_weight + self.plan_cum_weight


@dataclass(frozen=True)
class RhythmStageValidationContext:
    hparams: Mapping[str, Any]
    policy: RhythmHparamsPolicy
    stage: str
    profile: str
    knobs: RhythmStageKnobs


def build_stage_validation_context(
    hparams: Mapping[str, Any],
    *,
    config_path: str | None = None,
) -> RhythmStageValidationContext:
    policy = build_rhythm_hparams_policy(hparams, config_path=config_path)
    stage = policy.stage
    retimed_target_start = int(hparams.get("rhythm_retimed_target_start_steps", 0) or 0)
    source_boundary_scale = float(hparams.get("rhythm_source_boundary_scale", 1.0))
    pause_topk_ratio = float(hparams.get("rhythm_projector_pause_topk_ratio", 0.35))
    return RhythmStageValidationContext(
        hparams=hparams,
        policy=policy,
        stage=stage,
        profile=detect_rhythm_profile(hparams, config_path),
        knobs=RhythmStageKnobs(
            target_mode=policy.target_mode,
            primary_target_surface=policy.primary_target_surface,
            distill_surface=policy.distill_surface,
            require_cached_teacher=policy.require_cached_teacher,
            require_retimed_cache=policy.require_retimed_cache,
            enable_dual_mode_teacher=bool(hparams.get("rhythm_enable_dual_mode_teacher", False)),
            enable_learned_offline_teacher=bool(
                hparams.get("rhythm_enable_learned_offline_teacher", False)
            ),
            explicit_runtime_teacher_enable=hparams.get(
                "rhythm_runtime_enable_learned_offline_teacher", None
            ),
            runtime_offline_teacher_enabled=policy.runtime_offline_teacher_enabled,
            runtime_dual_mode_teacher_enabled=policy.runtime_dual_mode_teacher_enabled,
            legacy_schedule_flag=bool(hparams.get("rhythm_schedule_only_stage", False)),
            strict_mainline=policy.strict_mainline,
            optimize_module_only=bool(hparams.get("rhythm_optimize_module_only", False)),
            teacher_as_main=policy.teacher_as_main,
            apply_train_override=bool(hparams.get("rhythm_apply_train_override", False)),
            apply_valid_override=bool(hparams.get("rhythm_apply_valid_override", False)),
            use_retimed_target_if_available=bool(
                hparams.get("rhythm_use_retimed_target_if_available", False)
            ),
            retimed_target_mode=policy.retimed_target_mode,
            retimed_target_start_steps=retimed_target_start,
            online_retimed_target_start_steps=int(
                hparams.get("rhythm_online_retimed_target_start_steps", retimed_target_start)
                or retimed_target_start
            ),
            use_retimed_pitch_target=bool(
                hparams.get("rhythm_use_retimed_pitch_target", False)
            ),
            disable_pitch_when_retimed=bool(
                hparams.get("rhythm_disable_pitch_loss_when_retimed", False)
            ),
            retimed_mel_source=str(
                hparams.get("rhythm_binarize_retimed_mel_source", "guidance") or "guidance"
            )
            .strip()
            .lower(),
            binarize_teacher_targets=bool(hparams.get("rhythm_binarize_teacher_targets", False)),
            disable_mel_adv_when_retimed=bool(
                hparams.get("rhythm_disable_mel_adv_when_retimed", True)
            ),
            compact_joint_loss=bool(hparams.get("rhythm_compact_joint_loss", True)),
            lambda_distill=float(hparams.get("lambda_rhythm_distill", 0.0)),
            lambda_teacher_aux=float(hparams.get("lambda_rhythm_teacher_aux", 0.0) or 0.0),
            lambda_mel_adv=float(hparams.get("lambda_mel_adv", 0.0)),
            lambda_guidance=float(hparams.get("lambda_rhythm_guidance", 0.0)),
            lambda_plan=float(hparams.get("lambda_rhythm_plan", 0.0) or 0.0),
            lambda_budget=float(hparams.get("lambda_rhythm_budget", 0.0)),
            distill_exec_weight=float(hparams.get("rhythm_distill_exec_weight", 1.0)),
            distill_budget_weight=float(hparams.get("rhythm_distill_budget_weight", 0.5)),
            distill_allocation_weight=float(hparams.get("rhythm_distill_allocation_weight", 0.5)),
            distill_prefix_weight=float(hparams.get("rhythm_distill_prefix_weight", 0.25)),
            distill_speech_shape_weight=float(
                hparams.get("rhythm_distill_speech_shape_weight", 0.0)
            ),
            distill_pause_shape_weight=float(
                hparams.get("rhythm_distill_pause_shape_weight", 0.0)
            ),
            explicit_inactive_distill_weights=tuple(
                float(hparams[key]) for key in _DISTILL_COMPONENT_KEYS if key in hparams
            ),
            dedupe_teacher_primary_cache_distill=resolve_duplicate_primary_distill_dedupe_flag(
                hparams,
                default=True,
            ),
            plan_local_weight=float(hparams.get("rhythm_plan_local_weight", 0.5)),
            plan_cum_weight=float(hparams.get("rhythm_plan_cum_weight", 1.0)),
            budget_raw_weight=float(hparams.get("rhythm_budget_raw_weight", 1.0)),
            budget_exec_weight=float(hparams.get("rhythm_budget_exec_weight", 0.25)),
            pause_topk_ratio=pause_topk_ratio,
            pause_topk_ratio_train_start=float(
                hparams.get("rhythm_projector_pause_topk_ratio_train_start", 1.0)
            ),
            pause_topk_ratio_train_end=float(
                hparams.get("rhythm_projector_pause_topk_ratio_train_end", pause_topk_ratio)
            ),
            pause_topk_ratio_anneal_steps=int(
                hparams.get("rhythm_projector_pause_topk_ratio_anneal_steps", 20000) or 0
            ),
            pause_topk_ratio_warmup_steps=int(
                hparams.get("rhythm_projector_pause_topk_ratio_warmup_steps", 0) or 0
            ),
            distill_conf_floor=float(hparams.get("rhythm_distill_confidence_floor", 0.05)),
            distill_conf_power=float(hparams.get("rhythm_distill_confidence_power", 1.0)),
            source_boundary_scale=source_boundary_scale,
            source_boundary_scale_train_start=float(
                hparams.get("rhythm_source_boundary_scale_train_start", 1.0)
            ),
            source_boundary_scale_train_end=float(
                hparams.get("rhythm_source_boundary_scale_train_end", source_boundary_scale)
            ),
            source_boundary_scale_anneal_steps=int(
                hparams.get("rhythm_source_boundary_scale_anneal_steps", 20000) or 0
            ),
            source_boundary_scale_warmup_steps=int(
                hparams.get("rhythm_source_boundary_scale_warmup_steps", 0) or 0
            ),
            teacher_source_boundary_scale=float(
                hparams.get("rhythm_teacher_source_boundary_scale", source_boundary_scale)
            ),
            pause_boundary_weight=float(resolve_pause_boundary_weight(hparams)),
            pause_boundary_bias_weight=float(
                hparams.get("rhythm_projector_pause_boundary_bias_weight", 0.15)
            ),
            pause_selection_mode=str(
                hparams.get("rhythm_projector_pause_selection_mode", "sparse") or "sparse"
            ).strip().lower(),
            pause_event_weight=float(hparams.get("rhythm_pause_event_weight", 0.0) or 0.0),
            pause_support_weight=float(hparams.get("rhythm_pause_support_weight", 0.0) or 0.0),
            export_cache_audit_to_sample=bool(
                hparams.get("rhythm_export_cache_audit_to_sample", False)
            ),
            public_losses=tuple(hparams.get("rhythm_public_losses", []) or ()),
            configured_cache_version=int(
                hparams.get("rhythm_cache_version", RHYTHM_CACHE_VERSION)
            ),
        ),
    )


__all__ = [
    "RhythmStageKnobs",
    "RhythmStageValidationContext",
    "build_stage_validation_context",
]
