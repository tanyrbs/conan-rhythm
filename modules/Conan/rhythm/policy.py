from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .stages import (
    detect_rhythm_stage,
    resolve_runtime_dual_mode_teacher_enable as resolve_runtime_dual_mode_teacher_enable_from_stage,
    resolve_runtime_offline_teacher_enable as resolve_runtime_offline_teacher_enable_from_stage,
    resolve_teacher_as_main as resolve_teacher_as_main_from_stage,
)
from .surface_metadata import (
    RHYTHM_CACHE_VERSION,
    RHYTHM_REFERENCE_MODE_STATIC_REF_FULL,
    RHYTHM_TRACE_HOP_MS,
    RHYTHM_UNIT_HOP_MS,
    normalize_teacher_target_source,
    resolve_teacher_surface_name,
    resolve_teacher_target_source_id,
)

_TARGET_MODE_ALIASES = {
    "auto": "prefer_cache",
    "offline": "cached_only",
    "offline_only": "cached_only",
    "never": "cached_only",
    "runtime": "runtime_only",
    "always": "runtime_only",
}

_PRIMARY_TARGET_SURFACE_ALIASES = {
    "cache_teacher": "teacher",
    "offline": "teacher",
    "offline_teacher": "teacher",
    "teacher_surface": "teacher",
    "guidance_surface": "guidance",
    "self": "guidance",
}

_DISTILL_SURFACE_ALIASES = {
    "off": "none",
    "disable": "none",
    "disabled": "none",
    "false": "none",
    "cache_teacher": "cache",
    "cached_teacher": "cache",
    "full_context": "offline",
    "shared_offline": "offline",
    "algo": "algorithmic",
    "teacher": "cache",
}

_RETIMED_TARGET_MODE_ALIASES = {
    "cache": "cached",
    "cached_only": "cached",
    "teacher": "cached",
    "runtime": "online",
    "online_only": "online",
    "mixed": "hybrid",
}


def parse_optional_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"", "none", "null", "auto", "default"}:
        return None
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Unsupported optional bool value: {value}")


def use_strict_mainline(hparams: Mapping[str, Any]) -> bool:
    explicit = hparams.get("rhythm_strict_mainline", None)
    if explicit is not None:
        return bool(explicit)
    return bool(hparams.get("rhythm_minimal_v1_profile", False))


def normalize_rhythm_target_mode(value) -> str:
    mode = str(value or "prefer_cache").strip().lower()
    resolved = _TARGET_MODE_ALIASES.get(mode, mode)
    if resolved not in {"prefer_cache", "cached_only", "runtime_only"}:
        raise ValueError(f"Unsupported rhythm_dataset_target_mode: {value}")
    return resolved


def normalize_primary_target_surface(value) -> str:
    surface = str(value or "guidance").strip().lower()
    resolved = _PRIMARY_TARGET_SURFACE_ALIASES.get(surface, surface)
    if resolved not in {"guidance", "teacher"}:
        raise ValueError(f"Unsupported rhythm_primary_target_surface: {value}")
    return resolved


def normalize_distill_surface(value) -> str:
    surface = str(value or "auto").strip().lower()
    resolved = _DISTILL_SURFACE_ALIASES.get(surface, surface)
    if resolved not in {"auto", "none", "cache", "offline", "algorithmic"}:
        raise ValueError(f"Unsupported rhythm_distill_surface: {value}")
    return resolved


def normalize_retimed_target_mode(value) -> str:
    mode = str(value or "cached").strip().lower()
    resolved = _RETIMED_TARGET_MODE_ALIASES.get(mode, mode)
    if resolved not in {"cached", "online", "hybrid"}:
        raise ValueError(f"Unsupported rhythm_retimed_target_mode: {value}")
    return resolved


def resolve_pause_boundary_weight(hparams: Mapping[str, Any]) -> float:
    if "rhythm_pause_boundary_weight" in hparams:
        return float(hparams.get("rhythm_pause_boundary_weight", 0.35))
    if "rhythm_pause_exec_boundary_boost" in hparams:
        return float(hparams.get("rhythm_pause_exec_boundary_boost", 0.75))
    return 0.35


def resolve_prefix_state_lambda(hparams: Mapping[str, Any]) -> float:
    if "lambda_rhythm_cumplan" in hparams:
        return float(hparams.get("lambda_rhythm_cumplan", 0.15))
    return float(hparams.get("lambda_rhythm_carry", 0.15))


def resolve_cumplan_lambda(hparams: Mapping[str, Any]) -> float:
    return resolve_prefix_state_lambda(hparams)


def resolve_runtime_offline_teacher_enable(
    hparams: Mapping[str, Any],
    *,
    stage: str | None = None,
    config_path: str | None = None,
) -> bool:
    resolved_stage = stage
    if resolved_stage is None:
        resolved_stage = detect_rhythm_stage(hparams, config_name=config_path)
    return resolve_runtime_offline_teacher_enable_from_stage(hparams, stage=resolved_stage)


def resolve_runtime_dual_mode_teacher_enable(
    hparams: Mapping[str, Any],
    *,
    stage: str | None = None,
    config_path: str | None = None,
    infer: bool = False,
) -> bool:
    resolved_stage = stage
    if resolved_stage is None:
        resolved_stage = detect_rhythm_stage(hparams, config_name=config_path)
    return resolve_runtime_dual_mode_teacher_enable_from_stage(
        hparams,
        stage=resolved_stage,
        infer=infer,
    )


def resolve_teacher_as_main(
    hparams: Mapping[str, Any],
    *,
    stage: str | None = None,
    config_path: str | None = None,
    infer: bool = False,
) -> bool:
    resolved_stage = stage
    if resolved_stage is None:
        resolved_stage = detect_rhythm_stage(hparams, config_name=config_path)
    return resolve_teacher_as_main_from_stage(
        hparams,
        stage=resolved_stage,
        infer=infer,
    )


def should_optimize_render_params(hparams: Mapping[str, Any]) -> bool:
    if bool(hparams.get("rhythm_apply_train_override", False)):
        return True
    apply_mode = str(hparams.get("rhythm_apply_mode", "infer") or "infer").strip().lower()
    return apply_mode in {"always", "all", "train", "training"}


def resolve_apply_override(
    hparams: Mapping[str, Any],
    *,
    infer: bool,
    test: bool,
    current_step: int,
    explicit=None,
):
    explicit_value = parse_optional_bool(explicit)
    if explicit_value is not None:
        enabled = explicit_value
    else:
        split = "test" if test else ("valid" if infer else "train")
        enabled = parse_optional_bool(hparams.get(f"rhythm_apply_{split}_override", None))
    if enabled is None:
        return None
    effective_step = int(current_step)
    split = "test" if test else ("valid" if infer else "train")
    start_step = int(hparams.get(f"rhythm_{split}_render_start_steps", 0) or 0)
    if enabled and split in {"train", "valid"} and effective_step < start_step:
        return False
    retimed_target_start = int(hparams.get("rhythm_retimed_target_start_steps", 0) or 0)
    if (
        enabled
        and split in {"train", "valid"}
        and bool(hparams.get("rhythm_use_retimed_target_if_available", False))
        and effective_step < retimed_target_start
    ):
        return False
    return enabled


def expected_cache_contract(hparams: Mapping[str, Any]) -> dict[str, int | float]:
    return {
        "rhythm_cache_version": int(hparams.get("rhythm_cache_version", RHYTHM_CACHE_VERSION)),
        "rhythm_unit_hop_ms": int(hparams.get("rhythm_unit_hop_ms", RHYTHM_UNIT_HOP_MS)),
        "rhythm_trace_hop_ms": int(hparams.get("rhythm_trace_hop_ms", RHYTHM_TRACE_HOP_MS)),
        "rhythm_trace_bins": int(hparams.get("rhythm_trace_bins", 24)),
        "rhythm_trace_horizon": float(hparams.get("rhythm_trace_horizon", 0.35)),
        "rhythm_slow_topk": int(hparams.get("rhythm_slow_topk", 6)),
        "rhythm_selector_cell_size": int(hparams.get("rhythm_selector_cell_size", 3)),
        "rhythm_source_phrase_threshold": float(hparams.get("rhythm_source_phrase_threshold", 0.55)),
        "rhythm_reference_mode_id": int(
            hparams.get("rhythm_reference_mode_id", RHYTHM_REFERENCE_MODE_STATIC_REF_FULL)
        ),
    }


def exports_streaming_offline_sidecars(
    hparams: Mapping[str, Any],
    *,
    stage: str | None = None,
    config_path: str | None = None,
) -> bool:
    resolved_stage = stage
    if resolved_stage is None:
        resolved_stage = detect_rhythm_stage(hparams, config_name=config_path)
    return resolved_stage == "legacy_dual_mode_kd"


@dataclass(frozen=True)
class RhythmHparamsPolicy:
    hparams: Mapping[str, Any] = field(repr=False, compare=False)
    config_path: str | None = None
    stage: str = ""
    strict_mainline: bool = False
    target_mode: str = "prefer_cache"
    primary_target_surface: str = "guidance"
    distill_surface: str = "auto"
    retimed_target_mode: str = "cached"
    teacher_target_source: str = "algorithmic"
    teacher_surface_name: str = ""
    teacher_target_source_id: int = 0
    runtime_offline_teacher_enabled: bool = False
    runtime_dual_mode_teacher_enabled: bool = False
    teacher_as_main: bool = False
    require_cached_teacher: bool = False
    require_retimed_cache: bool = False
    use_retimed_target_if_available: bool = False
    lambda_teacher_aux: float = 0.0
    optimize_render_params: bool = False

    @classmethod
    def from_hparams(
        cls,
        hparams: Mapping[str, Any],
        *,
        config_path: str | None = None,
    ) -> "RhythmHparamsPolicy":
        stage = detect_rhythm_stage(hparams, config_name=config_path)
        teacher_target_source = normalize_teacher_target_source(
            hparams.get("rhythm_teacher_target_source", "algorithmic")
        )
        return cls(
            hparams=hparams,
            config_path=config_path,
            stage=stage,
            strict_mainline=use_strict_mainline(hparams),
            target_mode=normalize_rhythm_target_mode(
                hparams.get("rhythm_dataset_target_mode", "prefer_cache")
            ),
            primary_target_surface=normalize_primary_target_surface(
                hparams.get("rhythm_primary_target_surface", "guidance")
            ),
            distill_surface=normalize_distill_surface(
                hparams.get("rhythm_distill_surface", "auto")
            ),
            retimed_target_mode=normalize_retimed_target_mode(
                hparams.get("rhythm_retimed_target_mode", "cached")
            ),
            teacher_target_source=teacher_target_source,
            teacher_surface_name=resolve_teacher_surface_name(teacher_target_source),
            teacher_target_source_id=resolve_teacher_target_source_id(teacher_target_source),
            runtime_offline_teacher_enabled=resolve_runtime_offline_teacher_enable(
                hparams,
                stage=stage,
                config_path=config_path,
            ),
            runtime_dual_mode_teacher_enabled=resolve_runtime_dual_mode_teacher_enable(
                hparams,
                stage=stage,
                config_path=config_path,
                infer=False,
            ),
            teacher_as_main=resolve_teacher_as_main(
                hparams,
                stage=stage,
                config_path=config_path,
                infer=False,
            ),
            require_cached_teacher=bool(hparams.get("rhythm_require_cached_teacher", False)),
            require_retimed_cache=bool(hparams.get("rhythm_require_retimed_cache", False)),
            use_retimed_target_if_available=bool(
                hparams.get("rhythm_use_retimed_target_if_available", False)
            ),
            lambda_teacher_aux=float(hparams.get("lambda_rhythm_teacher_aux", 0.0) or 0.0),
            optimize_render_params=should_optimize_render_params(hparams),
        )

    def exports_streaming_offline_sidecars(self) -> bool:
        return exports_streaming_offline_sidecars(
            self.hparams,
            stage=self.stage,
            config_path=self.config_path,
        )

    def should_export_offline_teacher_aux(self) -> bool:
        return self.exports_streaming_offline_sidecars() and self.lambda_teacher_aux > 0.0

    def should_export_runtime_retimed_targets(self, *, split: str) -> bool:
        split = str(split or "").strip().lower()
        if self.require_retimed_cache:
            return True
        if not self.use_retimed_target_if_available:
            return False
        if split == "train":
            return bool(self.hparams.get("rhythm_apply_train_override", False))
        if split in {"valid", "dev"}:
            return bool(self.hparams.get("rhythm_apply_valid_override", False))
        return False

    def resolve_apply_override(
        self,
        *,
        infer: bool,
        test: bool,
        current_step: int,
        explicit=None,
    ):
        return resolve_apply_override(
            self.hparams,
            infer=infer,
            test=test,
            current_step=current_step,
            explicit=explicit,
        )


def build_rhythm_hparams_policy(
    hparams: Mapping[str, Any],
    *,
    config_path: str | None = None,
) -> RhythmHparamsPolicy:
    return RhythmHparamsPolicy.from_hparams(hparams, config_path=config_path)
