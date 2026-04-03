from __future__ import annotations


FORMAL_RHYTHM_STAGES = {
    "teacher_offline",
    "student_kd",
    "student_retimed",
}

LEGACY_RHYTHM_STAGES = {
    "legacy_schedule_only",
    "legacy_dual_mode_kd",
}

SPECIAL_RHYTHM_STAGES = {
    "minimal_v1",
    "transitional",
}

ALL_RHYTHM_STAGES = FORMAL_RHYTHM_STAGES | LEGACY_RHYTHM_STAGES | SPECIAL_RHYTHM_STAGES

_STAGE_ALIASES = {
    "teacher_offline": "teacher_offline",
    "offline_teacher": "teacher_offline",
    "teacher_only": "teacher_offline",
    "teacher_only_stage": "teacher_offline",
    "student_kd": "student_kd",
    "teacher_student_kd": "student_kd",
    "student_retimed": "student_retimed",
    "retimed_train": "student_retimed",
    "schedule_only": "legacy_schedule_only",
    "legacy_schedule_only": "legacy_schedule_only",
    "dual_mode_kd": "legacy_dual_mode_kd",
    "legacy_dual_mode_kd": "legacy_dual_mode_kd",
    "cached_only": "minimal_v1",
    "minimal_v1": "minimal_v1",
    "transitional": "transitional",
}


def normalize_rhythm_stage(value, *, default: str = "transitional") -> str:
    if value in {None, "", "none", "null"}:
        return default
    stage = str(value).strip().lower()
    normalized = _STAGE_ALIASES.get(stage, stage)
    if normalized not in ALL_RHYTHM_STAGES:
        raise ValueError(f"Unsupported rhythm stage: {value}")
    return normalized


def _normalize_distill_surface(value) -> str:
    surface = str(value or "auto").strip().lower()
    aliases = {
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
    return aliases.get(surface, surface)


def detect_rhythm_stage(hparams, *, config_name: str | None = None) -> str:
    explicit = hparams.get("rhythm_stage", None)
    if explicit not in {None, ""}:
        return normalize_rhythm_stage(explicit)

    config_name = str(config_name or "").strip().lower()
    if "teacher_offline" in config_name or "offline_teacher" in config_name or "teacher_only" in config_name:
        return "teacher_offline"
    if "student_kd" in config_name or "teacher_student_kd" in config_name:
        return "student_kd"
    if "student_retimed" in config_name or "retimed_train" in config_name:
        return "student_retimed"
    if "schedule_only" in config_name:
        return "legacy_schedule_only"
    if "dual_mode_kd" in config_name:
        return "legacy_dual_mode_kd"
    if "cached_only" in config_name:
        return "minimal_v1"
    if "minimal_v1" in config_name:
        return "minimal_v1"

    distill = _normalize_distill_surface(hparams.get("rhythm_distill_surface", "auto"))
    if bool(hparams.get("rhythm_teacher_as_main", False)) or bool(hparams.get("rhythm_teacher_only_stage", False)):
        return "teacher_offline"
    if (
        bool(hparams.get("rhythm_apply_train_override", False))
        or bool(hparams.get("rhythm_apply_valid_override", False))
        or bool(hparams.get("rhythm_require_retimed_cache", False))
    ):
        return "student_retimed"
    if (
        float(hparams.get("lambda_rhythm_distill", 0.0) or 0.0) > 0.0
        and distill == "cache"
        and not bool(hparams.get("rhythm_enable_dual_mode_teacher", False))
    ):
        return "student_kd"
    if bool(hparams.get("rhythm_enable_dual_mode_teacher", False)) or distill in {"offline", "algorithmic"}:
        return "legacy_dual_mode_kd"
    if bool(hparams.get("rhythm_schedule_only_stage", False)):
        return "legacy_schedule_only"
    if bool(hparams.get("rhythm_minimal_v1_profile", False)):
        return "minimal_v1"
    return "transitional"


def is_formal_rhythm_stage(stage: str) -> bool:
    return normalize_rhythm_stage(stage) in FORMAL_RHYTHM_STAGES


def is_student_mainline_stage(stage: str) -> bool:
    return normalize_rhythm_stage(stage) in {"student_kd", "student_retimed"}


def is_legacy_rhythm_stage(stage: str) -> bool:
    return normalize_rhythm_stage(stage) in LEGACY_RHYTHM_STAGES


def resolve_teacher_as_main(hparams, *, stage: str | None = None, infer: bool = False) -> bool:
    if bool(infer):
        return False
    resolved_stage = detect_rhythm_stage(hparams) if stage is None else normalize_rhythm_stage(stage)
    if resolved_stage != "teacher_offline":
        return False
    explicit = hparams.get("rhythm_teacher_as_main", None)
    if explicit is not None:
        return bool(explicit)
    return resolved_stage == "teacher_offline"


def resolve_runtime_offline_teacher_enable(hparams, *, stage: str | None = None) -> bool:
    resolved_stage = detect_rhythm_stage(hparams) if stage is None else normalize_rhythm_stage(stage)
    learned_teacher_requested = bool(hparams.get("rhythm_enable_learned_offline_teacher", False))
    explicit_runtime = hparams.get("rhythm_runtime_enable_learned_offline_teacher", None)
    if resolved_stage == "teacher_offline":
        if explicit_runtime is not None:
            return bool(explicit_runtime) and learned_teacher_requested
        return learned_teacher_requested
    if resolved_stage == "legacy_dual_mode_kd":
        if explicit_runtime is not None:
            return (
                bool(explicit_runtime)
                and learned_teacher_requested
                and bool(hparams.get("rhythm_enable_dual_mode_teacher", False))
            )
        return learned_teacher_requested and bool(hparams.get("rhythm_enable_dual_mode_teacher", False))
    return False


def resolve_runtime_dual_mode_teacher_enable(hparams, *, stage: str | None = None, infer: bool = False) -> bool:
    if bool(infer):
        return False
    resolved_stage = detect_rhythm_stage(hparams) if stage is None else normalize_rhythm_stage(stage)
    return (
        resolved_stage == "legacy_dual_mode_kd"
        and bool(hparams.get("rhythm_enable_dual_mode_teacher", False))
        and resolve_runtime_offline_teacher_enable(hparams, stage=resolved_stage)
    )
