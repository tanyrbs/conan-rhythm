from __future__ import annotations

from modules.Conan.rhythm.policy import expected_cache_contract as build_expected_cache_contract
from modules.Conan.rhythm.surface_metadata import RHYTHM_CACHE_VERSION

from .context import RhythmStageValidationContext


def validate_stage_post_rules(
    ctx: RhythmStageValidationContext,
    warnings: list[str],
) -> None:
    hparams = ctx.hparams
    if ctx.profile != "minimal_v1" and bool(hparams.get("rhythm_minimal_v1_profile", False)):
        warnings.append(
            "rhythm_minimal_v1_profile=true no longer implies the maintained chain; prefer explicit rhythm_stage={teacher_offline,student_kd,student_retimed}."
        )
    contract = build_expected_cache_contract(hparams)
    if int(contract["rhythm_cache_version"]) != int(RHYTHM_CACHE_VERSION):
        warnings.append(
            f"expected_cache_contract resolved rhythm_cache_version={int(contract['rhythm_cache_version'])} "
            f"while maintained cache version is {int(RHYTHM_CACHE_VERSION)}."
        )


__all__ = ["validate_stage_post_rules"]
