from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class RhythmConfigContractContext:
    hparams: Mapping[str, Any]
    config_path: str | None
    model_dry_run: bool
    profile: str
    stage: str
    policy: Any


@dataclass(frozen=True)
class RhythmConfigContractReport:
    profile: str
    stage: str
    required_field_groups: tuple[tuple[str, ...], ...] = ()
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def merged(self, *others: "RhythmConfigContractReport") -> "RhythmConfigContractReport":
        return merge_contract_reports(self, *others)


@dataclass(frozen=True)
class RhythmContractValidationResult:
    profile: str
    stage: str
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class RhythmConfigContractEvaluation:
    context: RhythmConfigContractContext
    report: RhythmConfigContractReport


def _dedup_groups(groups: Sequence[Sequence[str]]) -> tuple[tuple[str, ...], ...]:
    dedup: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for group in groups:
        normalized = tuple(group)
        if normalized in seen:
            continue
        seen.add(normalized)
        dedup.append(normalized)
    return tuple(dedup)


def _dedup_messages(messages: Sequence[str]) -> tuple[str, ...]:
    dedup: list[str] = []
    seen: set[str] = set()
    for message in messages:
        normalized = str(message)
        if normalized in seen:
            continue
        seen.add(normalized)
        dedup.append(normalized)
    return tuple(dedup)


def merge_contract_reports(*reports: RhythmConfigContractReport) -> RhythmConfigContractReport:
    profile = ""
    stage = ""
    groups: list[tuple[str, ...]] = []
    errors: list[str] = []
    warnings: list[str] = []
    for report in reports:
        if report is None:
            continue
        if not profile and report.profile:
            profile = report.profile
        if not stage and report.stage:
            stage = report.stage
        groups.extend(report.required_field_groups)
        errors.extend(report.errors)
        warnings.extend(report.warnings)
    return RhythmConfigContractReport(
        profile=profile,
        stage=stage,
        required_field_groups=_dedup_groups(groups),
        errors=_dedup_messages(errors),
        warnings=_dedup_messages(warnings),
    )


__all__ = [
    "RhythmContractValidationResult",
    "RhythmConfigContractContext",
    "RhythmConfigContractEvaluation",
    "RhythmConfigContractReport",
    "merge_contract_reports",
]
