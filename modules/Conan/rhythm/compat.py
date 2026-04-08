from __future__ import annotations


def resolve_bool_alias(
    *,
    default: bool | None,
    canonical_value,
    legacy_value,
    canonical_name: str,
    legacy_name: str,
    where: str,
) -> bool | None:
    if (
        canonical_value is not None
        and legacy_value is not None
        and bool(canonical_value) != bool(legacy_value)
    ):
        raise ValueError(
            f"{where}: conflicting values for {canonical_name} and deprecated {legacy_name}."
        )
    if canonical_value is not None:
        return bool(canonical_value)
    if legacy_value is not None:
        return bool(legacy_value)
    if default is None:
        return None
    return bool(default)


def resolve_float_alias(
    *,
    default: float | None,
    canonical_value,
    legacy_value,
    canonical_name: str,
    legacy_name: str,
    where: str,
) -> float | None:
    if (
        canonical_value is not None
        and legacy_value is not None
        and float(canonical_value) != float(legacy_value)
    ):
        raise ValueError(
            f"{where}: conflicting values for {canonical_name} and deprecated {legacy_name}."
        )
    if canonical_value is not None:
        return float(canonical_value)
    if legacy_value is not None:
        return float(legacy_value)
    if default is None:
        return None
    return float(default)


def resolve_phase_decoupled_flag(
    *,
    default: bool | None,
    phase_decoupled_timing,
    phase_free_timing,
    where: str,
) -> bool | None:
    return resolve_bool_alias(
        default=default,
        canonical_value=phase_decoupled_timing,
        legacy_value=phase_free_timing,
        canonical_name="phase_decoupled_timing",
        legacy_name="phase_free_timing",
        where=where,
    )


def resolve_phase_decoupled_threshold(
    *,
    default: float | None,
    phase_decoupled_phrase_gate_boundary_threshold,
    phase_free_phrase_boundary_threshold,
    where: str,
) -> float | None:
    return resolve_float_alias(
        default=default,
        canonical_value=phase_decoupled_phrase_gate_boundary_threshold,
        legacy_value=phase_free_phrase_boundary_threshold,
        canonical_name="phase_decoupled_phrase_gate_boundary_threshold",
        legacy_name="phase_free_phrase_boundary_threshold",
        where=where,
    )
