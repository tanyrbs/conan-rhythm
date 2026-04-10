from __future__ import annotations


def _coerce_alias_value(value, *, default, cast):
    if value is None:
        value = default
    return cast(value)


def resolve_legacy_alias_hparam(
    hparams,
    *,
    preferred_key: str,
    legacy_key: str,
    default,
    cast,
):
    has_preferred = preferred_key in hparams
    has_legacy = legacy_key in hparams
    if has_preferred and has_legacy:
        preferred_value = _coerce_alias_value(hparams.get(preferred_key), default=default, cast=cast)
        legacy_value = _coerce_alias_value(hparams.get(legacy_key), default=default, cast=cast)
        if preferred_value != legacy_value:
            raise ValueError(
                f"Do not set both {preferred_key} and legacy {legacy_key} to different values."
            )
        return preferred_value
    if has_preferred:
        return _coerce_alias_value(hparams.get(preferred_key), default=default, cast=cast)
    if has_legacy:
        return _coerce_alias_value(hparams.get(legacy_key), default=default, cast=cast)
    return cast(default)


def resolve_progress_bins(hparams, *, default: int = 4) -> int:
    return resolve_legacy_alias_hparam(
        hparams,
        preferred_key="rhythm_progress_bins",
        legacy_key="rhythm_coarse_bins",
        default=default,
        cast=int,
    )


def resolve_progress_support_tau(hparams, *, default: float = 8.0) -> float:
    return resolve_legacy_alias_hparam(
        hparams,
        preferred_key="rhythm_progress_support_tau",
        legacy_key="rhythm_coarse_support_tau",
        default=default,
        cast=float,
    )
