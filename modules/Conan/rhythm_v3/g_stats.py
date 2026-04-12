from __future__ import annotations

import torch


_VALID_G_VARIANTS = {"raw_median", "weighted_median", "trimmed_mean", "unit_norm"}
_VALID_EVAL_MODES = {"analytic", "coarse_only", "learned"}


def normalize_global_rate_variant(value) -> str:
    normalized = str(value or "raw_median").strip().lower()
    aliases = {
        "median": "raw_median",
        "raw": "raw_median",
        "wmed": "weighted_median",
        "weighted": "weighted_median",
        "tmean": "trimmed_mean",
        "trimmed": "trimmed_mean",
        "unit_normalized": "unit_norm",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in _VALID_G_VARIANTS:
        raise ValueError(
            f"Unsupported rhythm_v3 g_variant={value!r}. "
            f"Expected one of: {sorted(_VALID_G_VARIANTS)}"
        )
    return normalized


def normalize_falsification_eval_mode(value) -> str:
    normalized = str(value or "learned").strip().lower()
    aliases = {
        "full": "learned",
        "default": "learned",
        "coarse": "coarse_only",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in _VALID_EVAL_MODES:
        raise ValueError(
            f"Unsupported rhythm_v3 eval_mode={value!r}. "
            f"Expected one of: {sorted(_VALID_EVAL_MODES)}"
        )
    return normalized


def weighted_median_1d(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if values.ndim != 1 or weights.ndim != 1:
        raise ValueError("weighted_median_1d expects 1D tensors.")
    if values.numel() <= 0:
        raise ValueError("weighted_median_1d requires at least one value.")
    order = torch.argsort(values)
    values_sorted = values[order]
    weights_sorted = weights[order].float().clamp_min(0.0)
    total = weights_sorted.sum()
    if not torch.isfinite(total) or float(total.item()) <= 0.0:
        return values_sorted[values_sorted.numel() // 2]
    cdf = torch.cumsum(weights_sorted, dim=0)
    cutoff = 0.5 * total
    idx = torch.searchsorted(cdf, cutoff, right=False)
    idx = idx.clamp(max=values_sorted.numel() - 1)
    return values_sorted[idx]


def _resolve_unit_prior(
    *,
    unit_ids: torch.Tensor | None,
    unit_prior: torch.Tensor | None,
    mask: torch.Tensor,
) -> torch.Tensor:
    if unit_ids is None or unit_prior is None:
        raise ValueError("unit_norm g_variant requires both unit_ids and unit_prior.")
    if unit_prior.dim() == 1:
        prior = unit_prior.to(device=unit_ids.device, dtype=torch.float32)[unit_ids.long()]
    elif unit_prior.dim() == 2 and tuple(unit_prior.shape) == tuple(unit_ids.shape):
        prior = unit_prior.to(device=unit_ids.device, dtype=torch.float32)
    else:
        raise ValueError(
            "unit_prior must have shape [V] or match unit_ids shape [B, T]. "
            f"Got unit_prior={tuple(unit_prior.shape)}, unit_ids={tuple(unit_ids.shape)}."
        )
    return prior * mask.float()


def _compute_single_global_rate(
    *,
    log_dur: torch.Tensor,
    mask: torch.Tensor,
    variant: str,
    weight: torch.Tensor | None,
    trim_ratio: float,
    unit_ids: torch.Tensor | None,
    unit_prior: torch.Tensor | None,
) -> torch.Tensor:
    valid = mask.bool()
    if not bool(valid.any().item()):
        raise ValueError("No valid speech duration for global rate.")
    values = log_dur[valid].float()
    weights = None if weight is None else weight[valid].float().clamp_min(1.0e-6)
    if variant == "unit_norm":
        prior = _resolve_unit_prior(
            unit_ids=unit_ids,
            unit_prior=unit_prior,
            mask=mask,
        )
        prior_values = prior[valid]
        values = values - prior_values.float()
        if weights is not None:
            return weighted_median_1d(values, weights)
        return values.median()
    if variant == "raw_median":
        return values.median()
    if variant == "weighted_median":
        if weights is None:
            weights = torch.ones_like(values)
        return weighted_median_1d(values, weights)
    if variant == "trimmed_mean":
        sorted_values = values.sort().values
        trim = int(float(max(0.0, min(0.49, trim_ratio))) * int(sorted_values.numel()))
        if trim > 0 and (2 * trim) < int(sorted_values.numel()):
            sorted_values = sorted_values[trim:-trim]
        return sorted_values.mean()
    raise ValueError(f"Unsupported g_variant={variant!r}")


def compute_global_rate(
    *,
    log_dur: torch.Tensor,
    speech_mask: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    variant: str = "raw_median",
    weight: torch.Tensor | None = None,
    trim_ratio: float = 0.2,
    unit_ids: torch.Tensor | None = None,
    unit_prior: torch.Tensor | None = None,
) -> torch.Tensor:
    variant = normalize_global_rate_variant(variant)
    mask = speech_mask.bool()
    if valid_mask is not None:
        mask = mask & valid_mask.bool()
    log_dur = log_dur.float()
    if log_dur.ndim == 1:
        return _compute_single_global_rate(
            log_dur=log_dur,
            mask=mask,
            variant=variant,
            weight=weight,
            trim_ratio=trim_ratio,
            unit_ids=unit_ids,
            unit_prior=unit_prior,
        )
    if log_dur.ndim != 2:
        raise ValueError(f"compute_global_rate expects rank-1 or rank-2 log_dur, got {tuple(log_dur.shape)}")
    batch_size = int(log_dur.size(0))
    out = log_dur.new_zeros((batch_size, 1))
    for batch_idx in range(batch_size):
        unit_prior_row = unit_prior
        if isinstance(unit_prior, torch.Tensor) and unit_prior.ndim == 2 and int(unit_prior.size(0)) == batch_size:
            unit_prior_row = unit_prior[batch_idx]
        out[batch_idx, 0] = _compute_single_global_rate(
            log_dur=log_dur[batch_idx],
            mask=mask[batch_idx],
            variant=variant,
            weight=None if weight is None else weight[batch_idx],
            trim_ratio=trim_ratio,
            unit_ids=None if unit_ids is None else unit_ids[batch_idx],
            unit_prior=unit_prior_row,
        )
    return out


__all__ = [
    "compute_global_rate",
    "normalize_falsification_eval_mode",
    "normalize_global_rate_variant",
    "weighted_median_1d",
]
