from __future__ import annotations

import torch


EPS = 1.0e-6
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
        return true_median_1d(values_sorted)
    if values_sorted.numel() == 1:
        return values_sorted[0]
    cdf = torch.cumsum(weights_sorted, dim=0)
    cutoff = 0.5 * total
    idx = torch.searchsorted(cdf, cutoff, right=False).clamp(max=values_sorted.numel() - 1)
    next_idx = idx + 1
    if bool(torch.isclose(cdf[idx], cutoff)) and int(next_idx.item()) < int(values_sorted.numel()):
        return 0.5 * (values_sorted[idx] + values_sorted[next_idx])
    return values_sorted[idx]


def true_median_1d(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 1:
        raise ValueError("true_median_1d expects a 1D tensor.")
    if values.numel() <= 0:
        raise ValueError("true_median_1d requires at least one value.")
    sorted_values = values.float().sort().values
    count = int(sorted_values.numel())
    mid = count // 2
    if count % 2 == 1:
        return sorted_values[mid]
    return 0.5 * (sorted_values[mid - 1] + sorted_values[mid])


def _normalize_drop_edge_runs(value) -> int:
    return max(0, int(value or 0))


def _drop_edge_support_1d(mask: torch.Tensor, *, drop_edge_runs: int) -> torch.Tensor:
    keep = mask.bool().clone()
    drop = _normalize_drop_edge_runs(drop_edge_runs)
    if drop <= 0:
        return keep
    active = torch.nonzero(keep, as_tuple=False).reshape(-1)
    active_count = int(active.numel())
    if active_count <= (2 * drop):
        return keep
    keep[active[:drop]] = False
    keep[active[-drop:]] = False
    return keep


def _build_single_support_mask(
    *,
    speech_mask: torch.Tensor,
    valid_mask: torch.Tensor | None,
    drop_edge_runs: int,
) -> torch.Tensor:
    speech = speech_mask.bool()
    valid = torch.ones_like(speech, dtype=torch.bool) if valid_mask is None else valid_mask.bool()
    speech_valid = speech & valid
    if not bool(speech_valid.any().item()):
        return torch.zeros_like(speech_valid, dtype=torch.bool)
    support = _drop_edge_support_1d(speech_valid, drop_edge_runs=drop_edge_runs)
    if bool(support.any().item()):
        return support
    return speech_valid


def build_global_rate_support_mask(
    *,
    speech_mask: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    drop_edge_runs: int = 0,
) -> torch.Tensor:
    drop_edge_runs = _normalize_drop_edge_runs(drop_edge_runs)
    if speech_mask.ndim == 1:
        return _build_single_support_mask(
            speech_mask=speech_mask,
            valid_mask=valid_mask,
            drop_edge_runs=drop_edge_runs,
        )
    if speech_mask.ndim != 2:
        raise ValueError(
            "build_global_rate_support_mask expects rank-1 or rank-2 speech_mask, "
            f"got {tuple(speech_mask.shape)}"
        )
    if drop_edge_runs <= 0:
        speech = speech_mask.bool()
        valid = torch.ones_like(speech, dtype=torch.bool) if valid_mask is None else valid_mask.bool()
        return speech & valid
    batch_size = int(speech_mask.size(0))
    support = torch.zeros_like(speech_mask, dtype=torch.bool)
    for batch_idx in range(batch_size):
        support[batch_idx] = _build_single_support_mask(
            speech_mask=speech_mask[batch_idx],
            valid_mask=None if valid_mask is None else valid_mask[batch_idx],
            drop_edge_runs=drop_edge_runs,
        )
    return support


def _resolve_unit_prior(
    *,
    unit_ids: torch.Tensor | None,
    unit_prior: torch.Tensor | None,
    mask: torch.Tensor,
    unit_prior_default_value: float | torch.Tensor | None = None,
) -> torch.Tensor:
    if unit_prior is None:
        raise ValueError("unit_norm g_variant requires unit_prior.")
    if unit_ids is None:
        if tuple(unit_prior.shape) == tuple(mask.shape):
            return unit_prior.to(device=mask.device, dtype=torch.float32) * mask.float()
        raise ValueError("unit_norm g_variant requires unit_ids when unit_prior is a vocabulary vector.")
    if unit_prior.dim() == 1:
        unit_ids_long = unit_ids.long()
        if bool((unit_ids_long < 0).any().item()):
            raise ValueError("unit_ids contain negative values, which are not valid vocabulary ids.")
        prior_vocab = unit_prior.to(device=unit_ids.device, dtype=torch.float32)
        vocab_size = int(prior_vocab.numel())
        oov = unit_ids_long >= vocab_size
        if bool(oov.any().item()):
            if unit_prior_default_value is None:
                raise ValueError(
                    "unit_ids contain values outside the unit_prior vocabulary and no "
                    "unit_prior_default_value was provided. "
                    f"vocab_size={vocab_size}"
                )
            if torch.is_tensor(unit_prior_default_value):
                default_value = unit_prior_default_value.to(
                    device=unit_ids.device,
                    dtype=torch.float32,
                ).reshape(())
            else:
                default_value = prior_vocab.new_tensor(float(unit_prior_default_value))
            prior = torch.full_like(unit_ids_long, float(default_value.item()), dtype=torch.float32)
            in_vocab = ~oov
            if bool(in_vocab.any().item()):
                prior[in_vocab] = prior_vocab[unit_ids_long[in_vocab]]
        else:
            prior = prior_vocab[unit_ids_long]
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
    unit_prior_default_value: float | torch.Tensor | None,
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
            unit_prior_default_value=unit_prior_default_value,
        )
        prior_values = prior[valid]
        values = values - prior_values.float()
        if weights is not None:
            return weighted_median_1d(values, weights)
        return true_median_1d(values)
    if variant == "raw_median":
        return true_median_1d(values)
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


def compute_global_rate_1d(
    *,
    log_dur: torch.Tensor,
    speech_mask: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    variant: str = "raw_median",
    weight: torch.Tensor | None = None,
    trim_ratio: float = 0.2,
    drop_edge_runs: int = 0,
    unit_ids: torch.Tensor | None = None,
    unit_prior: torch.Tensor | None = None,
    unit_prior_default_value: float | torch.Tensor | None = None,
) -> torch.Tensor:
    support_mask = build_global_rate_support_mask(
        speech_mask=speech_mask,
        valid_mask=valid_mask,
        drop_edge_runs=drop_edge_runs,
    )
    return _compute_single_global_rate(
        log_dur=log_dur.float(),
        mask=support_mask,
        variant=normalize_global_rate_variant(variant),
        weight=weight,
        trim_ratio=trim_ratio,
        unit_ids=unit_ids,
        unit_prior=unit_prior,
        unit_prior_default_value=unit_prior_default_value,
    )


def compute_global_rate_batch(
    *,
    log_dur: torch.Tensor,
    speech_mask: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    variant: str = "raw_median",
    weight: torch.Tensor | None = None,
    trim_ratio: float = 0.2,
    drop_edge_runs: int = 0,
    unit_ids: torch.Tensor | None = None,
    unit_prior: torch.Tensor | None = None,
    unit_prior_default_value: float | torch.Tensor | None = None,
) -> torch.Tensor:
    variant = normalize_global_rate_variant(variant)
    log_dur = log_dur.float()
    if log_dur.ndim == 1:
        return compute_global_rate_1d(
            log_dur=log_dur,
            speech_mask=speech_mask,
            valid_mask=valid_mask,
            variant=variant,
            weight=weight,
            trim_ratio=trim_ratio,
            drop_edge_runs=drop_edge_runs,
            unit_ids=unit_ids,
            unit_prior=unit_prior,
            unit_prior_default_value=unit_prior_default_value,
        )
    if log_dur.ndim != 2:
        raise ValueError(f"compute_global_rate expects rank-1 or rank-2 log_dur, got {tuple(log_dur.shape)}")
    support = build_global_rate_support_mask(
        speech_mask=speech_mask,
        valid_mask=valid_mask,
        drop_edge_runs=drop_edge_runs,
    )
    resolved_unit_prior = None
    if variant == "unit_norm":
        resolved_unit_prior = _resolve_unit_prior(
            unit_ids=unit_ids,
            unit_prior=unit_prior,
            mask=support,
            unit_prior_default_value=unit_prior_default_value,
        )
    if variant == "raw_median" and weight is None and unit_prior is None:
        support_count = support.sum(dim=1, keepdim=True)
        if bool((support_count <= 0).any().item()):
            raise ValueError("No valid speech duration for global rate.")
        invalid_fill = torch.finfo(log_dur.dtype).max
        sorted_values = torch.sort(log_dur.masked_fill(~support, invalid_fill), dim=1).values
        counts = support_count.squeeze(1).long()
        lower_idx = ((counts - 1) // 2).clamp_min(0)
        upper_idx = (counts // 2).clamp_min(0)
        row_idx = torch.arange(log_dur.size(0), device=log_dur.device)
        median = 0.5 * (sorted_values[row_idx, lower_idx] + sorted_values[row_idx, upper_idx])
        return median.unsqueeze(1)
    if variant == "trimmed_mean" and _normalize_drop_edge_runs(drop_edge_runs) <= 0:
        support_count = support.sum(dim=1, keepdim=True)
        if bool((support_count <= 0).any().item()):
            raise ValueError("No valid speech duration for global rate.")
        invalid_fill = torch.finfo(log_dur.dtype).max
        sorted_values = torch.sort(log_dur.masked_fill(~support, invalid_fill), dim=1).values
        trim_ratio = float(max(0.0, min(0.49, trim_ratio)))
        trim = torch.floor(support_count.float() * trim_ratio).long()
        keep_start = torch.where((2 * trim) < support_count, trim, torch.zeros_like(trim))
        keep_end = torch.where((2 * trim) < support_count, support_count - trim, support_count)
        positions = torch.arange(log_dur.size(1), device=log_dur.device).unsqueeze(0)
        valid_sorted = positions < support_count
        keep_mask = (positions >= keep_start) & (positions < keep_end) & valid_sorted
        keep_weight = keep_mask.float()
        mean = (sorted_values * keep_weight).sum(dim=1, keepdim=True) / keep_weight.sum(dim=1, keepdim=True).clamp_min(1.0)
        return mean
    batch_size = int(log_dur.size(0))
    out = log_dur.new_zeros((batch_size, 1))
    for batch_idx in range(batch_size):
        unit_prior_row = unit_prior
        if isinstance(unit_prior, torch.Tensor) and unit_prior.ndim == 2 and int(unit_prior.size(0)) == batch_size:
            unit_prior_row = unit_prior[batch_idx]
        out[batch_idx, 0] = compute_global_rate_1d(
            log_dur=log_dur[batch_idx],
            speech_mask=support[batch_idx],
            valid_mask=None,
            variant=variant,
            weight=None if weight is None else weight[batch_idx],
            trim_ratio=trim_ratio,
            drop_edge_runs=0,
            unit_ids=None if resolved_unit_prior is not None or unit_ids is None else unit_ids[batch_idx],
            unit_prior=(
                resolved_unit_prior[batch_idx]
                if resolved_unit_prior is not None
                else unit_prior_row
            ),
            unit_prior_default_value=unit_prior_default_value,
        )
    return out


def compute_global_rate(
    *,
    log_dur: torch.Tensor,
    speech_mask: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    variant: str = "raw_median",
    weight: torch.Tensor | None = None,
    trim_ratio: float = 0.2,
    drop_edge_runs: int = 0,
    unit_ids: torch.Tensor | None = None,
    unit_prior: torch.Tensor | None = None,
    unit_prior_default_value: float | torch.Tensor | None = None,
) -> torch.Tensor:
    return compute_global_rate_batch(
        log_dur=log_dur,
        speech_mask=speech_mask,
        valid_mask=valid_mask,
        variant=variant,
        weight=weight,
        trim_ratio=trim_ratio,
        drop_edge_runs=drop_edge_runs,
        unit_ids=unit_ids,
        unit_prior=unit_prior,
        unit_prior_default_value=unit_prior_default_value,
    )


__all__ = [
    "EPS",
    "build_global_rate_support_mask",
    "compute_global_rate",
    "compute_global_rate_1d",
    "compute_global_rate_batch",
    "normalize_falsification_eval_mode",
    "normalize_global_rate_variant",
    "true_median_1d",
    "weighted_median_1d",
]
