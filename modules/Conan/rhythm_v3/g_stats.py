from __future__ import annotations

from dataclasses import dataclass

import torch


EPS = 1.0e-6
_VALID_G_VARIANTS = {"raw_median", "weighted_median", "trimmed_mean", "unit_norm"}
_VALID_EVAL_MODES = {"analytic", "coarse_only", "learned"}
_VALID_INVALID_WEIGHT_BEHAVIORS = {"raise", "nan", "fallback"}


@dataclass(frozen=True)
class GlobalRateSupportStats:
    support_mask: torch.Tensor
    support_count: torch.Tensor
    support_seed_count: torch.Tensor
    speech_count: torch.Tensor
    valid_count: torch.Tensor
    speech_ratio: torch.Tensor
    support_fraction: torch.Tensor
    edge_runs_dropped: torch.Tensor
    domain_valid: torch.Tensor
    clean_mask: torch.Tensor | None = None
    clean_count: torch.Tensor | None = None


def compute_duration_weighted_speech_ratio(
    *,
    duration_obs: torch.Tensor | None,
    speech_mask: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if speech_mask.ndim not in {1, 2}:
        raise ValueError(
            "compute_duration_weighted_speech_ratio expects rank-1 or rank-2 speech_mask, "
            f"got {tuple(speech_mask.shape)}"
        )
    speech = speech_mask.float().clamp(0.0, 1.0)
    valid = (
        torch.ones_like(speech, dtype=speech.dtype)
        if valid_mask is None
        else valid_mask.float().clamp(0.0, 1.0)
    )
    reduce_dim = 0 if speech_mask.ndim == 1 else 1
    speech = speech * valid
    count_ratio = speech.sum(dim=reduce_dim, keepdim=True) / valid.sum(dim=reduce_dim, keepdim=True).clamp_min(1.0)
    if not isinstance(duration_obs, torch.Tensor):
        return count_ratio
    if tuple(duration_obs.shape) != tuple(speech_mask.shape):
        raise ValueError(
            "compute_duration_weighted_speech_ratio duration_obs/speech_mask shape mismatch: "
            f"{tuple(duration_obs.shape)} vs {tuple(speech_mask.shape)}"
        )
    duration = duration_obs.float().clamp_min(0.0) * valid
    duration_mass = duration.sum(dim=reduce_dim, keepdim=True)
    duration_ratio = (duration * speech).sum(dim=reduce_dim, keepdim=True) / duration_mass.clamp_min(EPS)
    return torch.where(duration_mass > EPS, duration_ratio, count_ratio)


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


def normalize_invalid_weight_behavior(value) -> str:
    normalized = str(value or "raise").strip().lower()
    aliases = {
        "strict": "raise",
        "error": "raise",
        "none": "raise",
        "flag": "nan",
        "quiet_nan": "nan",
        "legacy": "fallback",
        "raw_median": "fallback",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in _VALID_INVALID_WEIGHT_BEHAVIORS:
        raise ValueError(
            f"Unsupported invalid_weight_behavior={value!r}. "
            f"Expected one of: {sorted(_VALID_INVALID_WEIGHT_BEHAVIORS)}"
        )
    return normalized


def weighted_median_1d(
    values: torch.Tensor,
    weights: torch.Tensor,
    *,
    invalid_weight_behavior: str = "raise",
) -> torch.Tensor:
    if values.ndim != 1 or weights.ndim != 1:
        raise ValueError("weighted_median_1d expects 1D tensors.")
    if values.numel() <= 0:
        raise ValueError("weighted_median_1d requires at least one value.")
    invalid_weight_behavior = normalize_invalid_weight_behavior(invalid_weight_behavior)
    order = torch.argsort(values)
    values_sorted = values[order]
    weights_sorted = weights[order].float().clamp_min(0.0)
    total = weights_sorted.sum()
    if not torch.isfinite(total) or float(total.item()) <= 0.0:
        if invalid_weight_behavior == "raise":
            raise ValueError("weighted_median_1d requires positive finite total weight.")
        if invalid_weight_behavior == "nan":
            return values_sorted.new_full((), float("nan"))
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


def _build_boundary_clean_seed(
    *,
    speech_valid: torch.Tensor,
    closed_mask: torch.Tensor | None,
    boundary_confidence: torch.Tensor | None,
    min_boundary_confidence: float | None,
) -> torch.Tensor:
    clean = speech_valid.bool().clone()
    if closed_mask is not None:
        clean &= closed_mask.bool()
    if boundary_confidence is not None and min_boundary_confidence is not None:
        clean &= boundary_confidence.float() >= float(min_boundary_confidence)
    return clean


def _build_single_support_mask(
    *,
    speech_mask: torch.Tensor,
    valid_mask: torch.Tensor | None,
    drop_edge_runs: int,
    closed_mask: torch.Tensor | None = None,
    boundary_confidence: torch.Tensor | None = None,
    min_boundary_confidence: float | None = None,
) -> torch.Tensor:
    _, support = _build_single_support_surface(
        speech_mask=speech_mask,
        valid_mask=valid_mask,
        drop_edge_runs=drop_edge_runs,
        closed_mask=closed_mask,
        boundary_confidence=boundary_confidence,
        min_boundary_confidence=min_boundary_confidence,
    )
    return support


def _build_single_support_surface(
    *,
    speech_mask: torch.Tensor,
    valid_mask: torch.Tensor | None,
    drop_edge_runs: int,
    closed_mask: torch.Tensor | None = None,
    boundary_confidence: torch.Tensor | None = None,
    min_boundary_confidence: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    speech = speech_mask.bool()
    valid = torch.ones_like(speech, dtype=torch.bool) if valid_mask is None else valid_mask.bool()
    speech_valid = speech & valid
    if not bool(speech_valid.any().item()):
        empty = torch.zeros_like(speech_valid, dtype=torch.bool)
        return empty, empty
    clean_seed = _build_boundary_clean_seed(
        speech_valid=speech_valid,
        closed_mask=closed_mask,
        boundary_confidence=boundary_confidence,
        min_boundary_confidence=min_boundary_confidence,
    )
    support_seed = clean_seed if bool(clean_seed.any().item()) else speech_valid
    support = _drop_edge_support_1d(support_seed, drop_edge_runs=drop_edge_runs)
    if bool(support.any().item()):
        return support_seed, support
    return support_seed, support_seed


def _resolve_support_mask_input(
    *,
    log_dur: torch.Tensor,
    support_mask: torch.Tensor | None,
    support_stats: GlobalRateSupportStats | None,
) -> torch.Tensor | None:
    if support_mask is not None and support_stats is not None:
        raise ValueError("Pass at most one of support_mask or support_stats.")
    resolved = support_stats.support_mask if support_stats is not None else support_mask
    if resolved is None:
        return None
    resolved = resolved.bool().to(device=log_dur.device)
    expected_shape = tuple(log_dur.shape)
    if tuple(resolved.shape) != expected_shape:
        raise ValueError(
            f"support_mask shape mismatch: expected {expected_shape}, got {tuple(resolved.shape)}"
        )
    return resolved


def _resolve_or_build_support_mask(
    *,
    log_dur: torch.Tensor,
    speech_mask: torch.Tensor,
    valid_mask: torch.Tensor | None,
    drop_edge_runs: int,
    closed_mask: torch.Tensor | None,
    boundary_confidence: torch.Tensor | None,
    min_boundary_confidence: float | None,
    support_mask: torch.Tensor | None,
    support_stats: GlobalRateSupportStats | None,
) -> torch.Tensor:
    resolved = _resolve_support_mask_input(
        log_dur=log_dur,
        support_mask=support_mask,
        support_stats=support_stats,
    )
    if resolved is not None:
        return resolved
    return build_global_rate_support_mask(
        speech_mask=speech_mask,
        valid_mask=valid_mask,
        drop_edge_runs=drop_edge_runs,
        closed_mask=closed_mask,
        boundary_confidence=boundary_confidence,
        min_boundary_confidence=min_boundary_confidence,
    )


def build_global_rate_support_mask(
    *,
    speech_mask: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    drop_edge_runs: int = 0,
    closed_mask: torch.Tensor | None = None,
    boundary_confidence: torch.Tensor | None = None,
    min_boundary_confidence: float | None = None,
) -> torch.Tensor:
    drop_edge_runs = _normalize_drop_edge_runs(drop_edge_runs)
    if speech_mask.ndim == 1:
        return _build_single_support_mask(
            speech_mask=speech_mask,
            valid_mask=valid_mask,
            drop_edge_runs=drop_edge_runs,
            closed_mask=closed_mask,
            boundary_confidence=boundary_confidence,
            min_boundary_confidence=min_boundary_confidence,
        )
    if speech_mask.ndim != 2:
        raise ValueError(
            "build_global_rate_support_mask expects rank-1 or rank-2 speech_mask, "
            f"got {tuple(speech_mask.shape)}"
        )
    if (
        drop_edge_runs <= 0
        and closed_mask is None
        and boundary_confidence is None
        and min_boundary_confidence is None
    ):
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
            closed_mask=None if closed_mask is None else closed_mask[batch_idx],
            boundary_confidence=(
                None if boundary_confidence is None else boundary_confidence[batch_idx]
            ),
            min_boundary_confidence=min_boundary_confidence,
        )
    return support


def summarize_global_rate_support(
    *,
    speech_mask: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    drop_edge_runs: int = 0,
    min_speech_ratio: float = 0.0,
    min_speech_runs: int = 1,
    closed_mask: torch.Tensor | None = None,
    boundary_confidence: torch.Tensor | None = None,
    min_boundary_confidence: float | None = None,
) -> GlobalRateSupportStats:
    speech = speech_mask.bool()
    valid = torch.ones_like(speech, dtype=torch.bool) if valid_mask is None else valid_mask.bool()
    clean_mask = _build_boundary_clean_seed(
        speech_valid=speech & valid,
        closed_mask=closed_mask,
        boundary_confidence=boundary_confidence,
        min_boundary_confidence=min_boundary_confidence,
    )
    if speech_mask.ndim == 1:
        support_seed_mask, support_mask = _build_single_support_surface(
            speech_mask=speech_mask,
            valid_mask=valid_mask,
            drop_edge_runs=drop_edge_runs,
            closed_mask=closed_mask,
            boundary_confidence=boundary_confidence,
            min_boundary_confidence=min_boundary_confidence,
        )
    elif speech_mask.ndim == 2:
        support_seed_mask = torch.zeros_like(speech_mask, dtype=torch.bool)
        support_mask = torch.zeros_like(speech_mask, dtype=torch.bool)
        batch_size = int(speech_mask.size(0))
        for batch_idx in range(batch_size):
            seed_row, support_row = _build_single_support_surface(
                speech_mask=speech_mask[batch_idx],
                valid_mask=None if valid_mask is None else valid_mask[batch_idx],
                drop_edge_runs=drop_edge_runs,
                closed_mask=None if closed_mask is None else closed_mask[batch_idx],
                boundary_confidence=(
                    None if boundary_confidence is None else boundary_confidence[batch_idx]
                ),
                min_boundary_confidence=min_boundary_confidence,
            )
            support_seed_mask[batch_idx] = seed_row
            support_mask[batch_idx] = support_row
    else:
        raise ValueError(
            "summarize_global_rate_support expects rank-1 or rank-2 speech_mask, "
            f"got {tuple(speech_mask.shape)}"
        )
    reduce_dim = 0 if speech_mask.ndim == 1 else 1
    support_count = support_mask.sum(dim=reduce_dim, keepdim=True).float()
    support_seed_count = support_seed_mask.sum(dim=reduce_dim, keepdim=True).float()
    speech_count = (speech & valid).sum(dim=reduce_dim, keepdim=True).float()
    valid_count = valid.sum(dim=reduce_dim, keepdim=True).float()
    clean_count = clean_mask.sum(dim=reduce_dim, keepdim=True).float()
    speech_ratio = speech_count / valid_count.clamp_min(1.0)
    support_fraction = support_count / speech_count.clamp_min(1.0)
    edge_runs_dropped = (support_seed_count - support_count).clamp_min(0.0)
    domain_valid = (
        (support_count >= float(max(1, int(min_speech_runs))))
        & (speech_ratio >= float(max(0.0, min(1.0, min_speech_ratio))) - 1.0e-6)
    ).float()
    return GlobalRateSupportStats(
        support_mask=support_mask,
        support_count=support_count,
        support_seed_count=support_seed_count,
        speech_count=speech_count,
        valid_count=valid_count,
        speech_ratio=speech_ratio,
        support_fraction=support_fraction,
        edge_runs_dropped=edge_runs_dropped,
        domain_valid=domain_valid,
        clean_mask=clean_mask,
        clean_count=clean_count,
    )


def _resolve_unit_count(
    *,
    unit_ids: torch.Tensor | None,
    unit_count: torch.Tensor | None,
    mask: torch.Tensor,
) -> torch.Tensor | None:
    if unit_count is None:
        return None
    if unit_ids is None:
        if tuple(unit_count.shape) == tuple(mask.shape):
            return unit_count.to(device=mask.device, dtype=torch.float32) * mask.float()
        return None
    if unit_count.dim() == 1:
        unit_ids_long = unit_ids.long()
        counts_vocab = unit_count.to(device=unit_ids.device, dtype=torch.float32).reshape(-1)
        vocab_size = int(counts_vocab.numel())
        counts = torch.zeros_like(unit_ids_long, dtype=torch.float32)
        in_vocab = (unit_ids_long >= 0) & (unit_ids_long < vocab_size)
        if bool(in_vocab.any().item()):
            counts[in_vocab] = counts_vocab[unit_ids_long[in_vocab]]
        return counts * mask.float()
    if unit_count.dim() == 2 and tuple(unit_count.shape) == tuple(unit_ids.shape):
        return unit_count.to(device=unit_ids.device, dtype=torch.float32) * mask.float()
    return None


def _resolve_unit_prior_backoff_value(
    *,
    unit_prior_default_value: float | torch.Tensor | None,
    unit_prior_global_backoff: float | torch.Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    value = unit_prior_global_backoff
    if value is None:
        value = unit_prior_default_value
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.to(device=device, dtype=torch.float32).reshape(())
    return torch.tensor(float(value), device=device, dtype=dtype).reshape(())


def _resolve_unit_prior(
    *,
    unit_ids: torch.Tensor | None,
    unit_prior: torch.Tensor | None,
    mask: torch.Tensor,
    unit_prior_default_value: float | torch.Tensor | None = None,
    unit_count: torch.Tensor | None = None,
    unit_prior_min_count: int | None = None,
    unit_prior_global_backoff: float | torch.Tensor | None = None,
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
    resolved_count = _resolve_unit_count(
        unit_ids=unit_ids,
        unit_count=unit_count,
        mask=mask,
    )
    if resolved_count is not None and unit_prior_min_count is not None and int(unit_prior_min_count) > 0:
        backoff = _resolve_unit_prior_backoff_value(
            unit_prior_default_value=unit_prior_default_value,
            unit_prior_global_backoff=unit_prior_global_backoff,
            device=prior.device,
            dtype=prior.dtype,
        )
        if backoff is not None:
            alpha = (resolved_count.float() / float(max(1, int(unit_prior_min_count)))).clamp_(0.0, 1.0)
            prior = (alpha * prior.float()) + ((1.0 - alpha) * backoff.float())
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
    unit_count: torch.Tensor | None,
    unit_prior_min_count: int | None,
    unit_prior_global_backoff: float | torch.Tensor | None,
    invalid_weight_behavior: str,
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
            unit_count=unit_count,
            unit_prior_min_count=unit_prior_min_count,
            unit_prior_global_backoff=unit_prior_global_backoff,
        )
        prior_values = prior[valid]
        values = values - prior_values.float()
        if weights is not None:
            return weighted_median_1d(
                values,
                weights,
                invalid_weight_behavior=invalid_weight_behavior,
            )
        return true_median_1d(values)
    if variant == "raw_median":
        return true_median_1d(values)
    if variant == "weighted_median":
        if weights is None:
            weights = torch.ones_like(values)
        return weighted_median_1d(
            values,
            weights,
            invalid_weight_behavior=invalid_weight_behavior,
        )
    if variant == "trimmed_mean":
        sorted_values = values.sort().values
        trim = int(float(max(0.0, min(0.49, trim_ratio))) * int(sorted_values.numel()))
        if trim > 0 and (2 * trim) < int(sorted_values.numel()):
            sorted_values = sorted_values[trim:-trim]
        return sorted_values.mean()
    raise ValueError(f"Unsupported g_variant={variant!r}")


def masked_true_median_batch(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = values.float()
    mask = mask.bool()
    if values.ndim == 1:
        values = values.unsqueeze(0)
        mask = mask.unsqueeze(0)
    if values.ndim != 2 or mask.ndim != 2:
        raise ValueError("masked_true_median_batch expects rank-1 or rank-2 values and mask.")
    if tuple(values.shape) != tuple(mask.shape):
        raise ValueError(
            f"masked_true_median_batch shape mismatch: values={tuple(values.shape)} mask={tuple(mask.shape)}"
        )
    support_count = mask.sum(dim=1, keepdim=True)
    if bool((support_count <= 0).any().item()):
        raise ValueError("masked_true_median_batch requires at least one valid item per batch row.")
    invalid_fill = torch.finfo(values.dtype).max
    sorted_values = torch.sort(values.masked_fill(~mask, invalid_fill), dim=1).values
    counts = support_count.squeeze(1).long()
    lower_idx = ((counts - 1) // 2).clamp_min(0)
    upper_idx = (counts // 2).clamp_min(0)
    row_idx = torch.arange(values.size(0), device=values.device)
    median = 0.5 * (sorted_values[row_idx, lower_idx] + sorted_values[row_idx, upper_idx])
    return median.unsqueeze(1)


def masked_weighted_median_batch(
    values: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
    *,
    invalid_weight_behavior: str = "raise",
) -> torch.Tensor:
    values = values.float()
    mask = mask.bool()
    weights = weights.float().clamp_min(0.0)
    invalid_weight_behavior = normalize_invalid_weight_behavior(invalid_weight_behavior)
    if values.ndim == 1:
        values = values.unsqueeze(0)
        mask = mask.unsqueeze(0)
        weights = weights.unsqueeze(0)
    if values.ndim != 2 or mask.ndim != 2 or weights.ndim != 2:
        raise ValueError("masked_weighted_median_batch expects rank-1 or rank-2 values/mask/weights.")
    if tuple(values.shape) != tuple(mask.shape) or tuple(values.shape) != tuple(weights.shape):
        raise ValueError(
            "masked_weighted_median_batch shape mismatch: "
            f"values={tuple(values.shape)} mask={tuple(mask.shape)} weights={tuple(weights.shape)}"
        )
    support_count = mask.sum(dim=1, keepdim=True)
    if bool((support_count <= 0).any().item()):
        raise ValueError("masked_weighted_median_batch requires at least one valid item per batch row.")
    safe_weights = torch.where(mask, weights, torch.zeros_like(weights))
    total = safe_weights.sum(dim=1, keepdim=True)
    no_mass = total <= 0.0
    invalid_total = no_mass | (~torch.isfinite(total))
    if bool(invalid_total.any().item()) and invalid_weight_behavior == "raise":
        raise ValueError("masked_weighted_median_batch requires positive finite total weight per batch row.")
    order = torch.argsort(values.masked_fill(~mask, torch.finfo(values.dtype).max), dim=1)
    sorted_values = torch.gather(values, 1, order)
    sorted_mask = torch.gather(mask.long(), 1, order).bool()
    sorted_weights = torch.gather(safe_weights, 1, order)
    sorted_weights = torch.where(sorted_mask, sorted_weights, torch.zeros_like(sorted_weights))
    cdf = torch.cumsum(sorted_weights, dim=1)
    cutoff = 0.5 * total
    idx = torch.searchsorted(cdf.contiguous(), cutoff.contiguous(), right=False)
    idx = idx.clamp(max=values.size(1) - 1)
    row_idx = torch.arange(values.size(0), device=values.device).unsqueeze(1)
    median = sorted_values[row_idx, idx].reshape(values.size(0), 1)
    next_idx = (idx + 1).clamp(max=values.size(1) - 1)
    at_cutoff = torch.isclose(
        cdf[row_idx, idx].reshape(values.size(0), 1),
        cutoff,
        atol=1.0e-6,
        rtol=1.0e-4,
    ) & (next_idx > idx)
    median = torch.where(
        at_cutoff,
        0.5 * (
            sorted_values[row_idx, idx].reshape(values.size(0), 1)
            + sorted_values[row_idx, next_idx].reshape(values.size(0), 1)
        ),
        median,
    )
    if bool(invalid_total.any().item()):
        if invalid_weight_behavior == "nan":
            nan_row = torch.full_like(median, float("nan"))
            median = torch.where(invalid_total, nan_row, median)
        else:
            raw = masked_true_median_batch(values, mask)
            median = torch.where(invalid_total, raw, median)
    return median


def compute_global_rate_1d(
    *,
    log_dur: torch.Tensor,
    speech_mask: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    variant: str = "raw_median",
    weight: torch.Tensor | None = None,
    trim_ratio: float = 0.2,
    drop_edge_runs: int = 0,
    closed_mask: torch.Tensor | None = None,
    boundary_confidence: torch.Tensor | None = None,
    min_boundary_confidence: float | None = None,
    unit_ids: torch.Tensor | None = None,
    unit_prior: torch.Tensor | None = None,
    unit_prior_default_value: float | torch.Tensor | None = None,
    unit_count: torch.Tensor | None = None,
    unit_prior_min_count: int | None = None,
    unit_prior_global_backoff: float | torch.Tensor | None = None,
    support_mask: torch.Tensor | None = None,
    support_stats: GlobalRateSupportStats | None = None,
    invalid_weight_behavior: str = "raise",
) -> torch.Tensor:
    support_mask = _resolve_or_build_support_mask(
        log_dur=log_dur,
        speech_mask=speech_mask,
        valid_mask=valid_mask,
        drop_edge_runs=drop_edge_runs,
        closed_mask=closed_mask,
        boundary_confidence=boundary_confidence,
        min_boundary_confidence=min_boundary_confidence,
        support_mask=support_mask,
        support_stats=support_stats,
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
        unit_count=unit_count,
        unit_prior_min_count=unit_prior_min_count,
        unit_prior_global_backoff=unit_prior_global_backoff,
        invalid_weight_behavior=invalid_weight_behavior,
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
    closed_mask: torch.Tensor | None = None,
    boundary_confidence: torch.Tensor | None = None,
    min_boundary_confidence: float | None = None,
    unit_ids: torch.Tensor | None = None,
    unit_prior: torch.Tensor | None = None,
    unit_prior_default_value: float | torch.Tensor | None = None,
    unit_count: torch.Tensor | None = None,
    unit_prior_min_count: int | None = None,
    unit_prior_global_backoff: float | torch.Tensor | None = None,
    support_mask: torch.Tensor | None = None,
    support_stats: GlobalRateSupportStats | None = None,
    invalid_weight_behavior: str = "raise",
) -> torch.Tensor:
    variant = normalize_global_rate_variant(variant)
    invalid_weight_behavior = normalize_invalid_weight_behavior(invalid_weight_behavior)
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
            closed_mask=closed_mask,
            boundary_confidence=boundary_confidence,
            min_boundary_confidence=min_boundary_confidence,
            unit_ids=unit_ids,
            unit_prior=unit_prior,
            unit_prior_default_value=unit_prior_default_value,
            unit_count=unit_count,
            unit_prior_min_count=unit_prior_min_count,
            unit_prior_global_backoff=unit_prior_global_backoff,
            support_mask=support_mask,
            support_stats=support_stats,
            invalid_weight_behavior=invalid_weight_behavior,
        )
    if log_dur.ndim != 2:
        raise ValueError(f"compute_global_rate expects rank-1 or rank-2 log_dur, got {tuple(log_dur.shape)}")
    support = _resolve_or_build_support_mask(
        log_dur=log_dur,
        speech_mask=speech_mask,
        valid_mask=valid_mask,
        drop_edge_runs=drop_edge_runs,
        closed_mask=closed_mask,
        boundary_confidence=boundary_confidence,
        min_boundary_confidence=min_boundary_confidence,
        support_mask=support_mask,
        support_stats=support_stats,
    )
    resolved_unit_prior = None
    if variant == "unit_norm":
        resolved_unit_prior = _resolve_unit_prior(
            unit_ids=unit_ids,
            unit_prior=unit_prior,
            mask=support,
            unit_prior_default_value=unit_prior_default_value,
            unit_count=unit_count,
            unit_prior_min_count=unit_prior_min_count,
            unit_prior_global_backoff=unit_prior_global_backoff,
        )
    if variant == "raw_median" and weight is None and unit_prior is None:
        return masked_true_median_batch(log_dur, support)
    if variant == "weighted_median":
        if weight is None:
            return masked_true_median_batch(log_dur, support)
        resolved_weight = weight.float()
        return masked_weighted_median_batch(
            log_dur,
            support,
            resolved_weight,
            invalid_weight_behavior=invalid_weight_behavior,
        )
    if variant == "unit_norm":
        normalized = log_dur - resolved_unit_prior.float()
        if weight is None:
            return masked_true_median_batch(normalized, support)
        resolved_weight = weight.float()
        return masked_weighted_median_batch(
            normalized,
            support,
            resolved_weight,
            invalid_weight_behavior=invalid_weight_behavior,
        )
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
            speech_mask=speech_mask[batch_idx],
            valid_mask=None if valid_mask is None else valid_mask[batch_idx],
            variant=variant,
            weight=None if weight is None else weight[batch_idx],
            trim_ratio=trim_ratio,
            drop_edge_runs=drop_edge_runs,
            closed_mask=None if closed_mask is None else closed_mask[batch_idx],
            boundary_confidence=(
                None if boundary_confidence is None else boundary_confidence[batch_idx]
            ),
            min_boundary_confidence=min_boundary_confidence,
            unit_ids=None if resolved_unit_prior is not None or unit_ids is None else unit_ids[batch_idx],
            unit_prior=(
                resolved_unit_prior[batch_idx]
                if resolved_unit_prior is not None
                else unit_prior_row
            ),
            unit_prior_default_value=unit_prior_default_value,
            unit_count=None if unit_count is None else unit_count[batch_idx] if unit_count.ndim == 2 and int(unit_count.size(0)) == batch_size else unit_count,
            unit_prior_min_count=unit_prior_min_count,
            unit_prior_global_backoff=unit_prior_global_backoff,
            support_mask=support[batch_idx],
            invalid_weight_behavior=invalid_weight_behavior,
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
    closed_mask: torch.Tensor | None = None,
    boundary_confidence: torch.Tensor | None = None,
    min_boundary_confidence: float | None = None,
    unit_ids: torch.Tensor | None = None,
    unit_prior: torch.Tensor | None = None,
    unit_prior_default_value: float | torch.Tensor | None = None,
    unit_count: torch.Tensor | None = None,
    unit_prior_min_count: int | None = None,
    unit_prior_global_backoff: float | torch.Tensor | None = None,
    support_mask: torch.Tensor | None = None,
    support_stats: GlobalRateSupportStats | None = None,
    invalid_weight_behavior: str = "raise",
) -> torch.Tensor:
    return compute_global_rate_batch(
        log_dur=log_dur,
        speech_mask=speech_mask,
        valid_mask=valid_mask,
        variant=variant,
        weight=weight,
        trim_ratio=trim_ratio,
        drop_edge_runs=drop_edge_runs,
        closed_mask=closed_mask,
        boundary_confidence=boundary_confidence,
        min_boundary_confidence=min_boundary_confidence,
        unit_ids=unit_ids,
        unit_prior=unit_prior,
        unit_prior_default_value=unit_prior_default_value,
        unit_count=unit_count,
        unit_prior_min_count=unit_prior_min_count,
        unit_prior_global_backoff=unit_prior_global_backoff,
        support_mask=support_mask,
        support_stats=support_stats,
        invalid_weight_behavior=invalid_weight_behavior,
    )


__all__ = [
    "EPS",
    "GlobalRateSupportStats",
    "build_global_rate_support_mask",
    "compute_global_rate",
    "compute_global_rate_1d",
    "compute_global_rate_batch",
    "masked_true_median_batch",
    "masked_weighted_median_batch",
    "normalize_falsification_eval_mode",
    "normalize_global_rate_variant",
    "summarize_global_rate_support",
    "true_median_1d",
    "weighted_median_1d",
]
