from __future__ import annotations

import torch

from .g_stats import (
    compute_global_rate_1d,
    normalize_global_rate_variant,
    summarize_global_rate_support,
    true_median_1d,
    weighted_median_1d,
    weighted_trimmed_mean_1d,
)


def apply_analytic_gap_clip(
    analytic_gap: torch.Tensor,
    clip_value: float | None,
) -> torch.Tensor:
    if clip_value is None:
        return analytic_gap
    clip = float(clip_value)
    if clip <= 0.0:
        return analytic_gap
    return analytic_gap.clamp(min=-clip, max=clip)


def first_valid_speech_init(observed_log: torch.Tensor, speech_mask: torch.Tensor) -> torch.Tensor:
    batch_size = int(observed_log.size(0))
    out = observed_log.new_zeros((batch_size, 1))
    for batch_idx in range(batch_size):
        keep = torch.nonzero(speech_mask[batch_idx] > 0.5, as_tuple=False).reshape(-1)
        if int(keep.numel()) > 0:
            out[batch_idx, 0] = observed_log[batch_idx, int(keep[0].item())]
    return out


def _resolve_initial_rate_column(
    *,
    observed_log: torch.Tensor,
    init_rate: torch.Tensor | None,
    default_init_rate: torch.Tensor | float | None = None,
) -> torch.Tensor:
    batch_size, _ = observed_log.shape
    if init_rate is None:
        if isinstance(default_init_rate, torch.Tensor):
            prev = default_init_rate.to(device=observed_log.device, dtype=observed_log.dtype).reshape(-1)
            if prev.numel() == 1:
                prev = prev.view(1, 1).expand(batch_size, 1)
            elif prev.numel() == batch_size:
                prev = prev.reshape(batch_size, 1)
            else:
                raise ValueError(
                    f"default_init_rate must be scalar or batch-sized tensor, got shape={tuple(default_init_rate.shape)}"
                )
        elif default_init_rate is None:
            prev = observed_log.new_zeros((batch_size, 1))
        else:
            prev = observed_log.new_full((batch_size, 1), float(default_init_rate))
    else:
        prev = init_rate.float().reshape(batch_size, 1)
    return prev


@torch.jit.script
def _build_causal_local_rate_seq_script(
    observed_log: torch.Tensor,
    speech_mask: torch.Tensor,
    prev: torch.Tensor,
    decay: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = int(observed_log.size(0))
    num_units = int(observed_log.size(1))
    seq = observed_log.new_zeros((batch_size, num_units))
    state = prev
    for unit_idx in range(num_units):
        seq[:, unit_idx : unit_idx + 1] = state
        use_t = speech_mask[:, unit_idx : unit_idx + 1] > 0.5
        cur_t = observed_log[:, unit_idx : unit_idx + 1]
        state = torch.where(use_t, decay * state + (1.0 - decay) * cur_t, state)
    return seq, state


def build_causal_local_rate_seq(
    *,
    observed_log: torch.Tensor,
    speech_mask: torch.Tensor,
    init_rate: torch.Tensor | None,
    default_init_rate: torch.Tensor | float | None = None,
    decay: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    observed_log: [B, U], usually normalized log-duration observations.
    speech_mask:  [B, U], 1 only for speech units that may update the EMA.
    init_rate:    [B, 1] or None, carried runtime state.
    Returns:
        local_rate_seq:  [B, U], rate state BEFORE consuming current unit
        local_rate_last: [B, 1], state AFTER consuming the whole chunk
    """
    batch_size, num_units = observed_log.shape
    prev = _resolve_initial_rate_column(
        observed_log=observed_log,
        init_rate=init_rate,
        default_init_rate=default_init_rate,
    )
    decay = float(max(0.0, min(0.999, decay)))

    if num_units <= 0:
        return observed_log.new_zeros((batch_size, 0)), prev
    return _build_causal_local_rate_seq_script(
        observed_log.float(),
        speech_mask.float(),
        prev.float(),
        float(decay),
    )


def _normalize_prefix_variant(value: str) -> str:
    variant = normalize_global_rate_variant(value)
    if variant == "unit_norm":
        return "raw_median"
    return variant


def normalize_src_prefix_stat_mode(value: str | None) -> str:
    normalized = str(value or "ema").strip().lower()
    aliases = {
        "": "ema",
        "auto": "ema",
        "legacy": "ema",
        "robust": "family_hybrid",
        "hybrid": "family_hybrid",
        "family": "family_hybrid",
        "exact": "exact_global_family",
        "exact_family": "exact_global_family",
        "global_family": "exact_global_family",
    }
    normalized = aliases.get(normalized, normalized)
    valid = {"ema", "family_hybrid", "exact_global_family"}
    if normalized not in valid:
        raise ValueError(
            f"Unsupported src_prefix_stat_mode={value!r}. Expected one of: {sorted(valid)}"
        )
    return normalized


def _compute_prefix_center_1d(
    *,
    values: torch.Tensor,
    weights: torch.Tensor | None,
    variant: str,
    trim_ratio: float,
) -> torch.Tensor:
    resolved_variant = _normalize_prefix_variant(variant)
    row_values = values.float().reshape(-1)
    if row_values.numel() <= 0:
        raise ValueError("_compute_prefix_center_1d requires at least one value.")
    row_weights = None if weights is None else weights.float().reshape(-1).clamp_min(0.0)
    if resolved_variant in {"raw_median", "weighted_median", "softclean_wmed"}:
        if row_weights is None:
            return true_median_1d(row_values)
        return weighted_median_1d(
            row_values,
            row_weights,
            invalid_weight_behavior="fallback",
        )
    if resolved_variant in {"trimmed_mean", "softclean_wtmean"}:
        if row_weights is None:
            row_weights = torch.ones_like(row_values)
        return weighted_trimmed_mean_1d(
            row_values,
            row_weights,
            trim_ratio=trim_ratio,
            invalid_weight_behavior="fallback",
        )
    return true_median_1d(row_values)


def build_causal_prefix_global_rate_seq(
    *,
    observed_log: torch.Tensor,
    speech_mask: torch.Tensor,
    init_rate: torch.Tensor | None,
    default_init_rate: torch.Tensor | float | None = None,
    decay: float = 0.95,
    variant: str = "raw_median",
    trim_ratio: float = 0.2,
    min_support: int = 3,
    weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if observed_log.ndim != 2 or speech_mask.ndim != 2:
        raise ValueError(
            "build_causal_prefix_global_rate_seq expects rank-2 observed_log and speech_mask, "
            f"got {tuple(observed_log.shape)} and {tuple(speech_mask.shape)}"
        )
    if tuple(observed_log.shape) != tuple(speech_mask.shape):
        raise ValueError(
            "build_causal_prefix_global_rate_seq shape mismatch: "
            f"{tuple(observed_log.shape)} vs {tuple(speech_mask.shape)}"
        )
    if weight is not None and tuple(weight.shape) != tuple(observed_log.shape):
        raise ValueError(
            "build_causal_prefix_global_rate_seq weight shape mismatch: "
            f"{tuple(weight.shape)} vs {tuple(observed_log.shape)}"
        )
    batch_size, num_units = observed_log.shape
    state = _resolve_initial_rate_column(
        observed_log=observed_log,
        init_rate=init_rate,
        default_init_rate=default_init_rate,
    )
    decay = float(max(0.0, min(0.999, decay)))
    trim_ratio = float(max(0.0, min(0.49, trim_ratio)))
    min_support = int(max(1, min_support))
    seq = observed_log.new_zeros((batch_size, num_units))
    value_cache: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
    weight_cache: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
    use_weight = isinstance(weight, torch.Tensor)
    for unit_idx in range(num_units):
        seq[:, unit_idx : unit_idx + 1] = state
        for batch_idx in range(batch_size):
            if float(speech_mask[batch_idx, unit_idx].item()) <= 0.5:
                continue
            value_t = observed_log[batch_idx, unit_idx]
            if not bool(torch.isfinite(value_t).item()):
                continue
            value_cache[batch_idx].append(value_t)
            if use_weight:
                weight_t = weight[batch_idx, unit_idx].float().clamp_min(0.0)
                weight_cache[batch_idx].append(weight_t)
            support_count = len(value_cache[batch_idx])
            if support_count >= min_support:
                state[batch_idx, 0] = _compute_prefix_center_1d(
                    values=torch.stack(value_cache[batch_idx]),
                    weights=(
                        torch.stack(weight_cache[batch_idx])
                        if use_weight and len(weight_cache[batch_idx]) == support_count
                        else None
                    ),
                    variant=variant,
                    trim_ratio=trim_ratio,
                )
            else:
                state[batch_idx, 0] = (
                    (decay * state[batch_idx, 0]) + ((1.0 - decay) * value_t.float())
                )
    return seq, state


def build_causal_prefix_global_rate_seq_exact(
    *,
    observed_log: torch.Tensor,
    speech_mask: torch.Tensor,
    init_rate: torch.Tensor | None,
    default_init_rate: torch.Tensor | float | None = None,
    valid_mask: torch.Tensor | None = None,
    closed_mask: torch.Tensor | None = None,
    boundary_confidence: torch.Tensor | None = None,
    min_boundary_confidence: float | None = None,
    variant: str = "raw_median",
    trim_ratio: float = 0.2,
    drop_edge_runs: int = 0,
    min_support: int = 3,
    min_speech_ratio: float = 0.0,
    weight: torch.Tensor | None = None,
    unit_ids: torch.Tensor | None = None,
    unit_prior: torch.Tensor | None = None,
    unit_prior_default_value: float | torch.Tensor | None = None,
    unit_count: torch.Tensor | None = None,
    unit_prior_min_count: int | None = None,
    unit_prior_global_backoff: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if observed_log.ndim != 2 or speech_mask.ndim != 2:
        raise ValueError(
            "build_causal_prefix_global_rate_seq_exact expects rank-2 observed_log and speech_mask, "
            f"got {tuple(observed_log.shape)} and {tuple(speech_mask.shape)}"
        )
    if tuple(observed_log.shape) != tuple(speech_mask.shape):
        raise ValueError(
            "build_causal_prefix_global_rate_seq_exact shape mismatch: "
            f"{tuple(observed_log.shape)} vs {tuple(speech_mask.shape)}"
        )
    if valid_mask is not None and tuple(valid_mask.shape) != tuple(observed_log.shape):
        raise ValueError(
            "build_causal_prefix_global_rate_seq_exact valid_mask shape mismatch: "
            f"{tuple(valid_mask.shape)} vs {tuple(observed_log.shape)}"
        )
    if closed_mask is not None and tuple(closed_mask.shape) != tuple(observed_log.shape):
        raise ValueError(
            "build_causal_prefix_global_rate_seq_exact closed_mask shape mismatch: "
            f"{tuple(closed_mask.shape)} vs {tuple(observed_log.shape)}"
        )
    if boundary_confidence is not None and tuple(boundary_confidence.shape) != tuple(observed_log.shape):
        raise ValueError(
            "build_causal_prefix_global_rate_seq_exact boundary_confidence shape mismatch: "
            f"{tuple(boundary_confidence.shape)} vs {tuple(observed_log.shape)}"
        )
    if weight is not None and tuple(weight.shape) != tuple(observed_log.shape):
        raise ValueError(
            "build_causal_prefix_global_rate_seq_exact weight shape mismatch: "
            f"{tuple(weight.shape)} vs {tuple(observed_log.shape)}"
        )
    batch_size, num_units = observed_log.shape
    state = _resolve_initial_rate_column(
        observed_log=observed_log,
        init_rate=init_rate,
        default_init_rate=default_init_rate,
    )
    seq = observed_log.new_zeros((batch_size, num_units))
    prefix_seen = torch.zeros_like(speech_mask, dtype=torch.bool)
    resolved_variant = normalize_global_rate_variant(variant)
    trim_ratio = float(max(0.0, min(0.49, trim_ratio)))
    min_support = int(max(1, min_support))
    valid_bool = None if valid_mask is None else valid_mask.bool()
    closed_bool = None if closed_mask is None else closed_mask.bool()
    boundary = None if boundary_confidence is None else boundary_confidence.float()
    for unit_idx in range(num_units):
        seq[:, unit_idx : unit_idx + 1] = state
        prefix_seen[:, unit_idx] = True
        for batch_idx in range(batch_size):
            if float(speech_mask[batch_idx, unit_idx].item()) <= 0.5:
                continue
            prefix_valid = prefix_seen[batch_idx]
            if valid_bool is not None:
                prefix_valid = prefix_valid & valid_bool[batch_idx]
            if not bool(prefix_valid.any().item()):
                continue
            prefix_closed = None
            if closed_bool is not None:
                prefix_closed = prefix_seen[batch_idx] & closed_bool[batch_idx]
            support_stats = summarize_global_rate_support(
                speech_mask=speech_mask[batch_idx],
                valid_mask=prefix_valid.float(),
                duration_obs=torch.exp(observed_log[batch_idx].float()),
                drop_edge_runs=drop_edge_runs,
                min_speech_ratio=min_speech_ratio,
                min_speech_runs=min_support,
                closed_mask=(
                    None if prefix_closed is None else prefix_closed.float()
                ),
                boundary_confidence=(
                    None if boundary is None else boundary[batch_idx]
                ),
                min_boundary_confidence=min_boundary_confidence,
            )
            domain_valid = bool(
                support_stats.domain_valid.reshape(-1)[0].detach().item() > 0.5
            )
            if not domain_valid:
                continue
            weight_row = None if weight is None else weight[batch_idx]
            unit_ids_row = None if unit_ids is None else unit_ids[batch_idx]
            unit_prior_row = unit_prior
            if isinstance(unit_prior, torch.Tensor) and unit_prior.ndim == 2 and int(unit_prior.size(0)) == batch_size:
                unit_prior_row = unit_prior[batch_idx]
            unit_count_row = unit_count
            if isinstance(unit_count, torch.Tensor) and unit_count.ndim == 2 and int(unit_count.size(0)) == batch_size:
                unit_count_row = unit_count[batch_idx]
            state[batch_idx, 0] = compute_global_rate_1d(
                log_dur=observed_log[batch_idx],
                speech_mask=speech_mask[batch_idx],
                valid_mask=prefix_valid.float(),
                variant=resolved_variant,
                weight=weight_row,
                trim_ratio=trim_ratio,
                drop_edge_runs=drop_edge_runs,
                closed_mask=(
                    None if prefix_closed is None else prefix_closed.float()
                ),
                boundary_confidence=(
                    None if boundary is None else boundary[batch_idx]
                ),
                min_boundary_confidence=min_boundary_confidence,
                unit_ids=unit_ids_row,
                unit_prior=unit_prior_row,
                unit_prior_default_value=unit_prior_default_value,
                unit_count=unit_count_row,
                unit_prior_min_count=unit_prior_min_count,
                unit_prior_global_backoff=unit_prior_global_backoff,
                support_stats=support_stats,
                invalid_weight_behavior="fallback",
            )
    return seq, state


def build_causal_source_prefix_rate_seq(
    *,
    observed_log: torch.Tensor,
    speech_mask: torch.Tensor,
    init_rate: torch.Tensor | None,
    default_init_rate: torch.Tensor | float | None = None,
    stat_mode: str = "ema",
    decay: float = 0.95,
    variant: str = "raw_median",
    trim_ratio: float = 0.2,
    min_support: int = 3,
    weight: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    closed_mask: torch.Tensor | None = None,
    boundary_confidence: torch.Tensor | None = None,
    min_boundary_confidence: float | None = None,
    drop_edge_runs: int = 0,
    min_speech_ratio: float = 0.0,
    unit_ids: torch.Tensor | None = None,
    unit_prior: torch.Tensor | None = None,
    unit_prior_default_value: float | torch.Tensor | None = None,
    unit_count: torch.Tensor | None = None,
    unit_prior_min_count: int | None = None,
    unit_prior_global_backoff: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    resolved_mode = normalize_src_prefix_stat_mode(stat_mode)
    if resolved_mode == "ema":
        return build_causal_local_rate_seq(
            observed_log=observed_log,
            speech_mask=speech_mask,
            init_rate=init_rate,
            default_init_rate=default_init_rate,
            decay=decay,
        )
    if resolved_mode == "exact_global_family":
        return build_causal_prefix_global_rate_seq_exact(
            observed_log=observed_log,
            speech_mask=speech_mask,
            init_rate=init_rate,
            default_init_rate=default_init_rate,
            valid_mask=valid_mask,
            closed_mask=closed_mask,
            boundary_confidence=boundary_confidence,
            min_boundary_confidence=min_boundary_confidence,
            variant=variant,
            trim_ratio=trim_ratio,
            drop_edge_runs=drop_edge_runs,
            min_support=min_support,
            min_speech_ratio=min_speech_ratio,
            weight=weight,
            unit_ids=unit_ids,
            unit_prior=unit_prior,
            unit_prior_default_value=unit_prior_default_value,
            unit_count=unit_count,
            unit_prior_min_count=unit_prior_min_count,
            unit_prior_global_backoff=unit_prior_global_backoff,
        )
    return build_causal_prefix_global_rate_seq(
        observed_log=observed_log,
        speech_mask=speech_mask,
        init_rate=init_rate,
        default_init_rate=default_init_rate,
        decay=decay,
        variant=variant,
        trim_ratio=trim_ratio,
        min_support=min_support,
        weight=weight,
    )


__all__ = [
    "apply_analytic_gap_clip",
    "build_causal_local_rate_seq",
    "build_causal_prefix_global_rate_seq",
    "build_causal_prefix_global_rate_seq_exact",
    "build_causal_source_prefix_rate_seq",
    "first_valid_speech_init",
    "normalize_src_prefix_stat_mode",
]
