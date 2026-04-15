from __future__ import annotations

import math
from typing import Any

import torch

from tasks.Conan.rhythm.common.metrics_impl import (
    build_rhythm_metric_dict,
    build_rhythm_metric_sections,
)


def _to_tensor(value: Any, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=dtype)
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    return torch.as_tensor(value, dtype=dtype)


def _flatten_valid_pair(x: Any, y: Any) -> tuple[torch.Tensor, torch.Tensor]:
    tx = _to_tensor(x).reshape(-1)
    ty = _to_tensor(y).reshape(-1)
    valid = torch.isfinite(tx) & torch.isfinite(ty)
    return tx[valid], ty[valid]


def _rank_1d(values: torch.Tensor) -> torch.Tensor:
    if values.numel() <= 0:
        return values
    order = torch.argsort(values, stable=True)
    sorted_vals = values[order]
    sorted_ranks = torch.empty_like(sorted_vals, dtype=torch.float32)
    start = 0
    total = int(sorted_vals.numel())
    while start < total:
        end = start + 1
        while end < total and torch.isclose(sorted_vals[end], sorted_vals[start], atol=1.0e-8, rtol=0.0):
            end += 1
        avg_rank = 0.5 * float(start + end - 1)
        sorted_ranks[start:end] = avg_rank
        start = end
    ranks = torch.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    return ranks


def _pearson_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.numel() <= 1 or y.numel() <= 1:
        return x.new_tensor(float("nan"))
    x_center = x - x.mean()
    y_center = y - y.mean()
    denom = torch.sqrt((x_center.square().sum() * y_center.square().sum()).clamp_min(1.0e-12))
    return (x_center * y_center).sum() / denom


def _theil_sen_slope(x: torch.Tensor, y: torch.Tensor, *, max_points: int = 256) -> torch.Tensor:
    if x.numel() <= 1 or y.numel() <= 1:
        return x.new_tensor(float("nan"))
    if x.numel() > max_points:
        step = max(1, int(math.ceil(float(x.numel()) / float(max_points))))
        x = x[::step]
        y = y[::step]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    valid = torch.triu(dx.abs() > 1.0e-8, diagonal=1)
    if not bool(valid.any().item()):
        return x.new_tensor(float("nan"))
    slopes = (dy[valid] / dx[valid]).float()
    return slopes.median()


def tempo_explainability(delta_g: Any, coarse_target: Any) -> dict[str, float]:
    x, y = _flatten_valid_pair(delta_g, coarse_target)
    if x.numel() <= 1:
        return {
            "spearman": float("nan"),
            "robust_slope": float("nan"),
            "r2_like": float("nan"),
            "count": float(x.numel()),
        }
    rank_x = _rank_1d(x)
    rank_y = _rank_1d(y)
    spearman = _pearson_corr(rank_x, rank_y)
    slope = _theil_sen_slope(x, y)
    intercept = torch.median(y - (slope * x))
    pred = (slope * x) + intercept
    denom = (y - y.mean()).square().sum().clamp_min(1.0e-12)
    r2_like = 1.0 - ((y - pred).square().sum() / denom)
    return {
        "spearman": float(spearman.item()),
        "robust_slope": float(slope.item()),
        "r2_like": float(r2_like.item()),
        "count": float(x.numel()),
    }


def tempo_monotonicity(
    tempo_slow: Any,
    tempo_mid: Any,
    tempo_fast: Any,
    *,
    margin: float = 0.0,
    increasing: bool = True,
) -> torch.Tensor:
    """Monotonicity over slow/mid/fast triplets.

    `increasing=True` is for tempo-like metrics where larger means faster.
    `increasing=False` is for duration-like metrics where larger means slower.
    """
    slow = _to_tensor(tempo_slow).reshape(-1)
    mid = _to_tensor(tempo_mid).reshape(-1)
    fast = _to_tensor(tempo_fast).reshape(-1)
    valid = torch.isfinite(slow) & torch.isfinite(mid) & torch.isfinite(fast)
    if not bool(valid.any().item()):
        return slow.new_tensor(float("nan"))
    if bool(increasing):
        monotonic = ((mid[valid] - slow[valid]) > float(margin)) & ((fast[valid] - mid[valid]) > float(margin))
    else:
        monotonic = ((slow[valid] - mid[valid]) > float(margin)) & ((mid[valid] - fast[valid]) > float(margin))
    return monotonic.float().mean()


def tempo_tie_rate(
    tempo_slow: Any,
    tempo_mid: Any,
    tempo_fast: Any,
    *,
    atol: float = 1.0e-4,
) -> torch.Tensor:
    slow = _to_tensor(tempo_slow).reshape(-1)
    mid = _to_tensor(tempo_mid).reshape(-1)
    fast = _to_tensor(tempo_fast).reshape(-1)
    valid = torch.isfinite(slow) & torch.isfinite(mid) & torch.isfinite(fast)
    if not bool(valid.any().item()):
        return slow.new_tensor(float("nan"))
    tie = torch.isclose(slow, mid, atol=float(atol), rtol=0.0) | torch.isclose(
        mid,
        fast,
        atol=float(atol),
        rtol=0.0,
    )
    return tie[valid].float().mean()


def transfer_slope(delta_g: Any, tempo_delta: Any) -> dict[str, float]:
    """Association summary for tempo transfer; not a causal coefficient."""
    return tempo_explainability(delta_g, tempo_delta)


def monotonic_triplet_table(
    sample_ids: Any,
    tempo_slow: Any,
    tempo_mid: Any,
    tempo_fast: Any,
    *,
    increasing: bool = True,
    margin: float = 0.0,
    tie_atol: float = 1.0e-4,
) -> dict[str, list[float | str]]:
    """Per-sample monotonicity table.

    `increasing=True` is for tempo-like metrics where larger means faster.
    `increasing=False` is for duration-like metrics where larger means slower.
    """
    slow = _to_tensor(tempo_slow).reshape(-1)
    mid = _to_tensor(tempo_mid).reshape(-1)
    fast = _to_tensor(tempo_fast).reshape(-1)
    count = min(int(slow.numel()), int(mid.numel()), int(fast.numel()))
    slow = slow[:count]
    mid = mid[:count]
    fast = fast[:count]
    valid = torch.isfinite(slow) & torch.isfinite(mid) & torch.isfinite(fast)
    tie_sm = torch.isclose(slow, mid, atol=float(tie_atol), rtol=0.0) & valid
    tie_mf = torch.isclose(mid, fast, atol=float(tie_atol), rtol=0.0) & valid
    tie_any = tie_sm | tie_mf
    if bool(increasing):
        strict = ((slow < mid) & (mid < fast)) & valid
        margin_ok = (((mid - slow) > float(margin)) & ((fast - mid) > float(margin))) & valid
    else:
        strict = ((slow > mid) & (mid > fast)) & valid
        margin_ok = (((slow - mid) > float(margin)) & ((mid - fast) > float(margin))) & valid
    values = sample_ids[:count] if isinstance(sample_ids, (list, tuple)) else None
    if values is None:
        values = [str(index) for index in range(count)]
    return {
        "sample_id": [str(value) for value in values],
        "tempo_slow": [float(value) for value in slow.tolist()],
        "tempo_mid": [float(value) for value in mid.tolist()],
        "tempo_fast": [float(value) for value in fast.tolist()],
        "valid": [float(value) for value in valid.float().tolist()],
        "mono_ok": [float(value) for value in strict.float().tolist()],
        "mono_ok_strict": [float(value) for value in strict.float().tolist()],
        "mono_ok_margin": [float(value) for value in margin_ok.float().tolist()],
        "tie_sm": [float(value) for value in tie_sm.float().tolist()],
        "tie_mf": [float(value) for value in tie_mf.float().tolist()],
        "tie_any": [float(value) for value in tie_any.float().tolist()],
    }


def silence_leakage(delta_z: Any, speech_mask: Any, silence_mask: Any) -> torch.Tensor:
    dz = _to_tensor(delta_z).float()
    speech = _to_tensor(speech_mask).float()
    silence = _to_tensor(silence_mask).float()
    numerator = (dz.abs() * silence).sum()
    denominator = (dz.abs() * speech).sum()
    if not torch.isfinite(denominator) or float(denominator.item()) <= 1.0e-3:
        return torch.tensor(float("nan"), device=dz.device)
    return numerator / denominator


def prefix_discrepancy(z_short: Any, z_long: Any, committed_mask: Any) -> torch.Tensor:
    short = _to_tensor(z_short).float()
    long = _to_tensor(z_long).float()
    committed = _to_tensor(committed_mask).float()
    diff = (short - long).abs() * committed
    return diff.sum() / committed.sum().clamp_min(1.0)


def budget_hit_rate(
    budget_hit_pos: Any | None,
    budget_hit_neg: Any | None = None,
) -> torch.Tensor:
    if budget_hit_pos is None and budget_hit_neg is None:
        return torch.tensor(float("nan"))
    hit_pos = torch.zeros((), dtype=torch.float32) if budget_hit_pos is None else _to_tensor(budget_hit_pos).float()
    hit_neg = torch.zeros_like(hit_pos) if budget_hit_neg is None else _to_tensor(budget_hit_neg).float()
    hits = torch.maximum(hit_pos.reshape(-1), hit_neg.reshape(-1))
    return hits.mean() if hits.numel() > 0 else torch.tensor(float("nan"))


def same_text_gap(
    same_text_value: Any,
    cross_text_value: Any | None = None,
    same_text_mask: Any | None = None,
) -> torch.Tensor:
    if same_text_mask is not None:
        values = _to_tensor(same_text_value).float().reshape(-1)
        mask = _to_tensor(same_text_mask).float().reshape(-1) > 0.5
        valid = torch.isfinite(values)
        same = values[valid & mask]
        cross = values[valid & (~mask)]
    else:
        same = _to_tensor(same_text_value).float().reshape(-1)
        cross = _to_tensor(cross_text_value).float().reshape(-1) if cross_text_value is not None else same.new_empty((0,))
        same = same[torch.isfinite(same)]
        cross = cross[torch.isfinite(cross)]
    if same.numel() <= 0 or cross.numel() <= 0:
        return torch.tensor(float("nan"))
    return same.mean() - cross.mean()


def cumulative_drift(prefix_offset: Any) -> torch.Tensor:
    return cumulative_drift_mean_abs(prefix_offset)


def final_prefix_drift_abs_mean(prefix_offset: Any) -> torch.Tensor:
    offset = _to_tensor(prefix_offset).float()
    if offset.numel() <= 0:
        return torch.tensor(float("nan"))
    if offset.ndim == 0:
        return offset.abs()
    final = offset[..., -1].reshape(-1)
    valid = final[torch.isfinite(final)]
    if valid.numel() <= 0:
        return torch.tensor(float("nan"))
    return valid.abs().mean()


def final_prefix_offset_abs_mean(prefix_offset: Any) -> torch.Tensor:
    return final_prefix_drift_abs_mean(prefix_offset)


def cumulative_drift_mean_abs(prefix_offset: Any) -> torch.Tensor:
    offset = _to_tensor(prefix_offset).float()
    if offset.numel() <= 0:
        return torch.tensor(float("nan"))
    if offset.ndim == 0:
        return offset.abs()
    flat = offset.reshape(offset.shape[0], -1)
    finite_mask = torch.isfinite(flat)
    finite_abs = torch.where(finite_mask, flat.abs(), torch.zeros_like(flat))
    count = finite_mask.float().sum(dim=1).clamp_min(1.0)
    mean_abs = finite_abs.sum(dim=1) / count
    valid = mean_abs[torch.isfinite(mean_abs)]
    if valid.numel() <= 0:
        return torch.tensor(float("nan"))
    return valid.mean()


def max_prefix_offset_abs(prefix_offset: Any) -> torch.Tensor:
    offset = _to_tensor(prefix_offset).float()
    if offset.numel() <= 0:
        return torch.tensor(float("nan"))
    if offset.ndim == 0:
        return offset.abs()
    flat = offset.reshape(offset.shape[0], -1).abs()
    finite_mask = torch.isfinite(flat)
    valid = torch.where(finite_mask, flat, torch.zeros_like(flat))
    count = finite_mask.any(dim=1)
    if not bool(count.any().item()):
        return torch.tensor(float("nan"))
    batch_max = valid.max(dim=1).values[count]
    return batch_max.mean()


def speech_weighted_mae(
    pred: Any,
    target: Any,
    speech_mask: Any,
    weight: Any | None = None,
) -> torch.Tensor:
    pred_t = _to_tensor(pred).float()
    target_t = _to_tensor(target).float()
    speech_t = _to_tensor(speech_mask).float()
    if weight is None:
        metric_weight = speech_t
    else:
        metric_weight = _to_tensor(weight).float() * speech_t
    diff = (pred_t - target_t).abs() * metric_weight
    return diff.sum() / metric_weight.sum().clamp_min(1.0)


def residual_bias_share(
    residual_pred: Any,
    speech_mask: Any,
    coarse_scalar: Any,
) -> torch.Tensor:
    residual = _to_tensor(residual_pred).float()
    speech = _to_tensor(speech_mask).float()
    coarse = _to_tensor(coarse_scalar).float().reshape(-1)

    if residual.ndim == 1:
        residual = residual.unsqueeze(0)
    if speech.ndim == 1:
        speech = speech.unsqueeze(0)

    batch = min(int(residual.size(0)), int(speech.size(0)))
    residual = residual[:batch]
    speech = speech[:batch]
    if coarse.numel() == 1 and batch > 1:
        coarse = coarse.repeat(batch)
    else:
        coarse = coarse[:batch]
    speech_mass = speech.sum(dim=1).clamp_min(1.0)
    mean_r = (residual * speech).sum(dim=1) / speech_mass
    rms_r = torch.sqrt(((residual.square() * speech).sum(dim=1) / speech_mass).clamp_min(1.0e-8))
    denom = coarse.abs() + rms_r + 1.0e-6
    return (mean_r.abs() / denom).mean()


def local_silence_delta_share(
    pred_learned: Any,
    pred_coarse: Any,
    speech_mask: Any,
    silence_mask: Any,
) -> torch.Tensor:
    learned = _to_tensor(pred_learned).float()
    coarse = _to_tensor(pred_coarse).float()
    speech = _to_tensor(speech_mask).float()
    silence = _to_tensor(silence_mask).float()
    delta = (learned - coarse).abs()
    num = (delta * silence).sum()
    den = (delta * speech).sum()
    if not torch.isfinite(den) or float(den.item()) <= 1.0e-3:
        return torch.tensor(float("nan"), device=delta.device)
    return num / den


def residual_target_stats(
    residual_pred: Any,
    residual_target: Any,
    speech_mask: Any,
) -> dict[str, float]:
    pred = _to_tensor(residual_pred).float().reshape(-1)
    target = _to_tensor(residual_target).float().reshape(-1)
    speech = _to_tensor(speech_mask).float().reshape(-1) > 0.5
    valid = speech & torch.isfinite(pred) & torch.isfinite(target)
    if not bool(valid.any().item()):
        return {
            "spearman": float("nan"),
            "robust_slope": float("nan"),
            "r2_like": float("nan"),
            "count": 0.0,
        }
    return tempo_explainability(pred[valid], target[valid])


__all__ = [
    "budget_hit_rate",
    "build_duration_v3_metric_sections",
    "build_rhythm_metric_dict",
    "build_rhythm_metric_sections",
    "cumulative_drift",
    "cumulative_drift_mean_abs",
    "final_prefix_offset_abs_mean",
    "final_prefix_drift_abs_mean",
    "local_silence_delta_share",
    "max_prefix_offset_abs",
    "prefix_discrepancy",
    "residual_bias_share",
    "residual_target_stats",
    "same_text_gap",
    "silence_leakage",
    "speech_weighted_mae",
    "monotonic_triplet_table",
    "tempo_explainability",
    "tempo_tie_rate",
    "tempo_monotonicity",
    "transfer_slope",
]


def build_duration_v3_metric_sections(output: dict[str, Any], sample: dict[str, Any] | None = None):
    return build_rhythm_metric_sections(output, sample=sample)
