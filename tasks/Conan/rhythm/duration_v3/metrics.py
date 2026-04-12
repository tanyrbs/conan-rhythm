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
) -> torch.Tensor:
    slow = _to_tensor(tempo_slow).reshape(-1)
    mid = _to_tensor(tempo_mid).reshape(-1)
    fast = _to_tensor(tempo_fast).reshape(-1)
    valid = torch.isfinite(slow) & torch.isfinite(mid) & torch.isfinite(fast)
    if not bool(valid.any().item()):
        return slow.new_tensor(float("nan"))
    return (((mid[valid] - slow[valid]) > float(margin)) & ((fast[valid] - mid[valid]) > float(margin))).float().mean()


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
    return tempo_explainability(delta_g, tempo_delta)


def monotonic_triplet_table(
    sample_ids: Any,
    tempo_slow: Any,
    tempo_mid: Any,
    tempo_fast: Any,
) -> dict[str, list[float | str]]:
    slow = _to_tensor(tempo_slow).reshape(-1)
    mid = _to_tensor(tempo_mid).reshape(-1)
    fast = _to_tensor(tempo_fast).reshape(-1)
    count = min(int(slow.numel()), int(mid.numel()), int(fast.numel()))
    mono = ((slow[:count] < mid[:count]) & (mid[:count] < fast[:count])).float()
    values = sample_ids[:count] if isinstance(sample_ids, (list, tuple)) else None
    if values is None:
        values = [str(index) for index in range(count)]
    return {
        "sample_id": [str(value) for value in values],
        "tempo_slow": [float(value) for value in slow[:count].tolist()],
        "tempo_mid": [float(value) for value in mid[:count].tolist()],
        "tempo_fast": [float(value) for value in fast[:count].tolist()],
        "mono_ok": [float(value) for value in mono[:count].tolist()],
    }


def silence_leakage(delta_z: Any, speech_mask: Any, silence_mask: Any) -> torch.Tensor:
    dz = _to_tensor(delta_z).float()
    speech = _to_tensor(speech_mask).float()
    silence = _to_tensor(silence_mask).float()
    numerator = (dz.abs() * silence).sum()
    denominator = (dz.abs() * speech).sum().clamp_min(1.0e-6)
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


__all__ = [
    "budget_hit_rate",
    "build_duration_v3_metric_sections",
    "build_rhythm_metric_dict",
    "build_rhythm_metric_sections",
    "cumulative_drift",
    "prefix_discrepancy",
    "same_text_gap",
    "silence_leakage",
    "monotonic_triplet_table",
    "tempo_explainability",
    "tempo_tie_rate",
    "tempo_monotonicity",
    "transfer_slope",
]


def build_duration_v3_metric_sections(output: dict[str, Any], sample: dict[str, Any] | None = None):
    return build_rhythm_metric_sections(output, sample=sample)
