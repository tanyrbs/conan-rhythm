from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


REF_RHYTHM_STATS_KEYS = (
    "pause_ratio",
    "mean_pause_frames",
    "mean_speech_frames",
    "rate_trend",
    "boundary_ratio",
    "voiced_ratio",
)

REF_RHYTHM_TRACE_KEYS = (
    "pause_indicator",
    "local_rate",
    "boundary_strength",
    "segment_duration_bias",
    "voiced_activity",
)


def _ensure_btf(ref_mel: torch.Tensor) -> torch.Tensor:
    if ref_mel.dim() == 2:
        ref_mel = ref_mel.unsqueeze(0)
    if ref_mel.dim() != 3:
        raise ValueError(f"ref_mel must be rank-3, got shape={tuple(ref_mel.shape)}")
    if ref_mel.size(-1) == 80:
        return ref_mel
    if ref_mel.size(1) == 80:
        return ref_mel.transpose(1, 2)
    if ref_mel.size(1) <= ref_mel.size(2):
        return ref_mel.transpose(1, 2)
    return ref_mel


def _masked_run_mean(mask: torch.Tensor) -> torch.Tensor:
    lengths = []
    for batch_mask in mask:
        values = batch_mask.tolist()
        runs = []
        run = 0
        target = int(values[0]) if len(values) > 0 else 0
        for value in values:
            value = int(value)
            if value == target:
                run += 1
            else:
                if target == 1:
                    runs.append(run)
                target = value
                run = 1
        if run > 0 and target == 1:
            runs.append(run)
        lengths.append(float(sum(runs) / max(len(runs), 1)) if runs else 0.0)
    return mask.new_tensor(lengths, dtype=torch.float32)


def _resample_by_progress(feature_track: torch.Tensor, progress: torch.Tensor, trace_bins: int) -> torch.Tensor:
    batch_size, total_frames, feat_dim = feature_track.shape
    trace = feature_track.new_zeros((batch_size, trace_bins, feat_dim))
    target_progress = torch.linspace(0.0, 1.0, trace_bins, device=feature_track.device)
    for batch_idx in range(batch_size):
        batch_progress = progress[batch_idx]
        batch_features = feature_track[batch_idx]
        for bin_idx, progress_value in enumerate(target_progress):
            right = int(torch.searchsorted(batch_progress, progress_value, right=False).item())
            if right <= 0:
                trace[batch_idx, bin_idx] = batch_features[0]
                continue
            if right >= total_frames:
                trace[batch_idx, bin_idx] = batch_features[-1]
                continue
            left = right - 1
            left_p = batch_progress[left]
            right_p = batch_progress[right]
            denom = (right_p - left_p).abs().clamp_min(1e-6)
            alpha = ((progress_value - left_p) / denom).clamp(0.0, 1.0)
            trace[batch_idx, bin_idx] = batch_features[left] * (1.0 - alpha) + batch_features[right] * alpha
    return trace


def sample_progress_trace(
    trace: torch.Tensor,
    phase_ptr: torch.Tensor,
    window_size: int,
    *,
    horizon: float = 0.35,
    visible_sizes: torch.Tensor | None = None,
) -> torch.Tensor:
    if trace.dim() != 3:
        raise ValueError(f"trace must be [B, bins, dim], got {tuple(trace.shape)}")
    batch_size, trace_bins, trace_dim = trace.shape
    if phase_ptr.dim() == 2 and phase_ptr.size(-1) == 1:
        phase_ptr = phase_ptr.squeeze(-1)
    if phase_ptr.dim() != 1 or phase_ptr.size(0) != batch_size:
        raise ValueError(
            f"phase_ptr must be [B] or [B,1], got {tuple(phase_ptr.shape)} for batch_size={batch_size}"
        )
    horizon = float(max(0.01, min(1.0, horizon)))
    if window_size <= 0:
        positions = trace.new_zeros((batch_size, 0))
    elif visible_sizes is None:
        offsets = torch.linspace(0.0, horizon, window_size, device=trace.device)
        positions = (phase_ptr[:, None] + offsets[None, :]).clamp(0.0, 1.0)
    else:
        visible_sizes = visible_sizes.long().to(device=trace.device)
        if visible_sizes.dim() != 1 or visible_sizes.size(0) != batch_size:
            raise ValueError(
                f"visible_sizes must be [B], got {tuple(visible_sizes.shape)} for batch_size={batch_size}"
            )
        step_ids = torch.arange(window_size, device=trace.device, dtype=torch.float32)[None, :]
        denom = (visible_sizes.float().clamp_min(1.0) - 1.0).unsqueeze(-1)
        denom = torch.where(visible_sizes.unsqueeze(-1) > 1, denom, torch.ones_like(denom))
        offsets = (step_ids / denom).clamp(0.0, 1.0) * horizon
        offsets = torch.where(
            visible_sizes.unsqueeze(-1) > 1,
            offsets,
            torch.zeros_like(offsets),
        )
        positions = (phase_ptr[:, None] + offsets).clamp(0.0, 1.0)
    scaled = positions * max(trace_bins - 1, 1)
    left = torch.floor(scaled).long().clamp(0, max(trace_bins - 1, 0))
    right = (left + 1).clamp(0, max(trace_bins - 1, 0))
    alpha = (scaled - left.float()).unsqueeze(-1)
    gathered_left = trace.gather(1, left.unsqueeze(-1).expand(-1, -1, trace_dim))
    gathered_right = trace.gather(1, right.unsqueeze(-1).expand(-1, -1, trace_dim))
    return gathered_left * (1.0 - alpha) + gathered_right * alpha


class ReferenceRhythmEncoder(nn.Module):
    """Progress-normalized rhythm conditioning."""

    def __init__(
        self,
        *,
        trace_bins: int = 24,
        smooth_kernel: int = 5,
        trace_horizon: float = 0.35,
        pause_energy_threshold_std: float = -0.5,
        pause_delta_quantile: float = 0.35,
        voiced_energy_threshold_std: float = -0.1,
        boundary_quantile: float = 0.75,
    ) -> None:
        super().__init__()
        self.trace_bins = max(4, int(trace_bins))
        self.smooth_kernel = max(1, int(smooth_kernel))
        self.trace_horizon = float(max(0.01, min(1.0, trace_horizon)))
        self.pause_energy_threshold_std = float(pause_energy_threshold_std)
        self.pause_delta_quantile = float(max(0.01, min(0.95, pause_delta_quantile)))
        self.voiced_energy_threshold_std = float(voiced_energy_threshold_std)
        self.boundary_quantile = float(max(0.50, min(0.99, boundary_quantile)))
        self.trace_dim = len(REF_RHYTHM_TRACE_KEYS)
        self.stats_dim = len(REF_RHYTHM_STATS_KEYS)

    def forward(self, ref_mel: torch.Tensor) -> dict[str, torch.Tensor]:
        ref_mel = _ensure_btf(ref_mel).float()
        energy = ref_mel.mean(dim=-1)
        energy_mean = energy.mean(dim=1, keepdim=True)
        energy_std = energy.std(dim=1, keepdim=True).clamp_min(1e-6)
        energy_z = (energy - energy_mean) / energy_std
        delta = torch.zeros_like(energy)
        delta[:, 1:] = (energy[:, 1:] - energy[:, :-1]).abs()
        delta_threshold = torch.quantile(delta, self.pause_delta_quantile, dim=1, keepdim=True)
        pause_mask = (energy_z <= self.pause_energy_threshold_std) & (delta <= delta_threshold)
        speech_mask = ~pause_mask
        voiced = energy_z.gt(self.voiced_energy_threshold_std).float()

        kernel = min(self.smooth_kernel, max(1, energy.size(1)))
        padding = kernel // 2
        local_rate = F.avg_pool1d(
            delta.unsqueeze(1),
            kernel_size=kernel,
            stride=1,
            padding=padding,
        ).squeeze(1)
        boundary_strength = F.avg_pool1d(
            delta.unsqueeze(1),
            kernel_size=min(kernel + 2, max(1, energy.size(1))),
            stride=1,
            padding=min(kernel + 2, max(1, energy.size(1))) // 2,
        ).squeeze(1)
        if local_rate.size(1) > energy.size(1):
            local_rate = local_rate[:, : energy.size(1)]
        if boundary_strength.size(1) > energy.size(1):
            boundary_strength = boundary_strength[:, : energy.size(1)]

        speech_progress = speech_mask.float().cumsum(dim=1)
        speech_total = speech_progress[:, -1:].clamp_min(1.0)
        progress = speech_progress / speech_total
        uniform = torch.linspace(0.0, 1.0, energy.size(1), device=energy.device)[None, :]
        segment_duration_bias = progress - uniform

        boundary_events = boundary_strength >= torch.quantile(
            boundary_strength,
            self.boundary_quantile,
            dim=1,
            keepdim=True,
        )

        feature_track = torch.stack(
            [
                pause_mask.float(),
                local_rate,
                boundary_events.float(),
                segment_duration_bias,
                voiced,
            ],
            dim=-1,
        )
        trace = _resample_by_progress(feature_track, progress, self.trace_bins)
        stats = torch.stack(
            [
                pause_mask.float().mean(dim=1),
                _masked_run_mean(pause_mask.long()),
                _masked_run_mean(speech_mask.long()),
                (local_rate[:, -1] - local_rate[:, 0]),
                boundary_events.float().mean(dim=1),
                voiced.mean(dim=1),
            ],
            dim=-1,
        )
        return {
            "ref_rhythm_stats": stats,
            "ref_rhythm_trace": trace,
        }

    def sample_trace_window(
        self,
        ref_rhythm_trace: torch.Tensor,
        phase_ptr: torch.Tensor,
        window_size: int,
        *,
        horizon: float | None = None,
        visible_sizes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return sample_progress_trace(
            ref_rhythm_trace,
            phase_ptr,
            window_size,
            horizon=self.trace_horizon if horizon is None else horizon,
            visible_sizes=visible_sizes,
        )
