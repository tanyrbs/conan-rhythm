from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts import RhythmTeacherTargets
from .reference_encoder import sample_progress_trace


@dataclass
class AlgorithmicTeacherConfig:
    rate_scale_min: float = 0.55
    rate_scale_max: float = 1.95
    local_rate_strength: float = 0.45
    segment_bias_strength: float = 0.30
    pause_strength: float = 1.10
    boundary_strength: float = 1.50
    source_boundary_pause_weight: float = 0.35
    source_boundary_prior_clip: float = 1.50
    source_boundary_gate_floor: float = 0.05
    source_boundary_gate_ceiling: float = 0.55
    source_boundary_agreement_center: float = 0.15
    source_boundary_agreement_scale: float = 4.0
    pause_budget_ratio_cap: float = 0.80
    speech_smooth_kernel: int = 3
    pause_topk_ratio: float = 0.30
    phrase_final_bonus: float = 0.20
    confidence_bonus: float = 0.05


def _ensure_batched_vector(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return x.unsqueeze(0)
    return x


def _masked_standardize(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    total = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (x * mask).sum(dim=1, keepdim=True) / total
    var = (((x - mean) ** 2) * mask).sum(dim=1, keepdim=True) / total
    return ((x - mean) / var.clamp_min(1e-6).sqrt()) * mask


def _masked_normalize(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    x = x * mask.float()
    return x / x.sum(dim=1, keepdim=True).clamp_min(1e-6)


def _masked_cosine_similarity(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    x_masked = x * mask
    y_masked = y * mask
    dot = (x_masked * y_masked).sum(dim=1)
    x_norm = x_masked.square().sum(dim=1).clamp_min(1e-6).sqrt()
    y_norm = y_masked.square().sum(dim=1).clamp_min(1e-6).sqrt()
    return dot / (x_norm * y_norm).clamp_min(1e-6)


def _smooth_1d(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = max(1, int(kernel_size))
    if kernel_size <= 1 or x.size(1) <= 1:
        return x
    kernel_size = min(kernel_size, int(x.size(1)))
    padding = kernel_size // 2
    y = F.avg_pool1d(x.unsqueeze(1), kernel_size=kernel_size, stride=1, padding=padding).squeeze(1)
    if y.size(1) > x.size(1):
        y = y[:, : x.size(1)]
    return y


def _sparsify_scores(x: torch.Tensor, mask: torch.Tensor, *, topk_ratio: float) -> torch.Tensor:
    mask = mask.float()
    out = x.new_zeros(x.shape)
    ratio = max(0.0, min(1.0, float(topk_ratio)))
    for batch_idx in range(x.size(0)):
        visible = int(mask[batch_idx].sum().item())
        if visible <= 0:
            continue
        topk = max(1, min(visible, int(round(visible * ratio))))
        values, indices = torch.topk((x[batch_idx] * mask[batch_idx])[:visible], k=topk, dim=0)
        out[batch_idx, indices] = values
    return out * mask


def _estimate_confidence(
    *,
    trace_context: torch.Tensor,
    ref_rhythm_stats: torch.Tensor,
    unit_mask: torch.Tensor,
    smoother_bonus: float = 0.0,
) -> torch.Tensor:
    unit_mask = unit_mask.float()
    visible = unit_mask.sum(dim=1).clamp_min(1.0)
    pause_track = (trace_context[:, :, 0] * unit_mask).sum(dim=1) / visible
    boundary_track = (trace_context[:, :, 2] * unit_mask).sum(dim=1) / visible
    local_rate_mean = (trace_context[:, :, 1] * unit_mask).sum(dim=1) / visible
    local_rate_var = (((trace_context[:, :, 1] - local_rate_mean[:, None]) ** 2) * unit_mask).sum(dim=1) / visible
    pause_ratio = ref_rhythm_stats[:, 0].clamp(0.0, 1.0)
    boundary_ratio = ref_rhythm_stats[:, 4].clamp(0.0, 1.0)
    confidence = 0.20 + 0.30 * pause_ratio + 0.25 * boundary_ratio + 0.20 * boundary_track
    confidence = confidence + 0.10 * torch.exp(-local_rate_var.clamp_min(0.0))
    confidence = confidence + float(smoother_bonus)
    return confidence.clamp(0.05, 1.0).unsqueeze(-1)


def _build_source_boundary_pause_prior(
    *,
    source_boundary_cue: torch.Tensor,
    trace_boundary_context: torch.Tensor,
    unit_mask: torch.Tensor,
    cfg: AlgorithmicTeacherConfig,
) -> torch.Tensor:
    source_prior = _masked_standardize(source_boundary_cue.float(), unit_mask)
    source_prior = source_prior.clamp(
        min=-float(cfg.source_boundary_prior_clip),
        max=float(cfg.source_boundary_prior_clip),
    )
    ref_boundary = _masked_standardize(trace_boundary_context.float(), unit_mask)
    agreement = _masked_cosine_similarity(source_prior, ref_boundary, unit_mask).clamp(-1.0, 1.0)
    gate = torch.sigmoid(
        (agreement - float(cfg.source_boundary_agreement_center)) * float(cfg.source_boundary_agreement_scale)
    )
    gate_floor = float(cfg.source_boundary_gate_floor)
    gate_ceiling = float(cfg.source_boundary_gate_ceiling)
    if gate_ceiling < gate_floor:
        gate_floor, gate_ceiling = gate_ceiling, gate_floor
    gate = gate_floor + (gate_ceiling - gate_floor) * gate
    return float(cfg.source_boundary_pause_weight) * gate.unsqueeze(-1) * source_prior


def _build_prefix_carry_targets(
    *,
    dur_anchor_src: torch.Tensor,
    speech_exec: torch.Tensor,
    pause_exec: torch.Tensor,
    unit_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    unit_mask = unit_mask.float()
    prefix_clock = torch.cumsum(
        ((speech_exec + pause_exec) - dur_anchor_src.float()) * unit_mask,
        dim=1,
    ) * unit_mask
    prefix_backlog = prefix_clock.clamp_min(0.0) * unit_mask
    return prefix_clock, prefix_backlog


def build_algorithmic_teacher_targets(
    *,
    dur_anchor_src: torch.Tensor,
    ref_rhythm_stats: torch.Tensor,
    ref_rhythm_trace: torch.Tensor,
    unit_mask: torch.Tensor | None = None,
    source_boundary_cue: torch.Tensor | None = None,
    config: AlgorithmicTeacherConfig | None = None,
) -> RhythmTeacherTargets:
    cfg = config or AlgorithmicTeacherConfig()
    dur_anchor_src = _ensure_batched_vector(dur_anchor_src.float())
    ref_rhythm_stats = _ensure_batched_vector(ref_rhythm_stats.float())
    if ref_rhythm_trace.dim() == 2:
        ref_rhythm_trace = ref_rhythm_trace.unsqueeze(0)
    if unit_mask is None:
        unit_mask = dur_anchor_src.gt(0).float()
    else:
        unit_mask = _ensure_batched_vector(unit_mask.float())
    if source_boundary_cue is None:
        source_boundary_cue = dur_anchor_src.new_zeros(dur_anchor_src.shape)
    else:
        source_boundary_cue = _ensure_batched_vector(source_boundary_cue.float())

    visible_sizes = unit_mask.sum(dim=1).long().clamp_min(1)
    trace_context = sample_progress_trace(
        ref_rhythm_trace,
        phase_ptr=torch.zeros(dur_anchor_src.size(0), device=dur_anchor_src.device),
        window_size=int(dur_anchor_src.size(1)),
        horizon=1.0,
        visible_sizes=visible_sizes,
    ) * unit_mask.unsqueeze(-1)

    src_total = (dur_anchor_src * unit_mask).sum(dim=1, keepdim=True).clamp_min(1.0)
    src_mean = src_total / visible_sizes.unsqueeze(-1).float().clamp_min(1.0)
    ref_mean_speech = ref_rhythm_stats[:, 2:3].clamp_min(1.0)
    rate_scale = (ref_mean_speech / src_mean.clamp_min(1.0)).clamp(cfg.rate_scale_min, cfg.rate_scale_max)
    speech_budget = src_total * rate_scale

    pause_ratio = ref_rhythm_stats[:, 0:1].clamp(0.0, 0.49)
    boundary_ratio = ref_rhythm_stats[:, 4:5].clamp(0.0, 1.0)
    mean_pause = ref_rhythm_stats[:, 1:2].clamp_min(0.0)
    pause_from_ratio = speech_budget * pause_ratio / (1.0 - pause_ratio).clamp_min(0.20)
    pause_from_events = visible_sizes.unsqueeze(-1).float() * boundary_ratio * mean_pause
    pause_budget = 0.35 * pause_from_ratio + 0.65 * pause_from_events
    pause_budget = pause_budget.clamp_min(0.0)
    pause_budget = torch.minimum(pause_budget, speech_budget * cfg.pause_budget_ratio_cap)

    local_rate = _masked_standardize(trace_context[:, :, 1], unit_mask)
    segment_bias = _masked_standardize(trace_context[:, :, 3], unit_mask)
    phrase_final = torch.roll(source_boundary_cue.float(), shifts=-1, dims=1)
    phrase_final[:, -1] = 1.0
    speech_scores = torch.exp(
        torch.log1p(dur_anchor_src.clamp_min(0.0))
        + cfg.local_rate_strength * local_rate
        + cfg.segment_bias_strength * segment_bias
        + cfg.phrase_final_bonus * phrase_final
    ) * unit_mask
    speech_scores = _smooth_1d(speech_scores, cfg.speech_smooth_kernel) * unit_mask
    speech_scores = _masked_normalize(speech_scores, unit_mask)
    speech_exec = speech_scores * speech_budget

    pause_seed = cfg.pause_strength * _masked_standardize(trace_context[:, :, 0], unit_mask)
    pause_seed = pause_seed + cfg.boundary_strength * _masked_standardize(trace_context[:, :, 2], unit_mask)
    pause_seed = pause_seed + _build_source_boundary_pause_prior(
        source_boundary_cue=source_boundary_cue,
        trace_boundary_context=trace_context[:, :, 2],
        unit_mask=unit_mask,
        cfg=cfg,
    )
    pause_scores = torch.exp(pause_seed) * unit_mask
    pause_scores = _sparsify_scores(
        pause_scores,
        unit_mask,
        topk_ratio=max(float(cfg.pause_topk_ratio), float(boundary_ratio.max().item())),
    )
    pause_scores = _masked_normalize(pause_scores, unit_mask)
    pause_exec = pause_scores * pause_budget

    allocation_tgt = _masked_normalize(speech_exec + pause_exec, unit_mask)
    prefix_clock_tgt, prefix_backlog_tgt = _build_prefix_carry_targets(
        dur_anchor_src=dur_anchor_src,
        speech_exec=speech_exec,
        pause_exec=pause_exec,
        unit_mask=unit_mask,
    )
    confidence = _estimate_confidence(
        trace_context=trace_context,
        ref_rhythm_stats=ref_rhythm_stats,
        unit_mask=unit_mask,
        smoother_bonus=cfg.confidence_bonus,
    )
    return RhythmTeacherTargets(
        speech_exec_tgt=speech_exec,
        pause_exec_tgt=pause_exec,
        speech_budget_tgt=speech_budget,
        pause_budget_tgt=pause_budget,
        allocation_tgt=allocation_tgt,
        confidence=confidence,
        trace_context=trace_context,
        prefix_clock_tgt=prefix_clock_tgt,
        prefix_backlog_tgt=prefix_backlog_tgt,
    )


class AlgorithmicRhythmTeacher(nn.Module):
    def __init__(self, config: AlgorithmicTeacherConfig | None = None) -> None:
        super().__init__()
        self.config = config or AlgorithmicTeacherConfig()

    def forward(
        self,
        *,
        dur_anchor_src: torch.Tensor,
        ref_rhythm_stats: torch.Tensor,
        ref_rhythm_trace: torch.Tensor,
        unit_mask: torch.Tensor | None = None,
        source_boundary_cue: torch.Tensor | None = None,
    ) -> RhythmTeacherTargets:
        return build_algorithmic_teacher_targets(
            dur_anchor_src=dur_anchor_src,
            ref_rhythm_stats=ref_rhythm_stats,
            ref_rhythm_trace=ref_rhythm_trace,
            unit_mask=unit_mask,
            source_boundary_cue=source_boundary_cue,
            config=self.config,
        )
