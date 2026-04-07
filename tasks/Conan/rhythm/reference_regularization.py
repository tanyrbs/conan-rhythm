from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from modules.Conan.rhythm.reference_encoder import _resample_by_progress
from modules.Conan.rhythm.reference_descriptor import mean_speech_frames_to_global_rate


@dataclass(frozen=True)
class CompactReferenceDescriptor:
    global_rate: torch.Tensor
    pause_ratio: torch.Tensor
    local_rate_trace: torch.Tensor
    boundary_trace: torch.Tensor

    def flatten(self) -> torch.Tensor:
        return torch.cat(
            [
                self.global_rate.float().reshape(self.global_rate.size(0), -1),
                self.pause_ratio.float().reshape(self.pause_ratio.size(0), -1),
                self.local_rate_trace.float().reshape(self.local_rate_trace.size(0), -1),
                self.boundary_trace.float().reshape(self.boundary_trace.size(0), -1),
            ],
            dim=-1,
        )


def _reshape_batch_scalar(value: torch.Tensor) -> torch.Tensor:
    if value.dim() == 0:
        return value.view(1, 1)
    return value.float().reshape(value.size(0), -1)[:, :1]


def _normalize_unit_mask(unit_mask: torch.Tensor) -> torch.Tensor:
    if unit_mask.dim() == 3 and unit_mask.size(-1) == 1:
        unit_mask = unit_mask.squeeze(-1)
    return unit_mask.float()


def _build_executed_boundary_proxy(
    *,
    pause_exec: torch.Tensor,
    unit_mask: torch.Tensor,
) -> torch.Tensor:
    """Build a planner-independent boundary proxy from executed pause mass.

    Descriptor consistency should describe the *executed* rhythm. Using
    `planner.boundary_score_unit` here leaks planner-side conditioning because
    that score already mixes source-boundary cues with reference boundary
    evidence. For pairwise/external-reference bootstrap, that creates a shortcut:
    the model can look descriptor-consistent without really executing B-like
    boundary placement.

    We therefore derive the predicted boundary trace from executed blank mass
    only. The proxy is intentionally shape-oriented:

      - log1p keeps long pauses from dominating
      - mean-centering removes the global pause budget, which is already covered
        by `pause_ratio`
      - positive-only normalization highlights relative boundary peaks instead
        of replaying planner-side priors
    """

    pause_exec = pause_exec.float().clamp_min(0.0) * unit_mask
    visible = unit_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    pause_feature = torch.log1p(pause_exec)
    pause_mean = (pause_feature * unit_mask).sum(dim=1, keepdim=True) / visible
    boundary_positive = (pause_feature - pause_mean).clamp_min(0.0) * unit_mask
    boundary_scale = boundary_positive.amax(dim=1, keepdim=True)
    has_boundary_variation = boundary_scale > 1.0e-6
    normalized = boundary_positive / boundary_scale.clamp_min(1.0e-6)
    return torch.where(has_boundary_variation, normalized, torch.zeros_like(normalized))


def build_target_compact_reference_descriptor(sample: dict) -> CompactReferenceDescriptor | None:
    ref_stats = sample.get("ref_rhythm_stats")
    ref_trace = sample.get("ref_rhythm_trace")
    if not isinstance(ref_stats, torch.Tensor) or not isinstance(ref_trace, torch.Tensor):
        return None
    mean_speech_frames = ref_stats.float()[:, 2:3]
    global_rate = sample.get("global_rate")
    if not isinstance(global_rate, torch.Tensor):
        global_rate = mean_speech_frames_to_global_rate(mean_speech_frames)
    else:
        global_rate = _reshape_batch_scalar(global_rate)
        global_rate = torch.where(
            mean_speech_frames > 0.0,
            global_rate,
            torch.zeros_like(mean_speech_frames),
        )
    pause_ratio = sample.get("pause_ratio")
    if not isinstance(pause_ratio, torch.Tensor):
        pause_ratio = ref_stats.float()[:, 0:1].clamp(0.0, 1.0)
    local_rate_trace = sample.get("local_rate_trace")
    if not isinstance(local_rate_trace, torch.Tensor):
        local_rate_trace = ref_trace.float()[:, :, 1:2]
    boundary_trace = sample.get("boundary_trace")
    if not isinstance(boundary_trace, torch.Tensor):
        boundary_trace = ref_trace.float()[:, :, 2:3]
    return CompactReferenceDescriptor(
        global_rate=_reshape_batch_scalar(global_rate),
        pause_ratio=_reshape_batch_scalar(pause_ratio),
        local_rate_trace=local_rate_trace.float(),
        boundary_trace=boundary_trace.float(),
    )


def build_predicted_compact_reference_descriptor(
    output: dict,
    *,
    trace_bins: int | None = None,
) -> CompactReferenceDescriptor | None:
    execution = output.get("rhythm_execution")
    unit_batch = output.get("rhythm_unit_batch")
    if execution is None or unit_batch is None:
        return None
    unit_mask = _normalize_unit_mask(unit_batch.unit_mask)
    dur_anchor_src = unit_batch.dur_anchor_src.float().reshape(unit_mask.shape)
    speech_exec = execution.speech_duration_exec.float().reshape(unit_mask.shape).clamp_min(0.0) * unit_mask
    pause_exec = getattr(execution, "blank_duration_exec", execution.pause_after_exec)
    pause_exec = pause_exec.float().reshape(unit_mask.shape).clamp_min(0.0) * unit_mask
    active_units = unit_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    speech_total = speech_exec.sum(dim=1, keepdim=True)
    global_rate = torch.where(
        speech_total > 1.0e-6,
        active_units / speech_total.clamp_min(1.0e-6),
        torch.zeros_like(speech_total),
    )
    pause_ratio = pause_exec.sum(dim=1, keepdim=True) / (speech_total + pause_exec.sum(dim=1, keepdim=True)).clamp_min(1.0e-6)

    relative_speech = speech_exec / dur_anchor_src.clamp_min(1.0)
    mean_relative_speech = (relative_speech * unit_mask).sum(dim=1, keepdim=True) / active_units
    local_rate_unit = relative_speech / mean_relative_speech.clamp_min(1.0e-6)

    boundary_score = _build_executed_boundary_proxy(
        pause_exec=pause_exec,
        unit_mask=unit_mask,
    )

    trace_bins = int(trace_bins or 0)
    if trace_bins <= 0:
        trace_bins = 24

    speech_progress = torch.cumsum(speech_exec, dim=1)
    speech_total_for_progress = speech_progress[:, -1:].clamp_min(1.0)
    progress_from_speech = speech_progress / speech_total_for_progress
    fallback_progress = torch.cumsum(unit_mask, dim=1) / active_units
    progress = torch.where(speech_total > 1.0e-6, progress_from_speech, fallback_progress)
    feature_track = torch.stack([local_rate_unit, boundary_score], dim=-1)
    trace = _resample_by_progress(feature_track, progress, trace_bins)

    return CompactReferenceDescriptor(
        global_rate=global_rate,
        pause_ratio=pause_ratio,
        local_rate_trace=trace[:, :, 0:1],
        boundary_trace=trace[:, :, 1:2].clamp(0.0, 1.0),
    )


def _normalize_trace_shape(trace: torch.Tensor) -> torch.Tensor:
    centered = trace.float() - trace.float().mean(dim=1, keepdim=True)
    scale = centered.abs().mean(dim=1, keepdim=True).clamp_min(1.0e-4)
    return centered / scale


def _compute_group_gap_scales(
    target_vec: torch.Tensor,
    *,
    gap_floor: float,
    min_scale: float,
    gap_power: float,
) -> torch.Tensor:
    if target_vec.size(0) <= 1:
        return target_vec.new_ones((target_vec.size(0),))
    similarity = target_vec @ target_vec.transpose(0, 1)
    gap = (1.0 - similarity).clamp_min(0.0)
    eye = torch.eye(gap.size(0), device=gap.device, dtype=torch.bool)
    gap = gap.masked_fill(eye, 0.0)
    mean_gap = gap.sum(dim=1) / float(max(gap.size(0) - 1, 1))
    denom = mean_gap + max(float(gap_floor), 1.0e-6)
    scaled = (mean_gap / denom.clamp_min(1.0e-6)).clamp(0.0, 1.0)
    scaled = scaled.pow(max(float(gap_power), 1.0e-6))
    return scaled.clamp(min=float(min_scale), max=1.0).detach()


def compute_descriptor_consistency_loss(
    pred: CompactReferenceDescriptor,
    target: CompactReferenceDescriptor,
    *,
    local_weight: float = 1.0,
    boundary_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    global_rate_loss = F.smooth_l1_loss(
        torch.log1p(pred.global_rate.float().clamp_min(0.0)),
        torch.log1p(target.global_rate.float().clamp_min(0.0)),
    )
    pause_ratio_loss = F.smooth_l1_loss(
        pred.pause_ratio.float().clamp(0.0, 1.0),
        target.pause_ratio.float().clamp(0.0, 1.0),
    )
    local_rate_loss = F.smooth_l1_loss(
        _normalize_trace_shape(pred.local_rate_trace),
        _normalize_trace_shape(target.local_rate_trace),
    )
    boundary_loss = F.smooth_l1_loss(
        pred.boundary_trace.float().clamp(0.0, 1.0),
        target.boundary_trace.float().clamp(0.0, 1.0),
    )
    total = global_rate_loss + pause_ratio_loss + float(local_weight) * local_rate_loss + float(boundary_weight) * boundary_loss
    return {
        "stats": global_rate_loss + pause_ratio_loss,
        "local_trace": local_rate_loss,
        "boundary_trace": boundary_loss,
        "total": total,
    }


def compute_group_reference_contrastive_loss(
    pred: CompactReferenceDescriptor,
    target: CompactReferenceDescriptor,
    group_ids: torch.Tensor | None,
    *,
    temperature: float = 0.10,
    gap_floor: float = 0.10,
    min_scale: float = 0.50,
    gap_power: float = 1.0,
) -> torch.Tensor | None:
    if group_ids is None:
        return None
    group_ids = group_ids.long().reshape(group_ids.size(0), -1)[:, 0]
    if group_ids.numel() <= 1:
        return None
    pred_vec = F.normalize(pred.flatten().float(), dim=-1)
    tgt_vec = F.normalize(target.flatten().float(), dim=-1)
    losses = []
    for group_id in torch.unique(group_ids):
        indices = torch.nonzero(group_ids == group_id, as_tuple=False).reshape(-1)
        if indices.numel() <= 1:
            continue
        group_pred = pred_vec.index_select(0, indices)
        group_tgt = tgt_vec.index_select(0, indices)
        sim = group_pred @ group_tgt.transpose(0, 1)
        sim = sim / max(float(temperature), 1.0e-6)
        per_anchor_loss = F.cross_entropy(
            sim,
            torch.arange(indices.numel(), device=sim.device),
            reduction="none",
        )
        gap_scale = _compute_group_gap_scales(
            group_tgt,
            gap_floor=float(gap_floor),
            min_scale=float(min_scale),
            gap_power=float(gap_power),
        )
        losses.append((per_anchor_loss * gap_scale).sum() / gap_scale.sum().clamp_min(1.0e-6))
    if not losses:
        return None
    return torch.stack(losses).mean()


__all__ = [
    "CompactReferenceDescriptor",
    "build_predicted_compact_reference_descriptor",
    "build_target_compact_reference_descriptor",
    "compute_descriptor_consistency_loss",
    "compute_group_reference_contrastive_loss",
]
