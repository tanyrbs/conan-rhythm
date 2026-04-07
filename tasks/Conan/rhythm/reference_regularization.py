from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from modules.Conan.rhythm.reference_encoder import _resample_by_progress
from modules.Conan.rhythm.reference_descriptor import mean_speech_frames_to_global_rate
from modules.Conan.rhythm.source_boundary import resolve_boundary_score_unit


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

    boundary_score = resolve_boundary_score_unit(execution.planner)
    if boundary_score is None:
        boundary_score = (pause_exec > 1.0e-3).float()
    if boundary_score.dim() == 3 and boundary_score.size(-1) == 1:
        boundary_score = boundary_score.squeeze(-1)
    boundary_score = boundary_score.float().reshape(unit_mask.shape).clamp(0.0, 1.0) * unit_mask

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
        sim = pred_vec.index_select(0, indices) @ tgt_vec.index_select(0, indices).transpose(0, 1)
        sim = sim / max(float(temperature), 1.0e-6)
        losses.append(F.cross_entropy(sim, torch.arange(indices.numel(), device=sim.device)))
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
