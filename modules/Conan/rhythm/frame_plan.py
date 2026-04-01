from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass
class BlankSlotSchedule:
    slot_duration_exec: torch.Tensor
    slot_mask: torch.Tensor
    slot_is_blank: torch.Tensor
    slot_unit_index: torch.Tensor


@dataclass
class RhythmFramePlan:
    frame_src_index: torch.Tensor
    frame_is_blank: torch.Tensor
    frame_slot_index: torch.Tensor
    frame_unit_index: torch.Tensor
    total_mask: torch.Tensor
    speech_mask: torch.Tensor
    blank_mask: torch.Tensor
    frame_phase_features: torch.Tensor


def _pad_sequences(
    sequences: list[torch.Tensor],
    *,
    pad_value: float = 0.0,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if len(sequences) <= 0:
        return torch.zeros((0, 0), dtype=dtype or torch.float32)
    max_len = max(int(seq.size(0)) for seq in sequences)
    tail_shape = tuple(sequences[0].shape[1:])
    out = sequences[0].new_full((len(sequences), max_len, *tail_shape), pad_value)
    if dtype is not None:
        out = out.to(dtype=dtype)
    for idx, seq in enumerate(sequences):
        if seq.numel() > 0:
            out[idx, : seq.size(0)] = seq.to(dtype=out.dtype)
    return out


def build_interleaved_blank_slot_schedule(
    *,
    speech_duration_exec: torch.Tensor,
    blank_duration_exec: torch.Tensor,
    unit_mask: torch.Tensor,
) -> BlankSlotSchedule:
    speech_duration_exec = speech_duration_exec.float()
    blank_duration_exec = blank_duration_exec.float()
    unit_mask = unit_mask.float()
    batch_size, num_units = speech_duration_exec.shape
    slot_count = num_units * 2
    slot_duration = speech_duration_exec.new_zeros((batch_size, slot_count))
    slot_mask = speech_duration_exec.new_zeros((batch_size, slot_count))
    slot_is_blank = torch.zeros((batch_size, slot_count), dtype=torch.long, device=speech_duration_exec.device)
    slot_unit_index = torch.zeros((batch_size, slot_count), dtype=torch.long, device=speech_duration_exec.device)
    unit_ids = torch.arange(num_units, device=speech_duration_exec.device)[None, :]
    slot_duration[:, 0::2] = speech_duration_exec
    slot_duration[:, 1::2] = blank_duration_exec
    slot_mask[:, 0::2] = unit_mask
    slot_mask[:, 1::2] = unit_mask
    slot_is_blank[:, 1::2] = 1
    slot_unit_index[:, 0::2] = unit_ids
    slot_unit_index[:, 1::2] = unit_ids
    return BlankSlotSchedule(
        slot_duration_exec=slot_duration,
        slot_mask=slot_mask,
        slot_is_blank=slot_is_blank,
        slot_unit_index=slot_unit_index,
    )


def build_frame_plan(
    *,
    dur_anchor_src: torch.Tensor,
    slot_duration_exec: torch.Tensor,
    slot_mask: torch.Tensor,
    slot_is_blank: torch.Tensor,
    slot_unit_index: torch.Tensor,
) -> RhythmFramePlan:
    device = slot_duration_exec.device
    slot_duration_exec = torch.round(slot_duration_exec.float()).long().clamp_min(0)
    slot_mask = slot_mask.float()
    slot_is_blank = slot_is_blank.long()
    slot_unit_index = slot_unit_index.long()
    src_anchor = torch.round(dur_anchor_src.float()).long().clamp_min(0)

    frame_src_index_list: list[torch.Tensor] = []
    frame_blank_list: list[torch.Tensor] = []
    frame_slot_index_list: list[torch.Tensor] = []
    frame_unit_index_list: list[torch.Tensor] = []
    frame_phase_feature_list: list[torch.Tensor] = []
    valid_lengths: list[int] = []

    for batch_idx in range(slot_duration_exec.size(0)):
        src_cursor = 0
        visible_slots = int(slot_mask[batch_idx].sum().item())
        src_total = int(src_anchor[batch_idx].sum().item())
        frame_src_index = []
        frame_blank = []
        frame_slot_index = []
        frame_unit_index = []
        frame_phase_features = []
        valid_len = 0
        for slot_idx in range(visible_slots):
            duration = int(slot_duration_exec[batch_idx, slot_idx].item())
            if duration <= 0:
                continue
            unit_idx = int(slot_unit_index[batch_idx, slot_idx].item())
            is_blank = int(slot_is_blank[batch_idx, slot_idx].item()) > 0
            if duration <= 1:
                phase = torch.zeros((duration,), dtype=torch.float32, device=device)
            else:
                phase = torch.linspace(0.0, 1.0, duration, dtype=torch.float32, device=device)
            edge = torch.abs(2.0 * phase - 1.0)
            phase_feat = torch.stack(
                [
                    phase,
                    1.0 - phase,
                    2.0 * phase - 1.0,
                    torch.full((duration,), float(math.log1p(float(duration))), dtype=torch.float32, device=device),
                    edge,
                ],
                dim=-1,
            )
            if is_blank:
                src_indices = torch.full((duration,), -1, dtype=torch.long, device=device)
            else:
                src_len = int(src_anchor[batch_idx, unit_idx].item()) if unit_idx < src_anchor.size(1) else 0
                src_start = src_cursor
                src_end = min(src_cursor + max(src_len, 0), src_total)
                src_cursor = src_end
                if src_total <= 0:
                    src_indices = torch.full((duration,), -1, dtype=torch.long, device=device)
                elif src_end <= src_start:
                    fallback = min(max(src_start - 1, 0), max(src_total - 1, 0))
                    src_indices = torch.full((duration,), int(fallback), dtype=torch.long, device=device)
                elif duration == 1:
                    src_indices = torch.tensor([src_start], dtype=torch.long, device=device)
                else:
                    src_indices = torch.linspace(
                        float(src_start),
                        float(max(src_end - 1, src_start)),
                        duration,
                        dtype=torch.float32,
                        device=device,
                    ).round().long().clamp(min=src_start, max=max(src_end - 1, src_start))
            frame_src_index.append(src_indices)
            frame_blank.append(torch.full((duration,), 1 if is_blank else 0, dtype=torch.float32, device=device))
            frame_slot_index.append(torch.full((duration,), int(slot_idx), dtype=torch.long, device=device))
            frame_unit_index.append(torch.full((duration,), int(unit_idx), dtype=torch.long, device=device))
            frame_phase_features.append(phase_feat)
            valid_len += duration
        if len(frame_src_index) <= 0:
            frame_src_index = [torch.full((1,), -1, dtype=torch.long, device=device)]
            frame_blank = [torch.ones((1,), dtype=torch.float32, device=device)]
            frame_slot_index = [torch.zeros((1,), dtype=torch.long, device=device)]
            frame_unit_index = [torch.zeros((1,), dtype=torch.long, device=device)]
            frame_phase_features = [torch.zeros((1, 5), dtype=torch.float32, device=device)]
        frame_src_index_list.append(torch.cat(frame_src_index, dim=0))
        frame_blank_list.append(torch.cat(frame_blank, dim=0))
        frame_slot_index_list.append(torch.cat(frame_slot_index, dim=0))
        frame_unit_index_list.append(torch.cat(frame_unit_index, dim=0))
        frame_phase_feature_list.append(torch.cat(frame_phase_features, dim=0))
        valid_lengths.append(valid_len)

    frame_src_index = _pad_sequences(frame_src_index_list, pad_value=-1, dtype=torch.long)
    blank_mask = _pad_sequences(frame_blank_list, pad_value=0.0)
    frame_slot_index = _pad_sequences(frame_slot_index_list, pad_value=-1, dtype=torch.long)
    frame_unit_index = _pad_sequences(frame_unit_index_list, pad_value=-1, dtype=torch.long)
    frame_phase_features = _pad_sequences(frame_phase_feature_list, pad_value=0.0)
    total_mask = torch.zeros_like(blank_mask)
    for batch_idx, valid_len in enumerate(valid_lengths):
        if valid_len > 0:
            total_mask[batch_idx, :valid_len] = 1.0
    blank_mask = blank_mask * total_mask
    speech_mask = (1.0 - blank_mask).clamp(0.0, 1.0) * total_mask
    return RhythmFramePlan(
        frame_src_index=frame_src_index,
        frame_is_blank=blank_mask.long(),
        frame_slot_index=frame_slot_index,
        frame_unit_index=frame_unit_index,
        total_mask=total_mask,
        speech_mask=speech_mask,
        blank_mask=blank_mask,
        frame_phase_features=frame_phase_features,
    )


def build_frame_plan_from_execution(
    *,
    dur_anchor_src: torch.Tensor,
    speech_exec: torch.Tensor,
    pause_exec: torch.Tensor,
    unit_mask: torch.Tensor,
) -> RhythmFramePlan:
    slot_schedule = build_interleaved_blank_slot_schedule(
        speech_duration_exec=speech_exec,
        blank_duration_exec=pause_exec,
        unit_mask=unit_mask,
    )
    return build_frame_plan(
        dur_anchor_src=dur_anchor_src,
        slot_duration_exec=slot_schedule.slot_duration_exec,
        slot_mask=slot_schedule.slot_mask,
        slot_is_blank=slot_schedule.slot_is_blank,
        slot_unit_index=slot_schedule.slot_unit_index,
    )


def sample_tensor_by_frame_plan(
    source: torch.Tensor,
    frame_plan: RhythmFramePlan,
    *,
    blank_fill: torch.Tensor | float | int | None = None,
) -> torch.Tensor:
    if source.dim() == 1:
        source = source.unsqueeze(0)
    if source.dim() not in {2, 3}:
        raise ValueError(f"Unsupported source shape for frame-plan sampling: {tuple(source.shape)}")
    if source.size(0) != frame_plan.frame_src_index.size(0):
        raise ValueError(
            f"Batch mismatch between source {tuple(source.shape)} and frame plan "
            f"{tuple(frame_plan.frame_src_index.shape)}."
        )
    outputs = []
    for batch_idx in range(source.size(0)):
        src = source[batch_idx]
        indices = frame_plan.frame_src_index[batch_idx]
        if src.size(0) <= 0:
            if src.dim() == 1:
                gathered = src.new_zeros(indices.shape)
            else:
                gathered = src.new_zeros((indices.size(0), *src.shape[1:]))
        else:
            safe_indices = indices.clamp(min=0, max=max(int(src.size(0)) - 1, 0))
            if src.dim() == 1:
                gathered = src[safe_indices]
            else:
                gathered = src.index_select(0, safe_indices)
        blank_mask = frame_plan.blank_mask[batch_idx] > 0.5
        if blank_mask.any():
            if blank_fill is None:
                fill = src.new_zeros(src.shape[1:]) if src.dim() > 1 else src.new_tensor(0.0)
            elif isinstance(blank_fill, torch.Tensor):
                fill = blank_fill[batch_idx] if blank_fill.dim() > src.dim() - 1 else blank_fill
                fill = fill.to(device=src.device, dtype=src.dtype)
            else:
                fill = src.new_tensor(blank_fill)
            if src.dim() == 1:
                gathered = gathered.clone()
                gathered[blank_mask] = fill
            else:
                gathered = gathered.clone()
                gathered[blank_mask] = fill.view(1, *src.shape[1:])
        if src.dim() == 1:
            gathered = gathered * frame_plan.total_mask[batch_idx]
        else:
            gathered = gathered * frame_plan.total_mask[batch_idx].unsqueeze(-1)
        outputs.append(gathered)
    return _pad_sequences(outputs, pad_value=0.0, dtype=source.dtype)


def build_frame_weight_from_plan(
    frame_plan: RhythmFramePlan,
    *,
    pause_frame_weight: float = 0.20,
    stretch_weight_min: float = 0.35,
) -> torch.Tensor:
    weights = frame_plan.total_mask.clone()
    for batch_idx in range(frame_plan.frame_src_index.size(0)):
        src_indices = frame_plan.frame_src_index[batch_idx]
        total_mask = frame_plan.total_mask[batch_idx] > 0.5
        blank_mask = frame_plan.blank_mask[batch_idx] > 0.5
        speech_mask = total_mask & (~blank_mask)
        if speech_mask.any():
            visible_src = src_indices[speech_mask].clamp_min(0)
            if visible_src.numel() > 0:
                counts = torch.bincount(visible_src, minlength=int(visible_src.max().item()) + 1).float().clamp_min(1.0)
                speech_weights = (1.0 / counts[visible_src]).clamp_min(float(stretch_weight_min))
                weights[batch_idx, speech_mask] = speech_weights
        weights[batch_idx, blank_mask] = float(pause_frame_weight)
        weights[batch_idx] = weights[batch_idx] * frame_plan.total_mask[batch_idx]
    return weights
