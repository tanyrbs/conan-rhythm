from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

from .reference_encoder import ReferenceRhythmEncoder, sample_progress_trace
from .unit_frontend import RhythmUnitFrontend


def _as_token_list(content_tokens) -> list[int]:
    if isinstance(content_tokens, str):
        return [int(float(x)) for x in content_tokens.split() if str(x).strip() != ""]
    if isinstance(content_tokens, np.ndarray):
        return [int(x) for x in content_tokens.tolist()]
    if torch.is_tensor(content_tokens):
        return [int(x) for x in content_tokens.detach().cpu().tolist()]
    return [int(x) for x in content_tokens]


def _as_mel_tensor(mel) -> torch.Tensor:
    if torch.is_tensor(mel):
        mel_tensor = mel.detach().float().cpu()
    else:
        mel_tensor = torch.tensor(np.asarray(mel), dtype=torch.float32)
    if mel_tensor.dim() == 2:
        mel_tensor = mel_tensor.unsqueeze(0)
    return mel_tensor


@lru_cache(maxsize=8)
def _cached_frontend(
    silent_token: int | None,
    separator_aware: bool,
    tail_open_units: int,
) -> RhythmUnitFrontend:
    return RhythmUnitFrontend(
        silent_token=silent_token,
        separator_aware=separator_aware,
        tail_open_units=tail_open_units,
    )


@lru_cache(maxsize=8)
def _cached_reference_encoder(trace_bins: int) -> ReferenceRhythmEncoder:
    return ReferenceRhythmEncoder(trace_bins=trace_bins)


def build_source_rhythm_cache(
    content_tokens,
    *,
    silent_token: int | None = None,
    separator_aware: bool = True,
    tail_open_units: int = 1,
) -> dict[str, np.ndarray]:
    frontend = _cached_frontend(
        silent_token=silent_token,
        separator_aware=separator_aware,
        tail_open_units=tail_open_units,
    )
    batch = frontend.from_token_lists(
        [_as_token_list(content_tokens)],
        mark_last_open=False,
    )
    return {
        "content_units": batch.content_units[0].cpu().numpy().astype(np.int64),
        "dur_anchor_src": batch.dur_anchor_src[0].cpu().numpy().astype(np.int64),
        "open_run_mask": batch.open_run_mask[0].cpu().numpy().astype(np.int64),
        "sep_hint": batch.sep_hint[0].cpu().numpy().astype(np.int64),
    }


def build_reference_rhythm_conditioning(
    ref_mel,
    *,
    trace_bins: int = 24,
) -> dict[str, np.ndarray]:
    encoder = _cached_reference_encoder(trace_bins=trace_bins)
    conditioning = encoder(_as_mel_tensor(ref_mel))
    return {
        "ref_rhythm_stats": conditioning["ref_rhythm_stats"][0].cpu().numpy().astype(np.float32),
        "ref_rhythm_trace": conditioning["ref_rhythm_trace"][0].cpu().numpy().astype(np.float32),
    }


def _masked_standardize(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    total = mask.sum().clamp_min(1.0)
    mean = (x * mask).sum() / total
    var = (((x - mean) ** 2) * mask).sum() / total
    return ((x - mean) / var.clamp_min(1e-6).sqrt()) * mask


def _masked_normalize(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    scores = scores.clamp_min(0.0) * mask
    return scores / scores.sum().clamp_min(1e-6)


def _smooth_1d(values: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = max(1, int(kernel_size))
    if kernel_size <= 1 or values.numel() <= 1:
        return values
    padded = F.avg_pool1d(
        values.view(1, 1, -1),
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    ).view(-1)
    if padded.size(0) > values.size(0):
        padded = padded[: values.size(0)]
    return padded


def _sparsify_scores(scores: torch.Tensor, mask: torch.Tensor, *, topk_ratio: float) -> torch.Tensor:
    mask = mask.float()
    valid = int(mask.sum().item())
    if valid <= 0:
        return scores.new_zeros(scores.shape)
    keep_k = max(1, min(valid, int(round(valid * float(max(0.0, min(1.0, topk_ratio)))))))
    sparse = scores.new_zeros(scores.shape)
    values, indices = torch.topk(scores[:valid], k=keep_k, dim=0)
    sparse[indices] = values
    return sparse * mask


def _resample_segment(segment: torch.Tensor, target_len: int) -> torch.Tensor:
    if target_len <= 0:
        return segment.new_zeros((0, segment.size(-1)))
    if segment.size(0) <= 0:
        return segment.new_zeros((target_len, segment.size(-1)))
    if segment.size(0) == target_len:
        return segment
    if segment.size(0) == 1:
        return segment.expand(target_len, -1)
    resized = F.interpolate(
        segment.transpose(0, 1).unsqueeze(0),
        size=target_len,
        mode="linear",
        align_corners=False,
    )
    return resized.squeeze(0).transpose(0, 1)


def _infer_silence_frame(mel: torch.Tensor) -> torch.Tensor:
    if mel.size(0) <= 0:
        return mel.new_zeros((mel.size(-1),))
    energy = mel.mean(dim=-1)
    min_idx = int(torch.argmin(energy).item())
    return mel[min_idx]


def build_retimed_mel_target(
    *,
    mel,
    dur_anchor_src,
    speech_exec_tgt,
    pause_exec_tgt,
    unit_mask=None,
    pause_frame_weight: float = 0.20,
    stretch_weight_min: float = 0.35,
) -> dict[str, np.ndarray]:
    mel = _as_mel_tensor(mel)[0]
    dur_anchor_src = torch.tensor(np.asarray(dur_anchor_src), dtype=torch.float32)
    speech_exec_tgt = torch.tensor(np.asarray(speech_exec_tgt), dtype=torch.float32)
    pause_exec_tgt = torch.tensor(np.asarray(pause_exec_tgt), dtype=torch.float32)
    if unit_mask is None:
        unit_mask = dur_anchor_src.gt(0).float()
    else:
        unit_mask = torch.tensor(np.asarray(unit_mask), dtype=torch.float32)

    visible = int(unit_mask.sum().item())
    if visible <= 0:
        empty = mel.new_zeros((0, mel.size(-1)))
        empty_weight = mel.new_zeros((0,))
        return {
            "rhythm_retimed_mel_tgt": empty.cpu().numpy().astype(np.float32),
            "rhythm_retimed_mel_len": np.asarray([0], dtype=np.int64),
            "rhythm_retimed_frame_weight": empty_weight.cpu().numpy().astype(np.float32),
        }

    src_dur = dur_anchor_src[:visible].clamp_min(0.0)
    speech_exec = speech_exec_tgt[:visible].clamp_min(0.0)
    pause_exec = pause_exec_tgt[:visible].clamp_min(0.0)

    src_int = torch.round(src_dur).long().clamp_min(0)
    speech_int = torch.round(speech_exec).long()
    pause_int = torch.round(pause_exec).long()

    positive_mask = unit_mask[:visible] > 0
    speech_int = torch.where(positive_mask, speech_int.clamp_min(1), speech_int.clamp_min(0))
    pause_int = pause_int.clamp_min(0)

    silence_frame = _infer_silence_frame(mel)
    pieces = []
    weight_pieces = []
    cursor = 0
    total_frames = int(mel.size(0))
    for unit_idx in range(visible):
        src_len = int(src_int[unit_idx].item())
        tgt_speech_len = int(speech_int[unit_idx].item())
        tgt_pause_len = int(pause_int[unit_idx].item())

        end = min(cursor + max(src_len, 0), total_frames)
        segment = mel[cursor:end]
        cursor = end
        if segment.size(0) <= 0:
            if total_frames > 0:
                fallback_idx = min(max(cursor - 1, 0), total_frames - 1)
                segment = mel[fallback_idx:fallback_idx + 1]
            else:
                segment = mel.new_zeros((1, mel.size(-1)))
        pieces.append(_resample_segment(segment, tgt_speech_len))
        if tgt_speech_len > 0:
            base_src_len = max(src_len, 1)
            ratio = min(base_src_len, tgt_speech_len) / max(base_src_len, tgt_speech_len)
            speech_weight = max(float(stretch_weight_min), float(ratio))
            weight_pieces.append(mel.new_full((tgt_speech_len,), speech_weight))
        if tgt_pause_len > 0:
            pieces.append(silence_frame.unsqueeze(0).expand(tgt_pause_len, -1))
            weight_pieces.append(mel.new_full((tgt_pause_len,), float(pause_frame_weight)))

    if len(pieces) <= 0:
        retimed = mel.new_zeros((0, mel.size(-1)))
        frame_weight = mel.new_zeros((0,))
    else:
        retimed = torch.cat(pieces, dim=0)
        frame_weight = torch.cat(weight_pieces, dim=0) if len(weight_pieces) > 0 else mel.new_zeros((retimed.size(0),))
    return {
        "rhythm_retimed_mel_tgt": retimed.cpu().numpy().astype(np.float32),
        "rhythm_retimed_mel_len": np.asarray([int(retimed.size(0))], dtype=np.int64),
        "rhythm_retimed_frame_weight": frame_weight.cpu().numpy().astype(np.float32),
    }


def build_reference_guided_targets(
    *,
    dur_anchor_src,
    ref_rhythm_stats,
    ref_rhythm_trace,
    unit_mask=None,
    rate_scale_min: float = 0.60,
    rate_scale_max: float = 1.80,
    local_rate_strength: float = 0.35,
    segment_bias_strength: float = 0.25,
    pause_strength: float = 1.00,
    boundary_strength: float = 1.25,
    pause_budget_ratio_cap: float = 0.75,
) -> dict[str, np.ndarray]:
    dur_anchor_src = torch.tensor(np.asarray(dur_anchor_src), dtype=torch.float32)
    ref_rhythm_stats = torch.tensor(np.asarray(ref_rhythm_stats), dtype=torch.float32)
    ref_rhythm_trace = torch.tensor(np.asarray(ref_rhythm_trace), dtype=torch.float32)
    if unit_mask is None:
        unit_mask = dur_anchor_src.gt(0).float()
    else:
        unit_mask = torch.tensor(np.asarray(unit_mask), dtype=torch.float32)

    unit_count = int(unit_mask.sum().item())
    if unit_count <= 0:
        zero_units = dur_anchor_src.new_zeros(dur_anchor_src.shape)
        zero_budget = dur_anchor_src.new_zeros((1,))
        return {
            "rhythm_speech_exec_tgt": zero_units.cpu().numpy().astype(np.float32),
            "rhythm_pause_exec_tgt": zero_units.cpu().numpy().astype(np.float32),
            "rhythm_speech_budget_tgt": zero_budget.cpu().numpy().astype(np.float32),
            "rhythm_pause_budget_tgt": zero_budget.cpu().numpy().astype(np.float32),
            "rhythm_guidance_speech_tgt": zero_units.cpu().numpy().astype(np.float32),
            "rhythm_guidance_pause_tgt": zero_units.cpu().numpy().astype(np.float32),
        }

    trace_context = sample_progress_trace(
        ref_rhythm_trace.unsqueeze(0),
        phase_ptr=torch.zeros(1, dtype=torch.float32),
        window_size=int(dur_anchor_src.size(0)),
        visible_sizes=torch.tensor([unit_count], dtype=torch.long),
    )[0]
    trace_context = trace_context * unit_mask.unsqueeze(-1)

    src_total = (dur_anchor_src * unit_mask).sum().clamp_min(1.0)
    src_mean = src_total / unit_mask.sum().clamp_min(1.0)
    ref_mean_speech = ref_rhythm_stats[2].clamp_min(1.0)
    rate_scale = (ref_mean_speech / src_mean.clamp_min(1.0)).clamp(rate_scale_min, rate_scale_max)
    speech_budget = src_total * rate_scale

    pause_ratio = ref_rhythm_stats[0].clamp(0.0, 0.49)
    boundary_ratio = ref_rhythm_stats[4].clamp(0.0, 1.0)
    mean_pause = ref_rhythm_stats[1].clamp_min(0.0)
    pause_from_ratio = speech_budget * pause_ratio / (1.0 - pause_ratio).clamp_min(0.20)
    pause_from_events = unit_mask.sum().clamp_min(1.0) * boundary_ratio * mean_pause
    pause_budget = 0.5 * (pause_from_ratio + pause_from_events)
    pause_budget = pause_budget.clamp(min=0.0, max=speech_budget * pause_budget_ratio_cap)

    local_rate = _masked_standardize(trace_context[:, 1], unit_mask)
    segment_bias = _masked_standardize(trace_context[:, 3], unit_mask)
    speech_logits = torch.log1p(dur_anchor_src.clamp_min(0.0))
    speech_logits = speech_logits + local_rate_strength * local_rate + segment_bias_strength * segment_bias
    speech_scores = torch.exp(speech_logits) * unit_mask
    speech_scores = speech_scores / speech_scores.sum().clamp_min(1e-6)
    speech_exec = speech_scores * speech_budget

    pause_seed = pause_strength * _masked_standardize(trace_context[:, 0], unit_mask)
    pause_seed = pause_seed + boundary_strength * _masked_standardize(trace_context[:, 2], unit_mask)
    pause_scores = torch.exp(pause_seed) * unit_mask
    pause_scores = pause_scores / pause_scores.sum().clamp_min(1e-6)
    pause_exec = pause_scores * pause_budget

    return {
        "rhythm_speech_exec_tgt": speech_exec.cpu().numpy().astype(np.float32),
        "rhythm_pause_exec_tgt": pause_exec.cpu().numpy().astype(np.float32),
        "rhythm_speech_budget_tgt": speech_budget.view(1).cpu().numpy().astype(np.float32),
        "rhythm_pause_budget_tgt": pause_budget.view(1).cpu().numpy().astype(np.float32),
        "rhythm_guidance_speech_tgt": speech_exec.cpu().numpy().astype(np.float32),
        "rhythm_guidance_pause_tgt": pause_exec.cpu().numpy().astype(np.float32),
    }


def build_reference_teacher_targets(
    *,
    dur_anchor_src,
    ref_rhythm_stats,
    ref_rhythm_trace,
    unit_mask=None,
    rate_scale_min: float = 0.55,
    rate_scale_max: float = 1.95,
    local_rate_strength: float = 0.45,
    segment_bias_strength: float = 0.30,
    pause_strength: float = 1.10,
    boundary_strength: float = 1.50,
    pause_budget_ratio_cap: float = 0.80,
    speech_smooth_kernel: int = 3,
    pause_topk_ratio: float = 0.30,
    **kwargs,
) -> dict[str, np.ndarray]:
    del kwargs
    dur_anchor_src = torch.tensor(np.asarray(dur_anchor_src), dtype=torch.float32)
    ref_rhythm_stats = torch.tensor(np.asarray(ref_rhythm_stats), dtype=torch.float32)
    ref_rhythm_trace = torch.tensor(np.asarray(ref_rhythm_trace), dtype=torch.float32)
    if unit_mask is None:
        unit_mask = dur_anchor_src.gt(0).float()
    else:
        unit_mask = torch.tensor(np.asarray(unit_mask), dtype=torch.float32)

    unit_count = int(unit_mask.sum().item())
    if unit_count <= 0:
        zero_units = dur_anchor_src.new_zeros(dur_anchor_src.shape)
        zero_budget = dur_anchor_src.new_zeros((1,))
        return {
            "rhythm_teacher_speech_exec_tgt": zero_units.cpu().numpy().astype(np.float32),
            "rhythm_teacher_pause_exec_tgt": zero_units.cpu().numpy().astype(np.float32),
            "rhythm_teacher_speech_budget_tgt": zero_budget.cpu().numpy().astype(np.float32),
            "rhythm_teacher_pause_budget_tgt": zero_budget.cpu().numpy().astype(np.float32),
        }

    trace_context = sample_progress_trace(
        ref_rhythm_trace.unsqueeze(0),
        phase_ptr=torch.zeros(1, dtype=torch.float32),
        window_size=int(dur_anchor_src.size(0)),
        visible_sizes=torch.tensor([unit_count], dtype=torch.long),
    )[0]
    trace_context = trace_context * unit_mask.unsqueeze(-1)

    src_total = (dur_anchor_src * unit_mask).sum().clamp_min(1.0)
    src_mean = src_total / unit_mask.sum().clamp_min(1.0)
    ref_mean_speech = ref_rhythm_stats[2].clamp_min(1.0)
    rate_scale = (ref_mean_speech / src_mean.clamp_min(1.0)).clamp(rate_scale_min, rate_scale_max)
    speech_budget = src_total * rate_scale

    pause_ratio = ref_rhythm_stats[0].clamp(0.0, 0.49)
    boundary_ratio = ref_rhythm_stats[4].clamp(0.0, 1.0)
    mean_pause = ref_rhythm_stats[1].clamp_min(0.0)
    pause_from_ratio = speech_budget * pause_ratio / (1.0 - pause_ratio).clamp_min(0.20)
    pause_from_events = unit_mask.sum().clamp_min(1.0) * boundary_ratio * mean_pause
    pause_budget = 0.35 * pause_from_ratio + 0.65 * pause_from_events
    pause_budget = pause_budget.clamp(min=0.0, max=speech_budget * pause_budget_ratio_cap)

    local_rate = _masked_standardize(trace_context[:, 1], unit_mask)
    segment_bias = _masked_standardize(trace_context[:, 3], unit_mask)
    speech_scores = torch.exp(
        torch.log1p(dur_anchor_src.clamp_min(0.0))
        + local_rate_strength * local_rate
        + segment_bias_strength * segment_bias
    ) * unit_mask
    speech_scores = _smooth_1d(speech_scores, speech_smooth_kernel) * unit_mask
    speech_scores = _masked_normalize(speech_scores, unit_mask)
    speech_exec = speech_scores * speech_budget

    pause_seed = pause_strength * _masked_standardize(trace_context[:, 0], unit_mask)
    pause_seed = pause_seed + boundary_strength * _masked_standardize(trace_context[:, 2], unit_mask)
    pause_scores = torch.exp(pause_seed) * unit_mask
    pause_scores = _sparsify_scores(
        pause_scores,
        unit_mask,
        topk_ratio=max(float(pause_topk_ratio), float(boundary_ratio.item())),
    )
    pause_scores = _masked_normalize(pause_scores, unit_mask)
    pause_exec = pause_scores * pause_budget

    return {
        "rhythm_teacher_speech_exec_tgt": speech_exec.cpu().numpy().astype(np.float32),
        "rhythm_teacher_pause_exec_tgt": pause_exec.cpu().numpy().astype(np.float32),
        "rhythm_teacher_speech_budget_tgt": speech_budget.view(1).cpu().numpy().astype(np.float32),
        "rhythm_teacher_pause_budget_tgt": pause_budget.view(1).cpu().numpy().astype(np.float32),
    }


def build_item_rhythm_bundle(
    *,
    content_tokens,
    mel,
    silent_token: int | None = None,
    separator_aware: bool = True,
    tail_open_units: int = 1,
    trace_bins: int = 24,
    include_self_targets: bool = True,
    include_teacher_targets: bool = False,
    include_retimed_mel_target: bool = False,
    retimed_mel_target_source: str = "guidance",
    retimed_pause_frame_weight: float = 0.20,
    retimed_stretch_weight_min: float = 0.35,
    teacher_kwargs: dict | None = None,
) -> dict[str, np.ndarray]:
    source = build_source_rhythm_cache(
        content_tokens,
        silent_token=silent_token,
        separator_aware=separator_aware,
        tail_open_units=tail_open_units,
    )
    conditioning = build_reference_rhythm_conditioning(mel, trace_bins=trace_bins)
    bundle = {**source, **conditioning}
    guided = None
    if include_self_targets or include_teacher_targets:
        guided = build_reference_guided_targets(
            dur_anchor_src=source["dur_anchor_src"],
            unit_mask=(np.asarray(source["dur_anchor_src"]) > 0).astype(np.float32),
            ref_rhythm_stats=conditioning["ref_rhythm_stats"],
            ref_rhythm_trace=conditioning["ref_rhythm_trace"],
        )
    if include_self_targets and guided is not None:
        bundle.update(guided)
    if include_teacher_targets:
        teacher_kwargs = dict(teacher_kwargs or {})
        bundle.update(
            build_reference_teacher_targets(
                dur_anchor_src=source["dur_anchor_src"],
                unit_mask=(np.asarray(source["dur_anchor_src"]) > 0).astype(np.float32),
                ref_rhythm_stats=conditioning["ref_rhythm_stats"],
                ref_rhythm_trace=conditioning["ref_rhythm_trace"],
                **teacher_kwargs,
            )
        )
    if include_retimed_mel_target:
        target_source = str(retimed_mel_target_source or "guidance").strip().lower()
        if target_source == "teacher" and "rhythm_teacher_speech_exec_tgt" in bundle:
            speech_key = "rhythm_teacher_speech_exec_tgt"
            pause_key = "rhythm_teacher_pause_exec_tgt"
        else:
            speech_key = "rhythm_speech_exec_tgt"
            pause_key = "rhythm_pause_exec_tgt"
        if speech_key in bundle and pause_key in bundle:
            bundle.update(
                build_retimed_mel_target(
                    mel=mel,
                    dur_anchor_src=source["dur_anchor_src"],
                    speech_exec_tgt=bundle[speech_key],
                    pause_exec_tgt=bundle[pause_key],
                    unit_mask=(np.asarray(source["dur_anchor_src"]) > 0).astype(np.float32),
                    pause_frame_weight=retimed_pause_frame_weight,
                    stretch_weight_min=retimed_stretch_weight_min,
                )
            )
    return bundle
