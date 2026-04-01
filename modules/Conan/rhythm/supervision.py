from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

from .reference_descriptor import RefRhythmDescriptor
from .reference_encoder import sample_progress_trace
from .source_boundary import build_source_boundary_cue
from .teacher import AlgorithmicTeacherConfig, build_algorithmic_teacher_targets
from .unit_frontend import RhythmUnitFrontend

RHYTHM_CACHE_VERSION = 4
RHYTHM_UNIT_HOP_MS = 20
RHYTHM_TRACE_HOP_MS = 80
RHYTHM_REFERENCE_MODE_STATIC_REF_FULL = 0
RHYTHM_GUIDANCE_SURFACE_NAME = "ref_guidance_v2"
RHYTHM_TEACHER_SURFACE_NAME = "offline_teacher_surface_v1"
RHYTHM_RETIMED_SOURCE_GUIDANCE = 0
RHYTHM_RETIMED_SOURCE_TEACHER = 1

_BLANK_ALIAS_KEYS = {
    "rhythm_pause_exec_tgt": "rhythm_blank_exec_tgt",
    "rhythm_pause_budget_tgt": "rhythm_blank_budget_tgt",
    "rhythm_guidance_pause_tgt": "rhythm_guidance_blank_tgt",
    "rhythm_teacher_pause_exec_tgt": "rhythm_teacher_blank_exec_tgt",
    "rhythm_teacher_pause_budget_tgt": "rhythm_teacher_blank_budget_tgt",
}


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


@lru_cache(maxsize=16)
def _cached_reference_descriptor(
    trace_bins: int,
    trace_horizon: float,
    smooth_kernel: int,
    slow_topk: int,
    selector_cell_size: int,
) -> RefRhythmDescriptor:
    return RefRhythmDescriptor(
        trace_bins=trace_bins,
        trace_horizon=trace_horizon,
        smooth_kernel=smooth_kernel,
        slow_topk=slow_topk,
        selector_cell_size=selector_cell_size,
    )


def build_source_rhythm_cache(
    content_tokens,
    *,
    silent_token: int | None = None,
    separator_aware: bool = True,
    tail_open_units: int = 1,
    phrase_boundary_threshold: float = 0.55,
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
    source_cache = {
        "content_units": batch.content_units[0].cpu().numpy().astype(np.int64),
        "dur_anchor_src": batch.dur_anchor_src[0].cpu().numpy().astype(np.int64),
        "open_run_mask": batch.open_run_mask[0].cpu().numpy().astype(np.int64),
        "sealed_mask": batch.sealed_mask[0].cpu().numpy().astype(np.int64),
        "sep_hint": batch.sep_hint[0].cpu().numpy().astype(np.int64),
        "boundary_confidence": batch.boundary_confidence[0].cpu().numpy().astype(np.float32),
    }
    source_cache.update(
        build_source_phrase_cache(
            dur_anchor_src=source_cache["dur_anchor_src"],
            sep_hint=source_cache["sep_hint"],
            open_run_mask=source_cache["open_run_mask"],
            sealed_mask=source_cache["sealed_mask"],
            boundary_confidence=source_cache["boundary_confidence"],
            phrase_boundary_threshold=phrase_boundary_threshold,
        )
    )
    return source_cache


def build_source_phrase_cache(
    *,
    dur_anchor_src,
    sep_hint,
    open_run_mask,
    sealed_mask,
    boundary_confidence,
    phrase_boundary_threshold: float = 0.55,
) -> dict[str, np.ndarray]:
    dur_anchor_src = torch.tensor(np.asarray(dur_anchor_src), dtype=torch.float32).unsqueeze(0)
    sep_hint = torch.tensor(np.asarray(sep_hint), dtype=torch.long).unsqueeze(0)
    open_run_mask = torch.tensor(np.asarray(open_run_mask), dtype=torch.long).unsqueeze(0)
    sealed_mask = torch.tensor(np.asarray(sealed_mask), dtype=torch.float32).unsqueeze(0)
    boundary_confidence = torch.tensor(np.asarray(boundary_confidence), dtype=torch.float32).unsqueeze(0)
    unit_mask = dur_anchor_src.gt(0).float()
    source_boundary_cue = build_source_boundary_cue(
        dur_anchor_src=dur_anchor_src,
        unit_mask=unit_mask,
        sep_hint=sep_hint,
        open_run_mask=open_run_mask,
        sealed_mask=sealed_mask,
        boundary_confidence=boundary_confidence,
    )[0]
    visible = int(unit_mask[0].sum().item())
    phrase_group_index = torch.zeros_like(source_boundary_cue, dtype=torch.long)
    phrase_group_pos = torch.zeros_like(source_boundary_cue)
    phrase_final_mask = torch.zeros_like(source_boundary_cue)
    if visible > 0:
        break_mask = (source_boundary_cue[:visible] >= float(phrase_boundary_threshold)).float()
        if sep_hint.size(1) >= visible:
            break_mask = torch.maximum(break_mask, sep_hint[0, :visible].float())
        phrase_starts = [0]
        for idx in range(max(visible - 1, 0)):
            if float(break_mask[idx].item()) > 0:
                phrase_starts.append(idx + 1)
                phrase_final_mask[idx] = 1.0
        phrase_final_mask[visible - 1] = 1.0
        phrase_starts = sorted(set(int(x) for x in phrase_starts if 0 <= int(x) < visible))
        for group_id, start in enumerate(phrase_starts):
            end = phrase_starts[group_id + 1] if group_id + 1 < len(phrase_starts) else visible
            length = max(1, end - start)
            phrase_group_index[start:end] = group_id
            if length == 1:
                phrase_group_pos[start] = 1.0
            else:
                phrase_group_pos[start:end] = torch.linspace(0.0, 1.0, steps=length)
    return {
        "source_boundary_cue": source_boundary_cue.cpu().numpy().astype(np.float32),
        "phrase_group_index": phrase_group_index.cpu().numpy().astype(np.int64),
        "phrase_group_pos": phrase_group_pos.cpu().numpy().astype(np.float32),
        "phrase_final_mask": phrase_final_mask.cpu().numpy().astype(np.float32),
    }


def build_reference_rhythm_conditioning(
    ref_mel,
    *,
    trace_bins: int = 24,
    trace_horizon: float = 0.35,
    smooth_kernel: int = 5,
    slow_topk: int = 6,
    selector_cell_size: int = 3,
) -> dict[str, np.ndarray]:
    descriptor = _cached_reference_descriptor(
        trace_bins=trace_bins,
        trace_horizon=float(trace_horizon),
        smooth_kernel=int(smooth_kernel),
        slow_topk=int(slow_topk),
        selector_cell_size=int(selector_cell_size),
    )
    conditioning = descriptor(_as_mel_tensor(ref_mel))
    out = {}
    for key, value in conditioning.items():
        array = value[0].detach().cpu().numpy()
        if key in {
            "selector_meta_indices",
            "selector_meta_starts",
            "selector_meta_ends",
        }:
            out[key] = array.astype(np.int64)
        else:
            out[key] = array.astype(np.float32)
    return out


def _masked_standardize(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    total = mask.sum().clamp_min(1.0)
    mean = (x * mask).sum() / total
    var = (((x - mean) ** 2) * mask).sum() / total
    return ((x - mean) / var.clamp_min(1e-6).sqrt()) * mask


def _estimate_surface_confidence(
    *,
    trace_context: torch.Tensor,
    ref_rhythm_stats: torch.Tensor,
    unit_mask: torch.Tensor,
    smoother_bonus: float = 0.0,
) -> float:
    unit_mask = unit_mask.float()
    visible = float(unit_mask.sum().item())
    if visible <= 0:
        return 0.0
    unit_support = min(1.0, visible / 6.0)
    local_rate_dyn = float(trace_context[:, 1].float().std().item()) if trace_context.size(0) > 1 else 0.0
    boundary_support = float(ref_rhythm_stats[4].clamp(0.0, 1.0).item())
    pause_support = float((ref_rhythm_stats[0].clamp(0.0, 0.49) / 0.49).item())
    trace_boundary_dyn = float(trace_context[:, 2].float().mean().clamp(0.0, 1.0).item())
    dynamics = float(np.tanh(local_rate_dyn + trace_boundary_dyn))
    confidence = 0.20 + 0.80 * (
        0.35 * unit_support
        + 0.35 * dynamics
        + 0.15 * boundary_support
        + 0.15 * pause_support
    )
    confidence = min(1.0, max(0.05, confidence + float(smoother_bonus)))
    return float(confidence)


def _build_cache_metadata(
    *,
    trace_bins: int,
    trace_horizon: float,
    slow_topk: int,
    selector_cell_size: int,
    source_phrase_threshold: float,
    target_confidence: float,
    guidance_confidence: float,
    retimed_target_source: str | None = None,
    retimed_target_confidence: float | None = None,
    teacher_confidence: float | None = None,
) -> dict[str, np.ndarray]:
    meta = {
        "rhythm_cache_version": np.asarray([RHYTHM_CACHE_VERSION], dtype=np.int64),
        "rhythm_unit_hop_ms": np.asarray([RHYTHM_UNIT_HOP_MS], dtype=np.int64),
        "rhythm_trace_hop_ms": np.asarray([RHYTHM_TRACE_HOP_MS], dtype=np.int64),
        "rhythm_trace_bins": np.asarray([int(trace_bins)], dtype=np.int64),
        "rhythm_trace_horizon": np.asarray([float(trace_horizon)], dtype=np.float32),
        "rhythm_slow_topk": np.asarray([int(slow_topk)], dtype=np.int64),
        "rhythm_selector_cell_size": np.asarray([int(selector_cell_size)], dtype=np.int64),
        "rhythm_source_phrase_threshold": np.asarray([float(source_phrase_threshold)], dtype=np.float32),
        "rhythm_reference_mode_id": np.asarray([RHYTHM_REFERENCE_MODE_STATIC_REF_FULL], dtype=np.int64),
        "rhythm_target_confidence": np.asarray([float(target_confidence)], dtype=np.float32),
        "rhythm_guidance_confidence": np.asarray([float(guidance_confidence)], dtype=np.float32),
        "rhythm_guidance_surface_name": np.asarray([RHYTHM_GUIDANCE_SURFACE_NAME], dtype=np.str_),
    }
    if teacher_confidence is not None:
        meta["rhythm_teacher_confidence"] = np.asarray([float(teacher_confidence)], dtype=np.float32)
        meta["rhythm_teacher_surface_name"] = np.asarray([RHYTHM_TEACHER_SURFACE_NAME], dtype=np.str_)
    if retimed_target_source is not None:
        target_source = str(retimed_target_source).strip().lower()
        source_id = RHYTHM_RETIMED_SOURCE_TEACHER if target_source == "teacher" else RHYTHM_RETIMED_SOURCE_GUIDANCE
        source_name = RHYTHM_TEACHER_SURFACE_NAME if source_id == RHYTHM_RETIMED_SOURCE_TEACHER else RHYTHM_GUIDANCE_SURFACE_NAME
        meta["rhythm_retimed_target_source_id"] = np.asarray([source_id], dtype=np.int64)
        meta["rhythm_retimed_target_surface_name"] = np.asarray([source_name], dtype=np.str_)
        if retimed_target_confidence is not None:
            meta["rhythm_retimed_target_confidence"] = np.asarray([float(retimed_target_confidence)], dtype=np.float32)
    return meta


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


def with_blank_aliases(bundle: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = dict(bundle)
    for pause_key, blank_key in _BLANK_ALIAS_KEYS.items():
        if pause_key in out and blank_key not in out:
            out[blank_key] = np.asarray(out[pause_key]).copy()
        if blank_key in out and pause_key not in out:
            out[pause_key] = np.asarray(out[blank_key]).copy()
    return out


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
        return with_blank_aliases({
            "rhythm_speech_exec_tgt": zero_units.cpu().numpy().astype(np.float32),
            "rhythm_pause_exec_tgt": zero_units.cpu().numpy().astype(np.float32),
            "rhythm_speech_budget_tgt": zero_budget.cpu().numpy().astype(np.float32),
            "rhythm_pause_budget_tgt": zero_budget.cpu().numpy().astype(np.float32),
            "rhythm_guidance_speech_tgt": zero_units.cpu().numpy().astype(np.float32),
            "rhythm_guidance_pause_tgt": zero_units.cpu().numpy().astype(np.float32),
        })

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
    guidance_confidence = _estimate_surface_confidence(
        trace_context=trace_context,
        ref_rhythm_stats=ref_rhythm_stats,
        unit_mask=unit_mask,
        smoother_bonus=0.0,
    )

    return with_blank_aliases({
        "rhythm_speech_exec_tgt": speech_exec.cpu().numpy().astype(np.float32),
        "rhythm_pause_exec_tgt": pause_exec.cpu().numpy().astype(np.float32),
        "rhythm_speech_budget_tgt": speech_budget.view(1).cpu().numpy().astype(np.float32),
        "rhythm_pause_budget_tgt": pause_budget.view(1).cpu().numpy().astype(np.float32),
        "rhythm_guidance_speech_tgt": speech_exec.cpu().numpy().astype(np.float32),
        "rhythm_guidance_pause_tgt": pause_exec.cpu().numpy().astype(np.float32),
        "rhythm_target_confidence": np.asarray([guidance_confidence], dtype=np.float32),
        "rhythm_guidance_confidence": np.asarray([guidance_confidence], dtype=np.float32),
    })


def build_reference_teacher_targets(
    *,
    dur_anchor_src,
    ref_rhythm_stats,
    ref_rhythm_trace,
    unit_mask=None,
    source_boundary_cue=None,
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
    if source_boundary_cue is not None:
        source_boundary_cue = torch.tensor(np.asarray(source_boundary_cue), dtype=torch.float32)
    teacher = build_algorithmic_teacher_targets(
        dur_anchor_src=dur_anchor_src,
        ref_rhythm_stats=ref_rhythm_stats,
        ref_rhythm_trace=ref_rhythm_trace,
        unit_mask=unit_mask,
        source_boundary_cue=source_boundary_cue,
        config=AlgorithmicTeacherConfig(
            rate_scale_min=rate_scale_min,
            rate_scale_max=rate_scale_max,
            local_rate_strength=local_rate_strength,
            segment_bias_strength=segment_bias_strength,
            pause_strength=pause_strength,
            boundary_strength=boundary_strength,
            pause_budget_ratio_cap=pause_budget_ratio_cap,
            speech_smooth_kernel=speech_smooth_kernel,
            pause_topk_ratio=pause_topk_ratio,
        ),
    )

    return with_blank_aliases({
        "rhythm_teacher_speech_exec_tgt": teacher.speech_exec_tgt[0].cpu().numpy().astype(np.float32),
        "rhythm_teacher_pause_exec_tgt": teacher.pause_exec_tgt[0].cpu().numpy().astype(np.float32),
        "rhythm_teacher_speech_budget_tgt": teacher.speech_budget_tgt[0].cpu().numpy().astype(np.float32),
        "rhythm_teacher_pause_budget_tgt": teacher.pause_budget_tgt[0].cpu().numpy().astype(np.float32),
        "rhythm_teacher_allocation_tgt": teacher.allocation_tgt[0].cpu().numpy().astype(np.float32),
        "rhythm_teacher_prefix_clock_tgt": teacher.prefix_clock_tgt[0].cpu().numpy().astype(np.float32),
        "rhythm_teacher_prefix_backlog_tgt": teacher.prefix_backlog_tgt[0].cpu().numpy().astype(np.float32),
        "rhythm_teacher_confidence": teacher.confidence[0].cpu().numpy().astype(np.float32),
    })


def build_item_rhythm_bundle(
    *,
    content_tokens,
    mel,
    silent_token: int | None = None,
    separator_aware: bool = True,
    tail_open_units: int = 1,
    trace_bins: int = 24,
    trace_horizon: float = 0.35,
    trace_smooth_kernel: int = 5,
    slow_topk: int = 6,
    selector_cell_size: int = 3,
    source_phrase_threshold: float = 0.55,
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
        phrase_boundary_threshold=source_phrase_threshold,
    )
    conditioning = build_reference_rhythm_conditioning(
        mel,
        trace_bins=trace_bins,
        trace_horizon=trace_horizon,
        smooth_kernel=trace_smooth_kernel,
        slow_topk=slow_topk,
        selector_cell_size=selector_cell_size,
    )
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
        bundle.update(with_blank_aliases(guided))
    if include_teacher_targets:
        teacher_kwargs = dict(teacher_kwargs or {})
        bundle.update(
            build_reference_teacher_targets(
                dur_anchor_src=source["dur_anchor_src"],
                unit_mask=(np.asarray(source["dur_anchor_src"]) > 0).astype(np.float32),
                source_boundary_cue=source.get("source_boundary_cue", source.get("boundary_confidence")),
                ref_rhythm_stats=conditioning["ref_rhythm_stats"],
                ref_rhythm_trace=conditioning["ref_rhythm_trace"],
                **teacher_kwargs,
            )
        )
    retimed_target_confidence = None
    retimed_target_source = None
    if include_retimed_mel_target:
        target_source = str(retimed_mel_target_source or "guidance").strip().lower()
        if target_source == "teacher" and "rhythm_teacher_speech_exec_tgt" in bundle:
            speech_key = "rhythm_teacher_speech_exec_tgt"
            pause_key = "rhythm_teacher_pause_exec_tgt"
            retimed_target_source = "teacher"
            retimed_target_confidence = bundle.get("rhythm_teacher_confidence")
        else:
            speech_key = "rhythm_speech_exec_tgt"
            pause_key = "rhythm_pause_exec_tgt"
            retimed_target_source = "guidance"
            retimed_target_confidence = bundle.get("rhythm_guidance_confidence", bundle.get("rhythm_target_confidence"))
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
    target_confidence = float(np.asarray(bundle.get("rhythm_target_confidence", [1.0])).reshape(-1)[0])
    guidance_confidence = float(np.asarray(bundle.get("rhythm_guidance_confidence", [target_confidence])).reshape(-1)[0])
    teacher_confidence = bundle.get("rhythm_teacher_confidence")
    if retimed_target_confidence is not None:
        retimed_target_confidence = float(np.asarray(retimed_target_confidence).reshape(-1)[0])
    if teacher_confidence is not None:
        teacher_confidence = float(np.asarray(teacher_confidence).reshape(-1)[0])
    bundle.update(
        _build_cache_metadata(
            trace_bins=trace_bins,
            trace_horizon=trace_horizon,
            slow_topk=slow_topk,
            selector_cell_size=selector_cell_size,
            source_phrase_threshold=source_phrase_threshold,
            target_confidence=target_confidence,
            guidance_confidence=guidance_confidence,
            retimed_target_source=retimed_target_source,
            retimed_target_confidence=retimed_target_confidence,
            teacher_confidence=teacher_confidence,
        )
    )
    return with_blank_aliases(bundle)
