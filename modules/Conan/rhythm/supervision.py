from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

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
    **kwargs,
) -> dict[str, np.ndarray]:
    targets = build_reference_guided_targets(
        dur_anchor_src=dur_anchor_src,
        ref_rhythm_stats=ref_rhythm_stats,
        ref_rhythm_trace=ref_rhythm_trace,
        unit_mask=unit_mask,
        **kwargs,
    )
    return {
        "rhythm_teacher_speech_exec_tgt": targets["rhythm_speech_exec_tgt"],
        "rhythm_teacher_pause_exec_tgt": targets["rhythm_pause_exec_tgt"],
        "rhythm_teacher_speech_budget_tgt": targets["rhythm_speech_budget_tgt"],
        "rhythm_teacher_pause_budget_tgt": targets["rhythm_pause_budget_tgt"],
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
        bundle.update(
            build_reference_teacher_targets(
                dur_anchor_src=source["dur_anchor_src"],
                unit_mask=(np.asarray(source["dur_anchor_src"]) > 0).astype(np.float32),
                ref_rhythm_stats=conditioning["ref_rhythm_stats"],
                ref_rhythm_trace=conditioning["ref_rhythm_trace"],
            )
        )
    return bundle
