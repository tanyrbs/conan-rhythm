from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

from .reference_descriptor import RefRhythmDescriptor
from .reference_encoder import sample_progress_trace
from .retimed_targets import build_online_retimed_bundle, build_retimed_mel_target
from .source_boundary import build_source_boundary_cue
from .surface_metadata import (
    RHYTHM_CACHE_VERSION,
    RHYTHM_GUIDANCE_SURFACE_NAME,
    RHYTHM_REFERENCE_MODE_STATIC_REF_FULL,
    RHYTHM_RETIMED_SOURCE_GUIDANCE,
    RHYTHM_RETIMED_SOURCE_TEACHER,
    RHYTHM_TEACHER_SURFACE_ALGORITHMIC_NAME,
    RHYTHM_TEACHER_SURFACE_LEARNED_OFFLINE_NAME,
    RHYTHM_TEACHER_SURFACE_NAME,
    RHYTHM_TEACHER_TARGET_SOURCE_ALGORITHMIC,
    RHYTHM_TEACHER_TARGET_SOURCE_LEARNED_OFFLINE,
    RHYTHM_TRACE_HOP_MS,
    RHYTHM_UNIT_HOP_MS,
    build_cache_metadata,
    compatible_rhythm_cache_versions,
    infer_retimed_target_source_id,
    infer_teacher_target_source_id_from_surface_name,
    is_rhythm_cache_version_compatible,
    materialize_rhythm_cache_compat_fields,
    normalize_teacher_target_source,
    resolve_retimed_target_surface_name,
    resolve_teacher_surface_name,
    resolve_teacher_target_source_from_id,
    resolve_teacher_target_source_id,
    with_blank_aliases,
)
from .teacher import AlgorithmicTeacherConfig, build_algorithmic_teacher_targets
from .teacher_cache import (
    build_learned_offline_teacher_bundle,
    build_learned_offline_teacher_export_bundle,
    complete_learned_teacher_bundle,
)
from modules.Conan.rhythm_v3.unit_frontend import RhythmUnitFrontend
from modules.Conan.rhythm_v3.source_cache import (
    build_source_phrase_cache as build_source_phrase_cache_v3,
    build_source_rhythm_cache_v3,
)

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
    emit_silence_runs: bool,
    debounce_min_run_frames: int,
) -> RhythmUnitFrontend:
    return RhythmUnitFrontend(
        silent_token=silent_token,
        separator_aware=separator_aware,
        tail_open_units=tail_open_units,
        emit_silence_runs=emit_silence_runs,
        debounce_min_run_frames=debounce_min_run_frames,
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
    emit_silence_runs: bool = False,
    debounce_min_run_frames: int = 1,
    phrase_boundary_threshold: float = 0.55,
) -> dict[str, np.ndarray]:
    return build_source_rhythm_cache_v3(
        content_tokens,
        silent_token=silent_token,
        separator_aware=separator_aware,
        tail_open_units=tail_open_units,
        emit_silence_runs=emit_silence_runs,
        debounce_min_run_frames=debounce_min_run_frames,
        phrase_boundary_threshold=phrase_boundary_threshold,
    )


def build_source_phrase_cache(
    *,
    dur_anchor_src,
    sep_hint,
    open_run_mask,
    sealed_mask,
    boundary_confidence,
    phrase_boundary_threshold: float = 0.55,
) -> dict[str, np.ndarray]:
    return build_source_phrase_cache_v3(
        dur_anchor_src=dur_anchor_src,
        sep_hint=sep_hint,
        open_run_mask=open_run_mask,
        sealed_mask=sealed_mask,
        boundary_confidence=boundary_confidence,
        phrase_boundary_threshold=phrase_boundary_threshold,
    )


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


def build_reference_guided_targets(
    *,
    dur_anchor_src,
    ref_rhythm_stats,
    ref_rhythm_trace,
    unit_mask=None,
    anchor_aware_trace_sampling: bool = False,
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
        anchor_durations=(
            (dur_anchor_src * unit_mask).unsqueeze(0)
            if bool(anchor_aware_trace_sampling)
            else None
        ),
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
    anchor_aware_trace_sampling: bool = False,
    rate_scale_min: float = 0.55,
    rate_scale_max: float = 1.95,
    local_rate_strength: float = 0.45,
    segment_bias_strength: float = 0.30,
    pause_strength: float = 1.10,
    boundary_strength: float = 1.50,
    pause_budget_ratio_cap: float = 0.80,
    speech_smooth_kernel: int = 3,
    pause_topk_ratio: float = 0.30,
) -> dict[str, np.ndarray]:
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
            anchor_aware_trace_sampling=anchor_aware_trace_sampling,
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
    emit_silence_runs: bool = False,
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
    teacher_target_source: str = "algorithmic",
    teacher_bundle_override: dict | None = None,
    teacher_kwargs: dict | None = None,
) -> dict[str, np.ndarray]:
    source = build_source_rhythm_cache(
        content_tokens,
        silent_token=silent_token,
        separator_aware=separator_aware,
        tail_open_units=tail_open_units,
        emit_silence_runs=emit_silence_runs,
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
    teacher_surface_name = None
    teacher_confidence = None
    target_source = str(retimed_mel_target_source or "guidance").strip().lower()
    need_teacher_surface = bool(include_teacher_targets or (include_retimed_mel_target and target_source == "teacher"))
    if include_self_targets or need_teacher_surface:
        guided = build_reference_guided_targets(
            dur_anchor_src=source["dur_anchor_src"],
            unit_mask=(np.asarray(source["dur_anchor_src"]) > 0).astype(np.float32),
            ref_rhythm_stats=conditioning["ref_rhythm_stats"],
            ref_rhythm_trace=conditioning["ref_rhythm_trace"],
        )
    if include_self_targets and guided is not None:
        bundle.update(with_blank_aliases(guided))
    if need_teacher_surface:
        teacher_source = normalize_teacher_target_source(teacher_target_source)
        teacher_surface_name = resolve_teacher_surface_name(teacher_source)
        if teacher_source == "algorithmic":
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
        else:
            if teacher_bundle_override is None:
                raise ValueError(
                    "teacher_target_source=learned_offline requires teacher_bundle_override "
                    "when building a cache bundle."
                )
            bundle.update(
                complete_learned_teacher_bundle(
                    teacher_bundle_override,
                    source_cache=source,
                    guidance_bundle=guided,
                )
            )
        teacher_confidence = bundle.get("rhythm_teacher_confidence")
    retimed_target_confidence = None
    retimed_target_source = None
    if include_retimed_mel_target:
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
    if retimed_target_confidence is not None:
        retimed_target_confidence = float(np.asarray(retimed_target_confidence).reshape(-1)[0])
    if teacher_confidence is not None:
        teacher_confidence = float(np.asarray(teacher_confidence).reshape(-1)[0])
    bundle.update(
        build_cache_metadata(
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
            teacher_target_source=teacher_target_source if need_teacher_surface else None,
            teacher_surface_name=teacher_surface_name,
        )
    )
    return with_blank_aliases(bundle)
