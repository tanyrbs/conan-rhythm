from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

from .frame_plan import build_frame_plan_from_execution, build_frame_weight_from_plan, sample_tensor_by_frame_plan
from .reference_descriptor import RefRhythmDescriptor
from .reference_encoder import sample_progress_trace
from .source_boundary import build_source_boundary_cue
from .teacher import AlgorithmicTeacherConfig, build_algorithmic_teacher_targets
from .unit_frontend import RhythmUnitFrontend

RHYTHM_CACHE_VERSION = 5
RHYTHM_UNIT_HOP_MS = 20
RHYTHM_TRACE_HOP_MS = 80
RHYTHM_REFERENCE_MODE_STATIC_REF_FULL = 0
RHYTHM_GUIDANCE_SURFACE_NAME = "ref_guidance_v2"
RHYTHM_TEACHER_SURFACE_ALGORITHMIC_NAME = "offline_teacher_surface_v1"
RHYTHM_TEACHER_SURFACE_LEARNED_OFFLINE_NAME = "offline_teacher_surface_learned_offline_v1"
RHYTHM_TEACHER_SURFACE_NAME = RHYTHM_TEACHER_SURFACE_ALGORITHMIC_NAME
RHYTHM_TEACHER_TARGET_SOURCE_ALGORITHMIC = 0
RHYTHM_TEACHER_TARGET_SOURCE_LEARNED_OFFLINE = 1
RHYTHM_RETIMED_SOURCE_GUIDANCE = 0
RHYTHM_RETIMED_SOURCE_TEACHER = 1
RHYTHM_CACHE_COMPATIBLE_VERSIONS = {
    RHYTHM_CACHE_VERSION: (4,),
}

_BLANK_ALIAS_KEYS = {
    "rhythm_pause_exec_tgt": "rhythm_blank_exec_tgt",
    "rhythm_pause_budget_tgt": "rhythm_blank_budget_tgt",
    "rhythm_guidance_pause_tgt": "rhythm_guidance_blank_tgt",
    "rhythm_teacher_pause_exec_tgt": "rhythm_teacher_blank_exec_tgt",
    "rhythm_teacher_pause_budget_tgt": "rhythm_teacher_blank_budget_tgt",
}


def normalize_teacher_target_source(value) -> str:
    source = str(value or "algorithmic").strip().lower()
    aliases = {
        "algo": "algorithmic",
        "heuristic": "algorithmic",
        "legacy": "algorithmic",
        "rule": "algorithmic",
        "rules": "algorithmic",
        "teacher": "learned_offline",
        "offline": "learned_offline",
        "offline_teacher": "learned_offline",
        "learned": "learned_offline",
        "learned-offline": "learned_offline",
        "cache": "learned_offline",
        "cached": "learned_offline",
        "cached_teacher": "learned_offline",
    }
    normalized = aliases.get(source, source)
    if normalized not in {"algorithmic", "learned_offline"}:
        raise ValueError(f"Unsupported rhythm_teacher_target_source: {value}")
    return normalized


def resolve_teacher_surface_name(teacher_target_source) -> str:
    normalized = normalize_teacher_target_source(teacher_target_source)
    if normalized == "learned_offline":
        return RHYTHM_TEACHER_SURFACE_LEARNED_OFFLINE_NAME
    return RHYTHM_TEACHER_SURFACE_ALGORITHMIC_NAME


def resolve_teacher_target_source_id(teacher_target_source) -> int:
    normalized = normalize_teacher_target_source(teacher_target_source)
    if normalized == "learned_offline":
        return RHYTHM_TEACHER_TARGET_SOURCE_LEARNED_OFFLINE
    return RHYTHM_TEACHER_TARGET_SOURCE_ALGORITHMIC


def resolve_teacher_target_source_from_id(source_id) -> str:
    value = int(source_id)
    if value == RHYTHM_TEACHER_TARGET_SOURCE_LEARNED_OFFLINE:
        return "learned_offline"
    if value == RHYTHM_TEACHER_TARGET_SOURCE_ALGORITHMIC:
        return "algorithmic"
    raise ValueError(f"Unsupported rhythm teacher target source id: {source_id}")


def infer_teacher_target_source_from_surface_name(surface_name) -> str | None:
    if surface_name in {None, ""}:
        return None
    surface = str(surface_name).strip()
    if surface == RHYTHM_TEACHER_SURFACE_LEARNED_OFFLINE_NAME:
        return "learned_offline"
    if surface == RHYTHM_TEACHER_SURFACE_ALGORITHMIC_NAME:
        return "algorithmic"
    return None


def infer_teacher_target_source_id_from_surface_name(surface_name) -> int | None:
    source = infer_teacher_target_source_from_surface_name(surface_name)
    if source is None:
        return None
    return resolve_teacher_target_source_id(source)


def resolve_retimed_target_surface_name(
    source_id,
    *,
    teacher_surface_name=None,
    teacher_target_source=None,
) -> str:
    source_value = int(source_id)
    if source_value == RHYTHM_RETIMED_SOURCE_GUIDANCE:
        return RHYTHM_GUIDANCE_SURFACE_NAME
    if source_value == RHYTHM_RETIMED_SOURCE_TEACHER:
        if teacher_surface_name not in {None, ""}:
            return str(teacher_surface_name)
        if teacher_target_source is None:
            teacher_target_source = "algorithmic"
        return resolve_teacher_surface_name(teacher_target_source)
    raise ValueError(f"Unsupported rhythm retimed target source id: {source_id}")


def infer_retimed_target_source_id(
    surface_name,
    *,
    teacher_surface_name=None,
) -> int | None:
    if surface_name in {None, ""}:
        return None
    surface = str(surface_name).strip()
    if surface == RHYTHM_GUIDANCE_SURFACE_NAME:
        return RHYTHM_RETIMED_SOURCE_GUIDANCE
    if surface in {
        RHYTHM_TEACHER_SURFACE_ALGORITHMIC_NAME,
        RHYTHM_TEACHER_SURFACE_LEARNED_OFFLINE_NAME,
    }:
        return RHYTHM_RETIMED_SOURCE_TEACHER
    if teacher_surface_name not in {None, ""} and surface == str(teacher_surface_name).strip():
        return RHYTHM_RETIMED_SOURCE_TEACHER
    return None


def compatible_rhythm_cache_versions(expected_version) -> tuple[int, ...]:
    expected = int(expected_version)
    compatible = [expected]
    for version in RHYTHM_CACHE_COMPATIBLE_VERSIONS.get(expected, ()):
        version = int(version)
        if version not in compatible:
            compatible.append(version)
    return tuple(compatible)


def is_rhythm_cache_version_compatible(found_version, expected_version) -> bool:
    found = int(found_version)
    return found in compatible_rhythm_cache_versions(expected_version)


def materialize_rhythm_cache_compat_fields(item: dict | None) -> dict | None:
    if item is None:
        return None
    adapted = dict(item)
    teacher_surface_name = None
    if "rhythm_teacher_surface_name" in adapted:
        teacher_surface_name = str(np.asarray(adapted["rhythm_teacher_surface_name"]).reshape(-1)[0])
    if "rhythm_teacher_target_source_id" in adapted:
        teacher_source_id = int(np.asarray(adapted["rhythm_teacher_target_source_id"]).reshape(-1)[0])
    else:
        teacher_source_id = infer_teacher_target_source_id_from_surface_name(teacher_surface_name)
        if teacher_source_id is not None:
            adapted["rhythm_teacher_target_source_id"] = np.asarray([teacher_source_id], dtype=np.int64)
    if "rhythm_teacher_surface_name" not in adapted and teacher_source_id is not None:
        adapted["rhythm_teacher_surface_name"] = np.asarray(
            [resolve_teacher_surface_name(resolve_teacher_target_source_from_id(teacher_source_id))],
            dtype=np.str_,
        )
        teacher_surface_name = str(np.asarray(adapted["rhythm_teacher_surface_name"]).reshape(-1)[0])
    if "rhythm_retimed_target_source_id" in adapted:
        retimed_source_id = int(np.asarray(adapted["rhythm_retimed_target_source_id"]).reshape(-1)[0])
    else:
        retimed_surface_name = None
        if "rhythm_retimed_target_surface_name" in adapted:
            retimed_surface_name = str(np.asarray(adapted["rhythm_retimed_target_surface_name"]).reshape(-1)[0])
        retimed_source_id = infer_retimed_target_source_id(
            retimed_surface_name,
            teacher_surface_name=teacher_surface_name,
        )
        if retimed_source_id is not None:
            adapted["rhythm_retimed_target_source_id"] = np.asarray([retimed_source_id], dtype=np.int64)
    if "rhythm_retimed_target_surface_name" not in adapted and retimed_source_id is not None:
        teacher_target_source = None
        if teacher_source_id is not None:
            teacher_target_source = resolve_teacher_target_source_from_id(teacher_source_id)
        adapted["rhythm_retimed_target_surface_name"] = np.asarray(
            [
                resolve_retimed_target_surface_name(
                    retimed_source_id,
                    teacher_surface_name=teacher_surface_name,
                    teacher_target_source=teacher_target_source,
                )
            ],
            dtype=np.str_,
        )
    return adapted


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
    teacher_target_source: str | None = None,
    teacher_surface_name: str | None = None,
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
    if teacher_target_source is not None or teacher_surface_name is not None or teacher_confidence is not None:
        teacher_source = normalize_teacher_target_source(teacher_target_source or "algorithmic")
        teacher_surface = str(teacher_surface_name or resolve_teacher_surface_name(teacher_source))
        meta["rhythm_teacher_target_source_id"] = np.asarray(
            [resolve_teacher_target_source_id(teacher_source)], dtype=np.int64
        )
        meta["rhythm_teacher_surface_name"] = np.asarray([teacher_surface], dtype=np.str_)
    if teacher_confidence is not None:
        meta["rhythm_teacher_confidence"] = np.asarray([float(teacher_confidence)], dtype=np.float32)
    if retimed_target_source is not None:
        target_source = str(retimed_target_source).strip().lower()
        source_id = RHYTHM_RETIMED_SOURCE_TEACHER if target_source == "teacher" else RHYTHM_RETIMED_SOURCE_GUIDANCE
        source_name = (
            str(teacher_surface_name or resolve_teacher_surface_name(teacher_target_source or "algorithmic"))
            if source_id == RHYTHM_RETIMED_SOURCE_TEACHER
            else RHYTHM_GUIDANCE_SURFACE_NAME
        )
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


def _sum_exec_budget(exec_value) -> np.ndarray:
    return np.asarray([float(np.asarray(exec_value, dtype=np.float32).sum())], dtype=np.float32)


def _build_prefix_targets_from_exec_numpy(
    speech_exec,
    pause_exec,
    dur_anchor_src,
) -> tuple[np.ndarray, np.ndarray]:
    speech_exec = np.asarray(speech_exec, dtype=np.float32).reshape(-1)
    pause_exec = np.asarray(pause_exec, dtype=np.float32).reshape(-1)
    dur_anchor_src = np.asarray(dur_anchor_src, dtype=np.float32).reshape(-1)
    visible = min(len(speech_exec), len(pause_exec), len(dur_anchor_src))
    speech_exec = speech_exec[:visible]
    pause_exec = pause_exec[:visible]
    dur_anchor_src = dur_anchor_src[:visible]
    unit_mask = (dur_anchor_src > 0).astype(np.float32)
    prefix_clock = np.cumsum(((speech_exec + pause_exec) - dur_anchor_src) * unit_mask, axis=0).astype(np.float32)
    prefix_backlog = np.maximum(prefix_clock, 0.0).astype(np.float32) * unit_mask
    return prefix_clock.astype(np.float32), prefix_backlog.astype(np.float32)


def _complete_learned_teacher_bundle(
    teacher_bundle_override: dict,
    *,
    source_cache: dict[str, np.ndarray],
    guidance_bundle: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    bundle = with_blank_aliases(dict(teacher_bundle_override or {}))
    if "rhythm_teacher_speech_exec_tgt" not in bundle:
        raise ValueError("learned_offline teacher bundle is missing rhythm_teacher_speech_exec_tgt.")
    if "rhythm_teacher_pause_exec_tgt" not in bundle:
        raise ValueError("learned_offline teacher bundle is missing rhythm_teacher_pause_exec_tgt.")
    expected_units = int(np.asarray(source_cache["dur_anchor_src"]).reshape(-1).shape[0])
    speech_exec = np.asarray(bundle["rhythm_teacher_speech_exec_tgt"], dtype=np.float32).reshape(-1)
    pause_exec = np.asarray(bundle["rhythm_teacher_pause_exec_tgt"], dtype=np.float32).reshape(-1)
    if speech_exec.shape[0] != expected_units or pause_exec.shape[0] != expected_units:
        raise ValueError(
            "learned_offline teacher bundle unit mismatch: "
            f"speech={speech_exec.shape[0]}, pause={pause_exec.shape[0]}, expected={expected_units}."
        )
    bundle["rhythm_teacher_speech_exec_tgt"] = speech_exec.astype(np.float32)
    bundle["rhythm_teacher_pause_exec_tgt"] = pause_exec.astype(np.float32)
    if "rhythm_teacher_speech_budget_tgt" not in bundle:
        bundle["rhythm_teacher_speech_budget_tgt"] = _sum_exec_budget(bundle["rhythm_teacher_speech_exec_tgt"])
    if "rhythm_teacher_pause_budget_tgt" not in bundle:
        bundle["rhythm_teacher_pause_budget_tgt"] = _sum_exec_budget(bundle["rhythm_teacher_pause_exec_tgt"])
    unit_mask = (np.asarray(source_cache["dur_anchor_src"]).reshape(-1) > 0).astype(np.float32)
    for key in (
        "rhythm_teacher_allocation_tgt",
        "rhythm_teacher_prefix_clock_tgt",
        "rhythm_teacher_prefix_backlog_tgt",
    ):
        if key in bundle and np.asarray(bundle[key]).reshape(-1).shape[0] != expected_units:
            raise ValueError(
                f"learned_offline teacher bundle field {key} has length "
                f"{np.asarray(bundle[key]).reshape(-1).shape[0]}, expected={expected_units}."
            )
    if "rhythm_teacher_allocation_tgt" not in bundle:
        allocation = np.zeros_like(unit_mask, dtype=np.float32)
        allocation[:] = (speech_exec + pause_exec) * unit_mask
        bundle["rhythm_teacher_allocation_tgt"] = allocation.astype(np.float32)
    if (
        "rhythm_teacher_prefix_clock_tgt" not in bundle
        or "rhythm_teacher_prefix_backlog_tgt" not in bundle
    ):
        prefix_clock, prefix_backlog = _build_prefix_targets_from_exec_numpy(
            bundle["rhythm_teacher_speech_exec_tgt"],
            bundle["rhythm_teacher_pause_exec_tgt"],
            source_cache["dur_anchor_src"],
        )
        if "rhythm_teacher_prefix_clock_tgt" not in bundle:
            bundle["rhythm_teacher_prefix_clock_tgt"] = prefix_clock
        if "rhythm_teacher_prefix_backlog_tgt" not in bundle:
            bundle["rhythm_teacher_prefix_backlog_tgt"] = prefix_backlog
    if "rhythm_teacher_confidence" not in bundle:
        fallback_confidence = 1.0
        if guidance_bundle is not None:
            fallback_confidence = float(
                np.asarray(
                    guidance_bundle.get(
                        "rhythm_guidance_confidence",
                        guidance_bundle.get("rhythm_target_confidence", [1.0]),
                    )
                ).reshape(-1)[0]
            )
        bundle["rhythm_teacher_confidence"] = np.asarray([fallback_confidence], dtype=np.float32)
    return with_blank_aliases(bundle)


def build_learned_offline_teacher_bundle(
    *,
    speech_exec_tgt,
    pause_exec_tgt,
    dur_anchor_src,
    unit_mask=None,
    confidence: float | np.ndarray = 1.0,
) -> dict[str, np.ndarray]:
    dur_anchor_src = np.asarray(dur_anchor_src, dtype=np.float32).reshape(-1)
    if unit_mask is None:
        unit_mask = (dur_anchor_src > 0).astype(np.float32)
    else:
        unit_mask = np.asarray(unit_mask, dtype=np.float32).reshape(-1)
    speech_exec = np.asarray(speech_exec_tgt, dtype=np.float32).reshape(-1)
    pause_exec = np.asarray(pause_exec_tgt, dtype=np.float32).reshape(-1)
    expected_units = int(dur_anchor_src.shape[0])
    if speech_exec.shape[0] != expected_units or pause_exec.shape[0] != expected_units:
        raise ValueError(
            "build_learned_offline_teacher_bundle expects full-length unit surfaces: "
            f"speech={speech_exec.shape[0]}, pause={pause_exec.shape[0]}, expected={expected_units}."
        )
    allocation = ((speech_exec + pause_exec) * unit_mask).astype(np.float32)
    prefix_clock, prefix_backlog = _build_prefix_targets_from_exec_numpy(
        speech_exec,
        pause_exec,
        dur_anchor_src,
    )
    confidence_value = float(np.asarray(confidence, dtype=np.float32).reshape(-1)[0])
    return with_blank_aliases({
        "rhythm_teacher_speech_exec_tgt": speech_exec.astype(np.float32),
        "rhythm_teacher_pause_exec_tgt": pause_exec.astype(np.float32),
        "rhythm_teacher_speech_budget_tgt": _sum_exec_budget(speech_exec),
        "rhythm_teacher_pause_budget_tgt": _sum_exec_budget(pause_exec),
        "rhythm_teacher_allocation_tgt": allocation,
        "rhythm_teacher_prefix_clock_tgt": prefix_clock.astype(np.float32),
        "rhythm_teacher_prefix_backlog_tgt": prefix_backlog.astype(np.float32),
        "rhythm_teacher_confidence": np.asarray([confidence_value], dtype=np.float32),
        "rhythm_teacher_surface_name": np.asarray([RHYTHM_TEACHER_SURFACE_LEARNED_OFFLINE_NAME], dtype=np.str_),
        "rhythm_teacher_target_source_id": np.asarray(
            [RHYTHM_TEACHER_TARGET_SOURCE_LEARNED_OFFLINE], dtype=np.int64
        ),
    })


def build_learned_offline_teacher_export_bundle(
    *,
    speech_exec_tgt,
    pause_exec_tgt,
    dur_anchor_src,
    unit_mask=None,
    confidence,
) -> dict[str, np.ndarray]:
    """Stable export contract for learned-offline teacher assets."""
    return build_learned_offline_teacher_bundle(
        speech_exec_tgt=speech_exec_tgt,
        pause_exec_tgt=pause_exec_tgt,
        dur_anchor_src=dur_anchor_src,
        unit_mask=unit_mask,
        confidence=confidence,
    )


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


def _as_runtime_batched_tensor(value, *, dtype=torch.float32) -> torch.Tensor:
    if torch.is_tensor(value):
        tensor = value.float() if dtype == torch.float32 else value.to(dtype=dtype)
    else:
        tensor = torch.tensor(np.asarray(value), dtype=dtype)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def build_online_retimed_bundle(
    *,
    mel,
    frame_plan,
    f0=None,
    uv=None,
    pause_frame_weight: float = 0.20,
    stretch_weight_min: float = 0.35,
) -> dict[str, torch.Tensor]:
    mel_tensor = _as_runtime_batched_tensor(mel, dtype=torch.float32)
    if mel_tensor.dim() != 3:
        raise ValueError(f"Expected mel with shape [B,T,C], got {tuple(mel_tensor.shape)}")
    silence_fill = torch.stack([
        _infer_silence_frame(mel_tensor[batch_idx])
        for batch_idx in range(mel_tensor.size(0))
    ], dim=0)
    mel_tgt = sample_tensor_by_frame_plan(
        mel_tensor,
        frame_plan,
        blank_fill=silence_fill,
    )
    frame_weight = build_frame_weight_from_plan(
        frame_plan,
        pause_frame_weight=float(pause_frame_weight),
        stretch_weight_min=float(stretch_weight_min),
    )
    bundle: dict[str, torch.Tensor] = {
        "mel_tgt": mel_tgt,
        "frame_weight": frame_weight,
        "mel_len": frame_plan.total_mask.sum(dim=1).long(),
    }
    if f0 is not None:
        f0_tensor = _as_runtime_batched_tensor(f0, dtype=torch.float32)
        bundle["f0_tgt"] = sample_tensor_by_frame_plan(
            f0_tensor,
            frame_plan,
            blank_fill=0.0,
        )
    if uv is not None:
        uv_tensor = _as_runtime_batched_tensor(uv, dtype=torch.float32)
        bundle["uv_tgt"] = sample_tensor_by_frame_plan(
            uv_tensor,
            frame_plan,
            blank_fill=1.0,
        )
    return bundle


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
    mel = _as_mel_tensor(mel)
    dur_anchor_src = torch.tensor(np.asarray(dur_anchor_src), dtype=torch.float32).unsqueeze(0)
    speech_exec_tgt = torch.tensor(np.asarray(speech_exec_tgt), dtype=torch.float32).unsqueeze(0)
    pause_exec_tgt = torch.tensor(np.asarray(pause_exec_tgt), dtype=torch.float32).unsqueeze(0)
    if unit_mask is None:
        unit_mask = dur_anchor_src.gt(0).float()
    else:
        unit_mask = torch.tensor(np.asarray(unit_mask), dtype=torch.float32).unsqueeze(0)

    visible = int(unit_mask.sum().item())
    if visible <= 0:
        empty = mel.new_zeros((0, mel.size(-1)))
        empty_weight = mel.new_zeros((0,))
        return {
            "rhythm_retimed_mel_tgt": empty.cpu().numpy().astype(np.float32),
            "rhythm_retimed_mel_len": np.asarray([0], dtype=np.int64),
            "rhythm_retimed_frame_weight": empty_weight.cpu().numpy().astype(np.float32),
        }
    frame_plan = build_frame_plan_from_execution(
        dur_anchor_src=dur_anchor_src,
        speech_exec=speech_exec_tgt,
        pause_exec=pause_exec_tgt,
        unit_mask=unit_mask,
    )
    bundle = build_online_retimed_bundle(
        mel=mel,
        frame_plan=frame_plan,
        pause_frame_weight=float(pause_frame_weight),
        stretch_weight_min=float(stretch_weight_min),
    )
    retimed_len = int(bundle["mel_len"][0].item())
    retimed = bundle["mel_tgt"][0, :retimed_len]
    frame_weight = bundle["frame_weight"][0, :retimed_len]
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
    teacher_target_source: str = "algorithmic",
    teacher_bundle_override: dict | None = None,
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
                _complete_learned_teacher_bundle(
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
            teacher_target_source=teacher_target_source if need_teacher_surface else None,
            teacher_surface_name=teacher_surface_name,
        )
    )
    return with_blank_aliases(bundle)
