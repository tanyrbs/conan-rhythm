from __future__ import annotations

from typing import Callable

import torch

from .renderer import render_rhythm_sequence


def resolve_content_lengths(content: torch.Tensor, content_lengths: torch.Tensor | None = None) -> torch.Tensor:
    if content_lengths is not None:
        return content_lengths.long().to(device=content.device)
    return torch.full(
        (content.size(0),),
        int(content.size(1)),
        dtype=torch.long,
        device=content.device,
    )


def build_content_nonpadding(content: torch.Tensor, content_lengths: torch.Tensor | None = None) -> torch.Tensor:
    if content_lengths is not None:
        steps = torch.arange(content.size(1), device=content.device)[None, :]
        return (steps < content_lengths[:, None].long()).float()
    return (content != -1).float()


def resolve_rhythm_apply_mode(hparams, *, infer: bool = False, override=None) -> bool:
    if override is not None:
        return bool(override)
    mode = str(hparams.get("rhythm_apply_mode", "infer") or "infer").strip().lower()
    if mode in {"off", "none", "false"}:
        return False
    if mode in {"always", "all"}:
        return True
    if mode in {"infer", "inference", "test"}:
        return bool(infer)
    if mode in {"train", "training"}:
        return not bool(infer)
    return bool(infer)


def run_rhythm_frontend(
    *,
    rhythm_enable_v2: bool,
    rhythm_unit_frontend,
    rhythm_module,
    content: torch.Tensor,
    ref: torch.Tensor | None,
    infer: bool = False,
    content_lengths: torch.Tensor | None = None,
    rhythm_state=None,
    rhythm_ref_conditioning=None,
    rhythm_source_cache: dict | None = None,
    rhythm_offline_source_cache: dict | None = None,
    enable_dual_mode_teacher: bool = False,
    enable_learned_offline_teacher: bool = True,
    enable_algorithmic_teacher: bool = False,
    projector_pause_topk_ratio_override: float | None = None,
    source_boundary_scale_override: float | None = None,
    teacher_source_boundary_scale_override: float | None = None,
):
    if not rhythm_enable_v2 or ref is None:
        return None
    if rhythm_source_cache is not None:
        unit_batch = rhythm_unit_frontend.from_precomputed(
            content_units=rhythm_source_cache["content_units"],
            dur_anchor_src=rhythm_source_cache["dur_anchor_src"],
            unit_mask=rhythm_source_cache.get("unit_mask"),
            open_run_mask=rhythm_source_cache.get("open_run_mask"),
            sealed_mask=rhythm_source_cache.get("sealed_mask"),
            sep_hint=rhythm_source_cache.get("sep_hint"),
            boundary_confidence=rhythm_source_cache.get("boundary_confidence"),
        )
    else:
        resolved_lengths = resolve_content_lengths(content, content_lengths=content_lengths)
        unit_batch = rhythm_unit_frontend.from_content_tensor(
            content,
            content_lengths=resolved_lengths,
            mark_last_open=bool(infer),
        )
    if rhythm_ref_conditioning is None:
        rhythm_ref_conditioning = rhythm_module.encode_reference(ref)
    offline_unit_batch = None
    if rhythm_offline_source_cache is not None:
        offline_unit_batch = rhythm_unit_frontend.from_precomputed(
            content_units=rhythm_offline_source_cache["content_units"],
            dur_anchor_src=rhythm_offline_source_cache["dur_anchor_src"],
            unit_mask=rhythm_offline_source_cache.get("unit_mask"),
            open_run_mask=rhythm_offline_source_cache.get("open_run_mask"),
            sealed_mask=rhythm_offline_source_cache.get("sealed_mask"),
            sep_hint=rhythm_offline_source_cache.get("sep_hint"),
            boundary_confidence=rhythm_offline_source_cache.get("boundary_confidence"),
        )
    if enable_dual_mode_teacher and not enable_learned_offline_teacher and not infer:
        raise ValueError(
            "rhythm_enable_dual_mode_teacher requires rhythm_enable_learned_offline_teacher: true "
            "in the maintained rhythm path."
        )
    if enable_dual_mode_teacher and not infer:
        dual_outputs = rhythm_module.forward_dual(
            content_units=unit_batch.content_units,
            dur_anchor_src=unit_batch.dur_anchor_src,
            unit_mask=unit_batch.unit_mask,
            open_run_mask=unit_batch.open_run_mask,
            sealed_mask=unit_batch.sealed_mask,
            sep_hint=unit_batch.sep_hint,
            boundary_confidence=unit_batch.boundary_confidence,
            ref_conditioning=rhythm_ref_conditioning,
            state=rhythm_state,
            projector_pause_topk_ratio_override=projector_pause_topk_ratio_override,
            source_boundary_scale_override=source_boundary_scale_override,
            teacher_source_boundary_scale_override=teacher_source_boundary_scale_override,
            offline_content_units=offline_unit_batch.content_units if offline_unit_batch is not None else None,
            offline_dur_anchor_src=offline_unit_batch.dur_anchor_src if offline_unit_batch is not None else None,
            offline_unit_mask=offline_unit_batch.unit_mask if offline_unit_batch is not None else None,
            offline_open_run_mask=offline_unit_batch.open_run_mask if offline_unit_batch is not None else None,
            offline_sealed_mask=offline_unit_batch.sealed_mask if offline_unit_batch is not None else None,
            offline_sep_hint=offline_unit_batch.sep_hint if offline_unit_batch is not None else None,
            offline_boundary_confidence=offline_unit_batch.boundary_confidence if offline_unit_batch is not None else None,
        )
        execution = dual_outputs["streaming_execution"]
        offline_execution = dual_outputs["offline_execution"]
        offline_confidence = dual_outputs.get("offline_confidence")
        algorithmic_teacher = dual_outputs["algorithmic_teacher"]
    else:
        execution = rhythm_module(
            content_units=unit_batch.content_units,
            dur_anchor_src=unit_batch.dur_anchor_src,
            unit_mask=unit_batch.unit_mask,
            open_run_mask=unit_batch.open_run_mask,
            sealed_mask=unit_batch.sealed_mask,
            sep_hint=unit_batch.sep_hint,
            boundary_confidence=unit_batch.boundary_confidence,
            ref_conditioning=rhythm_ref_conditioning,
            state=rhythm_state,
            projector_pause_topk_ratio_override=projector_pause_topk_ratio_override,
            source_boundary_scale_override=source_boundary_scale_override,
        )
        offline_execution = None
        offline_confidence = None
        algorithmic_teacher = None
        if enable_algorithmic_teacher and not infer:
            algorithmic_teacher = rhythm_module.compute_algorithmic_teacher(
                content_units=unit_batch.content_units,
                dur_anchor_src=unit_batch.dur_anchor_src,
                unit_mask=unit_batch.unit_mask,
                open_run_mask=unit_batch.open_run_mask,
                sealed_mask=unit_batch.sealed_mask,
                sep_hint=unit_batch.sep_hint,
                boundary_confidence=unit_batch.boundary_confidence,
                ref_conditioning=rhythm_ref_conditioning,
                source_boundary_scale_override=teacher_source_boundary_scale_override,
            )
    return {
        "unit_batch": unit_batch,
        "execution": execution,
        "ref_conditioning": rhythm_ref_conditioning,
        "offline_execution": offline_execution,
        "offline_confidence": offline_confidence,
        "offline_unit_batch": offline_unit_batch,
        "algorithmic_teacher": algorithmic_teacher,
    }


def attach_rhythm_outputs(
    *,
    ret: dict,
    rhythm_bundle,
    content_embed: torch.Tensor,
    tgt_nonpadding: torch.Tensor,
    hparams,
    infer: bool = False,
    rhythm_apply_override=None,
    speech_state_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    pause_state: torch.Tensor | None = None,
    frame_state_post_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
):
    if rhythm_bundle is None:
        return content_embed, tgt_nonpadding
    unit_batch = rhythm_bundle["unit_batch"]
    execution = rhythm_bundle["execution"]
    ref_conditioning = rhythm_bundle["ref_conditioning"]
    ret["rhythm_unit_batch"] = unit_batch
    ret["rhythm_execution"] = execution
    ret["rhythm_state_next"] = execution.next_state
    ret["rhythm_ref_conditioning"] = ref_conditioning
    ret["ref_rhythm_stats"] = ref_conditioning["ref_rhythm_stats"]
    ret["ref_rhythm_trace"] = ref_conditioning["ref_rhythm_trace"]
    ret["speech_budget_win"] = execution.planner.speech_budget_win
    ret["pause_budget_win"] = execution.planner.pause_budget_win
    ret["dur_logratio_unit"] = execution.planner.dur_logratio_unit
    ret["pause_weight_unit"] = execution.planner.pause_weight_unit
    ret["boundary_latent"] = execution.planner.boundary_latent
    ret["source_boundary_cue"] = execution.planner.source_boundary_cue
    ret["speech_duration_exec"] = execution.speech_duration_exec
    ret["blank_duration_exec"] = execution.blank_duration_exec
    ret["pause_after_exec"] = execution.pause_after_exec
    ret["effective_duration_exec"] = execution.effective_duration_exec
    ret["commit_frontier"] = execution.commit_frontier
    ret["slot_duration_exec"] = execution.slot_duration_exec
    ret["slot_mask"] = execution.slot_mask
    ret["slot_is_blank"] = execution.slot_is_blank
    ret["slot_unit_index"] = execution.slot_unit_index
    ret["rhythm_frame_plan"] = execution.frame_plan
    ret["blank_slot_duration_exec"] = execution.blank_slot_duration_exec
    ret["blank_slot_mask"] = execution.blank_slot_mask
    ret["blank_slot_is_blank"] = execution.blank_slot_is_blank
    ret["blank_slot_unit_index"] = execution.blank_slot_unit_index
    ret["sealed_mask"] = unit_batch.sealed_mask
    ret["boundary_confidence"] = unit_batch.boundary_confidence
    if rhythm_bundle.get("offline_execution") is not None:
        ret["rhythm_offline_execution"] = rhythm_bundle["offline_execution"]
    if rhythm_bundle.get("offline_confidence") is not None:
        offline_confidence = rhythm_bundle["offline_confidence"]
        if isinstance(offline_confidence, dict):
            for name, value in offline_confidence.items():
                if value is None:
                    continue
                ret[f"rhythm_offline_confidence_{name}"] = value
            if offline_confidence.get("overall") is not None:
                ret["rhythm_offline_confidence"] = offline_confidence["overall"]
        else:
            ret["rhythm_offline_confidence"] = offline_confidence
    if rhythm_bundle.get("offline_unit_batch") is not None:
        ret["rhythm_offline_unit_batch"] = rhythm_bundle["offline_unit_batch"]
    if rhythm_bundle.get("algorithmic_teacher") is not None:
        ret["rhythm_algorithmic_teacher"] = rhythm_bundle["algorithmic_teacher"]

    apply_rhythm_render = resolve_rhythm_apply_mode(
        hparams,
        infer=infer,
        override=rhythm_apply_override,
    )
    ret["rhythm_apply_render"] = float(apply_rhythm_render)
    if not apply_rhythm_render:
        return content_embed, tgt_nonpadding
    if speech_state_fn is None or pause_state is None:
        raise ValueError("speech_state_fn and pause_state are required when rhythm render is enabled.")
    rendered = render_rhythm_sequence(
        content_units=unit_batch.content_units,
        silent_token=hparams.get("silent_token", 57),
        speech_state_fn=speech_state_fn,
        pause_state=pause_state,
        frame_plan=execution.frame_plan,
        dur_anchor_src=unit_batch.dur_anchor_src,
        slot_duration_exec=execution.slot_duration_exec,
        slot_mask=execution.slot_mask,
        slot_is_blank=execution.slot_is_blank,
        slot_unit_index=execution.slot_unit_index,
        frame_state_post_fn=frame_state_post_fn,
    )
    ret["content"] = rendered.frame_tokens
    ret["content_rhythm_rendered"] = rendered.frame_tokens
    ret["content_embed_proj_rhythm"] = rendered.frame_states
    ret["rhythm_total_mask"] = rendered.total_mask
    ret["rhythm_speech_mask"] = rendered.speech_mask
    ret["rhythm_blank_mask"] = rendered.blank_mask
    ret["rhythm_render_slot_index"] = rendered.frame_slot_index
    ret["rhythm_render_unit_index"] = rendered.frame_unit_index
    ret["rhythm_render_phase_features"] = rendered.frame_phase_features
    ret["rhythm_frame_plan"] = rendered.frame_plan
    return rendered.frame_states, rendered.total_mask[:, :, None]
