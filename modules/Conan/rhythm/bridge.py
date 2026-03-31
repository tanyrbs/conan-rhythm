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
):
    if not rhythm_enable_v2 or ref is None:
        return None
    resolved_lengths = resolve_content_lengths(content, content_lengths=content_lengths)
    unit_batch = rhythm_unit_frontend.from_content_tensor(
        content,
        content_lengths=resolved_lengths,
        mark_last_open=bool(infer),
    )
    if rhythm_ref_conditioning is None:
        rhythm_ref_conditioning = rhythm_module.encode_reference(ref)
    execution = rhythm_module(
        content_units=unit_batch.content_units,
        dur_anchor_src=unit_batch.dur_anchor_src,
        unit_mask=unit_batch.unit_mask,
        open_run_mask=unit_batch.open_run_mask,
        ref_rhythm_stats=rhythm_ref_conditioning["ref_rhythm_stats"],
        ref_rhythm_trace=rhythm_ref_conditioning["ref_rhythm_trace"],
        state=rhythm_state,
    )
    return {
        "unit_batch": unit_batch,
        "execution": execution,
        "ref_conditioning": rhythm_ref_conditioning,
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
    ret["speech_duration_exec"] = execution.speech_duration_exec
    ret["pause_after_exec"] = execution.pause_after_exec
    ret["effective_duration_exec"] = execution.effective_duration_exec
    ret["commit_frontier"] = execution.commit_frontier

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
        speech_duration_exec=execution.speech_duration_exec,
        pause_after_exec=execution.pause_after_exec,
        unit_mask=unit_batch.unit_mask,
        silent_token=hparams.get("silent_token", 57),
        speech_state_fn=speech_state_fn,
        pause_state=pause_state,
    )
    ret["content"] = rendered["frame_tokens"]
    ret["content_rhythm_rendered"] = rendered["frame_tokens"]
    ret["content_embed_proj_rhythm"] = rendered["frame_states"]
    ret["rhythm_total_mask"] = rendered["total_mask"]
    ret["rhythm_speech_mask"] = rendered["frame_mask"]
    return rendered["frame_states"], rendered["total_mask"][:, :, None]
