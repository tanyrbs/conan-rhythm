from __future__ import annotations

import torch
import torch.nn as nn

from modules.Conan.rhythm.bridge import resolve_rhythm_apply_mode
from modules.Conan.rhythm.renderer import render_rhythm_sequence

from .contracts import (
    ReferenceDurationMemory,
    collect_duration_v3_source_cache,
    move_duration_runtime_state,
    move_reference_duration_memory,
    move_source_unit_batch,
)
from .module import StreamingDurationModule
from .unit_frontend import DurationUnitFrontend

_PROMPT_UNIT_REQUIRED_KEYS = (
    "prompt_content_units",
    "prompt_duration_obs",
)


class ConanDurationAdapter(nn.Module):
    def __init__(self, hparams, hidden_size: int, *, vocab_size: int) -> None:
        super().__init__()
        self.hparams = hparams
        self.unit_frontend = DurationUnitFrontend(
            vocab_size=vocab_size,
            silent_token=hparams.get("silent_token", 57),
            separator_aware=bool(hparams.get("rhythm_separator_aware", True)),
            tail_open_units=int(hparams.get("rhythm_tail_open_units", 1)),
            anchor_hidden_size=int(hparams.get("rhythm_anchor_hidden_size", 128)),
            anchor_min_frames=float(hparams.get("rhythm_anchor_min_frames", 1.0)),
            anchor_max_frames=float(hparams.get("rhythm_anchor_max_frames", 12.0)),
        )
        streaming_mode = str(hparams.get("rhythm_streaming_mode", "strict") or "strict")
        micro_lookahead_units = hparams.get("rhythm_micro_lookahead_units")
        self.module = StreamingDurationModule(
            vocab_size=vocab_size,
            hidden_size=int(hparams.get("rhythm_hidden_size", hidden_size)),
            basis_rank=int(hparams.get("rhythm_response_rank", 12)),
            response_window_left=int(hparams.get("rhythm_response_window_left", 4)),
            response_window_right=int(hparams.get("rhythm_response_window_right", 0)),
            trace_bins=int(hparams.get("rhythm_trace_bins", 24)),
            streaming_mode=streaming_mode,
            micro_lookahead_units=(None if micro_lookahead_units is None else int(micro_lookahead_units)),
            ridge_lambda=float(hparams.get("rhythm_operator_ridge_lambda", 1.0)),
        )
        self.pause_state = nn.Parameter(torch.zeros(hidden_size), requires_grad=False)
        self.render_phase_mlp = None
        self.render_phase_gain = None

    def render_frame_state_post(
        self,
        frame_states: torch.Tensor,
        frame_phase_features: torch.Tensor,
        blank_mask: torch.Tensor,
        total_mask: torch.Tensor,
    ) -> torch.Tensor:
        del frame_phase_features, blank_mask
        return frame_states * total_mask.unsqueeze(-1)

    @staticmethod
    def resolve_pause_topk_ratio(*, infer: bool, global_steps: int):
        del infer, global_steps
        return None

    @staticmethod
    def resolve_source_boundary_scale(*, infer: bool, global_steps: int, teacher: bool = False):
        del infer, global_steps, teacher
        return None

    def _build_source_batch(
        self,
        *,
        content: torch.Tensor,
        content_lengths: torch.Tensor | None,
        rhythm_source_cache: dict | None,
        infer: bool,
    ):
        precomputed_cache = collect_duration_v3_source_cache(rhythm_source_cache)
        if precomputed_cache is not None:
            return self.unit_frontend.from_precomputed(
                **precomputed_cache,
            )
        return self.unit_frontend.from_content_tensor(
            content,
            content_lengths=content_lengths,
            mark_last_open=bool(infer) or bool(self.hparams.get("rhythm_streaming_prefix_train", False)),
        )

    @staticmethod
    def _resolve_prompt_unit_mask(
        *,
        prompt_duration_obs: torch.Tensor,
        prompt_unit_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if isinstance(prompt_unit_mask, torch.Tensor):
            return prompt_unit_mask.float().clamp(0.0, 1.0)
        return prompt_duration_obs.float().gt(0.0).float()

    def _prepare_prompt_unit_conditioning(
        self,
        *,
        ref_conditioning,
        device: torch.device,
    ):
        if not isinstance(ref_conditioning, dict):
            return ref_conditioning
        if any(key not in ref_conditioning for key in _PROMPT_UNIT_REQUIRED_KEYS):
            return ref_conditioning
        prompt_content_units = ref_conditioning["prompt_content_units"].to(device=device).long()
        prompt_duration_obs = ref_conditioning["prompt_duration_obs"].to(device=device)
        prompt_unit_mask = self._resolve_prompt_unit_mask(
            prompt_duration_obs=prompt_duration_obs,
            prompt_unit_mask=ref_conditioning.get("prompt_unit_mask"),
        )
        enriched = dict(ref_conditioning)
        enriched["prompt_content_units"] = prompt_content_units
        enriched["prompt_duration_obs"] = prompt_duration_obs
        enriched["prompt_unit_mask"] = prompt_unit_mask
        if isinstance(enriched.get("prompt_unit_anchor_base"), torch.Tensor):
            enriched["prompt_unit_anchor_base"] = enriched["prompt_unit_anchor_base"].to(device=device)
        if isinstance(enriched.get("prompt_log_base"), torch.Tensor):
            enriched["prompt_log_base"] = enriched["prompt_log_base"].to(device=device)
        if enriched.get("prompt_log_base") is None and enriched.get("prompt_unit_anchor_base") is None:
            enriched["prompt_unit_anchor_base"] = self.unit_frontend.anchor_net(
                prompt_content_units,
                prompt_unit_mask,
            )
        return enriched

    def _attach_runtime_outputs(
        self,
        *,
        ret: dict,
        source_batch,
        ref_memory: ReferenceDurationMemory,
        execution,
        rhythm_state_prev,
    ) -> None:
        ret["rhythm_version"] = "v3"
        ret["rhythm_teacher_as_main"] = 0.0
        ret["rhythm_unit_batch"] = source_batch
        ret["rhythm_execution"] = execution
        ret["rhythm_state_prev"] = rhythm_state_prev
        ret["rhythm_state_next"] = execution.next_state
        ret["rhythm_ref_conditioning"] = ref_memory
        ret["speech_duration_exec"] = execution.speech_duration_exec
        ret["commit_frontier"] = execution.commit_frontier
        if execution.frame_plan is not None:
            ret["rhythm_frame_plan"] = execution.frame_plan

    def forward(
        self,
        *,
        ret: dict,
        content: torch.Tensor,
        ref: torch.Tensor | None,
        target: torch.Tensor | None,
        f0: torch.Tensor | None,
        uv: torch.Tensor | None,
        infer: bool,
        global_steps: int,
        content_embed: torch.Tensor,
        tgt_nonpadding: torch.Tensor,
        content_lengths: torch.Tensor | None = None,
        ref_lengths: torch.Tensor | None = None,
        rhythm_state=None,
        rhythm_ref_conditioning=None,
        rhythm_apply_override=None,
        rhythm_runtime_overrides: dict | None = None,
        rhythm_source_cache: dict | None = None,
        rhythm_offline_source_cache: dict | None = None,
        speech_state_fn=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        del target, global_steps, rhythm_runtime_overrides, rhythm_offline_source_cache
        if ref is None and rhythm_ref_conditioning is None:
            return content_embed, tgt_nonpadding, f0, uv
        if content_embed.device != content.device:
            raise ValueError(
                f"content/content_embed device mismatch: content={content.device}, content_embed={content_embed.device}"
            )
        if tgt_nonpadding.device != content.device:
            raise ValueError(
                f"content/tgt_nonpadding device mismatch: content={content.device}, tgt_nonpadding={tgt_nonpadding.device}"
            )
        source_batch = self._build_source_batch(
            content=content,
            content_lengths=content_lengths,
            rhythm_source_cache=rhythm_source_cache,
            infer=infer,
        )
        source_batch = move_source_unit_batch(source_batch, device=content.device)
        rhythm_state_prev = move_duration_runtime_state(rhythm_state, device=content.device)
        rhythm_ref_conditioning = self._prepare_prompt_unit_conditioning(
            ref_conditioning=rhythm_ref_conditioning,
            device=content.device,
        )
        ref_memory = self.module.build_reference_conditioning(
            ref_conditioning=rhythm_ref_conditioning,
            ref_mel=ref,
            ref_lengths=ref_lengths,
        )
        ref_memory = move_reference_duration_memory(
            ref_memory,
            device=content.device,
            dtype=content_embed.dtype if content_embed.is_floating_point() else None,
        )
        execution = self.module(
            source_batch=source_batch,
            ref_memory=ref_memory,
            state=rhythm_state_prev,
        )
        self._attach_runtime_outputs(
            ret=ret,
            source_batch=source_batch,
            ref_memory=ref_memory,
            execution=execution,
            rhythm_state_prev=rhythm_state_prev,
        )
        apply_rhythm_render = resolve_rhythm_apply_mode(
            self.hparams,
            infer=infer,
            override=rhythm_apply_override,
        )
        if (
            bool(apply_rhythm_render)
            and (
                execution.frame_plan is None
                or source_batch.content_units.size(1) <= 0
                or execution.frame_plan.total_mask.numel() <= 0
                or float(execution.frame_plan.total_mask.sum().item()) <= 0.0
            )
        ):
            ret["rhythm_apply_render"] = 0.0
            ret["rhythm_render_skipped_empty"] = 1.0
            return content_embed, tgt_nonpadding, f0, uv
        ret["rhythm_apply_render"] = float(apply_rhythm_render)
        if not apply_rhythm_render:
            return content_embed, tgt_nonpadding, f0, uv
        if speech_state_fn is None:
            raise ValueError("speech_state_fn is required when rhythm render is enabled.")
        rendered = render_rhythm_sequence(
            content_units=source_batch.content_units,
            silent_token=int(self.hparams.get("silent_token", 57)),
            speech_state_fn=speech_state_fn,
            pause_state=self.pause_state.to(device=content_embed.device, dtype=content_embed.dtype),
            frame_plan=execution.frame_plan,
            frame_state_post_fn=None,
        )
        ret["content"] = rendered.frame_tokens
        ret["content_rhythm_rendered"] = rendered.frame_tokens
        ret["content_embed_proj_rhythm"] = rendered.frame_states
        ret["rhythm_total_mask"] = rendered.total_mask
        ret["rhythm_speech_mask"] = rendered.speech_mask
        ret["rhythm_blank_mask"] = rendered.blank_mask
        ret["rhythm_render_unit_index"] = rendered.frame_unit_index
        ret["rhythm_frame_plan"] = rendered.frame_plan
        return rendered.frame_states, rendered.total_mask[:, :, None], f0, uv
